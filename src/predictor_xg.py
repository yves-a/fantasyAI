import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from datetime import datetime
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    playergamelog,
    teamgamelog,
    leaguedashteamstats,
    commonplayerinfo,
)
from nba_api.live.nba.endpoints import odds

# --- CONFIGURATION ---
# Update these paths to point to your XGBoost model file
MODEL_PATH = "../models/nba_xgboost_model.json"
DATA_FILE = "../data/processed_stats.csv"

_MODEL_CACHE = None


def calculate_custom_fpts(row):
    # Your specific fantasy scoring logic
    return (
        (row["FGM"] * 2)
        + (row["FGA"] * -1)
        + (row["FTM"] * 1)
        + (row["FTA"] * -1)
        + (row["FG3M"] * 1)
        + (row["REB"] * 1)
        + (row["AST"] * 2)
        + (row["STL"] * 4)
        + (row["BLK"] * 4)
        + (row["TOV"] * -2)
        + (row["PTS"] * 1)
    )


def get_current_team_id(p_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        info.columns = [c.upper() for c in info.columns]
        return int(info.iloc[0]["TEAM_ID"])
    except:
        return None


def get_live_vegas_data(game_id):
    try:
        target_id = str(game_id).strip()
        live_odds = odds.Odds().get_dict()
        game_list = live_odds.get("odds", [])
        for game in game_list:
            curr_id = str(game.get("gameId", "")).strip()
            if curr_id == target_id or curr_id.endswith(target_id[-5:]):
                return float(game.get("overUnder", 230.0)), abs(
                    float(game.get("pointSpread", 0.0))
                )
        return 230.0, 0.0
    except:
        return 230.0, 0.0


def get_next_game_context(team_id):
    try:
        log = teamgamelog.TeamGameLog(
            team_id=team_id, season="2025-26"
        ).get_data_frames()[0]
        log.columns = [c.upper() for c in log.columns]
        log["GAME_DATE_DT"] = pd.to_datetime(
            log["GAME_DATE"], format="%b %d, %Y"
        ).dt.date
        today = datetime.now().date()

        future_games = log[log["GAME_DATE_DT"] >= today].sort_values("GAME_DATE_DT")
        if future_games.empty:
            return 115.0, None, 1

        next_game = future_games.iloc[0]
        is_home = 1 if "vs." in next_game["MATCHUP"].lower() else 0
        opp_abbr = next_game["MATCHUP"].split(" ")[-1]

        ratings = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0]
        ratings.columns = [c.upper() for c in ratings.columns]
        opp_row = ratings[ratings["TEAM_ABBREVIATION"] == opp_abbr]

        def_rating = (
            float(opp_row.iloc[0]["DEF_RATING"]) if not opp_row.empty else 115.0
        )
        return def_rating, next_game["GAME_ID"], is_home
    except:
        return 115.0, None, 1


def get_live_player_stats(player_name):
    search = players.find_players_by_full_name(player_name)
    if not search:
        return None
    p_id = search[0]["id"]

    try:
        log_raw = playergamelog.PlayerGameLog(
            player_id=p_id, season="2025-26"
        ).get_data_frames()[0]
        log_raw.columns = [c.upper() for c in log_raw.columns]
        if len(log_raw) < 5:
            return None

        log_raw["GAME_DATE"] = pd.to_datetime(log_raw["GAME_DATE"], format="%b %d, %Y")
        df = log_raw.sort_values("GAME_DATE")
        df["FPTS"] = df.apply(calculate_custom_fpts, axis=1)

        t_id = get_current_team_id(p_id)
        if not t_id:
            return None

        opp_def, next_gid, next_home = get_next_game_context(t_id)
        ou, _ = get_live_vegas_data(next_gid) if next_gid else (230.0, 0.0)
        days_rest = (datetime.now().date() - df.iloc[-1]["GAME_DATE"].date()).days

        return {
            "ROLLING_FPTS": df["FPTS"].tail(5).mean(),
            "ROLLING_MIN": (
                df["MIN"].tail(5).mean()
                if isinstance(df["MIN"].iloc[-1], (int, float))
                else 30.0
            ),
            "OPP_DEF_RATING": opp_def,
            "VS_OPP_AVG": df["FPTS"].mean(),
            "STAR_OUT": 0,
            "USAGE_DELTA": 0.0,
            "IS_STARTER": 1,
            "IS_HOME": next_home,
            "DAYS_REST": days_rest,
            "IS_B2B": 1 if days_rest == 1 else 0,
            "GAME_PACE": ou / 2.3,
        }
    except Exception as e:
        print(f"Fetch Error: {e}")
        return None


def predict_player_xgboost(name):
    global _MODEL_CACHE

    # 1. Load XGBoost Model
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            return f"Error: Model file not found at {MODEL_PATH}"

        # Load the booster
        _MODEL_CACHE = xgb.XGBRegressor()
        _MODEL_CACHE.load_model(MODEL_PATH)

    features = [
        "ROLLING_FPTS",
        "ROLLING_MIN",
        "OPP_DEF_RATING",
        "VS_OPP_AVG",
        "STAR_OUT",
        "USAGE_DELTA",
        "IS_STARTER",
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "GAME_PACE",
    ]

    stats_dict = get_live_player_stats(name)
    if not stats_dict:
        return f"Error: No data for {name}."

    # 2. Prepare Data for XGBoost
    # Create a DataFrame with a single row to ensure feature names match
    stats_df = pd.DataFrame([stats_dict])[features]

    # 3. Inference
    prediction = _MODEL_CACHE.predict(stats_df)

    return float(prediction[0]), stats_dict


if __name__ == "__main__":
    player_input = input("Enter player name: ").strip()
    if player_input:
        result, raw_stats = predict_player_xgboost(player_input)
        if isinstance(result, str):
            print(result)
        else:
            print(f"\n{'='*45}")
            print(f"XGBOOST PREDICTION FOR {player_input.upper()}")
            print(f"Expected Score: {result:.2f} FPTS")
            print(f"{'='*45}")
            print(
                f"Context: Opp Def: {raw_stats['OPP_DEF_RATING']} | Pace: {raw_stats['GAME_PACE']:.1f}"
            )
