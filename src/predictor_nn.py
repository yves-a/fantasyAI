import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
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
MODEL_PATH = "../models/nba_nn_model.keras"
SCALER_PATH = "../models/nn_scaler.pkl"
DATA_FILE = "../data/processed_stats.csv"

# Global cache to prevent expensive re-loading and TF retracing
_MODEL_CACHE = None
_SCALER_CACHE = None


def calculate_custom_fpts(row):
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
            "ROLLING_MIN": df["MIN"].tail(5).mean(),
            "OPP_DEF_RATING": opp_def,
            "VS_OPP_AVG": df["FPTS"].mean(),
            "STAR_OUT": 0,
            "USAGE_DELTA": 0.0,  # Placeholder: Predictor assumes baseline until Star-Out is confirmed
            "IS_STARTER": 1,
            "IS_HOME": next_home,
            "DAYS_REST": days_rest,
            "IS_B2B": 1 if days_rest == 1 else 0,
            "GAME_PACE": ou / 2.3,
        }
    except Exception as e:
        print(f"Fetch Error: {e}")
        return None


def predict_any_player_nn(name, model=None, scaler=None):
    global _MODEL_CACHE, _SCALER_CACHE

    # 1. Handle Model/Scaler Loading & Caching
    if model is None or scaler is None:
        if _MODEL_CACHE is None or _SCALER_CACHE is None:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
                return "Error: Model or Scaler missing."
            _MODEL_CACHE = tf.keras.models.load_model(MODEL_PATH)
            _SCALER_CACHE = joblib.load(SCALER_PATH)
        model, scaler = _MODEL_CACHE, _SCALER_CACHE

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

    # 2. Vectorize and Scale
    stats_row = np.array([stats_dict.get(f, 0.0) for f in features]).reshape(1, -1)
    stats_scaled = scaler.transform(stats_row)

    # 3. Optimized Inference: Calling model directly avoids tf.function retracing
    prediction = model(stats_scaled, training=False)
    return float(prediction[0][0])


if __name__ == "__main__":
    player_input = input("Enter player name: ").strip()
    if player_input:
        result = predict_any_player_nn(player_input)
        if isinstance(result, str):
            print(result)
        else:
            print(
                f"\n{'='*45}\nPREDICTION FOR {player_input.upper()}\nExpected Score: {result:.2f} FPTS\n{'='*45}"
            )
