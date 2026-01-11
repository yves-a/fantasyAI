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


def calculate_custom_fpts(row):
    """Standardized Fantasy Point Calculation (Must match Trainer)."""
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
    """Fetches the player's active team from their profile (Trade-Proof)."""
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        info.columns = [c.upper() for c in info.columns]
        team_id = int(info.iloc[0]["TEAM_ID"])
        return team_id if team_id > 0 else None
    except:
        return None


def get_live_vegas_data(game_id):
    """Fetches Over/Under and Spread based on the specific upcoming GAME_ID."""
    try:
        target_id = str(game_id).strip()
        live_odds = odds.Odds().get_dict()
        game_list = live_odds.get("odds", [])

        for game in game_list:
            current_game_id = str(game.get("gameId", "")).strip()
            if current_game_id == target_id or current_game_id.endswith(target_id[-5:]):
                ou = float(game.get("overUnder", 230.0))
                spread = float(game.get("pointSpread", 0.0))
                return ou, abs(spread)
        return 230.0, 0.0
    except:
        return 230.0, 0.0


def get_next_game_context(team_id):
    """Scans the schedule for the first game on or after Today's date."""
    try:
        log = teamgamelog.TeamGameLog(
            team_id=team_id, season="2025-26"
        ).get_data_frames()[0]
        log.columns = [c.upper() for c in log.columns]

        # FIX: Explicit format to remove UserWarning and speed up parsing
        log["GAME_DATE_DT"] = pd.to_datetime(
            log["GAME_DATE"], format="%b %d, %Y"
        ).dt.date
        today = datetime.now().date()

        # Filter for upcoming games
        future_games = log[log["GAME_DATE_DT"] >= today].sort_values(
            "GAME_DATE_DT", ascending=True
        )
        if future_games.empty:
            return 115.0, None, 1

        next_game = future_games.iloc[0]
        matchup = next_game["MATCHUP"]
        g_id = next_game["GAME_ID"]
        is_home = 1 if "vs." in matchup.lower() else 0
        opp_abbrev = matchup.split(" ")[-1]

        # Fetch League Advanced Defense Stats
        ratings = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0]
        ratings.columns = [c.upper() for c in ratings.columns]

        opp_row = ratings[ratings["TEAM_ABBREVIATION"] == opp_abbrev]
        def_rating = (
            float(opp_row.iloc[0]["DEF_RATING"]) if not opp_row.empty else 115.0
        )

        print(f"📅 Next Matchup: {matchup} | Opp Def Rating: {def_rating}")
        return def_rating, g_id, is_home
    except:
        return 115.0, None, 1


def get_live_player_stats(player_name):
    """Fetches real-time data and context for the NEXT game."""
    search = players.find_players_by_full_name(player_name)
    if not search:
        return None
    p_id = search[0]["id"]

    try:
        # 1. Get History (Last 5 games)
        log_raw = playergamelog.PlayerGameLog(player_id=p_id, season="2025-26")
        df = log_raw.get_data_frames()[0]
        df.columns = [c.upper() for c in df.columns]
        if len(df) < 5:
            return None

        # FIX: Explicit format to remove UserWarning
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df = df.sort_values("GAME_DATE", ascending=True)
        df["FPTS"] = df.apply(calculate_custom_fpts, axis=1)

        # 2. Get Real-Time Context (Trade-Proof)
        current_team_id = get_current_team_id(p_id)
        if not current_team_id:
            return None

        opp_def, next_gid, next_home = get_next_game_context(current_team_id)
        ou, spread = get_live_vegas_data(next_gid) if next_gid else (230.0, 0.0)

        # Calculate rest based on today vs their last game played
        days_since_last = (datetime.now().date() - df.iloc[-1]["GAME_DATE"].date()).days

        return {
            "ROLLING_FPTS": df["FPTS"].tail(5).mean(),
            "ROLLING_MIN": df["MIN"].tail(5).mean(),
            "OPP_DEF_RATING": opp_def,
            "VS_OPP_AVG": df["FPTS"].mean(),
            "STAR_OUT": 0,
            "IS_STARTER": 1,
            "IS_HOME": next_home,
            "DAYS_REST": days_since_last,
            "IS_B2B": 1 if days_since_last == 1 else 0,
            "GAME_PACE": ou / 2.3,
        }
    except Exception as e:
        print(f"Fetch Error: {e}")
        return None


def predict_any_player_nn(name):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return "Error: Model or Scaler missing. Train the model first."

    # Force re-load to ensure we are using the 10-feature structure
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = [
        "ROLLING_FPTS",
        "ROLLING_MIN",
        "OPP_DEF_RATING",
        "VS_OPP_AVG",
        "STAR_OUT",
        "IS_STARTER",
        "IS_HOME",
        "DAYS_REST",
        "IS_B2B",
        "GAME_PACE",
    ]

    stats_dict = get_live_player_stats(name)
    if not stats_dict:
        return f"Error: No data for {name}."

    # Vectorize and Scale
    stats_row = np.array([stats_dict.get(f, 0.0) for f in features]).reshape(1, -1)
    stats_scaled = scaler.transform(stats_row)

    prediction = model.predict(stats_scaled, verbose=0)[0][0]
    return float(prediction)


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
