import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# --- CONFIGURATION ---
MODEL_PATH = "../models/nba_nn_model.keras"
SCALER_PATH = "../models/nn_scaler.pkl"
DATA_FILE = "../data/processed_stats.csv"


def calculate_custom_fpts(row):
    """Custom league scoring logic (must match trainer/fetcher)."""
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


def get_live_player_stats(player_name):
    """Fetches real-time data from NBA API if player is missing from CSV."""
    print(f"Player not in local data. Fetching live stats for {player_name}...")

    search = players.find_players_by_full_name(player_name)
    if not search:
        print("Player not found in NBA database.")
        return None
    p_id = search[0]["id"]

    try:
        # Fetch latest games for current season
        log = playergamelog.PlayerGameLog(player_id=p_id, season="2025-26")
        df = log.get_data_frames()[0]

        if len(df) < 5:
            return None

        # Calculate FPTS for each of the last 5 games
        df["FPTS"] = df.apply(calculate_custom_fpts, axis=1)

        # Build feature dictionary
        return {
            "ROLLING_FPTS": df["FPTS"].head(5).mean(),
            "ROLLING_MIN": df["MIN"].head(5).mean(),
            "OPP_DEF_RATING": 115.0,  # Proxy: League average
            "VS_OPP_AVG": df["FPTS"].mean(),  # Proxy: Season average
            "STAR_OUT": 0,  # Proxy: 0
            "IS_STARTER": 1,  # Proxy: 1
        }
    except Exception as e:
        print(f"API Error: {e}")
        return None


def predict_any_player_nn(name):
    # 1. Load Model and Scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return "Error: NN Model or Scaler not found. Train the NN first."

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Setup Features
    features = [
        "ROLLING_FPTS",
        "ROLLING_MIN",
        "OPP_DEF_RATING",
        "VS_OPP_AVG",
        "STAR_OUT",
        "IS_STARTER",
    ]

    # 3. Check Local Data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        player_data = df[df["PLAYER_NAME"].str.lower() == name.lower()]
    else:
        player_data = pd.DataFrame()

    if not player_data.empty:
        print(f"Found {name} in local data.")
        stats_row = player_data.iloc[-1][features].values.reshape(1, -1)
    else:
        # 4. Fallback to API
        live_stats_dict = get_live_player_stats(name)
        if not live_stats_dict:
            return "Player data unavailable."

        # Convert dict to array in the correct feature order
        stats_row = np.array([live_stats_dict[f] for f in features]).reshape(1, -1)

    # 5. Scale and Predict
    # IMPORTANT: The NN was trained on scaled data. We must scale live data too.
    stats_scaled = scaler.transform(stats_row)
    prediction = model.predict(stats_scaled, verbose=0)[0][0]

    return prediction


if __name__ == "__main__":
    name = input("Enter player name for NN prediction: ")
    result = predict_any_player_nn(name)

    if isinstance(result, str):
        print(result)
    else:
        print("-" * 30)
        print(f"NN PREDICTION FOR {name}: {result:.2f} FPTS")
        print("-" * 30)
