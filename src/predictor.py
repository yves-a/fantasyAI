import pandas as pd
import joblib
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# --- CONFIGURATION ---
MODEL_PATH = "../models/nba_v1_model.pkl"
DATA_FILE = "../data/processed_stats.csv"


def get_live_player_stats(player_name):
    """Fallback function to get data for players NOT in your CSV."""
    print(f"Player not in local data. Fetching live stats for {player_name}...")

    search = players.find_players_by_full_name(player_name)
    if not search:
        return None
    p_id = search[0]["id"]

    # Fetch latest 5 games from the API
    log = playergamelog.PlayerGameLog(player_id=p_id, season="2025-26")
    df = log.get_data_frames()[0]

    if len(df) < 5:
        return None

    # Calculate your custom FPTS (reuse your function here)
    # Then calculate ROLLING_FPTS and ROLLING_MIN from the head(5)
    # For VS_OPP_AVG and DEF_RATING, you can use season averages as a proxy
    return {
        "ROLLING_FPTS": df["PTS"].head(5).mean(),  # Simplified for example
        "ROLLING_MIN": df["MIN"].head(5).mean(),
        "OPP_DEF_RATING": 115.0,  # Average league defense as a placeholder
        "VS_OPP_AVG": df["PTS"].mean(),
    }


def predict_any_player(name):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_FILE)

    features = [
        "ROLLING_FPTS",
        "ROLLING_MIN",
        "OPP_DEF_RATING",
        "VS_OPP_AVG",
        "STAR_OUT",
        "IS_STARTER",
    ]

    player_data = df[df["PLAYER_NAME"] == name]

    if not player_data.empty:
        stats = player_data.iloc[-1]
        # Create DataFrame with the correct column names
        X = pd.DataFrame([stats[features].values], columns=features)
    else:
        live_stats = get_live_player_stats(name)
        if not live_stats:
            return "Player data unavailable."

        # Ensure the live_stats dictionary/DF has all 6 keys
        # If live_stats doesn't have STAR_OUT, we add them manually:
        live_stats["STAR_OUT"] = 0
        live_stats["IS_STARTER"] = 1

        X = pd.DataFrame([live_stats], columns=features)

    return model.predict(X)[0]


if __name__ == "__main__":
    name = input("Enter player name: ")
    print(f"Prediction: {predict_any_player(name):.2f}")
