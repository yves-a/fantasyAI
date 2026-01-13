import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
DATA_FILE = "../data/processed_stats.csv"
MODEL_PATH = "../models/nba_nn_model.keras"
SCALER_PATH = "../models/nn_scaler.pkl"


def run_backtest():
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_PATH):
        print(
            "Error: Required files missing. Ensure data is processed and model is trained."
        )
        return

    # Load Assets
    df = pd.read_csv(DATA_FILE)
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Must match your Trainer exactly
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

    # We test on the 15% of data the model (theoretically) didn't see during training
    test_size = int(len(df) * 0.15)
    test_df = df.tail(test_size).copy()

    X = test_df[features].values
    y_actual = test_df["TARGET_FPTS"].values

    # Scale and Predict
    X_scaled = scaler.transform(X)
    # Using model() directly is faster for inference
    y_pred = model(X_scaled, training=False).numpy().flatten()

    # Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    # Accuracy: % of games predicted within 6 fantasy points (about 1.5-2 buckets/assists)
    within_6 = np.mean(np.abs(y_actual - y_pred) <= 6) * 100

    print("\n" + "═" * 45)
    print(f"BACKTEST RESULTS ({test_size} Games)")
    print("═" * 45)
    print(f"R² Score:            {r2:.4f}")
    print(f"Avg. Point Error:    {mae:.2f} FPTS")
    print(f"Accuracy (±6 pts):   {within_6:.1f}%")
    print("═" * 45)

    # Identify the biggest "Hits" and "Misses"
    test_df["PRED"] = y_pred
    test_df["ERROR"] = test_df["PRED"] - test_df["TARGET_FPTS"]

    print("\nRecent Game Comparison:")
    print(
        test_df[["PLAYER_NAME", "TARGET_FPTS", "PRED", "ERROR"]]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_backtest()
