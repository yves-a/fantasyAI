import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA_FILE = "../data/processed_stats.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "nba_nn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "nn_scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_neural_network():
    if not os.path.exists(DATA_FILE):
        print("Error: No data found.")
        return

    df = pd.read_csv(DATA_FILE)

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

    df = df.dropna(subset=features + ["TARGET_FPTS"])
    X = df[features].values
    y = df["TARGET_FPTS"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(len(features),)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"],
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    print(f"Training Neural Network on {len(features)} features...")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    predictions = model.predict(X_test_scaled).flatten()
    r2 = r2_score(y_test, predictions)

    print("-" * 30)
    print(f"Final R^2 Score: {r2:.4f}")
    print("-" * 30)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


if __name__ == "__main__":
    train_neural_network()
