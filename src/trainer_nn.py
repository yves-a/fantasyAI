import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
DATA_FILE = "../data/processed_stats.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "nba_nn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "nn_scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_neural_network():
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print("Error: No data found. Run data_fetcher.py first.")
        return
    df = pd.read_csv(DATA_FILE)

    # 2. Define Features and Target
    features = [
        "ROLLING_FPTS",
        "ROLLING_MIN",
        "OPP_DEF_RATING",
        "VS_OPP_AVG",
        "STAR_OUT",
        "IS_STARTER",
    ]
    df = df.dropna(subset=features + ["TARGET_FPTS"])

    X = df[features].values
    y = df["TARGET_FPTS"].values

    # 3. Split and Scale (Mandatory for NNs)
    # Fit only on training data to prevent "Data Leakage"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Build Architecture (Multi-Layer Perceptron)
    model = Sequential(
        [
            # Input Layer + Hidden Layer 1: 64 neurons with ReLU
            layers.Dense(64, activation="relu", input_shape=(len(features),)),
            layers.Dropout(
                0.2
            ),  # Randomly disables 20% of neurons to prevent "memorizing" data
            # Hidden Layer 2: 32 neurons for complex pattern recognition
            layers.Dense(32, activation="relu"),
            # Output Layer: 1 node (Linear activation) to predict a continuous number
            layers.Dense(1),
        ]
    )

    # 5. Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",  # Mean Squared Error: standard for regression
        metrics=["mae"],  # Mean Absolute Error
    )

    # 6. Train the Model
    print("Training Neural Network...")
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.1,  # Check performance on a small holdout during training
        epochs=150,  # Passes through the entire dataset 150 times
        batch_size=32,  # Updates weights after every 32 games
        verbose=1,
    )

    # 7. Evaluate Performance
    predictions = model.predict(X_test_scaled).flatten()
    r2 = r2_score(y_test, predictions)

    print("-" * 30)
    print(f"NN TRAINING COMPLETE")
    print(f"R^2 Score: {r2:.4f}")
    print("-" * 30)

    # 8. Save the Model and Scaler
    # We must save the scaler so the predictor uses the same normalization
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    train_neural_network()
