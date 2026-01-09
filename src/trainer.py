import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

# --- CONFIGURATION ---
DATA_FILE = "../data/processed_stats.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "nba_v1_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print("Error: No data found. Run data_fetcher.py first.")
        return

    df = pd.read_csv(DATA_FILE)

    # 2. Define Features
    # Note: We now include STAR_OUT and IS_STARTER
    features = [
        "ROLLING_FPTS",
        "ROLLING_MIN",
        "OPP_DEF_RATING",
        "VS_OPP_AVG",
        "STAR_OUT",
        "IS_STARTER",
    ]

    # Drop rows where any of our training features are missing
    df = df.dropna(subset=features + ["TARGET_FPTS"])

    X = df[features]
    y = df["TARGET_FPTS"]

    # 3. Split Data (No shuffle for time-series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 4. Create the Pipeline (Standardization + Regression)
    # This automatically scales features to mean=0, variance=1
    model_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    # 5. Train
    print("Training model with normalized features...")
    model_pipeline.fit(X_train, y_train)

    # 6. Evaluate Performance
    predictions = model_pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("-" * 30)
    print(f"TRAINING COMPLETE")
    print(f"RMSE: {rmse:.2f} Fantasy Points")
    print(f"R^2 Score: {r2:.4f}")
    print("-" * 30)

    # 7. Interpret Scaled Coefficients
    # Since features are now on the same scale, the weight directly tells you importance
    regressor = model_pipeline.named_steps["regressor"]
    print("Feature Importance (Normalized Weights):")
    for feature, coef in zip(features, regressor.coef_):
        print(f"{feature}: {coef:.4f}")

    # 8. Save the Pipeline
    # This saves the scaler AND the model as one unit
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
