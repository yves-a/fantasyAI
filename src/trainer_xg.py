import pandas as pd
import xgboost as xgb
import os

# 1. Load your data
DATA_PATH = "../data/processed_stats.csv"
df = pd.read_csv(DATA_PATH)

# 2. MATCH THE COLUMN NAME
# Since your CSV uses 'FPTS', we set that as our target
target = "FPTS"

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

# 3. Data Cleaning
# Drop any rows where the target (FPTS) or features are missing
df = df.dropna(subset=[target] + features)

# 4. Initialize and Train
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    objective="reg:squarederror",
    random_state=42,
)

print(f"Training on {len(df)} games using target: {target}...")
model.fit(df[features], df[target])

# 5. Save the model
os.makedirs("../models", exist_ok=True)
model.save_model("../models/nba_xgboost_model.json")
print("Model successfully saved to ../models/nba_xgboost_model.json")
