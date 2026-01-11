# NBA Fantasy AI Predictor

This project uses machine learning to project NBA player performance for the 2025-26 season based on historical game logs, defensive ratings, and teammate context.

---

## Assumptions

### Historical Continuity (Stationarity)
* **Assumption:** Patterns found in the 2023–2025 seasons remain valid for the 2026 season.
* **Reality:** Changes in league rules, coaching philosophies, or team styles (e.g., a sudden league-wide increase in 3-point attempts) can make older data **"stale."**

### Proxy Reliability
* **Assumption:** Simplified metrics (proxies) like `IS_STARTER` or a static `100.0 Pace` correctly represent the true complexity of a game environment.
* **Reality:** A proxy for defensive rating might miss the impact of a specific **"shut-down" defender** who is uniquely capable of stopping your target player.


### Individual Independence
* **Assumption:** A player’s fantasy production is an isolated event that can be predicted without knowing their teammates' real-time performance.
* **Reality:** Basketball is a **zero-sum game**. If a teammate has a "career night" and takes more shots, there are fewer opportunities available for the player you are predicting.


### Linear Preprocessing (Scaling)
* **Assumption:** Normalizing all features to a standard scale (via `StandardScaler`) allows the Neural Network to weigh them appropriately.
* **Reality:** Non-linear relationships (like the difference between 0 and 1 days of rest being much more significant than 3 and 4 days) may be partially **"flattened"** by standard scaling.

### Distributional Consistency
* **Assumption:** Current player versions (e.g., 2026 Joel Embiid) belong to the same statistical distribution as their past versions.
* **Reality:** Age, injuries, or changes in team role can cause a player to **"break" the model** because they no longer play like the version of themselves the model memorized.

---

### Future Improvements

* Give more weight to recent games to indicate slump/hot streak
* Change proxies to dynamic data by fetching real time data from Vegas lines, opponent defensive ratings
* Investigate impact of the team on individual performance

---

### Features

* **Matchup Analysis**: Incorporates opponent defensive ratings and historical performance against specific teams.
* **Rolling Metrics**: Uses 5-game rolling windows for fantasy points and minutes to capture recent form.
* **Teammate Context**: Tracks "Star Out" scenarios where high-usage teammates are sidelined.
* **Automated Data Pipeline**: Dynamically identifies the top 5 players per team and fetches logs via the NBA API.
* **Scaling**: Implements a Scikit-Learn Pipeline with StandardScaler for normalized feature importance.

---

### Installation

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements.txt`

---

### Project Structure

* `data_fetcher.py`: Scrapes the NBA API for player logs, team ratings, and box score features.
* `trainer.py`: Processes the CSV data, trains a Linear Regression model, and saves a serialized pipeline.
* `predictor.py`: Facilitates single-player projections using the trained model and latest available data.
* `data/`: Directory containing the processed CSV dataset.
* `models/`: Directory containing the saved .pkl model.

---

### Usage

1. **Fetch Data**: Run `python src/data_fetcher.py` to build the training dataset.
2. **Train Model**: Run `python src/trainer.py` to generate the regression model and evaluate metrics (RMSE and R^2).
3. **Predict**: Use `python src/predictor.py` to output a specific player projection for the next game.

---

### Technical Specifications

* **Language**: Python 3.13
* **Model**: Linear Regression (Standardized)
* **Target Variable**: Custom League Fantasy Points (FPTS)
* **Pre-processing**: Features are normalized to mean 0 and variance 1 to ensure balanced feature weighting.