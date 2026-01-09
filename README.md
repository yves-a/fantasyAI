# NBA Fantasy AI Predictor

This project uses machine learning to project NBA player performance for the 2025-26 season based on historical game logs, defensive ratings, and teammate context.

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