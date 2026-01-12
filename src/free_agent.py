import os
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime
from dotenv import load_dotenv
from espn_api.basketball import League
from nba_api.live.nba.endpoints import scoreboard
from predictor_nn import predict_any_player_nn

load_dotenv()

LEAGUE_ID = int(os.getenv("LEAGUE_ID"))
SWID = os.getenv("SWID")
ESPN_S2 = os.getenv("ESPN_S2")
YEAR = 2026

TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = f"../data/waiver_scout_{TODAY_STR}.csv"

# Pre-load model to prevent warnings and speed up scanning
MODEL = tf.keras.models.load_model("../models/nba_nn_model.keras")
SCALER = joblib.load("../models/nn_scaler.pkl")


def get_nba_teams_playing_today():
    """Returns a set of NBA team abbreviations (LAL, GSW, etc.) playing today."""
    try:
        games = scoreboard.ScoreBoard().games.get_dict()
        active_teams = set()
        for game in games:
            active_teams.add(game["homeTeam"]["teamTricode"])
            active_teams.add(game["awayTeam"]["teamTricode"])
        print(f"🏀 NBA Teams Active Today: {', '.join(sorted(active_teams))}")
        return active_teams
    except Exception as e:
        print(f"Error fetching NBA Scoreboard: {e}")
        return set()


def run_waiver_scanner():
    try:
        league = League(league_id=LEAGUE_ID, year=YEAR, espn_s2=ESPN_S2, swid=SWID)

        # 1. Get Live NBA Schedule
        teams_active = get_nba_teams_playing_today()
        if not teams_active:
            print("No NBA games detected for today.")
            return

        print("Scanning Free Agents...")
        free_agents = league.free_agents(size=150)

        if os.path.exists(LOG_FILE):
            processed_set = set(pd.read_csv(LOG_FILE)["Player"].tolist())
        else:
            processed_set = set()

        for player in free_agents:
            # 2. Skip if already done
            if player.name in processed_set:
                continue

            # 3. FIX: Check Injury Status
            # Skip if status is not 'ACTIVE'. None usually means healthy.
            if player.injuryStatus not in ["ACTIVE", "DAY_TO_DAY"]:
                continue

            # 4. FIX: Check if Team is Playing Today
            if player.proTeam not in teams_active:
                continue

            print(f"Analyzing {player.name}...")
            prediction = predict_any_player_nn(player.name, model=MODEL, scaler=SCALER)

            if isinstance(prediction, (int, float)):
                new_row = pd.DataFrame(
                    [
                        {
                            "Player": player.name,
                            "Pos": player.position,
                            "Team": player.proTeam,
                            "Proj": round(prediction, 2),
                            "Avg": player.avg_points,
                            "Diff": round(prediction - player.avg_points, 2),
                            "Status": player.injuryStatus or "Healthy",
                        }
                    ]
                )

                # Append to file
                new_row.to_csv(
                    LOG_FILE, mode="a", index=False, header=not os.path.exists(LOG_FILE)
                )
                processed_set.add(player.name)

        if os.path.exists(LOG_FILE):
            final_df = pd.read_csv(LOG_FILE).sort_values(by="Proj", ascending=False)
            print("\n" + "=" * 60 + "\nTOP WAIVER PLAYS TONIGHT\n" + "=" * 60)
            print(final_df.head(10).to_string(index=False))

    except Exception as e:
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    run_waiver_scanner()
