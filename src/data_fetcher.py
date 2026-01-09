import pandas as pd
import time
import os
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    commonteamroster,
    leaguedashplayerstats,
    boxscoretraditionalv3,
)

# --- CONFIGURATION ---
SEASON = "2025-26"
DATA_DIR = "../data"
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_stats.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---


def get_defensive_ratings():
    """Fetches current league-wide defensive ratings."""
    print("Fetching team defensive ratings...")
    stats = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Advanced", season=SEASON
    ).get_data_frames()[0]
    return dict(zip(stats["TEAM_ID"], stats["DEF_RATING"]))


def get_dynamic_stars(threshold=0.25):
    """
    Identifies 'Stars' per team based on Usage Rate (USG%).
    threshold=0.25 means the player handles 25%+ of team possessions.
    """
    print(f"Dynamically identifying stars (USG% > {threshold*100}%)...")
    adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        measure_type_detailed_defense="Advanced", season=SEASON
    ).get_data_frames()[0]

    # Filter for players with at least 5 games played to avoid outliers
    stars_df = adv_stats[(adv_stats["GP"] >= 5) & (adv_stats["USG_PCT"] >= threshold)]

    dynamic_stars = {}
    for team_id in adv_stats["TEAM_ID"].unique():
        team_stars = stars_df[stars_df["TEAM_ID"] == team_id]["PLAYER_ID"].tolist()
        dynamic_stars[team_id] = team_stars
    return dynamic_stars


def get_top_players_per_team(limit=5):
    """Gets the top N players from every team's roster."""
    all_nba_teams = teams.get_teams()
    player_pool = []
    for team in all_nba_teams:
        print(f"Getting roster for {team['full_name']}...")
        roster = commonteamroster.CommonTeamRoster(
            team_id=team["id"], season=SEASON
        ).get_data_frames()[0]
        # Keep PLAYER_ID, Name, and TeamID
        team_players = roster.head(limit)[
            ["PLAYER_ID", "PLAYER", "TeamID"]
        ].values.tolist()
        player_pool.extend(team_players)
        time.sleep(0.6)
    return player_pool


def calculate_custom_fpts(row):
    """Your custom league scoring logic."""
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


def fetch_all_data():
    all_teams_map = {t["abbreviation"]: t["id"] for t in teams.get_teams()}
    def_ratings = get_defensive_ratings()
    dynamic_stars = get_dynamic_stars(threshold=0.26)
    player_pool = get_top_players_per_team(limit=5)

    all_player_data = []

    for p_id, full_name, t_id in player_pool:
        print(f"Processing {full_name}...")
        try:
            log = playergamelog.PlayerGameLog(
                player_id=p_id, season=SEASON
            ).get_data_frames()[0]
            if log.empty or len(log) < 10:
                continue

            log["PLAYER_NAME"] = full_name
            log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
            log = log.sort_values("GAME_DATE")

            # --- EXTRACT BOX SCORE FEATURES (IS_STARTER / STAR_OUT) ---
            is_starter_flags = []
            star_out_flags = []

            # We limit to the most recent 15 games to keep the script from taking hours
            game_subset = log.tail(15)
            for _, row in game_subset.iterrows():
                try:
                    box = boxscoretraditionalv3.boxscoretraditionalv3(
                        game_id=row["Game_ID"]
                    ).get_data_frames()[0]

                    # Feature: Is Starter?
                    p_row = box[box["PLAYER_ID"] == p_id]
                    is_starter_flags.append(
                        1
                        if not p_row.empty and p_row["START_POSITION"].iloc[0] != ""
                        else 0
                    )

                    # Feature: Star Out? (Checks if team's dynamic stars missed the game)
                    team_stars = dynamic_stars.get(t_id, [])
                    missing = 0
                    for s_id in team_stars:
                        if s_id != p_id and s_id not in box["PLAYER_ID"].values:
                            missing = 1
                    star_out_flags.append(missing)
                except:
                    is_starter_flags.append(1)  # Default fallback
                    star_out_flags.append(0)
                time.sleep(0.4)  # Respect API limits

            # Match log length to the subset we just processed
            final_log = game_subset.copy()
            final_log["IS_STARTER"] = is_starter_flags
            final_log["STAR_OUT"] = star_out_flags

            # --- CALCULATE CORE FEATURES ---
            final_log["FPTS"] = final_log.apply(calculate_custom_fpts, axis=1)
            final_log["OPP_ABBR"] = final_log["MATCHUP"].apply(
                lambda x: x.split(" ")[-1]
            )
            final_log["OPP_TEAM_ID"] = final_log["OPP_ABBR"].map(all_teams_map)
            final_log["OPP_DEF_RATING"] = final_log["OPP_TEAM_ID"].map(def_ratings)

            # Rolling Metrics
            final_log["ROLLING_FPTS"] = (
                final_log["FPTS"].shift(1).rolling(window=5).mean()
            )
            final_log["ROLLING_MIN"] = (
                final_log["MIN"].shift(1).rolling(window=5).mean()
            )

            # Historical Matchup
            final_log["VS_OPP_AVG"] = (
                final_log.groupby("OPP_ABBR")["FPTS"]
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(final_log["FPTS"].expanding().mean())
            )

            final_log["TARGET_FPTS"] = final_log["FPTS"]
            all_player_data.append(final_log)

            time.sleep(0.8)

        except Exception as e:
            print(f"Error with {full_name}: {e}")

    if all_player_data:
        final_df = pd.concat(all_player_data).dropna(
            subset=["ROLLING_FPTS", "TARGET_FPTS"]
        )
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved {len(final_df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_all_data()
