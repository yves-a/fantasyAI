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
SEASONS_TO_FETCH = ["2023-24", "2024-25", "2025-26"]
DATA_DIR = "../data"
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_stats.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---


def get_defensive_ratings(season):
    """Fetches league-wide defensive ratings for a specific season."""
    print(f"Fetching team defensive ratings for {season}...")
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced", season=season
        ).get_data_frames()[0]
        return dict(zip(stats["TEAM_ID"], stats["DEF_RATING"]))
    except Exception as e:
        print(f"Error fetching ratings for {season}: {e}")
        return {}


def get_dynamic_stars(season, threshold=0.26):
    """Identifies 'Stars' per team based on Usage Rate (USG%) for a specific season."""
    print(f"Identifying stars for {season} (USG% > {threshold*100}%)...")
    try:
        adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense="Advanced", season=season
        ).get_data_frames()[0]
        stars_df = adv_stats[
            (adv_stats["GP"] >= 5) & (adv_stats["USG_PCT"] >= threshold)
        ]

        dynamic_stars = {}
        for team_id in adv_stats["TEAM_ID"].unique():
            team_stars = stars_df[stars_df["TEAM_ID"] == team_id]["PLAYER_ID"].tolist()
            dynamic_stars[team_id] = team_stars
        return dynamic_stars
    except Exception as e:
        print(f"Error fetching stars for {season}: {e}")
        return {}


def get_top_players_per_team(season, limit=5):
    """Gets the top N players from every team's roster for a specific season."""
    all_nba_teams = teams.get_teams()
    player_pool = []
    for team in all_nba_teams:
        try:
            print(f"Getting {season} roster for {team['full_name']}...")
            roster = commonteamroster.CommonTeamRoster(
                team_id=team["id"], season=season
            ).get_data_frames()[0]
            team_players = roster.head(limit)[
                ["PLAYER_ID", "PLAYER", "TeamID"]
            ].values.tolist()
            player_pool.extend(team_players)
            time.sleep(0.6)
        except:
            continue
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


def fetch_season_data(target_season):
    """Fetches all player data for a single specific season."""
    all_teams_map = {t["abbreviation"]: t["id"] for t in teams.get_teams()}
    def_ratings = get_defensive_ratings(target_season)
    dynamic_stars = get_dynamic_stars(target_season)
    player_pool = get_top_players_per_team(target_season, limit=5)

    season_player_data = []

    for p_id, full_name, t_id in player_pool:
        print(f"Processing {full_name} ({target_season})...")
        try:
            log = playergamelog.PlayerGameLog(
                player_id=p_id, season=target_season
            ).get_data_frames()[0]
            if log.empty or len(log) < 10:
                continue

            log["PLAYER_NAME"] = full_name
            log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
            log = log.sort_values("GAME_DATE")

            is_starter_flags = []
            star_out_flags = []

            # Process a larger subset for historical seasons (e.g., last 40 games)
            # to give the NN more data to learn from.
            game_subset = log.tail(40) if "2025" not in target_season else log.tail(15)

            for _, row in game_subset.iterrows():
                try:
                    # Using TraditionalV2 for better stability
                    box = boxscoretraditionalv3.boxscoretraditionalv3(
                        game_id=row["Game_ID"]
                    ).get_data_frames()[0]

                    p_row = box[box["PLAYER_ID"] == p_id]
                    is_starter_flags.append(
                        1
                        if not p_row.empty and p_row["START_POSITION"].iloc[0] != ""
                        else 0
                    )

                    team_stars = dynamic_stars.get(t_id, [])
                    missing = 0
                    for s_id in team_stars:
                        if s_id != p_id and s_id not in box["PLAYER_ID"].values:
                            missing = 1
                    star_out_flags.append(missing)
                except:
                    is_starter_flags.append(1)
                    star_out_flags.append(0)
                time.sleep(0.4)

            final_log = game_subset.copy()
            final_log["IS_STARTER"] = is_starter_flags
            final_log["STAR_OUT"] = star_out_flags
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
            final_log["SEASON"] = target_season
            season_player_data.append(final_log)
            time.sleep(0.5)

        except Exception as e:
            print(f"Error with {full_name}: {e}")

    if season_player_data:
        return pd.concat(season_player_data)
    return pd.DataFrame()


def run_multi_season_pipeline():
    """Main orchestrator for multi-season data collection."""
    master_list = []

    for season in SEASONS_TO_FETCH:
        print(f"\nSTARTING SEASON: {season}")
        season_df = fetch_season_data(season)
        if not season_df.empty:
            master_list.append(season_df)
        print(f"Finished {season}. Resting...")
        time.sleep(5)

    if master_list:
        final_df = pd.concat(master_list).dropna(subset=["ROLLING_FPTS", "TARGET_FPTS"])
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Saved {len(final_df)} total rows to {OUTPUT_FILE}")


def upgrade_existing_data(file_path):
    df = pd.read_csv(file_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Sort by player and date to calculate rest correctly
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"])

    # Feature: IS_HOME
    # Matchup contains '@' for Away and 'vs.' for Home
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    # Feature: DAYS_REST
    # Calculate difference between current and previous game for each player
    df["DAYS_REST"] = df.groupby("PLAYER_NAME")["GAME_DATE"].diff().dt.days

    # Fill first game of season with 4 days (typical rest)
    df["DAYS_REST"] = df["DAYS_REST"].fillna(4)

    # Feature: IS_B2B (Back-to-Back)
    df["IS_B2B"] = (df["DAYS_REST"] == 1).astype(int)

    # We fetch Pace for every season currently in your CSV
    seasons_in_data = df["SEASON"].unique()
    pace_master_map = {}

    for season in seasons_in_data:
        print(f"Fetching pace ratings for {season}...")
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced", season=season
        ).get_data_frames()[0]
        # Create a dictionary of {TeamID: PaceValue}
        pace_master_map[season] = dict(zip(stats["TEAM_ID"], stats["PACE"]))
        time.sleep(0.6)  # API throttle

    def get_game_pace(row):
        season_pace = pace_master_map.get(row["SEASON"], {})

        # The NBA API is inconsistent with Team ID naming.
        # We check the row for 'TeamID', 'TEAM_ID', or 'Team_ID' dynamically.
        p_team_id = row.get("TeamID") or row.get("TEAM_ID") or row.get("Team_ID")
        o_team_id = row.get("OPP_TEAM_ID") or row.get("Opponent_Team_ID")

        # Predicted Pace = (Team A Pace + Team B Pace) / 2
        team_p = season_pace.get(p_team_id, 100.0)
        opp_p = season_pace.get(o_team_id, 100.0)

        return (team_p + opp_p) / 2

    print("Calculating game-specific pace ratings...")
    df["GAME_PACE"] = df.apply(get_game_pace, axis=1)

    # Save back to CSV
    df.to_csv(file_path, index=False)
    print(f"Successfully added IS_HOME, DAYS_REST, and IS_B2B to {file_path}")


if __name__ == "__main__":
    # run_multi_season_pipeline()
    upgrade_existing_data("../data/processed_stats.csv")
