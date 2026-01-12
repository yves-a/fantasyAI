import pandas as pd
import time
import os
import numpy as np
from datetime import datetime
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    leaguedashplayerstats,
    leagueleaders,
    leaguegamefinder,
)

# --- CONFIGURATION ---
DATA_DIR = "../data"
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_stats.csv")


def get_team_map():
    return {t["abbreviation"]: t["id"] for t in teams.get_teams()}


def process_and_feature_engineer(file_path):
    """
    Consolidated pipeline: Standardizes columns, maps Game/Team IDs,
    calculates rest/pace/defense, and injects Star-Out/Usage Delta logic.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print("🚀 Starting consolidated data processing...")
    df = pd.read_csv(file_path)

    # 1. STANDARDIZE COLUMNS (Prevents KeyErrors)
    df.columns = df.columns.str.strip().str.upper()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    team_map = get_team_map()

    # 2. MAP TEAM & GAME IDs
    print("Standardizing Team and Game IDs...")
    df["TEAM_ABBREVIATION"] = df["MATCHUP"].str.split(" ").str[0]
    df["TEAM_ID"] = df["TEAM_ABBREVIATION"].map(team_map)

    for season in df["SEASON"].unique():
        print(f"Fetching master schedule for {season}...")
        finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = finder.get_data_frames()[0]
        games.columns = games.columns.str.upper()
        games["DATE_KEY"] = pd.to_datetime(games["GAME_DATE"]).dt.date
        lookup = games.set_index(["TEAM_ID", "DATE_KEY"])["GAME_ID"].to_dict()

        season_mask = df["SEASON"] == season
        df.loc[season_mask, "GAME_ID"] = df[season_mask].apply(
            lambda x: lookup.get((int(x["TEAM_ID"]), x["GAME_DATE"].date())), axis=1
        )

    # 3. BASIC FEATURE ENGINEERING (Rest, Home, B2B)
    print("Calculating rest days and home/away flags...")
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"])
    df["DAYS_REST"] = df.groupby("PLAYER_NAME")["GAME_DATE"].diff().dt.days.fillna(4)
    df["IS_B2B"] = (df["DAYS_REST"] == 1).astype(int)
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x.lower() else 0)

    # 4. SEASON CONTEXT (Defense, Pace, and Star Mapping)
    # We identify stars from LeagueLeaders once to save API calls
    print("Analyzing star availability and league metrics...")
    leaders = leagueleaders.LeagueLeaders(season="2025-26").get_data_frames()[0]
    leaders.columns = leaders.columns.str.upper()

    # Map {TeamID: [Top 2 Star IDs]}
    team_stars = {}
    for _, row in leaders.head(60).iterrows():
        t_id = int(row["TEAM_ID"])
        if t_id not in team_stars:
            team_stars[t_id] = []
        if len(team_stars[t_id]) < 2:
            team_stars[t_id].append(int(row["PLAYER_ID"]))

    # Group by game to see which players actually recorded stats
    game_rosters = df.groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"].apply(set).to_dict()

    # 5. DEFENSE, PACE, STAR_OUT & USAGE DELTA
    for season in df["SEASON"].unique():
        print(f"Applying advanced metrics for {season}...")
        # Fetch Season Stats
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced", season=season
        ).get_data_frames()[0]
        stats.columns = stats.columns.str.upper()
        def_map = dict(zip(stats["TEAM_ID"], stats["DEF_RATING"]))
        pace_map = dict(zip(stats["TEAM_ID"], stats["PACE"]))

        season_mask = df["SEASON"] == season
        df.loc[season_mask, "OPP_ABBR"] = (
            df[season_mask]["MATCHUP"].str.split(" ").str[-1]
        )
        df.loc[season_mask, "OPP_TEAM_ID"] = df[season_mask]["OPP_ABBR"].map(team_map)
        df.loc[season_mask, "OPP_DEF_RATING"] = df[season_mask]["OPP_TEAM_ID"].map(
            def_map
        )

        # Game Pace Logic
        df.loc[season_mask, "GAME_PACE"] = df[season_mask].apply(
            lambda x: (
                pace_map.get(x["TEAM_ID"], 100) + pace_map.get(x["OPP_TEAM_ID"], 100)
            )
            / 2,
            axis=1,
        )

    # 6. STAR_OUT & USAGE BUMP
    print("Finalizing Star-Out and Usage Delta flags...")

    def check_star(row):
        t_id, g_id = int(row["TEAM_ID"]), row["GAME_ID"]
        roster = game_rosters.get((g_id, t_id), set())
        for s_id in team_stars.get(t_id, []):
            # If star is not in the box score and is not the current player being evaluated
            if s_id not in roster and s_id != row["PLAYER_ID"]:
                return 1
        return 0

    df["STAR_OUT"] = df.apply(check_star, axis=1)

    # Calculate Usage Bump based on USG comparison
    df["GAME_USG"] = df["FGA"] + (0.44 * df["FTA"]) + df["TOV"]
    usg_splits = df.groupby(["PLAYER_NAME", "STAR_OUT"])["GAME_USG"].mean().unstack()

    def get_delta(row):
        if row["STAR_OUT"] == 1:
            try:
                s_in, s_out = (
                    usg_splits.loc[row["PLAYER_NAME"], 0],
                    usg_splits.loc[row["PLAYER_NAME"], 1],
                )
                return s_out - s_in if (pd.notnull(s_in) and pd.notnull(s_out)) else 4.5
            except:
                return 4.5
        return 0.0

    df["USAGE_DELTA"] = df.apply(get_delta, axis=1)

    # Final Cleanup
    df.drop(columns=["GAME_USG", "DATE_KEY"], inplace=True, errors="ignore")
    df.to_csv(file_path, index=False)
    print(f"CLEANUP COMPLETE. Processed {len(df)} rows.")


if __name__ == "__main__":
    process_and_feature_engineer(OUTPUT_FILE)
