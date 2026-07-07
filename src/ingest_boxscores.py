"""Incremental data refresh so V5 features never go stale.

1. Refreshes the current season's rows in `games_clean` from the MLB schedule.
2. Finds completed games missing from `starter_game_pitching_lines` /
   `bullpen_team_game`, fetches their boxscores, and appends pitching lines.

Run daily before the report (wired into the GitHub Actions workflow).
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from .db import get_conn
from .ingest_games import parse_schedule_dates
from .mlb_api import fetch_game_boxscore, fetch_schedule_for_season
from .utils import ip_str_to_float, parse_pitcher_stat_value, to_datetime_flex


def refresh_current_season_games(season: int | None = None) -> pd.DataFrame:
    season = season or date.today().year
    rows = parse_schedule_dates(fetch_schedule_for_season(season), season)
    games = pd.DataFrame(rows)
    games_clean = games[(games["game_type"] == "R")
                        & (games["coded_status"] == "F")  # Final only: live games carry partial scores
                        & games["home_score"].notna() & games["away_score"].notna()
                        & games["home_win"].notna()].copy()
    games_clean["game_date"] = to_datetime_flex(games_clean["game_date"])
    games_clean = games_clean.drop_duplicates("game_id").sort_values(["game_date", "game_id"]).reset_index(drop=True)

    with get_conn() as conn:
        # heal any historical non-final rows that slipped in before this filter existed
        conn.execute("DELETE FROM games_clean WHERE coded_status != 'F'")
        conn.commit()
        existing = pd.read_sql("SELECT * FROM games_clean WHERE season != ?", conn, params=(season,))
        existing["game_date"] = to_datetime_flex(existing["game_date"])
        combined = pd.concat([existing, games_clean], ignore_index=True)
        combined = combined.drop_duplicates("game_id").sort_values(["game_date", "game_id"])
        combined.to_sql("games_clean", conn, if_exists="replace", index=False)
    print(f"games_clean refreshed: {len(games_clean)} completed {season} games, {len(combined)} total.")
    return games_clean


def parse_boxscore_pitching(game_id: int, game_date, season: int, box: dict) -> tuple[list[dict], list[dict]]:
    """Returns (starter_rows, bullpen_team_rows) for one game."""
    starter_rows, bullpen_rows = [], []
    for side in ["away", "home"]:
        team = box.get("teams", {}).get(side, {})
        team_abbr = team.get("team", {}).get("abbreviation")
        bp = {"ip": 0.0, "er": 0.0, "so": 0.0, "bb": 0.0, "hits": 0.0, "hr": 0.0, "pitches": 0.0, "n": 0}
        for player in team.get("players", {}).values():
            stats = player.get("stats", {}).get("pitching", {})
            if not stats:
                continue
            ip = ip_str_to_float(stats.get("inningsPitched", 0))
            pitches = parse_pitcher_stat_value(stats, "numberOfPitches", 0) or parse_pitcher_stat_value(stats, "pitchesThrown", 0)
            line = {
                "innings_pitched": ip,
                "earned_runs": parse_pitcher_stat_value(stats, "earnedRuns", 0),
                "strikeouts": parse_pitcher_stat_value(stats, "strikeOuts", 0),
                "walks": parse_pitcher_stat_value(stats, "baseOnBalls", 0),
                "hits_allowed": parse_pitcher_stat_value(stats, "hits", 0),
                "home_runs_allowed": parse_pitcher_stat_value(stats, "homeRuns", 0),
                "pitches_thrown": pitches,
            }
            if parse_pitcher_stat_value(stats, "gamesStarted", 0) >= 1:
                starter_rows.append({
                    "game_id": game_id, "side": side,
                    "pitcher_id": player.get("person", {}).get("id"),
                    "pitcher_name": player.get("person", {}).get("fullName"),
                    **line,
                })
            else:
                bp["ip"] += line["innings_pitched"]; bp["er"] += line["earned_runs"]
                bp["so"] += line["strikeouts"]; bp["bb"] += line["walks"]
                bp["hits"] += line["hits_allowed"]; bp["hr"] += line["home_runs_allowed"]
                bp["pitches"] += line["pitches_thrown"]; bp["n"] += 1
        bullpen_rows.append({
            "game_id": game_id, "game_date": str(pd.to_datetime(game_date).date()), "season": season,
            "team_abbr": team_abbr, "bullpen_ip": bp["ip"], "bullpen_er": bp["er"], "bullpen_so": bp["so"],
            "bullpen_bb": bp["bb"], "bullpen_hits": bp["hits"], "bullpen_hr": bp["hr"],
            "bullpen_pitches": bp["pitches"], "bullpen_pitchers_used": bp["n"],
        })
    return starter_rows, bullpen_rows


def ingest_missing_boxscores(limit: int | None = None) -> int:
    with get_conn() as conn:
        games = pd.read_sql("SELECT game_id, game_date, season FROM games_clean", conn)
        have = pd.read_sql("SELECT DISTINCT game_id FROM bullpen_team_game", conn)["game_id"]
    missing = games[~games["game_id"].isin(set(have))].sort_values("game_date")
    if limit:
        missing = missing.head(limit)
    if missing.empty:
        print("Pitching lines already up to date.")
        return 0

    starters, bullpens, failed = [], [], 0
    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="Fetching boxscores"):
        try:
            box = fetch_game_boxscore(int(row["game_id"]))
            s, b = parse_boxscore_pitching(int(row["game_id"]), row["game_date"], int(row["season"]), box)
            starters.extend(s)
            bullpens.extend(b)
        except Exception as exc:  # keep going; retry tomorrow
            failed += 1
            print(f"  boxscore {row['game_id']} failed: {exc}")

    with get_conn() as conn:
        if starters:
            pd.DataFrame(starters).to_sql("starter_game_pitching_lines", conn, if_exists="append", index=False)
        if bullpens:
            pd.DataFrame(bullpens).to_sql("bullpen_team_game", conn, if_exists="append", index=False)
    print(f"Ingested {len(missing) - failed} boxscores ({failed} failed) -> "
          f"{len(starters)} starter lines, {len(bullpens)} bullpen team-games.")
    return len(missing) - failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max boxscores to fetch this run.")
    args = parser.parse_args()
    refresh_current_season_games(args.season)
    ingest_missing_boxscores(args.limit)
