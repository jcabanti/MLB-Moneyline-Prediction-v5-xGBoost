from __future__ import annotations

from datetime import date
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from .db import get_conn
from .mlb_api import fetch_schedule_for_season
from .team_maps import TEAM_NAME_TO_ABBR
from .utils import to_datetime_flex


def parse_schedule_dates(schedule_dates: list, season: int) -> list[dict]:
    rows = []
    for date_block in schedule_dates:
        game_date = date_block.get("date")
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_team = home.get("team", {})
            away_team = away.get("team", {})
            home_score = home.get("score")
            away_score = away.get("score")
            home_name = home_team.get("name")
            away_name = away_team.get("name")
            home_abbr = TEAM_NAME_TO_ABBR.get(home_name)
            away_abbr = TEAM_NAME_TO_ABBR.get(away_name)
            venue = game.get("venue", {})
            rows.append({
                "game_id": game.get("gamePk"),
                "game_date": game_date,
                "season": season,
                "game_type": game.get("gameType"),
                "status": game.get("status", {}).get("detailedState"),
                "coded_status": game.get("status", {}).get("codedGameState"),
                "home_team_id": home_team.get("id"),
                "home_team": home_name,
                "home_team_abbr": home_abbr,
                "away_team_id": away_team.get("id"),
                "away_team": away_name,
                "away_team_abbr": away_abbr,
                "matchup": f"{away_abbr} @ {home_abbr}" if away_abbr and home_abbr else None,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": int(home_score > away_score) if home_score is not None and away_score is not None else None,
                "venue_id": venue.get("id"),
                "venue": venue.get("name"),
            })
    return rows


def ingest_games(start_season: int = 2021, end_season: int | None = None) -> pd.DataFrame:
    end_season = end_season or date.today().year
    rows = []
    for season in tqdm(range(start_season, end_season + 1), desc="Fetching seasons"):
        rows.extend(parse_schedule_dates(fetch_schedule_for_season(season), season))
    games = pd.DataFrame(rows)
    games_clean = games[
        (games["game_type"] == "R")
        & (games["coded_status"] == "F")  # Final only: in-progress games carry live scores
        & games["home_score"].notna()
        & games["away_score"].notna()
        & games["home_win"].notna()
    ].copy()
    games_clean["game_date"] = to_datetime_flex(games_clean["game_date"])
    games_clean = games_clean.drop_duplicates("game_id").sort_values(["game_date", "game_id"]).reset_index(drop=True)
    with get_conn() as conn:
        games_clean.to_sql("games_clean", conn, if_exists="replace", index=False)
    return games_clean


if __name__ == "__main__":
    df = ingest_games()
    print(f"Saved {len(df):,} rows to games_clean")
