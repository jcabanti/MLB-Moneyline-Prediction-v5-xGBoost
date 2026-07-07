from __future__ import annotations

import requests


def fetch_schedule_for_season(season: int) -> list:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "season": season, "gameType": "R", "hydrate": "team,venue"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("dates", [])


def fetch_schedule_for_date(game_date: str) -> list:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": game_date, "gameType": "R", "hydrate": "team,venue,probablePitcher,lineups"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("dates", [])


def fetch_game_boxscore(game_id: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()
