from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd
import requests

from .config import ODDS_API_KEY
from .team_maps import ODDS_TEAM_NAME_TO_ABBR
from .utils import american_to_decimal, american_to_implied_prob, decimal_to_american, probability_to_american_odds


def fetch_current_mlb_odds(api_key: str | None = None) -> list:
    api_key = api_key or ODDS_API_KEY
    if not api_key:
        raise ValueError("ODDS_API_KEY is missing. Add it to .env or environment variables.")
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_current_mlb_odds(odds_json: list) -> pd.DataFrame:
    rows = []
    snapshot_time = datetime.now(timezone.utc).isoformat()
    for game in odds_json:
        home_name = game.get("home_team")
        away_name = game.get("away_team")
        home_abbr = ODDS_TEAM_NAME_TO_ABBR.get(home_name)
        away_abbr = ODDS_TEAM_NAME_TO_ABBR.get(away_name)
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    moneyline = outcome.get("price")
                    team_name = outcome.get("name")
                    if moneyline is None:
                        continue
                    rows.append({
                        "snapshot_time_utc": snapshot_time,
                        "odds_game_id": game.get("id"),
                        "commence_time_utc": game.get("commence_time"),
                        "game_date_utc": game.get("commence_time", "")[:10],
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_team_abbr": home_abbr,
                        "away_team_abbr": away_abbr,
                        "matchup": f"{away_abbr} @ {home_abbr}" if away_abbr and home_abbr else None,
                        "sportsbook_key": bookmaker.get("key"),
                        "sportsbook": bookmaker.get("title"),
                        "sportsbook_last_update_utc": bookmaker.get("last_update"),
                        "team": team_name,
                        "team_abbr": ODDS_TEAM_NAME_TO_ABBR.get(team_name),
                        "moneyline": moneyline,
                        "implied_probability": american_to_implied_prob(moneyline),
                    })
    return pd.DataFrame(rows)


def build_market_tables(odds_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = odds_df[odds_df["snapshot_time_utc"] == odds_df["snapshot_time_utc"].max()].copy()
    latest["side"] = latest.apply(
        lambda r: "home" if r["team_abbr"] == r["home_team_abbr"] else ("away" if r["team_abbr"] == r["away_team_abbr"] else None),
        axis=1,
    )
    game_book = latest.pivot_table(
        index=["snapshot_time_utc", "odds_game_id", "commence_time_utc", "game_date_utc", "matchup", "home_team_abbr", "away_team_abbr", "sportsbook_key", "sportsbook"],
        columns="side",
        values=["moneyline", "implied_probability"],
        aggfunc="first",
    ).reset_index()
    game_book.columns = ["_".join([str(x) for x in col if x]).strip("_") for col in game_book.columns]
    game_book["market_hold"] = game_book["implied_probability_home"] + game_book["implied_probability_away"] - 1
    denom = game_book["implied_probability_home"] + game_book["implied_probability_away"]
    game_book["no_vig_home_probability"] = game_book["implied_probability_home"] / denom
    game_book["no_vig_away_probability"] = game_book["implied_probability_away"] / denom
    game_book["commence_time_utc"] = pd.to_datetime(game_book["commence_time_utc"], utc=True)
    game_book["game_date_et"] = game_book["commence_time_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)

    game_book["decimal_home"] = game_book["moneyline_home"].apply(american_to_decimal)
    game_book["decimal_away"] = game_book["moneyline_away"].apply(american_to_decimal)

    consensus = game_book.groupby([
        "snapshot_time_utc", "odds_game_id", "commence_time_utc", "game_date_utc", "game_date_et", "matchup", "home_team_abbr", "away_team_abbr"
    ]).agg(
        sportsbooks=("sportsbook", "nunique"),
        avg_home_moneyline=("moneyline_home", "mean"),
        avg_away_moneyline=("moneyline_away", "mean"),
        avg_market_hold=("market_hold", "mean"),
        consensus_home_no_vig_probability=("no_vig_home_probability", "mean"),
        consensus_away_no_vig_probability=("no_vig_away_probability", "mean"),
        best_home_decimal=("decimal_home", "max"),
        best_away_decimal=("decimal_away", "max"),
    ).reset_index()
    consensus["best_home_moneyline"] = consensus["best_home_decimal"].apply(decimal_to_american)
    consensus["best_away_moneyline"] = consensus["best_away_decimal"].apply(decimal_to_american)
    consensus["consensus_home_fair_moneyline"] = consensus["consensus_home_no_vig_probability"].apply(probability_to_american_odds)
    consensus["consensus_away_fair_moneyline"] = consensus["consensus_away_no_vig_probability"].apply(probability_to_american_odds)
    return game_book, consensus
