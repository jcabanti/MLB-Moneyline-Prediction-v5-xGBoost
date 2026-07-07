"""Daily V5 hybrid report.

Pipeline: fetch odds + schedule (with lineups) -> build V5 features for the
slate -> hybrid model win probabilities -> edge vs de-vigged consensus,
EV at the best available book price, quarter-Kelly stake fraction, lineup
and starter confirmation flags -> CSV + markdown report.
"""
from __future__ import annotations

import argparse
from datetime import date

import joblib
import numpy as np
import pandas as pd

from .build_features_v5 import build_game_features
from .config import MODELS_DIR, REPORTS_DIR
from .db import get_conn
from .mlb_api import fetch_schedule_for_date
from .odds import build_market_tables, fetch_current_mlb_odds, parse_current_mlb_odds
from .team_maps import TEAM_NAME_TO_ABBR
from .train_v5 import MODEL_PATH, predict_hybrid
from .utils import probability_to_american_odds

EDGE_STRONG, EDGE_PLAY, EDGE_LEAN = 0.05, 0.03, 0.015
KELLY_FRACTION = 0.25


def parse_daily_schedule(schedule_dates: list) -> pd.DataFrame:
    rows = []
    for date_block in schedule_dates:
        game_date = date_block.get("date")
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            home_team, away_team = home.get("team", {}), away.get("team", {})
            home_probable, away_probable = home.get("probablePitcher", {}), away.get("probablePitcher", {})
            home_name, away_name = home_team.get("name"), away_team.get("name")
            home_abbr, away_abbr = TEAM_NAME_TO_ABBR.get(home_name), TEAM_NAME_TO_ABBR.get(away_name)
            venue = game.get("venue", {})
            lineups = game.get("lineups", {}) or {}
            rows.append({
                "game_id": game.get("gamePk"),
                "game_date": game_date,
                "game_datetime_utc": game.get("gameDate"),
                "status": game.get("status", {}).get("detailedState"),
                "coded_status": game.get("status", {}).get("codedGameState"),
                "away_team": away_name, "home_team": home_name,
                "away_team_abbr": away_abbr, "home_team_abbr": home_abbr,
                "matchup": f"{away_abbr} @ {home_abbr}" if away_abbr and home_abbr else None,
                "away_probable_pitcher_id": away_probable.get("id"),
                "away_probable_pitcher": away_probable.get("fullName"),
                "home_probable_pitcher_id": home_probable.get("id"),
                "home_probable_pitcher": home_probable.get("fullName"),
                "venue_id": venue.get("id"), "venue": venue.get("name"),
                "home_lineup_posted": len(lineups.get("homePlayers") or []) >= 9,
                "away_lineup_posted": len(lineups.get("awayPlayers") or []) >= 9,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["season"] = pd.to_datetime(df["game_date"]).dt.year
    return df


def signal_label(edge: float, ev: float) -> str:
    if pd.isna(edge) or ev <= 0:
        return "Pass"
    if edge >= EDGE_STRONG:
        return "Strong"
    if edge >= EDGE_PLAY:
        return "Play"
    if edge >= EDGE_LEAN:
        return "Lean"
    return "Pass"


def run_daily_report(game_date: str | None = None) -> pd.DataFrame:
    game_date = game_date or date.today().isoformat()
    bundle = joblib.load(MODEL_PATH)

    odds_df = parse_current_mlb_odds(fetch_current_mlb_odds())
    market_game_book, market_consensus = build_market_tables(odds_df)
    today_games = parse_daily_schedule(fetch_schedule_for_date(game_date))
    if today_games.empty:
        print("No games scheduled.")
        return today_games

    today = today_games.merge(
        market_consensus,
        left_on=["game_date", "away_team_abbr", "home_team_abbr"],
        right_on=["game_date_et", "away_team_abbr", "home_team_abbr"],
        how="left", suffixes=("", "_market"))
    today["has_market_odds"] = today["odds_game_id"].notna()

    with get_conn() as conn:
        odds_df.to_sql("odds_snapshots", conn, if_exists="append", index=False)
        market_game_book.to_sql("market_game_book", conn, if_exists="replace", index=False)
        market_consensus.to_sql("market_consensus", conn, if_exists="replace", index=False)
        games_clean = pd.read_sql("SELECT * FROM games_clean", conn)
        starter_lines = pd.read_sql("SELECT * FROM starter_game_pitching_lines", conn)
        bullpen_team_game = pd.read_sql("SELECT * FROM bullpen_team_game", conn)

    feats = build_game_features(today, games_clean, starter_lines, bullpen_team_game)
    scores = predict_hybrid(bundle, feats)
    out = feats.merge(scores, on="game_id")

    out["model_away_win_prob"] = 1 - out["model_home_win_prob"]
    out["model_home_fair_moneyline"] = out["model_home_win_prob"].apply(probability_to_american_odds)
    out["model_away_fair_moneyline"] = out["model_away_win_prob"].apply(probability_to_american_odds)
    out["home_edge"] = out["model_home_win_prob"] - out["consensus_home_no_vig_probability"]
    out["away_edge"] = out["model_away_win_prob"] - out["consensus_away_no_vig_probability"]

    # EV and quarter-Kelly at the best available book price
    out["home_ev"] = out["model_home_win_prob"] * out["best_home_decimal"] - 1
    out["away_ev"] = out["model_away_win_prob"] * out["best_away_decimal"] - 1
    out["home_kelly"] = ((out["model_home_win_prob"] * out["best_home_decimal"] - 1)
                         / (out["best_home_decimal"] - 1)).clip(lower=0)
    out["away_kelly"] = ((out["model_away_win_prob"] * out["best_away_decimal"] - 1)
                         / (out["best_away_decimal"] - 1)).clip(lower=0)

    pick_home = out["home_ev"].fillna(-9) >= out["away_ev"].fillna(-9)
    out["recommended_side"] = np.where(pick_home, out["home_team_abbr"], out["away_team_abbr"])
    out["rec_edge"] = np.where(pick_home, out["home_edge"], out["away_edge"])
    out["rec_ev"] = np.where(pick_home, out["home_ev"], out["away_ev"])
    out["rec_best_moneyline"] = np.where(pick_home, out["best_home_moneyline"], out["best_away_moneyline"])
    out["rec_quarter_kelly_pct"] = np.where(pick_home, out["home_kelly"], out["away_kelly"]) * KELLY_FRACTION
    out["signal"] = [signal_label(e, v) for e, v in zip(out["rec_edge"], out["rec_ev"])]
    out["starter_confirmed_both"] = out["home_probable_pitcher_id"].notna() & out["away_probable_pitcher_id"].notna()
    out["lineups_posted_both"] = out["home_lineup_posted"] & out["away_lineup_posted"]

    output_cols = [
        "game_id", "game_date", "game_datetime_utc", "matchup", "away_team_abbr", "home_team_abbr",
        "away_probable_pitcher", "home_probable_pitcher", "venue", "sportsbooks",
        "starter_confirmed_both", "home_lineup_posted", "away_lineup_posted", "lineups_posted_both",
        "consensus_away_no_vig_probability", "consensus_home_no_vig_probability",
        "best_away_moneyline", "best_home_moneyline",
        "model_away_win_prob", "model_home_win_prob",
        "model_away_fair_moneyline", "model_home_fair_moneyline",
        "skellam_home_win_prob", "classifier_home_win_prob",
        "mu_away_runs", "mu_home_runs", "model_total_runs", "home_minus_1_5_prob", "away_plus_1_5_prob",
        "away_edge", "home_edge", "away_ev", "home_ev",
        "recommended_side", "rec_best_moneyline", "rec_edge", "rec_ev", "rec_quarter_kelly_pct", "signal",
    ]
    daily = out[out["has_market_odds"]][output_cols].copy().sort_values("rec_ev", ascending=False)
    if daily.empty:
        print("No model-ready games with odds.")
        return daily

    with get_conn() as conn:
        daily.assign(report_date=game_date).to_sql("daily_predictions_v5_hybrid", conn, if_exists="replace", index=False)
    csv_path = REPORTS_DIR / f"daily_predictions_v5_hybrid_{game_date}.csv"
    daily.to_csv(csv_path, index=False)
    write_markdown_report(daily, game_date)
    print(f"Saved {len(daily):,} V5 hybrid predictions to {csv_path}")
    return daily


def write_markdown_report(daily: pd.DataFrame, game_date: str) -> None:
    lines = [f"# MLB V5 Hybrid Report — {game_date}", ""]
    actionable = daily[daily["signal"] != "Pass"]
    lines.append(f"{len(daily)} games priced · {len(actionable)} actionable · "
                 f"model = GBT Poisson runs -> NegBin/Skellam layer blended with GBT classifier")
    lines.append("")
    if actionable.empty:
        lines.append("No positive-EV edges above threshold today.")
    for _, r in actionable.iterrows():
        conf = []
        if not r["starter_confirmed_both"]:
            conf.append("starter TBD")
        if not r["lineups_posted_both"]:
            conf.append("lineups not posted")
        conf_note = f" — WAIT: {', '.join(conf)}" if conf else " — confirmed"
        ml = int(r["rec_best_moneyline"]) if pd.notna(r["rec_best_moneyline"]) else "?"
        lines.append(
            f"- **{r['signal']}** {r['recommended_side']} ({r['matchup']}) @ {ml:+} best price · "
            f"model {r['model_home_win_prob']:.1%} home · edge {r['rec_edge']:+.1%} · EV {r['rec_ev']:+.1%} · "
            f"1/4-Kelly {r['rec_quarter_kelly_pct']:.1%} of bankroll · total {r['model_total_runs']:.1f}{conf_note}")
    lines.append("")
    lines.append("_Model signal, not betting advice. Stakes assume best listed price is available._")
    (REPORTS_DIR / f"report_v5_{game_date}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD. Defaults to today.")
    args = parser.parse_args()
    run_daily_report(args.date)
