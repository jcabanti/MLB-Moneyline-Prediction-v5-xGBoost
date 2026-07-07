"""Grade saved V5 predictions once games are final.

Appends to `graded_predictions_v5` (running history, not replaced) and prints
accuracy, log loss, and ROI (flat 1u and quarter-Kelly) at the recorded best
price for the recommended side.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .db import get_conn
from .utils import american_to_decimal, to_datetime_flex


def grade_predictions(game_date: str | None = None) -> pd.DataFrame:
    game_date = game_date or (date.today() - timedelta(days=1)).isoformat()
    with get_conn() as conn:
        try:
            daily = pd.read_sql("SELECT * FROM daily_predictions_v5_hybrid", conn)
        except Exception as exc:
            raise RuntimeError("daily_predictions_v5_hybrid missing. Run the daily report first.") from exc
        games = pd.read_sql("SELECT game_id, home_score, away_score, home_win FROM games_clean", conn)
        try:
            already = pd.read_sql("SELECT DISTINCT game_id FROM graded_predictions_v5", conn)["game_id"]
        except Exception:
            already = pd.Series([], dtype="int64")

    daily["game_date"] = to_datetime_flex(daily["game_date"]).dt.date.astype(str)
    preds = daily[(daily["game_date"] == game_date) & (~daily["game_id"].isin(set(already)))].copy()
    if preds.empty:
        print(f"No ungraded predictions for {game_date}.")
        return preds

    graded = preds.merge(games, on="game_id", how="left")
    final = graded["home_win"].notna()
    graded["actual_winner"] = np.where(graded["home_win"] == 1, graded["home_team_abbr"], graded["away_team_abbr"])
    graded.loc[~final, "actual_winner"] = None
    graded["pick_correct"] = np.where(final, graded["recommended_side"] == graded["actual_winner"], np.nan)

    dec = graded["rec_best_moneyline"].apply(american_to_decimal)
    graded["flat_profit_units"] = np.where(
        final, np.where(graded["pick_correct"] == 1, dec - 1, -1.0), np.nan)
    graded["quarter_kelly_profit_pct"] = graded["flat_profit_units"] * graded["rec_quarter_kelly_pct"]

    p = graded["model_home_win_prob"].clip(1e-6, 1 - 1e-6)
    graded["game_log_loss"] = np.where(
        final, -(graded["home_win"] * np.log(p) + (1 - graded["home_win"]) * np.log(1 - p)), np.nan)
    graded["final_score"] = np.where(
        final,
        graded["away_team_abbr"] + " " + graded["away_score"].astype("Int64").astype(str)
        + " - " + graded["home_team_abbr"] + " " + graded["home_score"].astype("Int64").astype(str),
        "not final")

    with get_conn() as conn:
        graded.to_sql("graded_predictions_v5", conn, if_exists="append", index=False)

    done = graded[final]
    if len(done):
        acted = done[done["signal"] != "Pass"]
        print(f"Graded {len(done)} games for {game_date}: "
              f"picks {int(done['pick_correct'].sum())}/{len(done)} correct, "
              f"log loss {done['game_log_loss'].mean():.4f}")
        if len(acted):
            print(f"Actionable ({len(acted)}): flat ROI {acted['flat_profit_units'].mean():+.1%}/bet, "
                  f"1/4-Kelly bankroll change {acted['quarter_kelly_profit_pct'].sum():+.2%}")
    return graded


def print_running_summary() -> None:
    with get_conn() as conn:
        try:
            hist = pd.read_sql("SELECT * FROM graded_predictions_v5 WHERE home_win IS NOT NULL", conn)
        except Exception:
            return
    if hist.empty:
        return
    acted = hist[hist["signal"] != "Pass"]
    print(f"\nRunning history: {len(hist)} graded games, "
          f"model log loss {hist['game_log_loss'].mean():.4f}, "
          f"pick accuracy {hist['pick_correct'].mean():.1%}")
    if len(acted):
        print(f"Actionable bets: {len(acted)}, flat ROI {acted['flat_profit_units'].mean():+.1%}/bet, "
              f"total flat P/L {acted['flat_profit_units'].sum():+.1f}u")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD. Defaults to yesterday.")
    args = parser.parse_args()
    df = grade_predictions(args.date)
    if not df.empty:
        cols = ["matchup", "recommended_side", "signal", "actual_winner", "pick_correct", "final_score"]
        print(df[cols].to_string(index=False))
    print_running_summary()
