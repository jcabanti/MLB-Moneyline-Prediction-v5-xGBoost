"""Offline smoke test: run the full daily feature + prediction path without
network access by simulating a slate from the most recent day in the database.

Usage:  python -m scripts.smoke_test
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.build_features_v5 import build_game_features
from src.db import get_conn
from src.train_v5 import MODEL_PATH, predict_hybrid
from src.utils import to_datetime_flex


def main() -> None:
    with get_conn() as conn:
        games = pd.read_sql("SELECT * FROM games_clean", conn)
        starters = pd.read_sql("SELECT * FROM starter_game_pitching_lines", conn)
        bullpen = pd.read_sql("SELECT * FROM bullpen_team_game", conn)
    games["game_date"] = to_datetime_flex(games["game_date"])

    last_date = games["game_date"].max()
    slate = games[games["game_date"] == last_date].copy()
    print(f"Simulating slate of {len(slate)} games on {last_date.date()}")

    sp = starters.pivot_table(index="game_id", columns="side", values="pitcher_id", aggfunc="first").reset_index()
    sp.columns = ["game_id"] + [f"{c}_probable_pitcher_id" for c in sp.columns[1:]]
    slate = slate.merge(sp, on="game_id", how="left")

    # pretend the slate is unplayed: history excludes that day
    history = games[games["game_date"] < last_date]
    slate_unplayed = slate.drop(columns=["home_score", "away_score", "home_win"])

    feats = build_game_features(slate_unplayed, history, starters, bullpen)
    bundle = joblib.load(MODEL_PATH)
    scores = predict_hybrid(bundle, feats)

    merged = scores.merge(slate[["game_id", "matchup", "home_win"]], on="game_id")
    print(merged[["matchup", "model_home_win_prob", "mu_home_runs", "mu_away_runs",
                  "model_total_runs", "home_minus_1_5_prob", "home_win"]].round(3).to_string(index=False))

    p = merged["model_home_win_prob"]
    assert p.between(0.01, 0.99).all(), "probabilities out of range"
    assert merged["model_total_runs"].between(5, 14).all(), "totals look wrong"
    assert not feats[["pyth_diff", "starter_fip_diff", "bullpen_pitches_last3d_diff"]].isna().any().any()
    acc = (np.round(p) == merged["home_win"]).mean()
    print(f"\nOK — sanity checks passed. Same-day hit rate (1 slate, noisy): {acc:.0%}")


if __name__ == "__main__":
    main()
