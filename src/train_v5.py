"""Train the V5 hybrid model.

Architecture:
1. Gradient-boosted Poisson regressor predicting runs scored per team-game
   (XGBoost `count:poisson` when available; sklearn HistGradientBoosting
   with Poisson loss as an automatic fallback).
2. NegBin dispersion estimated from training residuals; a Skellam-style
   joint-distribution layer converts (mu_home, mu_away) into win / total /
   run-line probabilities.
3. Gradient-boosted classifier predicting home_win directly from diff features.
4. Blend weight between the Skellam prob and classifier prob chosen on the
   calibration season by log loss, then Platt (sigmoid) calibration.

Splits: train 2021-2024, calibrate 2025, test 2026-to-date.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .build_features_v5 import CLASSIFIER_FEATURES_V5, RUN_MODEL_FEATURES, build_game_features, to_run_model_rows
from .config import MODELS_DIR
from .db import get_conn
from .skellam import batch_game_probabilities, estimate_dispersion
from .utils import to_datetime_flex

MODEL_PATH = MODELS_DIR / "hybrid_v5.joblib"
TRAIN_SEASONS = (2021, 2024)
CAL_SEASON = 2025
TEST_SEASON_MIN = 2026
MIN_PRIOR_GAMES = 10

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover - CI installs xgboost; local fallback
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    HAS_XGB = False


def make_run_regressor():
    if HAS_XGB:
        return XGBRegressor(
            objective="count:poisson", n_estimators=500, learning_rate=0.03,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20, reg_lambda=2.0, n_jobs=4, random_state=42)
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.05, max_depth=4,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42)


def make_classifier():
    if HAS_XGB:
        return XGBClassifier(
            objective="binary:logistic", n_estimators=400, learning_rate=0.03,
            max_depth=3, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=30, reg_lambda=2.0, n_jobs=4,
            eval_metric="logloss", random_state=42)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3,
        min_samples_leaf=60, l2_regularization=1.0, random_state=42)


def load_training_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with get_conn() as conn:
        games = pd.read_sql("SELECT * FROM games_clean", conn)
        starters = pd.read_sql("SELECT * FROM starter_game_pitching_lines", conn)
        bullpen = pd.read_sql("SELECT * FROM bullpen_team_game", conn)
    games["game_date"] = to_datetime_flex(games["game_date"])

    # attach starter ids per side from the pitching-line table
    sp = starters.pivot_table(index="game_id", columns="side", values="pitcher_id", aggfunc="first").reset_index()
    sp.columns = ["game_id"] + [f"{c}_starter_id" for c in sp.columns[1:]]
    games = games.merge(sp, on="game_id", how="left")
    return games, starters, bullpen


def predict_hybrid(bundle: dict, game_features: pd.DataFrame) -> pd.DataFrame:
    """Score a game-feature frame with a trained bundle. Returns per-game probs."""
    run_rows = to_run_model_rows(game_features, include_target=False)
    mu = np.clip(bundle["run_model"].predict(run_rows[RUN_MODEL_FEATURES]), 0.3, 12.0)
    run_rows = run_rows.assign(mu=mu)
    mu_h = run_rows[run_rows["side"] == "home"].set_index("game_id")["mu"]
    mu_a = run_rows[run_rows["side"] == "away"].set_index("game_id")["mu"]
    mu_h = mu_h.loc[game_features["game_id"]].values
    mu_a = mu_a.loc[game_features["game_id"]].values

    sk = batch_game_probabilities(mu_h, mu_a, bundle["alpha_home"], bundle["alpha_away"])
    p_clf = bundle["classifier"].predict_proba(game_features[CLASSIFIER_FEATURES_V5])[:, 1]
    w = bundle["blend_weight"]
    p_blend = w * sk["skellam_home_win_prob"] + (1 - w) * p_clf
    p_cal = bundle["calibrator"].predict_proba(p_blend.reshape(-1, 1))[:, 1]

    return pd.DataFrame({
        "game_id": game_features["game_id"].values,
        "model_home_win_prob": np.clip(p_cal, 1e-4, 1 - 1e-4),
        "skellam_home_win_prob": sk["skellam_home_win_prob"],
        "classifier_home_win_prob": p_clf,
        "mu_home_runs": mu_h,
        "mu_away_runs": mu_a,
        "model_total_runs": sk["model_total_runs"],
        "home_minus_1_5_prob": sk["home_minus_1_5_prob"],
        "away_plus_1_5_prob": sk["away_plus_1_5_prob"],
    })


def train_v5() -> dict:
    games, starters, bullpen = load_training_frame()
    feats = build_game_features(games, games, starters, bullpen)
    feats = feats.dropna(subset=["home_win"]).copy()
    feats = feats[(feats["home_off_games_played"] >= MIN_PRIOR_GAMES)
                  & (feats["away_off_games_played"] >= MIN_PRIOR_GAMES)]
    feats["home_win"] = feats["home_win"].astype(int)

    train = feats[(feats["season"] >= TRAIN_SEASONS[0]) & (feats["season"] <= TRAIN_SEASONS[1])]
    cal = feats[feats["season"] == CAL_SEASON]
    test = feats[feats["season"] >= TEST_SEASON_MIN]
    print(f"Rows -> train {len(train)}, cal {len(cal)}, test {len(test)}  (xgboost={HAS_XGB})")

    # 1) run regressor on offense-perspective rows
    run_train = to_run_model_rows(train).dropna(subset=["runs"])
    run_model = make_run_regressor()
    run_model.fit(run_train[RUN_MODEL_FEATURES], run_train["runs"])

    # 2) NegBin dispersion by side from training residuals
    mu_train = np.clip(run_model.predict(run_train[RUN_MODEL_FEATURES]), 0.3, 12.0)
    hm = run_train["side"] == "home"
    alpha_home = estimate_dispersion(run_train.loc[hm, "runs"].values, mu_train[hm.values])
    alpha_away = estimate_dispersion(run_train.loc[~hm, "runs"].values, mu_train[~hm.values])
    print(f"NegBin dispersion -> alpha_home {alpha_home:.1f}, alpha_away {alpha_away:.1f}")

    # 3) classifier
    clf = make_classifier()
    clf.fit(train[CLASSIFIER_FEATURES_V5], train["home_win"])

    # partial bundle for scoring the calibration set
    tmp = {"run_model": run_model, "classifier": clf, "alpha_home": alpha_home,
           "alpha_away": alpha_away, "blend_weight": 1.0,
           "calibrator": _identity_calibrator()}

    cal_scores = predict_hybrid(tmp, cal)
    y_cal = cal["home_win"].values

    # 4) blend weight on calibration season
    best_w, best_ll = 0.5, np.inf
    for w in np.linspace(0, 1, 21):
        p = w * cal_scores["skellam_home_win_prob"] + (1 - w) * cal_scores["classifier_home_win_prob"]
        ll = log_loss(y_cal, np.clip(p, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_ll, best_w = ll, float(w)
    print(f"Blend weight (skellam share) = {best_w:.2f}  cal log loss {best_ll:.4f}")

    # 5) Platt calibration of the blended probability
    p_blend_cal = (best_w * cal_scores["skellam_home_win_prob"]
                   + (1 - best_w) * cal_scores["classifier_home_win_prob"]).values.reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000)
    calibrator.fit(p_blend_cal, y_cal)

    bundle = {
        "run_model": run_model, "classifier": clf,
        "alpha_home": alpha_home, "alpha_away": alpha_away,
        "blend_weight": best_w, "calibrator": calibrator,
        "classifier_features": CLASSIFIER_FEATURES_V5,
        "run_model_features": RUN_MODEL_FEATURES,
        "min_prior_games": MIN_PRIOR_GAMES, "uses_xgboost": HAS_XGB,
        "train_seasons": list(TRAIN_SEASONS), "cal_season": CAL_SEASON,
    }

    # 6) held-out test metrics (current season)
    metrics = {}
    if len(test):
        scores = predict_hybrid(bundle, test)
        y = test["home_win"].values
        p = scores["model_home_win_prob"].values
        metrics = {
            "test_log_loss": float(log_loss(y, p)),
            "test_brier_score": float(brier_score_loss(y, p)),
            "test_auc": float(roc_auc_score(y, p)),
            "test_accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "test_home_win_rate": float(np.mean(y)),
            "avg_predicted_home_win_prob": float(np.mean(p)),
            "n_test_games": int(len(test)),
        }
        test_out = test[["game_id", "game_date", "season", "matchup", "away_team_abbr", "home_team_abbr", "home_win"]].copy()
        test_out = test_out.merge(scores, on="game_id")
        with get_conn() as conn:
            test_out.to_sql("hybrid_v5_test_predictions", conn, if_exists="replace", index=False)
    bundle["metrics"] = metrics

    joblib.dump(bundle, MODEL_PATH)
    print("Saved model to:", MODEL_PATH)
    print(json.dumps(metrics, indent=2))
    return metrics


def _identity_calibrator():
    lr = LogisticRegression()
    lr.classes_ = np.array([0, 1])
    lr.coef_ = np.array([[1.0]])
    lr.intercept_ = np.array([0.0])
    # exact identity on probabilities isn't needed pre-blend; predict_proba of
    # this near-identity is only used transiently before real calibration.
    return lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2021, help="Reserved; splits are fixed in-module.")
    parser.parse_args()
    train_v5()
