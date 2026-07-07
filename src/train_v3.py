from __future__ import annotations

import argparse
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .build_features import FEATURES_V3
from .config import MODELS_DIR
from .db import get_conn

TARGET = "home_win"


def train_v3_calibrated() -> dict:
    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM model_games_v3 ORDER BY game_date, game_id", conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    model_data = df.dropna(subset=FEATURES_V3 + [TARGET]).copy()
    model_data = model_data[(model_data["home_games_played_prior"] >= 10) & (model_data["away_games_played_prior"] >= 10)].copy()

    train_df = model_data[model_data["season"] <= 2023].copy()
    cal_df = model_data[model_data["season"] == 2024].copy()
    test_df = model_data[model_data["season"] >= 2025].copy()

    X_train, y_train = train_df[FEATURES_V3], train_df[TARGET].astype(int)
    X_cal, y_cal = cal_df[FEATURES_V3], cal_df[TARGET].astype(int)
    X_test, y_test = test_df[FEATURES_V3], test_df[TARGET].astype(int)

    base_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    base_model.fit(X_train, y_train)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_cal, y_cal)

    pred = calibrated_model.predict_proba(X_test)[:, 1]
    pred_class = (pred >= 0.5).astype(int)
    metrics = {
        "test_log_loss": log_loss(y_test, pred),
        "test_brier_score": brier_score_loss(y_test, pred),
        "test_auc": roc_auc_score(y_test, pred),
        "test_accuracy": accuracy_score(y_test, pred_class),
        "test_home_win_rate": y_test.mean(),
        "avg_predicted_home_win_prob": pred.mean(),
    }
    bundle = {
        "model": calibrated_model,
        "features": FEATURES_V3,
        "target": TARGET,
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "calibration_seasons": sorted(cal_df["season"].unique().tolist()),
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        "metrics": metrics,
        "calibration_method": "sigmoid",
        "min_prior_games": 10,
    }
    path = MODELS_DIR / "baseline_logistic_v3_calibrated.joblib"
    joblib.dump(bundle, path)

    test_predictions = test_df[["game_id", "game_date", "season", "matchup", "away_team_abbr", "home_team_abbr", "home_win", "away_starter_name", "home_starter_name"]].copy()
    test_predictions["model_home_win_prob"] = pred
    test_predictions["model_away_win_prob"] = 1 - pred
    with get_conn() as conn:
        test_predictions.to_sql("baseline_v3_calibrated_test_predictions", conn, if_exists="replace", index=False)
    print("Saved model to:", path)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2021, help="Reserved for future full rebuilds.")
    parser.parse_args()
    train_v3_calibrated()
