# MLB Moneyline Model — V5 Hybrid

Gradient-boosted Poisson run regressors feeding a NegBin/Skellam scoring layer,
blended with a gradient-boosted win classifier and Platt-calibrated.

## Architecture

1. **Run model** — XGBoost `count:poisson` regressor predicts expected runs per
   team-game (falls back to sklearn HistGradientBoosting with Poisson loss if
   xgboost is unavailable).
2. **Skellam layer** — negative-binomial run distributions (dispersion fit from
   training residuals) give win probability, expected total, and run-line probs.
3. **Classifier** — GBT on home−away diff features predicts the winner directly.
4. **Blend + calibration** — blend weight grid-searched on the calibration
   season by log loss, then sigmoid-calibrated.

Splits: train 2021–2024 · calibrate 2025 · test 2026-to-date.

## Signals

Team: Pythagorean expectation, season and last-15 form, rest days, games in
last 7 days. Starter: FIP, ERA, WHIP, K/9, BB/9, HR/9, last-10-start FIP,
pitches/start, with empirical-Bayes shrinkage toward league means. Bullpen:
FIP + rate stats, pitches thrown in the last 1 and 3 **calendar** days, IP in
last 3 team games. Context: park factor from prior seasons only, regressed.
All aggregates are strictly before the game date (no same-day leakage).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ODDS_API_KEY
```

GitHub: add `ODDS_API_KEY` under Settings → Secrets → Actions, and enable
**Settings → Actions → General → Workflow permissions → Read and write** so the
workflow can commit the updated database, reports, and model back to the repo.

## Commands

```bash
python -m src.ingest_boxscores      # refresh current-season games + missing pitching lines
python -m src.train_v5              # train hybrid, save models/hybrid_v5.joblib
python -m src.run_daily_report      # odds + slate -> V5 predictions CSV + markdown report
python -m src.grade_predictions     # grade yesterday (accuracy, log loss, flat & 1/4-Kelly ROI)
python -m scripts.smoke_test        # offline sanity check of the full inference path
```

## Automated workflow (`.github/workflows/mlb_reports.yml`)

- **Daily job** (morning + hourly lineup-watch + overnight): ingest new results
  and boxscores → grade yesterday → run the V5 report → commit the updated
  database and reports → upload CSV/markdown artifacts. This keeps features
  fresh as the season progresses.
- **Weekly retrain** (Mondays, or manual dispatch with `retrain=true`):
  retrains on all data and commits the new model.

## Outputs

`reports/daily/daily_predictions_v5_hybrid_<date>.csv` — per game: model win
probs (blend + components), fair moneylines, expected runs and total, run-line
probs, edge vs de-vigged consensus, EV at the best listed book price,
quarter-Kelly stake fraction, signal tier (Strong ≥5% edge / Play ≥3% /
Lean ≥1.5% / Pass), starter and lineup confirmation flags.

`reports/daily/report_v5_<date>.md` — readable summary of actionable edges.

## Tables

`games_clean` (Final games only), `starter_game_pitching_lines`,
`bullpen_team_game`, `daily_predictions_v5_hybrid`, `graded_predictions_v5`
(running history), `hybrid_v5_test_predictions`, plus odds snapshot tables.

## Notes

Test metrics on 2026 games are a modest improvement over the V3 logistic on
log loss and Brier score. Treat output as a model signal, not betting advice —
edges are small, variance is large, and stakes assume the best listed price.
