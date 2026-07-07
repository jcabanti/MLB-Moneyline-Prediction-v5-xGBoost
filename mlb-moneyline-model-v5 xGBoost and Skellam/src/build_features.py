from __future__ import annotations

import numpy as np
import pandas as pd

STARTER_DEFAULTS = {
    "starter_prior_era": 4.25,
    "starter_prior_whip": 1.30,
    "starter_prior_k_per_9": 8.4,
    "starter_prior_bb_per_9": 3.0,
    "starter_prior_hr_per_9": 1.1,
    "starter_prior_pitches_per_start": 85.0,
}

STARTER_CAPS = {
    "starter_prior_era": (1.50, 9.00),
    "starter_prior_whip": (0.70, 2.20),
    "starter_prior_k_per_9": (3.0, 14.0),
    "starter_prior_bb_per_9": (0.5, 7.0),
    "starter_prior_hr_per_9": (0.0, 3.0),
}

BULLPEN_DEFAULTS = {
    "bullpen_prior_era": 4.20,
    "bullpen_prior_whip": 1.30,
    "bullpen_prior_k_per_9": 8.8,
    "bullpen_prior_bb_per_9": 3.5,
    "bullpen_ip_last_3_team_games": 0.0,
    "bullpen_ip_last_5_team_games": 0.0,
    "bullpen_pitches_last_3_team_games": 0.0,
}

BULLPEN_CAPS = {
    "bullpen_prior_era": (2.00, 7.50),
    "bullpen_prior_whip": (0.90, 1.80),
    "bullpen_prior_k_per_9": (5.0, 13.0),
    "bullpen_prior_bb_per_9": (1.5, 6.0),
    "bullpen_ip_last_3_team_games": (0.0, 18.0),
    "bullpen_ip_last_5_team_games": (0.0, 28.0),
    "bullpen_pitches_last_3_team_games": (0.0, 300.0),
}

FEATURES_V3 = [
    "prior_win_pct_diff",
    "prior_runs_scored_pg_diff",
    "prior_runs_allowed_pg_diff",
    "prior_run_diff_pg_diff",
    "starter_era_diff",
    "starter_whip_diff",
    "starter_k_per_9_diff",
    "starter_bb_per_9_diff",
    "starter_hr_per_9_diff",
    "starter_prior_ip_diff",
    "starter_prior_games_started_diff",
    "bullpen_era_diff",
    "bullpen_whip_diff",
    "bullpen_k_per_9_diff",
    "bullpen_bb_per_9_diff",
    "bullpen_ip_last_3_diff",
    "bullpen_ip_last_5_diff",
    "bullpen_pitches_last_3_diff",
]


def add_team_features_for_date(today_games: pd.DataFrame, games_clean: pd.DataFrame) -> pd.DataFrame:
    target_game_date = pd.to_datetime(today_games["game_date"]).min()
    target_season = int(target_game_date.year)
    completed = games_clean[(games_clean["season"] == target_season) & (pd.to_datetime(games_clean["game_date"]) < target_game_date)].copy()
    home = completed[["game_id", "home_team_abbr", "away_team_abbr", "home_score", "away_score", "home_win"]].rename(
        columns={"home_team_abbr": "team", "away_team_abbr": "opponent", "home_score": "runs_scored", "away_score": "runs_allowed"}
    )
    home["team_win"] = home["home_win"]
    away = completed[["game_id", "away_team_abbr", "home_team_abbr", "away_score", "home_score", "home_win"]].rename(
        columns={"away_team_abbr": "team", "home_team_abbr": "opponent", "away_score": "runs_scored", "home_score": "runs_allowed"}
    )
    away["team_win"] = 1 - away["home_win"]
    tg = pd.concat([home, away], ignore_index=True)
    strength = tg.groupby("team").agg(
        games_played_prior=("game_id", "count"), wins_prior=("team_win", "sum"),
        runs_scored_prior=("runs_scored", "sum"), runs_allowed_prior=("runs_allowed", "sum")
    ).reset_index()
    strength["prior_win_pct"] = strength["wins_prior"] / strength["games_played_prior"]
    strength["prior_runs_scored_per_game"] = strength["runs_scored_prior"] / strength["games_played_prior"]
    strength["prior_runs_allowed_per_game"] = strength["runs_allowed_prior"] / strength["games_played_prior"]
    strength["prior_run_diff_per_game"] = strength["prior_runs_scored_per_game"] - strength["prior_runs_allowed_per_game"]
    home_f = strength.rename(columns={"team": "home_team_abbr", "games_played_prior": "home_games_played_prior", "prior_win_pct": "home_prior_win_pct", "prior_runs_scored_per_game": "home_prior_runs_scored_per_game", "prior_runs_allowed_per_game": "home_prior_runs_allowed_per_game", "prior_run_diff_per_game": "home_prior_run_diff_per_game"})
    away_f = strength.rename(columns={"team": "away_team_abbr", "games_played_prior": "away_games_played_prior", "prior_win_pct": "away_prior_win_pct", "prior_runs_scored_per_game": "away_prior_runs_scored_per_game", "prior_runs_allowed_per_game": "away_prior_runs_allowed_per_game", "prior_run_diff_per_game": "away_prior_run_diff_per_game"})
    out = today_games.merge(home_f, on="home_team_abbr", how="left").merge(away_f, on="away_team_abbr", how="left")
    out["prior_win_pct_diff"] = out["home_prior_win_pct"] - out["away_prior_win_pct"]
    out["prior_runs_scored_pg_diff"] = out["home_prior_runs_scored_per_game"] - out["away_prior_runs_scored_per_game"]
    out["prior_runs_allowed_pg_diff"] = out["home_prior_runs_allowed_per_game"] - out["away_prior_runs_allowed_per_game"]
    out["prior_run_diff_pg_diff"] = out["home_prior_run_diff_per_game"] - out["away_prior_run_diff_per_game"]
    return out


def add_starter_features_for_date(today_features: pd.DataFrame, games_clean: pd.DataFrame, starter_lines: pd.DataFrame) -> pd.DataFrame:
    target_game_date = pd.to_datetime(today_features["game_date"]).min()
    target_season = int(target_game_date.year)
    sl = starter_lines.merge(games_clean[["game_id", "game_date", "season"]], on="game_id", how="left")
    sl["game_date"] = pd.to_datetime(sl["game_date"])
    before = sl[(sl["season"] == target_season) & (sl["game_date"] < target_game_date)].copy()
    prior = before.groupby("pitcher_id").agg(
        starter_prior_games_started=("game_id", "count"), starter_prior_ip=("innings_pitched", "sum"), starter_prior_er=("earned_runs", "sum"),
        starter_prior_so=("strikeouts", "sum"), starter_prior_bb=("walks", "sum"), starter_prior_hits=("hits_allowed", "sum"),
        starter_prior_hr=("home_runs_allowed", "sum"), starter_prior_pitches=("pitches_thrown", "sum")
    ).reset_index()
    prior["starter_prior_era"] = np.where(prior["starter_prior_ip"] > 0, 9 * prior["starter_prior_er"] / prior["starter_prior_ip"], 4.25)
    prior["starter_prior_whip"] = np.where(prior["starter_prior_ip"] > 0, (prior["starter_prior_hits"] + prior["starter_prior_bb"]) / prior["starter_prior_ip"], 1.30)
    prior["starter_prior_k_per_9"] = np.where(prior["starter_prior_ip"] > 0, 9 * prior["starter_prior_so"] / prior["starter_prior_ip"], 8.4)
    prior["starter_prior_bb_per_9"] = np.where(prior["starter_prior_ip"] > 0, 9 * prior["starter_prior_bb"] / prior["starter_prior_ip"], 3.0)
    prior["starter_prior_hr_per_9"] = np.where(prior["starter_prior_ip"] > 0, 9 * prior["starter_prior_hr"] / prior["starter_prior_ip"], 1.1)
    prior["starter_prior_pitches_per_start"] = np.where(prior["starter_prior_games_started"] > 0, prior["starter_prior_pitches"] / prior["starter_prior_games_started"], 85.0)
    for c, (lo, hi) in STARTER_CAPS.items():
        prior[c] = prior[c].clip(lo, hi)
    home = prior.rename(columns={"pitcher_id": "home_probable_pitcher_id", **{c: "home_" + c for c in prior.columns if c.startswith("starter_")}})
    away = prior.rename(columns={"pitcher_id": "away_probable_pitcher_id", **{c: "away_" + c for c in prior.columns if c.startswith("starter_")}})
    out = today_features.merge(home, on="home_probable_pitcher_id", how="left").merge(away, on="away_probable_pitcher_id", how="left")
    for side in ["home", "away"]:
        out[f"{side}_starter_prior_games_started"] = out[f"{side}_starter_prior_games_started"].fillna(0)
        out[f"{side}_starter_prior_ip"] = out[f"{side}_starter_prior_ip"].fillna(0)
        for c, v in STARTER_DEFAULTS.items():
            out[f"{side}_{c}"] = out[f"{side}_{c}"].fillna(v)
    out["starter_era_diff"] = out["home_starter_prior_era"] - out["away_starter_prior_era"]
    out["starter_whip_diff"] = out["home_starter_prior_whip"] - out["away_starter_prior_whip"]
    out["starter_k_per_9_diff"] = out["home_starter_prior_k_per_9"] - out["away_starter_prior_k_per_9"]
    out["starter_bb_per_9_diff"] = out["home_starter_prior_bb_per_9"] - out["away_starter_prior_bb_per_9"]
    out["starter_hr_per_9_diff"] = out["home_starter_prior_hr_per_9"] - out["away_starter_prior_hr_per_9"]
    out["starter_prior_ip_diff"] = out["home_starter_prior_ip"] - out["away_starter_prior_ip"]
    out["starter_prior_games_started_diff"] = out["home_starter_prior_games_started"] - out["away_starter_prior_games_started"]
    return out


def add_bullpen_features_for_date(today_features: pd.DataFrame, bullpen_team_game: pd.DataFrame) -> pd.DataFrame:
    target_game_date = pd.to_datetime(today_features["game_date"]).min()
    target_season = int(target_game_date.year)
    bp = bullpen_team_game.copy()
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    before = bp[(bp["season"] == target_season) & (bp["game_date"] < target_game_date)].copy()
    prior = before.groupby("team_abbr").agg(
        bullpen_games_prior=("game_id", "count"), bullpen_prior_ip=("bullpen_ip", "sum"), bullpen_prior_er=("bullpen_er", "sum"), bullpen_prior_so=("bullpen_so", "sum"),
        bullpen_prior_bb=("bullpen_bb", "sum"), bullpen_prior_hits=("bullpen_hits", "sum"), bullpen_prior_hr=("bullpen_hr", "sum")
    ).reset_index()
    prior["bullpen_prior_era"] = np.where(prior["bullpen_prior_ip"] > 0, 9 * prior["bullpen_prior_er"] / prior["bullpen_prior_ip"], 4.20)
    prior["bullpen_prior_whip"] = np.where(prior["bullpen_prior_ip"] > 0, (prior["bullpen_prior_hits"] + prior["bullpen_prior_bb"]) / prior["bullpen_prior_ip"], 1.30)
    prior["bullpen_prior_k_per_9"] = np.where(prior["bullpen_prior_ip"] > 0, 9 * prior["bullpen_prior_so"] / prior["bullpen_prior_ip"], 8.8)
    prior["bullpen_prior_bb_per_9"] = np.where(prior["bullpen_prior_ip"] > 0, 9 * prior["bullpen_prior_bb"] / prior["bullpen_prior_ip"], 3.5)
    recent = before.sort_values(["team_abbr", "game_date", "game_id"]).copy()
    recent["rank_desc"] = recent.groupby("team_abbr").cumcount(ascending=False)
    last3 = recent[recent["rank_desc"] < 3].groupby("team_abbr").agg(bullpen_ip_last_3_team_games=("bullpen_ip", "sum"), bullpen_pitches_last_3_team_games=("bullpen_pitches", "sum")).reset_index()
    last5 = recent[recent["rank_desc"] < 5].groupby("team_abbr").agg(bullpen_ip_last_5_team_games=("bullpen_ip", "sum")).reset_index()
    prior = prior.merge(last3, on="team_abbr", how="left").merge(last5, on="team_abbr", how="left")
    for c, default in BULLPEN_DEFAULTS.items():
        prior[c] = prior[c].fillna(default)
    for c, (lo, hi) in BULLPEN_CAPS.items():
        prior[c] = prior[c].clip(lo, hi)
    home = prior.rename(columns={"team_abbr": "home_team_abbr", **{c: "home_" + c for c in prior.columns if c.startswith("bullpen_")}})
    away = prior.rename(columns={"team_abbr": "away_team_abbr", **{c: "away_" + c for c in prior.columns if c.startswith("bullpen_")}})
    out = today_features.merge(home, on="home_team_abbr", how="left").merge(away, on="away_team_abbr", how="left")
    for side in ["home", "away"]:
        for c, default in BULLPEN_DEFAULTS.items():
            out[f"{side}_{c}"] = out[f"{side}_{c}"].fillna(default)
    out["bullpen_era_diff"] = out["home_bullpen_prior_era"] - out["away_bullpen_prior_era"]
    out["bullpen_whip_diff"] = out["home_bullpen_prior_whip"] - out["away_bullpen_prior_whip"]
    out["bullpen_k_per_9_diff"] = out["home_bullpen_prior_k_per_9"] - out["away_bullpen_prior_k_per_9"]
    out["bullpen_bb_per_9_diff"] = out["home_bullpen_prior_bb_per_9"] - out["away_bullpen_prior_bb_per_9"]
    out["bullpen_ip_last_3_diff"] = out["home_bullpen_ip_last_3_team_games"] - out["away_bullpen_ip_last_3_team_games"]
    out["bullpen_ip_last_5_diff"] = out["home_bullpen_ip_last_5_team_games"] - out["away_bullpen_ip_last_5_team_games"]
    out["bullpen_pitches_last_3_diff"] = out["home_bullpen_pitches_last_3_team_games"] - out["away_bullpen_pitches_last_3_team_games"]
    return out
