"""Negative-binomial / Skellam-style scoring layer.

Converts per-team expected runs (mu_home, mu_away) from the gradient-boosted
run regressors into a joint run distribution, then derives:

- home win probability (with tie mass reallocated for extra innings)
- expected total runs
- run-line cover probabilities (home -1.5 / away +1.5)

Runs are modeled as independent negative-binomial variables. NegBin allows
over-dispersion relative to Poisson (MLB run distributions are over-dispersed),
with the dispersion parameter alpha estimated from training residuals via
method of moments:  Var(Y) = mu + mu^2 / alpha.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

MAX_RUNS = 30  # support grid 0..MAX_RUNS covers >99.999% of MLB scores
POISSON_ALPHA = 1e9  # alpha above this -> treat as Poisson


def estimate_dispersion(y_true: np.ndarray, mu_pred: np.ndarray) -> float:
    """Method-of-moments NegBin dispersion from regression residuals.

    excess variance = E[(y - mu)^2 - mu] = E[mu^2] / alpha
    Returns alpha clipped to a sane range; large alpha ~= Poisson.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(mu_pred, dtype=float), 0.05, None)
    excess = np.mean((y - mu) ** 2 - mu)
    if excess <= 1e-9:
        return POISSON_ALPHA
    alpha = float(np.mean(mu**2) / excess)
    return float(np.clip(alpha, 1.5, POISSON_ALPHA))


def run_pmf(mu: float, alpha: float, max_runs: int = MAX_RUNS) -> np.ndarray:
    """PMF over 0..max_runs runs for one team."""
    mu = max(float(mu), 0.05)
    ks = np.arange(max_runs + 1)
    if alpha >= POISSON_ALPHA:
        pmf = poisson.pmf(ks, mu)
    else:
        # scipy nbinom: n = alpha, p = alpha / (alpha + mu)
        p = alpha / (alpha + mu)
        pmf = nbinom.pmf(ks, alpha, p)
    total = pmf.sum()
    if total <= 0:
        pmf = poisson.pmf(ks, mu)
        total = pmf.sum()
    return pmf / total


def game_probabilities(mu_home: float, mu_away: float, alpha_home: float, alpha_away: float) -> dict:
    """Joint-distribution game markets from two independent NegBin PMFs."""
    ph = run_pmf(mu_home, alpha_home)
    pa = run_pmf(mu_away, alpha_away)
    joint = np.outer(ph, pa)  # joint[h, a]

    idx_h = np.arange(joint.shape[0])[:, None]
    idx_a = np.arange(joint.shape[1])[None, :]
    p_gt = float(joint[idx_h > idx_a].sum())   # home leads after 9
    p_lt = float(joint[idx_h < idx_a].sum())   # away leads after 9
    p_tie = max(0.0, 1.0 - p_gt - p_lt)

    # MLB games can't tie: reallocate the regulation-tie mass proportionally
    # to relative strength (extra innings roughly preserve the edge).
    denom = p_gt + p_lt
    share = p_gt / denom if denom > 0 else 0.5
    p_home_win = p_gt + p_tie * share

    p_home_minus_1_5 = float(joint[(idx_h - idx_a) >= 2].sum())

    return {
        "skellam_home_win_prob": float(np.clip(p_home_win, 1e-4, 1 - 1e-4)),
        "model_total_runs": float(mu_home + mu_away),
        "home_minus_1_5_prob": float(np.clip(p_home_minus_1_5, 1e-4, 1 - 1e-4)),
        "away_plus_1_5_prob": float(np.clip(1.0 - p_home_minus_1_5, 1e-4, 1 - 1e-4)),
    }


def batch_game_probabilities(mu_home: np.ndarray, mu_away: np.ndarray, alpha_home: float, alpha_away: float) -> dict:
    """Vectorized wrapper: returns dict of arrays aligned with the inputs."""
    keys = ["skellam_home_win_prob", "model_total_runs", "home_minus_1_5_prob", "away_plus_1_5_prob"]
    out = {k: [] for k in keys}
    for mh, ma in zip(np.asarray(mu_home, float), np.asarray(mu_away, float)):
        res = game_probabilities(mh, ma, alpha_home, alpha_away)
        for k in keys:
            out[k].append(res[k])
    return {k: np.array(v) for k, v in out.items()}
