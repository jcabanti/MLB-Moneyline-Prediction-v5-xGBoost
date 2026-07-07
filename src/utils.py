from __future__ import annotations

import math
import pandas as pd


def ip_str_to_float(ip_value) -> float:
    if pd.isna(ip_value):
        return 0.0
    ip_str = str(ip_value)
    if "." not in ip_str:
        return float(ip_str)
    whole, outs = ip_str.split(".")
    return int(whole) + int(outs) / 3


def american_to_implied_prob(odds) -> float:
    if pd.isna(odds):
        return math.nan
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds) -> float:
    if pd.isna(odds):
        return math.nan
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def decimal_to_american(dec) -> float | None:
    if pd.isna(dec) or dec <= 1:
        return None
    if dec >= 2:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))


def probability_to_american_odds(p: float):
    if pd.isna(p) or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def parse_pitcher_stat_value(stats: dict, key: str, default=0):
    value = stats.get(key, default)
    if value in [None, "", "-"]:
        return default
    try:
        return float(value)
    except Exception:
        return default
