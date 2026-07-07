"""V5 feature engineering — shared by historical training and the daily slate.

Signals added over V3:
- Team: Pythagorean expectation, last-15-game form (RS/G, RA/G, win pct),
  rest days, schedule density (games in last 7 calendar days)
- Starter: FIP, last-10-start rolling FIP/K9/BB9, pitches per start,
  empirical-Bayes shrinkage toward league means (replaces hard caps)
- Bullpen: FIP, true calendar-day fatigue (pitches thrown in the last 1 and
  3 calendar days), plus the existing last-3-team-games workload
- Context: park factor (venue run environment from prior seasons only,
  regressed toward 1.0)

All aggregates are strictly *before the game date* (no same-day leakage).
The daily path reuses the exact same machinery by appending today's slate
as unplayed rows and slicing them back out after the as-of computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import to_datetime_flex

# ---------------------------------------------------------------------------
# League priors for empirical-Bayes shrinkage (fixed constants -> leak-free)
# ---------------------------------------------------------------------------
LEAGUE = {
    "era": 4.30, "whip": 1.30, "k9": 8.6, "bb9": 3.2, "hr9": 1.15, "fip": 4.20,
    "bp_era": 4.20, "bp_whip": 1.32, "bp_k9": 8.9, "bp_bb9": 3.5, "bp_fip": 4.15,
    "rs_pg": 4.50, "pitches_per_start": 88.0,
}
STARTER_SHRINK_IP = 40.0     # IP of league-average performance blended in
STARTER_L10_SHRINK_IP = 25.0
BULLPEN_SHRINK_IP = 60.0
TEAM_SHRINK_GAMES = 25.0
FIP_CONSTANT = 3.15
PARK_SHRINK_GAMES = 100.0

CLASSIFIER_FEATURES_V5 = [
    "pyth_diff", "win_pct_diff", "rs_pg_diff", "ra_pg_diff",
    "form15_rs_diff", "form15_ra_diff", "form15_wpct_diff",
    "starter_fip_diff", "starter_era_diff", "starter_whip_diff",
    "starter_k9_diff", "starter_bb9_diff", "starter_hr9_diff",
    "starter_l10_fip_diff", "starter_pps_diff", "starter_ip_diff",
    "bullpen_fip_diff", "bullpen_k9_diff", "bullpen_bb9_diff",
    "bullpen_pitches_last3d_diff", "bullpen_pitches_last1d_diff",
    "bullpen_ip_last3g_diff",
    "rest_days_diff", "games_last7_diff", "park_factor",
]

RUN_MODEL_FEATURES = [
    "is_home",
    "off_rs_pg", "off_form15_rs_pg", "off_games_played",
    "opp_starter_fip", "opp_starter_era", "opp_starter_whip",
    "opp_starter_k9", "opp_starter_bb9", "opp_starter_hr9",
    "opp_starter_l10_fip", "opp_starter_pps", "opp_starter_ip",
    "opp_bullpen_fip", "opp_bullpen_k9", "opp_bullpen_bb9",
    "opp_bullpen_pitches_last3d", "opp_bullpen_pitches_last1d",
    "opp_bullpen_ip_last3g",
    "park_factor", "off_rest_days", "off_games_last7",
]


def _first_of_date(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """Make all same-date rows within a group use start-of-date values,
    so doubleheader game 2 never sees game 1 of the same day."""
    df[cols] = df.groupby([group_col, "game_date"])[cols].transform("first")
    return df


def _shifted_cum(df: pd.DataFrame, group_col: str, value_cols: list[str], prefix: str) -> pd.DataFrame:
    """Strictly-prior expanding sums per group (sorted by date, then game_id)."""
    df = df.sort_values([group_col, "game_date", "game_id"]).reset_index(drop=True)
    out_cols = []
    g = df.groupby(group_col, sort=False)
    for c in value_cols:
        name = f"{prefix}{c}_cum"
        df[name] = g[c].transform(lambda s: s.cumsum().shift(1))
        out_cols.append(name)
    name = f"{prefix}games_cum"
    df[name] = g.cumcount()
    out_cols.append(name)
    return _first_of_date(df, group_col, out_cols)


def _shifted_roll(df: pd.DataFrame, group_col: str, value_cols: list[str], window: int, prefix: str) -> pd.DataFrame:
    """Strictly-prior rolling sums over the last `window` games per group."""
    df = df.sort_values([group_col, "game_date", "game_id"]).reset_index(drop=True)
    out_cols = []
    g = df.groupby(group_col, sort=False)
    for c in value_cols:
        name = f"{prefix}{c}_r{window}"
        df[name] = g[c].transform(lambda s: s.rolling(window, min_periods=1).sum().shift(1))
        out_cols.append(name)
    name = f"{prefix}n_r{window}"
    df[name] = g[value_cols[0]].transform(lambda s: s.rolling(window, min_periods=1).count().shift(1))
    out_cols.append(name)
    return _first_of_date(df, group_col, out_cols)


def _calendar_window_sum(daily: pd.DataFrame, queries: pd.DataFrame, key: str, value_col: str, days: int, out_col: str) -> pd.DataFrame:
    """Sum of `value_col` over the `days` calendar days strictly before each
    query date. `daily` has one row per (key, date); `queries` has (key, game_date)."""
    daily = daily.sort_values([key, "game_date"]).copy()
    daily["_cum"] = daily.groupby(key)[value_col].cumsum()
    cum = daily[[key, "game_date", "_cum"]]

    q = queries[[key, "game_date"]].copy()
    q["_row"] = np.arange(len(q))
    q["_hi"] = q["game_date"] - pd.Timedelta(days=1)
    q["_lo"] = q["game_date"] - pd.Timedelta(days=days + 1)

    def asof(bound_col):
        left = q.sort_values(bound_col)
        merged = pd.merge_asof(left, cum.rename(columns={"game_date": bound_col}).sort_values(bound_col),
                               on=bound_col, by=key, direction="backward")
        return merged.set_index("_row")["_cum"]

    hi = asof("_hi").fillna(0.0)
    lo = asof("_lo").fillna(0.0)
    result = (hi - lo).sort_index()
    out = queries.copy()
    out[out_col] = result.values
    return out


# ---------------------------------------------------------------------------
# Team offense / record features
# ---------------------------------------------------------------------------

def _team_long(games: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["game_id", "game_date", "season"]
    home = games[base_cols + ["home_team_abbr", "home_score", "away_score", "home_win"]].rename(
        columns={"home_team_abbr": "team", "home_score": "rs", "away_score": "ra"})
    home["win"] = home["home_win"]
    away = games[base_cols + ["away_team_abbr", "away_score", "home_score", "home_win"]].rename(
        columns={"away_team_abbr": "team", "away_score": "rs", "home_score": "ra"})
    away["win"] = 1 - away["home_win"]
    return pd.concat([home, away], ignore_index=True)


def team_features_asof(games_completed: pd.DataFrame, query_games: pd.DataFrame) -> pd.DataFrame:
    """Per-(team, game) as-of features. `query_games` may include unplayed rows."""
    hist = _team_long(games_completed)
    todays = _team_long(query_games.assign(home_score=np.nan, away_score=np.nan, home_win=np.nan)) \
        if "home_score" not in query_games or query_games["home_score"].isna().all() else _team_long(query_games)
    todays = todays[~todays["game_id"].isin(hist["game_id"])]
    tg = pd.concat([hist, todays], ignore_index=True)
    tg["game_date"] = to_datetime_flex(tg["game_date"])

    # season-scoped expanding stats
    tg["skey"] = tg["team"] + "_" + tg["season"].astype(str)
    tg = _shifted_cum(tg, "skey", ["rs", "ra", "win"], "s_")
    tg = _shifted_roll(tg, "skey", ["rs", "ra", "win"], 15, "f_")

    n = tg["s_games_cum"].clip(lower=0)
    w = TEAM_SHRINK_GAMES
    tg["off_rs_pg"] = (tg["s_rs_cum"].fillna(0) + LEAGUE["rs_pg"] * w) / (n + w)
    tg["def_ra_pg"] = (tg["s_ra_cum"].fillna(0) + LEAGUE["rs_pg"] * w) / (n + w)
    tg["win_pct"] = (tg["s_win_cum"].fillna(0) + 0.5 * w) / (n + w)
    rs2, ra2 = tg["off_rs_pg"] ** 2, tg["def_ra_pg"] ** 2
    tg["pyth"] = rs2 / (rs2 + ra2)
    tg["off_games_played"] = n

    n15 = tg["f_n_r15"].clip(lower=0)
    w15 = 8.0
    tg["form15_rs_pg"] = (tg["f_rs_r15"].fillna(0) + LEAGUE["rs_pg"] * w15) / (n15 + w15)
    tg["form15_ra_pg"] = (tg["f_ra_r15"].fillna(0) + LEAGUE["rs_pg"] * w15) / (n15 + w15)
    tg["form15_wpct"] = (tg["f_win_r15"].fillna(0) + 0.5 * w15) / (n15 + w15)

    # rest days & schedule density (calendar-based, across season boundaries OK)
    tg = tg.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    prev_date = tg.groupby("team")["game_date"].shift(1)
    same_day = prev_date == tg["game_date"]
    prev_date = prev_date.where(~same_day, tg.groupby("team")["game_date"].shift(2))
    tg["rest_days"] = (tg["game_date"] - prev_date).dt.days.clip(upper=10).fillna(5).astype(float)
    ones = tg[["team", "game_date"]].copy(); ones["g"] = 1.0
    daily_games = ones.groupby(["team", "game_date"], as_index=False)["g"].sum()
    tg = _calendar_window_sum(daily_games, tg, "team", "g", 7, "games_last7")

    keep = ["game_id", "team", "off_rs_pg", "def_ra_pg", "win_pct", "pyth", "off_games_played",
            "form15_rs_pg", "form15_ra_pg", "form15_wpct", "rest_days", "games_last7"]
    return tg[keep]


# ---------------------------------------------------------------------------
# Starter features
# ---------------------------------------------------------------------------

def starter_features_asof(starter_lines: pd.DataFrame, games: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """As-of starter features for `queries` rows of (query_id, pitcher_id, game_date).

    Missing/TBD pitchers (NaN id) get league-mean features.
    """
    sl = starter_lines.merge(games[["game_id", "game_date", "season"]], on="game_id", how="left")
    sl["game_date"] = to_datetime_flex(sl["game_date"])
    sl = sl.dropna(subset=["pitcher_id", "game_date"]).copy()
    sl["pitcher_id"] = sl["pitcher_id"].astype(np.int64)
    sl["k"] = sl["strikeouts"].fillna(0)
    sl["bb"] = sl["walks"].fillna(0)
    sl["hr"] = sl["home_runs_allowed"].fillna(0)
    sl["h"] = sl["hits_allowed"].fillna(0)
    sl["er"] = sl["earned_runs"].fillna(0)
    sl["ip"] = sl["innings_pitched"].fillna(0)
    sl["pitches"] = sl["pitches_thrown"].fillna(0)
    sl["gs"] = 1.0

    q = queries.copy()
    q["game_date"] = to_datetime_flex(q["game_date"])
    known = q["pitcher_id"].notna()
    qk = q[known].copy()
    qk["pitcher_id"] = qk["pitcher_id"].astype(np.int64)
    for c in ["k", "bb", "hr", "h", "er", "ip", "pitches", "gs"]:
        qk[c] = 0.0
    qk["game_id"] = -qk["query_id"] - 1  # unique pseudo ids, never collide with real ones

    allrows = pd.concat([sl.assign(query_id=np.nan), qk], ignore_index=True)
    stat_cols = ["k", "bb", "hr", "h", "er", "ip", "pitches", "gs"]
    allrows = _shifted_cum(allrows, "pitcher_id", stat_cols, "c_")
    allrows = _shifted_roll(allrows, "pitcher_id", stat_cols, 10, "r_")

    res = allrows[allrows["query_id"].notna()].copy()

    def rates(ipcol, kcol, bbcol, hrcol, hcol, ercol, shrink_ip):
        ip = res[ipcol].fillna(0)
        w = shrink_ip
        # blend league-average numerators for w innings
        k9 = 9 * (res[kcol].fillna(0) + LEAGUE["k9"] / 9 * w) / (ip + w)
        bb9 = 9 * (res[bbcol].fillna(0) + LEAGUE["bb9"] / 9 * w) / (ip + w)
        hr9 = 9 * (res[hrcol].fillna(0) + LEAGUE["hr9"] / 9 * w) / (ip + w)
        era = 9 * (res[ercol].fillna(0) + LEAGUE["era"] / 9 * w) / (ip + w)
        whip = (res[hcol].fillna(0) + res[bbcol].fillna(0) + LEAGUE["whip"] * w) / (ip + w)
        fip = (13 * (res[hrcol].fillna(0) + LEAGUE["hr9"] / 9 * w)
               + 3 * (res[bbcol].fillna(0) + LEAGUE["bb9"] / 9 * w)
               - 2 * (res[kcol].fillna(0) + LEAGUE["k9"] / 9 * w)) / (ip + w) + FIP_CONSTANT
        return era, whip, k9, bb9, hr9, fip

    res["starter_era"], res["starter_whip"], res["starter_k9"], res["starter_bb9"], res["starter_hr9"], res["starter_fip"] = \
        rates("c_ip_cum", "c_k_cum", "c_bb_cum", "c_hr_cum", "c_h_cum", "c_er_cum", STARTER_SHRINK_IP)
    _, _, _, _, _, res["starter_l10_fip"] = \
        rates("r_ip_r10", "r_k_r10", "r_bb_r10", "r_hr_r10", "r_h_r10", "r_er_r10", STARTER_L10_SHRINK_IP)
    gs = res["c_gs_cum"].fillna(0)
    res["starter_pps"] = (res["c_pitches_cum"].fillna(0) + LEAGUE["pitches_per_start"] * 3) / (gs + 3)
    res["starter_ip"] = res["c_ip_cum"].fillna(0)
    res["starter_gs"] = gs

    feat_cols = ["starter_era", "starter_whip", "starter_k9", "starter_bb9", "starter_hr9",
                 "starter_fip", "starter_l10_fip", "starter_pps", "starter_ip", "starter_gs"]
    out = q[["query_id"]].merge(res[["query_id"] + feat_cols], on="query_id", how="left")
    defaults = {"starter_era": LEAGUE["era"], "starter_whip": LEAGUE["whip"], "starter_k9": LEAGUE["k9"],
                "starter_bb9": LEAGUE["bb9"], "starter_hr9": LEAGUE["hr9"], "starter_fip": LEAGUE["fip"],
                "starter_l10_fip": LEAGUE["fip"], "starter_pps": LEAGUE["pitches_per_start"],
                "starter_ip": 0.0, "starter_gs": 0.0}
    for c, v in defaults.items():
        out[c] = out[c].fillna(v)
    return out


# ---------------------------------------------------------------------------
# Bullpen features
# ---------------------------------------------------------------------------

def bullpen_features_asof(bullpen_team_game: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """As-of bullpen features for `queries` rows of (query_id, team_abbr, game_date)."""
    bp = bullpen_team_game.copy()
    bp["game_date"] = to_datetime_flex(bp["game_date"])
    for c in ["bullpen_ip", "bullpen_er", "bullpen_so", "bullpen_bb", "bullpen_hits", "bullpen_hr", "bullpen_pitches"]:
        bp[c] = bp[c].fillna(0)

    q = queries.copy()
    q["game_date"] = to_datetime_flex(q["game_date"])
    qb = q.rename(columns={"team_abbr": "team_abbr"}).copy()
    for c in ["bullpen_ip", "bullpen_er", "bullpen_so", "bullpen_bb", "bullpen_hits", "bullpen_hr", "bullpen_pitches"]:
        qb[c] = 0.0
    qb["game_id"] = -qb["query_id"] - 1
    qb["season"] = qb["game_date"].dt.year

    allrows = pd.concat([bp.assign(query_id=np.nan), qb], ignore_index=True)
    allrows["skey"] = allrows["team_abbr"] + "_" + allrows["season"].astype(int).astype(str)
    stat_cols = ["bullpen_ip", "bullpen_er", "bullpen_so", "bullpen_bb", "bullpen_hits", "bullpen_hr", "bullpen_pitches"]
    allrows = _shifted_cum(allrows, "skey", stat_cols, "c_")
    allrows = _shifted_roll(allrows, "skey", ["bullpen_ip"], 3, "g_")

    res = allrows[allrows["query_id"].notna()].copy()
    ip = res["c_bullpen_ip_cum"].fillna(0)
    w = BULLPEN_SHRINK_IP
    res["bullpen_era"] = 9 * (res["c_bullpen_er_cum"].fillna(0) + LEAGUE["bp_era"] / 9 * w) / (ip + w)
    res["bullpen_whip"] = (res["c_bullpen_hits_cum"].fillna(0) + res["c_bullpen_bb_cum"].fillna(0) + LEAGUE["bp_whip"] * w) / (ip + w)
    res["bullpen_k9"] = 9 * (res["c_bullpen_so_cum"].fillna(0) + LEAGUE["bp_k9"] / 9 * w) / (ip + w)
    res["bullpen_bb9"] = 9 * (res["c_bullpen_bb_cum"].fillna(0) + LEAGUE["bp_bb9"] / 9 * w) / (ip + w)
    res["bullpen_fip"] = (13 * (res["c_bullpen_hr_cum"].fillna(0) + LEAGUE["hr9"] / 9 * w)
                          + 3 * (res["c_bullpen_bb_cum"].fillna(0) + LEAGUE["bp_bb9"] / 9 * w)
                          - 2 * (res["c_bullpen_so_cum"].fillna(0) + LEAGUE["bp_k9"] / 9 * w)) / (ip + w) + FIP_CONSTANT
    res["bullpen_ip_last3g"] = res["g_bullpen_ip_r3"].fillna(0)

    # calendar fatigue
    daily = bp.groupby(["team_abbr", "game_date"], as_index=False)["bullpen_pitches"].sum()
    res3 = _calendar_window_sum(daily, res[["team_abbr", "game_date", "query_id"]], "team_abbr", "bullpen_pitches", 3, "bullpen_pitches_last3d")
    res1 = _calendar_window_sum(daily, res[["team_abbr", "game_date", "query_id"]], "team_abbr", "bullpen_pitches", 1, "bullpen_pitches_last1d")
    res = res.merge(res3[["query_id", "bullpen_pitches_last3d"]], on="query_id") \
             .merge(res1[["query_id", "bullpen_pitches_last1d"]], on="query_id")

    feat_cols = ["bullpen_era", "bullpen_whip", "bullpen_k9", "bullpen_bb9", "bullpen_fip",
                 "bullpen_ip_last3g", "bullpen_pitches_last3d", "bullpen_pitches_last1d"]
    return q[["query_id"]].merge(res[["query_id"] + feat_cols], on="query_id", how="left").fillna({
        "bullpen_era": LEAGUE["bp_era"], "bullpen_whip": LEAGUE["bp_whip"], "bullpen_k9": LEAGUE["bp_k9"],
        "bullpen_bb9": LEAGUE["bp_bb9"], "bullpen_fip": LEAGUE["bp_fip"],
        "bullpen_ip_last3g": 0.0, "bullpen_pitches_last3d": 0.0, "bullpen_pitches_last1d": 0.0})


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------

def park_factors_by_season(games_completed: pd.DataFrame) -> pd.DataFrame:
    """(venue_id, season) -> park factor computed from strictly earlier seasons,
    regressed toward 1.0 by PARK_SHRINK_GAMES pseudo-games."""
    g = games_completed.dropna(subset=["venue_id", "home_score", "away_score"]).copy()
    g["total_runs"] = g["home_score"] + g["away_score"]
    per = g.groupby(["venue_id", "season"], as_index=False).agg(n=("game_id", "count"), runs=("total_runs", "sum"))
    league = g.groupby("season", as_index=False).agg(ln=("game_id", "count"), lruns=("total_runs", "sum"))

    seasons = sorted(g["season"].unique().tolist())
    rows = []
    for s in seasons + [max(seasons) + 1]:
        prior = per[per["season"] < s].groupby("venue_id", as_index=False).agg(n=("n", "sum"), runs=("runs", "sum"))
        lp = league[league["season"] < s]
        if lp["ln"].sum() == 0:
            continue
        lrpg = lp["lruns"].sum() / lp["ln"].sum()
        prior["raw_pf"] = (prior["runs"] / prior["n"]) / lrpg
        prior["park_factor"] = (prior["n"] * prior["raw_pf"] + PARK_SHRINK_GAMES * 1.0) / (prior["n"] + PARK_SHRINK_GAMES)
        prior["season"] = s
        rows.append(prior[["venue_id", "season", "park_factor"]])
    if not rows:
        return pd.DataFrame(columns=["venue_id", "season", "park_factor"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Game-level assembly (shared by training + daily)
# ---------------------------------------------------------------------------

def build_game_features(target_games: pd.DataFrame, games_completed: pd.DataFrame,
                        starter_lines: pd.DataFrame, bullpen_team_game: pd.DataFrame) -> pd.DataFrame:
    """Attach every V5 feature to `target_games` (historical or today's slate).

    `target_games` needs: game_id, game_date, season, home/away_team_abbr,
    home/away starter id columns (home_starter_id/away_starter_id or
    home_probable_pitcher_id/away_probable_pitcher_id), venue_id.
    """
    tgt = target_games.copy()
    tgt["game_date"] = to_datetime_flex(tgt["game_date"])
    hid = "home_starter_id" if "home_starter_id" in tgt else "home_probable_pitcher_id"
    aid = "away_starter_id" if "away_starter_id" in tgt else "away_probable_pitcher_id"

    gc = games_completed.copy()
    gc["game_date"] = to_datetime_flex(gc["game_date"])

    # --- team features
    tf = team_features_asof(gc, tgt)
    home_tf = tf.add_prefix("home_").rename(columns={"home_game_id": "game_id", "home_team": "home_team_abbr"})
    away_tf = tf.add_prefix("away_").rename(columns={"away_game_id": "game_id", "away_team": "away_team_abbr"})
    tgt = tgt.merge(home_tf, on=["game_id", "home_team_abbr"], how="left")
    tgt = tgt.merge(away_tf, on=["game_id", "away_team_abbr"], how="left")

    # --- starter features
    sq = pd.concat([
        pd.DataFrame({"query_id": tgt.index * 2, "pitcher_id": tgt[hid], "game_date": tgt["game_date"]}),
        pd.DataFrame({"query_id": tgt.index * 2 + 1, "pitcher_id": tgt[aid], "game_date": tgt["game_date"]}),
    ], ignore_index=True)
    sf = starter_features_asof(starter_lines, gc, sq).set_index("query_id")
    for side, off in [("home", 0), ("away", 1)]:
        block = sf.loc[tgt.index * 2 + off].reset_index(drop=True)
        tgt = pd.concat([tgt.reset_index(drop=True), block.add_prefix(f"{side}_")], axis=1)

    # --- bullpen features
    bq = pd.concat([
        pd.DataFrame({"query_id": tgt.index * 2, "team_abbr": tgt["home_team_abbr"], "game_date": tgt["game_date"]}),
        pd.DataFrame({"query_id": tgt.index * 2 + 1, "team_abbr": tgt["away_team_abbr"], "game_date": tgt["game_date"]}),
    ], ignore_index=True)
    bf = bullpen_features_asof(bullpen_team_game, bq).set_index("query_id")
    for side, off in [("home", 0), ("away", 1)]:
        block = bf.loc[tgt.index * 2 + off].reset_index(drop=True)
        tgt = pd.concat([tgt.reset_index(drop=True), block.add_prefix(f"{side}_")], axis=1)

    # --- park factor
    pf = park_factors_by_season(gc)
    tgt["season"] = tgt["season"].astype(int)
    tgt = tgt.merge(pf, on=["venue_id", "season"], how="left")
    tgt["park_factor"] = tgt["park_factor"].fillna(1.0)

    # --- classifier diffs
    tgt["pyth_diff"] = tgt["home_pyth"] - tgt["away_pyth"]
    tgt["win_pct_diff"] = tgt["home_win_pct"] - tgt["away_win_pct"]
    tgt["rs_pg_diff"] = tgt["home_off_rs_pg"] - tgt["away_off_rs_pg"]
    tgt["ra_pg_diff"] = tgt["home_def_ra_pg"] - tgt["away_def_ra_pg"]
    tgt["form15_rs_diff"] = tgt["home_form15_rs_pg"] - tgt["away_form15_rs_pg"]
    tgt["form15_ra_diff"] = tgt["home_form15_ra_pg"] - tgt["away_form15_ra_pg"]
    tgt["form15_wpct_diff"] = tgt["home_form15_wpct"] - tgt["away_form15_wpct"]
    for stat in ["fip", "era", "whip", "k9", "bb9", "hr9", "l10_fip", "pps", "ip"]:
        tgt[f"starter_{stat}_diff"] = tgt[f"home_starter_{stat}"] - tgt[f"away_starter_{stat}"]
    for stat in ["fip", "k9", "bb9", "pitches_last3d", "pitches_last1d", "ip_last3g"]:
        tgt[f"bullpen_{stat}_diff"] = tgt[f"home_bullpen_{stat}"] - tgt[f"away_bullpen_{stat}"]
    tgt["rest_days_diff"] = tgt["home_rest_days"] - tgt["away_rest_days"]
    tgt["games_last7_diff"] = tgt["home_games_last7"] - tgt["away_games_last7"]
    return tgt


def to_run_model_rows(game_features: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Reshape game-level features into two offense-perspective rows per game
    for the run regressors."""
    rows = []
    for off, deff, is_home in [("home", "away", 1.0), ("away", "home", 0.0)]:
        r = pd.DataFrame({
            "game_id": game_features["game_id"],
            "is_home": is_home,
            "off_rs_pg": game_features[f"{off}_off_rs_pg"],
            "off_form15_rs_pg": game_features[f"{off}_form15_rs_pg"],
            "off_games_played": game_features[f"{off}_off_games_played"],
            "opp_starter_fip": game_features[f"{deff}_starter_fip"],
            "opp_starter_era": game_features[f"{deff}_starter_era"],
            "opp_starter_whip": game_features[f"{deff}_starter_whip"],
            "opp_starter_k9": game_features[f"{deff}_starter_k9"],
            "opp_starter_bb9": game_features[f"{deff}_starter_bb9"],
            "opp_starter_hr9": game_features[f"{deff}_starter_hr9"],
            "opp_starter_l10_fip": game_features[f"{deff}_starter_l10_fip"],
            "opp_starter_pps": game_features[f"{deff}_starter_pps"],
            "opp_starter_ip": game_features[f"{deff}_starter_ip"],
            "opp_bullpen_fip": game_features[f"{deff}_bullpen_fip"],
            "opp_bullpen_k9": game_features[f"{deff}_bullpen_k9"],
            "opp_bullpen_bb9": game_features[f"{deff}_bullpen_bb9"],
            "opp_bullpen_pitches_last3d": game_features[f"{deff}_bullpen_pitches_last3d"],
            "opp_bullpen_pitches_last1d": game_features[f"{deff}_bullpen_pitches_last1d"],
            "opp_bullpen_ip_last3g": game_features[f"{deff}_bullpen_ip_last3g"],
            "park_factor": game_features["park_factor"],
            "off_rest_days": game_features[f"{off}_rest_days"],
            "off_games_last7": game_features[f"{off}_games_last7"],
            "side": off,
        })
        if include_target and f"{off}_score" in game_features:
            r["runs"] = game_features[f"{off}_score"]
        if "_row" in game_features:
            r["_row"] = game_features["_row"].values
        rows.append(r)
    return pd.concat(rows, ignore_index=True)
