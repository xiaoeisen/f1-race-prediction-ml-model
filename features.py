"""Feature engineering. Every 'history' feature uses only races BEFORE the current one
(shift(1)), so there is no leakage. Weekend data (FP1/FP2/quali/weather) is known
before the race starts, so it's fair game."""
import numpy as np
import pandas as pd

from circuits import canon, circuit_facts

# Keep team identity across rebrands so team form carries over
TEAM_ALIASES = {
    "Toro Rosso": "RB", "AlphaTauri": "RB", "Racing Bulls": "RB",
    "Alfa Romeo": "Sauber", "Alfa Romeo Racing": "Sauber", "Kick Sauber": "Sauber", "Audi": "Sauber",
    "Racing Point": "Aston Martin", "Force India": "Aston Martin",
    "Renault": "Alpine", "Haas F1 Team": "Haas", "Red Bull Racing": "Red Bull",
    "Alpine F1 Team": "Alpine", "RB F1 Team": "RB",
}

FEATURES_WITH_PACE = [
    # this weekend (known before lights out)
    "grid", "grid_pen", "quali_pos", "q_best_gap", "q_best_gap_vs_tm",
    "fp1_best_gap", "fp2_best_gap", "fp3_best_gap", "fp1_pos", "fp2_pos", "fp3_pos",
    "fp1_long_gap", "fp2_long_gap", "long_pace", "long_pace_rank", "fp2_long_gap_vs_tm",
    "fp1_deg_per10", "fp2_deg_per10", "fp3_deg_per10",
    "q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap", "q_straight_bias",
    "air_temp", "track_temp", "humidity", "wind", "rain", "is_sprint",
    # driver skill / tendencies
    "drv_elo", "drv_finish_l5", "drv_finish_l10", "drv_grid_l5", "drv_gain_l10",
    "drv_dnf_l20", "drv_circuit_finish", "drv_circuit_n", "drv_wet_gain", "drv_races",
    "drv_pace_l5", "drv_pace_vs_tm_l5", "drv_deg_l5", "drv_lap1_gain_l10", "drv_stops_l10", "drv_ot_net_l10",
    "drv_qtm_l10", "drv_ftm_l10",   # driver vs teammate (same car) = driver skill isolated
    # car / team (finishing results AND true race pace)
    "team_elo", "team_finish_l5", "team_grid_l5", "team_dnf_l10",
    "team_pace_l3", "team_pace_l8", "team_pace_trend", "team_deg_l3", "team_deg_l8",
    "team_pitlane_l10", "team_stop_l10",
    # circuit character
    "circ_overtake", "circ_dnf", "circ_sc_rate", "circ_red_rate", "circ_sc_frac", "circ_stops",
    "street", "pit_loss", "tyre_wear",
    # season context
    "pts_before", "champ_pos", "round",
    # standing within THIS field (absolute values drift between eras)
    "elo_sum_rk", "drv_elo_rk", "team_elo_rk", "drv_finish_l5_rk", "team_finish_l5_rk",
    "team_pace_l3_rk", "q_best_gap_rk", "drv_pace_l5_rk",
]

# Added with the lap-level tyre-wear work; omit these from the prior-model baseline.
NEW_PACE_FEATURES = {"fp1_deg_per10", "fp2_deg_per10", "fp3_deg_per10",
                     "drv_deg_l5", "team_deg_l3", "team_deg_l8"}

# Signals whose value comes from long-run pace, tyre wear, qualifying sectors or
# historical race pace. Kept as one list so the chronological benchmark can run
# a clean pace-feature ablation against the same races and folds.
PACE_FEATURES = {
    "fp1_long_gap", "fp2_long_gap", "long_pace", "long_pace_rank", "fp2_long_gap_vs_tm",
    "fp1_deg_per10", "fp2_deg_per10", "fp3_deg_per10",
    "q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap", "q_straight_bias",
    "drv_pace_l5", "drv_pace_vs_tm_l5", "drv_deg_l5",
    "team_pace_l3", "team_pace_l8", "team_pace_trend", "team_deg_l3", "team_deg_l8",
    "team_pace_l3_rk", "drv_pace_l5_rk",
}
PACE_PL_FEATURES = ["fp1_long_gap", "fp2_long_gap", "drv_pace_l5", "team_pace_l3",
                    "drv_deg_l5", "team_deg_l3"]

# Keep the full signal set in the production model; the benchmark contains a
# paired no-pace ablation so sparse-data value is measured on each refresh.
FEATURES = FEATURES_WITH_PACE

# Before qualifying the grid and quali results don't exist yet: the "pre" model never sees them
QUALI_FEATURES = {"grid", "grid_pen", "grid_log", "quali_pos", "q_best_gap", "q_best_gap_vs_tm", "q_best_gap_rk",
                  "q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap", "q_straight_bias"}
FEATURES_PRE = [c for c in FEATURES if c not in QUALI_FEATURES]
# compact feature set for the linear Plackett-Luce part of the ensemble
PL_FEATURES = ["grid_log", "quali_pos", "q_best_gap", "fp1_best_gap", "fp2_best_gap", "drv_elo", "team_elo",
               "drv_finish_l5", "team_finish_l5", "team_grid_l5", "drv_grid_l5", "drv_gain_l10", "drv_dnf_l20",
               "team_dnf_l10", "pts_before", "champ_pos", "drv_qtm_l10", "drv_ftm_l10"]
PL_FEATURES_PRE = [c for c in PL_FEATURES if c not in QUALI_FEATURES]
# raw columns blanked to imitate "right after FP1" when training the pre-quali model
LATE_PRACTICE_COLS = ["fp2_best_gap", "fp3_best_gap", "fp2_long_gap",
                      "fp2_deg_per10", "fp3_deg_per10"]

RANK_SOURCES = ["elo_sum", "drv_elo", "team_elo", "drv_finish_l5", "team_finish_l5", "team_pace_l3", "q_best_gap", "drv_pace_l5"]
RANK_HIGH_GOOD = {"elo_sum", "drv_elo", "team_elo"}      # for these a HIGHER value is better
RANK_FEATURES = [c + "_rk" for c in RANK_SOURCES]

# columns older CSVs may lack -> created empty so nothing crashes
OPTIONAL_COLS = [
    "fp1_best_gap", "fp2_best_gap", "fp3_best_gap", "fp1_long_gap", "fp2_long_gap",
    "fp1_best_time_s", "fp2_best_time_s", "fp3_best_time_s", "fp1_gap_s", "fp2_gap_s", "fp3_gap_s",
    "fp1_laps", "fp2_laps", "fp3_laps", "q_best_time_s", "q_gap_s", "q_time_segment_valid", "quali_laps",
    "race_best_lap_s", "race_avg_clean_lap_s", "race_laps",
    "fp1_deg_per10", "fp2_deg_per10", "fp3_deg_per10", "race_deg_per10",
    "air_temp", "track_temp", "humidity", "wind", "rain", "is_sprint", "points", "dnf",
    "race_pace_gap", "lap1_gain", "race_sc", "race_vsc", "race_red", "sc_frac",
    "q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap",
    "q_rain", "q_air_temp", "q_track_temp", "q_humidity", "q_wind",
    "n_stops", "pit_lane_s", "stop_s", "overtakes_made", "overtakes_suffered", "n_sc", "n_vsc",
]


MECH_WORDS = ("engine", "gearbox", "hydraulic", "power", "brake", "electr", "suspension", "turbo", "fuel", "oil", "water",
              "cool", "exhaust", "transmission", "clutch", "mechanical", "overheat", "drive", "battery", "ers", "mgu",
              "differential", "radiator", "steering", "vibration", "pneumatic", "undertray", "skid", "ignition", "throttle",
              "wheel", "spark", "pressure", "leak", "failure")
INCIDENT_WORDS = ("collision", "accident", "spun", "crash", "contact", "damage", "puncture", "debris")


def _status_flag(status, dnf, words):
    """1.0 where the retirement reason matches `words`, 0.0 otherwise, NaN where the result isn't known."""
    s = status.fillna("").astype(str).str.lower()
    hit = s.str.contains("|".join(words)) & (dnf == 1)
    return pd.Series(np.where(dnf.isna(), np.nan, hit.astype(float)), index=dnf.index)


def _past(df, by, col, n=None):
    """Mean of the previous n races (or all previous if n is None) per group, NaNs skipped.
    Vectorised with cumulative sums: window sum = cumsum[k] - cumsum[k-n]."""
    g = df.groupby(by)[col]
    xs = g.shift(1)
    ok = xs.notna().astype(float)
    xv = xs.fillna(0.0)
    keys = [df[c] for c in ([by] if isinstance(by, str) else by)]
    cs = xv.groupby(keys).cumsum()
    cc = ok.groupby(keys).cumsum()
    if n is not None:
        cs = cs - cs.groupby(keys).shift(n, fill_value=0.0)
        cc = cc - cc.groupby(keys).shift(n, fill_value=0.0)
    return cs / cc.where(cc > 0)


# Tunable knobs (see benchmark.py). shrink = share of a rating forgotten at the start of each season
# (cars change a lot between seasons, drivers much less).
CFG = dict(elo_k=24.0, elo_team_shrink=0.3, elo_drv_shrink=0.1,
           elo_reg_shrink=0.7,      # extra team-rating reset at big rule changes (REG_CHANGES)
           rookie_impute=True,     # fill missing driver form from the team's form
           weather="pre")          # 'pre' = qualifying-session weather for training (no in-race info); 'race' = observed race weather (leaks)


REG_CHANGES = {2009, 2014, 2017, 2022, 2026}   # seasons that brought new technical regulations


def _elo(tbl, k=24.0, base=1500.0, shrink=0.0, reg_shrink=0.0):
    """tbl: columns race_id, id, finish, year (sorted by race_id). Returns pre-race Elo per row.
    Pairwise Elo: every competitor 'plays' every other one in the race."""
    race = tbl["race_id"].to_numpy()
    ids = tbl["id"].to_numpy()
    fin = tbl["finish"].to_numpy(float)
    years = tbl["year"].to_numpy()
    cuts = np.flatnonzero(np.diff(race)) + 1
    ratings, out, last_year = {}, np.zeros(len(tbl)), None
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(tbl)]):
        if last_year is not None and years[lo] != last_year:
            s = max(shrink, reg_shrink) if int(years[lo]) in REG_CHANGES else shrink
            if s:
                ratings = {i: base + (r - base) * (1 - s) for i, r in ratings.items()}
        last_year = years[lo]
        who = ids[lo:hi]
        r = np.array([ratings.get(i, base) for i in who])
        out[lo:hi] = r
        f_, n = fin[lo:hi], hi - lo
        if n < 2 or np.isnan(f_).any():
            continue  # upcoming race: nothing to update
        exp = 1.0 / (1.0 + 10.0 ** ((r[None, :] - r[:, None]) / 400.0))
        act = (f_[:, None] < f_[None, :]) + 0.5 * (f_[:, None] == f_[None, :])
        np.fill_diagonal(exp, 0.0)
        np.fill_diagonal(act, 0.0)
        delta = (act - exp).sum(axis=1)
        for i, d in zip(who, delta):
            ratings[i] = ratings.get(i, base) + k * d / (n - 1)
    return out


def weekend_practice(df):
    """Practice-derived columns (in place). Split out so a 'right after FP1' copy of the data
    can be rebuilt cheaply after blanking the later practice columns."""
    df["long_pace"] = df[["fp1_long_gap", "fp2_long_gap"]].mean(axis=1)
    df["long_pace_rank"] = df.groupby("race_id")["long_pace"].rank(pct=True)
    for p in ("fp1", "fp2", "fp3"):  # practice classification (position on the timesheet)
        df[f"{p}_pos"] = df.groupby("race_id")[f"{p}_best_gap"].rank(method="min")
    df["fp2_long_gap_vs_tm"] = df["fp2_long_gap"] - df.groupby(["race_id", "team"])["fp2_long_gap"].transform("mean")
    return df


def add_features(df, cfg=None):
    cfg = {**CFG, **(cfg or {})}
    df = df.copy()
    df["team"] = df["team"].replace(TEAM_ALIASES)
    df["circuit"] = df["circuit"].map(canon)
    df = df.sort_values(["year", "round", "driver"]).reset_index(drop=True)
    df["race_id"] = df.groupby(["year", "round"]).ngroup()

    for c in OPTIONAL_COLS:
        if c not in df:
            df[c] = np.nan

    if "status" not in df:
        df["status"] = np.nan
    # incident takes priority over a mechanical keyword ("collision damage" is an incident)
    df["dnf_inc"] = _status_flag(df["status"], df["dnf"], INCIDENT_WORDS)
    df["dnf_mech"] = _status_flag(df["status"], df["dnf"], MECH_WORDS) * (1 - df["dnf_inc"].fillna(0))
    # Weather: the frozen-at-lights-out rule. Weather measured DURING the race is not known beforehand, so for
    # finished races the model trains on the qualifying-session weather (cfg weather='pre'); upcoming races use the
    # live forecast that apply_live() writes into these same columns. weather='race' keeps the old behaviour (leaks).
    rain_seen = df["rain"].copy()          # what actually happened: only ever used for HISTORY (wet_gain below)
    if cfg["weather"] == "pre":
        finished = df["finish"].notna()
        for c in ("rain", "air_temp", "track_temp", "humidity", "wind"):
            if f"q_{c}" in df:
                df.loc[finished, c] = df.loc[finished, f"q_{c}"]
            else:
                df.loc[finished, c] = np.nan
    df["gain"] = df["grid"] - df["finish"]                       # positions gained on Sunday
    df["gain_clean"] = df["gain"].where(df["dnf"] == 0)          # ignore retirements
    df["wet_gain"] = df["gain_clean"].where(rain_seen >= 0.5)    # past wet races (shifted before use, so no leak)

    # ---- this-weekend features
    df["grid_pen"] = df["grid"] - df["quali_pos"]                # penalties / pit-lane starts
    df["grid_log"] = np.log(df["grid"].clip(lower=1))            # grid matters most at the front
    df["q_best_gap_vs_tm"] = df["q_best_gap"] - df.groupby(["race_id", "team"])["q_best_gap"].transform("mean")
    weekend_practice(df)
    # car character from qualifying: positive = relatively slower on straights than in corners
    q_corner = df[["q_s1_gap", "q_s2_gap", "q_s3_gap"]].mean(axis=1)
    df["q_straight_bias"] = df["q_topspeed_gap"] - q_corner

    # race pace is only known AFTER a race, so it is used through history only
    df["finish_vs_tm"] = df["finish"] - df.groupby(["race_id", "team"])["finish"].transform("mean")
    df["pace_vs_tm"] = df["race_pace_gap"] - df.groupby(["race_id", "team"])["race_pace_gap"].transform("mean")

    # ---- driver history
    df["drv_finish_l5"] = _past(df, "driver", "finish", 5)
    df["drv_finish_l10"] = _past(df, "driver", "finish", 10)
    df["drv_grid_l5"] = _past(df, "driver", "grid", 5)
    df["drv_gain_l10"] = _past(df, "driver", "gain_clean", 10)      # race craft
    df["drv_dnf_l20"] = _past(df, "driver", "dnf", 20)
    df["drv_inc_l20"] = _past(df, "driver", "dnf_inc", 20)          # error-proneness (collisions, spins)
    df["drv_wet_gain"] = _past(df, "driver", "wet_gain")            # wet-weather tendency
    df["drv_pace_l5"] = _past(df, "driver", "race_pace_gap", 5)     # true Sunday pace
    df["drv_pace_vs_tm_l5"] = _past(df, "driver", "pace_vs_tm", 5)  # pace vs teammate (driver, not car)
    df["drv_deg_l5"] = _past(df, "driver", "race_deg_per10", 5)    # historical race tyre degradation
    df["drv_lap1_gain_l10"] = _past(df, "driver", "lap1_gain", 10)  # first-lap tendency
    df["drv_qtm_l10"] = _past(df, "driver", "q_best_gap_vs_tm", 10)   # qualifying vs teammate, history
    df["drv_ftm_l10"] = _past(df, "driver", "finish_vs_tm", 10)       # race result vs teammate, history
    df["drv_stops_l10"] = _past(df, "driver", "n_stops", 10)        # strategy style
    df["ot_net"] = df["overtakes_made"] - df["overtakes_suffered"]
    df["drv_ot_net_l10"] = _past(df, "driver", "ot_net", 10)        # on-track passing record
    df["drv_circuit_finish"] = _past(df, ["driver", "circuit"], "finish")
    df["drv_circuit_n"] = df.groupby(["driver", "circuit"]).cumcount()
    df["drv_races"] = df.groupby("driver").cumcount()

    # ---- team history (team-level table: one row per team per race)
    t = (df.groupby(["race_id", "team"], as_index=False)
           .agg(finish=("finish", "mean"), grid=("grid", "mean"), dnf=("dnf", "mean"),
                pace=("race_pace_gap", "mean"), deg=("race_deg_per10", "mean"),
                pitlane=("pit_lane_s", "mean"), stop=("stop_s", "mean"),
                mech=("dnf_mech", "mean"))
           .sort_values(["race_id", "team"]).reset_index(drop=True))
    t["team_finish_l5"] = _past(t, "team", "finish", 5)
    t["team_grid_l5"] = _past(t, "team", "grid", 5)
    t["team_dnf_l10"] = _past(t, "team", "dnf", 10)
    t["team_mech_l10"] = _past(t, "team", "mech", 10)               # car reliability
    t["team_pace_l3"] = _past(t, "team", "pace", 3)
    t["team_pace_l8"] = _past(t, "team", "pace", 8)
    t["team_pace_trend"] = t["team_pace_l3"] - t["team_pace_l8"]    # negative = car getting faster (upgrades)
    t["team_deg_l3"] = _past(t, "team", "deg", 3)
    t["team_deg_l8"] = _past(t, "team", "deg", 8)
    t["team_pitlane_l10"] = _past(t, "team", "pitlane", 10)         # pit-crew / pit-lane efficiency
    t["team_stop_l10"] = _past(t, "team", "stop", 10)               # stationary stop time (2024+)
    t["year"] = t["race_id"].map(df.groupby("race_id")["year"].first())
    t["team_elo"] = _elo(t.rename(columns={"team": "id"})[["race_id", "id", "finish", "year"]],
                         k=cfg["elo_k"], shrink=cfg["elo_team_shrink"], reg_shrink=cfg["elo_reg_shrink"])
    team_cols = ["team_finish_l5", "team_grid_l5", "team_dnf_l10", "team_mech_l10", "team_pace_l3", "team_pace_l8",
                 "team_pace_trend", "team_deg_l3", "team_deg_l8", "team_pitlane_l10", "team_stop_l10", "team_elo"]
    df = df.merge(t[["race_id", "team"] + team_cols], on=["race_id", "team"], how="left")

    # ---- driver Elo
    d = df[["race_id", "driver", "finish", "year"]].rename(columns={"driver": "id"})
    df["drv_elo"] = _elo(d, k=cfg["elo_k"], shrink=cfg["elo_drv_shrink"])

    # ---- circuit character (shuffling, retirements, safety cars / red flags)
    rc = (df.groupby("race_id")
            .agg(circuit=("circuit", "first"),
                 abs_gain=("gain_clean", lambda s: s.abs().mean()),
                 dnf_rate=("dnf", "mean"),
                 sc=("race_sc", "first"), red=("race_red", "first"), scf=("sc_frac", "first"),
                 stops=("n_stops", "mean"))
            .reset_index().sort_values("race_id"))
    for new, col in (("circ_overtake", "abs_gain"), ("circ_dnf", "dnf_rate"), ("circ_sc_rate", "sc"),
                     ("circ_red_rate", "red"), ("circ_sc_frac", "scf"),
                     ("circ_stops", "stops")):
        rc[new] = rc.groupby("circuit")[col].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df = df.merge(rc[["race_id", "circ_overtake", "circ_dnf", "circ_sc_rate", "circ_red_rate", "circ_sc_frac", "circ_stops"]],
                  on="race_id", how="left")

    # ---- fixed circuit facts (street circuit, pit-lane time loss, tyre wear) from circuits.csv
    facts = circuit_facts(df["circuit"])
    for c in facts:
        df[c] = facts[c].to_numpy()

    # ---- seasons since the last big regulation change (cars/habits from before it matter less)
    reg = np.array(sorted(REG_CHANGES))
    df["era_age"] = [int(y - reg[reg <= y].max()) if (reg <= y).any() else 0 for y in df["year"]]

    # ---- rookies / reserve drivers: no history yet -> start from what the CAR does
    if cfg["rookie_impute"]:
        for drv, team in (("drv_finish_l5", "team_finish_l5"), ("drv_finish_l10", "team_finish_l5"),
                          ("drv_grid_l5", "team_grid_l5"), ("drv_pace_l5", "team_pace_l3")):
            df[drv] = df[drv].fillna(df[team])

    # ---- season context
    df["pts_before"] = df.groupby(["year", "driver"])["points"].transform(
        lambda s: s.fillna(0).cumsum().shift(1, fill_value=0))
    df["champ_pos"] = df.groupby("race_id")["pts_before"].rank(ascending=False, method="min")

    # ---- where does each driver stand relative to THIS field (absolute values drift between eras)
    df["elo_sum"] = df["drv_elo"] + df["team_elo"]
    for c in RANK_SOURCES:
        df[c + "_rk"] = df.groupby("race_id")[c].rank(pct=True, ascending=c not in RANK_HIGH_GOOD)

    return df
