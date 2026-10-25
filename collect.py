"""Download F1 weekend data with FastF1 -> one row per driver per race (data/races.csv).

Usage:
    python collect.py --start 2018 --end 2025
Re-running only fetches races that are not already in the CSV.
"""
import argparse
import logging
import ssl
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sources.tls import configure_tls

configure_tls()  # must run before FastF1/Requests are imported
import fastf1

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

DATA = Path("data")
DATA.mkdir(exist_ok=True)
Path("cache").mkdir(exist_ok=True)
fastf1.Cache.enable_cache("cache")
OUT = DATA / "races.csv"

# columns added by the richer feature set (older CSVs won't have them -> run --refresh)
NEW_COLS = ["q_rain", "q_air_temp", "q_track_temp", "q_humidity", "q_wind", "race_pace_gap", "race_deg_per10", "lap1_gain", "race_sc", "race_vsc", "race_red", "sc_frac",
            "q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap",
            "fp1_deg_per10", "fp2_deg_per10", "fp3_deg_per10",
            "fp1_best_time_s", "fp2_best_time_s", "fp3_best_time_s",
            "fp1_gap_s", "fp2_gap_s", "fp3_gap_s", "fp1_laps", "fp2_laps", "fp3_laps",
            "q_best_time_s", "q_gap_s", "q_time_segment_valid", "quali_laps",
            "race_best_lap_s", "race_avg_clean_lap_s", "race_laps",
            "n_stops", "pit_lane_s", "stop_s", "overtakes_made", "overtakes_suffered", "n_sc", "n_vsc"]
EXTRA = True  # extra sources (Jolpica, OpenF1); switch off with --no-extra
RATE_LIMITED = False
SPRINT_ROUNDS = {
    2021: {10, 14, 19},
    2022: {4, 11, 21},
    2023: {4, 9, 12, 17, 18, 20},
    2024: {5, 6, 11, 19, 21},
    2025: {2, 6, 13, 19, 21, 23},
}
SPRINT_EVENT_NAMES = {
    2026: ("chinese grand prix", "miami grand prix", "canadian grand prix",
           "british grand prix", "great britain", "dutch grand prix", "netherlands",
           "singapore grand prix"),
}


def load(year, rnd, code, laps=True, weather=False):
    """Load a session, or return None if it doesn't exist, hasn't happened yet, or the
    download failed (FastF1 logs failures instead of raising, so we verify the data)."""
    global RATE_LIMITED
    if RATE_LIMITED:
        return None
    try:
        s = fastf1.get_session(year, rnd, code)
        s.load(laps=laps, telemetry=False, weather=weather, messages=False)
        if laps:
            if s.laps is None or s.laps.empty:
                return None
        elif s.results is None or s.results.empty:
            return None
        return s
    except Exception as e:
        if type(e).__name__ == "RateLimitExceededError" or "calls/h" in str(e):
            RATE_LIMITED = True
            logging.getLogger("collect").warning("FastF1 rate limit reached: %s", e)
            return None
        if isinstance(e, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(e):
            logging.getLogger("collect").warning(
                "TLS verification failed for %s %s R%02d; configure F1_CA_BUNDLE with the trusted PEM bundle: %s",
                code, year, rnd, e)
        return None


def load_best(year, rnd, code, weather=True):
    """Prefer a load that includes lap data; fall back to results-only."""
    return load(year, rnd, code, laps=True, weather=weather) or load(year, rnd, code, laps=False, weather=weather)


def has_laps(s):
    try:
        return s is not None and s.laps is not None and not s.laps.empty
    except Exception:
        return False


def adjusted_pace(df, numeric, min_laps=8):
    """Driver pace gap (%) after controlling for tyre compound and the numeric columns
    (tyre age, lap number = fuel load). Fixed-effects regression on log lap time.
    If identical strategies make the full model ambiguous, it falls back to fewer controls.
    Returns a Series (driver -> % slower than the fastest) or None if it can't be fitted."""
    try:
        d = df.dropna(subset=["t", "Driver"] + numeric).copy()
        counts = d["Driver"].value_counts()
        d = d[d["Driver"].isin(counts[counts >= min_laps].index)]
        if d["Driver"].nunique() < 5:
            return None
        d["Compound"] = d["Compound"].fillna("UNKNOWN").astype(str)
        drivers = pd.get_dummies(d["Driver"]).astype(float)
        comp = pd.get_dummies(d["Compound"], drop_first=True).astype(float)
        # a compound used by < 3 drivers can't be told apart from driver pace -> leave it out
        keep = [c for c in comp if (comp[c].groupby(d["Driver"]).max() > 0).sum() >= 3]
        y = np.log(d["t"].to_numpy(float))

        def fit(nums):
            X = pd.concat([drivers, comp[keep], d[nums].astype(float)], axis=1).to_numpy(float)
            if np.linalg.matrix_rank(X) < X.shape[1]:
                return None  # ambiguous design (e.g. everyone pitted on the same lap)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(((X @ coef - y) ** 2).sum())
            eff = pd.Series(coef[: drivers.shape[1]], index=drivers.columns)
            gaps = (np.exp(eff - eff.min()) - 1) * 100
            return (rss, gaps) if gaps.max() < 20 else None  # >20% spread = garbage, reject

        res = fit(numeric)
        if res is None and len(numeric) > 1:
            singles = [r for r in (fit([c]) for c in numeric) if r is not None]
            res = min(singles, key=lambda r: r[0]) if singles else None
        if res is None:
            res = fit([])
        return None if res is None else res[1]
    except Exception:
        return None


def practice_features(s, prefix):
    """Best lap, tyre-adjusted long-run pace and degradation from clean practice laps."""
    cols = [f"{prefix}_best_gap", f"{prefix}_long_gap", f"{prefix}_deg_per10"]
    if s is None:
        return pd.DataFrame(columns=cols)
    laps = s.laps
    if laps is None or laps.empty:
        return pd.DataFrame(columns=cols)
    laps = _clean_timing_laps(laps, min_lap=1)
    if laps.empty:
        return pd.DataFrame(columns=cols)
    laps["t"] = laps["LapTime"].dt.total_seconds()

    best = laps.groupby("Driver")["t"].min()
    res = pd.DataFrame({cols[0]: (best / best.min() - 1) * 100})
    fastest = best.min()
    res[f"{prefix}_best_time_s"] = best
    res[f"{prefix}_gap_s"] = best - fastest
    res[f"{prefix}_laps"] = laps.groupby("Driver")["LapNumber"].count()
    res[cols[2]] = _degradation_per_driver(laps)

    # Long runs: stints with >= 5 clean laps. Compare with each driver's own best;
    # a single session-wide cutoff would discard slower cars' otherwise clean runs.
    clean = laps.copy()
    for c in ("PitInTime", "PitOutTime"):
        if c in clean:
            clean = clean[clean[c].isna()]
    clean = clean[clean["t"] <= clean["Driver"].map(best) * 1.18]
    clean = _trim_stint_outliers(clean)
    stint = clean.groupby(["Driver", "Stint"])["t"].agg(["median", "count"])
    stint = stint[stint["count"] >= 5]
    res[cols[1]] = np.nan
    if not stint.empty:
        # tyre-aware: regress out compound + tyre age; fall back to raw stint medians
        in_runs = pd.MultiIndex.from_frame(clean[["Driver", "Stint"]]).isin(stint.index)
        adj = adjusted_pace(clean[in_runs], ["TyreLife"], min_laps=5)
        if adj is None:
            long = stint.groupby("Driver")["median"].min()
            adj = (long / long.min() - 1) * 100
        res.loc[adj.index.intersection(res.index), cols[1]] = adj
    return res


def quali_features(s):
    cols = ["quali_pos", "q_best_gap"]
    if s is None or s.results is None or s.results.empty:
        return pd.DataFrame(columns=cols)
    r = s.results.copy()
    # Classification time is from the last segment reached (Q3, otherwise Q2,
    # otherwise Q1), not the driver's fastest lap across all three segments.
    r["best"] = (r["Q3"].combine_first(r["Q2"]).combine_first(r["Q1"]).dt.total_seconds())
    r["q_best_gap"] = (r["best"] / r["best"].min() - 1) * 100
    r["quali_pos"] = r["Position"]
    r["q_best_time_s"] = r["best"]
    r["q_gap_s"] = r["best"] - r["best"].min()
    r["q_time_segment_valid"] = r["best"].notna().astype(float)
    out = r.set_index("Abbreviation")[cols + ["q_best_time_s", "q_gap_s"]]
    if has_laps(s):
        counts = s.laps.groupby("Driver")["LapNumber"].count()
        out["quali_laps"] = counts.reindex(out.index)
    else:
        out["quali_laps"] = np.nan
    return out


def quali_profile(s):
    """Sector-time and top-speed gaps (%) from qualifying laps: where is the car strong?"""
    cols = ["q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap"]
    if not has_laps(s):
        return pd.DataFrame(columns=cols)
    laps = _clean_timing_laps(s.laps, min_lap=1)
    if laps.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(index=pd.Index(laps["Driver"].unique(), name="Driver"), columns=cols, dtype=float)
    for i, c in enumerate(("Sector1Time", "Sector2Time", "Sector3Time"), 1):
        if c in laps:
            best = laps.groupby("Driver")[c].min().dt.total_seconds()
            out[f"q_s{i}_gap"] = (best / best.min() - 1) * 100
    if "SpeedST" in laps and laps["SpeedST"].notna().any():
        top = laps.groupby("Driver")["SpeedST"].max()
        out["q_topspeed_gap"] = (1 - top / top.max()) * 100
    return out


def race_lap_features(s, grid):
    """From the race laps: tyre/fuel-adjusted clean-air race pace, lap-1 places gained,
    and race-level safety car / VSC / red flag flags."""
    laps = s.laps.copy()
    laps["t"] = laps["LapTime"].dt.total_seconds()
    ts = laps["TrackStatus"].fillna("").astype(str) if "TrackStatus" in laps else pd.Series("", index=laps.index)
    sc = ts.str.contains("4")
    vsc = ts.str.contains("6|7")
    red = bool(ts.str.contains("5").any())
    try:
        red = red or bool((s.track_status["Status"].astype(str) == "5").any())
    except Exception:
        pass
    race = dict(race_sc=float(sc.any()), race_vsc=float(vsc.any()), race_red=float(red),
                sc_frac=float((sc | vsc).mean()))

    res = pd.DataFrame(index=pd.Index(laps["Driver"].unique(), name="Driver"),
                       columns=["race_pace_gap", "race_deg_per10", "lap1_gain"], dtype=float)
    if "Position" in laps:
        l1 = laps[laps["LapNumber"] == 1].drop_duplicates("Driver").set_index("Driver")["Position"]
        res["lap1_gain"] = grid.reindex(res.index) - l1.reindex(res.index)

    clean = _trim_stint_outliers(_clean_timing_laps(laps, min_lap=2))
    if "t" not in clean:
        clean["t"] = clean["LapTime"].dt.total_seconds()
    clean = clean[clean["t"] < clean.groupby("Driver")["t"].transform("median") * 1.08]
    if not clean.empty:
        res["race_best_lap_s"] = clean.groupby("Driver")["t"].min().reindex(res.index)
        res["race_avg_clean_lap_s"] = clean.groupby("Driver")["t"].mean().reindex(res.index)
    res["race_laps"] = laps.groupby("Driver")["LapNumber"].nunique().reindex(res.index)
    adj = adjusted_pace(clean, ["LapNumber", "TyreLife"])
    if adj is None:  # fall back to plain medians
        m = clean.groupby("Driver")["t"].median()
        adj = (m / m.min() - 1) * 100
    res["race_pace_gap"] = adj.reindex(res.index)
    res["race_deg_per10"] = _degradation_per_driver(_clean_timing_laps(laps, min_lap=2))
    return res, race


def _clean_timing_laps(laps, min_lap=1):
    """Keep timed green laps and remove pit entry/exit laps before pace fitting."""
    if laps is None or laps.empty or "LapTime" not in laps:
        return pd.DataFrame() if laps is None else laps.iloc[0:0].copy()
    d = laps.copy()
    d["t"] = d["LapTime"].dt.total_seconds()
    d = d[d["t"].notna() & np.isfinite(d["t"]) & (d["t"] > 0)]
    if "TrackStatus" in d:
        # FastF1 status 1 is green; all yellow, SC, VSC and red-flag statuses are excluded.
        d = d[d["TrackStatus"].fillna("").astype(str).str.strip().eq("1")]
    for col in ("PitInTime", "PitOutTime"):
        if col in d:
            d = d[d[col].isna()]
    if "LapNumber" in d:
        d = d[pd.to_numeric(d["LapNumber"], errors="coerce") >= min_lap]
    return d.copy()


def _trim_stint_outliers(laps, floor_fraction=0.03):
    """Drop large within-stint timing outliers while retaining normal tyre-age drift."""
    if laps is None or laps.empty or not {"Driver", "Stint", "t"}.issubset(laps.columns):
        return laps
    groups = laps.groupby(["Driver", "Stint"])["t"]
    center = groups.transform("median")
    mad = groups.transform(lambda x: (x - x.median()).abs().median())
    limit = np.maximum(4.0 * 1.4826 * mad, center * floor_fraction)
    return laps[(laps["t"] - center).abs() <= limit].copy()


def _degradation_per_driver(laps, min_stint_laps=5):
    """Estimate lap-time loss per ten tyre-age laps, controlling for driver, compound and fuel burn.

    Only drivers with at least two usable stints are scored: within one stint, tyre age and
    race/session lap number are nearly identical, so tyre wear cannot be separated from fuel.
    A shared lap-number term controls fuel burn; driver-specific tyre-age slopes capture wear.
    """
    empty = pd.Series(dtype=float)
    required = {"Driver", "Stint", "Compound", "TyreLife", "LapNumber", "t"}
    if laps is None or laps.empty or not required.issubset(laps.columns):
        return empty
    d = laps.dropna(subset=["Driver", "Stint", "TyreLife", "LapNumber", "t"]).copy()
    for col in ("TyreLife", "LapNumber", "t"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["TyreLife", "LapNumber", "t"])
    d = d[(d["TyreLife"] >= 0) & (d["t"] > 0)]
    if d.empty:
        return empty

    counts = d.groupby(["Driver", "Stint"])["t"].transform("size")
    d = d[counts >= min_stint_laps].copy()
    stint_counts = d.groupby("Driver")["Stint"].nunique()
    eligible = stint_counts[stint_counts >= 2].index
    d = d[d["Driver"].isin(eligible)].copy()
    if d["Driver"].nunique() < 3 or len(d) < 30:
        return empty

    d["Compound"] = d["Compound"].fillna("UNKNOWN").astype(str)
    drivers = pd.get_dummies(d["Driver"], dtype=float)
    compounds = pd.get_dummies(d["Compound"], drop_first=True, dtype=float)
    age = d["TyreLife"] - d.groupby("Driver")["TyreLife"].transform("mean")
    age_terms = pd.DataFrame({driver: age.where(d["Driver"].eq(driver), 0.0)
                              for driver in drivers.columns}, index=d.index)
    lap = (d["LapNumber"] - d["LapNumber"].mean()).rename("lap_number")
    X = pd.concat([drivers, compounds, lap, age_terms], axis=1).to_numpy(float)
    y = np.log(d["t"].to_numpy(float))

    def fit(mask):
        xx, yy = X[mask], y[mask]
        if xx.shape[0] <= xx.shape[1] or np.linalg.matrix_rank(xx) < xx.shape[1]:
            return None
        coef, *_ = np.linalg.lstsq(xx, yy, rcond=None)
        return coef, yy - xx @ coef

    fit_all = fit(np.ones(len(d), dtype=bool))
    if fit_all is None:
        return empty
    _, residual = fit_all
    d["residual"] = residual
    groups = d.groupby(["Driver", "Stint"])["residual"]
    center = groups.transform("median")
    spread = groups.transform(lambda x: (x - x.median()).abs().median())
    keep = (d["residual"] - center).abs() <= np.maximum(4.0 * 1.4826 * spread, np.log(1.03))
    mask = keep.to_numpy()
    result = fit(mask)
    if result is None:
        return empty
    coef, _ = result
    slopes = pd.Series(coef[-len(drivers.columns):], index=drivers.columns)
    return ((np.exp(slopes * 10.0) - 1.0) * 100.0).clip(-15.0, 30.0)


def weather_features(*sessions):
    for s in sessions:
        if s is None:
            continue
        try:
            w = s.weather_data
        except Exception:
            continue
        if w is not None and len(w):
            return dict(
                air_temp=w["AirTemp"].mean(),
                track_temp=w["TrackTemp"].mean(),
                humidity=w["Humidity"].mean(),
                wind=w["WindSpeed"].mean(),
                rain=float(w["Rainfall"].any()),
            )
    return dict(air_temp=np.nan, track_temp=np.nan, humidity=np.nan, wind=np.nan, rain=np.nan)


def is_dnf(status):
    st = str(status)
    return not (st == "Finished" or st.startswith("+") or st == "Lapped")


def collect_weekend(year, rnd, ev):
    """Return a DataFrame (one row per driver) for one race weekend.

    Works for finished races (has target columns) and for an upcoming race
    (finish/dnf/points are NaN; grid falls back to qualifying position).
    """
    race = load_best(year, rnd, "R")
    quali = load_best(year, rnd, "Q")
    fps = {p: load(year, rnd, p) for p in ("FP1", "FP2", "FP3")}
    open_rows = None
    q_weather_missing = not np.isfinite(weather_features(quali).get("rain", np.nan))
    sprint = str(ev.get("EventFormat", "")).lower().startswith("sprint")
    expected_practice = ("FP1",) if sprint else ("FP1", "FP2", "FP3")
    if EXTRA and year >= 2023 and (quali is None or any(fps[p] is None for p in expected_practice)
                                   or q_weather_missing):
        try:
            from sources import openf1
            open_rows = openf1.weekend_features(ev["EventDate"])
        except Exception:
            open_rows = None

    # --- base table: who is racing
    base = None
    if race is not None and race.results is not None and not race.results.empty:
        r = race.results
        base = pd.DataFrame({
            "driver": r["Abbreviation"].values,
            "full_name": r["FullName"].values,
            "team": r["TeamName"].values,
            "grid": r["GridPosition"].values,
            "finish": r["Position"].values,
            "status": r["Status"].values,
            "points": r["Points"].values,
        })
    elif quali is not None and quali.results is not None and not quali.results.empty:
        r = quali.results
        base = pd.DataFrame({"driver": r["Abbreviation"].values, "full_name": r["FullName"].values,
                             "team": r["TeamName"].values})
    else:
        for p in ("FP2", "FP1", "FP3"):
            if fps[p] is not None and not fps[p].laps.empty:
                u = fps[p].laps[["Driver", "Team"]].drop_duplicates("Driver")
                base = pd.DataFrame({"driver": u["Driver"].values, "full_name": u["Driver"].values,
                                     "team": u["Team"].values})
                break
    if base is None and open_rows is not None and not open_rows.empty:
        base = open_rows[["driver", "full_name", "team"]].drop_duplicates("driver").copy()
    if base is None and EXTRA:  # FastF1 / OpenF1 had nothing: fall back to Jolpica, then f1db
        try:
            from sources import jolpica
            rows = jolpica.weekend_rows(year, rnd, ev)
            if len(rows):
                return rows
        except Exception:
            pass
        try:
            from sources import f1db
            rows = f1db.frame(year, year)
            rows = rows[rows["round"] == rnd].copy()
            if len(rows):
                rows["src"] = "f1db"
                return rows
        except Exception:
            pass
    if base is None:
        return pd.DataFrame()
    for c in ("grid", "finish", "status", "points"):
        if c not in base:
            base[c] = np.nan
    base = base.set_index("driver")

    # --- qualifying + practice features
    base = base.join(quali_features(quali))
    for name, s in fps.items():
        base = base.join(practice_features(s, name.lower()))
    if open_rows is not None and not open_rows.empty:
        live = open_rows.drop_duplicates("driver").set_index("driver")
        for c in live.columns:
            if c in base:
                base[c] = base[c].combine_first(live[c])
            else:
                base[c] = live[c]
    for c in ("fp1_best_gap", "fp1_long_gap", "fp2_best_gap", "fp2_long_gap", "fp3_best_gap"):
        if c not in base:
            base[c] = np.nan

    # --- grid (0 = pit lane start -> treat as last); fall back to quali position
    base["grid"] = base["grid"].replace(0, 20)
    base["grid"] = base["grid"].fillna(base.get("quali_pos"))

    # --- richer features (never allowed to break the basic data if they fail)
    try:
        base = base.join(quali_profile(quali))
    except Exception:
        pass
    try:
        if has_laps(race):
            rl, rd = race_lap_features(race, base["grid"])
            base = base.join(rl)
            for k, v in rd.items():
                base[k] = v
    except Exception:
        pass
    if EXTRA and ("race_pace_gap" not in base or not base["race_pace_gap"].notna().any()):
        try:
            from sources import openf1
            race_pace = openf1.race_pace_features(ev["EventDate"])
            if race_pace is not None:
                base = base.join(race_pace, rsuffix="_openf1")
        except Exception:
            pass
    # --- extra sources: pit stops, strategy, overtakes, safety cars (finished races only)
    if EXTRA and base["finish"].notna().any():
        try:
            from sources import enrich
            ex, rl = enrich.race_extras(year, rnd, ev["EventDate"])
            if ex is not None:
                base = base.join(ex)
            if rl:
                base["n_sc"], base["n_vsc"] = rl["n_sc"], rl["n_vsc"]
                for col, val in (("race_sc", rl["n_sc"] > 0), ("race_vsc", rl["n_vsc"] > 0),
                                 ("race_red", bool(rl["red"]))):
                    # either source seeing a safety car / red flag counts (FastF1 lap flags can miss one)
                    base[col] = float(val) if col not in base else np.fmax(base[col], float(val))
        except Exception:
            pass
    for c in NEW_COLS:
        if c not in base:
            base[c] = np.nan

    base["dnf"] = [float(is_dnf(s)) if pd.notna(s) else np.nan for s in base["status"]]
    for k, v in weather_features(race, quali).items():
        base[k] = v                        # observed during the race (known only after the start)
    for k, v in weather_features(quali).items():
        col = "q_" + k
        # Keep an OpenF1 qualifying-session observation if FastF1 has no weather sample.
        base[col] = base[col].fillna(v) if col in base else v

    base = base.reset_index()
    base["year"] = year
    base["round"] = rnd
    base["event"] = ev["EventName"]
    base["circuit"] = ev["Location"]
    base["date"] = pd.Timestamp(ev["EventDate"]).date().isoformat()
    base["is_sprint"] = float(str(ev.get("EventFormat", "")).startswith("sprint"))
    base["src"] = "fastf1+openf1" if open_rows is not None and not open_rows.empty else "fastf1"
    return base


def _source_parts(value):
    return {part.strip() for part in str(value or "").split("+") if part.strip() and part != "nan"}


def merge_weekend(existing, fresh):
    """Merge a newly fetched race with stored rows, keeping non-missing values from both."""
    if existing is None or existing.empty:
        return fresh.copy()
    if fresh is None or fresh.empty:
        return existing.copy()
    authoritative = (("finish" in fresh and fresh["finish"].notna().any()) or
                     ("quali_pos" in fresh and fresh["quali_pos"].notna().any()))
    old = {str(r["driver"]): r.to_dict() for _, r in existing.iterrows()}
    rows = []
    drivers = list(dict.fromkeys(fresh["driver"].astype(str).tolist()))
    if not authoritative:
        drivers += [d for d in old if d not in drivers]
    for driver in drivers:
        previous = old.get(driver, {})
        incoming = fresh[fresh["driver"].astype(str) == driver]
        row = previous.copy()
        if len(incoming):
            new = incoming.iloc[-1].to_dict()
            for key, value in new.items():
                if pd.notna(value):
                    row[key] = value
        if previous.get("src") and len(incoming):
            parts = _source_parts(previous["src"]) | _source_parts(row.get("src"))
            row["src"] = "+".join(sorted(parts))
        rows.append(row)
    return pd.DataFrame(rows)


def _contains_source(value, source):
    return source in _source_parts(value)


def missing_weekend_data(g):
    """Return missing session/pace signals that justify a retry for this weekend."""
    required = ["quali_pos", "fp1_best_gap", "q_best_time_s", "q_time_segment_valid", "fp3_best_time_s"]
    if not _is_sprint_weekend(g):
        required.extend(("fp2_best_gap", "fp3_best_gap"))
    missing = [c for c in required if c not in g or not g[c].notna().any()]
    # Old f1db imports carry basic times but have no lap-level pace. Re-fetch those
    # weekends so the same retry command also backfills the newer pace signals.
    if not any(c in g and g[c].notna().any() for c in ("fp1_long_gap", "fp2_long_gap")):
        missing.append("practice_long_run")
    if "race_pace_gap" not in g or not g["race_pace_gap"].notna().any():
        missing.append("race_pace")
    if "q_s1_gap" not in g or not g["q_s1_gap"].notna().any():
        missing.append("quali_sectors")
    return missing


def _is_sprint_weekend(g):
    if bool(g.get("is_sprint", pd.Series(0, index=g.index)).fillna(0).max()):
        return True
    if "year" not in g or "round" not in g or g.empty:
        return False
    year = int(g["year"].iloc[0])
    if year in SPRINT_ROUNDS and int(g["round"].iloc[0]) in SPRINT_ROUNDS[year]:
        return True
    event = str(g.get("event", pd.Series("", index=g.index)).iloc[0]).lower()
    return any(name in event for name in SPRINT_EVENT_NAMES.get(year, ()))


def audit_data(df):
    columns = ["quali_pos", "fp1_best_gap", "fp2_best_gap", "fp3_best_gap",
               "fp1_long_gap", "fp2_long_gap", "fp1_deg_per10", "fp2_deg_per10",
               "fp3_deg_per10", "race_pace_gap", "race_deg_per10", "q_s1_gap",
               "q_topspeed_gap", "q_rain", "q_air_temp"]
    columns = [c for c in columns if c in df]
    print(f"Data coverage: {df.groupby(['year', 'round']).ngroups} races, {len(df)} driver rows")
    print(df[columns].notna().mean().round(3).to_string())
    missing = []
    for (year, rnd), g in df.groupby(["year", "round"]):
        required = ["quali_pos", "fp1_best_gap"]
        if not _is_sprint_weekend(g):
            required.extend(("fp2_best_gap", "fp3_best_gap"))
        absent = [c for c in required if c not in g or not g[c].notna().any()]
        if absent:
            missing.append((int(year), int(rnd), str(g["event"].iloc[0]), ",".join(absent)))
    print(f"Races missing core session data: {len(missing)}")
    for year, rnd, event, absent in missing[:40]:
        print(f"  {year} R{rnd:02d} {event}: {absent}")
    if len(missing) > 40:
        print(f"  ... and {len(missing) - 40} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2022, help="first season (FastF1 goes back to 2018)")
    ap.add_argument("--end", type=int, default=pd.Timestamp.now().year)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download every race (needed once after the feature set changes)")
    ap.add_argument("--f1db", type=int, default=None, metavar="YEAR",
                    help="FAST bulk load of every race from YEAR (one ~4.5 MB download, seconds): results, qualifying, "
                         "practice gaps, pit stops. No tyre/lap/weather data; FastF1 rows replace these later")
    ap.add_argument("--no-extra", action="store_true",
                    help="skip the extra sources (Jolpica / OpenF1): FastF1 only")
    ap.add_argument("--backfill-from", type=int, default=None, metavar="YEAR",
                    help="also load older seasons (YEAR .. start-1) from Jolpica: results, qualifying, pit stops")
    ap.add_argument("--check", action="store_true",
                    help="download ONE recent race and print it, to verify real data works")
    ap.add_argument("--audit", action="store_true", help="report session-data coverage and missing races")
    ap.add_argument("--retry-incomplete", action="store_true",
                    help="retry FastF1 races missing session timing or lap-level pace signals")
    ap.add_argument("--only-race", action="append", default=[], metavar="YEAR:ROUND",
                    help="process only this race; repeat to select multiple races")
    args = ap.parse_args()
    global EXTRA, RATE_LIMITED
    EXTRA = not args.no_extra
    RATE_LIMITED = False

    try:
        only_races = {(int(item.split(":", 1)[0]), int(item.split(":", 1)[1])) for item in args.only_race}
    except (ValueError, IndexError):
        ap.error("--only-race values must use YEAR:ROUND, for example 2026:18")

    if args.check:
        year = args.end
        sched = fastf1.get_event_schedule(year, include_testing=False)
        done = sched[pd.to_datetime(sched["EventDate"]) < pd.Timestamp.now().normalize()]
        if done.empty:
            year -= 1
            done = fastf1.get_event_schedule(year, include_testing=False)
        ev = done.iloc[-1]
        print(f"Checking {year} {ev['EventName']} ...")
        w = collect_weekend(year, int(ev["RoundNumber"]), ev)
        if w.empty:
            raise SystemExit("No data came back. Check your internet connection / try again.")
        cols = ["driver", "team", "grid", "quali_pos", "q_best_gap", "q_topspeed_gap", "fp1_best_gap",
                "fp2_long_gap", "fp3_best_gap", "race_pace_gap", "lap1_gain", "finish", "dnf", "rain"]
        print(w[cols].round(2).to_string(index=False))
        return

    df = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(df) and "src" not in df:
        df["src"] = "fastf1"           # older files only ever held FastF1 rows
    if args.audit:
        audit_data(df)
        return
    if args.refresh:
        df = pd.DataFrame()
    elif len(df) and "race_pace_gap" not in df:
        print("NOTE: your data was collected before the newer features existed "
              "(race pace, safety cars, sectors). Run: python collect.py --refresh\n")
    if args.f1db:
        from sources import f1db
        print(f"Loading all races from {args.f1db} via f1db ...", flush=True)
        bulk = f1db.frame(args.f1db)
        bulk["src"] = "f1db"
        if len(df):                     # keep richer FastF1 rows where we already have them
            fast_rows = df["src"].map(lambda x: _contains_source(x, "fastf1"))
            fast = set(zip(df.loc[fast_rows, "year"], df.loc[fast_rows, "round"]))
            df = df[fast_rows]
            bulk = bulk[[(y, r) not in fast for y, r in zip(bulk["year"], bulk["round"])]]
        df = pd.concat([df, bulk], ignore_index=True)
        df.to_csv(OUT, index=False)
        print(f"   {bulk['round'].size} rows added ({bulk.groupby(['year', 'round']).ngroups} races). "
              f"Run `python collect.py --start 2022` to upgrade recent races with FastF1 lap data.")
    # a race counts as 'collected' for FastF1 only if its rows came from FastF1
    fast_rows = df["src"].map(lambda x: _contains_source(x, "fastf1")) if len(df) else pd.Series(dtype=bool)
    have = set(zip(df.loc[fast_rows, "year"], df.loc[fast_rows, "round"])) if len(df) else set()
    if args.retry_incomplete and len(df):
        for (year, rnd), g in df.loc[fast_rows].groupby(["year", "round"]):
            if missing_weekend_data(g):
                have.discard((year, rnd))
    today = pd.Timestamp.now().normalize()

    if args.backfill_from and EXTRA:
        from sources import jolpica
        for year in range(args.backfill_from, args.start):
            if len(df) and (df["year"] == year).any():
                continue
            print(f"{year} (history from Jolpica) ...", flush=True)
            try:
                f = jolpica.season_frame(year, pits=True)
            except Exception as e:
                print("   skipped:", e)
                continue
            if not f.empty:
                df = pd.concat([df, f], ignore_index=True)
                df.to_csv(OUT, index=False)

    years = sorted({year for year, _ in only_races}) if only_races else range(args.start, args.end + 1)
    for year in years:
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            if type(e).__name__ == "RateLimitExceededError" or "calls/h" in str(e):
                RATE_LIMITED = True
                print(f"FastF1's hourly limit was reached while loading the {year} schedule. "
                      "Progress is saved; resume the same command after the limit resets.", flush=True)
                break
            raise
        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            if only_races and (year, rnd) not in only_races:
                continue
            if (year, rnd) in have or pd.Timestamp(ev["EventDate"]) > today:
                continue
            print(f"{year} R{rnd:02d} {ev['EventName']} ...", flush=True)
            w = collect_weekend(year, rnd, ev)
            has_result = not w.empty and "finish" in w and w["finish"].notna().any()
            session_cols = ("quali_pos", "fp1_best_gap", "fp2_best_gap", "fp3_best_gap",
                            "fp1_long_gap", "fp2_long_gap", "q_s1_gap", "race_pace_gap")
            has_session_data = not w.empty and any(c in w and w[c].notna().any() for c in session_cols)
            if w.empty or (not has_result and not has_session_data):
                print("   skipped (no published session data; existing rows retained)")
                continue
            previous = df[(df["year"] == year) & (df["round"] == rnd)] if len(df) else pd.DataFrame()
            merged = merge_weekend(previous, w)
            if len(df):
                df = df[~((df["year"] == year) & (df["round"] == rnd))]
            df = pd.concat([df, merged], ignore_index=True)
            df.to_csv(OUT, index=False)
            have.add((year, rnd))
            if RATE_LIMITED:
                print("   FastF1 limit reached; saved available OpenF1 timing and will continue this-season backfill", flush=True)
        if RATE_LIMITED:
            break
    print(f"Saved {len(df)} rows to {OUT}")
    if len(df):
        key = ["grid", "q_best_gap", "fp1_best_gap", "fp2_best_gap", "fp2_long_gap", "fp2_deg_per10", "rain",
               "race_pace_gap", "race_deg_per10", "lap1_gain", "race_sc", "q_s1_gap", "q_topspeed_gap", "q_rain", "finish"]
        key = [k for k in key if k in df]
        print("\nData coverage (share of rows with a value):")
        print(df[key].notna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
