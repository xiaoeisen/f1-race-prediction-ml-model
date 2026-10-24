"""Download F1 weekend data with FastF1 -> one row per driver per race (data/races.csv).

Usage:
    python collect.py --start 2018 --end 2025
Re-running only fetches races that are not already in the CSV.
"""
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

DATA = Path("data")
DATA.mkdir(exist_ok=True)
Path("cache").mkdir(exist_ok=True)
fastf1.Cache.enable_cache("cache")
OUT = DATA / "races.csv"

# columns added by the richer feature set (older CSVs won't have them -> run --refresh)
NEW_COLS = ["q_rain", "q_air_temp", "q_track_temp", "q_humidity", "q_wind", "race_pace_gap", "lap1_gain", "race_sc", "race_vsc", "race_red", "sc_frac",
            "q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap",
            "n_stops", "pit_lane_s", "stop_s", "overtakes_made", "overtakes_suffered", "n_sc", "n_vsc"]
EXTRA = True  # extra sources (Jolpica, OpenF1); switch off with --no-extra


def load(year, rnd, code, laps=True, weather=False):
    """Load a session, or return None if it doesn't exist, hasn't happened yet, or the
    download failed (FastF1 logs failures instead of raising, so we verify the data)."""
    try:
        s = fastf1.get_session(year, rnd, code)
        s.load(laps=laps, telemetry=False, weather=weather, messages=False)
        if laps:
            if s.laps is None or s.laps.empty:
                return None
        elif s.results is None or s.results.empty:
            return None
        return s
    except Exception:
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
    """Best-lap gap and long-run pace gap (both in % vs the fastest driver)."""
    cols = [f"{prefix}_best_gap", f"{prefix}_long_gap"]
    if s is None:
        return pd.DataFrame(columns=cols)
    laps = s.laps
    if laps is None or laps.empty:
        return pd.DataFrame(columns=cols)
    laps = laps[laps["LapTime"].notna()].copy()
    if laps.empty:
        return pd.DataFrame(columns=cols)
    laps["t"] = laps["LapTime"].dt.total_seconds()

    best = laps.groupby("Driver")["t"].min()
    res = pd.DataFrame({cols[0]: (best / best.min() - 1) * 100})

    # Long runs: stints with >= 5 clean laps (no in/out laps, within 110% of best)
    clean = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    clean = clean[clean["t"] <= best.min() * 1.10]
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
    r["best"] = r[["Q1", "Q2", "Q3"]].min(axis=1).dt.total_seconds()
    r["q_best_gap"] = (r["best"] / r["best"].min() - 1) * 100
    r["quali_pos"] = r["Position"]
    return r.set_index("Abbreviation")[cols]


def quali_profile(s):
    """Sector-time and top-speed gaps (%) from qualifying laps: where is the car strong?"""
    cols = ["q_s1_gap", "q_s2_gap", "q_s3_gap", "q_topspeed_gap"]
    if not has_laps(s):
        return pd.DataFrame(columns=cols)
    laps = s.laps
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
                       columns=["race_pace_gap", "lap1_gain"], dtype=float)
    if "Position" in laps:
        l1 = laps[laps["LapNumber"] == 1].drop_duplicates("Driver").set_index("Driver")["Position"]
        res["lap1_gain"] = grid.reindex(res.index) - l1.reindex(res.index)

    clean = laps[(ts == "1") & laps["t"].notna() & laps["PitInTime"].isna()
                 & laps["PitOutTime"].isna() & (laps["LapNumber"] > 1)].copy()
    clean = clean[clean["t"] < clean.groupby("Driver")["t"].transform("median") * 1.05]
    adj = adjusted_pace(clean, ["LapNumber", "TyreLife"])
    if adj is None:  # fall back to plain medians
        m = clean.groupby("Driver")["t"].median()
        adj = (m / m.min() - 1) * 100
    res["race_pace_gap"] = adj.reindex(res.index)
    return res, race


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
    if base is None and EXTRA:  # FastF1 had nothing: fall back to Jolpica, then f1db, for a finished race
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
        base["q_" + k] = v                 # qualifying session: genuinely pre-race information

    base = base.reset_index()
    base["year"] = year
    base["round"] = rnd
    base["event"] = ev["EventName"]
    base["circuit"] = ev["Location"]
    base["date"] = pd.Timestamp(ev["EventDate"]).date().isoformat()
    base["is_sprint"] = float(str(ev.get("EventFormat", "")).startswith("sprint"))
    base["src"] = "fastf1"
    return base


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
    args = ap.parse_args()
    global EXTRA
    EXTRA = not args.no_extra

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
        cols = ["driver", "team", "grid", "q_best_gap", "q_topspeed_gap", "fp1_best_gap",
                "fp2_long_gap", "race_pace_gap", "lap1_gain", "finish", "dnf", "rain"]
        print(w[cols].round(2).to_string(index=False))
        return

    df = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(df) and "src" not in df:
        df["src"] = "fastf1"           # older files only ever held FastF1 rows
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
            fast = set(zip(df.loc[df["src"] == "fastf1", "year"], df.loc[df["src"] == "fastf1", "round"]))
            df = df[df["src"] == "fastf1"]
            bulk = bulk[[(y, r) not in fast for y, r in zip(bulk["year"], bulk["round"])]]
        df = pd.concat([df, bulk], ignore_index=True)
        df.to_csv(OUT, index=False)
        print(f"   {bulk['round'].size} rows added ({bulk.groupby(['year', 'round']).ngroups} races). "
              f"Run `python collect.py --start 2022` to upgrade recent races with FastF1 lap data.")
    # a race counts as 'collected' for FastF1 only if its rows came from FastF1
    have = set(zip(df.loc[df["src"] == "fastf1", "year"], df.loc[df["src"] == "fastf1", "round"])) if len(df) else set()
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

    for year in range(args.start, args.end + 1):
        sched = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            if (year, rnd) in have or pd.Timestamp(ev["EventDate"]) >= today:
                continue
            print(f"{year} R{rnd:02d} {ev['EventName']} ...", flush=True)
            w = collect_weekend(year, rnd, ev)
            if w.empty or w["finish"].isna().all():
                print("   skipped (no results)")
                continue
            if len(df):                 # replace any bulk-loaded (f1db / Jolpica) rows of this race
                df = df[~((df["year"] == year) & (df["round"] == rnd))]
            df = pd.concat([df, w], ignore_index=True)
            df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
    if len(df):
        key = ["grid", "q_best_gap", "fp1_best_gap", "fp2_best_gap", "fp2_long_gap", "rain",
               "race_pace_gap", "lap1_gain", "race_sc", "q_s1_gap", "q_topspeed_gap", "finish"]
        key = [k for k in key if k in df]
        print("\nData coverage (share of rows with a value):")
        print(df[key].notna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
