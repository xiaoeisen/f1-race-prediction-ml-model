# Load historical F1 results from f1db (CC BY 4.0) for fast season imports.
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"
CACHE = Path("cache") / "f1db"
UA = "f1-predictor/1.0 (personal research project)"
MAX_AGE = 6 * 3600  # re-download after 6 hours (it changes after each race)

_zip = None


def _archive(refresh=False):
    global _zip
    f = CACHE / "f1db-csv.zip"
    if refresh or not f.exists() or time.time() - f.stat().st_mtime > MAX_AGE:
        CACHE.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            zipfile.ZipFile(io.BytesIO(data)).testzip()  # sanity check before replacing the cache
            f.write_bytes(data)
            _zip = None
        except Exception:
            if not f.exists():
                raise
    if _zip is None:
        _zip = zipfile.ZipFile(f)
    return _zip


def _read(name):
    return pd.read_csv(_archive().open(f"f1db-{name}.csv"), low_memory=False)


def _gap(df, ms_col, by):
    """% slower than the quickest time in each group."""
    best = df.groupby(by)[ms_col].transform("min")
    return (df[ms_col] / best - 1) * 100


def frame(start=2014, end=None):
    """Finished races from `start` as a DataFrame in the races.csv schema."""
    races = _read("races")
    races = races[races["year"] >= start]
    if end:
        races = races[races["year"] <= end]
    circuits = _read("circuits").set_index("id")
    gps = _read("grands-prix").set_index("id")
    drivers = _read("drivers").set_index("id")
    cons = _read("constructors").set_index("id")

    res = _read("races-race-results")
    res = res[res["raceId"].isin(races["id"])].copy()
    if res.empty:
        return pd.DataFrame()
    meta = races.set_index("id")
    k = ["raceId"]

    out = pd.DataFrame({
        "year": res["year"], "round": res["round"], "raceId": res["raceId"],
        "driver": res["driverId"].map(drivers["abbreviation"]),
        "full_name": res["driverId"].map(drivers["fullName"]),
        "team": res["constructorId"].map(cons["name"]),
        "grid": res["gridPositionNumber"],
        "finish": res["positionDisplayOrder"].astype(float),
        "points": res["points"].fillna(0.0),
        "dnf": res["positionNumber"].isna().astype(float),   # not classified = retired/DSQ/DNS
        "status": np.where(res["positionNumber"].isna(), res["reasonRetired"].fillna("Retired"), "Finished"),
    })
    out["driver"] = out["driver"].fillna(res["driverId"].str.split("-").str[-1].str[:3].str.upper())
    out["grid"] = out["grid"].fillna(20.0)                    # pit-lane start / no grid slot -> last
    out["event"] = out["raceId"].map(meta["grandPrixId"]).map(gps["name"])
    out["circuit"] = out["raceId"].map(meta["circuitId"]).map(circuits["placeName"])
    out["date"] = out["raceId"].map(meta["date"])
    out["is_sprint"] = out["raceId"].map(meta["sprintRaceDate"].notna().astype(float))

    key = out[["raceId", "driver"]].copy()
    drv_of = dict(zip(zip(res["raceId"], res["driverId"]), out["driver"]))

    def by_driver(df):
        df = df.copy()
        df["driver"] = [drv_of.get((r, d)) for r, d in zip(df["raceId"], df["driverId"])]
        return df.dropna(subset=["driver"])

    # qualifying: best of Q1/Q2/Q3, gap to pole, position
    q = by_driver(_read("races-qualifying-results"))
    q = q[q["raceId"].isin(races["id"])].copy()
    q["best"] = q[["q1Millis", "q2Millis", "q3Millis"]].min(axis=1)
    q["q_best_gap"] = _gap(q, "best", "raceId")
    q = q.rename(columns={"positionNumber": "quali_pos"})[["raceId", "driver", "quali_pos", "q_best_gap"]]
    out = out.merge(q.drop_duplicates(["raceId", "driver"]), on=["raceId", "driver"], how="left")

    # free practice: gap to the quickest lap of the session
    for n in (1, 2, 3):
        try:
            p = by_driver(_read(f"races-free-practice-{n}-results"))
        except KeyError:
            continue
        p = p[p["raceId"].isin(races["id"]) & p["timeMillis"].notna()].copy()
        p[f"fp{n}_best_gap"] = _gap(p, "timeMillis", "raceId")
        out = out.merge(p.drop_duplicates(["raceId", "driver"])[["raceId", "driver", f"fp{n}_best_gap"]],
                        on=["raceId", "driver"], how="left")

    # pit stops: count and mean pit-stop time
    try:
        ps = by_driver(_read("races-pit-stops"))
        ps = ps[ps["raceId"].isin(races["id"])]
        agg = ps.groupby(["raceId", "driver"]).agg(n_stops=("stop", "max"), pit_lane_s=("timeMillis", "mean")).reset_index()
        agg["pit_lane_s"] /= 1000.0
        out = out.merge(agg, on=["raceId", "driver"], how="left")
        has = out.groupby("raceId")["n_stops"].transform(lambda s: s.notna().any())
        out.loc[has & out["n_stops"].isna() & (out["dnf"] == 0), "n_stops"] = 0.0   # finished without stopping
    except KeyError:
        pass

    out = out[out["finish"].notna()].drop(columns="raceId")
    return out.sort_values(["year", "round", "finish"]).reset_index(drop=True)
