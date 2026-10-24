# Fetch historical results and pit stops through cached, rate-limited queries.
import numpy as np
import pandas as pd

from .http import Client

PAGE = 100  # Jolpica max page size
client = Client("jolpica", "https://api.jolpi.ca/ergast/f1", per_second=3, per_hour=450)


def _secs(t):
    """'1:30.123' -> 90.123 ; '23.456' -> 23.456"""
    if t in (None, ""):
        return np.nan
    try:
        parts = str(t).split(":")
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
    except ValueError:
        return np.nan


def _pages(path, limit=None):
    limit = limit or PAGE
    out, offset = [], 0
    while True:
        mr = client.get(path, {"limit": limit, "offset": offset})["MRData"]
        out.append(mr)
        offset += limit
        if offset >= int(mr.get("total", 0)):
            return out


def _merge(pages, listkey):
    """Races can be split across pages -> merge by (season, round)."""
    merged = {}
    for mr in pages:
        for r in mr["RaceTable"]["Races"]:
            k = (int(r["season"]), int(r["round"]))
            m = merged.setdefault(k, {**{x: v for x, v in r.items() if x != listkey}, listkey: []})
            m[listkey].extend(r.get(listkey, []))
    return merged


def _is_dnf(status):
    s = str(status)
    return not (s == "Finished" or s.startswith("+") or s == "Lapped")


def _code(d):
    return d.get("code") or str(d.get("familyName", "UNK"))[:3].upper()


def _race_frame(race, qrace):
    """One race's Results (+ qualifying) -> rows in the races.csv schema."""
    q = {}
    for r in (qrace or {}).get("QualifyingResults", []):
        times = [_secs(r.get(k)) for k in ("Q1", "Q2", "Q3")]
        best = np.nanmin(times) if not np.all(np.isnan(times)) else np.nan
        q[_code(r["Driver"])] = (float(r["position"]), best)
    pole = np.nanmin([b for _, b in q.values()]) if q and not all(np.isnan(b) for _, b in q.values()) else np.nan
    rows = []
    for r in race["Results"]:
        code = _code(r["Driver"])
        grid = int(r.get("grid", 0) or 0)
        qp, qb = q.get(code, (np.nan, np.nan))
        rows.append(dict(
            year=int(race["season"]), round=int(race["round"]), event=race.get("raceName"),
            circuit=race["Circuit"]["Location"]["locality"], date=race.get("date"),
            driver=code, full_name=f"{r['Driver'].get('givenName', '')} {r['Driver'].get('familyName', '')}".strip(),
            team=r["Constructor"]["name"], grid=float(grid if grid > 0 else 20),
            quali_pos=qp, q_best_gap=(qb / pole - 1) * 100 if pd.notna(qb) and pd.notna(pole) else np.nan,
            finish=float(r["position"]), status=r.get("status"), points=float(r.get("points", 0)),
            dnf=float(_is_dnf(r.get("status"))), is_sprint=0.0, src="jolpica",
            _driver_id=r["Driver"].get("driverId")))
    return pd.DataFrame(rows)


def pitstops_frame(year, rnd, id2code):
    """-> DataFrame indexed by driver code: n_stops, pit_lane_s (pit-lane time, not stationary)."""
    if int(year) < 2011:
        return None
    mr = client.get(f"/{year}/{rnd}/pitstops.json", {"limit": 100})["MRData"]
    races = mr["RaceTable"]["Races"]
    if not races:
        return None
    stops = {}
    for p in races[0].get("PitStops", []):
        c = id2code.get(p.get("driverId"))
        if c:
            stops.setdefault(c, []).append(_secs(p.get("duration")))
    if not stops:
        return None
    return pd.DataFrame({"n_stops": {c: float(len(v)) for c, v in stops.items()},
                         "pit_lane_s": {c: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan
                                        for c, v in stops.items()}}).rename_axis("driver")


def weekend_rows(year, rnd, ev=None):
    """One finished race from Jolpica (fallback when FastF1 has nothing)."""
    res = _merge([client.get(f"/{year}/{rnd}/results.json", {"limit": 100})["MRData"]], "Results")
    if not res:
        return pd.DataFrame()
    qual = _merge([client.get(f"/{year}/{rnd}/qualifying.json", {"limit": 100})["MRData"]], "QualifyingResults")
    k = next(iter(res))
    return _race_frame(res[k], qual.get(k)).drop(columns="_driver_id")


def season_frame(year, pits=True):
    """A whole season (results + qualifying [+ pit stops]) in the races.csv schema."""
    res = _merge(_pages(f"/{year}/results.json"), "Results")
    qual = _merge(_pages(f"/{year}/qualifying.json"), "QualifyingResults")
    frames = []
    for k, race in sorted(res.items()):
        f = _race_frame(race, qual.get(k))
        if f.empty:
            continue
        if pits:
            try:
                ps = pitstops_frame(k[0], k[1], dict(zip(f["_driver_id"], f["driver"])))
                if ps is not None:
                    f = f.merge(ps, left_on="driver", right_index=True, how="left")
            except Exception:
                pass
        frames.append(f.drop(columns="_driver_id"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
