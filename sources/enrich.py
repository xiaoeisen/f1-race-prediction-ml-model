# Combine optional sources and fall back gracefully when one fails.
import pandas as pd

from . import apisports, hyprace, jolpica, openf1


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def race_extras(year, rnd, race_date, id2code=None):
    """Pit stops / strategy / overtakes / safety cars for a FINISHED race.
    OpenF1 for 2023+, otherwise Jolpica pit stops (2011+).
    -> (DataFrame indexed by driver code or None, race-level dict)"""
    ex, rl = None, {}
    if int(year) >= 2023:
        got = _try(openf1.race_extras, race_date)
        if got and got[0] is not None:
            ex, rl = got
    if ex is None and int(year) >= 2011:
        if id2code is None:
            res = _try(lambda: jolpica.client.get(f"/{year}/{rnd}/results.json", {"limit": 100})["MRData"]["RaceTable"]["Races"])
            id2code = {r["Driver"]["driverId"]: jolpica._code(r["Driver"]) for r in (res[0]["Results"] if res else [])}
        ex = _try(jolpica.pitstops_frame, year, rnd, id2code)
    return ex, rl


def official_grid(year, rnd, race_date):
    """-> ({driver code: position}, source name) or (None, None). OpenF1 first, then API-Sports."""
    if int(year) >= 2023:
        g = _try(openf1.starting_grid, race_date)
        if g:
            return g, "openf1"
    if apisports.available():
        g = _try(apisports.starting_grid, race_date)
        if g:
            return g, "apisports"
    return None, None


def race_start(year, rnd, ev=None):
    """Race start (UTC): FastF1 calendar first, Hyprace as fallback."""
    if ev is not None:
        v = ev.get("Session5DateUtc") if hasattr(ev, "get") else None
        if v is not None and pd.notna(v):
            return v
    if hyprace.available():
        return _try(hyprace.race_start_utc, year, rnd)
    return None
