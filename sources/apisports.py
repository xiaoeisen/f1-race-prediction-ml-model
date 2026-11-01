# Use API-Sports as a fallback for official grids; configure APISPORTS_KEY to enable it.
import os

import pandas as pd

from .http import Client


def available():
    return bool(os.environ.get("APISPORTS_KEY"))


def _client():
    return Client("apisports", "https://v1.formula-1.api-sports.io",
                  headers={"x-apisports-key": os.environ.get("APISPORTS_KEY", "")}, per_minute=8)


def starting_grid(race_date):
    """{driver code: position} for the race on this date, or None."""
    if not available():
        return None
    c = _client()
    d = pd.Timestamp(race_date).tz_localize(None) if pd.Timestamp(race_date).tzinfo else pd.Timestamp(race_date)
    races = c.get("/races", {"season": d.year, "type": "Race"}, ttl=6 * 3600).get("response", [])
    race_id = None
    for r in races:
        rd = pd.Timestamp(r.get("date"))
        rd = rd.tz_localize(None) if rd.tzinfo is None else rd.tz_convert("UTC").tz_localize(None)
        if abs((rd.normalize() - d.normalize()).days) <= 1:
            race_id = r.get("id")
            break
    if race_id is None:
        return None
    rows = c.get("/rankings/startinggrid", {"race": race_id}, ttl=900).get("response", [])
    grid = {}
    for r in rows:
        abbr = (r.get("driver") or {}).get("abbr")
        if abbr and r.get("position") is not None:
            grid[abbr] = int(r["position"])
    return grid or None
