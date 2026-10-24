# Use Hyprace for current-season start times when FastF1 has none; configure RAPIDAPI_KEY.
import os

from .http import Client


def available():
    return bool(os.environ.get("RAPIDAPI_KEY"))


def _client():
    return Client("hyprace", "https://hyprace-api.p.rapidapi.com/v2",
                  headers={"x-rapidapi-key": os.environ.get("RAPIDAPI_KEY", ""),
                           "x-rapidapi-host": "hyprace-api.p.rapidapi.com"}, per_second=2)


def race_start_utc(year, rnd):
    if not available():
        return None
    d = _client().get("/grands-prix", {"isCurrent": "true", "pageSize": 25}, ttl=6 * 3600)
    for gp in d.get("items", []):
        if gp.get("round") == rnd and (gp.get("season") or {}).get("year") == year:
            for s in gp.get("schedule", []):
                if s.get("type") == "MainRace":
                    return s.get("startDate")
    return None
