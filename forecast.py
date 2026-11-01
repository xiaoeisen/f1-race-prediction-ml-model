# Fetch race-hour weather when a forecast is available.
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from circuits import lookup

_cache = {}
TTL = 3 * 3600


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def coords(location):
    row = lookup(location)
    if row is not None:
        return float(row["lat"]), float(row["lon"])
    try:  # fall back to geocoding the circuit's location name
        d = _get("https://geocoding-api.open-meteo.com/v1/search?"
                 + urllib.parse.urlencode({"name": location, "count": 1}))
        r = d["results"][0]
        return r["latitude"], r["longitude"]
    except Exception:
        return None


def get_forecast(location, race_date, start_utc=None):
    """-> {'rain_prob': 0..1, 'air_temp': deg C} or None.
    With a race start time the forecast covers the race window (start .. +2h);
    without it, the daily maximum rain probability and high temperature."""
    start = None
    if start_utc is not None:
        try:
            start = pd.Timestamp(start_utc)
            start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        except Exception:
            start = None
    key = (str(location), str(race_date), str(start))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    out = None
    try:
        c = coords(location)
        if c:
            day = str(start.date()) if start is not None else str(race_date)
            q = dict(latitude=c[0], longitude=c[1], timezone="UTC", start_date=day, end_date=day)
            if start is not None:
                q["hourly"] = "precipitation_probability,temperature_2m"
                h = _get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))["hourly"]
                t0 = start.floor("h").tz_localize(None)
                idx = [i for i, t in enumerate(h["time"])
                       if t0 <= pd.Timestamp(t) <= t0 + pd.Timedelta(hours=2)]
                p = [h["precipitation_probability"][i] for i in idx if h["precipitation_probability"][i] is not None]
                tt = [h["temperature_2m"][i] for i in idx if h["temperature_2m"][i] is not None]
                if p and tt:
                    out = dict(rain_prob=max(p) / 100.0, air_temp=float(sum(tt) / len(tt)))
            else:
                q["daily"] = "precipitation_probability_max,temperature_2m_max"
                d = _get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))["daily"]
                p, t = d["precipitation_probability_max"][0], d["temperature_2m_max"][0]
                if p is not None and t is not None:
                    out = dict(rain_prob=float(p) / 100.0, air_temp=float(t))
    except Exception:
        out = None
    _cache[key] = (time.time(), out)
    return out
