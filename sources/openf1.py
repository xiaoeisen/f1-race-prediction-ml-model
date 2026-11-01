# Use OpenF1 data (CC BY-NC-SA) for race extras and official grids from 2023 onward.
import numpy as np
import pandas as pd

from .http import Client

client = Client("openf1", "https://api.openf1.org/v1", per_second=2, per_minute=25)


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _list(path, **params):
    r = client.get(path, params, allow_404=True, ttl=None)
    return r if isinstance(r, list) else []


def meeting_for(race_date):
    """The meeting (race weekend) containing this race date."""
    d = _utc(race_date)
    ms = client.get("/meetings", {"year": d.year}, allow_404=True, ttl=6 * 3600)
    best, best_gap = None, None
    for m in ms if isinstance(ms, list) else []:
        if m.get("is_cancelled") or "testing" in str(m.get("meeting_name", "")).lower():
            continue
        start, end = _utc(m["date_start"]), _utc(m["date_end"])
        if start - pd.Timedelta(days=1) <= d <= end + pd.Timedelta(days=1):
            gap = abs((end - d).total_seconds())
            if best is None or gap < best_gap:
                best, best_gap = m, gap
    return best


def session_key(meeting_key, name):
    s = _list("/sessions", meeting_key=meeting_key, session_name=name)
    return s[0]["session_key"] if s else None


def driver_codes(sk):
    return {d["driver_number"]: d["name_acronym"] for d in _list("/drivers", session_key=sk) if d.get("name_acronym")}


def _best_duration(value):
    """OpenF1 qualifying duration may be a scalar or a Q1/Q2/Q3 list."""
    values = value if isinstance(value, (list, tuple)) else [value]
    numeric = []
    for item in values:
        try:
            x = float(item)
            if np.isfinite(x) and x > 0:
                numeric.append(x)
        except (TypeError, ValueError):
            pass
    return min(numeric) if numeric else np.nan


def _classification_duration(value):
    """Use the last completed qualifying segment when the API returns Q1/Q2/Q3."""
    values = value if isinstance(value, (list, tuple)) else [value]
    numeric = []
    for item in values:
        try:
            number = float(item)
            if np.isfinite(number) and number > 0:
                numeric.append(number)
        except (TypeError, ValueError):
            pass
    return numeric[-1] if numeric else np.nan


def _qualifying_weather(sk):
    """Summarize weather observed during qualifying, never during the race."""
    rows = _list("/weather", session_key=sk)
    if not rows:
        return {}
    data = pd.DataFrame(rows)
    mapping = {"air_temperature": "q_air_temp", "track_temperature": "q_track_temp",
               "humidity": "q_humidity", "wind_speed": "q_wind"}
    result = {}
    for source, target in mapping.items():
        values = pd.to_numeric(data.get(source, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(values):
            result[target] = float(values.mean())
    if "rainfall" in data:
        rain = pd.to_numeric(data["rainfall"], errors="coerce").fillna(0)
        result["q_rain"] = float(rain.gt(0).any())
    return result


def weekend_features(race_date):
    """Return published practice/quali timing for one weekend (historical OpenF1 data)."""
    meeting = meeting_for(race_date)
    if not meeting:
        return None
    sessions = _list("/sessions", meeting_key=meeting["meeting_key"])
    names = {"Practice 1": "fp1", "Practice 2": "fp2", "Practice 3": "fp3",
             "Qualifying": "quali"}
    drivers, features, q_weather = {}, {}, {}
    for session in sessions:
        name = names.get(session.get("session_name"))
        sk = session.get("session_key")
        if not name or sk is None:
            continue
        driver_rows = _list("/drivers", session_key=sk)
        codes = {d["driver_number"]: d.get("name_acronym") for d in driver_rows
                 if d.get("driver_number") is not None and d.get("name_acronym")}
        for d in driver_rows:
            code = codes.get(d.get("driver_number"))
            if code:
                drivers[code] = dict(full_name=d.get("full_name") or code,
                                     team=d.get("team_name") or "")
        results = _list("/session_result", session_key=sk)
        if name == "quali":
            q_weather = _qualifying_weather(sk)
        times = {}
        for row in results:
            code = codes.get(row.get("driver_number"))
            duration = (_classification_duration(row.get("duration")) if name == "quali"
                        else _best_duration(row.get("duration")))
            if code and np.isfinite(duration):
                times[code] = duration
                if name == "quali":
                    features.setdefault(code, {})["q_time_segment_valid"] = 1.0
            if name == "quali" and code and row.get("position") is not None:
                features.setdefault(code, {})["quali_pos"] = float(row["position"])
        laps = _list("/laps", session_key=sk)
        lap_counts, lap_bests = {}, {}
        for lap in laps:
            code = codes.get(lap.get("driver_number"))
            duration = _best_duration(lap.get("lap_duration"))
            if not code or not np.isfinite(duration):
                continue
            lap_counts[code] = lap_counts.get(code, 0) + 1
            if not lap.get("is_pit_out_lap", False):
                lap_bests[code] = min(lap_bests.get(code, np.inf), duration)
        for code, duration in lap_bests.items():
            if name == "quali":
                times.setdefault(code, duration)
            else:
                times[code] = min(times.get(code, np.inf), duration)
        if times:
            fastest = min(times.values())
            col = "q_best_gap" if name == "quali" else f"{name}_best_gap"
            time_col = "q_best_time_s" if name == "quali" else f"{name}_best_time_s"
            gap_col = "q_gap_s" if name == "quali" else f"{name}_gap_s"
            laps_col = "quali_laps" if name == "quali" else f"{name}_laps"
            for code, duration in times.items():
                values = features.setdefault(code, {})
                values[col] = (duration / fastest - 1) * 100
                values[time_col] = duration
                values[gap_col] = duration - fastest
            for code, count in lap_counts.items():
                features.setdefault(code, {})[laps_col] = count
        if name == "quali" and laps:
            for raw_col, out_col, reducer in (
                    ("duration_sector_1", "q_s1_gap", "min"),
                    ("duration_sector_2", "q_s2_gap", "min"),
                    ("duration_sector_3", "q_s3_gap", "min"),
                    ("st_speed", "q_topspeed_gap", "max")):
                vals = {}
                for lap in laps:
                    code = codes.get(lap.get("driver_number"))
                    try:
                        value = float(lap.get(raw_col))
                    except (TypeError, ValueError):
                        continue
                    if code and np.isfinite(value) and value > 0:
                        vals[code] = (min if reducer == "min" else max)(vals.get(code, value), value)
                if vals:
                    best = max(vals.values()) if reducer == "max" else min(vals.values())
                    for code, value in vals.items():
                        features.setdefault(code, {})[out_col] = ((1 - value / best) * 100 if reducer == "max"
                                                                 else (value / best - 1) * 100)
    if q_weather:
        for code in drivers:
            features.setdefault(code, {}).update(q_weather)
    if not features:
        return None
    rows = []
    for code, values in features.items():
        rows.append(dict(driver=code, **drivers.get(code, {}), **values))
    return pd.DataFrame(rows)


def race_extras(race_date):
    """-> (DataFrame indexed by driver code, race-level dict) for a FINISHED race, or (None, {})."""
    m = meeting_for(race_date)
    sk = session_key(m["meeting_key"], "Race") if m else None
    if not sk:
        return None, {}
    codes = driver_codes(sk)
    pit, stints = _list("/pit", session_key=sk), _list("/stints", session_key=sk)
    over, rc = _list("/overtakes", session_key=sk), _list("/race_control", session_key=sk)

    ex = pd.DataFrame(index=pd.Index(sorted(set(codes.values())), name="driver"),
                      columns=["n_stops", "pit_lane_s", "stop_s", "overtakes_made", "overtakes_suffered"], dtype=float)
    lane, stop, npit = {}, {}, {}
    for p in pit:
        c = codes.get(p.get("driver_number"))
        if not c:
            continue
        npit[c] = npit.get(c, 0) + 1
        d = p.get("lane_duration", p.get("pit_duration"))
        if d is not None:
            lane.setdefault(c, []).append(float(d))
        if p.get("stop_duration") is not None:
            stop.setdefault(c, []).append(float(p["stop_duration"]))
    nstint = {}
    for s in stints:
        c = codes.get(s.get("driver_number"))
        if c:
            nstint[c] = max(nstint.get(c, 0), int(s.get("stint_number", 0)))
    made, suffered = {}, {}
    for o in over:
        a, b = codes.get(o.get("overtaking_driver_number")), codes.get(o.get("overtaken_driver_number"))
        if a:
            made[a] = made.get(a, 0) + 1
        if b:
            suffered[b] = suffered.get(b, 0) + 1
    for c in ex.index:
        n = nstint[c] - 1 if c in nstint else npit.get(c)
        ex.loc[c, "n_stops"] = np.nan if n is None else max(n, 0)
        ex.loc[c, "pit_lane_s"] = np.mean(lane[c]) if c in lane else np.nan
        ex.loc[c, "stop_s"] = np.mean(stop[c]) if c in stop else np.nan
        if over:  # overtakes feed is "may be incomplete": only trust it if present at all
            ex.loc[c, "overtakes_made"] = made.get(c, 0)
            ex.loc[c, "overtakes_suffered"] = suffered.get(c, 0)

    n_sc = n_vsc = 0
    red = False
    for r in rc:
        msg = str(r.get("message", "")).upper()
        if "DEPLOYED" in msg and "SAFETY CAR" in msg:
            if "VIRTUAL" in msg:
                n_vsc += 1
            else:
                n_sc += 1
        if str(r.get("flag", "")).upper() == "RED":
            red = True
    return ex, dict(n_sc=float(n_sc), n_vsc=float(n_vsc), red=float(red)) if rc else {}


def race_pace_features(race_date):
    """Return robust race-lap pace summaries keyed by driver code.

    OpenF1 laps do not provide a complete track-status timeline. The clean-lap
    summary therefore removes pit-out laps and strong within-stint outliers; it
    is a pace estimate, not an official green-flag classification.
    """
    meeting = meeting_for(race_date)
    sk = session_key(meeting["meeting_key"], "Race") if meeting else None
    if not sk:
        return None
    codes = driver_codes(sk)
    raw_laps = _list("/laps", session_key=sk)
    if not raw_laps:
        return None
    laps = pd.DataFrame(raw_laps)
    laps["driver"] = laps["driver_number"].map(codes)
    laps["lap"] = pd.to_numeric(laps.get("lap_number"), errors="coerce")
    laps["time_s"] = pd.to_numeric(laps.get("lap_duration"), errors="coerce")
    laps = laps.dropna(subset=["driver", "lap", "time_s"])
    laps = laps[(laps["lap"] >= 2) & (laps["time_s"] > 0)]
    if "is_pit_out_lap" in laps:
        pit_out = laps["is_pit_out_lap"].map(lambda x: str(x).lower() in ("true", "1"))
        laps = laps[~pit_out]
    if laps.empty:
        return None

    stints = _list("/stints", session_key=sk)
    if stints:
        stint_rows = []
        for stint in stints:
            code = codes.get(stint.get("driver_number"))
            try:
                start, end = int(stint.get("lap_start")), int(stint.get("lap_end"))
                number = int(stint.get("stint_number", 1))
            except (TypeError, ValueError):
                continue
            if code and end >= start:
                stint_rows.append((code, start, end, number, stint.get("compound"),
                                   pd.to_numeric(stint.get("tyre_age_at_start"), errors="coerce")))
        if stint_rows:
            laps["stint"] = np.nan
            laps["compound"] = np.nan
            laps["tyre_age"] = np.nan
            for code, start, end, number, compound, age in stint_rows:
                mask = laps["driver"].eq(code) & laps["lap"].between(start, end)
                laps.loc[mask, "stint"] = number
                laps.loc[mask, "compound"] = compound
                laps.loc[mask, "tyre_age"] = age + laps.loc[mask, "lap"] - start
    if "stint" not in laps or laps["stint"].isna().all():
        laps["stint"] = 1
    laps["stint"] = laps["stint"].fillna(1).astype(int)
    center = laps.groupby(["driver", "stint"])["time_s"].transform("median")
    mad = laps.groupby(["driver", "stint"])["time_s"].transform(
        lambda values: (values - values.median()).abs().median())
    cutoff = np.maximum(4 * 1.4826 * mad, center * 0.03)
    clean = laps[(laps["time_s"] - center).abs() <= cutoff].copy()
    clean = clean[clean["time_s"] <= clean.groupby("driver")["time_s"].transform("median") * 1.08]
    if clean.empty:
        return None

    means = clean.groupby("driver")["time_s"].mean()
    fastest = means.min()
    result = pd.DataFrame(index=means.index)
    result["race_pace_gap"] = (means / fastest - 1) * 100
    result["race_best_lap_s"] = clean.groupby("driver")["time_s"].min()
    result["race_avg_clean_lap_s"] = means
    result["race_laps"] = laps.groupby("driver")["lap"].nunique()
    deg = {}
    if "tyre_age" in clean:
        for code, group in clean.dropna(subset=["tyre_age"]).groupby("driver"):
            if len(group) >= 8 and group["tyre_age"].nunique() >= 5:
                slope = np.polyfit(group["tyre_age"].to_numpy(float),
                                   group["time_s"].to_numpy(float), 1)[0]
                deg[code] = slope * 10 / group["time_s"].median() * 100
    result["race_deg_per10"] = pd.Series(deg)
    return result


def starting_grid(race_date):
    """Official grid (penalties included): {driver code: position}, or None if not published yet."""
    m = meeting_for(race_date)
    if not m:
        return None
    # docs don't say whether the key is the race or the qualifying session, so try both
    for name in ("Race", "Qualifying"):
        sk = session_key(m["meeting_key"], name)
        if not sk:
            continue
        rows = client.get("/starting_grid", {"session_key": sk}, allow_404=True, ttl=900)
        if isinstance(rows, list) and rows:
            codes = driver_codes(sk)
            grid = {codes[r["driver_number"]]: int(r["position"]) for r in rows
                    if r.get("driver_number") in codes and r.get("position") is not None}
            if grid:
                return grid
    return None
