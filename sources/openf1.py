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
