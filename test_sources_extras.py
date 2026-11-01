import email.message
import json
import urllib.error
import urllib.parse

import numpy as np
import pandas as pd
import pytest

import extras
import forecast
import sources.http as H
from sources import apisports, hyprace, jolpica, openf1


class Resp:
    def __init__(self, obj):
        self.b = json.dumps(obj).encode()

    def read(self):
        return self.b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_err(code):
    return urllib.error.HTTPError("http://x", code, "e", email.message.Message(), None)


@pytest.fixture
def net(monkeypatch, tmp_path):
    """Fake internet: net.routes[(host, path)] = function(params) -> JSON or an exception."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    routes, calls = {}, []

    def fake(req, timeout=20):
        u = urllib.parse.urlparse(req.full_url)
        calls.append((u.netloc, u.path))
        fn = routes.get((u.netloc, u.path))
        if fn is None:
            raise http_err(404)
        out = fn({k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()})
        if isinstance(out, Exception):
            raise out
        return Resp(out)
    monkeypatch.setattr(H, "_urlopen", fake)
    fake.routes, fake.calls = routes, calls
    return fake


def test_http_client_caches_retries_and_handles_404(net):
    n = {"c": 0}

    def flaky(p):
        n["c"] += 1
        return http_err(429) if n["c"] == 1 else {"ok": True}
    net.routes[("h", "/x")] = flaky
    c = H.Client("t", "https://h")
    assert c.get("/x") == {"ok": True} and n["c"] == 2          # one 429, then success
    assert c.get("/x") == {"ok": True} and n["c"] == 2          # second call served from the disk cache
    assert c.get("/missing", allow_404=True) == []
    with pytest.raises(H.SourceError):
        c.get("/missing-and-not-allowed")                       # a 404 is an error unless the caller says it is fine


def test_openf1_extras_and_official_grid(net):
    h = "api.openf1.org"
    net.routes[(h, "/v1/meetings")] = lambda p: [
        {"meeting_key": 1, "meeting_name": "Pre-Season Testing", "date_start": "2025-02-26T07:00:00+00:00", "date_end": "2025-02-28T16:00:00+00:00"},
        {"meeting_key": 9, "meeting_name": "GP", "date_start": "2025-12-05T09:30:00+00:00", "date_end": "2025-12-07T14:00:00+00:00"}]
    net.routes[(h, "/v1/sessions")] = lambda p: [{"session_key": 90 if p["session_name"] == "Race" else 89}]
    net.routes[(h, "/v1/drivers")] = lambda p: [{"driver_number": 1, "name_acronym": "VER"}, {"driver_number": 4, "name_acronym": "NOR"}]
    net.routes[(h, "/v1/pit")] = lambda p: [{"driver_number": 1, "lane_duration": 22.0, "stop_duration": 2.0},
                                            {"driver_number": 1, "lane_duration": 24.0, "stop_duration": 2.4}]
    net.routes[(h, "/v1/stints")] = lambda p: [{"driver_number": 1, "stint_number": n} for n in (1, 2, 3)] + [{"driver_number": 4, "stint_number": 1}]
    net.routes[(h, "/v1/overtakes")] = lambda p: [{"overtaking_driver_number": 1, "overtaken_driver_number": 4}]
    net.routes[(h, "/v1/race_control")] = lambda p: [{"category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
                                                     {"category": "SafetyCar", "message": "VIRTUAL SAFETY CAR DEPLOYED"},
                                                     {"category": "Flag", "message": "RED FLAG", "flag": "RED"}]
    ex, rl = openf1.race_extras("2025-12-07")
    assert ex.loc["VER", "n_stops"] == 2 and ex.loc["NOR", "n_stops"] == 0
    assert ex.loc["VER", "pit_lane_s"] == 23.0 and ex.loc["VER", "stop_s"] == 2.2
    assert ex.loc["VER", "overtakes_made"] == 1 and ex.loc["NOR", "overtakes_suffered"] == 1
    assert rl == dict(n_sc=1.0, n_vsc=1.0, red=1.0)
    # the docs don't say which session the grid belongs to: Race first, then Qualifying
    net.routes[(h, "/v1/starting_grid")] = lambda p: ([{"position": 1, "driver_number": 4}, {"position": 2, "driver_number": 1}]
                                                      if p["session_key"] == "89" else http_err(404))
    assert openf1.starting_grid("2025-12-07") == {"NOR": 1, "VER": 2}


def test_jolpica_merges_split_pages_and_reads_pit_stops(net, monkeypatch):
    drv = lambda c, f: {"driverId": f.lower(), "code": c, "givenName": "G", "familyName": f}
    res = lambda pos, grid, c, f, team, st="Finished": {"position": str(pos), "grid": str(grid), "points": "0", "status": st,
                                                       "Driver": drv(c, f), "Constructor": {"name": team}}
    meta = lambda rn, name, loc: dict(season="2022", round=str(rn), raceName=name, date="2022-03-20", Circuit={"Location": {"locality": loc}})
    races = {1: (meta(1, "Bahrain", "Sakhir"), [res(1, 1, "LEC", "Leclerc", "Ferrari"), res(2, 3, "SAI", "Sainz", "Ferrari"), res(3, 0, "HAM", "Hamilton", "Mercedes", "+1 Lap")]),
             2: (meta(2, "Saudi", "Jeddah"), [res(1, 1, "VER", "Verstappen", "Red Bull"), res(2, 2, "LEC", "Leclerc", "Ferrari"), res(20, 5, "HAM", "Hamilton", "Mercedes", "Engine")])}

    def page(p):
        flat = [(rn, r) for rn, (m, rs) in races.items() for r in rs]
        off, lim = int(p["offset"]), int(p["limit"])
        out = {}
        for rn, r in flat[off:off + lim]:
            out.setdefault(rn, {**races[rn][0], "Results": []})["Results"].append(r)
        return {"MRData": {"total": str(len(flat)), "RaceTable": {"Races": list(out.values())}}}
    net.routes[("api.jolpi.ca", "/ergast/f1/2022/results.json")] = page
    net.routes[("api.jolpi.ca", "/ergast/f1/2022/qualifying.json")] = lambda p: {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
    for rn in (1, 2):
        net.routes[("api.jolpi.ca", f"/ergast/f1/2022/{rn}/pitstops.json")] = lambda p, rn=rn: {"MRData": {"total": "2", "RaceTable": {"Races": [dict(
            races[rn][0], PitStops=[{"driverId": "leclerc", "duration": "23.4"}, {"driverId": "leclerc", "duration": "24.6"}])]}}}
    monkeypatch.setattr(jolpica, "PAGE", 4)                     # forces race 1 to be split across two pages
    f = jolpica.season_frame(2022)
    assert f.groupby("round").size().to_dict() == {1: 3, 2: 3}  # nothing lost or duplicated by the split
    assert f[(f["round"] == 1) & (f.driver == "HAM")].grid.iloc[0] == 20   # pit-lane start (grid 0) -> last
    assert f[(f["round"] == 2) & (f.driver == "HAM")].dnf.iloc[0] == 1
    lec = f[(f["round"] == 1) & (f.driver == "LEC")].iloc[0]
    assert lec.n_stops == 2 and abs(lec.pit_lane_s - 24.0) < 1e-9


def test_optional_sources_need_keys(net, monkeypatch):
    monkeypatch.delenv("APISPORTS_KEY", raising=False)
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    assert apisports.starting_grid("2025-12-07") is None and hyprace.race_start_utc(2026, 3) is None
    monkeypatch.setenv("APISPORTS_KEY", "k")
    net.routes[("v1.formula-1.api-sports.io", "/races")] = lambda p: {"response": [{"id": 7, "date": "2025-12-07T13:00:00+00:00"}]}
    net.routes[("v1.formula-1.api-sports.io", "/rankings/startinggrid")] = lambda p: {"response": [{"position": 1, "driver": {"abbr": "NOR"}}]}
    assert apisports.starting_grid("2025-12-07") == {"NOR": 1}


def test_penalties_shift_the_grid(tmp_path):
    rows = pd.DataFrame({"driver": [f"D{i:02d}" for i in range(1, 21)], "grid": np.arange(1.0, 21)})
    p = tmp_path / "pen.csv"
    p.write_text("year,round,driver,places\n2025,7,D01,10\n2025,7,D05,99\n2024,7,D02,5\n")
    assert extras.apply_penalties(rows, 2025, 7, path=p) == 2
    g = dict(zip(rows.driver, rows.grid.astype(int)))
    assert g["D05"] == 20 and g["D02"] == 1 and sorted(g.values()) == list(range(1, 21))   # still a clean 1..20 grid


def test_forecast_uses_the_race_hour_window(monkeypatch):
    hours = [f"2026-10-11T{h:02d}:00" for h in range(24)]
    prob = [5] * 24
    prob[15], prob[20] = 70, 95                                  # 95% late evening must be ignored
    monkeypatch.setattr(forecast, "_get", lambda url, timeout=8: {"hourly": {
        "time": hours, "precipitation_probability": prob, "temperature_2m": [20.0 + h * 0.5 for h in range(24)]}})
    forecast._cache.clear()
    fc = forecast.get_forecast("Marina Bay", "2026-10-11", "2026-10-11T14:00:00Z")
    assert fc == {"rain_prob": 0.7, "air_temp": 27.5}
    forecast._cache.clear()
    monkeypatch.setattr(forecast, "_get", lambda url, timeout=8: (_ for _ in ()).throw(OSError("offline")))
    assert forecast.get_forecast("Marina Bay", "2026-10-11", "2026-10-11T14:00:00Z") is None   # fails soft


def test_apply_live_prefers_official_grid_then_penalties(monkeypatch, tmp_path):
    drivers = [f"D{i:02d}" for i in range(1, 21)]
    rows = lambda: pd.DataFrame({"driver": drivers, "grid": np.arange(1.0, 21), "finish": np.nan, "rain": 0.0, "air_temp": 20.0})
    ev = pd.Series({"EventDate": pd.Timestamp("2026-10-11"), "Location": "Marina Bay"})
    monkeypatch.setattr(extras, "get_forecast", lambda *a, **k: None)
    monkeypatch.setattr(extras, "PENALTIES", tmp_path / "pen.csv")
    (tmp_path / "pen.csv").write_text("year,round,driver,places\n2026,99,D01,10\n")
    official = {d: i + 2 for i, d in enumerate(drivers)}
    official["D05"] = 1
    monkeypatch.setattr(extras.enrich, "official_grid", lambda *a: (official, "openf1"))
    r = rows()
    assert extras.apply_live(r, 2026, 99, ev)["grid_source"] == "openf1" and r.set_index("driver").grid["D05"] == 1
    monkeypatch.setattr(extras.enrich, "official_grid", lambda *a: (None, None))
    r = rows()
    info = extras.apply_live(r, 2026, 99, ev)
    assert info["grid_source"] == "penalties.csv" and r.set_index("driver").grid["D01"] == 11
    done = rows().assign(finish=1.0)
    assert extras.apply_live(done, 2026, 99, ev)["grid_source"] is None      # finished races are never touched
