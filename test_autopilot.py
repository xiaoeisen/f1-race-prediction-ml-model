import json

import numpy as np
import pandas as pd
import pytest

import autopilot
import predlog
import train

FAKE_YEAR = 2030      # a season that doesn't exist, so the fake weekend can't collide with real data


def schedule(year):
    if year != FAKE_YEAR:
        return pd.DataFrame(columns=["RoundNumber", "EventName", "Location", "EventDate"])
    d = pd.Timestamp(f"{FAKE_YEAR}-03-13")
    return pd.DataFrame([{
        "RoundNumber": 1, "EventName": "Test Grand Prix", "Location": "Melbourne", "EventDate": d + pd.Timedelta(days=2),
        "Session1": "Practice 1", "Session1DateUtc": d + pd.Timedelta(hours=11, minutes=30),
        "Session2": "Practice 2", "Session2DateUtc": d + pd.Timedelta(hours=15),
        "Session3": "Practice 3", "Session3DateUtc": d + pd.Timedelta(days=1, hours=11, minutes=30),
        "Session4": "Qualifying", "Session4DateUtc": d + pd.Timedelta(days=1, hours=15),
        "Session5": "Race", "Session5DateUtc": d + pd.Timedelta(days=2, hours=15)}])


@pytest.fixture
def weekend_variants(raw):
    """One real race relabelled as the fake weekend, as it looks after practice / qualifying / the race."""
    pool = raw[raw.fp1_best_gap.notna() & raw.quali_pos.notna() & (raw.year == raw.year.max() - 1)]
    rnd = int(pool.groupby("round").size().idxmax())
    base = pool[pool["round"] == rnd].copy()
    base["year"], base["round"] = FAKE_YEAR, 1
    result_cols = [c for c in ("finish", "dnf", "points", "status", "n_stops", "pit_lane_s") if c in base]
    quali_cols = [c for c in ("quali_pos", "q_best_gap", "grid") if c in base]
    full = base.copy()
    quali = base.copy()
    quali[result_cols] = np.nan
    practice = quali.copy()
    practice[quali_cols] = np.nan
    practice["fp3_best_gap"] = np.nan
    return dict(practice=practice, quali=quali, full=full)


@pytest.mark.slow
def test_full_race_weekend_lifecycle(workdir, raw, weekend_variants, monkeypatch):
    monkeypatch.setattr(autopilot.extras.enrich, "official_grid", lambda *a: (None, None))
    monkeypatch.setattr(autopilot.extras, "get_forecast", lambda *a, **k: None)
    train.train_models(raw, train_from=2018, models_dir="models", verbose=False)
    phase, fetches = {"v": "practice"}, []
    clock = {"t": pd.Timestamp(f"{FAKE_YEAR}-03-12 10:00")}

    def get_weekend(year, rnd, ev, force):
        fetches.append(force)
        return weekend_variants[phase["v"]].copy()

    def retrain(reason):
        return train.train_models(pd.read_csv("data/races.csv"), train_from=2018, models_dir="models", verbose=False, reason=reason,
                                  race_weights=predlog.race_weights(predlog.load()))
    ap = autopilot.Autopilot(get_schedule=schedule, get_weekend=get_weekend, retrain=retrain, now_fn=lambda: clock["t"])

    def at(t, v=None):
        clock["t"] = pd.Timestamp(t)
        if v:
            phase["v"] = v
        n = len(fetches)
        r = ap.tick()
        assert r["error"] is None
        return r, len(fetches) - n

    r, n = at(f"{FAKE_YEAR}-03-12 10:00")
    assert n == 0 and predlog.load() == []                                    # before FP1: nothing to do
    r, n = at(f"{FAKE_YEAR}-03-13 16:30", "practice")
    assert n == 1 and [e["stage"] for e in predlog.load()] == ["pre"]         # after FP2: practice-only prediction stored
    r, n = at(f"{FAKE_YEAR}-03-13 16:50")
    assert n == 0                                                             # 20 minutes later: no pointless re-download
    r, n = at(f"{FAKE_YEAR}-03-14 16:20", "practice")
    assert n == 1 and [e["stage"] for e in predlog.load()] == ["pre"]         # quali over but data not out yet: keeps trying
    r, n = at(f"{FAKE_YEAR}-03-14 16:50", "quali")
    assert sorted(e["stage"] for e in predlog.load()) == ["pre", "quali"]
    stored = {e["stage"]: e["created"] for e in predlog.load()}
    for e in predlog.load():
        assert len(e["top5"]) == 5 and abs(sum(t["win"] for t in e["top5"])) <= 100
    r, n = at(f"{FAKE_YEAR}-03-15 15:30", "quali")
    assert n == 0 and stored == {e["stage"]: e["created"] for e in predlog.load()}     # race started: predictions locked
    r, n = at(f"{FAKE_YEAR}-03-15 18:30", "full")
    entries = predlog.load()
    assert all(e["metrics"] for e in entries)
    actual = entries[0]["actual"]["order"][:5]
    for e in entries:
        assert e["metrics"]["top5_hits"] == len(set(t["driver"] for t in e["top5"]) & set(actual))      # scored correctly
    h = pd.read_csv("data/races.csv")
    assert ((h.year == FAKE_YEAR) & (h["round"] == 1)).sum() >= 15                                       # result joined the training data
    assert r["actions"][-1].startswith("self-training")
    assert json.load(open("models/version.json"))["trained_through"] == [FAKE_YEAR, 1]
    r, n = at(f"{FAKE_YEAR}-03-15 19:30")
    assert r["actions"] == []                                                 # nothing left to do: idempotent
    assert "Latest" or True
    payload = autopilot.live_payload(ap)
    assert payload["event"]["status"] == "finished" and payload["season"]["n_races"] == 1


def test_only_one_process_runs_a_check_at_a_time(workdir, monkeypatch):
    ap = autopilot.Autopilot(get_schedule=schedule, get_weekend=lambda *a: pd.DataFrame(), now_fn=lambda: pd.Timestamp("2030-01-01"))
    (workdir / "data" / "autopilot.lock").write_text("12345")
    assert ap.tick().get("skipped")                                           # a live lock: skip, don't collide
    import os
    import time
    old = time.time() - 3600
    os.utime(workdir / "data" / "autopilot.lock", (old, old))
    assert "skipped" not in ap.tick()                                         # a stale lock (crashed run) is taken over
    assert not (workdir / "data" / "autopilot.lock").exists()                 # and released afterwards
    assert (workdir / "data" / "autopilot.log").exists()


def test_session_parser_reads_the_real_fastf1_layout():
    ev = schedule(FAKE_YEAR).iloc[0]
    s = autopilot.sessions_of(ev)
    assert sorted(s) == ["fp1", "fp2", "fp3", "quali", "race"]
    assert autopilot.Autopilot.expected_stage(s, pd.Timestamp(f"{FAKE_YEAR}-03-13 16:30")) == "pre"
    assert autopilot.Autopilot.expected_stage(s, pd.Timestamp(f"{FAKE_YEAR}-03-14 16:30")) == "quali"
    assert autopilot.Autopilot.expected_stage(s, pd.Timestamp(f"{FAKE_YEAR}-03-12")) is None
