"""Autopilot: keeps the predictor up to date by itself.

    python autopilot.py --once          # one check (run it from cron / Task Scheduler)
    python autopilot.py --loop          # keep checking every 20 minutes
    python autopilot.py --status

The web server (server.py) runs this loop in the background automatically.

What one check does:
  * race weekend in progress -> when FP1/FP2 or qualifying data becomes available, fetch it
    (including the official starting grid / forecast) and store a new top-5 prediction.
    Predictions are LOCKED once the race starts, so they can't be tampered with afterwards.
  * race finished -> fetch the real result, add it to the training data, score every stored
    prediction (accuracy), then retrain the models (self-training). A retrained model only goes
    live if it passes a sanity check against the "finish = grid" baseline.

Needs data/races.csv and trained models first (python collect.py, python train.py)."""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import joblib
import pandas as pd

import extras
import predlog
from predict import predict_race

DATA = Path("data")
STATE = DATA / "autopilot_state.json"
LOCK = DATA / "autopilot.lock"
LOGFILE = DATA / "autopilot.log"
LOCK_STALE_MIN = 30      # a lock older than this is from a crashed run
HIST = DATA / "races.csv"
MODELS = Path("models")

SESSION_KEYS = {"Practice 1": "fp1", "Practice 2": "fp2", "Practice 3": "fp3", "Qualifying": "quali",
                "Sprint Qualifying": "sq", "Sprint Shootout": "sq", "Sprint": "sprint", "Race": "race"}
SESSION_LABEL = {"fp1": "Free Practice 1", "fp2": "Free Practice 2", "fp3": "Free Practice 3", "sq": "Sprint qualifying",
                 "sprint": "Sprint", "quali": "Qualifying (Q1-Q3)", "race": "Race"}
DURATION_MIN = {"fp1": 60, "fp2": 60, "fp3": 60, "sq": 45, "sprint": 40, "quali": 60, "race": 120}
DATA_DELAY_MIN = 15      # wait this long after a session ends before expecting its data
RACE_OVER_MIN = 180      # race considered over this long after the start
REFRESH_MIN = 55         # minimum gap between refreshes of the same stage
WINDOW_DAYS = 4          # start watching a weekend this many days before its first session
FOCUS_DAYS = 3           # the page switches to the next weekend this many days before its first session
STAGE_RANK = {None: 0, "pre": 1, "quali": 2}


def utcnow():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def sessions_of(ev):
    """{key: (start, estimated end)} in naive UTC, from a FastF1 schedule row."""
    out = {}
    for i in range(1, 6):
        name, start = ev.get(f"Session{i}"), ev.get(f"Session{i}DateUtc")
        key = SESSION_KEYS.get(name)
        if key and start is not None and not pd.isna(start):
            s = pd.Timestamp(start)
            s = s.tz_convert("UTC").tz_localize(None) if s.tzinfo else s
            out[key] = (s, s + pd.Timedelta(minutes=DURATION_MIN[key]))
    return out


def _log(line, max_bytes=1_000_000):
    """Append to data/autopilot.log (rotates once at ~1 MB so it can never fill the disk)."""
    try:
        DATA.mkdir(exist_ok=True)
        if LOGFILE.exists() and LOGFILE.stat().st_size > max_bytes:
            LOGFILE.replace(LOGFILE.with_suffix(".log.1"))
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(f"{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')}Z {line}\n")
    except OSError:
        pass


def _acquire_lock():
    """Cross-process lock (works on Windows too): only one process may run a check at a time."""
    DATA.mkdir(exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - LOCK.stat().st_mtime > LOCK_STALE_MIN * 60:
                    LOCK.unlink()          # crashed run: take over
                    continue
            except OSError:
                pass
            return False
    return False


def _release_lock():
    try:
        LOCK.unlink()
    except OSError:
        pass


def read_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"weekends": {}, "learned": []}


def write_state(state):
    DATA.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, default=str))


def _model(stage):
    f = MODELS / ("ranker.joblib" if stage == "quali" else "ranker_pre.joblib")
    return joblib.load(f) if f.exists() else None


def default_schedule(year):
    import fastf1
    return fastf1.get_event_schedule(year, include_testing=False)


def default_weekend(year, rnd, ev, force):
    """Fresh data for one weekend. `force` bypasses FastF1's cache (a session loaded while it was
    still running can be cached half-empty)."""
    import fastf1
    import collect
    if force:
        try:
            fastf1.Cache.enable_cache("cache", force_renew=True)
        except Exception:
            pass
    try:
        return collect.collect_weekend(year, rnd, ev)
    finally:
        if force:
            try:
                fastf1.Cache.enable_cache("cache")
            except Exception:
                pass


def default_retrain(reason):
    import train
    raw = pd.read_csv(HIST)
    entries = predlog.load()
    return train.train_models(
        raw, train_from=int(os.environ["TRAIN_FROM"]) if os.environ.get("TRAIN_FROM") else None,
        race_weights=predlog.race_weights(entries),
        live_items={"full": predlog.calibration_items(entries, "quali"),
                    "pre": predlog.calibration_items(entries, "pre")},
        reason=reason, verbose=False)


class Autopilot:
    def __init__(self, get_schedule=default_schedule, get_weekend=default_weekend,
                 retrain=default_retrain, now_fn=utcnow):
        self.get_schedule, self.get_weekend, self.retrain, self.now_fn = get_schedule, get_weekend, retrain, now_fn
        self.lock = threading.Lock()
        self._sched = {}

    # ------------------------------------------------------------------ calendar
    def events(self, now):
        """All events around now: list of dict(year, round, name, ev, sessions, first, race)."""
        out = []
        for year in (now.year - 1, now.year, now.year + 1):
            hit = self._sched.get(year)
            if hit is None or time.time() - hit[0] > 6 * 3600:
                try:
                    sched = self.get_schedule(year)
                except Exception:
                    sched = None
                self._sched[year] = hit = (time.time(), sched)
            if hit[1] is None:
                continue
            for _, ev in hit[1].iterrows():
                s = sessions_of(ev)
                if "race" in s:
                    out.append(dict(year=year, round=int(ev["RoundNumber"]), name=ev["EventName"], ev=ev, sessions=s,
                                    first=min(a for a, _ in s.values()), race=s["race"][0]))
        return sorted(out, key=lambda e: e["race"])

    def focus(self, now, events=None):
        """The weekend the page should show: the latest one whose window has opened."""
        events = events if events is not None else self.events(now)
        started = [e for e in events if e["first"] - pd.Timedelta(days=FOCUS_DAYS) <= now]
        return started[-1] if started else (events[0] if events else None)

    @staticmethod
    def expected_stage(sessions, now):
        """Best stage that the clock says should have data by now."""
        def ready(k):
            return k in sessions and now >= sessions[k][1] + pd.Timedelta(minutes=DATA_DELAY_MIN)
        if ready("quali"):
            return "quali"
        if ready("fp1") or ready("fp2") or ready("fp3"):
            return "pre"
        return None

    # ------------------------------------------------------------------ one check
    def tick(self):
        with self.lock:
            if not _acquire_lock():
                _log("skipped: another autopilot process is running")
                return dict(now=str(self.now_fn()), actions=[], error=None, skipped="another process is running")
            try:
                return self._tick()
            finally:
                _release_lock()

    def _tick(self):
        if True:
            now = self.now_fn()
            state = read_state()
            state.setdefault("weekends", {})
            state.setdefault("learned", [])
            actions, error = [], None
            try:
                events = self.events(now)
                if not events:
                    error = "Could not load the race calendar."
                for e in events:
                    key = f"{e['year']}-{e['round']}"
                    if e["first"] - pd.Timedelta(days=WINDOW_DAYS) <= now < e["race"]:
                        self._weekend(e, key, now, state, actions)
                    elif (now >= e["race"] + pd.Timedelta(minutes=RACE_OVER_MIN) and key not in state["learned"]
                          and e["race"] >= now - pd.Timedelta(days=14)):
                        self._finished(e, key, now, state, actions)
            except Exception as ex:  # never let the loop die
                error = f"{type(ex).__name__}: {ex}"
            state.update(last_tick=str(now), last_actions=actions, last_error=error)
            write_state(state)
            for line in actions or ["checked, nothing to do"]:
                _log(line)
            if error:
                _log("ERROR " + error)
            return dict(now=str(now), actions=actions, error=error)

    def _weekend(self, e, key, now, state, actions):
        st = state["weekends"].setdefault(key, {})
        expected = self.expected_stage(e["sessions"], now)
        if expected is None:
            return
        have = st.get("stage")
        force = STAGE_RANK[expected] > STAGE_RANK[have]
        fresh = st.get("snap_at") and (now - pd.Timestamp(st["snap_at"])) < pd.Timedelta(minutes=REFRESH_MIN)
        if not force and have == expected and fresh:
            return
        rows = self.get_weekend(e["year"], e["round"], e["ev"], force)
        if rows is None or rows.empty:
            actions.append(f"{e['name']}: no data yet")
            return
        has_q = rows["quali_pos"].notna().any() if "quali_pos" in rows else False
        has_fp = any(c in rows and rows[c].notna().any() for c in ("fp1_best_gap", "fp2_best_gap", "fp3_best_gap"))
        stage = "quali" if has_q else ("pre" if has_fp else None)
        if stage is None:
            actions.append(f"{e['name']}: sessions finished but no data published yet")
            return
        bundle = _model(stage)
        if bundle is None or not HIST.exists():
            actions.append("no trained model / race data yet: run collect.py and train.py")
            return
        live = extras.apply_live(rows, e["year"], e["round"], e["ev"])
        res = predict_race(pd.read_csv(HIST), rows, bundle)
        if self.now_fn() >= e["race"]:   # the race started while we were working: predictions are locked
            actions.append(f"{e['name']}: race started, prediction not stored")
            return
        entry = predlog.make_entry(
            res, dict(year=e["year"], round=e["round"], event=e["name"], grid_source=live["grid_source"],
                      forecast=live["forecast"]), stage, bundle.get("version"), now)
        if predlog.upsert(entry):
            st.update(stage=stage, snap_at=str(now))
            top = ", ".join(t["driver"] for t in entry["top5"])
            actions.append(f"{e['name']}: stored '{stage}' prediction, top 5 = {top}")

    def _finished(self, e, key, now, state, actions):
        rows = self.get_weekend(e["year"], e["round"], e["ev"], True)
        if rows is None or rows.empty or rows["finish"].notna().sum() < 10:
            actions.append(f"{e['name']}: result not published yet")
            return
        hist = pd.read_csv(HIST) if HIST.exists() else pd.DataFrame()
        src = hist["src"] if "src" in hist else pd.Series("fastf1", index=hist.index)
        mine = (hist["year"] == e["year"]) & (hist["round"] == e["round"]) if len(hist) else pd.Series(dtype=bool)
        new_src = rows["src"].iloc[0] if "src" in rows else "fastf1"
        # keep what we have unless the new rows are richer (FastF1) than the stored ones
        present = bool(len(hist) and mine.any() and ((src[mine] == "fastf1").any() or new_src != "fastf1"))
        if not present:
            if len(hist):
                hist = hist[~((hist["year"] == e["year"]) & (hist["round"] == e["round"]))]
            pd.concat([hist, rows], ignore_index=True).to_csv(HIST, index=False)
            actions.append(f"{e['name']}: result added to the training data")
        scored = predlog.finalize(e["year"], e["round"], rows)
        for s in scored:
            m = s["metrics"]
            actions.append(f"{e['name']} [{s['stage']}]: top-5 hits {m['top5_hits']}/5, exact {m['exact']}/5, score {m['score']}")
        state["learned"].append(key)
        bundle = _model("quali")
        if bundle is None or tuple(bundle.get("trained_through", (0, 0))) < (e["year"], e["round"]):
            rep = self.retrain(f"learned {e['name']} {e['year']}")
            state["last_retrain"] = dict(time=str(now), event=e["name"], promoted=rep["promoted"],
                                         version=rep["version"], gate=rep["reason"])
            actions.append(f"self-training: model v{rep['version']}, {rep['reason']}")


# ---------------------------------------------------------------------- payload for the web page
def _iso(t):
    return pd.Timestamp(t).isoformat() + "Z"


def live_payload(ap, year=None, rnd=None):
    now = ap.now_fn()
    events = ap.events(now)
    if not events:
        return dict(error="Could not load the race calendar (offline?).")
    e = next((x for x in events if x["year"] == year and x["round"] == rnd), None) if year else ap.focus(now, events)
    if e is None:
        return dict(error="Unknown race.")
    entries = predlog.load()
    mine = {s["stage"]: s for s in predlog.for_race(e["year"], e["round"], entries)}

    actual = next((s["actual"] for s in mine.values() if s.get("actual")), None)
    if actual is None and HIST.exists():  # raced before the log existed: still show the real top 5
        h = pd.read_csv(HIST)
        h = h[(h["year"] == e["year"]) & (h["round"] == e["round"]) & h["finish"].notna()].sort_values("finish")
        if len(h) >= 5:
            actual = dict(order=h["driver"].tolist()[:10], top5=[
                dict(pos=i + 1, driver=r["driver"], name=r.get("full_name", r["driver"]), team=r["team"])
                for i, (_, r) in enumerate(h.head(5).iterrows())])

    sess = [dict(key=k, label=SESSION_LABEL[k], start=_iso(a), done=bool(now >= b + pd.Timedelta(minutes=DATA_DELAY_MIN)))
            for k, (a, b) in sorted(e["sessions"].items(), key=lambda kv: kv[1][0])]
    if actual:
        status = "finished"
    elif now >= e["race"]:
        status = "race_started"
    elif now >= e["first"]:
        status = "weekend"
    else:
        status = "upcoming"

    state = read_state()
    version = json.loads((MODELS / "version.json").read_text()) if (MODELS / "version.json").exists() else None
    tlog = []
    if (DATA / "training_log.jsonl").exists():
        for line in (DATA / "training_log.jsonl").read_text().splitlines()[-5:]:
            try:
                tlog.append(json.loads(line))
            except ValueError:
                pass
    return dict(
        now=_iso(now),
        event=dict(year=e["year"], round=e["round"], name=e["name"], race_start=_iso(e["race"]), status=status, sessions=sess),
        predictions=mine, actual=actual,
        season=predlog.summary(entries),
        races=[dict(year=x["year"], round=x["round"], name=x["name"]) for x in events if x["year"] == now.year],
        autopilot=dict(enabled=os.environ.get("AUTOPILOT", "1") != "0", last_tick=state.get("last_tick"),
                       last_actions=state.get("last_actions", []), last_error=state.get("last_error"),
                       interval_min=int(os.environ.get("AUTOPILOT_INTERVAL_MIN", "20")),
                       model=version, last_retrain=state.get("last_retrain"), training_log=tlog))


def run_loop(ap, interval_min):
    while True:
        try:
            ap.tick()
        except Exception:
            pass
        time.sleep(interval_min * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--interval", type=int, default=20, help="minutes between checks with --loop")
    a = p.parse_args()
    if a.status:
        print(json.dumps(read_state(), indent=1))
        return
    ap = Autopilot()
    if a.loop:
        run_loop(ap, a.interval)
    else:
        r = ap.tick()
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
