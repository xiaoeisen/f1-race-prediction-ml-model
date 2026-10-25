"""Web app: JSON API + static frontend.

    uvicorn server:app --host 127.0.0.1 --port 8000

Bind to 0.0.0.0 only behind a firewall and HTTPS reverse proxy.

Needs data/races.csv (python collect.py) and the models (python train.py) first.
Run it from the project folder.
"""
import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # collect.py uses relative paths (data/, cache/)

import fastf1
import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import autopilot
import collect
import extras
import predlog
from predict import predict_race

HIST_CSV = ROOT / "data" / "races.csv"
FILES = {"full": (ROOT / "models" / "ranker.joblib", ROOT / "models" / "ranker_holdout.joblib"),
         "pre": (ROOT / "models" / "ranker_pre.joblib", ROOT / "models" / "ranker_pre_holdout.joblib")}
METRICS = ROOT / "models" / "metrics.json"
WEEKEND_TTL = 30 * 60  # re-fetch a live weekend after 30 min (quali may have finished)

AP = autopilot.Autopilot()


@asynccontextmanager
async def lifespan(app):
    """Runs the autopilot (auto-update, predict, score, self-train) in the background.
    Disable with AUTOPILOT=0; change the check interval with AUTOPILOT_INTERVAL_MIN (default 20)."""
    if os.environ.get("AUTOPILOT", "1") != "0":
        interval = int(os.environ.get("AUTOPILOT_INTERVAL_MIN", "20"))

        def loop():
            time.sleep(15)
            autopilot.run_loop(AP, interval)
        threading.Thread(target=loop, daemon=True, name="autopilot").start()
    yield


DOCS_ENABLED = os.environ.get("ENABLE_API_DOCS", "0") == "1"
app = FastAPI(
    title="F1 Race Prediction",
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
    lifespan=lifespan,
)

STARTED = time.time()
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))     # requests per minute per client, 0 = off
MAX_TRACKED_RATE_CLIENTS = 5000
MAX_API_BODY_BYTES = 16 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; "
        "form-action 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https://media.formula1.com https://flagcdn.com; connect-src 'self'"
    ),
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
}
TRUST_PROXY = os.environ.get("TRUST_PROXY", "0") == "1"          # read the client IP from X-Forwarded-For
_hits = defaultdict(deque)
_hits_lock = threading.Lock()
_predict_slots = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_PREDICTIONS", "4")))


def _secure_response(response, path):
    response.headers.update(SECURITY_HEADERS)
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _client_ip(request):
    if TRUST_PROXY and request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RequestBodyLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or not scope.get("path", "").startswith("/api/")
                or scope.get("method") not in {"POST", "PUT", "PATCH"}):
            await self.app(scope, receive, send)
            return

        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                response = JSONResponse(
                    {"detail": "Request body too large."},
                    status_code=413,
                    headers={
                        "Cache-Control": "no-store",
                        **SECURITY_HEADERS,
                    },
                )
                await response(scope, receive, send)
                return
            chunks.append(message)
            if not message.get("more_body", False):
                break

        pending = iter(chunks)

        async def replay_receive():
            try:
                return next(pending)
            except StopIteration:
                return await receive()

        await self.app(scope, replay_receive, send)


app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_API_BODY_BYTES)


@app.middleware("http")
async def guard(request, call_next):
    path = request.url.path
    if RATE_LIMIT and path.startswith("/api/") and path != "/api/health":
        now, ip = time.time(), _client_ip(request)
        with _hits_lock:
            if ip not in _hits and len(_hits) >= MAX_TRACKED_RATE_CLIENTS:
                stale = [k for k, v in _hits.items() if not v or now - v[-1] > 60]
                for k in stale:
                    del _hits[k]
                    if len(_hits) < MAX_TRACKED_RATE_CLIENTS:
                        break
                if len(_hits) >= MAX_TRACKED_RATE_CLIENTS:
                    return _secure_response(
                        JSONResponse({"detail": "Server is busy. Please try again shortly."},
                                     status_code=503, headers={"Retry-After": "60"}),
                        path,
                    )
            q = _hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_LIMIT:
                retry = max(1, int(60 - (now - q[0])))
                return _secure_response(
                    JSONResponse({"detail": "Too many requests. Please slow down."}, status_code=429,
                                 headers={"Retry-After": str(retry)}),
                    path,
                )
            q.append(now)
    resp = await call_next(request)
    return _secure_response(resp, path)


# ---------------------------------------------------------------- cached loaders
_files = {}
_sched = {}
_weekends = {}
_fetch_lock = threading.Lock()  # one FastF1 download at a time


def _cached(path, loader):
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    hit = _files.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    obj = loader(path)
    _files[path] = (mtime, obj)
    return obj


def history():
    h = _cached(HIST_CSV, pd.read_csv)
    if h is None:
        raise HTTPException(503, "No race data yet. Run: python collect.py")
    return h


def schedule(year):
    if year not in _sched:
        try:
            s = fastf1.get_event_schedule(year, include_testing=False)
        except Exception:
            raise HTTPException(502, "Could not load the race calendar right now.")
        _sched[year] = s.set_index("RoundNumber")
    return _sched[year]


def pick_model(year, rnd, stage="full"):
    """Returns (bundle, in_sample). Races the final model trained on are shown with the
    hold-out model when possible so backtests stay honest. stage 'pre' = before qualifying."""
    final_f, hold_f = FILES[stage]
    final = _cached(final_f, joblib.load)
    if final is None and stage == "pre":  # models from before the stage split
        final_f, hold_f = FILES["full"]
        final = _cached(final_f, joblib.load)
    if final is None:
        raise HTTPException(503, "Model not trained yet. Run: python train.py")
    hold = _cached(hold_f, joblib.load)
    seen = (year, rnd) <= tuple(final.get("trained_through", (0, 0)))
    if seen:
        start = tuple(hold.get("test_from") or (hold.get("test_year", 10 ** 9), 0)) if hold else None
        if hold and (year, rnd) >= start:
            return hold, False                     # the hold-out model never saw these races
        return final, True
    return final, False


def _session_data_missing(rows):
    required = ["quali_pos", "fp1_best_gap", "q_best_time_s", "q_time_segment_valid", "fp3_best_time_s"]
    sprint = ("is_sprint" in rows and
              pd.to_numeric(rows["is_sprint"], errors="coerce").fillna(0).max() > 0)
    if not sprint:
        required.extend(("fp2_best_gap", "fp3_best_gap"))
    current_season = ("year" in rows and
                      pd.to_numeric(rows["year"], errors="coerce").max() == pd.Timestamp.now().year)
    if (current_season and "finish" in rows and
            pd.to_numeric(rows["finish"], errors="coerce").notna().any()):
        required.append("race_pace_gap")
    return any(column not in rows or
               not pd.to_numeric(rows[column], errors="coerce").notna().any()
               for column in required)


def _merge_session_data(saved, fetched):
    if fetched is None or fetched.empty or "driver" not in fetched:
        return saved.copy()
    base = saved.set_index("driver", drop=False).copy()
    fresh = fetched.drop_duplicates("driver", keep="last").set_index("driver")
    session_columns = (
        "quali_pos", "q_best_gap", "fp1_best_gap", "fp2_best_gap", "fp3_best_gap",
        "q_best_time_s", "q_gap_s", "q_time_segment_valid", "quali_laps",
        "fp1_best_time_s", "fp1_gap_s", "fp1_laps", "fp2_best_time_s", "fp2_gap_s", "fp2_laps",
        "fp3_best_time_s", "fp3_gap_s", "fp3_laps", "fp1_long_gap", "fp2_long_gap",
        "fp1_deg_per10", "fp2_deg_per10", "fp3_deg_per10",
        "race_pace_gap", "race_deg_per10", "race_best_lap_s", "race_avg_clean_lap_s", "race_laps",
    )
    for column in session_columns:
        if column not in fresh:
            continue
        incoming = fresh[column].reindex(base.index)
        if column not in base:
            base[column] = incoming
        else:
            base[column] = base[column].where(base[column].notna(), incoming)
    return base.reset_index(drop=True)


def weekend_rows(year, rnd):
    h = history()
    done = h[(h["year"] == year) & (h["round"] == rnd)]
    key = (year, rnd)
    hit = _weekends.get(key)
    if hit and time.time() - hit[0] < WEEKEND_TTL:
        return hit[1].copy()
    if len(done) and not _session_data_missing(done):
        return done.copy()

    with _fetch_lock:
        try:
            df = collect.collect_weekend(year, rnd, schedule(year).loc[rnd])
        except HTTPException:
            raise
        except Exception:
            if len(done):
                _weekends[key] = (time.time(), done)
                return done.copy()
            raise HTTPException(502, "Could not fetch session data from FastF1 right now.")
    if df.empty:
        if len(done):
            _weekends[key] = (time.time(), done)
            return done.copy()
        raise HTTPException(404, "No session data is available for that weekend yet.")
    if len(done):
        df = _merge_session_data(done, df)
    _weekends[key] = (time.time(), df)
    return df.copy()

# ---------------------------------------------------------------- API
class PredictReq(BaseModel):
    year: int = Field(ge=2018, le=2100)
    round: int = Field(ge=1, le=30)
    grid: Optional[Dict[str, int]] = Field(default=None, max_length=30)  # bounded what-if input
    rain: Optional[bool] = None            # what-if: force wet/dry

    class Config:
        extra = "forbid"


@app.get("/api/seasons")
def seasons():
    ys = sorted(set(history()["year"].astype(int)) | {pd.Timestamp.now().year})
    return {"seasons": ys}


@app.get("/api/races")
def races(year: int = Query(ge=2018, le=2100)):
    sched = schedule(year)
    today = pd.Timestamp.now().normalize()
    out = []
    for rnd, ev in sched.iterrows():
        date = pd.Timestamp(ev["EventDate"])
        out.append(dict(round=int(rnd), name=ev["EventName"], country=ev.get("Country", ""),
                        date=date.date().isoformat(), completed=bool(date < today)))
    return {"year": year, "races": out}


@app.get("/api/standings")
def standings(year: int = Query(ge=1950, le=2100)):
    """Championship points standings for a season, based on stored completed race results."""
    rows = history()
    rows = rows[(rows["year"] == year) & rows["finish"].notna()].copy()
    if rows.empty:
        return {"year": year, "through_round": None, "drivers": [], "teams": []}

    rows["points"] = pd.to_numeric(rows["points"], errors="coerce").fillna(0.0)
    team_aliases = {
        "Red Bull Racing": "Red Bull", "RB": "Racing Bulls", "RB F1 Team": "Racing Bulls",
        "Alpine F1 Team": "Alpine", "Haas F1 Team": "Haas", "Cadillac F1 Team": "Cadillac",
    }
    rows["team"] = rows["team"].replace(team_aliases)
    rows = rows.sort_values(["round", "driver"])
    through_round = int(rows["round"].max())
    drivers = (rows.groupby("driver", as_index=False)
        .agg(full_name=("full_name", "last"), team=("team", "last"), points=("points", "sum"),
             wins=("finish", lambda finish: int((finish == 1).sum())))
        .sort_values(["points", "wins", "full_name"], ascending=[False, False, True]))
    teams = (rows.groupby("team", as_index=False)
        .agg(points=("points", "sum"), wins=("finish", lambda finish: int((finish == 1).sum())))
        .sort_values(["points", "wins", "team"], ascending=[False, False, True]))
    driver_rows = [dict(position=i, **row) for i, row in enumerate(drivers.to_dict("records"), 1)]
    team_rows = [dict(position=i, **row) for i, row in enumerate(teams.to_dict("records"), 1)]
    for row in driver_rows + team_rows:
        row["points"] = round(float(row["points"]), 1)
        row["wins"] = int(row["wins"])
    return {"year": year, "through_round": through_round, "drivers": driver_rows, "teams": team_rows}


@app.get("/api/model")
def model_info():
    if not METRICS.exists():
        return {"available": False}
    info = json.loads(METRICS.read_text())
    info["available"] = True
    return info


@app.post("/api/predict")
def predict(req: PredictReq):
    sched = schedule(req.year)
    if req.round not in sched.index:
        raise HTTPException(404, "That round doesn't exist for this season.")
    rows = weekend_rows(req.year, req.round)
    ev = sched.loc[req.round]
    live = extras.apply_live(rows, req.year, req.round, ev)

    if req.grid:
        unknown = set(req.grid) - set(rows["driver"])
        if unknown:
            raise HTTPException(400, f"Unknown drivers: {', '.join(sorted(unknown))}")
        if any(not 1 <= v <= 30 for v in req.grid.values()):
            raise HTTPException(400, "Grid positions must be between 1 and 30.")
        key = rows["driver"].map(req.grid).fillna(rows["grid"]).fillna(99)
        rows["grid"] = key.rank(method="first")  # renumber 1..N so there are no duplicates
    if req.rain is not None:
        rows["rain"] = float(req.rain)

    has_quali = bool(rows["quali_pos"].notna().any())
    completed = bool(rows["finish"].notna().any())
    stage = "full" if (completed or has_quali) else "pre"   # before qualifying: practice-only model
    bundle, in_sample = pick_model(req.year, req.round, stage)
    if not _predict_slots.acquire(timeout=10):
        raise HTTPException(503, "The server is busy. Try again in a moment.")
    try:
        res = predict_race(history(), rows, bundle)
        display_round = [column for column in ("exp_pos", "win_%", "podium_%", "top10_%", "score")
                         if column in res]
        res[display_round] = res[display_round].round(2)
    finally:
        _predict_slots.release()

    meta = dict(
        year=req.year, round=req.round, event=sched.loc[req.round, "EventName"],
        completed=bool(rows["finish"].notna().any()),
        in_sample=bool(in_sample and rows["finish"].notna().any()),
        quali_available=has_quali, stage=stage,
        what_if=dict(grid=bool(req.grid), rain=req.rain),
        forecast=live["forecast"], penalties_applied=live["penalties"], grid_source=live["grid_source"],
        trained_through=bundle.get("trained_through"),
    )
    return {"meta": meta, "rows": json.loads(res.to_json(orient="records"))}


@app.get("/api/health")
def health():
    """For uptime monitors: 200 when data + model are present and the autopilot is alive, else 503."""
    problems = []
    info = dict(uptime_s=int(time.time() - STARTED))
    try:
        h = history()
        info["data"] = dict(rows=int(len(h)), latest=[int(h["year"].max()), int(h[h["year"] == h["year"].max()]["round"].max())])
    except HTTPException:
        problems.append("no race data (run collect.py)")
    vf = ROOT / "models" / "version.json"
    if vf.exists() and FILES["full"][0].exists():
        info["model"] = json.loads(vf.read_text())
    else:
        problems.append("no trained model (run train.py)")
    state = autopilot.read_state()
    enabled = os.environ.get("AUTOPILOT", "1") != "0"
    interval = int(os.environ.get("AUTOPILOT_INTERVAL_MIN", "20"))
    last = state.get("last_tick")
    age = None if not last else round((pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timestamp(last)).total_seconds() / 60, 1)
    info["autopilot"] = dict(enabled=enabled, last_tick=last, minutes_since=age, last_error=state.get("last_error"))
    if enabled and time.time() - STARTED > 3 * interval * 60 and (age is None or age > max(3 * interval, 90)):
        problems.append(f"autopilot has not run for {age if age is not None else 'ever'} minutes")
    info["ok"], info["problems"] = not problems, problems
    return JSONResponse(info, status_code=200 if not problems else 503)


@app.get("/api/live")
def live(year: Optional[int] = None, round: Optional[int] = None):
    """Race Center: the current/next weekend, stored predictions, real result, accuracy, self-training."""
    return autopilot.live_payload(AP, year, round)


@app.get("/api/accuracy")
def accuracy():
    return predlog.summary(predlog.load())


@app.post("/api/autopilot/run")
def autopilot_run(x_admin_token: Optional[str] = Header(default=None)):
    """Trigger one check now. Disabled unless the ADMIN_TOKEN environment variable is set."""
    token = os.environ.get("ADMIN_TOKEN")
    if not token or not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), token.encode()):
        raise HTTPException(403, "Not allowed.")
    return AP.tick()


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

