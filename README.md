# F1 Race Prediction

Predicts the finishing order of an F1 race from FP1/FP2/FP3 pace, qualifying, weather,
driver skill (Elo, form, race craft, wet/circuit tendencies), team form and circuit character.
Model: ensemble of LightGBM LambdaRank + Plackett-Luce + grid prior (orders the field per race), Monte Carlo with retirement risk for win/podium/top-10 %.

## Setup
    pip install -r runtime-requirements.txt

## Run
    python collect.py --check                     # quick test: downloads ONE recent race and prints it
    python collect.py --start 2022                # full download (slow the first time, cached after)
    python train.py                               # backtests on the latest season, saves models/ranker.joblib
    python predict.py --year 2025 --round 20      # run after quali; use --grid to apply penalties

Data comes from FastF1 (real F1 timing + results). Default start is 2022 (current car regs); use `--start 2018` for more history, but expect a long first download (an hour or more per few seasons). It resumes where it stopped if interrupted.

## What the model looks at
- This weekend: grid, qualifying (position, gap to pole, sector gaps, top speed), FP1/FP2/FP3 (positions, best-lap gaps,
  tyre-aware long-run pace), weather, grid penalties.
- Driver: Elo, recent form, race craft, retirements, wet-weather record, circuit record, true race pace and pace vs
  teammate (tyre- and fuel-adjusted), first-lap gains.
- Team: Elo, form, reliability, true race pace and its recent trend (a proxy for car upgrades).
- Circuit: shuffling, retirements, safety car / red flag rates, plus street / pit-loss / tyre-wear facts from `circuits.csv`.
- Race-day weather forecast (Open-Meteo) for upcoming races, and penalties from `penalties.csv`.

**After updating from an older version run `python collect.py --refresh` once** (new columns), then `python train.py`.

`circuits.csv` holds approximate starting values (pit-lane time loss, tyre wear 1-3, street circuit 0/1, coordinates).
Edit them if you know better. `train.py` prints any circuit names it couldn't match so you can add rows.

`penalties.csv` (year,round,driver,places): list grid penalties before the race, e.g. `2025,20,VER,10`
(places dropped; 99 = back of grid / pit lane). They are applied to upcoming races only.

## Fast start with real data (seconds, not hours)
    python collect.py --f1db 2014        # all races since 2014: results, qualifying, practice, grid, pit stops
    python train.py
    python collect.py --start 2022       # optional: upgrade recent races with FastF1 lap data (slow, one-off)
f1db has no tyre/lap/weather data, so those columns stay empty for the bulk-loaded races; FastF1 rows replace them.

## Is the model any good? Measure it
    python benchmark.py                  # walk-forward on past seasons vs the "finish where you start" baseline
See EVALUATION.md for the checklist audit and measured numbers (189 real races): the ensemble ranks the whole
field clearly better than the grid (Spearman 0.667 vs 0.625) but ties it on winner / podium / exact order.

## Tests
    pip install -r test-requirements.txt
    pytest                    # ~45 s: features/leakage, metrics, prediction log, data sources (mocked), model, autopilot weekend, server
    pytest -m "not slow"      # skip the tests that train models
The tests use your real `data/races.csv` (and `models/` for the server tests). They include a leakage test that scrambles
every post-race value and checks no pre-race feature changes, and a simulated race weekend (practice -> quali -> locked ->
result -> scored -> retrained). Without a trained model the server tests are skipped.

## Running it for other people
The server is meant for a small audience, not heavy public traffic. It does not require login for normal prediction requests. Keep it on localhost for personal use; public deployment needs HTTPS, a firewall, and an authentication layer at a reverse proxy. Settings (environment variables):
`RATE_LIMIT_PER_MIN` (default 60 per client, 0 = off), `TRUST_PROXY=1` (read the client IP from X-Forwarded-For when
behind a trusted proxy that overwrites caller-supplied X-Forwarded-For), `MAX_CONCURRENT_PREDICTIONS` (default 4), `ADMIN_TOKEN` (enables `POST /api/autopilot/run`).
Monitoring: point an uptime checker at `GET /api/health` (200 = data, model and autopilot all fine; 503 + reasons otherwise).
The autopilot writes `data/autopilot.log` (rotates at ~1 MB) and uses a lock file, so two server processes can't both run
it; still, run ONE process.

## Race Center: automatic updates, top-5 predictions, accuracy, self-training
Start the web app and leave it running. A background autopilot checks every 20 minutes:

1. **Auto-update:** when FP1/FP2 data is published it stores a top-5 prediction (practice-only model);
   when qualifying data (and the official starting grid, if OpenF1 has it) is published it stores a new
   one (full model). Predictions are **locked when the race starts**.
2. **Predicted vs actual:** the Race Center tab shows the stored top 5 next to the real top 5 once the race
   is over, with a mark per position (right position / in the top 5 / missed).
3. **Accuracy indicator:** top-5 accuracy, exact positions, winner, a 0-100 score, and the same numbers
   for the naive guess "drivers finish where they started", plus a per-race history.
4. **Self-training:** after each race the real result is added to the training data, the models are
   retrained, and the prediction log is used to (a) weight races the model got wrong up to 1.5x and
   (b) recalibrate the win/podium percentages. A retrained model only goes live if it is not clearly
   worse than the "finish = grid" baseline on the held-out season; otherwise the old model stays and
   the rejection is logged. Old models are archived in `models/archive/`.

Run it without the web app: `python autopilot.py --once` (cron / Task Scheduler) or `--loop`.
Settings (environment variables): `AUTOPILOT=0` to turn it off, `AUTOPILOT_INTERVAL_MIN`,
`TRAIN_FROM` (e.g. 2022), `ADMIN_TOKEN` (enables `POST /api/autopilot/run` with header `x-admin-token`).
Use ONE server process (several workers would each run the autopilot). Needs `python collect.py` and
`python train.py` to have been run once. Files it writes: `data/predictions.jsonl`, `data/autopilot_state.json`,
`data/training_log.jsonl`, `models/version.json`.

Honest expectations: about 24 races a year means the accuracy indicator is noisy for a long time, and one
extra race per week changes the model only slightly. The weighting heuristics are my design choices, not
proven improvements: the accuracy history and the baseline comparison are how you check whether they help.

## Web app (frontend)
    uvicorn server:app --host 127.0.0.1 --port 8000 # open http://127.0.0.1:8000

Keep the app on localhost for personal use. Exposing it on a network gives callers access to prediction endpoints;
use a firewall and an HTTPS reverse proxy before binding to 0.0.0.0. Set TRUST_PROXY=1 only when that proxy
removes caller-supplied X-Forwarded-For and writes its own. API docs are disabled by default; set ENABLE_API_DOCS=1
for local development only.

Pick a season and race, press Predict, and you get the predicted order with win / podium / top-10 %.
For finished races it shows the actual result next to the prediction. What-ifs: edit a grid number
(penalties) or force rain or dry, then press Predict again. The page also shows how the model scores
against simple baselines on the held-out season.

Run `collect.py` and `train.py` first (the app needs `data/races.csv` and the files in `models/`).
Upcoming weekends are fetched live from FastF1 and cached for 30 minutes.

To share it with anyone on the internet, deploy the folder to a Python host (Render, Railway, Fly.io,
a VPS). Put the service behind an HTTPS reverse proxy, firewall, and authentication layer; the app does not authenticate regular prediction requests. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`. The host needs
`data/races.csv` and `models/*.joblib`; generate them by running `collect.py` and `train.py` on the host.
The FastF1 cache needs enough disk space. Treat model .joblib files as executable input: only load artifacts from a trusted source.

## Files
- collect.py: FastF1 -> data/races.csv (one row per driver per race)
- features.py: leak-free features (all history uses shift(1))
- train.py: time-based split, compares against grid-order and Elo baselines; pooled probability calibration
- model.py: the ranking ensemble (trees + Plackett-Luce + grid prior)
- metrics.py, benchmark.py: shared metrics (incl. NDCG) and the walk-forward benchmark
- EVALUATION.md: audit against the best-practice checklist
- predict.py: ranking + probabilities for one weekend
- autopilot.py: auto-update, store predictions, score them, self-train
- predlog.py: prediction log and accuracy metrics
- sources/, sources_check.py: Jolpica / OpenF1 / API-Sports / Hyprace adapters and a connection test
- server.py: FastAPI backend (API + serves the frontend)
- web/: index.html, style.css, app.js (no build step)
