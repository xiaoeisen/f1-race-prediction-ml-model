# Codebase evaluation against the F1 best-practice checklist

Method: every checklist item was checked in the code, and anything that could change accuracy was
measured with a walk-forward test on **189 real races (2018-2026)** from the open f1db dataset
(`python benchmark.py`). Caveat: f1db has results, qualifying, practice times, grid and pit stops only,
so lap-level features (tyre-aware race pace, safety cars, sectors, weather) could not be tested there.
"Neutral" below means: no difference beyond the test's noise (about +/-0.002 to 0.01).

## 1. Problem formulation
| Item | Result |
|---|---|
| Learning-to-rank, not unconstrained regression | PASS. LightGBM LambdaRank + Plackett-Luce; each race is ranked once, so two drivers can never share a position. |
| Beats the starting-grid baseline | PARTLY. Whole-field ordering: Spearman 0.667 vs 0.625 (clear). Top-5 overlap 0.746 vs 0.730 (+0.016, suggestive, ~1.5-2 standard errors). Winner 54.0% vs 54.0%, podium 0.695 vs 0.688, exact top-5 order 0.292 vs 0.297: tied, not better. |

Note: the checklist says the grid explains 60-70% of outcome variance. In this data the grid's rank correlation is
0.625 (r-squared about 0.39) and pole wins 54% of races, so "grid is hard to beat" holds, the exact figure doesn't.

## 2. Data integrity and leakage
| Item | Result |
|---|---|
| Formation-lap cutoff | PASS. Every feature is frozen at lights-out (tested by scrambling all post-race values; predictions are locked at race start). **Weather was a leak and is now fixed:** finished races train on the qualifying-session weather (`weather='pre'`), live predictions use the forecast; the observed race weather is only used for history (past wet races). `weather='race'` restores the old behaviour. The model hasn't been re-benchmarked with lap-level weather data (none in f1db). |
| Grid-penalty separation | PASS. `quali_pos`, `grid`, `grid_pen = grid - quali_pos` and `grid_log` are separate; the official grid (OpenF1) or `penalties.csv` feeds `grid`. |
| DNF factorisation | PASS (added). Retirement reasons are split into mechanical (engine, gearbox, hydraulics, ...) and incident (collision, accident, spun). Driver error-proneness and car unreliability feed the race simulation's retirement risk. As model features the split was neutral to slightly negative, so they are not in the model's feature list. |

## 3. Feature engineering
| Item | Result |
|---|---|
| Regulation / era flagging | PASS (added). Team ratings reset 70% at regulation changes (2014, 2017, 2022, 2026), `era_age` computed, short rolling windows (5/10 races), Elo shrinks at every season start. Measured: neutral. |
| Teammate baseline delta | PASS. This-weekend qualifying and long-run deltas, plus (added) teammate-relative history over 10 races. Measured: neutral, kept. |
| Track profiles | PARTLY. Street/road, pit-lane loss, tyre wear, altitude, downforce level in `circuits.csv`. **These values are my rough estimates, not measured data**; edit them. altitude/downforce/era_age were slightly negative (-0.002 +/- 0.001) so they are computed but not used by default. |

## 4. Validation
| Item | Result |
|---|---|
| Time-ordered validation | PASS. No random K-fold or shuffling anywhere in the code (searched). `train.py` tests on the latest season; `benchmark.py` walks forward over every season. |
| Roster changes / rookies | PASS (added). Missing driver form is filled from the team's form (not zero); the linear part imputes with the race median; `drv_races` tells the model how little history exists. Measured: neutral. |

## 5. Metrics (189 real races, walk-forward)
| Metric | Grid baseline | Previous model | Current model |
|---|---|---|---|
| NDCG@5 (added) | 0.878 | 0.876 | 0.883 |
| MAE, positions per driver | 3.50 | 3.42 | 3.32 |
| MAE, finishers only | 2.99 | 2.88 | 2.76 |
| Top-1 (winner) accuracy | 54.0% | 47.6% | 54.0% |

**The MAE < 2.5 target is not met** (3.3 overall, 2.8 for finishers). About 15% of starters retire in this data and
the grid baseline itself scores 3.5, so the target looks unrealistic for this problem rather than fixable by tuning.

## Added after the first audit
Automated tests (`pytest`, 34 tests), the weather fix above, rate limiting, `/api/health`, a cross-process autopilot lock, a rotating log, and a hold-out set of the latest 12 races (a fixed season gave a coin-flip safety check right after round 1). Street/road circuit types were checked against f1db: all 33 circuits match; lap length and corner counts now come from f1db.

## What changed because of this evaluation
- NDCG@5/@10, exact-order and finisher-only MAE added everywhere (`metrics.py`, shown on the page).
- The model is now an ensemble (trees + Plackett-Luce + grid prior); the old single tree model was clearly worse than the grid on winner (47.6%), podium and exact order.
- Probabilities: the old temperature was fitted on one season and was overconfident (a "98%" podium chance came true 79% of the time). Now pooled over three out-of-sample seasons plus retirement risk: Brier scores improved win 0.0333 to 0.0310, podium 0.0702 to 0.0658, top-5 0.0965 to 0.0908; the same bucket is now "93%" vs 85% actual.
- Speed: feature building 2.7-3x faster (identical output), a prediction request 1.29 s to 0.45 s, a full retrain 10.5 s to 8.8 s, and `collect.py --f1db 2014` loads 107 races in about 3 seconds.

## Open risks
1. Weather: training uses Saturday's weather and live use a race-hour forecast, so a dry-Saturday/wet-Sunday race is the hard case.
2. No lap-level data in the benchmark, so the gain from tyre-aware pace, safety cars and sectors is unmeasured until you run it on FastF1 data.
3. A 189-race sample can't separate improvements smaller than about +/-0.01 top-5 overlap.
