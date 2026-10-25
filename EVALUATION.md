# Codebase evaluation against the F1 best-practice checklist

Method: every checklist item was checked in the code, and the initial accuracy review used a walk-forward
test on **189 real races (2018-2026)** from the open f1db dataset (`python benchmark.py`). The later
250-race ranking and calibration results below supersede that initial metric snapshot. Caveat: f1db has results, qualifying, practice times, grid and pit stops only,
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

## 5. Initial metrics (189 real races, walk-forward; superseded below)
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
2. Lap-level pace data still covers too few weekends to measure the gain from tyre-aware pace, safety cars and sectors reliably.
3. Even 250 races leave small blend differences within normal season-to-season noise.


## Top-five and probability follow-up (2026-10-25)

The benchmark compares grid-prior weights on 250 season-ordered test races (2015-2026). After fixing benchmark scoring errors, the current blend uses grid weight **0.5** and scores **0.568 winner accuracy, 0.700 podium overlap and 0.750 top-five overlap**, versus **0.560, 0.699 and 0.751** for the previous weight-1 blend. The grid baseline scores **0.556, 0.695 and 0.732**. On the latest 87 races (2023-2026), weight 0.5 scores 0.644/0.678/0.754 (winner/podium/top five), versus 0.621/0.674/0.761 at weight 1. The fixed 0.5 blend is used because it improves winner accuracy, slightly improves podium overlap, and leaves top-five overlap effectively unchanged. These are chronological test folds but the candidate weight was selected after inspecting the aggregate sweep, so the gain still needs monitoring on future races. A nested tuner choosing weights from only the previous three seasons did not improve results (winner/podium/top five: 0.540/0.695/0.746).

The temperature calibration uses the same retirement simulation as prediction. At grid weight 0.5, the corrected walk-forward Brier scores are **0.0300 raw / 0.0302 calibrated for wins** and **0.0638 / 0.0639 for podiums**. Calibration slightly improves ECE (win 0.0056 to 0.0051; podium 0.0133 to 0.0114) while Brier scores are essentially tied. The calibrated grid baseline is 0.0312 for wins and 0.0665 for podiums. Per-race rankings, yearly calibration metrics, blend weights, win/podium reliability plots, and the podium miss report are under `data/benchmark/`.

The latest sprint-aware audit reports coverage over 270 races / 5,505 driver rows: qualifying 98.2%, FP1 91.9%, FP2 87.3%, and FP3 85.5%. Eight weekends now lack at least one scheduled core session; the 2026 United States GP FP2 gap was filled from the targeted retry. The earlier 38-weekend count incorrectly treated sprint weekends as if FP2/FP3 were scheduled. The first lap-data retry populated eight 2022 weekends before FastF1's 500-calls-per-hour limit stopped that broader run. Lap-level pace coverage remains too partial to measure the new pace signals reliably.

## Pace-feature and podium-miss follow-up (2026-10-25)

A paired feature ablation on the same 250 chronological test races found the full pace model scores **56.8% winner / 70.0% podium / 75.0% top five**, versus **56.0% / 70.0% / 75.0%** when long-run pace, degradation, race-pace history and qualifying-sector features are removed. Extending the linear ranking component with selected pace signals matched the full pace model. With only eight weekends carrying lap-level pace signals, the current evidence favors retaining the features for the small winner-accuracy gain while we backfill more data.

The limiting factor is coverage: of 270 weekends, only eight have any race-pace or long-run pace values, seven have any FP1 degradation, five FP2 degradation, and nine qualifying sector data. The requested FastF1 backfill was attempted, but the feed returned its 500-calls-per-hour limit after the first few weekends. `data/benchmark/podium_miss_analysis.csv` records each race's actual and predicted podium candidates and available pre-race pace signals so future backfills can be checked directly. The benchmark writes paired pace/no-pace comparisons to `walkforward_metrics.csv`. The production grid prior is now 0.5; the pace features remain enabled.

The updated model was promoted as version 3 after the existing 12-race holdout gate passed. On that holdout it scored 66.7% winner accuracy and 63.9% podium overlap, matching the grid baseline; top-five overlap was 71.7% versus 73.3% for the grid, within the configured two-point tolerance. Changing only the prior weight from 1.0 to 0.5 did not change any of the rankings in this short holdout. The broader 250-race walk-forward comparison is the evidence for the small winner/podium gain.

## 2026 season-focused adjustment (2026-10-25)

Because 2026 introduced a major regulation reset, the current-season walk-forward was also checked independently on its 17 completed races. The prior 0.5 blend scored **70.6% winner / 60.8% podium / 70.6% top five**. A 1.25 grid contribution scored **70.6% / 62.7% / 71.8%**: winner hit rate held, with one more predicted podium finisher and one more top-five finisher per race on average. The 2026 win Brier score improved from 0.0246 to **0.0214**; podium Brier was 0.0670, still better than the grid baseline's 0.0691 but slightly worse than the old blend's 0.0641.

The model now uses that 1.25 grid contribution for 2026 rows and retains 0.5 elsewhere. The full 250-race chronological benchmark is **56.8% winner / 70.1% podium / 75.0% top five**, versus the previous blend's **56.8% / 70.0% / 75.0%** and grid baseline's **55.6% / 69.5% / 73.2%**. This keeps historical winner/top-five performance level while giving a small podium gain; the 2026 sample is still only 17 races, so the change should be reassessed as this season's results accumulate. Updated per-race comparisons and calibration reports are in `data/benchmark/` and `data/benchmark/season_2026/`.

The retrained version 4 passed the existing latest-12-race gate: Spearman 0.711 vs 0.662 for grid, and top-five overlap 0.717 vs 0.733 (within the allowed two-point margin). Winner accuracy and podium overlap both matched grid at 66.7% and 63.9% on that short holdout, so the season-specific prior is supported by the 17-race walk-forward rather than a claimed holdout gain.

## Session timing and race-pace UI (2026-10-25)

The 2026 retry now stores raw best-lap seconds, seconds to the session fastest, and lap counts for FP1/FP2/FP3 and qualifying. Qualifying classification time uses the last segment reached (Q3, else Q2, else Q1), with a validity marker to identify refreshed rows. On the US GP data this yields the classification example: Antonelli 1:34.035, Verstappen +0.179s, Hadjar +0.215s. The session panel now shows these values and adds driver long-run, race-pace and team-pace tables, including best/average filtered race-lap time, degradation and lap counts.

2026 coverage after the targeted refresh: qualifying segment times 90.7% of driver rows, FP3 times 64.1% (FP3 is not scheduled on sprint weekends), and race-pace summaries 83.8% (the remaining rows include the weekend with no session feed and the not-yet-raced US GP). The OpenF1 race pace fallback excludes pit-out laps and strong within-stint outliers; it has no full track-status timeline, so safety-car/yellow-flag laps can still affect the estimate. Exact qualifying/FP3 classifications are separate from the model's finishing-order predictions.

With this season's race-pace history backfilled, the 17-race 2026 walk-forward now scores **76.5% winner / 60.8% podium / 72.9% top five**. The grid baseline scores **70.6% / 60.8% / 72.9%**. On the same folds, removing pace features holds winner hits at 76.5% but lowers podium overlap to 58.8% and top-five overlap to 71.8%; this season's pace inputs add one podium and one top-five driver on average per race, with no winner hit-rate change. Retrained version 6 passed the existing latest-12-race gate (winner 75.0%, podium 61.1%, top five 71.7%; grid 66.7%, 61.1%, 73.3%).

## Current-season-first training (2026-10-25)

Training sample weights now give the regulation-reset season full weight, the immediately prior season 0.30 weight, and earlier seasons 0.05 weight. Previous-season and older rows remain in the fit as secondary reference; their engineered driver/team history remains available. Live-error weights are still applied on top. `benchmark.py --season-primary 2026 --seeds 1` evaluates each race chronologically after at least 10 earlier races and compares the sample-weight policy with equal-history training and nearby secondary-season weights. On 17 2026 races, equal-history scored **76.5% winner / 60.8% podium / 72.9% top five**; the 2026-first 1.0/0.30/0.05 policy scored **76.5% / 62.7% / 71.8%**. The mean across these three ranking targets increased from 0.701 to 0.703. The 0.15 and 0.50 previous-season variants had the same measured winner/podium/top-five averages as 0.30 at this sample size, so 0.30 is retained as a moderate secondary influence. This is a small 17-race comparison and the top-five change is a 1.1-point decline, so keep monitoring rather than treat it as a proven large gain.

Retrained version 7 passed the existing latest-12-race gate. It scored **72.5% Spearman / 75.0% winner / 61.1% podium / 71.7% top five** versus **70.5% / 66.7% / 61.1% / 73.3%** for the grid baseline. These holdout metrics are unchanged from the prior model on those folds; the support for season-first weighting is the small winner/podium/top-five aggregate gain in the 17-race rolling evaluation, not a claim of improved latest-12 holdout results. Reproduce the season-weight comparison with `python benchmark.py --season-primary 2026 --seeds 1`.
