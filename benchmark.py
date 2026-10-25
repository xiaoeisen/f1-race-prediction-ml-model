# Compare ranking variants with earlier seasons only to avoid future-data leakage.
import argparse
import warnings
from pathlib import Path

import sklearn  # load its packaged OpenMP runtime before LightGBM on Windows
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (FEATURES, FEATURES_WITH_PACE, NEW_PACE_FEATURES, PACE_FEATURES, PACE_PL_FEATURES,
                      PL_FEATURES, add_features)
from metrics import (calibration_error, per_race, probability_metrics,
                     race_outcome_probabilities, write_calibration_svg)
from model import (FULL_GRID_PRIOR_WEIGHT, REG_RESET_GRID_PRIOR_WEIGHT,
                   REG_RESET_SEASON, RankEnsemble, labels)
from train import calibrate_temperature

# Suppress warnings to keep benchmark output readable.
warnings.filterwarnings("ignore")

# Keep the previous model's feature set and settings for a consistent baseline comparison.
OLD_CFG = dict(elo_team_shrink=0.0, elo_drv_shrink=0.0, elo_reg_shrink=0.0, rookie_impute=False)
OLD_FEATURES = [f for f in FEATURES_WITH_PACE if not f.endswith("_rk") and f not in
                ("drv_qtm_l10", "drv_ftm_l10") and f not in NEW_PACE_FEATURES]


# Train and predict with the older LightGBM ranker to compare it against current variants.
def old_model(tr, te, feats):
    tr = tr.sort_values("race_id")
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, label_gain=list(range(31)),
                       verbose=-1, random_state=42)
    m.fit(tr[feats], labels(tr), group=tr.groupby("race_id").size().to_numpy())
    return m.predict(te[feats])


def _prepare_blend_races(df):
    """Cache compact per-race arrays for fast chronological blend selection."""
    races = []
    for _, g in df.groupby("race_id", sort=False):
        finish = g["finish"].to_numpy(float)
        actual_order = np.argsort(finish, kind="stable")
        actual_top3 = np.zeros(len(g), dtype=bool)
        actual_top3[actual_order[:3]] = True
        actual_top5 = np.zeros(len(g), dtype=bool)
        actual_top5[actual_order[:5]] = True
        races.append((g["trees"].to_numpy(float), g["linear"].to_numpy(float),
                      g["grid"].to_numpy(float), int(actual_order[0]), actual_top3, actual_top5))
    return races


def _blend_objective(races, linear_weight, grid_weight):
    """Mean of winner accuracy and top-five overlap on prior out-of-sample races."""
    winners, top5 = [], []
    for trees, linear, grid, actual_winner, _, actual_top5 in races:
        order = np.argsort(-(trees + linear_weight * linear + grid_weight * grid), kind="stable")
        winners.append(float(order[0] == actual_winner))
        top5.append(float(actual_top5[order[:5]].mean()))
    return (float(np.mean(winners)) + float(np.mean(top5))) / 2


def _winner_podium_objective(races, grid_weight):
    """Mean of winner accuracy and podium overlap on prior out-of-sample races."""
    winners, podiums = [], []
    for trees, linear, grid, actual_winner, actual_top3, _ in races:
        order = np.argsort(-(trees + linear + grid_weight * grid), kind="stable")
        winners.append(float(order[0] == actual_winner))
        podiums.append(float(actual_top3[order[:3]].mean()))
    return (float(np.mean(winners)) + float(np.mean(podiums))) / 2


def _season_primary_benchmark(data, season, seeds):
    """Within-season walk-forward evaluation with explicit primary/secondary season weights."""
    raw = pd.read_csv(data)
    df = add_features(raw)
    df = df[df["finish"].notna()].reset_index(drop=True)
    season_rows = df[df["year"] == season]
    if season_rows.empty:
        raise ValueError(f"No finished race data found for season {season}.")

    # Compare the requested season-first policy to an unweighted fit and to
    # nearby secondary-season strengths. Older rows still supply engineered
    # history features, but have little influence on the fitted outcome models.
    policies = {
        "equal_history": (1.0, 1.0, 1.0),
        "season_primary_0.30": (1.0, 0.30, 0.05),
        "season_primary_0.50": (1.0, 0.50, 0.05),
        "season_primary_0.15": (1.0, 0.15, 0.05),
    }
    rows = {name: [] for name in policies}
    for rid, test in season_rows.groupby("race_id", sort=True):
        prior = df[df["race_id"] < rid].copy()
        if prior["race_id"].nunique() < 10:
            continue
        for name, (primary_w, secondary_w, old_w) in policies.items():
            weights = np.where(prior["year"].to_numpy() == season, primary_w,
                               np.where(prior["year"].to_numpy() == season - 1, secondary_w, old_w))
            model = RankEnsemble(FEATURES, PL_FEATURES, prior_weight=FULL_GRID_PRIOR_WEIGHT,
                                 seeds=seeds).fit(prior.assign(_season_w=weights), "_season_w")
            pred = test.copy()
            pred["score"] = model.predict(pred)
            score = per_race(pred, "score").iloc[0].to_dict()
            score["round"] = int(test["round"].iloc[0])
            rows[name].append(score)
    results = {name: pd.DataFrame(values) for name, values in rows.items() if values}
    if not results:
        raise ValueError(f"Fewer than 11 completed races before season {season} tests.")
    print(f"{season} within-season walk-forward, {len(next(iter(results.values())))} races; "
          "each prediction trained only on earlier races\n")
    print(f"{'policy':26s}{'winner':>9s}{'podium':>9s}{'top5':>9s}{'spearman':>11s}")
    for name, frame in results.items():
        print(f"{name:26s}{frame['winner'].mean():9.3f}{frame['podium'].mean():9.3f}"
              f"{frame['top5'].mean():9.3f}{frame['spearman'].mean():11.3f}")
    return results


# Run a season-by-season backtest so every test season uses only earlier seasons for training.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/races.csv")
    ap.add_argument("--from-year", type=int, default=None, help="first test season")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--season-primary", type=int, default=None,
                    help="compare season-first sample weights using within-season chronological predictions")
    ap.add_argument("--report-dir", default="data/benchmark",
                    help="directory for per-race walk-forward scores and calibration plots")
    a = ap.parse_args()

    if a.season_primary is not None:
        _season_primary_benchmark(a.data, a.season_primary, tuple(range(1, a.seeds + 1)))
        return

    # Build current and baseline features from the same race data for a fair comparison.
    raw = pd.read_csv(a.data)
    new = add_features(raw)
    new = new[new["finish"].notna()].reset_index(drop=True)
    old = add_features(raw, OLD_CFG)
    old = old[old["finish"].notna()].reset_index(drop=True)
    years = sorted(new["year"].unique())
    test_years = [y for y in years[1:] if y >= (a.from_year or years[1])]
    seeds = tuple(range(1, a.seeds + 1))

    # Score each season independently to avoid training on future results.
    blend_weights = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    linear_weights = (1.0, 0.75, 1.25, 0.5, 1.5)
    tuning_grid_weights = (1.0, 0.75, 1.25, 0.5, 1.5, 0.25, 0.0)
    out = {k: [] for k in ("grid", "previous", "trees", "linear", "current", "no_pace",
                           "pace_pl", "tuned", "tuned_wp")}
    out.update({f"grid_blend_{w:g}": [] for w in blend_weights})
    tuning_rows, weights_by_year, podium_diagnostics = [], [], []
    calibration = {"current": [], "grid": []}
    probability_results = {(kind, cal): {"win": [], "podium": []}
                           for kind in calibration for cal in ("raw", "calibrated")}
    probability_samples = {(kind, cal, event): {"p": [], "y": []}
                           for kind in calibration for cal in ("raw", "calibrated")
                           for event in ("win", "podium")}
    probability_samples_by_year = {}
    for y in test_years:
        tr, te = new[new["year"] < y], new[new["year"] == y].copy()
        otr, ote = old[old["year"] < y], old[old["year"] == y].copy()
        if tr["race_id"].nunique() < 10:
            continue
        ens = RankEnsemble(FEATURES, PL_FEATURES, prior_weight=1.0, seeds=seeds).fit(tr)
        parts = ens.components(te)
        # `components()` applies the regulation-reset prior for 2026. Normalize
        # it back to a unit grid signal here so the reported blend sweep remains
        # comparable across seasons and each candidate weight keeps its meaning.
        component_prior = np.where(te["year"].to_numpy() == REG_RESET_SEASON,
                                   REG_RESET_GRID_PRIOR_WEIGHT, 1.0)
        grid_signal = parts["grid"] / component_prior
        # Paired feature checks on the exact same chronological fold: compare
        # live pace features with a no-pace tree ablation and pace-extended linear variant.
        no_pace = RankEnsemble([f for f in FEATURES if f not in PACE_FEATURES],
                               [f for f in PL_FEATURES if f not in PACE_FEATURES],
                               prior_weight=FULL_GRID_PRIOR_WEIGHT, seeds=seeds).fit(tr)
        pace_pl = RankEnsemble(FEATURES_WITH_PACE,
                               PL_FEATURES + [f for f in PACE_PL_FEATURES if f not in PL_FEATURES],
                               prior_weight=FULL_GRID_PRIOR_WEIGHT, seeds=seeds).fit(tr)
        no_pace_score = no_pace.predict(te)
        pace_pl_score = pace_pl.predict(te)
        te["start_grid"] = te["grid"]
        te["grid"] = -te["grid"].fillna(21)
        te["trees"], te["linear"] = parts["trees"], parts["linear"]
        # Use the already-computed components. The `grid` column is replaced with
        # a baseline score below, so calling predict() after that would feed the
        # baseline value back into the model's grid feature.
        current_prior = np.where(te["year"].to_numpy() == REG_RESET_SEASON,
                                 REG_RESET_GRID_PRIOR_WEIGHT, FULL_GRID_PRIOR_WEIGHT)
        te["current"] = (parts["trees"] + parts["linear"]
                          + current_prior * grid_signal)
        te["no_pace"] = no_pace_score
        te["pace_pl"] = pace_pl_score
        for w in blend_weights:
            te[f"grid_blend_{w:g}"] = parts["trees"] + parts["linear"] + grid_signal * w
        # Nested chronological tuning: choose a small linear/grid blend using only
        # the previous three seasons' out-of-sample component predictions.
        recent = pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame()
        if len(recent):
            recent = recent[recent["year"] >= y - 3]
        best_weights, best_objective = (1.0, 1.0), -np.inf
        best_wp_weight, best_wp_objective = 1.0, -np.inf
        if len(recent) and recent["race_id"].nunique() >= 30:
            recent_races = _prepare_blend_races(recent)
            for linear_weight in linear_weights:
                for grid_weight in tuning_grid_weights:
                    objective = _blend_objective(recent_races, linear_weight, grid_weight)
                    # Iteration order puts the production blend first, so ties retain it.
                    if objective > best_objective + 1e-12:
                        best_objective = objective
                        best_weights = (linear_weight, grid_weight)
            for grid_weight in tuning_grid_weights:
                objective = _winner_podium_objective(recent_races, grid_weight)
                if objective > best_wp_objective + 1e-12:
                    best_wp_objective = objective
                    best_wp_weight = grid_weight
        linear_weight, grid_weight = best_weights
        te["tuned"] = (parts["trees"] + linear_weight * parts["linear"]
                       + grid_weight * parts["grid"])
        te["tuned_wp"] = parts["trees"] + parts["linear"] + best_wp_weight * parts["grid"]
        weights_by_year.append(dict(year=int(y), linear_weight=linear_weight,
                                    grid_weight=grid_weight, train_races=int(recent["race_id"].nunique())
                                    if len(recent) else 0, winner_podium_grid_weight=best_wp_weight))
        te["previous"] = old_model(otr, ote, OLD_FEATURES)
        # Keep the actual and predicted podium candidates so recurring winner /
        # podium misses can be traced back to their pre-race pace inputs.
        for rid, g in te.groupby("race_id"):
            pred_rank = g["current"].rank(ascending=False, method="first")
            actual_rank = g["finish"].rank(method="first")
            selected = g.index[(pred_rank <= 3) | (actual_rank <= 3)]
            winner_hit = bool(g.loc[pred_rank.idxmin(), "finish"] == 1)
            for idx in selected:
                row = g.loc[idx]
                record = {
                    "race_id": int(rid), "year": int(y), "round": int(row.get("round", 0)),
                    "event": row.get("event", ""), "driver": row.get("driver", ""),
                    "team": row.get("team", ""), "finish": row["finish"],
                    "predicted_rank": int(pred_rank.loc[idx]), "start_grid": row["start_grid"],
                    "winner_hit": winner_hit, "actual_podium": bool(actual_rank.loc[idx] <= 3),
                    "predicted_podium": bool(pred_rank.loc[idx] <= 3),
                }
                for col in ("quali_pos", "q_best_gap", "fp1_long_gap", "fp2_long_gap",
                            "fp1_deg_per10", "fp2_deg_per10", "q_s1_gap", "q_s2_gap",
                            "q_s3_gap", "race_pace_gap", "race_deg_per10", "current"):
                    if col in g:
                        record[col] = row[col]
                podium_diagnostics.append(record)
        for k in out:
            out[k].append(per_race(te, k))

        tuning_rows.append(pd.DataFrame({"race_id": te["race_id"].to_numpy(), "year": int(y),
                                         "finish": te["finish"].to_numpy(),
                                         "trees": parts["trees"], "linear": parts["linear"],
                                         "grid": parts["grid"]}))

        for kind, score_col in (("current", "current"), ("grid", "grid")):
            history = calibration[kind][-40:]
            temp = calibrate_temperature(history, n_sims=256) if history else 1.0
            for cal, t in (("raw", 1.0), ("calibrated", temp)):
                for rid, g in te.groupby("race_id"):
                    risk = (g["drv_inc_l20"].fillna(0.06) + g["team_mech_l10"].fillna(0.08)).clip(0.03, 0.35).to_numpy()
                    probs = race_outcome_probabilities(g[score_col].to_numpy(float), risk, t,
                                                       n_sims=4096, seed=int(y * 1000 + rid) % (2**32))
                    for cutoff, event in ((1, "win"), (3, "podium")):
                        sample = probability_samples[(kind, cal, event)]
                        sample["p"].append(probs[event])
                        outcome = (g["finish"].to_numpy(float) <= cutoff).astype(float)
                        sample["y"].append(outcome)
                        brier = probability_metrics(probs[event], g["finish"].to_numpy(float), cutoff)["brier"]
                        probability_results[(kind, cal)][event].append(brier)
                        year_sample = probability_samples_by_year.setdefault(
                            (int(y), kind, cal, event), {"p": [], "y": [], "brier": []})
                        year_sample["p"].append(probs[event])
                        year_sample["y"].append(outcome)
                        year_sample["brier"].append(brier)
            calibration[kind].extend(
                (g[score_col].to_numpy(float), g["finish"].to_numpy(float),
                 (g["drv_inc_l20"].fillna(0.06) + g["team_mech_l10"].fillna(0.08)).clip(0.03, 0.35).to_numpy(float))
                for _, g in te.groupby("race_id"))

    # Summarize per-race scores and paired differences to show how the variants compare.
    res = {k: pd.concat(v) for k, v in out.items() if v}
    n = len(res["grid"])
    print(f"{n} races, test seasons {test_years[0]}-{test_years[-1]}, walk-forward (train on earlier seasons only)\n")
    base = res["previous"]
    cols = ["winner", "podium", "top5", "exact", "ndcg5", "spearman", "mae", "mae_fin"]
    print(f"{'':10s}" + "".join(f"{c:>9s}" for c in cols) + f"{'top5 diff vs previous':>26s}{'exact diff vs grid':>22s}")
    for k, r in res.items():
        d5, dg = r["top5"] - base["top5"], r["exact"] - res["grid"]["exact"]
        print(f"{k:10s}" + "".join(f"{r[c].mean():9.3f}" for c in cols)
              + f"   {d5.mean():+.3f} +/- {d5.std() / np.sqrt(n):.3f}   {dg.mean():+.3f} +/- {dg.std() / np.sqrt(n):.3f}")
    print("\nTop-five overlap by grid-prior blend weight:")
    for w in blend_weights:
        r = res[f"grid_blend_{w:g}"]
        print(f"  grid weight {w:g}: top5={r['top5'].mean():.3f}, winner={r['winner'].mean():.3f}, podium={r['podium'].mean():.3f}")
    tuned = res["tuned"]
    print(f"  nested tuning: top5={tuned['top5'].mean():.3f}, winner={tuned['winner'].mean():.3f}")
    tuned_wp = res["tuned_wp"]
    print(f"  nested winner/podium tuning: winner={tuned_wp['winner'].mean():.3f}, "
          f"podium={tuned_wp['podium'].mean():.3f}, top5={tuned_wp['top5'].mean():.3f}")
    print("\nOut-of-sample win/podium Brier (lower is better; calibrated uses only earlier seasons):")
    for kind in ("grid", "current"):
        for cal in ("raw", "calibrated"):
            vals = probability_results[(kind, cal)]
            win = probability_samples[(kind, cal, "win")]
            podium = probability_samples[(kind, cal, "podium")]
            wp, wy = np.concatenate(win["p"]), np.concatenate(win["y"])
            pp, py = np.concatenate(podium["p"]), np.concatenate(podium["y"])
            we, pe = calibration_error(wp, wy), calibration_error(pp, py)
            print(f"  {kind:7s} {cal:10s}: win={np.mean(vals['win']):.4f} (ECE {we['ece']:.4f}), "
                  f"podium={np.mean(vals['podium']):.4f} (ECE {pe['ece']:.4f})")

    report_dir = Path(a.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    walk = pd.concat([res["grid"]["year"].rename("year")] +
                     [frame.drop(columns="year").add_prefix(f"{kind}_") for kind, frame in res.items()], axis=1)
    walk.index.name = "race"
    walk.reset_index().to_csv(report_dir / "walkforward_metrics.csv", index=False)

    probability_summary = []
    for event in ("win", "podium"):
        curves = {}
        for kind in ("grid", "current"):
            for cal in ("raw", "calibrated"):
                sample = probability_samples[(kind, cal, event)]
                p, y_true = np.concatenate(sample["p"]), np.concatenate(sample["y"])
                errors = calibration_error(p, y_true)
                label = f"{kind} {cal}"
                curves[label] = (p, y_true)
                probability_summary.append(dict(event=event, model=kind, calibration=cal,
                                                brier=float(np.mean(probability_results[(kind, cal)][event])),
                                                ece=errors["ece"], mce=errors["mce"], drivers=len(p)))
        write_calibration_svg(curves, f"Walk-forward {event} calibration",
                              report_dir / f"calibration_{event}.svg")
    pd.DataFrame(probability_summary).to_csv(report_dir / "probability_calibration.csv", index=False)
    by_year = []
    for (year, kind, cal, event), sample in sorted(probability_samples_by_year.items()):
        p, y_true = np.concatenate(sample["p"]), np.concatenate(sample["y"])
        errors = calibration_error(p, y_true)
        by_year.append(dict(year=year, event=event, model=kind, calibration=cal,
                            brier=float(np.mean(sample["brier"])), ece=errors["ece"],
                            mce=errors["mce"], drivers=len(p)))
    pd.DataFrame(by_year).to_csv(report_dir / "probability_calibration_by_year.csv", index=False)
    pd.DataFrame(weights_by_year).to_csv(report_dir / "blend_weights_by_year.csv", index=False)
    pd.DataFrame(podium_diagnostics).to_csv(report_dir / "podium_miss_analysis.csv", index=False)
    print(f"\nSaved per-race metrics and out-of-sample calibration plots to {report_dir}")


# Run the benchmark only when this file is executed directly.
if __name__ == "__main__":
    main()
