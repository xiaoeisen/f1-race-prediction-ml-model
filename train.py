"""Train the ranking models.

    python train.py                  # backtest on the latest season, then refit on everything
    python train.py --test-year 2024 # test on the whole 2024 season instead (default: the latest 12 races)
    python train.py --train-from 2022

Two models are trained because the information differs during a race weekend:
  full  everything incl. qualifying / grid          (used after qualifying and for backtests)
  pre   no qualifying/grid, trained on practice-only (used after FP1/FP2, before qualifying)

train_models() is also what the autopilot calls to retrain itself after every race.
A retrained model only replaces the live one if it passes a sanity check (see GATE_MARGIN)."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from circuits import unmatched
from features import (FEATURES, FEATURES_PRE, LATE_PRACTICE_COLS, PL_FEATURES, PL_FEATURES_PRE,
                      add_features, weekend_practice)
from metrics import evaluate, probability_metrics, race_outcome_probabilities
from model import FULL_GRID_PRIOR_WEIGHT, RankEnsemble

MODELS = Path("models")
STAGE_FEATURES = {"full": FEATURES, "pre": FEATURES_PRE}
STAGE_FILES = {"full": ("ranker.joblib", "ranker_holdout.joblib"),
               "pre": ("ranker_pre.joblib", "ranker_pre_holdout.joblib")}
GATE_MARGIN = 0.02   # tolerate at most a 2-point drop against the grid on Spearman or top-five overlap
KEEP_ARCHIVES = 5
TRAINING_LOG = Path("data") / "training_log.jsonl"


def fit_ranker(train, stage, weight_col=None):
    """Ensemble for one stage: 'full' (after qualifying) or 'pre' (practice only, no grid prior)."""
    full = stage == "full"
    return RankEnsemble(FEATURES if full else FEATURES_PRE, PL_FEATURES if full else PL_FEATURES_PRE,
                        prior_weight=FULL_GRID_PRIOR_WEIGHT if full else 0.0).fit(train, weight_col)


def calibrate_temperature(items, n_sims=2048):
    """Choose temperature by held-out win + podium Brier score.

    Items are (scores, finishing positions[, per-driver retirement risk]) from races
    the model did not train on. Legacy two-item records use a 14% risk.
    """
    prepared = []
    for item in items:
        scores, finish = item[:2]
        p_out = item[2] if len(item) > 2 else np.full(len(scores), 0.14)
        prepared.append((np.asarray(scores, float), np.asarray(finish, float),
                         np.asarray(p_out, float)))
    if not prepared:
        return 1.5
    best_t, best_loss = 1.5, np.inf
    for t in np.linspace(0.3, 4.0, 38):
        losses = []
        for i, (scores, finish, p_out) in enumerate(prepared):
            probs = race_outcome_probabilities(scores, p_out, t, n_sims, seed=1000 + i)
            win = probability_metrics(probs["win"], finish, 1)["brier"]
            podium = probability_metrics(probs["podium"], finish, 3)["brier"]
            losses.append(win + podium)
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_t, best_loss = t, loss
    return float(best_t)


def items_from_df(df, score_col):
    items = []
    for _, g in df.groupby("race_id"):
        p_out = (g["drv_inc_l20"].fillna(0.06) + g["team_mech_l10"].fillna(0.08)).clip(0.03, 0.35)
        items.append((g[score_col].to_numpy(float), g["finish"].to_numpy(float), p_out.to_numpy(float)))
    return items


def fold_items(al, stage, test_year, n_folds=3):
    """Out-of-sample scores for the seasons before the test season (single-seed models: cheap).
    Pooling several seasons stops one lucky/unlucky season from skewing the probabilities."""
    items = []
    for y in sorted(al["year"].unique()):
        if test_year - n_folds <= y < test_year and (al["year"] < y).any() and al.loc[al["year"] < y, "race_id"].nunique() >= 10:
            tr = al[al["year"] < y]
            season = al[al["year"] == y]
            # The pre-quali model is validated on the FP1-only copy, with FP2/FP3
            # fields blanked just as they are for a live early-weekend prediction.
            te = season[season["is_aug"]].copy() if stage == "pre" else season[~season["is_aug"]].copy()
            full = stage == "full"
            m = RankEnsemble(FEATURES if full else FEATURES_PRE, PL_FEATURES if full else PL_FEATURES_PRE,
                             prior_weight=FULL_GRID_PRIOR_WEIGHT if full else 0.0, seeds=(1,)).fit(tr, "w")
            te["s"] = m.predict(te)
            items += items_from_df(te, "s")
    return items


def _prep(raw, train_from=None):
    df = add_features(raw)
    df = df[df["finish"].notna()].reset_index(drop=True)
    return df[df["year"] >= train_from].reset_index(drop=True) if train_from else df


def _add_weights(df, order, race_weights, use_weights, primary_year=None):
    """Prioritize the current regulation season; retain prior seasons as weaker context."""
    if not use_weights:
        df["w"] = 1.0
        return df
    primary_year = int(primary_year if primary_year is not None else df["year"].max())
    season_weight = {int(y): (1.0 if int(y) == primary_year else
                              0.30 if int(y) == primary_year - 1 else 0.05)
                     for y in df["year"].unique()}
    live = race_weights or {}
    df["w"] = [season_weight.get(int(y), 0.05) * live.get((int(y), int(r)), 1.0)
                for y, r in zip(df["year"], df["round"])]
    return df


def _last_race(d):
    r = d.sort_values(["year", "round"]).iloc[-1]
    return [int(r["year"]), int(r["round"])]


def _version():
    f = MODELS / "version.json"
    return json.loads(f.read_text()) if f.exists() else {"version": 0}


def _archive_current():
    cur = _version()
    files = [MODELS / n for pair in STAGE_FILES.values() for n in pair] + [MODELS / "metrics.json", MODELS / "version.json"]
    files = [f for f in files if f.exists()]
    if not files:
        return
    dest = MODELS / "archive" / f"v{cur.get('version', 0)}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)
    old = sorted((MODELS / "archive").iterdir(), key=lambda p: p.stat().st_mtime)
    for p in old[:-KEEP_ARCHIVES]:
        shutil.rmtree(p, ignore_errors=True)


def _log(entry):
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def train_models(raw, test_year=None, train_from=None, race_weights=None, live_items=None, test_races=12,
                 use_weights=True, models_dir=None, reason="manual", verbose=True):
    """Backtest + refit both stage models. Returns a report dict (see keys at the bottom)."""
    global MODELS
    if models_dir:
        MODELS = Path(models_dir)
    MODELS.mkdir(parents=True, exist_ok=True)
    live_items = live_items or {}

    df = _prep(raw, train_from)
    if df.empty:
        raise ValueError("No finished races to train on.")
    order = sorted({(int(y), int(r)) for y, r in zip(df["year"], df["round"])})
    df = _add_weights(df, order, race_weights, use_weights,
                      primary_year=test_year if test_year is not None else max(y for y, _ in order))

    # "right after FP1" copy of the data: practice 2/3 blanked, only the practice columns rebuilt
    aug = df.copy()
    df["is_aug"] = False
    aug["is_aug"] = True
    for c in LATE_PRACTICE_COLS:
        if c in aug:
            aug[c] = np.nan
    weekend_practice(aug)
    aug["race_id"] += int(df["race_id"].max()) + 1000

    # Hold-out set: a whole season if --test-year is given, otherwise the latest `test_races` races (spanning
    # a season boundary if needed). A fixed season would be a coin-flip right after round 1 of a new year.
    key = lambda d: [(int(y), int(r)) for y, r in zip(d["year"], d["round"])]
    if test_year:
        is_test, is_aug_test = df["year"] == test_year, aug["year"] == test_year
        test_from = [int(test_year), int(df.loc[df["year"] == test_year, "round"].min()) if is_test.any() else 1]
    else:
        n_test = min(test_races, max(3, len(order) // 3))
        keys = set(order[-n_test:])
        is_test = pd.Series([k in keys for k in key(df)], index=df.index)
        is_aug_test = pd.Series([k in keys for k in key(aug)], index=aug.index)
        test_from = list(order[-n_test])
        test_year = test_from[0]
    train, test = df[~is_test].copy(), df[is_test].copy()
    if train.empty or test.empty:
        raise ValueError("Need at least two seasons of data (collect more years).")
    aug_train = aug[~is_aug_test]

    tables, bundles, holdouts, imps, temps = {}, {}, {}, {}, {}
    for stage, feats in STAGE_FEATURES.items():
        tr = train if stage == "full" else pd.concat([train, aug_train], ignore_index=True)
        al = df if stage == "full" else pd.concat([df, aug], ignore_index=True)
        stage_test = test.copy() if stage == "full" else aug.loc[is_aug_test].copy()
        hold = fit_ranker(tr, stage, "w")
        stage_test[f"score_{stage}"] = hold.predict(stage_test)
        tables[stage] = pd.DataFrame({
            "grid order (baseline)": evaluate(stage_test.assign(g=-stage_test["grid"].fillna(21)), "g"),
            "driver+team Elo": evaluate(stage_test.assign(e=stage_test["drv_elo"] + stage_test["team_elo"]), "e"),
            "LightGBM ranker": evaluate(stage_test, f"score_{stage}")}).round(3)
        al_stage = df if stage == "full" else pd.concat([df, aug], ignore_index=True)
        temps[stage] = calibrate_temperature(items_from_df(stage_test, f"score_{stage}")
                                             + fold_items(al_stage, stage, test_year) + live_items.get(stage, []))
        imps[stage] = hold.feature_importance()
        holdouts[stage] = {"model": hold, "features": feats, "temperature": temps[stage], "stage": stage,
                           "trained_through": _last_race(train), "test_year": int(test_year), "test_from": test_from}
        bundles[stage] = {"model": fit_ranker(al, stage, "w"), "features": feats, "temperature": temps[stage],
                          "stage": stage, "trained_through": _last_race(df)}

    t = tables["full"]
    ranker_sp, grid_sp = float(t.loc["spearman", "LightGBM ranker"]), float(t.loc["spearman", "grid order (baseline)"])
    ranker_top5, grid_top5 = float(t.loc["top5", "LightGBM ranker"]), float(t.loc["top5", "grid order (baseline)"])
    first = not (MODELS / STAGE_FILES["full"][0]).exists()
    passed = bool(ranker_sp >= grid_sp - GATE_MARGIN and ranker_top5 >= grid_top5 - GATE_MARGIN)
    promoted = first or passed
    detail = f"Spearman {ranker_sp:.3f}/{grid_sp:.3f}, top-five {ranker_top5:.3f}/{grid_top5:.3f} (model/grid)"
    reason_txt = ("first model" if first else
                  f"passed ({detail})" if passed else
                  f"REJECTED, kept the previous model ({detail})")

    version = _version().get("version", 0)
    if promoted:
        _archive_current()
        version += 1
        for stage in STAGE_FILES:
            bundles[stage]["version"] = holdouts[stage]["version"] = version
            joblib.dump(bundles[stage], MODELS / STAGE_FILES[stage][0])
            joblib.dump(holdouts[stage], MODELS / STAGE_FILES[stage][1])
        (MODELS / "metrics.json").write_text(json.dumps({
            "test_year": int(test_year), "test_from": test_from, "train_races": int(train["race_id"].nunique()),
            "test_races": int(test["race_id"].nunique()), "results": tables["full"].to_dict(),
            "pre_results": tables["pre"].to_dict()}, indent=2))
        (MODELS / "version.json").write_text(json.dumps({
            "version": version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_through": _last_race(df), "races": len(order), "gate": reason_txt}))
    n_up = sum(1 for v in (race_weights or {}).values() if v > 1.0)
    _log(dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), reason=reason, promoted=promoted,
              version=version, gate=reason_txt, trained_through=_last_race(df), races=len(order),
              spearman=ranker_sp, grid_spearman=grid_sp, races_upweighted=n_up))
    if verbose:
        print(f"Train: {train['race_id'].nunique()} races | Test: the latest {test['race_id'].nunique()} races (from {test_from[0]} R{test_from[1]})")
        for stage in ("full", "pre"):
            print(f"\nHold-out results, {stage} model (per-race averages):")
            print(tables[stage].to_string())
        print("\nTop features, full model (gain):")
        print((imps["full"] / imps["full"].sum()).head(12).round(3).to_string())
        print(f"\nGate: {reason_txt}")
    return dict(promoted=promoted, reason=reason_txt, version=version, tables=tables, test_year=int(test_year),
                trained_through=_last_race(df), races=len(order), spearman=ranker_sp, grid_spearman=grid_sp,
                races_upweighted=n_up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/races.csv")
    ap.add_argument("--test-year", type=int, default=None)
    ap.add_argument("--train-from", type=int, default=None,
                    help="only train/test on seasons >= this year (older races still feed history/Elo)")
    ap.add_argument("--no-weights", action="store_true", help="plain training: no recency / live-error weighting")
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    if "race_pace_gap" not in raw or raw["race_pace_gap"].notna().mean() < 0.2:
        print("NOTE: most rows have no lap-level data (tyre-aware race pace, safety cars, sectors), e.g. because "
              "they came from the fast f1db load. Training works without it; to add it for recent seasons run "
              "`python collect.py --start 2022` (slow, one-off).\n")
    missing = unmatched(raw["circuit"].unique())
    if missing:
        print("NOTE: no circuit facts for these locations (add rows to circuits.csv):", ", ".join(missing), "\n")

    import predlog  # live log (if any) refines weights + probability calibration
    entries = predlog.load()
    rep = train_models(
        raw, test_year=args.test_year, train_from=args.train_from, use_weights=not args.no_weights,
        race_weights=predlog.race_weights(entries) if not args.no_weights else None,
        live_items={"full": predlog.calibration_items(entries, "quali"),
                    "pre": predlog.calibration_items(entries, "pre")},
        reason="manual")
    if rep["promoted"]:
        print(f"\nSaved models v{rep['version']}: ranker.joblib, ranker_pre.joblib (+ hold-out versions), metrics.json")


if __name__ == "__main__":
    main()
