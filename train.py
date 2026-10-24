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
from metrics import evaluate
from model import RankEnsemble

MODELS = Path("models")
STAGE_FEATURES = {"full": FEATURES, "pre": FEATURES_PRE}
STAGE_FILES = {"full": ("ranker.joblib", "ranker_holdout.joblib"),
               "pre": ("ranker_pre.joblib", "ranker_pre_holdout.joblib")}
GATE_MARGIN = 0.02   # new model's Spearman may be at most this far below "finish = grid"
KEEP_ARCHIVES = 5
TRAINING_LOG = Path("data") / "training_log.jsonl"


def fit_ranker(train, stage, weight_col=None):
    """Ensemble for one stage: 'full' (after qualifying) or 'pre' (practice only, no grid prior)."""
    full = stage == "full"
    return RankEnsemble(FEATURES if full else FEATURES_PRE, PL_FEATURES if full else PL_FEATURES_PRE,
                        prior_weight=1.0 if full else 0.0).fit(train, weight_col)


def calibrate_temperature(items):
    """Pick T so softmax(score/T) gives honest win/podium odds. items = (scores, finishing rank) of races
    the model did NOT train on; minimises the Plackett-Luce loss of the real top 3 (pooled over seasons)."""
    prepared = []
    for scores, finish in items:
        order = np.argsort(finish, kind="stable")
        prepared.append((scores, order, min(3, len(order))))
    best_t, best_nll = 1.5, np.inf
    for t in np.linspace(0.3, 4.0, 38):
        nll = 0.0
        for scores, order, k in prepared:
            z = scores[order] / t
            m = z.max()
            for i in range(k):
                nll -= z[i] - (np.log(np.exp(z[i:] - m).sum()) + m)
        if nll < best_nll:
            best_t, best_nll = t, nll
    return float(best_t)


def items_from_df(df, score_col):
    return [(g[score_col].to_numpy(float), g["finish"].to_numpy(float)) for _, g in df.groupby("race_id")]


def fold_items(al, stage, test_year, n_folds=3):
    """Out-of-sample scores for the seasons before the test season (single-seed models: cheap).
    Pooling several seasons stops one lucky/unlucky season from skewing the probabilities."""
    items = []
    for y in sorted(al["year"].unique()):
        if test_year - n_folds <= y < test_year and (al["year"] < y).any() and al.loc[al["year"] < y, "race_id"].nunique() >= 10:
            tr, te = al[al["year"] < y], al[al["year"] == y].copy()
            full = stage == "full"
            m = RankEnsemble(FEATURES if full else FEATURES_PRE, PL_FEATURES if full else PL_FEATURES_PRE,
                             prior_weight=1.0 if full else 0.0, seeds=(1,)).fit(tr, "w")
            te["s"] = m.predict(te)
            items += items_from_df(te[te["race_id"] < 10 ** 6], "s")
    return items


def _prep(raw, train_from=None):
    df = add_features(raw)
    df = df[df["finish"].notna()].reset_index(drop=True)
    return df[df["year"] >= train_from].reset_index(drop=True) if train_from else df


def _add_weights(df, order, race_weights, use_weights):
    """Recency (older races count a bit less) x emphasis on races the live model got wrong."""
    if not use_weights:
        df["w"] = 1.0
        return df
    rec = dict(zip(order, np.linspace(0.6, 1.0, len(order)) if len(order) > 1 else [1.0]))
    live = race_weights or {}
    df["w"] = [rec.get((y, r), 1.0) * live.get((y, r), 1.0) for y, r in zip(df["year"], df["round"])]
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
    df = _add_weights(df, order, race_weights, use_weights)

    # "right after FP1" copy of the data: practice 2/3 blanked, only the practice columns rebuilt
    aug = df.copy()
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
        hold = fit_ranker(tr, stage, "w")
        test[f"score_{stage}"] = hold.predict(test)
        tables[stage] = pd.DataFrame({
            "grid order (baseline)": evaluate(test.assign(g=-test["grid"].fillna(21)), "g"),
            "driver+team Elo": evaluate(test.assign(e=test["drv_elo"] + test["team_elo"]), "e"),
            "LightGBM ranker": evaluate(test, f"score_{stage}")}).round(3)
        al_stage = df if stage == "full" else pd.concat([df, aug], ignore_index=True)
        temps[stage] = calibrate_temperature(items_from_df(test, f"score_{stage}")
                                             + fold_items(al_stage, stage, test_year) + live_items.get(stage, []))
        imps[stage] = hold.feature_importance()
        holdouts[stage] = {"model": hold, "features": feats, "temperature": temps[stage], "stage": stage,
                           "trained_through": _last_race(train), "test_year": int(test_year), "test_from": test_from}
        bundles[stage] = {"model": fit_ranker(al, stage, "w"), "features": feats, "temperature": temps[stage],
                          "stage": stage, "trained_through": _last_race(df)}

    t = tables["full"]
    ranker_sp, grid_sp = float(t.loc["spearman", "LightGBM ranker"]), float(t.loc["spearman", "grid order (baseline)"])
    first = not (MODELS / STAGE_FILES["full"][0]).exists()
    passed = bool(ranker_sp >= grid_sp - GATE_MARGIN)
    promoted = first or passed
    reason_txt = ("first model" if first else
                  f"passed (Spearman {ranker_sp:.3f} vs grid {grid_sp:.3f})" if passed else
                  f"REJECTED, kept the previous model (Spearman {ranker_sp:.3f} vs grid {grid_sp:.3f})")

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
