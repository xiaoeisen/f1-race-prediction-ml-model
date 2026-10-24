# Compare ranking variants with earlier seasons only to avoid future-data leakage.
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, PL_FEATURES, add_features
from metrics import per_race
from model import RankEnsemble, labels

# Suppress warnings to keep benchmark output readable.
warnings.filterwarnings("ignore")

# Keep the previous model's feature set and settings for a consistent baseline comparison.
OLD_CFG = dict(elo_team_shrink=0.0, elo_drv_shrink=0.0, elo_reg_shrink=0.0, rookie_impute=False)
OLD_FEATURES = [f for f in FEATURES if not f.endswith("_rk") and f not in ("drv_qtm_l10", "drv_ftm_l10")]


# Train and predict with the older LightGBM ranker to compare it against current variants.
def old_model(tr, te, feats):
    tr = tr.sort_values("race_id")
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, label_gain=list(range(31)),
                       verbose=-1, random_state=42)
    m.fit(tr[feats], labels(tr), group=tr.groupby("race_id").size().to_numpy())
    return m.predict(te[feats])


# Run a season-by-season backtest so every test season uses only earlier seasons for training.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/races.csv")
    ap.add_argument("--from-year", type=int, default=None, help="first test season")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

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
    out = {k: [] for k in ("grid", "previous", "trees", "linear", "current")}
    for y in test_years:
        tr, te = new[new["year"] < y], new[new["year"] == y].copy()
        otr, ote = old[old["year"] < y], old[old["year"] == y].copy()
        if tr["race_id"].nunique() < 10:
            continue
        ens = RankEnsemble(FEATURES, PL_FEATURES, prior_weight=1.0, seeds=seeds).fit(tr)
        parts = ens.components(te)
        te["grid"] = -te["grid"].fillna(21)
        te["trees"], te["linear"], te["current"] = parts["trees"], parts["linear"], ens.predict(te)
        te["previous"] = old_model(otr, ote, OLD_FEATURES)
        for k in out:
            out[k].append(per_race(te, k))

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


# Run the benchmark only when this file is executed directly.
if __name__ == "__main__":
    main()
