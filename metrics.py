# Calculate shared race-level metrics for training and benchmarking.
import numpy as np
import pandas as pd


def _dcg(rel):
    return float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())


def ndcg(pred_rank, finish, k):
    """NDCG@k with relevance = 21 - finish position (P1 -> 20). 1.0 = perfect ordering of the top k."""
    rel = (21 - finish).clip(lower=0).to_numpy(float)
    top = np.argsort(pred_rank.to_numpy(), kind="stable")[:k]
    ideal = np.sort(rel)[::-1][:k]
    return _dcg(rel[top]) / _dcg(ideal) if _dcg(ideal) > 0 else np.nan


def race_metrics(g, col):
    pred = g[col].rank(ascending=False, method="first")
    act = g["finish"].rank(method="first")
    first5 = g.assign(a=act).loc[pred.sort_values().index[:5], "a"].to_numpy()
    fin = g["dnf"].eq(0) if "dnf" in g else pd.Series(True, index=g.index)
    return dict(
        spearman=pred.corr(act, method="spearman"),
        winner=float(act[pred.idxmin()] == 1),
        podium=len(set(g.index[pred <= 3]) & set(g.index[act <= 3])) / 3,
        top5=len(set(g.index[pred <= 5]) & set(g.index[act <= 5])) / 5,
        top10=len(set(g.index[pred <= 10]) & set(g.index[act <= 10])) / 10,
        exact=float(np.mean(first5 == np.arange(1, 6))),
        ndcg5=ndcg(pred, g["finish"], 5), ndcg10=ndcg(pred, g["finish"], 10),
        mae=(pred - act).abs().mean(),
        mae_fin=(pred[fin] - g.loc[fin, "finish"]).abs().mean() if fin.any() else np.nan)


def per_race(df, col):
    rows = []
    for rid, g in df.groupby("race_id"):
        rows.append(dict(race=rid, year=int(g["year"].iloc[0]) if "year" in g else 0, **race_metrics(g, col)))
    return pd.DataFrame(rows).set_index("race")


def evaluate(df, col):
    return per_race(df, col).drop(columns="year").mean()
