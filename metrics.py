# Calculate shared race-level metrics for training and benchmarking.
import numpy as np
import pandas as pd
from pathlib import Path


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


def race_outcome_probabilities(scores, p_out=None, temperature=1.0, n_sims=20000, seed=0):
    """Estimate race outcome probabilities from rank scores and per-driver retirement risk."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if not n:
        return dict(expected_pos=np.array([]), win=np.array([]), podium=np.array([]),
                    top5=np.array([]), top10=np.array([]))
    p_out = np.zeros(n, dtype=float) if p_out is None else np.asarray(p_out, dtype=float)
    p_out = np.clip(np.nan_to_num(p_out, nan=0.0), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    noisy = scores[None, :] / max(float(temperature), 1e-6) + rng.gumbel(size=(n_sims, n))
    noisy[rng.random(noisy.shape) < p_out[None, :]] = -1e9
    order = np.argsort(-noisy, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, n + 1)[None, :], axis=1)
    return dict(expected_pos=ranks.mean(axis=0), win=(ranks == 1).mean(axis=0),
                podium=(ranks <= 3).mean(axis=0), top5=(ranks <= 5).mean(axis=0),
                top10=(ranks <= 10).mean(axis=0))


def probability_metrics(probability, finish, cutoff):
    """Per-driver Brier and log loss for a race-level event (finish <= cutoff)."""
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    y = (np.asarray(finish, dtype=float) <= cutoff).astype(float)
    return dict(brier=float(np.mean((p - y) ** 2)),
                logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))


def calibration_bins(probability, outcome, n_bins=10):
    """Observed event rate and mean probability in fixed-width reliability bins."""
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    y = np.asarray(outcome, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probability and outcome must have the same shape")
    bucket = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bucket == i
        if mask.any():
            rows.append(dict(bin=i, count=int(mask.sum()), mean_probability=float(p[mask].mean()),
                             event_rate=float(y[mask].mean())))
    return pd.DataFrame(rows, columns=["bin", "count", "mean_probability", "event_rate"])


def calibration_error(probability, outcome, n_bins=10):
    """Return expected and maximum absolute calibration error for fixed-width bins."""
    bins = calibration_bins(probability, outcome, n_bins)
    if bins.empty:
        return dict(ece=np.nan, mce=np.nan)
    error = (bins["mean_probability"] - bins["event_rate"]).abs()
    weights = bins["count"] / bins["count"].sum()
    return dict(ece=float((weights * error).sum()), mce=float(error.max()))


def write_calibration_svg(curves, title, path, n_bins=10):
    """Write a dependency-free SVG reliability plot from out-of-sample probabilities.

    `curves` maps a display label to (predicted probabilities, binary outcomes).
    """
    import html

    width, height = 680, 510
    left, top, size = 72, 58, 390
    colors = ("#e4572e", "#f3a712", "#1d7874", "#4c78a8")
    sx = lambda x: left + float(x) * size
    sy = lambda y: top + size - float(y) * size
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{left + size / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600">{html.escape(title)}</text>']
    for value in np.linspace(0, 1, 6):
        x, y = sx(value), sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + size}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + size}" stroke="#f0f1f3"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#4b5563">{value:.1f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{top + size + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#4b5563">{value:.1f}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke="#6b7280" stroke-dasharray="5 5"/>',
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="#6b7280"/>',
        f'<text x="{left + size / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">Mean predicted probability</text>',
        f'<text x="19" y="{top + size / 2}" text-anchor="middle" transform="rotate(-90 19 {top + size / 2})" font-family="sans-serif" font-size="14">Observed event rate</text>',
    ])
    for ix, (label, (probability, outcome)) in enumerate(curves.items()):
        bins = calibration_bins(probability, outcome, n_bins)
        color = colors[ix % len(colors)]
        points = " ".join(f"{sx(row.mean_probability):.1f},{sy(row.event_rate):.1f}" for row in bins.itertuples())
        if points:
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for row in bins.itertuples():
                parts.append(f'<circle cx="{sx(row.mean_probability):.1f}" cy="{sy(row.event_rate):.1f}" r="4" fill="{color}"/>')
        y = top + 18 * ix + 12
        parts.append(f'<line x1="492" y1="{y}" x2="514" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="522" y="{y + 4}" font-family="sans-serif" font-size="12" fill="#374151">{html.escape(label)}</text>')
    parts.append("</svg>")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    return target
