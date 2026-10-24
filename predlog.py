# Save predictions before races and score them after results arrive.
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

LOG = Path("data") / "predictions.jsonl"
STAGE_ORDER = {"pre": 1, "quali": 2}
STAGE_LABEL = {"pre": "After practice", "quali": "After qualifying"}
TOP_N = 5


def load(path=None):
    path = Path(path or LOG)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def save(entries, path=None):
    path = Path(path or LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(json.dumps(e) for e in entries) + ("\n" if entries else ""))
    os.replace(tmp, path)


def _key(e):
    return (int(e["year"]), int(e["round"]), e["stage"])


def upsert(entry, path=None):
    """One snapshot per (race, stage): newer replaces older, but a scored one is never overwritten."""
    entries = load(path)
    for i, e in enumerate(entries):
        if _key(e) == _key(entry):
            if e.get("metrics"):
                return False
            entries[i] = entry
            break
    else:
        entries.append(entry)
    save(entries, path)
    return True


def for_race(year, rnd, entries=None):
    return [e for e in (entries if entries is not None else load()) if e["year"] == year and e["round"] == rnd]


def make_entry(res, meta, stage, model_version, created):
    """res = DataFrame from predict_race()."""
    res = res.sort_values("pred_pos")
    top = [dict(pos=int(r.pred_pos), driver=r.driver, name=r.full_name, team=r.team,
                win=round(float(r["win_%"]), 1), podium=round(float(r["podium_%"]), 1))
           for _, r in res.head(TOP_N).iterrows()]
    return dict(
        year=int(meta["year"]), round=int(meta["round"]), event=meta["event"], stage=stage,
        created=str(created), model_version=model_version, grid_source=meta.get("grid_source"),
        forecast=meta.get("forecast"), top5=top, order=res["driver"].tolist(),
        scores={d: round(float(s), 4) for d, s in zip(res["driver"], res["score"])},
        grid={d: float(g) for d, g in zip(res["driver"], res["grid"]) if pd.notna(g)},
        actual=None, metrics=None)


def evaluate(entry, actual_order, grid_map=None):
    """Score one snapshot against the real finishing order (list of driver codes, P1 first)."""
    pred5 = [t["driver"] for t in entry["top5"]]
    act = list(actual_order)
    act5 = act[:TOP_N]
    hits = len(set(pred5) & set(act5))
    exact = sum(p == a for p, a in zip(pred5, act5))
    winner = bool(pred5 and act5 and pred5[0] == act5[0])
    podium = len(set(pred5[:3]) & set(act[:3]))
    pr = {d: i for i, d in enumerate(entry["order"])}
    ar = {d: i for i, d in enumerate(act)}
    common = [d for d in pr if d in ar]
    rho = (pd.Series([pr[d] for d in common]).corr(pd.Series([ar[d] for d in common]), method="spearman")
           if len(common) > 2 else np.nan)
    base5 = [d for d, _ in sorted((grid_map or {}).items(), key=lambda kv: kv[1])][:TOP_N]
    base_hits = len(set(base5) & set(act5)) if base5 else None
    return dict(
        top5_hits=hits, top5_pct=100.0 * hits / TOP_N, exact=exact, exact_pct=100.0 * exact / TOP_N,
        winner=winner, podium_hits=podium, rank_corr=None if pd.isna(rho) else round(float(rho), 3),
        baseline_top5_hits=base_hits, baseline_top5_pct=None if base_hits is None else 100.0 * base_hits / TOP_N,
        score=round(100 * (0.5 * hits / TOP_N + 0.3 * exact / TOP_N + 0.2 * winner)))


def finalize(year, rnd, rows, path=None):
    """Attach the real result to every snapshot of this race and score it."""
    done = rows[rows["finish"].notna()].sort_values("finish")
    order = done["driver"].tolist()
    grid_map = {d: g for d, g in zip(rows["driver"], rows["grid"]) if pd.notna(g)}
    names = dict(zip(rows["driver"], rows.get("full_name", rows["driver"])))
    teams = dict(zip(rows["driver"], rows["team"]))
    actual = dict(order=order[:10], top5=[dict(pos=i + 1, driver=d, name=names.get(d, d), team=teams.get(d, ""))
                                         for i, d in enumerate(order[:TOP_N])])
    entries = load(path)
    touched = []
    for e in entries:
        if e["year"] == year and e["round"] == rnd:
            e["actual"] = actual
            e["metrics"] = evaluate(e, order, grid_map)
            touched.append(e)
    if touched:
        save(entries, path)
    return touched


def final_per_race(entries):
    """The latest-stage scored snapshot of each race."""
    best = {}
    for e in entries:
        if e.get("metrics"):
            k = (e["year"], e["round"])
            if k not in best or STAGE_ORDER[e["stage"]] > STAGE_ORDER[best[k]["stage"]]:
                best[k] = e
    return [best[k] for k in sorted(best)]


def summary(entries):
    races = final_per_race(entries)
    if not races:
        return dict(n_races=0, series=[])
    m = [r["metrics"] for r in races]
    base = [x["baseline_top5_pct"] for x in m if x["baseline_top5_pct"] is not None]
    return dict(
        n_races=len(races),
        avg_top5_pct=round(float(np.mean([x["top5_pct"] for x in m])), 1),
        avg_exact_pct=round(float(np.mean([x["exact_pct"] for x in m])), 1),
        winner_pct=round(100 * float(np.mean([x["winner"] for x in m])), 1),
        avg_score=round(float(np.mean([x["score"] for x in m])), 1),
        baseline_top5_pct=round(float(np.mean(base)), 1) if base else None,
        last10_top5_pct=round(float(np.mean([x["top5_pct"] for x in m[-10:]])), 1),
        series=[dict(year=r["year"], round=r["round"], event=r["event"], stage=r["stage"],
                     top5_pct=r["metrics"]["top5_pct"], exact_pct=r["metrics"]["exact_pct"],
                     baseline_top5_pct=r["metrics"]["baseline_top5_pct"], score=r["metrics"]["score"],
                     winner=r["metrics"]["winner"]) for r in races])


def race_weights(entries):
    """Self-training emphasis: races the model got badly wrong count up to 1.5x next time."""
    return {(r["year"], r["round"]): 1.0 + 0.5 * (1.0 - r["metrics"]["top5_pct"] / 100.0)
            for r in final_per_race(entries)}


def calibration_items(entries, stage):
    """(scores, finishing rank) per scored snapshot of this stage, to calibrate probabilities.
    Only the top 10 of the real order is stored; the rest get a shared worse rank."""
    items = []
    for e in entries:
        if e["stage"] != stage or not e.get("metrics") or not e.get("actual"):
            continue
        order = e["actual"]["order"]
        drivers = list(e["scores"])
        if len(order) >= 3 and all(d in drivers for d in order[:3]):
            rank = {d: i + 1 for i, d in enumerate(order)}
            items.append((np.array([e["scores"][d] for d in drivers], float), np.array([rank.get(d, 15) for d in drivers], float)))
    return items
