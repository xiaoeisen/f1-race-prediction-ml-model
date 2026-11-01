import numpy as np
import pandas as pd

import metrics
import predlog


def _race(order, score):
    return pd.DataFrame({"finish": [float(i + 1) for i in range(len(order))], "score": score, "dnf": 0.0, "race_id": 1,
                         "year": 2025}, index=list(order))


def test_perfect_prediction_scores_one():
    g = _race("ABCDEFGHIJ", np.arange(10, 0, -1))
    m = metrics.race_metrics(g, "score")
    assert m["winner"] == 1 and m["top5"] == 1 and m["podium"] == 1 and m["exact"] == 1
    assert abs(m["ndcg5"] - 1) < 1e-9 and m["mae"] == 0 and m["spearman"] > 0.999


def test_reversed_prediction_is_bad():
    g = _race("ABCDEFGHIJKLMNOPQRST", np.arange(0, 20))      # a full 20-car grid, predicted exactly backwards
    m = metrics.race_metrics(g, "score")
    assert m["winner"] == 0 and m["top5"] == 0 and m["spearman"] < -0.99 and m["ndcg5"] < 0.2


def _entry(top, order):
    return dict(year=2025, round=1, stage="quali", event="X", top5=[{"driver": d} for d in top], order=order,
                scores={d: float(len(order) - i) for i, d in enumerate(order)}, actual=None, metrics=None, created="t")


def test_evaluate_scores_known_cases():
    act = list("ABCDEFG")
    perfect = predlog.evaluate(_entry("ABCDE", act), act, {d: i for i, d in enumerate(act)})
    assert perfect["top5_hits"] == 5 and perfect["exact"] == 5 and perfect["winner"] and perfect["score"] == 100
    assert perfect["baseline_top5_hits"] == 5
    shuffled = predlog.evaluate(_entry("EDCBA", list("EDCBAFG")), act)
    assert shuffled["top5_hits"] == 5 and shuffled["exact"] == 1 and not shuffled["winner"] and shuffled["score"] == 56
    miss = predlog.evaluate(_entry("FGXYZ", list("FGXYZABC")), act)
    assert miss["top5_hits"] == 0 and miss["score"] == 0


def test_log_roundtrip_lock_and_summary(tmp_path):
    p = tmp_path / "p.jsonl"
    assert predlog.upsert(_entry("ABCDE", list("ABCDEFG")), p)
    rows = pd.DataFrame({"driver": list("ABCDEFG"), "finish": [2.0, 1, 3, 4, 6, 5, 7], "grid": [1.0, 2, 3, 4, 5, 6, 7],
                         "team": "T", "full_name": list("ABCDEFG")})
    scored = predlog.finalize(2025, 1, rows, p)
    assert scored[0]["metrics"]["top5_hits"] == 4
    assert predlog.upsert(dict(_entry("EDCBA", list("EDCBAFG")), created="later"), p) is False   # scored entries are locked
    s = predlog.summary(predlog.load(p))
    assert s["n_races"] == 1 and s["avg_top5_pct"] == 80.0
    assert predlog.race_weights(predlog.load(p))[(2025, 1)] == 1.1                                 # 4/5 right -> 1.1x emphasis
    items = predlog.calibration_items(predlog.load(p), "quali")
    assert len(items) == 1 and items[0][1].min() == 1
