import json

import joblib
import numpy as np
import pandas as pd
import pytest

import features
import model
import train
from features import FEATURES, PL_FEATURES, add_features


@pytest.fixture(scope="module")
def table(raw):
    f = add_features(raw[raw.year >= 2018])
    return f[f.finish.notna()].reset_index(drop=True)


def test_ensemble_scores_every_driver_and_survives_pickling(table, tmp_path):
    tr, te = table[table.year < 2025], table[table.year == 2025].copy()
    ens = model.RankEnsemble(FEATURES, PL_FEATURES, seeds=(1,)).fit(tr)
    s = ens.predict(te)
    assert len(s) == len(te) and np.isfinite(s).all()
    joblib.dump(ens, tmp_path / "m.joblib")
    assert np.allclose(joblib.load(tmp_path / "m.joblib").predict(te), s)           # same answers after reload
    one = te[te.race_id == te.race_id.iloc[0]]
    assert np.allclose(ens.predict(one), s[(te.race_id == te.race_id.iloc[0]).to_numpy()])   # a single race scores the same alone
    assert ens.beta_grid() < 0 if hasattr(ens, "beta_grid") else True


def test_linear_part_learns_that_a_worse_grid_means_a_worse_finish(table):
    ens = model.RankEnsemble(FEATURES, PL_FEATURES, seeds=(1,)).fit(table)
    assert ens.pl_beta[PL_FEATURES.index("grid_log")] < 0


def test_pre_qualifying_model_has_no_grid_prior(table):
    ens = model.RankEnsemble(features.FEATURES_PRE, features.PL_FEATURES_PRE, prior_weight=0.0, seeds=(1,)).fit(table[table.year < 2025])
    assert "grid" not in ens.components(table[table.year == 2025])


def test_ensemble_beats_chance_on_the_latest_season(table):
    """Loose sanity bar (not the accuracy claim): ordering clearly better than random."""
    import metrics
    tr, te = table[table.year < 2025], table[table.year == 2025].copy()
    te["s"] = model.RankEnsemble(FEATURES, PL_FEATURES, seeds=(1,)).fit(tr).predict(te)
    m = metrics.evaluate(te, "s")
    assert m["spearman"] > 0.4 and m["top5"] > 0.6


def test_temperature_calibration_tracks_score_scale():
    rng = np.random.default_rng(0)
    items = []
    for _ in range(80):
        skill = rng.normal(size=20)
        finish = np.argsort(np.argsort(-(skill + rng.gumbel(size=20)))) + 1.0     # a Plackett-Luce race with true T = 1
        items.append((skill, finish))
    t1 = train.calibrate_temperature(items)
    t2 = train.calibrate_temperature([(s * 2, f) for s, f in items])
    assert 0.6 < t1 < 1.8 and abs(t2 - 2 * t1) < 0.5


@pytest.mark.slow
def test_training_promotes_first_model_and_gate_can_reject(workdir, raw):
    rep = train.train_models(raw, train_from=2018, models_dir="models", verbose=False)
    assert rep["promoted"] and rep["version"] == 1
    for f in ("ranker.joblib", "ranker_holdout.joblib", "ranker_pre.joblib", "ranker_pre_holdout.joblib", "metrics.json", "version.json"):
        assert (workdir / "models" / f).exists()
    b = joblib.load(workdir / "models" / "ranker.joblib")
    assert 0.3 <= b["temperature"] <= 4.0 and b["stage"] == "full" and b["trained_through"][0] >= 2025
    old = train.GATE_MARGIN
    train.GATE_MARGIN = -5.0                                     # nothing can pass: the live model must stay
    try:
        rep2 = train.train_models(raw, train_from=2018, models_dir="models", verbose=False)
    finally:
        train.GATE_MARGIN = old
    assert not rep2["promoted"] and json.loads((workdir / "models" / "version.json").read_text())["version"] == 1
    log = [json.loads(x) for x in (workdir / "data" / "training_log.jsonl").read_text().splitlines()]
    assert [x["promoted"] for x in log] == [True, False]
