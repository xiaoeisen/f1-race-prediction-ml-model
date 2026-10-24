# Combine tree, linear, and grid signals to rank drivers within a race.
import numpy as np
import pandas as pd
import sklearn  # registers its packaged OpenMP runtime before LightGBM loads on Windows
import lightgbm as lgb
from scipy.optimize import minimize

GBM_PARAMS = dict(objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=7,
                  min_child_samples=40, colsample_bytree=0.6, reg_lambda=5.0, subsample=0.8,
                  subsample_freq=1, lambdarank_truncation_level=10, verbose=-1)
LABEL_GAIN = [int(round((lvl / 20) ** 2 * 100)) for lvl in range(31)]   # top positions matter most
SEEDS = (1, 2, 3)


def labels(df):
    return (21 - df["finish"]).clip(lower=0, upper=30).astype(int)  # P1 -> 20, P20 -> 1


def _z(values, race):
    s = pd.Series(values)
    g = s.groupby(np.asarray(race))
    return ((s - g.transform("mean")) / (g.transform("std").fillna(1.0) + 1e-9)).to_numpy()


def _pl_matrix(df, cols, stats=None):
    """Impute (race median, then 0) and standardise. Returns (matrix, stats)."""
    X = df[cols].astype(float)
    X = X.fillna(X.groupby(df["race_id"]).transform("median")).fillna(0.0)
    if stats is None:
        stats = (X.mean().to_numpy(), X.std().replace(0, 1.0).fillna(1.0).to_numpy())
    return ((X.to_numpy() - stats[0]) / stats[1]), stats


def fit_plackett_luce(X, race, order_key, weights, lam=30.0, top=10):
    """Maximise the Plackett-Luce likelihood of the top `top` finishers of every race (L2 penalty)."""
    races = np.unique(race)
    n_max = max(int((race == r).sum()) for r in races)
    d = X.shape[1]
    Xp = np.zeros((len(races), n_max, d))
    mask = np.zeros((len(races), n_max), bool)
    w = np.zeros(len(races))
    for i, r in enumerate(races):
        idx = np.flatnonzero(race == r)
        idx = idx[np.argsort(order_key[idx], kind="stable")]       # finishing order, P1 first
        Xp[i, : len(idx)] = X[idx]
        mask[i, : len(idx)] = True
        w[i] = weights[idx].mean()
    k = min(top, n_max)
    tri = np.arange(n_max)[None, :] >= np.arange(k)[:, None]         # j >= k

    def objective(beta):
        s = np.where(mask, Xp @ beta, -1e9)
        lse = np.logaddexp.accumulate(s[:, ::-1], axis=1)[:, ::-1]   # log-sum-exp over positions k..end
        valid = mask[:, :k] * w[:, None]
        ll = ((s[:, :k] - lse[:, :k]) * valid).sum()
        p = np.exp(s[:, None, :] - lse[:, :k, None]) * tri[None] * mask[:, None, :]
        grad = ((Xp[:, :k] - np.einsum("rkn,rnd->rkd", p, Xp)) * valid[:, :, None]).sum((0, 1))
        return -ll + lam * beta @ beta, -grad + 2 * lam * beta

    return minimize(objective, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


class RankEnsemble:
    """fit(df) / predict(df): df needs a race_id column. Higher score = finishes better."""
    needs_frame = True  # predict() takes the whole feature frame, not just model.features

    def __init__(self, gbm_features, pl_features, prior_weight=1.0, seeds=SEEDS, pl_lam=30.0):
        self.gbm_features, self.pl_features = list(gbm_features), list(pl_features)
        self.prior_weight, self.seeds, self.pl_lam = prior_weight, tuple(seeds), pl_lam

    def fit(self, df, weight_col=None):
        df = df.sort_values("race_id")
        w = df[weight_col].to_numpy(float) if weight_col else np.ones(len(df))
        groups = df.groupby("race_id").size().to_numpy()
        y = labels(df)
        self.gbms = []
        for s in self.seeds:
            m = lgb.LGBMRanker(label_gain=LABEL_GAIN, random_state=s, **GBM_PARAMS)
            m.fit(df[self.gbm_features], y, group=groups, sample_weight=w)  # weights are constant within a race
            self.gbms.append(m)
        X, self.pl_stats = _pl_matrix(df, self.pl_features)
        self.pl_beta = fit_plackett_luce(X, df["race_id"].to_numpy(), df["finish"].to_numpy(float), w, self.pl_lam)
        return self

    def components(self, df):
        race = df["race_id"].to_numpy()
        gbm = np.mean([m.predict(df[self.gbm_features]) for m in self.gbms], axis=0)
        X, _ = _pl_matrix(df, self.pl_features, self.pl_stats)
        parts = {"trees": _z(gbm, race), "linear": _z(X @ self.pl_beta, race)}
        if self.prior_weight:
            parts["grid"] = self.prior_weight * _z(-df["grid_log"].fillna(df["grid_log"].max()).to_numpy(float), race)
        return parts

    def predict(self, df):
        return np.sum(list(self.components(df).values()), axis=0)

    def feature_importance(self):
        gain = np.mean([m.booster_.feature_importance("gain") for m in self.gbms], axis=0)
        return pd.Series(gain, index=self.gbm_features).sort_values(ascending=False)
