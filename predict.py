# Produce finishing-order probabilities from the trained ranking model.
import argparse

import joblib
import numpy as np
import pandas as pd

import extras

from features import add_features


def predict_race(hist, new, bundle, n_sims=20000, seed=0):
    """hist: past races (with results). new: rows for the race to predict."""
    year, rnd = int(new["year"].iloc[0]), int(new["round"].iloc[0])
    hist = hist[~((hist["year"] == year) & (hist["round"] == rnd))]
    df = add_features(pd.concat([hist, new], ignore_index=True))
    cur = df[(df["year"] == year) & (df["round"] == rnd)].copy()

    model = bundle["model"]
    scores = model.predict(cur) if getattr(model, "needs_frame", False) else model.predict(cur[bundle["features"]])
    cur["score"] = scores

    # Plackett-Luce Monte Carlo: sample many plausible finishing orders
    rng = np.random.default_rng(seed)
    noisy = scores / bundle["temperature"] + rng.gumbel(size=(n_sims, len(cur)))
    # retirements: a driver's chance = own error-proneness + car unreliability (history; ~15% on average).
    # A retired driver drops to the back, so nobody gets a 99.9% podium chance.
    p_out = (cur["drv_inc_l20"].fillna(0.06) + cur["team_mech_l10"].fillna(0.08)).clip(0.03, 0.35).to_numpy()
    noisy = np.where(rng.random(noisy.shape) < p_out, -1e9, noisy)
    ranks = (-noisy).argsort(axis=1).argsort(axis=1) + 1   # 1 = winner

    cur["exp_pos"] = ranks.mean(axis=0)
    cur["win_%"] = (ranks == 1).mean(axis=0) * 100
    cur["podium_%"] = (ranks <= 3).mean(axis=0) * 100
    cur["top10_%"] = (ranks <= 10).mean(axis=0) * 100
    cur = cur.sort_values("score", ascending=False).reset_index(drop=True)
    cur["pred_pos"] = np.arange(1, len(cur) + 1)
    return cur[["pred_pos", "driver", "full_name", "team", "grid", "quali_pos", "q_best_gap",
                "fp1_pos", "fp1_best_gap", "fp2_pos", "fp2_best_gap", "fp3_pos", "fp3_best_gap", "exp_pos", "win_%", "podium_%", "top10_%", "score"]
               + (["finish"] if cur["finish"].notna().any() else [])]


def main():
    from collect import OUT, collect_weekend  # lazy: needs fastf1 + internet
    import fastf1

    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--grid", default=None, help="comma-separated driver codes in grid order")
    ap.add_argument("--model", default="models/ranker.joblib")
    args = ap.parse_args()

    hist = pd.read_csv(OUT)
    sched = fastf1.get_event_schedule(args.year, include_testing=False).set_index("RoundNumber")
    new = collect_weekend(args.year, args.round, sched.loc[args.round])
    if new.empty:
        raise SystemExit("No session data available for that weekend yet.")
    pre = not new["quali_pos"].notna().any() and not new["finish"].notna().any()
    path = args.model.replace("ranker.joblib", "ranker_pre.joblib") if pre else args.model
    bundle = joblib.load(path if pre and __import__("os").path.exists(path) else args.model)
    if pre:
        print("Qualifying not available yet: using the practice-only model.")

    ev = sched.loc[args.round]
    live = extras.apply_live(new, args.year, args.round, ev)
    if live["grid_source"]:
        print(f"Starting grid source: {live['grid_source']}")
    if live["penalties"]:
        print(f"Applied {live['penalties']} grid penalties from penalties.csv")
    if live["forecast"]:
        print(f"Using forecast: {live['forecast']['rain_prob']:.0%} chance of rain, "
              f"{live['forecast']['air_temp']:.0f} C high")

    if args.grid:
        order = [d.strip().upper() for d in args.grid.split(",")]
        new["grid"] = new["driver"].map({d: i + 1 for i, d in enumerate(order)}).fillna(20)
    if new["grid"].isna().all():
        print("Note: qualifying not available yet -> grid unknown, prediction is less reliable.")

    res = predict_race(hist, new, bundle)
    print(f"\n{sched.loc[args.round, 'EventName']} {args.year}")
    print(res.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
