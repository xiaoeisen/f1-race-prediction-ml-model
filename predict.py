# Produce finishing-order probabilities from the trained ranking model.
import argparse

import joblib
import numpy as np
import pandas as pd

import extras

from features import add_features
from metrics import race_outcome_probabilities


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
    # Share the seeded simulator with training-time probability calibration.
    p_out = (cur["drv_inc_l20"].fillna(0.06) + cur["team_mech_l10"].fillna(0.08)).clip(0.03, 0.35).to_numpy()
    probs = race_outcome_probabilities(scores, p_out, bundle["temperature"], n_sims, seed)
    cur["exp_pos"] = probs["expected_pos"]
    cur["win_%"] = probs["win"] * 100
    cur["podium_%"] = probs["podium"] * 100
    cur["top10_%"] = probs["top10"] * 100
    cur = cur.sort_values("score", ascending=False).reset_index(drop=True)
    cur["pred_pos"] = np.arange(1, len(cur) + 1)
    return cur[["pred_pos", "driver", "full_name", "team", "grid", "quali_pos", "q_best_gap",
                "fp1_pos", "fp1_best_gap", "fp2_pos", "fp2_best_gap", "fp3_pos", "fp3_best_gap",
                "fp1_best_time_s", "fp1_gap_s", "fp1_laps", "fp2_best_time_s", "fp2_gap_s", "fp2_laps",
                "fp3_best_time_s", "fp3_gap_s", "fp3_laps", "q_best_time_s", "q_gap_s", "q_time_segment_valid", "quali_laps",
                "fp1_long_gap", "fp2_long_gap", "fp1_deg_per10", "fp2_deg_per10", "fp3_deg_per10",
                "race_pace_gap", "race_deg_per10", "race_best_lap_s", "race_avg_clean_lap_s", "race_laps",
                "exp_pos", "win_%", "podium_%", "top10_%", "score"]
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
