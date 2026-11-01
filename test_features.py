import numpy as np
import pandas as pd

import features
from features import FEATURES, add_features


def test_rolling_mean_matches_a_plain_loop():
    df = pd.DataFrame({"driver": list("AABAABBA"), "x": [1.0, 2, 3, np.nan, 5, 6, 7, 8]})
    got = features._past(df, "driver", "x", 2)
    expect = []                      # rule: mean of the driver's previous 2 races, NaNs skipped
    for i, row in df.iterrows():
        past = df.loc[:i - 1]
        past = past[past["driver"] == row.driver]["x"].tail(2)
        expect.append(past.mean() if past.notna().any() else np.nan)
    assert np.allclose(got.fillna(-1), pd.Series(expect).fillna(-1))


def test_no_future_leaks_into_prerace_features(recent):
    """Scramble every value that is only known AFTER a race: no pre-race feature may change."""
    rng = np.random.default_rng(0)
    last = recent[(recent.year == recent.year.max())]
    target = (int(last.year.iloc[0]), int(sorted(last["round"].unique())[len(last["round"].unique()) // 2]))
    m = (recent.year == target[0]) & (recent["round"] == target[1])
    d2 = recent.copy()
    for c in ["finish", "points", "dnf", "n_stops", "pit_lane_s", "status"]:
        if c in d2:
            d2.loc[m, c] = rng.permutation(d2.loc[m, c].to_numpy())
    a, b = add_features(recent), add_features(d2)
    pick = lambda x: x[(x.year == target[0]) & (x["round"] == target[1])].set_index("driver")[FEATURES].sort_index().astype(float)
    ra, rb = pick(a), pick(b)
    changed = [c for c in FEATURES if not np.allclose(ra[c], rb[c], equal_nan=True)]
    assert changed == []


def test_features_do_not_depend_on_later_races(recent):
    """Features of race k are identical whether or not the races after it exist yet."""
    full = add_features(recent)
    races = sorted({(y, r) for y, r in zip(recent.year, recent["round"])})
    y, r = races[len(races) // 2]
    cut = recent[(recent.year < y) | ((recent.year == y) & (recent["round"] <= r))]
    part = add_features(cut)
    pick = lambda x: x[(x.year == y) & (x["round"] == r)].set_index("driver")[FEATURES].sort_index().astype(float)
    a, b = pick(full), pick(part)
    assert [c for c in FEATURES if not np.allclose(a[c], b[c], equal_nan=True)] == []


def test_grid_and_quali_are_separate_features(recent):
    f = add_features(recent)
    ok = f[["grid", "quali_pos"]].notna().all(axis=1)
    assert np.allclose(f.loc[ok, "grid_pen"], f.loc[ok, "grid"] - f.loc[ok, "quali_pos"])
    assert (f.loc[ok, "grid_pen"] != 0).any()          # penalties / pit-lane starts exist in real data


def test_retirement_reasons_are_split():
    df = pd.DataFrame({"status": ["Engine", "Collision", "Collision damage", "Finished", "+1 Lap", "Gearbox", "Retired"],
                       "dnf": [1.0, 1, 1, 0, 0, 1, 1]})
    inc = features._status_flag(df["status"], df["dnf"], features.INCIDENT_WORDS)
    mech = features._status_flag(df["status"], df["dnf"], features.MECH_WORDS) * (1 - inc)
    assert inc.tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert mech.tolist() == [1, 0, 0, 0, 0, 1, 0]       # "Retired" with no reason counts as neither


def test_rookies_start_from_their_teams_form(raw):
    f = add_features(raw[raw.year >= 2022])
    first = f[f["drv_races"] == 0]
    later = first[first.year > 2022]
    assert len(later) > 0
    assert later["drv_finish_l5"].notna().mean() > 0.9   # filled from team form, not left empty


def test_regulation_change_resets_team_ratings(raw):
    base = raw[raw.year.between(2020, 2023)]
    a = add_features(base, dict(elo_reg_shrink=0.0))
    b = add_features(base, dict(elo_reg_shrink=0.9))
    first22 = lambda f: f[(f.year == 2022) & (f["round"] == 1)].team_elo.std()
    assert first22(b) < first22(a) * 0.8


def test_weather_modes(recent):
    d = recent.copy()
    d["rain"], d["q_rain"], d["air_temp"], d["q_air_temp"] = 1.0, 0.0, 30.0, 20.0
    pre, race = add_features(d), add_features(d, dict(weather="race"))
    assert (pre.loc[pre.finish.notna(), "rain"] == 0.0).all()       # no in-race information for finished races
    assert (race.loc[race.finish.notna(), "rain"] == 1.0).all()     # the old (leaky) behaviour is still available
    live = d.copy().iloc[-20:]
    live[["finish", "dnf", "points"]] = np.nan
    live["rain"] = 0.7                                              # a forecast written by extras.apply_live
    both = add_features(pd.concat([d.iloc[:-20], live], ignore_index=True))
    assert (both[both.finish.isna()].rain == 0.7).all()             # forecasts are never overwritten
