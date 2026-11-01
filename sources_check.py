# Verify configured data sources before collection depends on them.
import warnings

import fastf1
import pandas as pd

from sources import apisports, hyprace, jolpica, openf1

warnings.filterwarnings("ignore")


def show(name, fn):
    try:
        out = fn()
    except Exception as e:
        print(f"[FAIL   ] {name}: {type(e).__name__}: {str(e)[:120]}")
        return
    empty = out is None or (hasattr(out, "__len__") and len(out) == 0)
    print(f"[{'EMPTY  ' if empty else 'OK     '}] {name}: {'nothing returned' if empty else out}")


def main():
    year = pd.Timestamp.now().year
    sched = fastf1.get_event_schedule(year, include_testing=False)
    done = sched[pd.to_datetime(sched["EventDate"]) < pd.Timestamp.now().normalize()]
    if done.empty:
        year -= 1
        done = fastf1.get_event_schedule(year, include_testing=False)
    ev = done.iloc[-1]
    rnd, date = int(ev["RoundNumber"]), ev["EventDate"]
    print(f"Testing with {year} round {rnd}: {ev['EventName']}\n")

    show("Jolpica results + qualifying",
         lambda: (lambda d: f"{len(d)} rows, e.g. {d.iloc[0]['driver']} P{int(d.iloc[0]['finish'])} from grid {int(d.iloc[0]['grid'])}")(
             jolpica.weekend_rows(year, rnd)))
    show("Jolpica pit stops", lambda: (lambda p: None if p is None else f"{len(p)} drivers, {p['n_stops'].mean():.1f} stops avg")(
        jolpica.pitstops_frame(year, rnd, {r["Driver"]["driverId"]: jolpica._code(r["Driver"]) for r in
                                           jolpica.client.get(f"/{year}/{rnd}/results.json", {"limit": 100})["MRData"]["RaceTable"]["Races"][0]["Results"]})))

    def of1():
        ex, rl = openf1.race_extras(date)
        return None if ex is None else (f"{len(ex)} drivers, pit-lane {ex['pit_lane_s'].mean():.1f}s, "
                                        f"stationary {ex['stop_s'].mean():.1f}s, safety cars {rl.get('n_sc')}")
    show("OpenF1 pit / stints / overtakes / race control", of1)
    show("OpenF1 official starting grid", lambda: (lambda g: None if not g else f"{len(g)} drivers, pole = {min(g, key=g.get)}")(openf1.starting_grid(date)))

    if apisports.available():
        show("API-Sports starting grid", lambda: (lambda g: None if not g else f"{len(g)} drivers")(apisports.starting_grid(date)))
    else:
        print("[SKIPPED] API-Sports: set APISPORTS_KEY to test it")
    if hyprace.available():
        show("Hyprace race start (current season only)", lambda: hyprace.race_start_utc(year, rnd))
    else:
        print("[SKIPPED] Hyprace: set RAPIDAPI_KEY to test it")


if __name__ == "__main__":
    main()
