"""Pre-race adjustments for a weekend that hasn't been raced yet:
  1. the starting grid: official grid (OpenF1 / API-Sports) if published, else penalties.csv
  2. race-day weather forecast (replaces observed weather, which isn't known yet)

penalties.csv columns: year,round,driver,places
    places = grid places dropped (5 = five-place drop). Use 99 for a back-of-grid / pit-lane start.
    Example:  2025,20,VER,10
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import get_forecast
from sources import enrich

PENALTIES = Path(__file__).resolve().parent / "penalties.csv"


def apply_penalties(rows, year, rnd, path=None):
    """Modifies rows['grid'] in place. Returns the number of penalised drivers."""
    path = Path(path or PENALTIES)        # resolved at call time so the file location can be overridden
    if not path.exists() or rows["grid"].isna().all():
        return 0
    p = pd.read_csv(path)
    if p.empty:
        return 0
    p = p[(p["year"] == year) & (p["round"] == rnd)]
    if p.empty:
        return 0
    places = rows["driver"].map(dict(zip(p["driver"].astype(str).str.upper(), p["places"]))).fillna(0)
    n = int((places > 0).sum())
    if n == 0:
        return 0
    # a driver dropping k places lands just behind whoever is k places further back
    key = rows["grid"] + places + np.where(places > 0, 0.5, 0.0)
    rows["grid"] = key.rank(method="first")
    return n


def apply_live(rows, year, rnd, ev):
    """Prepare an upcoming weekend (no-op for finished races):
      1. starting grid: the OFFICIAL grid (penalties included) from OpenF1 / API-Sports if published,
         else qualifying order + penalties.csv
      2. weather: race-hour forecast replaces the not-yet-observed weather"""
    info = dict(forecast=None, penalties=0, grid_source=None)
    if rows["finish"].notna().any():
        return info
    grid, src = enrich.official_grid(year, rnd, ev["EventDate"])
    if grid:
        mapped = rows["driver"].map(grid)
        if mapped.notna().sum() >= 0.75 * len(rows):
            rows["grid"] = mapped.fillna(99).rank(method="first")  # unlisted (pit lane) -> back
            info["grid_source"] = src
    if info["grid_source"] is None:
        info["penalties"] = apply_penalties(rows, year, rnd)
        if info["penalties"]:
            info["grid_source"] = "penalties.csv"
        elif rows["grid"].notna().any():
            info["grid_source"] = "qualifying"
    fc = get_forecast(ev["Location"], pd.Timestamp(ev["EventDate"]).date(), enrich.race_start(year, rnd, ev))
    if fc:
        rows["rain"] = fc["rain_prob"]
        rows["air_temp"] = fc["air_temp"]
        info["forecast"] = fc
    return info
