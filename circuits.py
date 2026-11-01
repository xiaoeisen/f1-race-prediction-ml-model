"""Fixed circuit facts (street circuit?, pit-lane time loss, tyre wear, altitude, downforce level) and coordinates.

Data lives in circuits.csv (approximate starting values, edit freely). Rows are matched
on FastF1's circuit 'Location' after lower-casing and stripping accents/punctuation, so
"Montréal" -> "montreal". If a location isn't found its facts are simply left empty.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent / "circuits.csv"
FACT_COLS = ["street", "pit_loss", "tyre_wear", "altitude", "downforce", "length_km", "turns"]
_table = None


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def table():
    global _table
    if _table is None:
        _table = pd.read_csv(CSV).drop_duplicates("key").set_index("key")
    return _table


def lookup(location):
    t = table()
    k = norm(location)
    return t.loc[k] if k in t.index else None


def canon(name):
    """One stable id per circuit, whichever source spelled the name ('Spa' / 'Spa-Francorchamps')."""
    t = table()
    k = norm(name)
    return t.loc[k, "canon"] if k in t.index else k


def circuit_facts(locations):
    """DataFrame (street, pit_loss, tyre_wear) aligned with a Series of locations."""
    t = table()
    keys = locations.map(norm)
    out = pd.DataFrame(index=locations.index, columns=FACT_COLS, dtype=float)
    for c in FACT_COLS:
        out[c] = keys.map(t[c]).astype(float).to_numpy() if len(keys) else np.array([])
    return out


def unmatched(locations):
    t = table()
    return sorted({str(x) for x in locations if norm(x) not in t.index})
