"""Shared fixtures. The tests use the project's REAL data (data/races.csv) wherever a model or feature
needs data, and tiny inline frames only for pure functions. Run from the project folder:  pytest"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "races.csv"


@pytest.fixture(scope="session")
def raw():
    if not DATA.exists():
        pytest.skip("data/races.csv missing (run: python collect.py --f1db 2014)")
    return pd.read_csv(DATA)


@pytest.fixture(scope="session")
def recent(raw):
    """Last few seasons: enough to test features quickly."""
    return raw[raw["year"] >= raw["year"].max() - 2].copy()


@pytest.fixture
def workdir(tmp_path, monkeypatch, raw):
    """A scratch project folder (data/ + models/) so tests never touch the real files."""
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    raw.to_csv(tmp_path / "data" / "races.csv", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path
