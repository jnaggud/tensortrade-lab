import copy
import json

import numpy as np
import pandas as pd
import pytest

from tensortrade_lab import rotation_forward as forward
from tensortrade_lab.config import Market
from tensortrade_lab.market_data import expected_grid
from tensortrade_lab.portfolio import make_panel


@pytest.fixture
def paper_setup(tmp_path, monkeypatch):
    grid = expected_grid("2024-01-01", "2026-10-10", Market(calendar="XNYS", bar_minutes=1440))
    frames = {
        s: grid.assign(
            open=100 + np.arange(len(grid)) * (j + 1) / 100,
            close=101 + np.arange(len(grid)) * (j + 1) / 100,
            dividend=0.0,
        )
        for j, s in enumerate(("A", "B", "C", "SPY"))
    }
    rates = pd.DataFrame(
        {"date": pd.date_range("2023-12-01", "2026-10-10", freq="B").astype(str), "discount_yield": 0.04}
    )
    panel = make_panel(frames, rates)
    holder = {"panel": panel.prefix("2026-09-19")}
    monkeypatch.setattr(forward, "load_panel", lambda spec: (holder["panel"], {}))
    spec = {
        "universe": ["A", "B", "C"],
        "benchmark": "SPY",
        "skip_recent": 21,
        "top_k": 3,
        "commission": 0.0005,
        "slippage": 0.0005,
        "initial_cash": 10000.0,
        "data_directory": str(tmp_path),
    }
    selection = {
        "candidate": {
            "name": "test",
            "family": "momentum",
            "frequency": "weekly",
            "lookback": 252,
            "weighting": "equal",
        }
    }
    for name, obj in {
        "results": {},
        "protocol": {"engine_sha256": forward.engine_hash()},
        "selection": selection,
        "preregistration": {"spec": spec},
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(obj))
    return tmp_path, panel, holder


def test_paper_records_before_open_and_is_idempotent(paper_setup):
    study, panel, holder = paper_setup
    first = forward.poll(study, now="2026-09-18T21:00:00Z")
    assert first["recorded_decisions"] == 1
    assert first["observed_sessions"] == 0
    again = forward.poll(study, now="2026-09-18T21:01:00Z")
    assert again["recorded_decisions"] == 1
    holder["panel"] = panel.prefix("2026-09-23")
    later = forward.poll(study, data_directory=study, now="2026-09-22T21:00:00Z")
    assert later["observed_sessions"] == 2
    assert later["metrics"]["spy"]["start"] == later["metrics"]["strategy"]["start"]
    assert not later["deployment_approved"]
    # No poll recorded the September 25 rebalance; Monday must hold rather than backfill.
    holder["panel"] = panel.prefix("2026-09-29")
    missed = forward.poll(study, data_directory=study, now="2026-09-28T21:00:00Z")
    assert missed["missed_rebalances"] == 1
    assert missed["recorded_decisions"] == 1


def test_paper_does_not_backfill_missed_first_open(paper_setup):
    study, _, _ = paper_setup
    status = forward.poll(study, now="2026-09-21T15:00:00Z")
    assert status["recorded_decisions"] == 0
    assert status["observed_sessions"] == 0


def test_paper_rejects_revised_history(paper_setup):
    study, _, holder = paper_setup
    forward.poll(study, now="2026-09-18T21:00:00Z")
    changed = copy.deepcopy(holder["panel"])
    changed.close[100, 0] += 1
    holder["panel"] = changed
    with pytest.raises(ValueError, match="history changed"):
        forward.poll(study, now="2026-09-18T21:01:00Z")


def test_paper_rejects_backdated_record(paper_setup):
    study, _, _ = paper_setup
    forward.poll(study, now="2026-09-18T21:00:00Z")
    path = study / "prospective/decisions.json"
    decisions = json.loads(path.read_text())
    decisions[0]["recorded_at"] = "2026-09-21T15:00:00Z"
    path.write_text(json.dumps(decisions))
    with pytest.raises(ValueError, match="backfilled"):
        forward.poll(study, now="2026-09-21T16:00:00Z")
