"""Economic accounting and causality checks for the expanded paper screen."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/paper151"))
from expanded_study import SINGLE, causal_knn, model_features, portfolio_target, prepare, rank, simulate
from run_study import panel_from


@pytest.fixture
def frame():
    n = 1050
    x = np.arange(n)
    close = 100 * np.exp(0.0003 * x + 0.08 * np.sin(x / 19))
    ts = pd.date_range("2014-01-01", periods=n, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "bar_end": ts + pd.Timedelta(days=1),
            "open": close * 1.003,
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "volume": 1000 + 100 * np.sin(x / 9),
            "dividend": np.zeros(n),
            "split": np.zeros(n),
        }
    )


@pytest.mark.parametrize("rule", [s for s in SINGLE if s != "KNN"])
def test_future_prices_cannot_change_past_fills_or_equity(frame, rule):
    p = panel_from({"A": frame}, False, 365)
    d, _ = prepare(frame, p, True, False)
    _, h, _, _ = simulate(p, {"A": frame}, d, rule, 300, 1000, 0.0005, 0.0005)
    changed = frame.copy()
    changed.loc[851:, ["open", "close", "high", "low"]] *= 5
    p2 = panel_from({"A": changed}, False, 365)
    d2, _ = prepare(changed, p2, True, False)
    _, h2, _, _ = simulate(p2, {"A": changed}, d2, rule, 300, 1000, 0.0005, 0.0005)
    pd.testing.assert_frame_equal(h.iloc[:552], h2.iloc[:552])


def test_buyhold_fees_both_sides_and_gap_fills(frame):
    f = frame.copy()
    f["open"] = 100.0
    f["close"] = 120.0
    f["high"] = 121.0
    f["low"] = 99.0
    p = panel_from({"A": f}, False, 365)
    metrics, h, fills, _ = simulate(p, {"A": f}, {}, "BuyHold", 300, 310, 0.01, 0.02)
    expected = 10000 / (100 * 1.02 * 1.01) * 120 * 0.98 * 0.99
    assert h.equity.iloc[-1] == pytest.approx(expected, abs=1e-8)
    assert metrics["return"] == pytest.approx(expected / 10000 - 1)
    assert fills[0]["price"] == pytest.approx(102)
    assert fills[-1]["price"] == pytest.approx(117.6)
    assert h.units_A.iloc[-1] == pytest.approx(0)


def test_model_labels_wait_until_execution_prices_are_known(frame):
    p = panel_from({"A": frame}, False, 365)
    x, _, y = model_features(frame, p)
    _, expected = causal_knn(x, y, 0.002)
    changed = y.copy()
    changed[599:] = 1000000
    xx = x.copy()
    xx[601:] *= 100
    _, actual = causal_knn(xx, changed, 0.002)
    # At decision600 only training example indices <=598 have mature labels.
    assert actual[600] == pytest.approx(expected[600])
    assert y[598] == pytest.approx(frame.open.iloc[600] / frame.open.iloc[599] - 1)


def test_entry_dividend_excluded_and_exit_dividend_paid(frame):
    f = frame.copy()
    f[["open", "close"]] = 100.0
    f["high"] = 101.0
    f["low"] = 99.0
    f.loc[300:302, "dividend"] = 1.0
    p = panel_from({"A": f}, False, 365)
    metrics, h, _, _ = simulate(p, {"A": f}, {}, "BuyHold", 300, 302, 0, 0)
    assert metrics["dividends"] == pytest.approx(201)
    assert h.equity.iloc[-1] == pytest.approx(10201)


def test_portfolio_caps_and_causality(frame):
    names = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY", "IEF"]
    frames = {s: frame.copy() for s in names}
    p = panel_from(frames, False)
    for rule in [
        "SectorRotation",
        "SectorRotationMA",
        "SectorRotationDual",
        "SectorMomentumSkip21",
        "SectorLowVol",
        "SectorIBS",
        "SectorContrarian",
    ]:
        w = portfolio_target(rule, 500, p, frames)
        assert np.isfinite(w).all() and min(w) >= 0 and sum(w) <= 1 + 1e-12
        p.total_index[501:] *= 2
        np.testing.assert_array_equal(w, portfolio_target(rule, 500, p, frames))


def test_ranking_does_not_turn_underperformers_into_winners():
    rows = []
    for strategy, excess in [("SMA200", -0.01), ("MA50_200", -0.02)]:
        for window, case in [("full", "base"), ("full", "double_cost"), ("2023_latest", "base")]:
            rows.append(
                {
                    "strategy": strategy,
                    "market": "A",
                    "window": window,
                    "case": case,
                    "pass": False,
                    "excess_cagr": excess,
                    "drawdown_deterioration": -0.01,
                }
            )
    r = rank(pd.DataFrame(rows))
    assert not r.qualified.any()
    assert r.strategy.iloc[0] == "SMA200"
