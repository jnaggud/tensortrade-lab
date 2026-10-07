"""Economic invariants of short positions and future-data exclusion."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/paper151"))
from run_study import panel_from
from signed_study import GROUPS, RULES, SignedLedger, features, run, target


def panel():
    ts = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"], tz="UTC")
    return SimpleNamespace(
        symbols=["A", "B"],
        timestamp=ts,
        bar_end=ts + pd.Timedelta(hours=6),
        opening=np.full((5, 2), 100.0),
        close=np.full((5, 2), 100.0),
        dividend=np.zeros((5, 2)),
        cash_rate=np.zeros(5),
    )


def test_short_proceeds_are_not_profit_and_costs_reconcile():
    p = panel()
    ledger = SignedLedger(p, 1, 4, fee=0.001, slip=0.002, borrow=0)
    ledger.step(np.array([0.5, -0.5]))
    gross = np.abs(ledger.units * p.opening[1]).sum()
    assert ledger.equity_at(p.opening[1]) == pytest.approx(10000 - ledger.fees - ledger.slippage)
    assert ledger.equity_at(p.opening[1]) == pytest.approx(gross)
    assert ledger.cash > 0 and ledger.free_cash(p.opening[1]) > 0
    for _ in range(3):
        ledger.step()
    assert ledger.units == pytest.approx([0, 0])
    assert ledger.cash == pytest.approx(10000 - ledger.fees - ledger.slippage)


def test_short_ex_date_entitlement_and_weekend_borrow():
    p = panel()
    p.dividend[1] = [0, 20]
    p.dividend[2] = [0, 2]
    ledger = SignedLedger(p, 1, 4, fee=0, slip=0, borrow=0.0365)
    ledger.step(np.array([0.5, -0.5]))
    assert ledger.short_dividends == 0  # opened on ex-date; no earlier liability
    ledger.step(np.zeros(2))  # cover at Monday open, still owes distribution
    assert ledger.short_dividends == 100
    assert ledger.borrow_paid == pytest.approx(5000 * 0.0365 * 3 / 365)
    assert ledger.cash == pytest.approx(10000 - 100 - 1.5)


def test_only_cash_outside_short_collateral_earns_interest():
    p = panel()
    p.cash_rate[:] = 0.0365
    ledger = SignedLedger(p, 1, 4, fee=0, slip=0, borrow=0)
    ledger.step(np.array([0.5, -0.5]))
    old = ledger.interest
    free = ledger.free_cash(p.close[1])
    cash = ledger.cash
    ledger.step()
    assert ledger.interest - old == pytest.approx(free * 0.0365 * 3 / 365)
    assert ledger.interest - old < cash * 0.0365 * 3 / 365


def test_close_collateral_breach_fills_next_open_not_known_close():
    p = panel()
    p.close[2, 1] = 140.0
    p.opening[3, 1] = 150.0
    p.close[3, 1] = 150.0
    ledger = SignedLedger(p, 1, 4, fee=0, slip=0, borrow=0)
    ledger.step(np.array([0.5, -0.5]))
    ledger.step()
    assert ledger.pending_liquidation and len(ledger.fills) == 2
    ledger.step(np.array([0.5, -0.5]))
    assert ledger.margin_liquidations == 1 and not np.any(ledger.units)
    assert ledger.fills[-1]["reference_price"] == 150
    assert ledger.cash == 7500


def test_gap_insolvency_retains_debt():
    p = panel()
    p.opening[2:, 1] = 300.0
    p.close[2:, 1] = 300.0
    ledger = SignedLedger(p, 1, 4, fee=0, slip=0, borrow=0)
    ledger.step(np.array([0.0, -1.0]))
    ledger.step()
    ledger.step()
    ledger.step()
    assert ledger.bankrupt and ledger.cash == -10000 and not np.any(ledger.units)
    assert ledger.history[-1]["equity"] == -10000


def test_gross_target_limit():
    ledger = SignedLedger(panel(), 1, 4)
    with pytest.raises(ValueError):
        ledger.step(np.array([1.0, -1.0]))


@pytest.fixture
def frames():
    n = 380
    x = np.arange(n)
    ts = pd.date_range("2014-01-01", periods=n, tz="UTC")
    names = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY", "IEF"]
    out = {}
    for k, s in enumerate(names):
        close = 100 * np.exp(0.0002 * x + 0.08 * np.sin(x / (18 + k)) + 0.015 * np.cos(x / 5 + k))
        out[s] = pd.DataFrame(
            {
                "timestamp": ts,
                "bar_end": ts + pd.Timedelta(hours=6),
                "open": close * 1.003,
                "close": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "volume": 1000.0,
                "dividend": 0.0,
                "split": 0.0,
            }
        )
    return out


@pytest.mark.parametrize("rule", RULES)
def test_future_mutations_do_not_change_earlier_trades(frames, rule):
    p = panel_from(frames, False, 365)
    d = features(p, frames)
    _, h, _, _ = run(p, frames, d, rule, 270, 370)
    changed = {s: f.copy() for s, f in frames.items()}
    for k, f in enumerate(changed.values()):
        f.loc[321:, ["open", "close", "high", "low"]] *= 2 + k / 10
    p2 = panel_from(changed, False, 365)
    d2 = features(p2, changed)
    _, h2, _, _ = run(p2, changed, d2, rule, 270, 370)
    pd.testing.assert_frame_equal(h.iloc[:52], h2.iloc[:52])
    np.testing.assert_array_equal(target(rule, 320, p, d), target(rule, 320, p2, d2))


def test_dollar_and_factor_neutrality(frames):
    p = panel_from(frames, False, 365)
    d = features(p, frames)
    for rule in RULES:
        w = target(rule, 300, p, d)
        assert abs(w.sum()) < 1e-10 and np.abs(w).sum() <= 1 + 1e-10
    w = target("LSWeightedResidual", 300, p, d)
    assert abs(w[:9] @ d["beta"][300, :9]) < 1e-10
    w = target("LSMultiCluster", 300, p, d)
    for group in GROUPS:
        assert abs(w[list(group)].sum()) < 1e-10
