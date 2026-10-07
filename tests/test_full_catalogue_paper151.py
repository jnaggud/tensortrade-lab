import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "research/paper151"))
from full_catalogue_activity import oi_at
from full_catalogue_options import payoff, registry, run_one
from full_catalogue_prices import hp_signal, hp_weights
from full_catalogue_scenarios import duration_wings, tranche_loss, variance_swap
from full_catalogue_volatility import black76, iv_delta


@pytest.mark.parametrize(
    "long,short",
    [
        ("2.6", "2.8"),
        ("2.7", "2.9"),
        ("2.10", "2.11"),
        ("2.12", "2.13"),
        ("2.14", "2.16"),
        ("2.15", "2.17"),
        ("2.22", "2.25"),
        ("2.23", "2.26"),
        ("2.24", "2.27"),
        ("2.28", "2.30"),
        ("2.29", "2.31"),
        ("2.40", "2.42"),
        ("2.41", "2.43"),
        ("2.44", "2.45"),
        ("2.46", "2.48"),
        ("2.47", "2.49"),
        ("2.50", "2.51"),
        ("2.54", "2.55"),
        ("2.56", "2.57"),
    ],
)
def test_opposite_payoff_portfolios_cancel(long, short):
    s = np.r_[np.linspace(0, 220, 441), 1e7]
    assert np.allclose(payoff(registry()[long], s) + payoff(registry()[short], s), 0)


def test_registry_exact_coverage_and_calendar_expiries():
    expected = {f"2.{k}" for k in range(2, 58)} | {"2.40.1", "2.41.1"}
    r = registry()
    assert set(r) == expected
    assert {s for s, legs in r.items() if any(x[2] for x in legs)} == {"2.18", "2.19", "2.20", "2.21"}
    for s in ["2.18", "2.19", "2.20", "2.21"]:
        assert sum(x[3] for x in r[s] if x[2] == 1) == 1
        assert sum(x[3] for x in r[s] if x[2] == 0) == -1
        with pytest.raises(ValueError):
            payoff(r[s], [100.0])


def test_independent_option_identities_and_paper_iron_naming():
    s = np.linspace(0, 250, 501)
    r = registry()
    assert np.allclose(payoff(r["2.10"], s), s - 100)
    assert np.allclose(payoff(r["2.22"], s), abs(s - 100))
    assert np.allclose(payoff(r["2.28"], s), abs(s - 100))
    assert np.allclose(payoff(r["2.29"], s), abs(s - 100))
    assert np.allclose(payoff(r["2.52"], s), 20)
    assert np.allclose(payoff(r["2.40"], s), np.maximum(10 - abs(s - 100), 0))
    assert np.allclose(payoff(r["2.41"], s), payoff(r["2.40"], s))
    assert np.allclose(payoff(r["2.44"], s), -np.minimum(abs(s - 100), 10))
    assert np.allclose(payoff(r["2.46"], s), np.clip(15 - abs(s - 100), 0, 10))
    assert np.allclose(payoff(r["2.47"], s), payoff(r["2.46"], s))
    assert np.allclose(payoff(r["2.50"], s), -np.clip(abs(s - 100) - 5, 0, 10))
    assert np.allclose(payoff(r["2.53"], s), np.clip(s - 100, -10, 10))


def quote_fixture():
    day = pd.Timestamp("2025-07-07", tz="UTC")
    rows = []
    for stamp, spot, premium in [
        (day + pd.Timedelta(hours=14), 100, 5),
        (day + pd.Timedelta(days=2, hours=19, minutes=55), 102, 6),
    ]:
        for iid, price in [(1, spot), (2, premium)]:
            rows.append(
                {
                    "ts_recv": stamp,
                    "instrument_id": iid,
                    "bid_px_00": price - 0.1,
                    "ask_px_00": price + 0.1,
                    "bid_sz_00": 2,
                    "ask_sz_00": 2,
                }
            )
    return day, pd.DataFrame(rows), {"mapping": {"F:0:0": 1, "C:0:0": 2}}


def test_option_quote_cashflow_fees_size_and_no_future_fill():
    day, f, selection = quote_fixture()
    legs = [("C", 0, 0, -2)]
    m, h, fills = run_one(f, selection, legs, day, 0)
    assert np.isclose(m["return"], -130 / 1e6)  # -2*(6.1-4.9)*50 minus four $2.50 sides.
    assert m["missing_marks"] == 214 and m["marks"] == 2
    stressed, _, _ = run_one(f, selection, legs, day, 1)
    assert stressed["return"] < m["return"]
    f.loc[f.instrument_id == 2, "bid_sz_00"] = 1
    with pytest.raises(ValueError, match="entry or exit"):
        run_one(f, selection, legs, day, 0)
    day, f, selection = quote_fixture()
    f.loc[0, "ts_recv"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="entry or exit"):
        run_one(f, selection, legs, day, 0)


def test_causal_hp_agrees_with_full_past_window_solve_and_ignores_future():
    rng = np.random.default_rng(321)
    c = 100 + np.cumsum(rng.normal(size=500))
    d = np.diff(np.eye(252), n=2, axis=0)
    trend = np.linalg.solve(np.eye(252) + 1600 * d.T @ d, c[:252])
    assert np.isclose(c[:252] @ hp_weights(), trend[-20:].mean() - trend[-50:].mean(), atol=1e-8)
    before = hp_signal(c)
    c[400:] += 100000
    assert np.array_equal(before[:400], hp_signal(c)[:400])


def test_hedge_checks_do_not_assume_all_risks_removed():
    durations = np.array([2.0, 5.0, 10.0])
    w = duration_wings(durations)
    assert np.isclose(w.sum(), 0) and np.isclose(w @ durations, 0)
    assert not np.isclose(w @ (durations**2), 0)
    assert np.allclose(tranche_loss([0, 0.02, 0.04, 0.6], 0.03, 0.1), [0, 0, 0.01, 0.07])
    assert variance_swap([100, 110, 121], 0) > 0  # Do not subtract the nonzero mean log return.


def test_oi_availability_corrections_deletes_and_staleness():
    t = pd.Timestamp("2018-01-10", tz="UTC")
    table = pd.DataFrame(
        [
            {
                "symbol": "ESH8",
                "ts_ref": t - pd.Timedelta(days=2),
                "ts_recv": t - pd.Timedelta(days=1),
                "quantity": 100,
                "update_action": 1,
            },
            {
                "symbol": "ESH8",
                "ts_ref": t - pd.Timedelta(days=1),
                "ts_recv": t + pd.Timedelta(hours=1),
                "quantity": 200,
                "update_action": 1,
            },
            {
                "symbol": "ESH8",
                "ts_ref": t - pd.Timedelta(days=3),
                "ts_recv": t + pd.Timedelta(hours=2),
                "quantity": 999,
                "update_action": 1,
            },
        ]
    )
    assert oi_at(table, "ESH8", t) == 100
    assert oi_at(table, "ESH8", t + pd.Timedelta(hours=3)) == 200
    assert np.isnan(oi_at(table, "ESH8", t + pd.Timedelta(days=20)))
    deleted = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {
                        "symbol": "ESH8",
                        "ts_ref": t - pd.Timedelta(days=1),
                        "ts_recv": t + pd.Timedelta(hours=4),
                        "quantity": 0,
                        "update_action": 2,
                    }
                ]
            ),
        ]
    )
    assert np.isnan(oi_at(deleted, "ESH8", t + pd.Timedelta(hours=5)))


@pytest.mark.parametrize("right", ["C", "P"])
def test_black76_iv_inversion_and_delta_by_finite_difference(right):
    value, delta = black76(6000, 6100, 0.1, 0.25, right)
    recovered, recovered_delta = iv_delta(6000, 6100, 0.1, value, right)
    finite = (
        black76(6000.001, 6100, 0.1, 0.25, right)[0] - black76(5999.999, 6100, 0.1, 0.25, right)[0]
    ) / 0.002
    assert np.isclose(recovered, 0.25, atol=1e-8)
    assert np.isclose(delta, finite, atol=1e-7) and np.isclose(delta, recovered_delta)
    with pytest.raises(ValueError):
        iv_delta(6000, 6100, 0, value, right)
