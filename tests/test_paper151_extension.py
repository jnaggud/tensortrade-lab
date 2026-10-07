"""Regression tests for financially material timing, gaps and audit integrity."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/paper151"))
from futures_extension import ROOTS, prepare, run
from prospective import SPECS, advance, append_event, load_log, new_account, next_execution
from velocity_retest import protective_fill

from tensortrade_lab.market_data import schedule


def test_stop_gap_does_not_fill_at_unavailable_stop():
    p, reason = protective_fill(80, 85, 75, 100, 100, {"stop_loss_pct": 10, "take_profit_pct": 20})
    assert (p, reason) == (80, "stop_gap")


def test_current_high_cannot_create_earlier_trailing_stop():
    cfg = {
        "stop_loss_pct": 20,
        "take_profit_pct": 50,
        "use_trailing_stop": True,
        "trailing_stop_pct": 5,
        "trailing_stop_activation_pct": 1,
    }
    assert protective_fill(100, 120, 95, 100, 100, cfg) == (None, None)
    assert protective_fill(100, 121, 99, 100, 120, cfg) == (100, "stop_gap")


def test_ambiguous_stop_target_bar_is_conservative():
    assert protective_fill(100, 130, 80, 100, 100, {"stop_loss_pct": 10, "take_profit_pct": 20}) == (
        90.0,
        "stop",
    )


def test_log_chain_detects_tampering(tmp_path):
    p = tmp_path / "log.jsonl"
    append_event({"state": {"cash": 10000}}, p)
    append_event({"state": {"cash": 9990}}, p)
    assert len(load_log(p)) == 2
    p.write_text(p.read_text().replace("9990", "9999"))
    with pytest.raises(ValueError, match="integrity"):
        load_log(p)


def test_future_open_is_strictly_after_recording_and_respects_weekend():
    now = pd.Timestamp("2026-10-02T22:15:00Z")
    assert next_execution(now) == pd.Timestamp("2026-10-05T13:30:00Z")
    assert next_execution(now, True) == pd.Timestamp("2026-10-03T00:00:00Z")


def us_frame():
    cal = schedule("2024-01-01", "2026-10-05")
    return pd.DataFrame(
        {
            "timestamp": cal.open.to_numpy(),
            "bar_end": cal.close.to_numpy(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "dividend": 0.0,
            "split": 1.0,
        }
    )


def test_no_retroactive_prospective_fill():
    spec = SPECS["TLT_MA50_200"]
    account = new_account(spec)
    account["pending_orders"] = [
        {"recorded_at": "2026-10-05T15:00:00Z", "execute_at": "2026-10-05T13:30:00Z", "weights": [1.0]}
    ]
    with pytest.raises(ValueError, match="retrospective"):
        advance(
            account,
            spec,
            {"TLT": us_frame()},
            pd.Timestamp("2026-10-05T22:15:00Z"),
            {"frozen_at": "2026-10-01T22:00:00Z"},
        )


def test_precommitted_order_fills_without_historical_signal_backfill():
    spec = SPECS["TLT_MA50_200"]
    account = new_account(spec)
    account["pending_orders"] = [
        {"recorded_at": "2026-10-02T22:15:00Z", "execute_at": "2026-10-05T13:30:00Z", "weights": [1.0]}
    ]
    out, events = advance(
        account,
        spec,
        {"TLT": us_frame()},
        pd.Timestamp("2026-10-05T22:15:00Z"),
        {"frozen_at": "2026-10-01T22:00:00Z"},
    )
    fills = [e for e in events if e["type"] == "simulated_fills"]
    assert len(fills) == 1 and out["units"]["TLT"] > 99
    assert out["equity"] < 10000 and out["cash"] >= 0
    assert out["pending_orders"][0]["execute_at"] == "2026-10-06T13:30:00+00:00"


def small_contract_frame():
    rows = []
    for day in pd.date_range("2024-01-02", periods=4, tz="UTC"):
        for root in ROOTS:
            for month, volume in [("M", 1000), ("U", 100)]:
                rows.append(
                    {
                        "ts_event": day,
                        "root": root,
                        "symbol": root + month + "4",
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0,
                        "volume": volume,
                    }
                )
    return pd.DataFrame(rows)


def test_contract_selection_does_not_see_current_day_volume():
    f = small_contract_frame()
    a = prepare(f)
    changed = f.copy()
    mask = (changed.ts_event == pd.Timestamp("2024-01-03", tz="UTC")) & changed.symbol.str.contains("U4")
    changed.loc[mask, "volume"] = 100000
    b = prepare(changed)
    assert a[2][1] == b[2][1]
    assert a[2][2] != b[2][2]


def test_missing_actual_contract_price_is_not_forward_filled():
    f = small_contract_frame()
    data = prepare(f)
    d = list(data)
    d[1] = dict(data[1])
    day = data[0][2]
    d[1][day] = dict(d[1][day])
    d[1][day].pop("ESM4")
    with pytest.raises(ValueError, match="Missing"):
        run(tuple(d), "PassiveFutures", 1, 3)


def test_roll_gap_does_not_become_profit():
    f = small_contract_frame()
    for col in ["open", "high", "low", "close"]:
        f[col] = np.where(f.symbol.str.contains("M4"), 100.0, 200.0)
    f.loc[(f.ts_event >= pd.Timestamp("2024-01-03", tz="UTC")) & f.symbol.str.contains("U4"), "volume"] = (
        10000
    )
    metrics, _, fills = run(prepare(f), "PassiveFutures", 1, 3, ticks=0, fee=0)
    assert any(x["symbol"].endswith("U4") for x in fills)
    assert metrics["total_return"] == 0.0


def test_bitcoin_can_queue_next_order_before_previous_bar_is_complete():
    spec = SPECS["BTC_Momentum252_Delay2"]
    account = new_account(spec)
    times = pd.date_range("2024-01-01", "2026-10-02", tz="UTC")
    f = pd.DataFrame(
        {
            "timestamp": times,
            "bar_end": times + pd.Timedelta(days=1),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "dividend": 0.0,
            "split": 1.0,
        }
    )
    account["pending_orders"] = [
        {"recorded_at": "2026-10-02T22:15:00Z", "execute_at": "2026-10-03T00:00:00Z", "weights": [1.0]}
    ]
    account["last_target"] = [1.0]
    account["last_signal_end"] = "2026-10-02T00:00:00+00:00"
    out, events = advance(
        account,
        spec,
        {"BTC-USD": f},
        pd.Timestamp("2026-10-03T22:15:00Z"),
        {"frozen_at": "2026-10-01T22:00:00Z"},
    )
    assert len(out["pending_orders"]) == 2
    assert out["pending_orders"][1]["execute_at"] == "2026-10-04T00:00:00+00:00"
    assert out["pending_orders"][1]["weights"] == [0.0]


def test_single_missed_session_is_flagged():
    spec = SPECS["TLT_MA50_200"]
    account = new_account(spec)
    account["last_signal_end"] = "2026-10-01T20:00:00+00:00"
    _, events = advance(
        account,
        spec,
        {"TLT": us_frame()},
        pd.Timestamp("2026-10-05T22:15:00Z"),
        {"frozen_at": "2026-10-01T19:00:00Z"},
    )
    assert next(e for e in events if e["type"] == "missed_signal_sessions")["count"] == 1
