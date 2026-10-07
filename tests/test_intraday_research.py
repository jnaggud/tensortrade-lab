"""Material accounting, order timing and signal-data causality checks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/paper151"))
from futures_intraday_retest import build_root, run
from intraday_common import spot_run


def frame():
    ts = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "bar_end": ts + pd.Timedelta(minutes=15),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 100.0,
            "symbol": "ESM5",
            "root": "ES",
            "old_symbol": None,
            "old_open": np.nan,
            "scale": 1.0,
        }
    )


def signals(f):
    return pd.DataFrame({"buy": True, "sell": False, "osc": -1.0, "accel": 0.0, "jerk": 0.0}, index=f.index)


def test_spot_costs_balance_without_price_changes():
    f = frame()
    m, eq, fills = spot_run(f, None, {}, 1, 7, fee=0.001, slip=0.0005, kind="benchmark")
    assert eq[-1] == pytest.approx(10000 - m["fees"] - m["slippage"])
    assert len(fills) == 2 and m["entries"] == 1


def test_signal_is_not_executed_before_next_open():
    f = frame()
    s = signals(f)
    s["buy"] = False
    s.loc[2, "buy"] = True
    m, eq, fill = spot_run(f, s, {"stop_loss_pct": 50, "take_profit_pct": 50}, 1, 7, fee=0, slip=0)
    assert fill[0]["timestamp"] == str(f.timestamp.iloc[3])
    assert m["total_return"] == 0


def test_futures_roll_gap_is_not_profit_and_both_legs_cost():
    f = frame()
    f.loc[4:, ["open", "high", "low", "close"]] = 200.0
    f.loc[4:, "symbol"] = "ESU5"
    f.loc[4:, "scale"] = 0.5
    f.loc[4, "old_open"] = 100.0
    f.loc[4, "old_symbol"] = "ESM5"
    m, eq, fills = run(f, None, {}, 1, 7, "ES", kind="benchmark", ticks=0)
    assert m["rolls"] == 1 and {x["reason"] for x in fills} >= {"roll_in", "roll_out"}
    assert eq[-1] == pytest.approx(1e6 - m["fees"])


def test_current_bar_high_cannot_set_earlier_trailing_stop():
    f = frame()
    f.loc[2, ["high", "low", "close"]] = [130.0, 99.0, 120.0]
    d = {
        "stop_loss_pct": 30,
        "take_profit_pct": 90,
        "use_trailing_stop": True,
        "trailing_stop_pct": 5,
        "trailing_stop_activation_pct": 1,
    }
    _, _, fill = run(f, signals(f), d, 1, 7, "ES", ticks=0)
    stops = [x for x in fill if x["reason"].startswith("stop")]
    assert not stops or stops[0]["timestamp"] != str(f.timestamp.iloc[2])


def test_linked_signal_history_never_revised_by_future_contract():
    f = frame()
    rows = []
    for _, r in f.iterrows():
        for symbol, price in [("ESM5", 100.0), ("ESU5", 200.0)]:
            rows.append(
                {
                    "timestamp": r.timestamp,
                    "bar_end": r.bar_end,
                    "symbol": symbol,
                    "root": "ES",
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1.0,
                    "first_print_minute": r.timestamp,
                    "minute_rows": 15,
                }
            )
    # One-day fixture: changing only a future selected contract must not alter earlier linked OHLC.
    data = pd.DataFrame(rows)
    selection = {"20250101": {"front": {"ES": "ESM5"}}}
    a = build_root("ES", data, selection)
    b = build_root("ES", data[data.timestamp <= f.timestamp.iloc[4]], selection)
    assert np.allclose(a.close.iloc[:5] * a.scale.iloc[:5], b.close * b.scale)


def test_unprinted_selected_bucket_does_not_invent_a_trade_bar():
    f = frame()
    rows = []
    for i, r in f.iterrows():
        for symbol, price in [("ESM5", 100.0), ("ESU5", 200.0)]:
            if i == 3 and symbol == "ESM5":
                continue
            rows.append(
                {
                    "timestamp": r.timestamp,
                    "bar_end": r.bar_end,
                    "symbol": symbol,
                    "root": "ES",
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1.0,
                    "first_print_minute": r.timestamp,
                    "minute_rows": 15,
                }
            )
    z = build_root("ES", pd.DataFrame(rows), {"20250101": {"front": {"ES": "ESM5"}}})
    assert len(z) == 7 and f.timestamp.iloc[3] not in set(z.timestamp)
    assert z.attrs["audit"]["unprinted_selected_buckets"] == 1


from options_pilot import synchronized, valid_quotes


def test_option_quote_clock_is_not_last_trade_clock():
    t = pd.Timestamp("2026-06-01T14:00:00Z")
    f = pd.DataFrame(
        [
            {
                "instrument_id": 1,
                "ts_recv": t,
                "ts_event": t - pd.Timedelta(days=1),
                "bid_px_00": 10.0,
                "ask_px_00": 11.0,
                "bid_sz_00": 1,
                "ask_sz_00": 1,
            }
        ]
    )
    assert synchronized(f, [1], t)[1].bid_px_00 == 10
    with pytest.raises(ValueError, match="Missing fresh"):
        synchronized(f, [1], t + pd.Timedelta(minutes=2))


def test_options_require_size_and_non_crossed_sides():
    f = pd.DataFrame(
        {
            "bid_px_00": [11.0, 10.0, 10.0],
            "ask_px_00": [10.0, 11.0, 11.0],
            "bid_sz_00": [1, 0, 1],
            "ask_sz_00": [1, 1, 1],
        }
    )
    assert valid_quotes(f).tolist() == [False, False, True]


def test_futures_fill_ledger_independently_reconciles_roll_and_reduction():
    f = frame()
    for t in range(len(f)):
        price = 100 + t * 2
        f.loc[t, ["open", "high", "low", "close"]] = [price, price + 1, price - 1, price + 0.5]
    f.loc[4:, ["open", "high", "low", "close"]] += 100
    f.loc[4:, "symbol"] = "ESU5"
    f.loc[4, "old_symbol"] = "ESM5"
    f.loc[4, "old_open"] = 108.0
    f.loc[4:, "scale"] = 108 / 208
    m, eq, fills = run(f, None, {}, 1, 7, "ES", kind="benchmark", ticks=3)
    # A separate, signed fill cash ledger must equal final variation-margin equity.
    independently_replayed = 1e6 - sum(z["units"] * z["modeled_price"] * 50 + z["fee"] for z in fills)
    assert eq[-1] == pytest.approx(independently_replayed)
    assert m["gross_cap_reductions"] > 0
    for symbol in {"ESM5", "ESU5"}:
        assert sum(z["units"] for z in fills if z["symbol"] == symbol) == 0


def test_futures_short_profit_and_costs_reconcile_independently():
    f = frame()
    f.loc[3:, ["open", "high", "low", "close"]] = 90.0
    s = pd.DataFrame(
        {
            k: False
            for k in [
                "core_long",
                "cap_long",
                "participation_long",
                "trend_carry_long",
                "core_short",
                "core_long_close_exit",
                "core_short_close_exit",
            ]
        },
        index=f.index,
    )
    s.loc[0, "core_short"] = True
    s["atr"] = 1.0
    _, eq, fills = run(
        f, s, {"allow_short": True, "stop_atr": 50.0, "trail_atr": 50.0}, 1, 7, "ES", kind="quant", ticks=1
    )
    assert fills[0]["units"] < 0
    assert eq[-1] > 1e6
    assert eq[-1] == pytest.approx(1e6 - sum(z["units"] * z["modeled_price"] * 50 + z["fee"] for z in fills))


def test_option_pilot_realized_cashflows_and_preexpiry_limit():
    from options_pilot import run as option_run

    start = pd.Timestamp("2026-06-01T14:00:00Z")
    end = pd.Timestamp("2026-06-05T19:55:00Z")
    rows = []
    for t, ub, ua, ob, oa in [(start, 100.0, 101.0, 10.0, 11.0), (end, 110.0, 111.0, 4.0, 5.0)]:
        for iid, bid, ask in [(1, ub, ua), (2, ob, oa)]:
            rows.append(
                {
                    "ts_recv": t,
                    "instrument_id": iid,
                    "bid_px_00": bid,
                    "ask_px_00": ask,
                    "bid_sz_00": 1,
                    "ask_sz_00": 1,
                }
            )
    f = pd.DataFrame(rows)
    c = pd.DataFrame(
        [
            {"instrument_id": 1, "instrument_class": "F", "raw_symbol": "ESM6"},
            {
                "instrument_id": 2,
                "instrument_class": "C",
                "raw_symbol": "EW2M6 C105",
                "expiration": pd.Timestamp("2026-06-12T20:00:00Z"),
                "strike_price": 105.0,
            },
        ]
    )
    m, h, _ = option_run(f, c, "covered_call")
    assert h.equity.iloc[-1] == 1e6 + (110 - 101 + 10 - 5) * 50 - 10
    assert not m["qualified"]
    c.loc[c.instrument_class == "C", "expiration"] = start
    with pytest.raises(ValueError, match="Expiry"):
        option_run(f, c, "covered_call")
