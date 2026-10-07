import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "research/paper151"))
from expanded_study import load_jobs
from reconcile_tradingview import OUT, compare, emulate, native_events, replay


@pytest.fixture(scope="module")
def tv_frames():
    f = next(j[1]["TLT"] for j in load_jobs()[0] if j[0] == "TLT")
    tv = pd.read_json(OUT / "TLT_tradingview_prices.json")
    tv["timestamp"] = pd.to_datetime(tv.time, unit="s", utc=True)
    return f.merge(
        tv[["timestamp", "open", "high", "low", "close", "volume"]],
        on="timestamp",
        suffixes=("_yahoo", ""),
        validate="one_to_one",
    )


@pytest.mark.external_data
@pytest.mark.parametrize("rule", ["MA50_200", "MA20_50_200", "AlphaCombo", "IBS"])
def test_independent_emulator_matches_native_events_and_cash(rule, tv_frames):
    all_reports = json.loads((OUT / "TLT_native_reports.json").read_text())
    tv = next(v["report"] for k, v in all_reports.items() if k.split()[2] == rule)
    expected = native_events(tv["trades"], str(tv_frames.timestamp.iloc[-1].date()))
    fills, h, rejected, start = emulate(rule, tv_frames)
    check = compare(fills, expected)
    assert check["events_match"] and check["quantities_match"] and check["max_fill_price_difference"] == 0
    m, _ = replay(tv_frames, expected, start)
    assert abs(h[-1]["equity"] / 10000 - 1 - m["return"]) < 1e-10
    if rule != "IBS":
        assert abs(m["return"] - tv["performance"]["all"]["netProfit"] / 10000) < 1e-10
        assert (
            abs(m["closed_peak_intrabar_drawdown"] - tv["performance"]["maxStrategyDrawDownPercent"]) < 1e-10
        )
    if rule == "AlphaCombo":
        assert len(rejected) == 4 and sum(x["role"] == "margin" for x in fills) == 2
    if rule == "IBS":
        assert sum(x["role"] == "margin" for x in fills) == 12


def test_split_trades_do_not_duplicate_original_entry_and_open_marks_are_not_fills():
    t = 1704205800000
    trades = [
        {"q": 1, "e": {"tm": t, "p": 100}, "x": {"tm": t, "p": 100, "c": "Margin call"}},
        {"q": 9, "e": {"tm": t, "p": 100}, "x": {"tm": t + 86400000, "p": 101, "c": "Next-open exit"}},
        {"q": 8, "e": {"tm": t + 172800000, "p": 101}, "x": {"tm": t + 259200000, "p": 105, "c": ""}},
    ]
    out = native_events(trades, "2024-01-10")
    assert [(x["role"], x["units"]) for x in out] == [
        ("entry", 10),
        ("margin", -1),
        ("exit", -9),
        ("entry", 8),
    ]
