import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "research/paper151"))
from ensemble_all_clocks import causal_daily_state
from ensemble_intraday import SPECS, fit_weights, paper_daily
from ensemble_intraday import execute as future_execute
from ensemble_study import execute as daily_execute
from ensemble_study import features, fit_ridge, mature_labels, simple_weights
from filing_study import dated_facts, first_quarters, sue_at
from macro_study import allocation, known_at
from options_weekly import combine, entry_available, unresolved_entry
from run_study import panel_from


def frame(n=400):
    t = pd.date_range("2017-01-01", periods=n, tz="UTC")
    c = 100 + np.arange(n) * 0.02 + np.sin(np.arange(n) / 7)
    return pd.DataFrame(
        {
            "timestamp": t,
            "bar_end": t + pd.Timedelta(hours=8),
            "open": c - 0.1,
            "close": c,
            "high": c + 1,
            "low": c - 1,
            "dividend": 0.0,
            "split": 1.0,
            "volume": 1000.0,
            "scale": 1.0,
            "old_open": np.nan,
            "symbol": "ESH7",
            "old_symbol": None,
        }
    )


def test_ensemble_features_do_not_depend_on_future():
    f = frame()
    p = panel_from({"SPY": f}, False)
    t = np.ones((400, 2, 1)) * 0.4
    r = np.column_stack([np.sin(np.arange(400)) * 0.01, np.cos(np.arange(400)) * 0.01])
    short = panel_from({"SPY": f.iloc[:250]}, False)
    np.testing.assert_allclose(features(p, t, r)[:250], features(short, t[:250], r[:250]))


@pytest.mark.parametrize("method", ["EqualFamilies", "InverseVol", "PastSharpe"])
def test_family_weight_prefix_and_normalization(method):
    r = np.random.default_rng(8).normal(0, 0.01, (400, 4))
    a = simple_weights(r, 200, method)
    r[201:] = 100
    np.testing.assert_allclose(a, simple_weights(r, 200, method))
    assert a.sum() == pytest.approx(1)
    assert min(a) >= 0


def test_mature_labels_exclude_current_and_unavailable_future():
    r = np.zeros((30, 1))
    r[0] = 100
    r[1] = 0.1
    r[21] = 0.2
    y = mature_labels(r)
    assert y[0, 0] == pytest.approx(1.1 * 1.2 - 1)
    assert np.isnan(y[9:]).all()


def test_ridge_training_embargo_ignores_later_outcomes():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 5))
    r = rng.normal(0, 0.01, (400, 3))
    s, m, a = fit_ridge(x, r, 300)
    x[300:] = 1e6
    r[300:] = 5
    s2, m2, a2 = fit_ridge(x, r, 300)
    assert a["max_label_index"] < 300
    assert a == a2
    np.testing.assert_allclose(s.mean_, s2.mean_)
    np.testing.assert_allclose(m.coef_, m2.coef_)


def test_daily_target_waits_until_following_open_and_delay():
    f = frame(7)
    p = panel_from({"SPY": f}, False)
    s = np.full((7, 1), np.nan)
    s[2] = 1
    a = daily_execute(p, s, 1, 6, 0, 0)
    b = daily_execute(p, s, 1, 6, 0, 0, delay=2)
    assert pd.Timestamp(a.fills[0]["timestamp"]) == p.timestamp[3]
    assert pd.Timestamp(b.fills[0]["timestamp"]) == p.timestamp[4]
    assert a.fills[0]["price"] == p.opening[3, 0]


def test_completed_day_paper_targets_prefix():
    f = frame()
    full, names = paper_daily(f)
    part, _ = paper_daily(f.iloc[:280].copy())
    np.testing.assert_allclose(full[:280], part)
    changed = f.copy()
    changed.loc[280:, ["close", "high", "open"]] *= 50
    changed["high"] = changed[["open", "high", "close"]].max(axis=1)
    np.testing.assert_allclose(full[:280], paper_daily(changed)[0][:280])


def test_daily_shadow_waits_for_completed_day_and_rejects_staleness():
    d = frame(3)
    intra = d.iloc[:2].copy()
    intra["bar_end"] = d.bar_end.iloc[:2] + pd.Timedelta(hours=1)
    w = causal_daily_state(intra, d, np.array([0.1, 0.2, 0.9]))
    np.testing.assert_allclose(w, [0.1, 0.2])
    future = d.copy()
    future.loc[2, "bar_end"] += pd.Timedelta(days=10)
    np.testing.assert_allclose(causal_daily_state(intra, future, np.array([0.1, 0.2, 999.0])), w)
    intra.loc[1, "bar_end"] += pd.Timedelta(days=10)
    with pytest.raises(ValueError):
        causal_daily_state(intra, d, np.array([0.1, 0.2, 0.9]))


@pytest.mark.parametrize(
    "method", ["EqualFamilies", "InverseVol", "PastSharpe", "WithoutPatternFindr", "WithoutQuant"]
)
def test_intraday_family_weights_prefix(method):
    f = frame()
    r = np.random.default_rng(9).normal(0, 0.01, (400, 3))
    names = ["Paper", "PatternFindr", "Quant"]
    a = fit_weights(f, r, names, method)
    b = fit_weights(f.iloc[:250], r[:250], names, method)
    np.testing.assert_allclose(a[:250], b)
    np.testing.assert_allclose(a.sum(axis=1), 1)


def test_futures_roll_reconciles_both_legs_and_integer_positions():
    f = frame(8)
    f.loc[4:, "symbol"] = "ESM7"
    f.loc[4, "old_symbol"] = "ESH7"
    f.loc[4, "old_open"] = f.open.iloc[4]
    f.loc[4:, ["open", "high", "low", "close"]] += 7
    m, h, fills = future_execute(f, np.full(8, 0.5), 1, 7, "ES")
    mult = SPECS["ES"][0]
    cash = 1e6
    positions = {}
    for x in fills:
        assert x["units"] == int(x["units"])
        cash -= x["units"] * x["modeled_price"] * mult + x["fee"]
        positions[x["symbol"]] = positions.get(x["symbol"], 0) + x["units"]
    assert all(v == 0 for v in positions.values())
    assert cash == pytest.approx(h[-1], abs=1e-6)
    assert {"roll_in", "roll_out"}.issubset({x["reason"] for x in fills})


def vint(values):
    return pd.DataFrame(values, columns=["period_start_date", "value", "available_at"]).assign(
        period_start_date=lambda f: pd.to_datetime(f.period_start_date, utc=True),
        available_at=lambda f: pd.to_datetime(f.available_at, utc=True),
    )


def test_macro_vintages_reject_future_revision():
    a = vint(
        [
            ("2020-01-01", 100, "2020-02-15"),
            ("2021-01-01", 104, "2021-02-15"),
            ("2020-01-01", 90, "2022-02-15"),
        ]
    )
    b = a.copy()
    b.loc[1, "value"] = 102
    at = pd.Timestamp("2021-02-20", tz="UTC")
    w, d = allocation(a, b, at)
    assert w == pytest.approx(0.5)
    assert known_at(a, at).loc[pd.Timestamp("2020-01-01", tz="UTC"), "value"] == 100
    assert all(pd.Timestamp(r["available_at"]) <= at for r in d["vintages_used"])


def test_macro_missing_prior_is_not_filled():
    a = vint([("2021-01-01", 104, "2021-02-15")])
    w, d = allocation(a, a, pd.Timestamp("2021-02-20", tz="UTC"))
    assert w is None
    assert "prior-year" in d["reason"]


def test_macro_stale_release_and_zero_inflation():
    a = vint([("2020-01-01", 100, "2020-02-15"), ("2021-01-01", 100, "2021-02-15")])
    assert allocation(a, a, pd.Timestamp("2021-02-20", tz="UTC"))[0] == 0
    assert allocation(a, a, pd.Timestamp("2021-06-20", tz="UTC"))[0] is None


def test_first_filed_eps_not_replaced_by_later_revision():
    f = pd.DataFrame(
        {
            "start": ["2018-01-01"] * 2,
            "end": pd.to_datetime(["2018-03-31"] * 2, utc=True),
            "available_at": pd.to_datetime(["2018-05-01", "2019-05-01"], utc=True),
            "accn": ["a", "b"],
            "val": [1.0, 999.0],
        }
    )
    q = first_quarters(f)
    assert len(q) == 1
    assert q.val.iloc[0] == 1


def test_sec_late_acceptance_controls_availability():
    facts = {
        "facts": {
            "us-gaap": {
                "T": {
                    "units": {
                        "USD": [
                            {
                                "form": "10-Q",
                                "end": "2020-03-31",
                                "filed": "2020-05-01",
                                "accn": "a",
                                "val": 1,
                            }
                        ]
                    }
                }
            }
        }
    }
    sub = [{"accessionNumber": "a", "acceptanceDateTime": "2020-05-02T12:00:00Z"}]
    f = dated_facts(facts, sub, "T", "USD")
    assert f.available_at.iloc[0] == pd.Timestamp("2020-05-02T12:01:00Z")
    with pytest.raises(ValueError):
        dated_facts(facts, [], "T", "USD")


def test_missing_eps_quarter_blocks_sue():
    q = pd.DataFrame(
        {
            "quarter": pd.period_range("2017Q1", "2019Q4", freq="Q").delete(4),
            "available_at": pd.Timestamp("2020-01-01", tz="UTC"),
            "end": pd.Timestamp("2019-12-31", tz="UTC"),
            "filed": pd.Timestamp("2020-01-01", tz="UTC"),
            "val": 1.0,
        }
    )
    splits = pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC"), "split": []})
    value, reason = sue_at(q, splits, pd.Timestamp("2020-02-01", tz="UTC"))
    assert value is None
    assert reason == "missing consecutive quarters"


def test_option_dollar_chain_does_not_recapitalize_or_compound():
    h = [
        pd.DataFrame({"timestamp": [1, 2], "equity": [1e6, 1.1e6]}),
        pd.DataFrame({"timestamp": [3, 4], "equity": [1e6, 0.9e6]}),
    ]
    out = combine(h)
    np.testing.assert_allclose(out.equity, [1e6, 1.1e6, 1.1e6, 1e6])


def test_option_dollar_chain_preserves_losses_beyond_initial_capital():
    h = [
        pd.DataFrame({"timestamp": [1, 2], "equity": [1e6, 100000.0]}),
        pd.DataFrame({"timestamp": [3, 4], "equity": [1e6, 800000.0]}),
    ]
    assert combine(h).equity.iloc[-1] == -100000.0


def test_option_missing_contract_never_invents_entry():
    ok, why = entry_available(
        pd.DataFrame(), {"mapping": {"F:0:0": 1}}, [("C", 1, 0, 1)], pd.Timestamp("2025-01-01", tz="UTC")
    )
    assert not ok
    assert "definition" in why


def test_option_halt_keeps_known_entry_and_unknown_terminal_equity():
    day = pd.Timestamp("2025-01-06", tz="UTC")
    at = day + pd.Timedelta(hours=14)
    f = pd.DataFrame(
        {
            "ts_recv": [at],
            "instrument_id": [2],
            "bid_px_00": [4.0],
            "ask_px_00": [5.0],
            "bid_sz_00": [3],
            "ask_sz_00": [3],
        }
    )
    r = unresolved_entry(f, {"mapping": {"C:0:0": 2}}, [("C", 0, 0, 1)], day, 0)
    assert r["terminal_equity"] is None
    assert r["unresolved_positions"] == {"2": 1}
    assert r["entry_fills"][0]["price"] == 5
    assert len(r["entry_fills"]) == 1


def test_option_negative_adverse_entry_is_not_fabricated():
    day = pd.Timestamp("2025-01-06", tz="UTC")
    at = day + pd.Timedelta(hours=14)
    f = pd.DataFrame(
        {
            "ts_recv": [at],
            "instrument_id": [2],
            "bid_px_00": [0.1],
            "ask_px_00": [0.2],
            "bid_sz_00": [3],
            "ask_sz_00": [3],
        }
    )
    r = unresolved_entry(f, {"mapping": {"C:0:0": 2}}, [("C", 0, 0, -1)], day, 1)
    assert not r["entry_executable"]
    assert not r["entry_fills"]


def test_weekly_options_stop_after_entered_trade_missing_exit(tmp_path, monkeypatch):
    import json

    import options_weekly as weekly

    days = pd.date_range("2025-01-06", periods=3, freq="W-MON", tz="UTC")
    monkeypatch.setattr(weekly, "OUT", tmp_path)
    monkeypatch.setattr(weekly, "HERE", tmp_path)
    parent = tmp_path / "full_catalogue" / "options_protocol.json"
    parent.parent.mkdir()
    parent.write_text(json.dumps({"synthetic": True, "purpose": "weekly account regression fixture"}))
    monkeypatch.setattr(weekly, "DAYS", days)
    monkeypatch.setattr(weekly.old, "registry", lambda: {"2.4": [("C", 0, 0, 1)]})
    for week, day in enumerate(days):
        folder = tmp_path / "options" / day.strftime("%Y%m%d")
        folder.mkdir(parents=True)
        (folder / "selection.json").write_text(json.dumps({"mapping": {"F:0:0": 1, "C:0:0": 2}}))
        for j, date in enumerate(pd.date_range(day, periods=3)):
            rows = []
            for time in [date + pd.Timedelta(hours=14), date + pd.Timedelta(hours=19, minutes=55)]:
                if week == 1 and j == 2:
                    continue
                for iid, price in [(1, 100), (2, 5 + week)]:
                    rows.append(
                        {
                            "ts_recv": time,
                            "instrument_id": iid,
                            "bid_px_00": price,
                            "ask_px_00": price + 1,
                            "bid_sz_00": 10,
                            "ask_sz_00": 10,
                        }
                    )
            f = pd.DataFrame(
                rows, columns=["ts_recv", "instrument_id", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
            )
            f.ts_recv = pd.to_datetime(f.ts_recv, utc=True)
            path = folder / ("quotes_" + date.strftime("%Y%m%d") + ".parquet")
            f.to_parquet(path, index=False)
            path.with_suffix(".audit.json").write_text(json.dumps({"extract_sha256": weekly.old.sha(path)}))
    weekly.score()
    rows = json.loads((tmp_path / "results.json").read_text())
    events = json.loads((tmp_path / "events.json").read_text())
    for r in rows:
        assert not r["complete"]
        assert r["entered_weeks"] == 1
        assert r["full_year_return"] is None
        assert r["halted"]["week"] == "2025-01-13"
        assert r["halted"]["unresolved"]["entry_executable"]
    assert not any(e["week"] == "2025-01-20" for e in events)
