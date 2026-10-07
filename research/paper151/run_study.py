"""Reproduce the fixed paper151 study. Run with the repository .venv Python.

Reuses the existing self-financing ledger without changing any frozen study.
No data downloads, parameter optimization, broker calls, or live orders.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tensortrade_lab.portfolio import Panel, Ledger
from tensortrade_lab.market_data import schedule

OUT = Path(__file__).parent / "results"
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DAILY = ["SMA200", "MA50_200", "Momentum252", "Donchian55_20", "IBS", "Vol12"]


def read_frame(path):
    path = ROOT / path
    f = pd.read_parquet(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == meta["sha256"], f"Changed source: {path}"
    assert f.timestamp.is_monotonic_increasing and not f.timestamp.duplicated().any()
    assert np.isfinite(f[["open", "high", "low", "close"]]).all().all()
    assert (f.low <= f[["open", "close"]].min(axis=1) + 1e-6).all()
    assert (f.high >= f[["open", "close"]].max(axis=1) - 1e-6).all()
    return f, {"path": str(path), "sha256": digest, **meta}


def panel_from(frames, cash_interest=True, periods=252):
    symbols = list(frames)
    timestamp = pd.DatetimeIndex(frames[symbols[0]].timestamp)
    for f in frames.values():
        assert pd.DatetimeIndex(f.timestamp).equals(timestamp), "Never fill missing tradable prices"
    opening = np.column_stack([f.open for f in frames.values()])
    close = np.column_stack([f.close for f in frames.values()])
    dividend = np.column_stack([f.dividend for f in frames.values()])
    returns = np.vstack([np.zeros(len(symbols)), (close[1:] + dividend[1:]) / close[:-1] - 1])
    total_index = np.cumprod(1 + returns, axis=0)
    vol = pd.DataFrame(returns).rolling(63).std().to_numpy() * np.sqrt(periods)
    rates = np.zeros(len(timestamp))
    if cash_interest:
        rate = pd.read_parquet(ROOT / "data/rotation/cash_yield.parquet")
        rate["day"] = pd.to_datetime(rate.date, utc=True)
        rate["observed"] = rate.day
        join = pd.merge_asof(pd.DataFrame({"day": timestamp.normalize()}), rate, on="day", allow_exact_matches=False)
        # Warmup's first bar can predate rates; never used for evaluation.
        rates = (join.discount_yield / (1 - join.discount_yield * 91 / 360) * 365 / 360).fillna(0).to_numpy()
        assert ((join.day - join.observed).dt.days.iloc[253:] <= 7).all()
    return Panel(symbols, timestamp, pd.DatetimeIndex(frames[symbols[0]].bar_end), opening,
                 close, dividend, np.zeros((len(timestamp), len(symbols), 1)), total_index, vol, rates)


def signals(f, p, kind):
    """At index i, uses only OHLC through the completed close i."""
    c = f.close
    if kind.startswith("SMA"):
        return (c > c.rolling(200).mean()).astype(float).to_numpy()
    if kind in ("MA50_200", "MA10_40"):
        fast, slow = (50, 200) if kind == "MA50_200" else (10, 40)
        return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float).to_numpy()
    if kind == "Momentum252":
        x = pd.Series(p.total_index[:, 0])
        return (x > x.shift(252)).astype(float).to_numpy()
    if kind == "Vol12":
        return np.nan_to_num(np.clip(0.12 / np.maximum(p.volatility[:, 0], 1e-12), 0, 1))
    if kind == "IBS":
        return np.divide(c - f.low, f.high - f.low, out=np.full(len(f), 0.5), where=(f.high != f.low))
    if kind.startswith("Donchian"):
        a, b = (55, 20) if kind == "Donchian55_20" else (20, 10)
        entry = c > f.high.rolling(a).max().shift(1)
        leave = c < f.low.rolling(b).min().shift(1)
        return np.column_stack([entry, leave])
    raise ValueError(kind)


def simulation(p, frames, start, end, kind, fee, slip, delay=1, intraday=False):
    ledger = Ledger(p, start, end, commission=fee, slippage=slip)
    f = next(iter(frames.values()))
    sig = None if kind in ("BuyHold", "PassiveSectors") or kind.startswith("Rotation") else signals(f, p, kind)
    target = None
    held = 0
    bounds_peak = ledger.initial
    bound = 0.0
    if intraday:
        cal = schedule(str(p.timestamp[0].date()), str(p.timestamp[-1].date()))
        session_close = {str(r.open.date()): r.close for r in cal.itertuples()}
    for t in range(start, end + 1):
        j = t - delay
        w = None
        if kind in ("BuyHold", "PassiveSectors"):
            if t == start:
                w = np.zeros(len(p.symbols))
                if kind == "PassiveSectors":
                    w[:9] = 1 / 9
                else:
                    w[p.symbols.index("SPY") if len(p.symbols) > 1 else 0] = 1
        elif kind.startswith("Rotation"):
            monthly = t == start or p.timestamp[t].month != p.timestamp[t - 1].month
            if monthly:
                scores = p.total_index[j, :9] / p.total_index[j - 252, :9] - 1
                winners = np.argsort(-scores, kind="stable")[:3]
                w = np.zeros(len(p.symbols))
                if kind == "RotationDual" and p.close[j, 9] <= p.close[j - 199:j + 1, 9].mean():
                    w[10] = 1
                else:
                    for k in winners:
                        if kind != "RotationOwnMA" or p.close[j, k] > p.close[j - 199:j + 1, k].mean():
                            w[k] = 1 / 3
        else:
            in_position = ledger.shares[0] > 1e-10
            held = held + 1 if in_position else 0
            if kind == "IBS":
                value = float(in_position)
                if in_position and (sig[j] > 0.8 or held >= 5):
                    value = 0.0
                elif not in_position and sig[j] < 0.2:
                    value = 1.0
            elif kind.startswith("Donchian"):
                value = float(in_position)
                if in_position and sig[j, 1]:
                    value = 0.0
                elif not in_position and sig[j, 0]:
                    value = 1.0
            else:
                value = float(sig[j])
            if intraday:
                close_time = session_close[str(p.timestamp[t].date())]
                if p.timestamp[t] >= close_time - pd.Timedelta(minutes=15) or p.timestamp[t].date() != p.timestamp[j].date():
                    value = 0.0
            if value != target or kind == "Vol12":
                w = np.array([value])
                target = value

        # Conservative OHLC bound, calculated on the actual post-open holdings.
        ledger.step(w, drip=kind in ("BuyHold", "PassiveSectors"), liquidate=False)
        high = np.array([fr.high.iloc[t] for fr in frames.values()])
        low = np.array([fr.low.iloc[t] for fr in frames.values()])
        high_nav = float(ledger.cash + ledger.shares @ high)
        low_nav = float(ledger.cash + ledger.shares @ low)
        # DRIP at close occurs after the low/high; for passive DRIP days reconstruct pre-DRIP holdings.
        drips = [fill for fill in ledger.fills[-len(p.symbols):] if fill["reason"] == "dividend_reinvestment" and fill["timestamp"] == p.bar_end[t]]
        if drips:
            units = ledger.shares.copy()
            cash = ledger.cash
            for fill in drips:
                units[p.symbols.index(fill["symbol"])] -= fill["units"]
                cash += fill["units"] * fill["price"] + fill["fee"]
            high_nav = float(cash + units @ high)
            low_nav = float(cash + units @ low)
        bounds_peak = max(bounds_peak, high_nav, ledger.equity)
        bound = max(bound, 1 - low_nav / bounds_peak, 1 - ledger.equity / bounds_peak)
        if t == end:
            ledger._execute(-ledger.shares, p.close[t], "terminal_liquidation")
            ledger.history.pop()
            ledger.record()
            bound = max(bound, 1 - ledger.equity / bounds_peak)
    history = pd.DataFrame(ledger.history)
    eq = history.equity.to_numpy()
    days = (p.bar_end[end] - p.timestamp[start]).total_seconds() / 86400
    metrics = {
        "return": eq[-1] / eq[0] - 1,
        "cagr": (eq[-1] / eq[0]) ** (365.25 / days) - 1 if days >= 365 else None,
        "max_drawdown": (1 - eq / np.maximum.accumulate(eq)).max(),
        "ohlc_drawdown_bound": bound, "fills": len(ledger.fills),
        "fees_dollars": ledger.fees, "slippage_dollars": ledger.slip_paid,
        "cash_interest_dollars": ledger.interest, "dividends_dollars": ledger.dividends,
    }
    return metrics, history, pd.DataFrame(ledger.fills)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "ledgers").mkdir(exist_ok=True)
    manifest, rows, curves = {}, [], {}
    jobs = []
    for symbol, path, first in [("SPY", "data/rotation/SPY.parquet", "2005-01-01"),
                                ("QQQ", "data/yahoo/QQQ_1d.parquet", "2017-01-01"),
                                ("BTC", "data/local/BTCUSD_1d.parquet", "2017-01-01")]:
        f, meta = read_frame(path)
        manifest[symbol + "_daily"] = meta
        f = f[f.timestamp.dt.date <= pd.Timestamp("2026-09-23").date()].reset_index(drop=True)
        jobs.append((symbol + "_daily", {symbol: f}, DAILY, first, symbol == "BTC", False))
    frames = {}
    for symbol in SECTORS + ["SPY", "IEF"]:
        f, meta = read_frame(f"data/rotation/{symbol}.parquet")
        manifest["rotation_" + symbol] = meta
        # A documented missing session starts 2026-09-22; stop the panel before it.
        frames[symbol] = f[f.timestamp.dt.date <= pd.Timestamp("2026-09-21").date()].reset_index(drop=True)
    jobs.append(("Sectors_daily", frames, ["Rotation", "RotationOwnMA", "RotationDual", "PassiveSectors"], "2005-01-01", False, False))
    for symbol in ["SPY", "QQQ", "AAPL"]:
        f, meta = read_frame(f"data/yahoo/{symbol}_15m.parquet")
        manifest[symbol + "_15m"] = meta
        f = f[f.timestamp.dt.date < pd.Timestamp("2026-09-24").date()].reset_index(drop=True)
        jobs.append((symbol + "_15m", {symbol: f}, ["MA10_40", "Donchian20_10"], str(f.timestamp.iloc[253].date()), False, True))
    for market, frames, rules, first, crypto, intraday in jobs:
        p = panel_from(frames, cash_interest=not (crypto or intraday), periods=365 if crypto else 252)
        windows = [("full", first, "2027-01-01")]
        if not intraday:
            for label, a, b in [("2005_2012", "2005-01-01", "2013-01-01"), ("2013_2019", "2013-01-01", "2020-01-01"),
                                ("2020_2022", "2020-01-01", "2023-01-01"), ("2023_latest", "2023-01-01", "2027-01-01")]:
                a = max(a, first)
                if a < b:
                    windows.append((label if a == label[:4] + "-01-01" else a[:4] + "_" + label.split("_")[1], a, b))
        for label, a, b in windows:
            indices = np.flatnonzero((p.timestamp >= pd.Timestamp(a, tz="UTC")) & (p.timestamp < pd.Timestamp(b, tz="UTC")))
            if len(indices) < 20:
                continue
            start, end = max(253, int(indices[0])), int(indices[-1])
            assert start >= 253
            cases = [("base", 1, 1, False)]
            if label == "full":
                cases += [("double_cost", 2, 1, False)]
                if not intraday:
                    cases += [("extra_bar_delay", 1, 2, False)]
                    if not crypto:
                        cases += [("zero_cash_yield", 1, 1, True)]
            for case, cost_mult, delay, zero in cases:
                old_rates = p.cash_rate.copy()
                if zero:
                    p.cash_rate = np.zeros_like(p.cash_rate)
                for rule in ["BuyHold"] + rules:
                    metrics, hist, fills = simulation(p, frames, start, end, rule,
                        (0.001 if crypto else 0.0005) * cost_mult, 0.0005 * cost_mult, delay, intraday)
                    rows.append({"market": market, "window": label, "case": case, "strategy": rule,
                                 "start": str(p.timestamp[start]), "end": str(p.bar_end[end]), **metrics})
                    if label == "full" and case == "base":
                        key = market + "__" + rule
                        hist.to_parquet(OUT / "ledgers" / (key + "_equity.parquet"), index=False)
                        fills.to_json(OUT / "ledgers" / (key + "_fills.json"), orient="records", date_format="iso", indent=2)
                        curves[key] = hist
                p.cash_rate = old_rates
        print("Completed", market, flush=True)
    result = pd.DataFrame(rows)
    bench = result[result.strategy == "BuyHold"][["market", "window", "case", "return", "max_drawdown", "ohlc_drawdown_bound"]]
    result = result.merge(bench, on=["market", "window", "case"], suffixes=("", "_bh"))
    result["excess_return_pp"] = 100 * (result["return"] - result.return_bh)
    result["pass_return_drawdown"] = (result["return"] > result.return_bh) & (result.max_drawdown <= result.max_drawdown_bh)
    result["pass_including_ohlc_bound"] = result.pass_return_drawdown & (result.ohlc_drawdown_bound <= result.ohlc_drawdown_bound_bh)
    result.to_json(OUT / "metrics.json", orient="records", indent=2)
    source_manifest = {"sources": manifest, "protocol_sha256": hashlib.sha256((Path(__file__).parent / "PROTOCOL.md").read_bytes()).hexdigest(),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "disclosure": "Retrospective. Previously seen history, not a pristine holdout. BTC vendor/venue unverified."}
    (OUT / "manifest.json").write_text(json.dumps(source_manifest, indent=2))
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, subplot_titles=["SPY daily", "QQQ daily", "BTC daily"])
    for row, market in enumerate(["SPY_daily", "QQQ_daily", "BTC_daily"], 1):
        for key, hist in curves.items():
            if key.startswith(market + "__"):
                fig.add_trace(go.Scatter(x=hist.timestamp, y=hist.equity / 10000, name=key.split("__")[1],
                    legendgroup=key.split("__")[1], showlegend=row == 1), row=row, col=1)
        fig.update_yaxes(type="log", title_text="Growth of $1 (log)", row=row, col=1)
    fig.update_layout(height=1000, title="Fixed rules from 151 Trading Strategies — retrospective, after costs", template="plotly_white")
    fig.write_html(OUT / "equity_curves.html", include_plotlyjs=True)
    cols = ["market", "strategy", "return", "cagr", "max_drawdown", "return_bh", "max_drawdown_bh", "pass_return_drawdown"]
    print(result[(result.window == "full") & (result.case == "base")][cols].to_string(index=False))


if __name__ == "__main__":
    main()
