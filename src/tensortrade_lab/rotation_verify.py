"""Independent cash/share reconciliation of saved fills against raw snapshots.

Does not call the portfolio ledger, feature builder or simulation functions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import digest
from .experiments import atomic_json


def reconcile(study):
    study = Path(study)
    protocol = json.loads((study / "protocol.json").read_text())
    spec = json.loads((study / "preregistration.json").read_text())["spec"]
    symbols = [*spec["universe"], spec["benchmark"]]
    raw = {}
    for name, source in protocol["data"]["sources"].items():
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("A registered source changed")
        frame = pd.read_parquet(source["path"])
        if name == "cash_yield":
            rates = frame
        else:
            frame["day"] = pd.to_datetime(frame.timestamp).dt.strftime("%Y-%m-%d")
            raw[name] = frame.set_index("day")
    rate_dates = np.array(rates.date)
    reports = []
    files = [
        p
        for folder in ("development", "confirmation", "stress")
        for p in (study / folder).rglob("equity.csv")
    ]
    for path in files:
        history = pd.read_csv(path)
        try:
            fills = pd.read_csv(path.with_name("fills.csv"))
        except pd.errors.EmptyDataError:
            fills = pd.DataFrame(columns=["timestamp"])
        multiplier = 2 if "double_costs" in str(path) else 1
        commission, slip = spec["commission"] * multiplier, spec["slippage"] * multiplier
        grouped = {}
        for fill in fills.to_dict("records"):
            grouped.setdefault(pd.Timestamp(fill["timestamp"]).strftime("%Y-%m-%d"), []).append(fill)
        cash = spec["initial_cash"]
        shares = dict.fromkeys(symbols, 0.0)
        total_fee = total_slip = total_div = total_interest = 0.0
        largest = 0.0
        previous_date = pd.Timestamp(history.timestamp.iloc[0]).normalize()
        for row in history.iloc[1:].itertuples():
            day = pd.Timestamp(row.timestamp).normalize()
            date = day.strftime("%Y-%m-%d")
            rate_index = np.searchsorted(rate_dates, date, side="left") - 1
            if rate_index < 0:
                raise ValueError("Missing independent rate observation")
            discount = float(rates.discount_yield.iloc[rate_index])
            annual_rate = discount * 365 / (360 - 91 * discount)
            interest = cash * annual_rate * (day - previous_date).days / 365
            total_interest += interest
            cash += interest
            dividend = sum(shares[s] * float(raw[s].loc[date, "dividend"]) for s in symbols)
            total_div += dividend
            cash += dividend
            for fill in grouped.get(date, []):
                symbol, amount = fill["symbol"], fill["units"]
                quote = raw[symbol].loc[date, "open" if fill["reason"] == "rebalance" else "close"]
                expected_price = quote * (1 + slip * np.sign(amount))
                fee = abs(amount * expected_price) * commission
                slippage = abs(amount * quote) * slip
                if not np.allclose(
                    [fill["price"], fill["fee"], fill["slippage"]],
                    [expected_price, fee, slippage],
                    rtol=1e-11,
                    atol=1e-8,
                ):
                    raise ArithmeticError("Fill price/cost differs from independent execution assumptions")
                cash -= amount * expected_price + fee
                shares[symbol] += amount
                total_fee += fee
                total_slip += slippage
            equity = cash + sum(shares[s] * float(raw[s].loc[date, "close"]) for s in symbols)
            errors = [
                abs(equity - row.equity),
                abs(cash - row.cash),
                abs(total_fee - row.fees),
                abs(total_slip - row.slippage),
                abs(total_div - row.dividends),
                abs(total_interest - row.interest),
            ]
            errors.extend(abs(shares[s] - getattr(row, "units_" + s)) for s in symbols)
            largest = max(largest, *errors)
            if largest > 1e-6 or cash < -1e-6 or min(shares.values()) < -1e-8:
                raise ArithmeticError(f"Independent cash/share reconciliation failed at {path} on {date}")
            previous_date = day
        decisions_path = path.with_name("decisions.csv")
        if decisions_path.exists():
            decisions = pd.read_csv(decisions_path)
            if not (pd.to_datetime(decisions.decided_at) < pd.to_datetime(decisions.execute_at)).all():
                raise ValueError("A decision uses an uncompleted candle")
        reports.append(
            {
                "path": str(path.relative_to(study)),
                "rows": len(history),
                "fills": len(fills),
                "maximum_absolute_error": largest,
            }
        )
    result = {
        "passed": True,
        "portfolios_checked": len(reports),
        "maximum_absolute_error": max(r["maximum_absolute_error"] for r in reports),
        "scope": "Independent cash, shares, mark-to-market equity, ex-dividends, lagged yield interest, trade quotes, commission/slippage, timestamps. Not independent verification of vendor prices or investment signal quality.",
        "checks": reports,
    }
    atomic_json(study / "reconciliation.json", result)
    return result
