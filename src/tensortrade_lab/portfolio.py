"""Causal multi-asset features and self-financing, long-only execution.

Prices and dividends must already be split-adjusted. Each action is decided
after a completed close and executes at the following open. No forward fill of
tradable prices is permitted. The same ledger powers rules and TensorTrade.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Panel:
    symbols: list[str]
    timestamp: pd.DatetimeIndex
    bar_end: pd.DatetimeIndex
    opening: np.ndarray
    close: np.ndarray
    dividend: np.ndarray
    features: np.ndarray
    total_index: np.ndarray
    volatility: np.ndarray
    cash_rate: np.ndarray

    def prefix(self, end):
        """Physically truncate a fold before fitting, including label inputs."""
        n = int((self.bar_end < pd.Timestamp(end, tz="UTC")).sum())
        return Panel(
            self.symbols, *[getattr(self, k)[:n] for k in self.__dataclass_fields__ if k != "symbols"]
        )


def make_panel(frames, rates):
    symbols = list(frames)
    indexed = {s: f.set_index("timestamp") for s, f in frames.items()}
    start = max(f.index[0] for f in indexed.values())
    end = min(f.index[-1] for f in indexed.values())
    base = indexed[symbols[0]].loc[start:end].index
    for s, frame in indexed.items():
        if not frame.loc[start:end].index.equals(base):
            raise ValueError(f"Unaligned or missing sessions: {s}")
    opening = np.column_stack([f.loc[base, "open"] for f in indexed.values()])
    close = np.column_stack([f.loc[base, "close"] for f in indexed.values()])
    dividend = np.column_stack([f.loc[base, "dividend"] for f in indexed.values()])
    ends = pd.DatetimeIndex(indexed[symbols[0]].loc[base, "bar_end"])
    for f in indexed.values():
        if not pd.DatetimeIndex(f.loc[base, "bar_end"]).equals(ends):
            raise ValueError("Assets have different completed-session times")
    returns = pd.DataFrame((close[1:] + dividend[1:]) / close[:-1] - 1)
    returns = pd.concat([pd.DataFrame(np.zeros((1, len(symbols)))), returns], ignore_index=True)
    index = (1 + returns).cumprod()
    vol = returns.rolling(63, min_periods=63).std() * np.sqrt(252)
    features = []
    for lookback in (21, 63, 126, 252):
        r = index / index.shift(lookback) - 1
        features.extend([r.to_numpy(), r.rank(axis=1, pct=True).to_numpy()])
    features.extend([vol.to_numpy(), (index / index.rolling(126).max() - 1).to_numpy()])
    rate_frame = rates.copy()
    rate_frame["day"] = pd.to_datetime(rate_frame.date, utc=True)
    rate_frame["published_day"] = rate_frame.day
    days = pd.DataFrame({"day": base.normalize()})
    # Strictly earlier observation; no same-day yield can influence interest.
    joined = pd.merge_asof(days, rate_frame.sort_values("day"), on="day", allow_exact_matches=False)
    if joined.discount_yield.isna().any() or ((joined.day - joined.published_day).dt.days > 7).any():
        raise ValueError("Missing or stale prior-session cash yield")
    discount = joined.discount_yield.to_numpy()
    cash_rate = discount / (1 - discount * 91 / 360) * 365 / 360
    return Panel(
        symbols,
        base,
        ends,
        opening,
        close,
        dividend,
        np.stack(features, axis=-1),
        index.to_numpy(),
        vol.to_numpy(),
        cash_rate,
    )


def rebalance_days(panel, frequency):
    """Use the exchange calendar, including holidays, not a truncated future frame."""
    from .market_data import schedule

    dates = panel.timestamp.tz_convert("America/New_York").tz_localize(None).normalize()
    calendar = schedule(str(dates[0].date()), str((dates[-1] + pd.Timedelta(days=10)).date()))
    sessions = pd.DatetimeIndex(calendar.index).tz_localize(None).normalize()
    key = sessions.to_period("W-FRI" if frequency == "weekly" else "M")
    last = pd.Series(sessions, index=key).groupby(level=0).max()
    return dates.isin(last.to_numpy())


def weights_from_scores(scores, volatility, count=3, weighting="equal"):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Cannot allocate with missing scores")
    winners = np.argsort(-scores, kind="stable")[:count]
    weights = np.zeros(len(scores))
    raw = np.ones(len(winners)) if weighting == "equal" else 1 / np.maximum(volatility[winners], 0.01)
    weights[winners] = raw / raw.sum()
    return weights


def momentum_weights(panel, index, lookback, skip=21, top_k=3, weighting="equal", investable=None):
    if index < lookback:
        raise ValueError("Insufficient momentum warmup")
    investable = investable or len(panel.symbols)
    scores = (
        panel.total_index[index - skip, :investable] / panel.total_index[index - lookback, :investable] - 1
    )
    return np.pad(
        weights_from_scores(scores, panel.volatility[index, :investable], top_k, weighting),
        (0, len(panel.symbols) - investable),
    )


class Ledger:
    def __init__(self, panel, start, end, commission=0.0005, slippage=0.0005, initial=10000.0):
        if start < 1 or end <= start or end >= len(panel.timestamp):
            raise ValueError("Invalid execution interval")
        self.panel, self.start, self.end = panel, start, end
        self.commission, self.slippage, self.initial = commission, slippage, initial
        self.reset()

    def reset(self):
        self.index = self.start - 1
        self.cash = self.initial
        self.shares = np.zeros(len(self.panel.symbols))
        self.fees = self.slip_paid = self.interest = self.dividends = self.turnover = 0.0
        self.reward_value = 0.0
        self.history, self.fills = [], []
        self.record()

    @property
    def equity(self):
        return float(self.cash + self.shares @ self.panel.close[self.index])

    def record(self):
        self.history.append(
            {
                "timestamp": self.panel.bar_end[self.index],
                "equity": self.equity,
                "cash": self.cash,
                "fees": self.fees,
                "slippage": self.slip_paid,
                "interest": self.interest,
                "dividends": self.dividends,
                "turnover": self.turnover,
                **{f"units_{s}": float(u) for s, u in zip(self.panel.symbols, self.shares)},
            }
        )

    def _execute(self, delta, prices, reason):
        trade_price = prices * (1 + self.slippage * np.sign(delta))
        fees = np.abs(delta * trade_price) * self.commission
        slip = np.abs(delta * prices) * self.slippage
        self.cash -= float(delta @ trade_price + fees.sum())
        self.shares += delta
        self.fees += float(fees.sum())
        self.slip_paid += float(slip.sum())
        self.turnover += float(np.abs(delta * prices).sum())
        for j in np.flatnonzero(np.abs(delta) > 1e-10):
            self.fills.append(
                {
                    "timestamp": self.panel.timestamp[self.index]
                    if reason == "rebalance"
                    else self.panel.bar_end[self.index],
                    "symbol": self.panel.symbols[j],
                    "units": float(delta[j]),
                    "price": float(trade_price[j]),
                    "reference_price": float(prices[j]),
                    "fee": float(fees[j]),
                    "slippage": float(slip[j]),
                    "reason": reason,
                }
            )
        if self.cash < -1e-6 or self.shares.min() < -1e-8:
            raise ArithmeticError("Portfolio is not self-financing")
        self.cash = max(0.0, self.cash)

    def rebalance(self, weights, prices):
        weights = np.asarray(weights, dtype=float)
        if (
            weights.shape != self.shares.shape
            or not np.isfinite(weights).all()
            or weights.min() < 0
            or weights.sum() > 1 + 1e-10
        ):
            raise ValueError("Weights must be finite, nonnegative, unlevered and match the universe")
        equity = float(self.cash + self.shares @ prices)
        net = equity
        # Solve post-cost equity so simultaneous sells fund buys, including fees.
        for _ in range(30):
            delta = weights * net / prices - self.shares
            cost = (
                np.abs(delta * prices)
                * (self.slippage + self.commission * (1 + self.slippage * np.sign(delta)))
            ).sum()
            updated = equity - cost
            if abs(updated - net) < 1e-10:
                net = updated
                break
            net = updated
        self._execute(weights * net / prices - self.shares, prices, "rebalance")

    def step(self, weights=None, drip=False, liquidate=True):
        if self.index >= self.end:
            raise ValueError("Episode already finished")
        before = self.equity
        previous = self.index
        self.index += 1
        p, t = self.panel, self.index
        days = (p.timestamp[t].normalize() - p.timestamp[previous].normalize()).days
        earned = self.cash * p.cash_rate[t] * days / 365
        self.cash += earned
        self.interest += earned
        distribution = self.shares * p.dividend[t]
        self.cash += float(distribution.sum())
        self.dividends += float(distribution.sum())
        if weights is not None:
            self.rebalance(weights, p.opening[t])
        if drip and distribution.sum() > 0:
            # Passive benchmarks reinvest each asset's own distribution at its close.
            delta = distribution / (p.close[t] * (1 + self.slippage) * (1 + self.commission))
            self._execute(delta, p.close[t], "dividend_reinvestment")
        if liquidate and t == self.end:
            self._execute(-self.shares, p.close[t], "terminal_liquidation")
        self.reward_value = float(np.log(self.equity / before))
        self.record()
        return self.reward_value


def performance(ledger):
    history = pd.DataFrame(ledger.history)
    eq = history.equity.to_numpy()
    returns = eq[1:] / eq[:-1] - 1
    days = (history.timestamp.iloc[-1] - ledger.panel.timestamp[ledger.start]).total_seconds() / 86400
    vol = returns.std(ddof=1) * np.sqrt(252)
    periods = np.array(
        [
            (ledger.panel.timestamp[t].normalize() - ledger.panel.timestamp[t - 1].normalize()).days
            for t in range(ledger.start, ledger.end + 1)
        ]
    )
    excess = returns - ledger.panel.cash_rate[ledger.start : ledger.end + 1] * periods / 365
    return {
        "total_return": float(eq[-1] / eq[0] - 1),
        "cagr": float((eq[-1] / eq[0]) ** (365.25 / max(1, days)) - 1),
        "max_drawdown": float((1 - eq / np.maximum.accumulate(eq)).max()),
        "volatility": float(vol),
        "sharpe": float(excess.mean() * 252 / vol) if vol > 1e-12 else 0.0,
        "fees": ledger.fees,
        "slippage": ledger.slip_paid,
        "interest": ledger.interest,
        "dividends": ledger.dividends,
        "fills": len(ledger.fills),
        "turnover_initial_capital": ledger.turnover / ledger.initial,
        "start": str(ledger.panel.timestamp[ledger.start]),
        "end": str(history.timestamp.iloc[-1]),
    }
