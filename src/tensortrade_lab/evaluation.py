from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .environment import ResearchEnv
from .features import Scaler


@dataclass(frozen=True)
class Partitions:
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


def chronological_split(length: int, config: Config) -> Partitions:
    train_end = int(length * config.split.train_fraction)
    validation_end = int(length * (config.split.train_fraction + config.split.validation_fraction))
    gap = config.split.purge_bars
    result = Partitions((0, train_end), (train_end + gap, validation_end), (validation_end + gap, length))
    for name in ("train", "validation", "test"):
        start, end = getattr(result, name)
        if end - start < config.window + 10:
            raise ValueError(f"{name} partition too short after warmup and purge; provide more candles")
    return result


def metrics(history: pd.DataFrame, fills: pd.DataFrame, periods_per_year: int) -> dict:
    equity = history.equity.to_numpy(float)
    returns = np.divide(equity[1:], equity[:-1], out=np.ones_like(equity[1:]), where=equity[:-1] > 0) - 1
    total = equity[-1] / equity[0] - 1
    volatility = returns.std(ddof=1) if len(returns) > 1 else 0.0
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    drawdown = 1 - equity / np.maximum.accumulate(equity)
    # No annualized return extrapolation from a short sample. Sharpe assumes configured cadence.
    return {
        "initial_equity": float(equity[0]),
        "final_equity": float(equity[-1]),
        "net_profit": float(equity[-1] - equity[0]),
        "total_return": float(total),
        "max_drawdown": float(drawdown.max()),
        "sharpe": float(returns.mean() / volatility * np.sqrt(periods_per_year))
        if volatility > 1e-12
        else 0.0,
        "sortino": float(returns.mean() / downside * np.sqrt(periods_per_year)) if downside > 1e-12 else 0.0,
        "annualized_volatility": float(volatility * np.sqrt(periods_per_year)),
        "fills": len(fills),
        "fees": float(history.fees.iloc[-1]),
        "borrow_cost": float(history.borrow_cost.iloc[-1]) if "borrow_cost" in history else 0.0,
        "dividends": float(history.dividends.iloc[-1]) if "dividends" in history else 0.0,
        "slippage": float(history.slippage.iloc[-1]),
        "turnover": float(fills.notional.sum() / equity[0]) if len(fills) else 0.0,
        "mean_exposure": float(history.exposure.iloc[:-1].mean()),
        "mean_gross_exposure": float(history.exposure.iloc[:-1].abs().mean()),
        "bars": len(returns),
        "halted": bool(history.halted.any()),
        "start": str(history.timestamp.iloc[0]),
        "end": str(history.timestamp.iloc[-1]),
    }


FILL_COLUMNS = [
    "timestamp",
    "side",
    "units",
    "price",
    "reference_price",
    "fee",
    "slippage",
    "notional",
    "reason",
    "cash_after",
    "units_after",
]


def rollout(bars: pd.DataFrame, scaler: Scaler, config: Config, policy="cash", start_index=None):
    benchmark = (isinstance(policy, str) and policy == "buy_hold") or getattr(policy, "bypass_risk", False)
    env = ResearchEnv(bars, scaler.transform(bars), config, start_index, benchmark)
    obs, _ = env.reset(seed=config.training.seed)
    if hasattr(policy, "prepare"):
        policy.prepare(bars)
    done = False
    first = True
    while not done:
        if isinstance(policy, str):
            if policy == "cash":
                action = 1
            elif policy == "buy_hold":
                action = 5 if first else 0
            elif policy == "momentum":
                action = 5 if bars.return_24.iloc[env.state.index] > 0 else 1
            else:
                raise ValueError(f"Unknown baseline {policy}")
        elif hasattr(policy, "action"):
            action = policy.action(bars, env.state.index)
        else:
            action = int(policy.predict(obs, deterministic=True)[0])
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        first = False
    history = pd.DataFrame(env.state.history)
    fills = pd.DataFrame(env.state.fills, columns=FILL_COLUMNS)
    result = metrics(history, fills, config.market.periods_per_year)
    env.close()
    return result, history, fills


def evaluate_suite(bars, scaler, config, model, output: Path, start_index=None):
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, policy in [
        ("agent", model),
        ("buy_hold", "buy_hold"),
        ("momentum", "momentum"),
        ("cash", "cash"),
    ]:
        result, history, fills = rollout(bars, scaler, config, policy, start_index)
        history.to_csv(output / f"{name}_equity.csv", index=False)
        fills.to_csv(output / f"{name}_fills.csv", index=False)
        results[name] = result
    return results
