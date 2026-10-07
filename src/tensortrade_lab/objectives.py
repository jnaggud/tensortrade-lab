"""Benchmark-relative selection with an explicit feasibility constraint."""

import numpy as np


def selection_score(agent, benchmark, config):
    if config.objective == "return_drawdown":
        return agent["total_return"] - agent["max_drawdown"]
    excess = agent["total_return"] - benchmark["total_return"]
    if agent["max_drawdown"] > config.selection_drawdown_limit:
        return -100.0 - agent["max_drawdown"]
    return float(np.clip(excess, -10, 10))


def aggregate(rows, limit, positive_fraction):
    excess = np.array([r["excess_return"] for r in rows])
    feasible = all(r["max_drawdown"] <= limit for r in rows)
    score = float(excess.mean() - 0.5 * excess.std())
    positive = float(np.mean(excess > 0))
    return {
        "score": score,
        "mean_excess_return": float(excess.mean()),
        "std_excess_return": float(excess.std()),
        "worst_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_fraction": positive,
        "risk_feasible": feasible,
        "eligible": bool(feasible and score > 0 and positive >= positive_fraction),
    }


def learning_gain(log, budget):
    """Promotion sees inner validation only, never outer scores or final returns."""
    earlier = [r["score"] for r in log if r["timesteps"] <= budget / 2]
    later = [r["score"] for r in log if budget / 2 < r["timesteps"] <= budget + 4096]
    return float(max(later) - max(earlier)) if earlier and later else 0.0
