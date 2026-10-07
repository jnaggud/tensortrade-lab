"""Conditional PPO comparison, entered only after the simple-strategy gate."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .compute import configure_threads, parallel_jobs
from .config import Compute
from .experiments import atomic_json
from .portfolio import performance
from .rotation import load_panel


def rl_job(payload, device, threads):
    configure_threads(threads)
    from stable_baselines3 import PPO

    from .portfolio_env import PortfolioEnv

    spec, fold, frequency, seed, output, identity = payload
    panel, observed_identity = load_panel(spec)
    if identity != observed_identity:
        raise ValueError("Registered PPO data changed")
    train = panel.prefix(fold["start"])
    count = len(spec["universe"])
    features = train.features[252:, :count].reshape(-1, panel.features.shape[-1])
    mean, scale = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-6)
    costs = {"commission": spec["commission"], "slippage": spec["slippage"], "initial": spec["initial_cash"]}
    env = PortfolioEnv(train, 253, len(train.timestamp) - 1, count, frequency, mean, scale, **costs)
    args = spec["rl"]
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device=device,
        n_steps=args["n_steps"],
        batch_size=args["batch_size"],
        n_epochs=args["n_epochs"],
        learning_rate=args["learning_rate"],
        gamma=0.995,
        ent_coef=0.005,
        policy_kwargs={"net_arch": args["hidden_sizes"]},
    )
    start_time = time.monotonic()
    model.learn(total_timesteps=args["timesteps"])
    destination = Path(output) / fold["name"] / str(seed)
    destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / "model")
    panel = panel.prefix(fold["end"])
    indices = np.flatnonzero(panel.timestamp >= pd.Timestamp(fold["start"], tz="UTC"))
    test = PortfolioEnv(panel, int(indices[0]), int(indices[-1]), count, frequency, mean, scale, **costs)
    obs, _ = test.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test.step(action)
    result = {
        "fold": fold,
        "seed": seed,
        "metrics": performance(test.state),
        "training_end": str(train.bar_end[-1]),
        "timesteps": model.num_timesteps,
        "elapsed_seconds": time.monotonic() - start_time,
        "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "device": device,
    }
    pd.DataFrame(test.state.history).to_csv(destination / "equity.csv", index=False)
    pd.DataFrame(test.state.fills).to_csv(destination / "fills.csv", index=False)
    atomic_json(destination / "result.json", result)
    return result


def run_rl(spec, output, winner, development, identity):
    jobs = [
        (spec, fold, winner["frequency"], seed, str(output / "rl"), identity)
        for fold in [*spec["folds"], spec["confirmation"]]
        for seed in spec["rl"]["seeds"]
    ]
    rows = parallel_jobs(rl_job, jobs, Compute.model_validate(spec["compute"]))
    reference = {
        r["fold"]["name"]: r["metrics"] for r in development if r["candidate"]["name"] == winner["name"]
    }
    comparisons = [
        {
            "fold": r["fold"]["name"],
            "seed": r["seed"],
            "incremental_cagr": r["metrics"]["cagr"] - reference[r["fold"]["name"]]["cagr"],
            "drawdown_within_tolerance": r["metrics"]["max_drawdown"]
            <= reference[r["fold"]["name"]]["max_drawdown"] + spec["gate"]["drawdown_tolerance"],
        }
        for r in rows
        if r["fold"]["name"] in reference
    ]
    useful = (
        np.mean([r["incremental_cagr"] for r in comparisons]) > 0
        and np.mean([r["incremental_cagr"] > 0 for r in comparisons]) >= 2 / 3
        and all(r["drawdown_within_tolerance"] for r in comparisons)
    )
    return {
        "status": "completed",
        "rows": rows,
        "incremental_comparison": comparisons,
        "incremental_development_evidence": bool(useful),
        "deployment_approved": False,
        "note": "Fixed-budget PPO; all seeds retained, no best-seed selection. Historical comparisons are exploratory.",
    }
