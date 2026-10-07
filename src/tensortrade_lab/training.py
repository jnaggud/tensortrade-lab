from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from .artifacts import digest, new_run, save_manifest, write_json
from .compute import configure_threads, parallel_jobs, resolve_device
from .config import Config
from .data import load_bars
from .environment import ResearchEnv
from .evaluation import Partitions, chronological_split, evaluate_suite, rollout
from .features import Scaler, engineer
from .objectives import selection_score
from .report import write_report


class ValidationCallback(BaseCallback):
    def __init__(self, bars, scaler, config, run):
        super().__init__()
        self.bars, self.scaler, self.config, self.run = bars, scaler, config, run
        self.best_score = -np.inf
        self.stale = 0
        self.log = []
        self.baseline = (
            rollout(bars, scaler, config, "buy_hold")[0] if config.objective == "excess_return" else None
        )

    def evaluate(self):
        result, _, _ = rollout(self.bars, self.scaler, self.config, self.model)
        # Validation chooses a cost-aware, drawdown-penalized checkpoint. Test is never consulted.
        score = selection_score(result, self.baseline, self.config)
        improved = score > self.best_score + 1e-8
        if improved:
            self.best_score, self.stale = score, 0
            self.model.save(self.run / "best_model.zip")
        else:
            self.stale += 1
        self.log.append(
            {
                "timesteps": self.num_timesteps,
                "score": score,
                "selected": improved,
                "excess_return": result["total_return"] - self.baseline["total_return"]
                if self.baseline
                else None,
                **result,
            }
        )
        write_json(self.run / "validation.json", self.log)
        self.model.save(self.run / "last_model.zip")
        print(
            f"Validation @ {self.num_timesteps}: return={result['total_return']:.2%}, "
            f"drawdown={result['max_drawdown']:.2%}, fills={result['fills']}",
            flush=True,
        )

    def _on_step(self):
        if self.num_timesteps // self.config.training.eval_every > getattr(self, "last_eval_bucket", 0):
            self.last_eval_bucket = self.num_timesteps // self.config.training.eval_every
            self.evaluate()
        return self.stale < self.config.training.patience


def _training_env(frame, features, config, start):
    return Monitor(ResearchEnv(frame, features, config, start_index=start, training=True))


def fit_ppo(train, validation, scaler, config, run, device=None, resume_from=None):
    configure_threads(config.compute.threads_per_worker)
    count = config.compute.vector_envs
    if count == 0:
        import os

        from .compute import hardware

        count = min(
            max(1, (os.cpu_count() or 1) // config.compute.threads_per_worker),
            max(1, config.training.total_timesteps // config.training.n_steps),
            max(1, len(train) // (config.window + 10)),
            max(1, int(hardware()["available_memory_gib"] // 2)),
        )
    features = scaler.transform(train)
    if count > 1:
        import os

        if count * config.compute.threads_per_worker > (os.cpu_count() or 1):
            raise ValueError("Vector environments exceed the CPU budget")
        starts = [config.window - 1 + i * (len(train) - config.window) // count for i in range(count)]
        env = SubprocVecEnv(
            [partial(_training_env, train, features, config, start) for start in starts], start_method="spawn"
        )
    else:
        env = _training_env(train, features, config, config.window - 1)
    t = config.training
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=t.learning_rate,
        n_steps=t.n_steps,
        batch_size=t.batch_size,
        n_epochs=t.n_epochs,
        gamma=t.gamma,
        gae_lambda=t.gae_lambda,
        clip_range=t.clip_range,
        ent_coef=t.entropy_coefficient,
        policy_kwargs={"net_arch": t.hidden_sizes},
        seed=t.seed,
        device=resolve_device(device or config.compute.device),
        verbose=0,
    )
    callback = ValidationCallback(validation, scaler, config, run)
    if resume_from:
        import json
        import shutil

        previous = Path(resume_from)
        model = PPO.load(
            previous / "last_model.zip", env=env, device=resolve_device(device or config.compute.device)
        )
        callback.log = json.loads((previous / "validation.json").read_text())
        callback.best_score = max(row["score"] for row in callback.log)
        callback.last_eval_bucket = model.num_timesteps // t.eval_every
        if previous.resolve() != run.resolve():
            shutil.copy2(previous / "best_model.zip", run / "best_model.zip")
    remaining = max(0, t.total_timesteps - model.num_timesteps)
    if remaining:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=not bool(resume_from),
        )
    else:
        callback.init_callback(model)
        callback.num_timesteps = model.num_timesteps
    if not callback.log or callback.log[-1]["timesteps"] != model.num_timesteps:
        callback.evaluate()
    model.save(run / "last_model.zip")
    env.close()
    return PPO.load(
        run / "best_model.zip", device=resolve_device(device or config.compute.device)
    ), model.num_timesteps


def train_partition(
    frame, source: Path, config: Config, split: Partitions, run: Path, synthetic=False, device=None
):
    train = frame.iloc[slice(*split.train)].reset_index(drop=True)
    validation = frame.iloc[slice(*split.validation)].reset_index(drop=True)
    test = frame.iloc[slice(*split.test)].reset_index(drop=True)
    scaler = Scaler.fit(train)
    manifest = save_manifest(run, config, scaler, source, split, frame, synthetic)
    best, timesteps = fit_ppo(train, validation, scaler, config, run, device)
    manifest["actual_timesteps"] = timesteps
    manifest["model_sha256"] = digest(run / "best_model.zip")
    write_json(run / "manifest.json", manifest)
    results = evaluate_suite(test, scaler, config, best, run / "test")
    write_json(run / "test" / "metrics.json", results)
    write_report(run, results, synthetic)
    return results


def train(source: str | Path, config: Config, output="runs", synthetic=False):
    source = Path(source)
    frame = engineer(load_bars(source, config.market), config)
    split = chronological_split(len(frame), config)
    run = new_run(output)
    train_partition(frame, source, config, split, run, synthetic)
    return run


def walk_forward(source, config, output="runs", folds=3, synthetic=False):
    if folds < 2:
        raise ValueError("Walk-forward needs at least two folds")
    source = Path(source)
    frame = engineer(load_bars(source, config.market), config)
    n = len(frame)
    initial = int(n * config.split.train_fraction)
    validation_size = int(n * config.split.validation_fraction)
    gap = config.split.purge_bars
    test_size = (n - initial - validation_size - 2 * gap) // folds
    if test_size < config.window + 10:
        raise ValueError("Not enough candles for the requested walk-forward folds")
    run = new_run(output, "walk-forward")
    jobs = []
    for fold in range(folds):
        train_end = initial + fold * test_size
        validation_start = train_end + gap
        validation_end = validation_start + validation_size
        test_start = validation_end + gap
        test_end = n if fold == folds - 1 else test_start + test_size
        split = Partitions((0, train_end), (validation_start, validation_end), (test_start, test_end))
        fold_dir = run / f"fold-{fold + 1:02d}"
        fold_dir.mkdir()
        raw = config.model_dump()
        raw["compute"]["vector_envs"] = 1
        jobs.append((frame, source, Config.model_validate(raw), split, fold_dir, synthetic, fold))
    rows = parallel_jobs(_walk_forward_job, jobs, config.compute)
    pd.DataFrame(rows).to_csv(run / "folds.csv", index=False)
    write_json(
        run / "summary.json",
        {
            "folds": rows,
            "synthetic": synthetic,
            "note": "Each fold starts with fresh capital; returns are not a stitched portfolio.",
            "fraction_beating_buy_hold": float(
                np.mean([r["agent_return"] > r["buy_hold_return"] for r in rows])
            ),
        },
    )
    return run


def _walk_forward_job(payload, device, threads):
    configure_threads(threads)
    frame, source, config, split, fold_dir, synthetic, fold = payload
    results = train_partition(frame, source, config, split, fold_dir, synthetic, device)
    return {
        "fold": fold + 1,
        "agent_return": results["agent"]["total_return"],
        "buy_hold_return": results["buy_hold"]["total_return"],
        "max_drawdown": results["agent"]["max_drawdown"],
        "start": results["agent"]["start"],
        "end": results["agent"]["end"],
    }
