"""Resumable process-parallel grid/TPE research with a sealed final holdout.

Only the coordinator writes Optuna storage. Workers see development snapshots;
final-test candles are loaded only after selection is frozen on disk.
"""

import hashlib
import itertools
import json
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import pandas as pd
import yaml
from filelock import FileLock
from pydantic import Field, model_validator

from .artifacts import digest, write_json
from .compute import configure_threads, hardware, plan_resources
from .config import Config, StrictConfig
from .data import load_bars
from .features import Scaler, engineer
from .market_data import resample_bars, trading_periods_per_year


class Dataset(StrictConfig):
    name: str
    path: str


class SearchSpec(StrictConfig):
    datasets: list[Dataset] = Field(min_length=1)
    timeframes: list[int] = Field(default_factory=lambda: [15, 60, 1440], min_length=1)
    strategies: list[Literal["momentum", "mean_reversion", "breakout", "supervised", "ppo"]] = Field(
        default_factory=lambda: ["momentum", "mean_reversion", "breakout", "supervised", "ppo"], min_length=1
    )
    directions: list[Literal["long_only", "long_short"]] = Field(
        default_factory=lambda: ["long_only"], min_length=1
    )
    parameters: dict[str, list] = Field(
        default_factory=lambda: {"policy.lookback_minutes": [240, 1440], "policy.threshold": [0.001, 0.005]}
    )
    method: Literal["grid", "tpe"] = "tpe"
    trials: int = Field(default=64, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [17, 42, 83], min_length=1)
    folds: int = Field(default=3, ge=2)
    holdout_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    start: str | None = None
    end: str | None = None
    config: Config = Field(default_factory=Config)

    @model_validator(mode="after")
    def check(self):
        if len({d.name for d in self.datasets}) != len(self.datasets):
            raise ValueError("Dataset names must be unique")
        if any(not values for values in self.parameters.values()):
            raise ValueError("Parameter choices cannot be empty")
        if any(x < 1 or x > 1440 for x in self.timeframes):
            raise ValueError("Timeframes must be 1..1440 minutes")
        allowed = {
            "training.learning_rate",
            "training.gamma",
            "training.entropy_coefficient",
            "training.n_steps",
            "training.batch_size",
            "training.n_epochs",
            "risk.max_position",
            "risk.stop_loss",
            "risk.take_profit",
            "risk.cooldown_minutes",
            "features.observation_minutes",
            "features.reference_minutes",
            "policy.lookback_minutes",
            "policy.threshold",
            "policy.forecast_minutes",
            "policy.trees",
            "policy.leaves",
            "policy.tree_learning_rate",
        }
        if set(self.parameters) - allowed:
            raise ValueError(
                f"Unsupported search parameters: {set(self.parameters) - allowed}. Execution costs are fixed assumptions, never optimized."
            )
        return self


def load_spec(path):
    return SearchSpec.model_validate(yaml.safe_load(Path(path).read_text()))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def prepare(spec, output):
    sources, starts, ends = {}, [], []
    for dataset in spec.datasets:
        path = Path(dataset.path).resolve()
        meta = json.loads(path.with_suffix(".metadata.json").read_text())
        if digest(path) != meta.get("sha256"):
            raise ValueError(f"Dataset checksum changed: {path}")
        from .config import Market

        market = Market.model_validate(meta["market"])
        bars = load_bars(path, market.model_copy(update={"require_regular_bars": False}))
        sources[dataset.name] = (path, meta, bars, market)
        starts.append(bars.timestamp.iloc[0])
        ends.append(bars.bar_end.iloc[-1])
    start, end = max(starts), min(ends)
    if spec.start:
        start = max(start, pd.Timestamp(spec.start, tz="UTC"))
    if spec.end:
        end = min(end, pd.Timestamp(spec.end, tz="UTC"))
    if end <= start:
        raise ValueError(
            "Datasets have no common historical date range; use separate studies for non-overlapping archives"
        )
    holdout = start + (end - start) * (1 - spec.holdout_fraction)
    markets = {}
    output.mkdir(parents=True, exist_ok=True)
    for name, (path, meta, bars, source_market) in sources.items():
        for minutes in spec.timeframes:
            if minutes < source_market.bar_minutes or minutes % source_market.bar_minutes:
                continue
            sampled = resample_bars(
                bars, source_market.model_copy(update={"require_regular_bars": False}), minutes
            )
            sampled = sampled[(sampled.timestamp >= start) & (sampled.bar_end <= end)].reset_index(drop=True)
            development = sampled[sampled.bar_end <= holdout].reset_index(drop=True)
            if len(development) < 100:
                continue
            key = f"{name}:{minutes}"
            target = source_market.model_copy(update={"bar_minutes": minutes})
            # Gaps remain explicit and affect results; they are never filled with prices.
            target = target.model_copy(
                update={
                    "require_regular_bars": False,
                    "periods_per_year": int(trading_periods_per_year(target)),
                }
            )
            destination = output / f"market-{len(markets):03d}.parquet"
            development.to_parquet(destination, index=False)
            markets[key] = {
                "development_path": str(destination.resolve()),
                "source_path": str(path),
                "source_sha256": meta["sha256"],
                "market": target.model_dump(),
                "source_market": source_market.model_dump(),
                "development_rows": len(development),
                "source_quality": meta["quality"],
                "synthetic": bool(meta.get("synthetic", False)),
            }
    if not markets:
        raise ValueError("No compatible dataset/timeframe has 100 development candles")
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "holdout_start": holdout.isoformat(),
        "markets": markets,
    }


def candidate_config(base, market, params, seed):
    raw = base.model_dump()
    raw["market"] = market
    raw["risk"]["direction"] = params["direction"]
    raw["training"]["seed"] = seed
    # Outer process parallelism owns the CPU budget.
    raw["compute"]["vector_envs"] = 1
    for name, value in params.items():
        if "." in name and not name.startswith("policy."):
            section, key = name.split(".", 1)
            raw[section][key] = value
    return Config.model_validate(raw)


def fold_dates(start, end, folds):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    edges = [start + (end - start) * (0.5 + 0.5 * i / folds) for i in range(folds + 1)]
    return [(edges[i], edges[i + 1]) for i in range(folds)]


def fit_candidate(train, validation, config, strategy, params, directory, device):
    from .strategies import fit_strategy

    scaler = Scaler.fit(train)
    directory.mkdir(parents=True, exist_ok=True)
    if strategy == "ppo":
        from .training import fit_ppo

        model, _ = fit_ppo(train, validation, scaler, config, directory, device)
    else:
        model = fit_strategy(
            strategy,
            train,
            config,
            {k.removeprefix("policy."): v for k, v in params.items() if k.startswith("policy.")},
        )
    return model, scaler


def run_candidate(payload):
    configure_threads(payload["threads"])
    from .evaluation import rollout

    spec = SearchSpec.model_validate(payload["spec"])
    params, item = payload["params"], payload["market"]
    raw = pd.read_parquet(item["development_path"])
    result = []
    for seed in spec.seeds:
        config = candidate_config(spec.config, item["market"], params, seed)
        frame = engineer(raw, config)
        for number, (score_start, score_end) in enumerate(
            fold_dates(payload["start"], payload["holdout_start"], spec.folds)
        ):
            gap = pd.Timedelta(minutes=config.split.purge_bars * config.market.bar_minutes)
            before = frame[frame.bar_end < score_start - gap].reset_index(drop=True)
            train_end = int(len(before) * 0.8)
            train = before.iloc[:train_end].reset_index(drop=True)
            validation = before.iloc[train_end + config.split.purge_bars :].reset_index(drop=True)
            # Include observation/rule warmup, but open the account at the score boundary.
            eligible = frame.index[(frame.timestamp >= score_start) & (frame.bar_end <= score_end)]
            if not len(eligible) or min(len(train), len(validation)) < config.window + 10:
                raise ValueError(
                    "Insufficient rows for folds, purge and observation window; use more history or smaller windows"
                )
            first, last = int(eligible[0]), int(eligible[-1])
            if first < config.window or last - first < 5:
                raise ValueError("Scoring interval too short")
            scoring = frame.iloc[: last + 1].reset_index(drop=True)
            directory = Path(payload["directory"]) / f"seed-{seed}" / f"fold-{number}"
            model, scaler = fit_candidate(
                train, validation, config, params["strategy"], params, directory, payload["device"]
            )
            metric, history, fills = rollout(scoring, scaler, config, model, first - 1)
            baseline, _, _ = rollout(scoring, scaler, config, "buy_hold", first - 1)
            history.to_csv(directory / "equity.csv", index=False)
            fills.to_csv(directory / "fills.csv", index=False)
            score = metric["total_return"] - metric["max_drawdown"]
            result.append(
                {
                    "seed": seed,
                    "fold": number,
                    "score": score,
                    "buy_hold_return": baseline["total_return"],
                    **metric,
                }
            )
    scores = [r["score"] for r in result]
    value = float(np.mean(scores) - 0.5 * np.std(scores))
    report = {
        "params": params,
        "value": value,
        "folds": result,
        "device": payload["device"],
        "mean_return": float(np.mean([r["total_return"] for r in result])),
        "worst_drawdown": max(r["max_drawdown"] for r in result),
        "fraction_beating_buy_hold": float(
            np.mean([r["total_return"] > r["buy_hold_return"] for r in result])
        ),
    }
    write_json(Path(payload["directory"]) / "result.json", report)
    return report


def leaderboard(study, output):
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "trial": trial.number,
                "status": trial.state.name,
                "score": trial.value,
                **trial.params,
                **{k: v for k, v in trial.user_attrs.items() if not isinstance(v, (dict, list))},
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values("score", ascending=False, na_position="last")
    temporary = output / "leaderboard.tmp.csv"
    frame.to_csv(temporary, index=False)
    temporary.replace(output / "leaderboard.csv")


def search(spec_path, output, max_trials=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "coordinator.lock"), timeout=0):
        return _search(load_spec(spec_path), output, max_trials)


def _search(spec, output, max_trials):
    if (output / "selection.json").exists():
        raise ValueError("Selection is frozen; this study cannot resume tuning after holdout access")
    signature_spec = spec.model_dump()
    signature_spec.pop("trials")
    for d in signature_spec["datasets"]:
        d["sha256"] = digest(Path(d["path"]))
        d["metadata_sha256"] = digest(Path(d["path"]).with_suffix(".metadata.json"))
    engine_files = (
        "config",
        "compute",
        "data",
        "environment",
        "evaluation",
        "experiments",
        "features",
        "margin",
        "market_data",
        "strategies",
        "training",
    )
    signature_spec["engine_sha256"] = hashlib.sha256(
        "".join(digest(Path(__file__).with_name(name + ".py")) for name in engine_files).encode()
    ).hexdigest()
    signature = hashlib.sha256(json.dumps(signature_spec, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "study.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Study configuration or input data changed; choose a new output directory")
    else:
        manifest = {
            "signature": signature,
            "spec": spec.model_dump(),
            **prepare(spec, output / "development"),
        }
        write_json(manifest_path, manifest)
    choices = {
        "market": list(manifest["markets"]),
        "strategy": spec.strategies,
        "direction": spec.directions,
        **spec.parameters,
    }
    sampler = (
        optuna.samplers.TPESampler(seed=spec.config.training.seed, constant_liar=True)
        if spec.method == "tpe"
        else optuna.samplers.RandomSampler(seed=spec.config.training.seed)
    )
    study = optuna.create_study(
        study_name="research",
        storage=f"sqlite:///{output / 'study.sqlite3'}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    interrupted_path = output / "interrupted.json"
    interrupted = set(json.loads(interrupted_path.read_text())) if interrupted_path.exists() else set()
    for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        interrupted.add(trial.number)
        atomic_json(interrupted_path, sorted(interrupted))
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        study.enqueue_trial(trial.params)
    target = max_trials or spec.trials
    if spec.method == "grid":
        keys = list(choices)
        # enqueue lazily up to this run's budget; avoid materializing a huge Cartesian product.
        existing = {
            json.dumps(t.params or t.system_attrs.get("fixed_params", {}), sort_keys=True)
            for t in study.trials
        }
        combinations = (
            list(itertools.product(*(choices[k] for k in keys)))
            if np.prod([len(v) for v in choices.values()]) <= 100000
            else None
        )
        if combinations is None:
            raise ValueError("Grid exceeds 100,000 combinations; narrow the grid or use TPE")
        np.random.default_rng(spec.config.training.seed).shuffle(combinations)
        for values in combinations:
            if len(study.trials) - len(interrupted) >= target:
                break
            params = dict(zip(keys, values, strict=True))
            if json.dumps(params, sort_keys=True) not in existing:
                study.enqueue_trial(params)

    completed = sum(
        t.number not in interrupted
        for t in study.get_trials(states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL))
    )
    remaining = max(0, target - completed)
    if spec.method == "grid":
        remaining = min(remaining, len(study.get_trials(states=(optuna.trial.TrialState.WAITING,))))
    if not remaining:
        leaderboard(study, output)
        return output
    plan = plan_resources(spec.config.compute, remaining)
    configure_threads(plan.threads_per_worker)
    write_json(output / "hardware.json", {"detected": hardware(), "plan": plan.to_dict()})
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) for _ in plan.devices]
    pending, available, dispatched = {}, list(range(len(pools))), 0
    try:
        while pending or dispatched < remaining:
            while available and dispatched < remaining:
                trial = study.ask()
                params = {k: trial.suggest_categorical(k, values) for k, values in choices.items()}
                preferred = [
                    i for i in available if (plan.devices[i] != "cpu") == (params["strategy"] == "ppo")
                ]
                slot = preferred[0] if preferred else available[0]
                available.remove(slot)
                directory = output / f"trial-{trial.number:05d}"
                directory.mkdir(exist_ok=True)
                device = plan.devices[slot] if params["strategy"] == "ppo" else "cpu"
                payload = {
                    "spec": spec.model_dump(),
                    "params": params,
                    "market": manifest["markets"][params["market"]],
                    "start": manifest["start"],
                    "holdout_start": manifest["holdout_start"],
                    "directory": str(directory),
                    "device": device,
                    "threads": plan.threads_per_worker,
                }
                pending[pools[slot].submit(run_candidate, payload)] = (slot, trial)
                dispatched += 1
            atomic_json(
                output / "status.json",
                {
                    "state": "running",
                    "completed": completed,
                    "target": target,
                    "active_trials": [t.number for _, t in pending.values()],
                    "workers": plan.worker_count,
                },
            )
            finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                slot, trial = pending.pop(future)
                available.append(slot)
                try:
                    result = future.result()
                    for key in ("mean_return", "worst_drawdown", "fraction_beating_buy_hold", "device"):
                        trial.set_user_attr(key, result[key])
                    study.tell(trial, result["value"])
                    print(f"Trial {trial.number}: score={result['value']:.4f} {trial.params}", flush=True)
                except Exception as error:  # noqa: BLE001 - record isolated worker failures without losing the study
                    trial.set_user_attr("error", str(error)[:2000])
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    print(f"Trial {trial.number} failed: {error}", flush=True)
                completed += 1
                leaderboard(study, output)
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
    atomic_json(
        output / "status.json",
        {
            "state": "complete"
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            else "failed",
            "completed": completed,
            "target": target,
            "active_trials": [],
        },
    )
    return output


def finalize(output):
    output = Path(output).resolve()
    with FileLock(str(output / "coordinator.lock"), timeout=0):
        return _finalize(output)


def _finalize(output):
    import joblib

    from .evaluation import evaluate_suite, rollout

    if (output / "holdout.json").exists():
        return json.loads((output / "holdout.json").read_text())
    manifest = json.loads((output / "study.json").read_text())
    spec = SearchSpec.model_validate(manifest["spec"])
    study = optuna.load_study(study_name="research", storage=f"sqlite:///{output / 'study.sqlite3'}")
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        best = study.best_trial
        selection = {
            "trial": best.number,
            "params": best.params,
            "validation_score": best.value,
            "holdout_start": manifest["holdout_start"],
            "seed": spec.seeds[0],
        }
        # Freeze before loading any final-test candles. Re-running cannot choose another trial.
        write_json(selection_path, selection)
    params = selection["params"]
    item = manifest["markets"][params["market"]]
    if digest(Path(item["source_path"])) != item["source_sha256"]:
        raise ValueError("Dataset changed since search")
    config = candidate_config(spec.config, item["market"], params, selection["seed"])
    configure_threads(config.compute.threads_per_worker)
    from .config import Market

    bars = load_bars(
        item["source_path"],
        Market.model_validate(item["source_market"]).model_copy(update={"require_regular_bars": False}),
    )
    bars = resample_bars(
        bars,
        Market.model_validate(item["source_market"]).model_copy(update={"require_regular_bars": False}),
        config.market.bar_minutes,
    )
    bars = bars[
        (bars.timestamp >= pd.Timestamp(manifest["start"])) & (bars.bar_end <= pd.Timestamp(manifest["end"]))
    ].reset_index(drop=True)
    frame = engineer(bars, config)
    boundary = pd.Timestamp(manifest["holdout_start"])
    gap = pd.Timedelta(minutes=config.split.purge_bars * config.market.bar_minutes)
    development = frame[frame.bar_end < boundary - gap].reset_index(drop=True)
    train_end = int(len(development) * 0.8)
    train = development.iloc[:train_end].reset_index(drop=True)
    validation = development.iloc[train_end + config.split.purge_bars :].reset_index(drop=True)
    future = frame.index[frame.timestamp >= boundary]
    if not len(future) or min(len(train), len(validation)) < config.window + 10:
        raise ValueError("Insufficient final training/validation/holdout candles")
    directory = output / "finalist"
    directory.mkdir(exist_ok=True)
    model, scaler = fit_candidate(train, validation, config, params["strategy"], params, directory, "cpu")
    if params["strategy"] != "ppo":
        joblib.dump(model, directory / "policy.joblib")
    write_json(
        directory / "policy.json",
        {
            "config": config.model_dump(),
            "scaler": asdict(scaler),
            "strategy": params["strategy"],
            "selection_end": boundary.isoformat(),
        },
    )
    from .artifacts import UPSTREAM_COMMIT

    model_file = "best_model.zip" if params["strategy"] == "ppo" else "policy.joblib"
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "upstream_commit": UPSTREAM_COMMIT,
            "config": config.model_dump(),
            "scaler": asdict(scaler),
            "model_file": model_file,
            "model_sha256": digest(directory / model_file),
            "source": item["source_path"],
            "source_sha256": item["source_sha256"],
            "train_end": train.timestamp.iloc[-1].isoformat(),
            "selection_end": manifest["end"],
            "synthetic": item.get("synthetic", False),
            "note": "Paper trading starts after the entire research/holdout period.",
        },
    )
    results = evaluate_suite(frame, scaler, config, model, directory / "holdout", int(future[0]) - 1)
    stresses = {}
    for label, multiplier, delay in (
        ("double_costs", 2, 0),
        ("one_bar_delay", 1, 1),
        ("double_costs_and_delay", 2, 1),
    ):
        raw = config.model_dump()
        raw["execution"].update(
            commission=config.execution.commission * multiplier,
            slippage_bps=config.execution.slippage_bps * multiplier,
            delay_bars=config.execution.delay_bars + delay,
        )
        stressed = Config.model_validate(raw)
        stresses[label] = rollout(frame, scaler, stressed, model, int(future[0]) - 1)[0]
    report = {
        "selection": selection,
        "metrics": results,
        "stress_tests": stresses,
        "note": "Final holdout accessed after selection was frozen. Any subsequent research needs a new untouched period. No profitability guarantee.",
    }
    write_json(output / "holdout.json", report)
    return report
