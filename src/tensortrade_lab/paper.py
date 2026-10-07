"""Deterministic paper replay of newly appended, completed candles. No exchange API calls."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout

from .artifacts import digest, load_bundle, new_run, write_json
from .data import load_bars
from .environment import ResearchEnv
from .evaluation import FILL_COLUMNS, evaluate_suite
from .features import engineer
from .report import write_report


def future_frame(source, config, manifest, raw_bars=None):
    if manifest.get("feature_builder") == "etf":
        raise ValueError("ETF bundles require both assets: use focused-audit or focused-forward")
    frame = engineer(load_bars(source, config.market) if raw_bars is None else raw_bars, config)
    # The validation boundary is the latest timestamp used to select the model.
    boundary = pd.Timestamp(manifest["selection_end"])
    future = frame.index[frame.timestamp > boundary]
    if not len(future):
        raise ValueError("CSV has no candles after the model-selection period")
    start = max(config.window - 1, int(future[0]) - 1)
    if start >= len(frame) - 1:
        raise ValueError("Include enough candles for indicator warmup, observation history, and execution")
    return frame, start


def backtest(run, source, output="runs"):
    model, scaler, config, manifest = load_bundle(run)
    frame, start = future_frame(source, config, manifest)
    destination = new_run(output, "backtest")
    results = evaluate_suite(frame, scaler, config, model, destination, start)
    write_json(destination / "metrics.json", results)
    write_json(
        destination / "evaluation.json",
        {
            "model_run": str(Path(run).resolve()),
            "source_sha256": digest(Path(source)),
            "selection_end": manifest["selection_end"],
            "synthetic": manifest["synthetic"],
        },
    )
    write_report(destination, results, manifest["synthetic"])
    return destination


def replay(run, source, state_path):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(state_path) + ".lock", timeout=0):
            return _replay(run, source, state_path)
    except Timeout as error:
        raise ValueError("Another paper replay is writing this account") from error


def _replay(run, source, state_path):
    """Replay full history to reconstruct paper state; verify immutable prefix on each call.

    Replaying avoids unsafe serialized object state and makes reruns idempotent. Cost
    is O(history). There is no terminal liquidation here: the last candle stays open
    for the next invocation's next-open action.
    """
    model, scaler, config, manifest = load_bundle(run)
    raw_bars = load_bars(source, config.market)
    frame, start = future_frame(source, config, manifest, raw_bars)
    state_path = Path(state_path)
    canonical = raw_bars.to_csv(index=False)
    old = None
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get("schema_version") != 2:
            raise ValueError("Unsupported paper state version; choose a new state path")
        if old["model_sha256"] != manifest["model_sha256"]:
            raise ValueError("Paper state belongs to a different model; choose a new state path")
        prefix = raw_bars.iloc[: old["source_rows"]].to_csv(index=False)
        if (
            len(raw_bars) < old["source_rows"]
            or hashlib.sha256(prefix.encode()).hexdigest() != old["prefix_sha256"]
        ):
            raise ValueError("Historical candles changed; paper replay requires an append-only CSV")
    env = ResearchEnv(frame, scaler.transform(frame), config, start, liquidate_on_end=False)
    obs, _ = env.reset(seed=config.training.seed)
    if hasattr(model, "prepare"):
        model.prepare(frame)

    def action():
        return (
            model.action(frame, env.state.index)
            if hasattr(model, "action")
            else int(model.predict(obs, deterministic=True)[0])
        )

    while env.state.index < len(frame) - 1:
        obs, _, _, _, _ = env.step(action())
    fill_path = state_path.with_suffix(".fills.csv")
    equity_path = state_path.with_suffix(".equity.csv")
    snapshot = {
        "schema_version": 2,
        "mode": "paper_replay",
        "model_sha256": manifest["model_sha256"],
        "prefix_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "rows": len(frame),
        "source_rows": len(raw_bars),
        "account": env.state.snapshot(),
        "fills": len(env.state.fills),
        "next_action": action(),
        "new_fills": len(env.state.fills) - (old["fills"] if old else 0),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(env.state.fills, columns=FILL_COLUMNS).to_csv(fill_path, index=False)
    pd.DataFrame(env.state.history).to_csv(equity_path, index=False)
    temporary = state_path.with_suffix(".tmp")
    write_json(temporary, snapshot)
    temporary.replace(state_path)
    env.close()
    return snapshot
