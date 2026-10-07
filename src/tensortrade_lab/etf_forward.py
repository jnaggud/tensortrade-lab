"""Append-only forward decisions; past missing decisions are never invented."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .artifacts import digest, load_bundle
from .data import load_bars
from .environment import ResearchEnv
from .evaluation import FILL_COLUMNS, metrics
from .experiments import atomic_json
from .focused import build_frame, engine_hash
from .market_data import schedule


def now_utc():
    return pd.Timestamp.now(tz="UTC")


def register(study):
    study = Path(study).resolve()
    if not (study / "final_results.json").exists():
        raise ValueError("Freeze and evaluate the historical finalists before registering forward decisions")
    directory = study / "prospective"
    directory.mkdir(exist_ok=True)
    assets = {}
    for asset in ("SPY", "QQQ"):
        bundle = study / "finalists" / asset
        _, _, _, manifest = load_bundle(bundle)
        assets[asset] = {
            "bundle": str(bundle),
            "manifest_sha256": digest(bundle / "manifest.json"),
            "model_sha256": manifest["model_sha256"],
        }
    with FileLock(str(directory / "registration.lock"), timeout=0):
        path = directory / "registration.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if (
                existing["assets"] != assets
                or existing["engine_sha256"] != engine_hash()
                or existing["forward_engine_sha256"] != digest(Path(__file__))
            ):
                raise ValueError("A registered policy or execution engine changed")
            return existing
        value = {
            "registered_at": now_utc().isoformat(),
            "assets": assets,
            "engine_sha256": engine_hash(),
            "forward_engine_sha256": digest(Path(__file__)),
            "status": "awaiting future observations",
            "mode": "forward paper decisions; no broker orders",
            "minimum_observation_sessions": 126,
            "assessment": "Compare net return and maximum drawdown with contemporaneous 100% buy-and-hold. Require positive excess and <=30% drawdown; neither condition alone establishes statistical evidence.",
            "missing_decisions": "Hold the existing position; never create retrospective actions.",
            "deployment_approved": False,
        }
        atomic_json(path, value)
        return value


def frame_digest(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def next_open(closed_at):
    sessions = schedule(str(closed_at.date()), str((closed_at + pd.Timedelta(days=10)).date()))
    return sessions.loc[sessions.open > closed_at, "open"].iloc[0]


def reconstruct(frame, scaler, config, signals):
    """Replay only previously recorded actions; hold through gaps in manual polling."""
    flat = np.r_[scaler.transform(frame)[-config.window :].reshape(-1), [0, 1, 0, 0, 0, 0]]
    observation = np.clip(flat, -10, 10).astype(np.float32)
    empty = {
        "equity": config.execution.initial_cash,
        "cash": config.execution.initial_cash,
        "units": 0.0,
        "exposure": 0.0,
    }
    if not signals:
        return observation, empty, None, None, None, 0
    first = pd.Timestamp(signals[0]["decision_at"])
    starts = frame.index[frame.bar_end == first]
    if not len(starts):
        raise ValueError("Source no longer contains the first forward decision")
    start = int(starts[0])
    if start == len(frame) - 1:
        return observation, empty, None, None, None, 0
    recorded = {signal["decision_at"]: signal for signal in signals}
    env = ResearchEnv(frame, scaler.transform(frame), config, start, liquidate_on_end=False)
    benchmark = ResearchEnv(
        frame, scaler.transform(frame), config, start, benchmark=True, liquidate_on_end=False
    )
    observation, _ = env.reset(seed=config.training.seed)
    benchmark.reset(seed=config.training.seed)
    missing = 0
    first_step = True
    while env.state.index < len(frame) - 1:
        i = env.state.index
        signal = recorded.get(frame.bar_end.iloc[i].isoformat())
        if signal:
            if pd.Timestamp(signal["recorded_at"]) >= frame.timestamp.iloc[i + 1]:
                raise ValueError("A recorded decision arrived after its execution candle opened")
            action = signal["action"]
        else:
            action = 0
            missing += 1
        observation, _, _, _, _ = env.step(action)
        benchmark.step(5 if first_step else 0)
        first_step = False
    history = pd.DataFrame(env.state.history)
    fills = pd.DataFrame(env.state.fills, columns=FILL_COLUMNS)
    a = metrics(history, fills, config.market.periods_per_year)
    b = metrics(
        pd.DataFrame(benchmark.state.history),
        pd.DataFrame(benchmark.state.fills, columns=FILL_COLUMNS),
        config.market.periods_per_year,
    )
    comparison = {
        "agent": a,
        "buy_hold": b,
        "excess_return": a["total_return"] - b["total_return"],
        "terminal_liquidation": False,
    }
    account = env.state.snapshot()
    env.close()
    benchmark.close()
    return observation, account, history, fills, comparison, missing


def poll(study, sources):
    study = Path(study).resolve()
    registration = register(study)
    directory = study / "prospective"
    results = {}
    with FileLock(str(directory / "poll.lock"), timeout=0):
        for asset, identity in registration["assets"].items():
            model, scaler, config, manifest = load_bundle(identity["bundle"])
            raw = {}
            for symbol, source in sources.items():
                data = load_bars(source, config.market.model_copy(update={"symbol": symbol}))
                raw[symbol] = data[data.bar_end <= now_utc()].reset_index(drop=True)
            state_path = directory / f"{asset}.json"
            old = (
                json.loads(state_path.read_text()) if state_path.exists() else {"signals": [], "prefixes": {}}
            )
            for symbol, previous in old["prefixes"].items():
                if (
                    len(raw[symbol]) < previous["rows"]
                    or frame_digest(raw[symbol].iloc[: previous["rows"]]) != previous["sha256"]
                ):
                    raise ValueError("Historical candles changed; forward sources must be append-only")
            frame = build_frame(raw[asset], raw[manifest["peer_symbol"]], config, manifest["feature_set"])
            if len(frame) < config.window:
                raise ValueError("Insufficient completed data for forward feature warmup")
            signals = old["signals"]
            obs, account, history, fills, comparison, missing = reconstruct(frame, scaler, config, signals)
            close = frame.bar_end.iloc[-1]
            opening = next_open(close)
            status = "decision already recorded"
            latest_matches = close == raw[asset].bar_end.iloc[-1]
            if not latest_matches:
                status = "awaiting completed peer data"
            elif now_utc() >= opening:
                status = "awaiting next completed session; past execution window missed"
            elif not signals or signals[-1]["decision_at"] != close.isoformat():
                if hasattr(model, "prepare"):
                    model.prepare(frame)
                action = (
                    int(model.action(frame, len(frame) - 1))
                    if hasattr(model, "action")
                    else int(model.predict(obs, deterministic=True)[0])
                )
                recorded_at = now_utc()
                if recorded_at >= opening:
                    status = "execution window closed during inference; no action recorded"
                else:
                    signals.append(
                        {
                            "decision_at": close.isoformat(),
                            "recorded_at": recorded_at.isoformat(),
                            "execution_not_before": opening.isoformat(),
                            "action": action,
                            "model_sha256": manifest["model_sha256"],
                        }
                    )
                    status = "recorded before next session open"
            value = {
                "asset": asset,
                "status": status,
                "updated_at": now_utc().isoformat(),
                "signals": signals,
                "account": account,
                "comparison": comparison,
                "missing_decision_sessions": missing,
                "prefixes": {
                    symbol: {"rows": len(data), "sha256": frame_digest(data)} for symbol, data in raw.items()
                },
                "prospective_sessions": 0 if comparison is None else comparison["agent"]["bars"],
                "deployment_approved": False,
            }
            if history is not None:
                history.to_csv(directory / f"{asset}_equity.csv", index=False)
                fills.to_csv(directory / f"{asset}_fills.csv", index=False)
            atomic_json(state_path, value)
            results[asset] = value
    return results
