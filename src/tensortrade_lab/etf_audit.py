"""Diagnostics on explicitly previously seen market regimes. Never selects a model."""

import json
from pathlib import Path

import pandas as pd

from .artifacts import digest, load_bundle
from .compute import configure_threads, parallel_jobs
from .config import Config
from .data import load_bars
from .experiments import atomic_json
from .focused import build_frame, evaluate_policy, utc

REGIMES = [
    ("2016_2019", "2016-01-01", "2020-01-01"),
    ("2020_2021", "2020-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023_onward", "2023-01-01", "2027-01-01"),
    ("all_previously_seen", "2016-01-01", "2027-01-01"),
]


def audit_job(payload, device, threads):
    configure_threads(threads)
    study, asset, regime, begin, end, source_paths = payload
    study = Path(study)
    model, scaler, config, manifest = load_bundle(study / "finalists" / asset)
    protocol = json.loads((study / "protocol.json").read_text())
    raw = {}
    for symbol, recent in source_paths.items():
        market = config.market.model_copy(update={"symbol": symbol})
        original = protocol["datasets"][symbol]
        if digest(Path(original["source_path"])) != original["source_sha256"]:
            raise ValueError("Registered historical source changed")
        metadata = json.loads(Path(recent).with_suffix(".metadata.json").read_text())
        if (
            metadata["sha256"] != digest(Path(recent))
            or metadata["market"]["symbol"] != symbol
            or metadata["market"]["price_adjustment"] != "split_adjusted"
        ):
            raise ValueError("Diagnostic source identity, checksum or adjustment mismatch")
        history = load_bars(original["source_path"], market)
        future = load_bars(recent, market)
        raw[symbol] = (
            pd.concat([history, future])
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        raw[symbol] = raw[symbol][raw[symbol].bar_end <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)
    frame = build_frame(raw[asset], raw[manifest["peer_symbol"]], config, manifest["feature_set"])
    frame = frame[frame.bar_end < utc(end)].reset_index(drop=True)
    eligible = frame.index[frame.timestamp >= utc(begin)]
    if not len(eligible) or eligible[0] < config.window:
        raise ValueError("Insufficient history for the requested regime")
    final = json.loads((study / "finalists" / asset / "result.json").read_text())
    destination = study / "previously_seen" / asset / regime
    metrics, _ = evaluate_policy(
        frame, scaler, config, model, int(eligible[0]) - 1, destination, final["matched_fraction_from_tuning"]
    )
    result = {
        "asset": asset,
        "regime": regime,
        "metrics": metrics,
        "excess_return": metrics["agent"]["total_return"] - metrics["buy_hold"]["total_return"],
        "previously_seen": True,
    }
    atomic_json(destination / "result.json", result)
    return result


def audit(study, sources=None):
    study = Path(study).resolve()
    if not (study / "final_results.json").exists():
        raise ValueError("Finish and freeze the focused experiment before running regime diagnostics")
    sources = sources or {symbol: f"data/yahoo/{symbol}_1d.parquet" for symbol in ("SPY", "QQQ")}
    identity = {
        symbol: {"path": str(Path(path).resolve()), "sha256": digest(Path(path))}
        for symbol, path in sources.items()
    }
    protocol = json.loads((study / "protocol.json").read_text())
    jobs = [
        (str(study), asset, *regime, sources) for asset in protocol["spec"]["assets"] for regime in REGIMES
    ]
    results = parallel_jobs(audit_job, jobs, Config.model_validate(protocol["spec"]["config"]).compute)
    output = {
        "results": results,
        "sources": identity,
        "classification": "Previously seen 2016–2026 diagnostics; not pristine out-of-sample evidence. Each regime resets capital. No model selection or retraining occurs.",
    }
    atomic_json(study / "previously_seen.json", output)
    from .etf_report import write_report

    write_report(study)
    return output
