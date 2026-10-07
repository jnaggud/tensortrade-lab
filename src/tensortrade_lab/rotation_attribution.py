"""Frozen-policy counterfactuals; descriptive decomposition, never selection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import digest, load_bundle
from .compute import configure_threads, parallel_jobs
from .config import Config
from .data import load_bars
from .etf_strategies import ETFPolicy
from .evaluation import rollout
from .experiments import atomic_json
from .focused import build_frame, utc


def attribution_job(payload, device, threads):
    configure_threads(threads)
    study, asset, destination = map(Path, payload)
    asset = str(asset)
    protocol = json.loads((study / "protocol.json").read_text())
    model, scaler, config, manifest = load_bundle(study / "finalists" / asset)
    raw = {}
    for symbol, source in protocol["datasets"].items():
        path = Path(source["source_path"])
        if digest(path) != source["source_sha256"]:
            raise ValueError("Frozen attribution source changed")
        raw[symbol] = load_bars(path, config.market.model_copy(update={"symbol": symbol}))
    frame = build_frame(raw[asset], raw[manifest["peer_symbol"]], config, manifest["feature_set"])
    frame = frame[frame.bar_end < utc(protocol["spec"]["holdout_end"])].reset_index(drop=True)
    start = int(frame.index[frame.timestamp >= utc(protocol["spec"]["holdout_start"])][0]) - 1
    original = json.loads((study / "finalists" / asset / "holdout" / "metrics.json").read_text())
    # Quantization matches the original six-action environment. Ex post and diagnostic.
    fraction = float(np.clip(round(original["agent"]["mean_exposure"] * 4) / 4, 0, 1))
    gross = config.model_copy(
        update={"execution": config.execution.model_copy(update={"commission": 0.0, "slippage_bps": 0.0})}
    )
    values = {}
    destination = destination / asset
    destination.mkdir(parents=True, exist_ok=True)
    for name, policy in [
        ("gross_strategy", model),
        ("gross_buy_hold", "buy_hold"),
        ("gross_constant_allocation", ETFPolicy("fixed", fraction=fraction, bypass_risk=True)),
    ]:
        metric, history, _ = rollout(frame, scaler, gross, policy, start)
        values[name] = metric["total_return"]
        history.to_csv(destination / f"{name}.csv", index=False)
    a, b = original["agent"]["total_return"], original["buy_hold"]["total_return"]
    g, h, c = values["gross_strategy"], values["gross_buy_hold"], values["gross_constant_allocation"]
    components = {
        "strategy_cost_effect": a - g,
        "timing_vs_constant_allocation": g - c,
        "constant_allocation_vs_full_exposure": c - h,
        "benchmark_cost_offset": h - b,
    }
    if not np.isclose(sum(components.values()), a - b, atol=1e-10):
        raise ArithmeticError("Attribution does not reconcile")
    return {
        "asset": asset,
        "net_strategy": a,
        "net_buy_hold": b,
        "excess": a - b,
        "constant_fraction": fraction,
        "actual_mean_exposure": original["agent"]["mean_exposure"],
        **values,
        "components": components,
        "note": "Sequential counterfactuals, not causal estimates. Allocation fraction rounded to original action grid using ex post average exposure. Zero cash interest and cash dividends preserve original accounting. Timing term includes differences from this constant allocation.",
    }


def attribute(study, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = parallel_jobs(
        attribution_job, [(str(study), s, str(output)) for s in ("SPY", "QQQ")], Config().compute
    )
    atomic_json(output / "attribution.json", rows)
    pd.DataFrame(
        [{"asset": r["asset"], **r["components"], "total_excess": r["excess"]} for r in rows]
    ).to_csv(output / "attribution.csv", index=False)
    return rows
