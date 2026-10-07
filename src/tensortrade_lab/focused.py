"""Preregistered ETF experiments: staged learning, nested dates and sealed evaluation."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from filelock import FileLock

from .artifacts import UPSTREAM_COMMIT, digest, write_json
from .compute import configure_threads, hardware, parallel_jobs
from .config import Config, Market
from .data import load_bars
from .etf_features import etf_frame
from .etf_strategies import ETFPolicy, fit_etf_policy
from .evaluation import rollout
from .experiments import atomic_json
from .features import Scaler
from .objectives import aggregate, learning_gain
from .training import fit_ppo


def utc(value):
    return (
        pd.Timestamp(value).tz_localize("UTC")
        if pd.Timestamp(value).tz is None
        else pd.Timestamp(value).tz_convert("UTC")
    )


def read_spec(path):
    spec = yaml.safe_load(Path(path).read_text())
    if spec["assets"] != ["SPY", "QQQ"]:
        raise ValueError("This focused protocol requires SPY and QQQ")
    if (
        any(b <= 0 for b in spec["stage_budgets"])
        or sorted(set(spec["stage_budgets"])) != spec["stage_budgets"]
    ):
        raise ValueError("Stage budgets must strictly increase")
    if len(spec["seeds"]) != len(set(spec["seeds"])):
        raise ValueError("Seeds must be unique")
    boundary = utc(spec["holdout_start"])
    previous = None
    for fold in spec["folds"]:
        a, b, c = map(utc, (fold["training_end"], fold["tuning_end"], fold["scoring_end"]))
        if not a < b < c <= boundary or (previous is not None and b < previous):
            raise ValueError("Folds must be chronological and non-overlapping, entirely before holdout")
        previous = c
    Config.model_validate(spec["config"])
    return spec


def engine_hash():
    files = (
        "focused",
        "etf_features",
        "etf_strategies",
        "objectives",
        "features",
        "environment",
        "margin",
        "training",
        "config",
        "evaluation",
    )
    return hashlib.sha256(
        "".join(digest(Path(__file__).with_name(f"{name}.py")) for name in files).encode()
    ).hexdigest()


def prepare(spec, output):
    manifest_path = output / "protocol.json"
    fingerprint = {
        "spec": spec,
        "engine": engine_hash(),
        "sources": {asset: digest(Path(path)) for asset, path in spec["sources"].items()},
    }
    fingerprint["metadata"] = {
        asset: digest(Path(path).with_suffix(".metadata.json")) for asset, path in spec["sources"].items()
    }
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Protocol, engine or data changed; use a new output directory")
        return manifest
    development = output / "development"
    development.mkdir(parents=True, exist_ok=True)
    details = {}
    for asset, path in spec["sources"].items():
        source = Path(path)
        meta = json.loads(source.with_suffix(".metadata.json").read_text())
        if (
            digest(source) != meta["sha256"]
            or meta["market"]["symbol"] != asset
            or meta["market"]["price_adjustment"] != "split_adjusted"
        ):
            raise ValueError("Input identity, checksum or price-adjustment mismatch")
        bars = load_bars(
            source, Market.model_validate(meta["market"]).model_copy(update={"require_regular_bars": True})
        )
        dev = bars[bars.bar_end < utc(spec["holdout_start"])].reset_index(drop=True)
        dev.to_parquet(development / f"{asset}.parquet", index=False)
        details[asset] = {
            "source_path": str(source.resolve()),
            "source_sha256": digest(source),
            "development_sha256": digest(development / f"{asset}.parquet"),
            "development_rows": len(dev),
            "quality": meta["quality"],
            "synthetic": meta.get("synthetic", meta["quality"].get("synthetic", False)),
        }
    manifest = {
        "signature": signature,
        "engine": fingerprint["engine"],
        "created_at": datetime.now(UTC).isoformat(),
        "spec": spec,
        "datasets": details,
        "holdout_accessed": False,
        "primary_target": "Excess return over 100% same-asset buy-and-hold, after equal costs",
        "previously_seen_stress_period": "2016-2026; not pristine holdout evidence",
    }
    write_json(manifest_path, manifest)
    write_json(output / "hardware.json", hardware())
    return manifest


def config_for(spec, asset, candidate, seed, budget):
    raw = json.loads(json.dumps(spec["config"]))
    raw["market"]["symbol"] = asset
    raw["objective"] = "excess_return"
    raw["selection_drawdown_limit"] = spec["max_drawdown"]
    raw["training"].update(seed=seed, total_timesteps=budget)
    for key in ("learning_rate", "gamma"):
        if key in candidate:
            raw["training"][key] = candidate[key]
    return Config.model_validate(raw)


def build_frame(own, peer, config, feature_set):
    # Own/cross ablations use exactly the same valid dates and own features.
    frame = etf_frame(own, peer, config, "cross")
    if feature_set == "own":
        frame = frame.drop(columns=[c for c in frame if c.startswith(("context_peer_", "context_relative_"))])
    return frame


def partitions(frame, fold, config):
    train = frame[frame.bar_end < utc(fold["training_end"])]
    if config.split.purge_bars:
        train = train.iloc[: -config.split.purge_bars]
    train = train.reset_index(drop=True)
    tune = frame[
        (frame.timestamp >= utc(fold["training_end"])) & (frame.bar_end < utc(fold["tuning_end"]))
    ].reset_index(drop=True)
    score = frame[frame.bar_end < utc(fold["scoring_end"])].reset_index(drop=True)
    eligible = score.index[score.timestamp >= utc(fold["tuning_end"])]
    if min(len(train), len(tune)) < config.window + 50 or not len(eligible) or eligible[0] < config.window:
        raise ValueError("Insufficient chronological history for the protocol")
    return train, tune, score, int(eligible[0]) - 1


def evaluate_policy(frame, scaler, config, model, start, directory, matched_fraction):
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    histories = {}
    policies = {
        "agent": model,
        "buy_hold": "buy_hold",
        "cash": "cash",
        "fixed_50": ETFPolicy("fixed", fraction=0.5, bypass_risk=True),
        "volatility_12": ETFPolicy("volatility", bypass_risk=True),
        "matched_exposure": ETFPolicy("fixed", fraction=matched_fraction, bypass_risk=True),
    }
    for name, policy in policies.items():
        metric, history, fills = rollout(frame, scaler, config, policy, start)
        history.to_csv(directory / f"{name}_equity.csv", index=False)
        fills.to_csv(directory / f"{name}_fills.csv", index=False)
        results[name] = metric
        histories[name] = history
    write_json(directory / "metrics.json", results)
    return results, histories


def fit_job(payload, device, threads):
    configure_threads(threads)
    output = Path(payload["output"])
    spec = payload["spec"]
    candidate = payload["candidate"]
    asset, seed, fold, budget = payload["asset"], payload["seed"], payload["fold"], payload["budget"]
    folder = output / "fits" / asset / candidate["name"] / f"seed-{seed}" / fold["name"] / f"steps-{budget}"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "fit.lock"), timeout=0):
        if (folder / "result.json").exists():
            return json.loads((folder / "result.json").read_text())
        config = config_for(spec, asset, candidate, seed, budget)
        peer = next(a for a in spec["assets"] if a != asset)
        own = pd.read_parquet(output / "development" / f"{asset}.parquet")
        other = pd.read_parquet(output / "development" / f"{peer}.parquet")
        frame = build_frame(own, other, config, candidate.get("feature_set", "cross"))
        train, tune, scoring, start = partitions(frame, fold, config)
        scaler = Scaler.fit(train)
        if candidate["strategy"] == "ppo":
            previous = payload.get("previous")
            if all(
                (folder / name).exists() for name in ("last_model.zip", "best_model.zip", "validation.json")
            ):
                previous = folder
            model, steps = fit_ppo(train, tune, scaler, config, folder, device, previous)
            log = json.loads((folder / "validation.json").read_text())
            gain = learning_gain(log, budget)
        else:
            model = fit_etf_policy(candidate, train, seed)
            joblib.dump(model, folder / "policy.joblib")
            steps = 0
            gain = 0
        tuning = rollout(tune, scaler, config, model)[0]
        matched = round(tuning["mean_gross_exposure"] * 4) / 4
        results, _ = evaluate_policy(scoring, scaler, config, model, start, folder / "score", matched)
        metric = results["agent"]
        baseline = results["buy_hold"]
        report = {
            "asset": asset,
            "candidate": candidate["name"],
            "strategy": candidate["strategy"],
            "feature_set": candidate.get("feature_set", "cross"),
            "seed": seed,
            "fold": fold["name"],
            "budget": budget,
            "actual_steps": steps,
            "learning_gain": gain,
            "device": device,
            "directory": str(folder.resolve()),
            "excess_return": metric["total_return"] - baseline["total_return"],
            "buy_hold_return": baseline["total_return"],
            "matched_fraction_from_tuning": matched,
            "metrics": results,
            **metric,
        }
        write_json(folder / "config.json", config.model_dump())
        write_json(folder / "scaler.json", asdict(scaler))
        atomic_json(folder / "result.json", report)
        print(
            f"{asset} {candidate['name']} seed={seed} fold={fold['name']} steps={steps} excess={report['excess_return']:.2%}",
            flush=True,
        )
        return report


def paired_bootstrap(rows, samples, block, seed=2026):
    # Seeds share the same market path. Resample the SAME blocks for all seeds
    # within a fold; never pretend seeds are independent market observations.
    rng = np.random.default_rng(seed)
    distributions = []
    for fold in sorted({r["fold"] for r in rows}):
        group = [r for r in rows if r["fold"] == fold]
        arrays = []
        for row in group:
            path = Path(row["directory"]) / "score"
            a = pd.read_csv(path / "agent_equity.csv")
            b = pd.read_csv(path / "buy_hold_equity.csv")
            if not a.timestamp.equals(b.timestamp):
                raise ValueError("Benchmark timestamps differ")
            arrays.append(
                (a.equity.pct_change().iloc[1:].to_numpy(), b.equity.pct_change().iloc[1:].to_numpy())
            )
        n = len(arrays[0][0])
        count = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(samples, count))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(samples, -1)[:, :n]
        per_seed = [np.prod(1 + a[indices], axis=1) - np.prod(1 + b[indices], axis=1) for a, b in arrays]
        distributions.append(np.mean(per_seed, axis=0))
    values = np.mean(distributions, axis=0)
    return {
        "paired_block_excess_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "block_bars": block,
        "samples": samples,
        "note": "Descriptive bootstrap; not adjusted for strategy selection or regime uncertainty.",
    }


def summarize(rows, spec, output):
    summaries = []
    for asset in spec["assets"]:
        names = sorted({r["candidate"] for r in rows if r["asset"] == asset})
        for name in names:
            group = [r for r in rows if r["asset"] == asset and r["candidate"] == name]
            entry = {
                "asset": asset,
                "candidate": name,
                "budget": max(r["budget"] for r in group),
                "fits": len(group),
                **aggregate(group, spec["max_drawdown"], spec["min_positive_fold_fraction"]),
            }
            entry["mean_return"] = float(np.mean([r["total_return"] for r in group]))
            entry["mean_exposure"] = float(np.mean([r["mean_gross_exposure"] for r in group]))
            summaries.append(entry)
    board = pd.DataFrame(summaries).sort_values(
        ["asset", "risk_feasible", "score"], ascending=[True, False, False]
    )
    board.to_csv(output / "leaderboard.csv", index=False)
    atomic_json(output / "development_results.json", rows)
    return summaries


def run(spec_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "coordinator.lock"), timeout=0):
        if (output / "final_results.json").exists():
            return output
        spec = read_spec(spec_path)
        preregistration = output / "preregistration.json"
        if preregistration.exists():
            registered = json.loads(preregistration.read_text())
            if registered["spec_sha256"] != digest(Path(spec_path)):
                raise ValueError("Configuration differs from the preregistered protocol")
        manifest = prepare(spec, output)
        if (output / "selection.json").exists():
            finalize_focused(spec, manifest, output)
            return output
        config = Config.model_validate(spec["config"])
        ppo = [dict(c, strategy="ppo") for c in spec["ppo_candidates"]]
        rules = spec["rule_candidates"]
        active = ppo
        latest = {}
        promotions = []
        for stage, budget in enumerate(spec["stage_budgets"]):
            candidates = rules + active if stage == 0 else active
            if not candidates:
                break
            jobs = []
            for candidate in candidates:
                for asset in spec["assets"]:
                    for seed in spec["seeds"]:
                        for fold in spec["folds"]:
                            key = (asset, candidate["name"], seed, fold["name"])
                            job = {
                                "output": str(output),
                                "spec": spec,
                                "candidate": candidate,
                                "asset": asset,
                                "seed": seed,
                                "fold": fold,
                                "budget": budget,
                            }
                            if key in latest:
                                job["previous"] = latest[key]["directory"]
                            jobs.append(job)
            atomic_json(
                output / "status.json",
                {
                    "state": "training",
                    "stage": stage + 1,
                    "budget": budget,
                    "jobs": len(jobs),
                    "note": "Only inner validation controls budget promotion; holdout remains sealed.",
                },
            )
            results = parallel_jobs(fit_job, jobs, config.compute)
            for row in results:
                latest[(row["asset"], row["candidate"], row["seed"], row["fold"])] = row
            summarize(list(latest.values()), spec, output)
            promoted = []
            for candidate in active:
                group = [r for r in results if r["candidate"] == candidate["name"]]
                fraction = float(np.mean([r["learning_gain"] >= spec["promotion_min_gain"] for r in group]))
                advance = fraction >= spec["promotion_fraction"]
                promotions.append(
                    {
                        "candidate": candidate["name"],
                        "budget": budget,
                        "fraction_improving": fraction,
                        "advance": advance,
                        "criterion": "Inner-validation best score improves >=0.5 percentage points between first and second half of this cumulative budget in >=50% of fits.",
                    }
                )
                if advance:
                    promoted.append(candidate)
            write_json(output / "promotion.json", promotions)
            active = promoted
        rows = list(latest.values())
        summaries = summarize(rows, spec, output)
        selected = {}
        for asset in spec["assets"]:
            choices = [s for s in summaries if s["asset"] == asset]
            winner = max(choices, key=lambda s: (s["risk_feasible"], s["score"]))
            winner.update(
                paired_bootstrap(
                    [r for r in rows if r["asset"] == asset and r["candidate"] == winner["candidate"]],
                    spec["bootstrap_samples"],
                    spec["bootstrap_block_bars"],
                )
            )
            candidate = next(c for c in ppo + rules if c["name"] == winner["candidate"])
            selected[asset] = {
                "development": winner,
                "candidate": candidate,
                "seed": spec["seeds"][0],
                "budget": winner["budget"],
                "deployment_approved": False,
                "interpretation": "Selected diagnostic candidate; eligibility is reported, never assumed.",
            }
        write_json(
            output / "selection.json",
            {
                "frozen_at": datetime.now(UTC).isoformat(),
                "assets": selected,
                "protocol_signature": manifest["signature"],
            },
        )
        finalize_focused(spec, manifest, output)
        return output


def finalist_job(payload, device, threads):
    spec, manifest, output, asset, chosen = payload
    output = Path(output)
    configure_threads(threads)
    directory = output / "finalists" / asset
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return asset, json.loads(result_path.read_text())
    raw = {}
    for name, item in manifest["datasets"].items():
        if digest(Path(item["source_path"])) != item["source_sha256"]:
            raise ValueError("Source changed after selection")
        meta = json.loads(Path(item["source_path"]).with_suffix(".metadata.json").read_text())
        raw[name] = load_bars(item["source_path"], Market.model_validate(meta["market"]))
    candidate = chosen["candidate"]
    config = config_for(spec, asset, candidate, chosen["seed"], chosen["budget"])
    peer = next(name for name in spec["assets"] if name != asset)
    frame = build_frame(raw[asset], raw[peer], config, candidate.get("feature_set", "cross"))
    # Reserve the final two development years for checkpoint selection.
    tune_start = utc(spec["holdout_start"]) - pd.DateOffset(years=2)
    fold = {
        "training_end": tune_start.isoformat(),
        "tuning_end": spec["holdout_start"],
        "scoring_end": spec["holdout_end"],
    }
    train, tune, holdout, start = partitions(frame, fold, config)
    scaler = Scaler.fit(train)
    configure_threads(config.compute.threads_per_worker)
    if candidate["strategy"] == "ppo":
        model, actual = fit_ppo(train, tune, scaler, config, directory, "cpu")
        model_file = "best_model.zip"
    else:
        model = fit_etf_policy(candidate, train, chosen["seed"])
        joblib.dump(model, directory / "policy.joblib")
        actual = 0
        model_file = "policy.joblib"
    tuning = rollout(tune, scaler, config, model)[0]
    fraction = round(tuning["mean_gross_exposure"] * 4) / 4
    write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "upstream_commit": UPSTREAM_COMMIT,
            "config": config.model_dump(),
            "scaler": asdict(scaler),
            "model_file": model_file,
            "model_sha256": digest(directory / model_file),
            "source_sha256": manifest["datasets"][asset]["source_sha256"],
            "source": manifest["datasets"][asset]["source_path"],
            "train_end": train.timestamp.iloc[-1].isoformat(),
            "selection_end": spec["holdout_end"],
            "synthetic": any(d["synthetic"] for d in manifest["datasets"].values()),
            "feature_builder": "etf",
            "feature_set": candidate.get("feature_set", "cross"),
            "peer_symbol": peer,
            "actual_timesteps": actual,
        },
    )
    metrics, _ = evaluate_policy(holdout, scaler, config, model, start, directory / "holdout", fraction)
    stress = {}
    for name, multiple, delay in (
        ("double_costs", 2, 0),
        ("one_session_delay", 1, 1),
        ("double_costs_and_delay", 2, 1),
    ):
        values = config.model_dump()
        values["execution"].update(
            commission=config.execution.commission * multiple,
            slippage_bps=config.execution.slippage_bps * multiple,
            delay_bars=delay,
        )
        stressed = Config.model_validate(values)
        a = rollout(holdout, scaler, stressed, model, start)[0]
        b = rollout(holdout, scaler, stressed, "buy_hold", start)[0]
        stress[name] = {"agent": a, "buy_hold": b, "excess_return": a["total_return"] - b["total_return"]}
    excess = metrics["agent"]["total_return"] - metrics["buy_hold"]["total_return"]
    result = {
        "metrics": metrics,
        "excess_return": excess,
        "risk_feasible": metrics["agent"]["max_drawdown"] <= spec["max_drawdown"],
        "stress_tests": stress,
        "development": chosen["development"],
        "matched_fraction_from_tuning": fraction,
        "historical_holdout": [spec["holdout_start"], spec["holdout_end"]],
        "prospective_status": "Pending future candles after a policy freeze; historical simulation is not prospective evidence.",
    }
    atomic_json(result_path, result)
    return asset, result


def finalize_focused(spec, manifest, output):
    selection = json.loads((output / "selection.json").read_text())
    atomic_json(
        output / "status.json",
        {
            "state": "final_evaluation",
            "note": "Selection frozen; opening reserved 2014–2015 historical period.",
        },
    )
    jobs = [(spec, manifest, str(output), asset, chosen) for asset, chosen in selection["assets"].items()]
    results = dict(parallel_jobs(finalist_job, jobs, Config.model_validate(spec["config"]).compute))
    atomic_json(
        output / "final_results.json",
        {
            "selection": selection,
            "assets": results,
            "note": "No parameters were retuned on the reserved historical holdout. Historical bootstrap intervals are descriptive, not selection-adjusted guarantees.",
        },
    )
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "holdout_evaluated": True,
            "prospective_status": "awaiting future observations",
        },
    )
