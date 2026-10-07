"""Registered, bounded portfolio research. Historical evidence remains exploratory."""

import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from filelock import FileLock
from sklearn.ensemble import HistGradientBoostingRegressor

from .artifacts import digest
from .compute import configure_threads, parallel_jobs, plan_resources
from .config import Compute, Market
from .data import load_bars
from .experiments import atomic_json
from .market_data import expected_grid
from .portfolio import Ledger, make_panel, momentum_weights, performance, rebalance_days, weights_from_scores


def engine_hash():
    root = Path(__file__).parent
    return hashlib.sha256(
        "".join(
            digest(root / name)
            for name in (
                "portfolio.py",
                "portfolio_env.py",
                "rotation.py",
                "rotation_data.py",
                "rotation_rl.py",
            )
        ).encode()
    ).hexdigest()


def candidates(spec):
    rows = [
        {
            "name": f"momentum_{lb}_{freq}_{weight}",
            "family": "momentum",
            "lookback": lb,
            "frequency": freq,
            "weighting": weight,
        }
        for lb, freq, weight in itertools.product(spec["lookbacks"], spec["frequencies"], spec["weightings"])
    ]
    return rows + [
        {"name": f"supervised_{freq}", "family": "supervised", "frequency": freq, "weighting": "equal"}
        for freq in spec["supervised_frequencies"]
    ]


def load_panel(spec):
    root = Path(spec["data_directory"])
    frames, provenance = {}, {}
    cutoff = pd.Timestamp(spec["download_end"], tz="UTC")
    for symbol in [*spec["universe"], spec["benchmark"]]:
        path = root / f"{symbol}.parquet"
        meta = json.loads(path.with_suffix(".metadata.json").read_text())
        if (
            meta["sha256"] != digest(path)
            or meta["source_ticker"] != symbol
            or meta["market"]["price_adjustment"] != "split_adjusted"
        ):
            raise ValueError(f"Source identity mismatch: {symbol}")
        market = Market.model_validate(meta["market"]).model_copy(update={"require_regular_bars": False})
        frame = load_bars(path, market)
        expected = expected_grid(frame.timestamp.iloc[0], frame.timestamp.iloc[-1], market)
        gaps = expected.timestamp[~expected.timestamp.isin(frame.timestamp)]
        if len(gaps):
            # A recent provider gap trims *all* portfolios before that session.
            # An older gap is a hard error; prices are never synthesized.
            if gaps.iloc[0] < frame.timestamp.iloc[-1] - pd.Timedelta(days=7):
                raise ValueError(f"Interior missing sessions: {symbol}")
            cutoff = min(cutoff, gaps.iloc[0])
        frames[symbol] = frame
        provenance[symbol] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "missing_sessions": [str(t) for t in gaps],
        }
    frames = {s: f[f.bar_end < cutoff].reset_index(drop=True) for s, f in frames.items()}
    rates = root / "cash_yield.parquet"
    metadata = json.loads(rates.with_suffix(".metadata.json").read_text())
    if metadata["sha256"] != digest(rates) or metadata["symbol"] != "^IRX":
        raise ValueError("Cash yield source changed")
    provenance["cash_yield"] = {"path": str(rates.resolve()), "sha256": digest(rates)}
    panel = make_panel(frames, pd.read_parquet(rates))
    return panel, {
        "sources": provenance,
        "common_start": str(panel.timestamp[0]),
        "common_end": str(panel.bar_end[-1]),
        "cutoff_exclusive": str(cutoff),
        "rows": len(panel.timestamp),
    }


def training_examples(panel, investable, horizon):
    """Labels are next-open entry to 21 sessions later; no entry-day dividend."""
    end = len(panel.timestamp) - horizon - 1
    times = np.arange(252, end)
    entry, exit_ = times + 1, times + 1 + horizon
    cumulative = np.cumsum(panel.dividend, axis=0)
    labels = (panel.opening[exit_] + cumulative[exit_] - cumulative[entry]) / panel.opening[entry] - 1
    labels = labels[:, :investable]
    labels -= labels.mean(axis=1, keepdims=True)
    x = panel.features[times, :investable]
    if not len(times) or not np.isfinite(x).all():
        raise ValueError("Insufficient finite training history")
    return (
        x.reshape(-1, x.shape[-1]),
        labels.ravel(),
        {
            "feature_end": str(panel.bar_end[times[-1]]),
            "label_end": str(panel.timestamp[exit_[-1]]),
            "examples": len(labels.ravel()),
            "horizon": horizon,
        },
    )


def fit_ranker(panel, fold, spec):
    train = panel.prefix(fold["start"])
    x, y, info = training_examples(train, len(spec["universe"]), spec["label_horizon"])
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        learning_rate=0.03,
        l2_regularization=10,
        early_stopping=False,
        random_state=spec["supervised_seed"],
    )
    model.fit(x, y)
    return model, info


def simulate(panel, fold, candidate, spec, model=None, cost_multiplier=1, delay=0, save=None):
    indices = np.flatnonzero(
        (panel.timestamp >= pd.Timestamp(fold["start"], tz="UTC"))
        & (panel.bar_end < pd.Timestamp(fold["end"], tz="UTC"))
    )
    if not len(indices) or indices[0] <= 252 + delay:
        raise ValueError("Insufficient evaluation history")
    start, end = int(indices[0]), int(indices[-1])
    ledger = Ledger(
        panel,
        start,
        end,
        spec["commission"] * cost_multiplier,
        spec["slippage"] * cost_multiplier,
        spec["initial_cash"],
    )
    count, n = len(spec["universe"]), len(panel.symbols)
    family = candidate["family"]
    schedule = rebalance_days(panel, candidate.get("frequency", "monthly"))
    predictions = None
    if family == "supervised":
        features = panel.features[start - 1 - delay : end, :count]
        predictions = model.predict(features.reshape(-1, features.shape[-1])).reshape(len(features), count)
    decisions = []
    for t in range(start, end + 1):
        decision = t - 1 - delay
        weights = None
        first = t == start + delay
        active = t >= start + delay
        if active and (first or (family in ("momentum", "supervised") and schedule[decision])):
            if family == "momentum":
                weights = momentum_weights(
                    panel,
                    decision,
                    candidate["lookback"],
                    spec["skip_recent"],
                    spec["top_k"],
                    candidate["weighting"],
                    count,
                )
            elif family == "supervised":
                scores = predictions[decision - (start - 1 - delay)]
                weights = np.pad(
                    weights_from_scores(scores, panel.volatility[decision, :count], spec["top_k"]),
                    (0, n - count),
                )
            elif family == "spy":
                weights = np.zeros(n)
                weights[panel.symbols.index(spec["benchmark"])] = 1
            elif family == "passive":
                weights = np.r_[np.full(count, 1 / count), np.zeros(n - count)]
            elif family == "cash":
                weights = np.zeros(n)
            else:
                raise ValueError(f"Unknown family {family}")
            decisions.append(
                {
                    "decided_at": str(panel.bar_end[decision]),
                    "execute_at": str(panel.timestamp[t]),
                    **dict(zip(panel.symbols, map(float, weights))),
                }
            )
        ledger.step(weights, drip=family in ("spy", "passive"))
    if save:
        save = Path(save)
        save.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ledger.history).to_csv(save / "equity.csv", index=False)
        pd.DataFrame(ledger.fills).to_csv(save / "fills.csv", index=False)
        pd.DataFrame(decisions).to_csv(save / "decisions.csv", index=False)
    return ledger


def evaluate_job(payload, device, threads):
    configure_threads(threads)
    spec, candidate, fold, output, expected_data = payload
    panel, identity = load_panel(spec)
    if identity != expected_data:
        raise ValueError("Source snapshot changed during study")
    panel = panel.prefix(fold["end"])
    destination = Path(output) / candidate["name"] / fold["name"]
    model, training = None, None
    if candidate["family"] == "supervised":
        import joblib

        model, training = fit_ranker(panel, fold, spec)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination / "model.joblib")
    ledger = simulate(panel, fold, candidate, spec, model, save=destination)
    metrics = performance(ledger)
    history = pd.DataFrame(ledger.history)
    result = {
        "candidate": candidate,
        "fold": fold,
        "metrics": metrics,
        "training": training,
        "log_returns": np.diff(np.log(history.equity)).tolist(),
        "timestamps": history.timestamp.iloc[1:].astype(str).tolist(),
    }
    atomic_json(destination / "result.json", result)
    return result


def family_bootstrap(candidate_rows, baseline_rows, spec):
    """Joint max-mean bootstrap, 63-session blocks within each regime.

    This exploratory correction covers this registered family only; it cannot
    undo earlier experiments, public knowledge of history or universe selection.
    """
    names = sorted(candidate_rows)
    folds = [f["name"] for f in spec["folds"]]
    matrices = []
    for fold in folds:
        b = baseline_rows[fold]
        rows = [candidate_rows[n][fold] for n in names]
        if any(r["timestamps"] != b["timestamps"] for r in rows):
            raise ValueError("Bootstrap requires aligned returns")
        matrices.append(np.column_stack([np.array(r["log_returns"]) - b["log_returns"] for r in rows]))
    observed = np.concatenate(matrices).mean(axis=0)
    rng = np.random.default_rng(spec["seed"])
    means = np.empty((spec["bootstrap_samples"], len(names)))
    null_max = np.empty(spec["bootstrap_samples"])
    for i in range(spec["bootstrap_samples"]):
        samples, null = [], []
        for matrix in matrices:
            n = len(matrix)
            starts = rng.integers(0, n, size=int(np.ceil(n / spec["bootstrap_block"])))
            ix = ((starts[:, None] + np.arange(spec["bootstrap_block"])) % n).ravel()[:n]
            samples.append(matrix[ix])
            null.append(matrix[ix] - matrix.mean(axis=0))
        means[i] = np.concatenate(samples).mean(axis=0)
        null_max[i] = np.concatenate(null).mean(axis=0).max()
    return {
        name: {
            "annual_log_excess": float(observed[j] * 252),
            "pointwise_95_interval": (np.quantile(means[:, j], [0.025, 0.975]) * 252).tolist(),
            "family_adjusted_p": float((1 + (null_max >= observed[j]).sum()) / (len(null_max) + 1)),
        }
        for j, name in enumerate(names)
    }


def summarize(rows, baseline, spec):
    indexed = {c["name"]: {} for c in candidates(spec)}
    by_baseline = {family: {} for family in ("spy", "passive", "cash")}
    for row in rows:
        indexed[row["candidate"]["name"]][row["fold"]["name"]] = row
    for row in baseline:
        by_baseline[row["candidate"]["family"]][row["fold"]["name"]] = row
    corrected = family_bootstrap(indexed, by_baseline["spy"], spec)
    summaries = []
    gate = spec["gate"]
    for name, values in indexed.items():
        excess, passive_excess, risk = [], [], []
        for fold, row in values.items():
            m, b, passive = (
                row["metrics"],
                by_baseline["spy"][fold]["metrics"],
                by_baseline["passive"][fold]["metrics"],
            )
            excess.append(m["cagr"] - b["cagr"])
            passive_excess.append(m["cagr"] - passive["cagr"])
            risk.append(
                m["max_drawdown"] <= b["max_drawdown"] + gate["drawdown_tolerance"]
                and m["volatility"] <= b["volatility"] * gate["volatility_ratio_limit"]
            )
        positive = float(np.mean(np.array(excess) > 0))
        passes = bool(
            np.mean(excess) > 0
            and np.mean(passive_excess) > 0
            and all(risk)
            and positive >= gate["positive_fold_fraction"]
            and corrected[name]["family_adjusted_p"] <= gate["adjusted_p_limit"]
        )
        summaries.append(
            {
                "candidate": name,
                "mean_cagr_excess_spy": float(np.mean(excess)),
                "mean_cagr_excess_passive": float(np.mean(passive_excess)),
                "positive_fold_fraction": positive,
                "risk_feasible_all_folds": all(risk),
                "passes_rl_gate": passes,
                **corrected[name],
            }
        )
    return sorted(summaries, key=lambda r: (r["passes_rl_gate"], r["mean_cagr_excess_spy"]), reverse=True)


def stress_job(payload, device, threads):
    configure_threads(threads)
    spec, candidate, fold, output, multiplier, delay, label, expected_data = payload
    panel, identity = load_panel(spec)
    if identity != expected_data:
        raise ValueError("Stress source snapshot changed")
    model = fit_ranker(panel, fold, spec)[0] if candidate["family"] == "supervised" else None
    rows = {}
    for c in (candidate, {"name": "spy", "family": "spy"}, {"name": "passive", "family": "passive"}):
        ledger = simulate(panel, fold, c, spec, model, multiplier, delay, Path(output) / label / c["name"])
        rows[c["name"]] = performance(ledger)
    return {"scenario": label, "cost_multiplier": multiplier, "delay_sessions": delay, "metrics": rows}


def _run(spec_path, output):
    output = Path(output).resolve()
    spec = yaml.safe_load(Path(spec_path).read_text())
    registration_path = output / "preregistration.json"
    if not registration_path.exists():
        atomic_json(
            registration_path,
            {
                "registered_at": datetime.now(UTC).isoformat(),
                "spec": spec,
                "spec_sha256": digest(Path(spec_path)),
                "status": "Retrospective research",
            },
        )
    registration = json.loads(registration_path.read_text())
    if registration["spec"] != spec or registration["spec_sha256"] != digest(Path(spec_path)):
        raise ValueError("Registered specification changed; use a separately registered study")
    _, identity = load_panel(spec)
    protocol = {
        "registration_sha256": digest(registration_path),
        "engine_sha256": engine_hash(),
        "data": identity,
        "candidates": candidates(spec),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Frozen engine or sources changed; do not overwrite research")
    if (output / "results.json").exists():
        return output
    atomic_json(protocol_path, protocol)
    compute = Compute.model_validate(spec["compute"])
    baselines = [{"name": family, "family": family} for family in ("spy", "passive", "cash")]
    jobs = [
        (spec, candidate, fold, str(output / "development"), identity)
        for candidate in candidates(spec) + baselines
        for fold in spec["folds"]
    ]
    plan = plan_resources(compute, len(jobs))
    atomic_json(output / "compute.json", plan.to_dict())
    atomic_json(
        output / "trial_registry.json",
        {
            "family_size": len(candidates(spec)),
            "development_jobs": len(jobs),
            "trials": [{"candidate": c, "fold": f} for _, c, f, _, _ in jobs],
            "prior_experiments": ["research-daily", "research-intraday", "research-archive", "focused-etf"],
            "correction_scope": "Current ten-candidate family only. Earlier studies remain additional researcher discretion.",
        },
    )
    atomic_json(
        output / "status.json", {"stage": "development", "jobs": len(jobs), "workers": plan.worker_count}
    )
    all_rows = parallel_jobs(evaluate_job, jobs, compute)
    rows = [r for r in all_rows if r["candidate"]["family"] not in ("spy", "passive", "cash")]
    baseline = [r for r in all_rows if r not in rows]
    summaries = summarize(rows, baseline, spec)
    pd.DataFrame(summaries).to_csv(output / "leaderboard.csv", index=False)
    winner = next(c for c in candidates(spec) if c["name"] == summaries[0]["candidate"])
    selection = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "candidate": winner,
        "passes_rl_gate": summaries[0]["passes_rl_gate"],
        "deployment_approved": False,
        "note": "Diagnostic finalist if gate fails. Selection uses development folds only.",
    }
    atomic_json(output / "selection.json", selection)
    atomic_json(
        output / "status.json",
        {"stage": "confirmation_and_stress", "jobs": len(jobs), "workers": plan.worker_count},
    )
    # All ten fixed candidates are retained in the recent-period diagnostic.
    # Their relative results do not reselect the frozen finalist.
    confirm_jobs = [
        (spec, c, spec["confirmation"], str(output / "confirmation"), identity)
        for c in candidates(spec) + baselines
    ]
    confirmation = parallel_jobs(evaluate_job, confirm_jobs, compute)
    stresses = parallel_jobs(
        stress_job,
        [
            (spec, winner, spec["confirmation"], str(output / "stress"), mult, delay, name, identity)
            for mult, delay, name in (
                (2, 0, "double_costs"),
                (1, 1, "one_session_delay"),
                (2, 1, "double_costs_and_delay"),
            )
        ],
        compute,
    )
    rl = {
        "status": "not_started",
        "reason": "No simple candidate passed the registered evidence and risk gate.",
    }
    if selection["passes_rl_gate"]:
        from .rotation_rl import run_rl

        rl = run_rl(spec, output, winner, rows, identity)
    result = {
        "completed_at": datetime.now(UTC).isoformat(),
        "development": all_rows,
        "leaderboard": summaries,
        "selection": selection,
        "confirmation": confirmation,
        "stress": stresses,
        "rl": rl,
        "historical_evidence": "exploratory_previously_seen_regimes",
        "deployment_approved": False,
        "prospective_sessions": 0,
    }
    if engine_hash() != protocol["engine_sha256"]:
        raise ValueError("Engine changed during execution")
    atomic_json(output / "results.json", result)
    atomic_json(
        output / "status.json",
        {
            "stage": "complete",
            "development_jobs": len(jobs),
            "confirmation_jobs": len(confirm_jobs),
            "workers": plan.worker_count,
        },
    )
    from .rotation_report import write_report

    write_report(output)
    return output


def run(spec_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "study.lock"), timeout=1):
        return _run(spec_path, output)
