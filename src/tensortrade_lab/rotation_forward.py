"""Append-only, manually polled paper decisions. No broker or background scheduler."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .artifacts import digest
from .config import Market
from .experiments import atomic_json
from .market_data import expected_grid
from .portfolio import Ledger, momentum_weights, performance, rebalance_days, weights_from_scores
from .rotation import engine_hash, fit_ranker, load_panel


def refresh(study):
    """Download into a new directory; never mutate registered research prices."""
    import yaml

    from .rotation_data import acquire

    study = Path(study).resolve()
    now = pd.Timestamp.now(tz="UTC")
    spec = json.loads((study / "preregistration.json").read_text())["spec"]
    root = study / "prospective" / "snapshots" / now.strftime("%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True, exist_ok=False)
    spec["data_directory"] = str(root / "data")
    spec["download_end"] = str((now + pd.Timedelta(days=1)).date())
    spec_path = root / "download_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    return acquire(spec_path)


def panel_digest(panel, count=None):
    count = len(panel.timestamp) if count is None else count
    h = hashlib.sha256(json.dumps(panel.symbols).encode())
    h.update(panel.timestamp.asi8[:count].tobytes())
    for name in ("opening", "close", "dividend", "cash_rate"):
        h.update(np.ascontiguousarray(getattr(panel, name)[:count], dtype="float64").tobytes())
    return h.hexdigest()


def next_open(panel):
    start = panel.timestamp[-1] + pd.Timedelta(days=1)
    grid = expected_grid(start, start + pd.Timedelta(days=10), Market(calendar="XNYS", bar_minutes=1440))
    return grid.timestamp.iloc[0]


def register(study, now):
    study = Path(study)
    root = study / "prospective"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "registration.json"
    if path.exists():
        return json.loads(path.read_text())
    if not (study / "results.json").exists():
        raise ValueError("Complete the historical study before prospective registration")
    protocol = json.loads((study / "protocol.json").read_text())
    selection = json.loads((study / "selection.json").read_text())
    spec = json.loads((study / "preregistration.json").read_text())["spec"]
    if engine_hash() != protocol["engine_sha256"]:
        raise ValueError("Research engine changed before prospective registration")
    panel, _ = load_panel(spec)
    model_sha, training = None, None
    if selection["candidate"]["family"] == "supervised":
        import joblib

        # Refit the frozen model recipe once using only available labels.
        end = str((panel.bar_end[-1] + pd.Timedelta(days=1)).date())
        model, training = fit_ranker(panel, {"start": end}, spec)
        joblib.dump(model, root / "model.joblib")
        model_sha = digest(root / "model.joblib")
    registration = {
        "registered_at": now.isoformat(),
        "candidate": selection["candidate"],
        "spec": spec,
        "selection_sha256": digest(study / "selection.json"),
        "engine_sha256": engine_hash(),
        "forward_engine_sha256": digest(Path(__file__)),
        "model_sha256": model_sha,
        "training": training,
        "known_rows": len(panel.timestamp),
        "known_history_sha256": panel_digest(panel),
        "known_end": str(panel.bar_end[-1]),
        "minimum_sessions_before_review": 126,
        "deployment_approved": False,
        "note": "Diagnostic paper strategy if historical gate failed. No backfilled decisions or live orders.",
    }
    atomic_json(path, registration)
    atomic_json(root / "decisions.json", [])
    return registration


def poll(study, data_directory=None, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tz is None:
        raise ValueError("Paper clock must be timezone-aware")
    study = Path(study)
    root = study / "prospective"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "poll.lock"), timeout=1):
        registration = register(study, now)
        if now < pd.Timestamp(registration["registered_at"]):
            raise ValueError("Paper clock precedes registration")
        if (
            engine_hash() != registration["engine_sha256"]
            or digest(Path(__file__)) != registration["forward_engine_sha256"]
            or digest(study / "selection.json") != registration["selection_sha256"]
        ):
            raise ValueError("Frozen prospective identity changed")
        spec = dict(registration["spec"])
        if data_directory:
            spec["data_directory"] = str(data_directory)
            spec["download_end"] = str((now + pd.Timedelta(days=1)).date())
        panel, _ = load_panel(spec)
        panel = panel.prefix(now.isoformat())
        count = registration["known_rows"]
        if len(panel.timestamp) < count or panel_digest(panel, count) != registration["known_history_sha256"]:
            raise ValueError("Known source history changed; investigate vendor revisions before continuing")
        state_path = root / "observed_prefix.json"
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if (
                len(panel.timestamp) < previous["rows"]
                or panel_digest(panel, previous["rows"]) != previous["sha256"]
            ):
                raise ValueError("Previously observed paper history changed")
        decisions_path = root / "decisions.json"
        decisions = json.loads(decisions_path.read_text())
        opening = next_open(panel)
        candidate = registration["candidate"]
        # Each poll can record only the immediately upcoming session.
        scheduled = rebalance_days(panel, candidate["frequency"])[-1]
        reason = "awaiting next completed session; previous execution window missed"
        if opening > now and panel.bar_end[-1] <= now and (scheduled or not decisions):
            if not any(d["execute_at"] == str(opening) for d in decisions):
                if candidate["family"] == "momentum":
                    weights = momentum_weights(
                        panel,
                        len(panel.timestamp) - 1,
                        candidate["lookback"],
                        spec["skip_recent"],
                        spec["top_k"],
                        candidate["weighting"],
                        len(spec["universe"]),
                    )
                else:
                    import joblib

                    if digest(root / "model.joblib") != registration["model_sha256"]:
                        raise ValueError("Frozen paper ranker changed")
                    model = joblib.load(root / "model.joblib")
                    scores = model.predict(panel.features[-1, : len(spec["universe"])])
                    weights = np.pad(
                        weights_from_scores(scores, panel.volatility[-1], spec["top_k"]),
                        (0, len(panel.symbols) - len(spec["universe"])),
                    )
                decisions.append(
                    {
                        "recorded_at": now.isoformat(),
                        "decision_at": str(panel.bar_end[-1]),
                        "execute_at": str(opening),
                        "weights": weights.tolist(),
                    }
                )
                atomic_json(decisions_path, decisions)
            reason = "decision recorded before the next open"
        elif opening > now:
            reason = "next session requires no scheduled rebalance"
        for decision in decisions:
            if not (
                pd.Timestamp(registration["registered_at"])
                <= pd.Timestamp(decision["recorded_at"])
                < pd.Timestamp(decision["execute_at"])
                and pd.Timestamp(decision["decision_at"]) <= pd.Timestamp(decision["recorded_at"])
            ):
                raise ValueError("Invalid or backfilled paper decision")
        completed = [d for d in decisions if pd.Timestamp(d["execute_at"]) <= panel.timestamp[-1]]
        metrics, observed_sessions, missed = {}, 0, 0
        if completed:
            start = int(panel.timestamp.get_loc(pd.Timestamp(completed[0]["execute_at"])))
            end = len(panel.timestamp) - 1
            observed_sessions = end - start + 1
            if end > start:
                action_map = {d["execute_at"]: d["weights"] for d in completed}
                scheduled = rebalance_days(panel, candidate["frequency"])
                ledgers = {
                    name: Ledger(
                        panel, start, end, spec["commission"], spec["slippage"], spec["initial_cash"]
                    )
                    for name in ("strategy", "spy", "passive")
                }
                for t in range(start, end + 1):
                    action = action_map.get(str(panel.timestamp[t]))
                    if t > start and scheduled[t - 1] and action is None:
                        missed += 1  # An unrecorded rebalance holds; never reconstruct it from history.
                    ledgers["strategy"].step(action, liquidate=False)
                    for name in ("spy", "passive"):
                        w = None
                        if t == start:
                            w = np.zeros(len(panel.symbols))
                            if name == "spy":
                                w[panel.symbols.index(spec["benchmark"])] = 1
                            else:
                                w[: len(spec["universe"])] = 1 / len(spec["universe"])
                        ledgers[name].step(w, drip=True, liquidate=False)
                metrics = {name: performance(ledger) for name, ledger in ledgers.items()}
                for name, ledger in ledgers.items():
                    pd.DataFrame(ledger.history).to_csv(root / f"{name}_equity.csv", index=False)
                observed_sessions = end - start + 1
        atomic_json(state_path, {"rows": len(panel.timestamp), "sha256": panel_digest(panel)})
        status = {
            "as_of": now.isoformat(),
            "status": reason,
            "last_completed_session": str(panel.bar_end[-1]),
            "next_open": str(opening),
            "recorded_decisions": len(decisions),
            "observed_sessions": observed_sessions,
            "missed_rebalances": missed,
            "metrics": metrics,
            "minimum_sessions_before_review": registration["minimum_sessions_before_review"],
            "deployment_approved": False,
            "manual_polling_required": True,
            "data_directory": str(Path(spec["data_directory"]).resolve()),
        }
        atomic_json(root / "status.json", status)
        return status
