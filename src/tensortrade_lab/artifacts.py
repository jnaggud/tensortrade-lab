import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .config import Config
from .features import Scaler

UPSTREAM_COMMIT = "d58afba23deb1fded39793203b7997c3990bb032"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def new_run(output: str | Path, prefix="run") -> Path:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    run = root / f"{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
    run.mkdir()
    return run


def save_manifest(
    run: Path, config: Config, scaler: Scaler, source: Path, partitions, frame, synthetic=False
):
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "upstream_commit": UPSTREAM_COMMIT,
        "config": config.model_dump(),
        "scaler": asdict(scaler),
        "source": str(source.resolve()),
        "source_sha256": digest(source),
        "synthetic": synthetic,
        "partitions": asdict(partitions),
        "feature_rows": len(frame),
        "train_end": frame.timestamp.iloc[partitions.train[1] - 1].isoformat(),
        "selection_end": frame.timestamp.iloc[partitions.validation[1] - 1].isoformat(),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("tensortrade", "stable-baselines3", "torch", "numpy", "pandas", "gymnasium")
        },
    }
    write_json(run / "manifest.json", manifest)
    return manifest


def load_bundle(run: str | Path):
    from stable_baselines3 import PPO

    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Unsupported model manifest")
    model_path = run / manifest.get("model_file", "best_model.zip")
    if digest(model_path) != manifest.get("model_sha256"):
        raise ValueError("Model checksum mismatch")
    config = Config.model_validate(manifest["config"])
    scaler = Scaler(**manifest["scaler"])
    # Only load models created locally or obtained from a trusted source: SB3 uses pickle.
    if manifest.get("model_file") == "policy.joblib":
        import joblib

        model = joblib.load(model_path)
    else:
        model = PPO.load(model_path, device="cpu")
    return model, scaler, config, manifest
