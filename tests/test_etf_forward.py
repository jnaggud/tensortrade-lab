from dataclasses import asdict

import joblib
import pandas as pd
import pytest

from tensortrade_lab import etf_forward
from tensortrade_lab.artifacts import UPSTREAM_COMMIT, digest, write_json
from tensortrade_lab.config import Config
from tensortrade_lab.data import synthetic_bars, validate_bars
from tensortrade_lab.etf_strategies import ETFPolicy
from tensortrade_lab.features import Scaler
from tensortrade_lab.focused import build_frame
from tensortrade_lab.market_data import expected_grid


def test_forward_records_before_execution_and_never_backfills(tmp_path, monkeypatch):
    config = Config.model_validate(
        {
            "market": {
                "calendar": "XNYS",
                "bar_minutes": 1440,
                "periods_per_year": 252,
                "price_adjustment": "split_adjusted",
            },
            "features": {"window": 5, "reference_minutes": 1440},
            "risk": {"max_position": 1.0, "max_drawdown": 1.0, "stop_loss": 1.0, "take_profit": 100.0},
        }
    )
    grid = expected_grid("2010-01-01", "2013-12-31", config.market)
    data = {}
    sources = {}
    for seed, asset in enumerate(("SPY", "QQQ")):
        raw = synthetic_bars(len(grid), seed=seed)
        raw["timestamp"] = grid.timestamp
        data[asset] = validate_bars(raw, config.market)
        sources[asset] = tmp_path / f"{asset}.parquet"
        data[asset].iloc[:500].to_parquet(sources[asset], index=False)
    for asset, peer in (("SPY", "QQQ"), ("QQQ", "SPY")):
        directory = tmp_path / "finalists" / asset
        directory.mkdir(parents=True)
        c = config.model_copy(update={"market": config.market.model_copy(update={"symbol": asset})})
        frame = build_frame(data[asset].iloc[:450], data[peer].iloc[:450], c, "cross")
        scaler = Scaler.fit(frame)
        joblib.dump(ETFPolicy("fixed", fraction=1), directory / "policy.joblib")
        write_json(
            directory / "manifest.json",
            {
                "schema_version": 1,
                "upstream_commit": UPSTREAM_COMMIT,
                "config": c.model_dump(),
                "scaler": asdict(scaler),
                "model_file": "policy.joblib",
                "model_sha256": digest(directory / "policy.joblib"),
                "feature_builder": "etf",
                "feature_set": "cross",
                "peer_symbol": peer,
            },
        )
    write_json(tmp_path / "final_results.json", {})
    clock = [data["SPY"].bar_end.iloc[499] + pd.Timedelta(minutes=5)]
    monkeypatch.setattr(etf_forward, "now_utc", lambda: clock[0])
    first = etf_forward.poll(tmp_path, sources)
    assert first["SPY"]["prospective_sessions"] == 0
    assert first["SPY"]["signals"][0]["action"] == 5
    repeated = etf_forward.poll(tmp_path, sources)
    assert repeated["SPY"]["signals"] == first["SPY"]["signals"]
    for asset, source in sources.items():
        data[asset].iloc[:501].to_parquet(source, index=False)
    clock[0] = data["SPY"].bar_end.iloc[500] + pd.Timedelta(minutes=5)
    second = etf_forward.poll(tmp_path, sources)
    assert second["SPY"]["prospective_sessions"] == 1
    assert second["SPY"]["comparison"]["excess_return"] == pytest.approx(0)
    assert second["SPY"]["account"]["units"] > 0
    for asset, source in sources.items():
        data[asset].iloc[:504].to_parquet(source, index=False)
    clock[0] = data["SPY"].bar_end.iloc[503] + pd.Timedelta(minutes=5)
    later = etf_forward.poll(tmp_path, sources)
    assert later["SPY"]["prospective_sessions"] == 4
    assert later["SPY"]["missing_decision_sessions"] == 2
    assert len(later["SPY"]["signals"]) == 3
    changed = data["SPY"].iloc[:504].copy()
    changed.loc[10, "volume"] *= 2
    changed.to_parquet(sources["SPY"], index=False)
    with pytest.raises(ValueError, match="append-only"):
        etf_forward.poll(tmp_path, sources)


def test_late_decisions_are_rejected(frame, scaler, config):
    frame = frame.assign(bar_end=frame.timestamp + pd.Timedelta(minutes=config.market.bar_minutes))
    signals = [
        {
            "decision_at": frame.bar_end.iloc[config.window].isoformat(),
            "recorded_at": frame.timestamp.iloc[config.window + 2].isoformat(),
            "action": 5,
        }
    ]
    with pytest.raises(ValueError, match="execution candle"):
        etf_forward.reconstruct(frame, scaler, config, signals)


def test_generic_paper_rejects_etf_without_peer():
    from tensortrade_lab.paper import future_frame

    with pytest.raises(ValueError, match="both assets"):
        future_frame("unused.csv", Config(), {"feature_builder": "etf"})
