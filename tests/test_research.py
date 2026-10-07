import json

import numpy as np
import pandas as pd
import pytest

from tensortrade_lab.compute import plan_resources
from tensortrade_lab.config import Compute, Config, Market
from tensortrade_lab.data import synthetic_bars, validate_bars
from tensortrade_lab.environment import ResearchEnv
from tensortrade_lab.experiments import SearchSpec, candidate_config, prepare, run_candidate
from tensortrade_lab.features import Scaler, engineer
from tensortrade_lab.margin import MarginLedger, target_action
from tensortrade_lab.market_data import attach_calendar, expected_grid, resample_bars
from tensortrade_lab.sources import normalize, save_dataset


def test_short_accounting_reversal_borrow_and_dividends():
    account = MarginLedger(10000)
    account.transact(-0.75, 100, 0.001, 5)
    assert account.units * 100 / account.equity(100) == pytest.approx(-0.75)
    before = account.equity(90)
    assert before > 10000
    account.accrue_borrow(100, 3, 0.365)
    assert account.borrow_cost == pytest.approx(-account.units * 100 * 0.003)
    old_units = account.units
    account.corporate_action(2, 1, "raw")
    assert account.units == old_units * 2
    assert account.dividends == old_units * 2
    account.transact(0.5, 50, 0.001, 5)
    assert account.units * 50 / account.equity(50) == pytest.approx(0.5)
    account.transact(0, 50, 0.001, 5)
    assert account.units == 0
    assert target_action(-0.001, True) == 1


def test_exchange_holiday_early_close_and_resampling():
    market = Market(symbol="SPY", calendar="XNYS", bar_minutes=15)
    grid = expected_grid(pd.Timestamp("2025-07-02", tz="UTC"), pd.Timestamp("2025-07-07", tz="UTC"), market)
    assert not any(grid.timestamp.dt.date.astype(str) == "2025-07-04")
    july3 = grid[grid.timestamp.dt.date.astype(str) == "2025-07-03"]
    assert len(july3) == 14
    assert july3.bar_end.iloc[-1] == pd.Timestamp("2025-07-03T17:00Z")
    bars = grid.assign(open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0)
    hourly = resample_bars(bars, market, 60)
    daily = resample_bars(hourly, market.model_copy(update={"bar_minutes": 60}), 1440)
    assert daily.volume.tolist() == [26.0, 14.0, 26.0]
    with pytest.raises(ValueError, match="Missing"):
        attach_calendar(bars.drop(index=3).reset_index(drop=True), market)
    incomplete = resample_bars(
        bars.drop(index=3).reset_index(drop=True),
        market.model_copy(update={"require_regular_bars": False}),
        60,
    )
    assert len(incomplete) == len(hourly) - 1
    with pytest.raises(ValueError, match="never upsample"):
        resample_bars(bars, market, 5)


def test_multitimeframe_features_are_causal():
    config = Config.model_validate(
        {"market": {"bar_minutes": 15}, "features": {"higher_timeframes": [60, 1440]}}
    )
    bars = validate_bars(synthetic_bars(1000, bar_minutes=15), config.market)
    before = engineer(bars.iloc[:850], config)
    changed = bars.copy()
    changed.loc[850:, "close"] *= 10
    after = engineer(changed, config)
    pd.testing.assert_frame_equal(before, after.iloc[: len(before)].reset_index(drop=True))
    assert "context_60_return" in before
    assert np.isfinite(Scaler.fit(before).transform(before)).all()


def test_short_environment_delay_recall_and_margin():
    raw = {
        "features": {"window": 2},
        "risk": {
            "direction": "long_short",
            "borrow_available": True,
            "stop_loss": 1.0,
            "take_profit": 10.0,
            "max_drawdown": 1.0,
            "max_position": 1.0,
        },
        "execution": {"commission": 0.0, "slippage_bps": 0.0, "delay_bars": 1},
    }
    config = Config.model_validate(raw)
    bars = synthetic_bars(100).assign(open=100.0, high=101.0, low=99.0, close=100.0, borrow_available=True)
    env = ResearchEnv(bars, np.zeros((100, 2), dtype=np.float32), config)
    env.reset()
    env.step(9)
    assert env.state.units == 0
    env.step(9)
    assert env.state.units == -100
    env.state.bars.loc[env.state.index + 1, "borrow_available"] = False
    env.step(9)
    assert env.state.units == 0
    assert env.state.fills[-1]["reason"] == "borrow_unavailable"
    assert env.state.ledger.borrow_cost > 0
    env.close()
    raw["execution"]["delay_bars"] = 0
    env = ResearchEnv(bars, np.zeros((100, 2), dtype=np.float32), Config.model_validate(raw))
    env.reset()
    env.step(9)
    env.state.bars.loc[env.state.index + 1, ["open", "close"]] = 250.0
    env.step(0)
    assert env.state.units == 0 and env.state.equity < 0 and env.state.halted
    assert env.state.fills[-1]["reason"] == "margin_liquidation"
    env.close()


def test_cpu_budget_and_gpu_slots():
    info = {"cpu_cores": 32, "available_memory_gib": 400, "mps_available": True, "cuda_devices": 0}
    plan = plan_resources(Compute(), 100, info)
    assert plan.worker_count == 32 and plan.devices.count("mps") == 1
    plan = plan_resources(Compute(threads_per_worker=4), 100, info)
    assert plan.worker_count == 8
    plan = plan_resources(Compute(device="cpu"), 2, info)
    assert plan.devices == ["cpu", "cpu"]
    assert plan_resources(Compute(device="mps", gpu_jobs=1), 100, info).worker_count == 1


def test_import_timezone_provenance_and_sealed_development(tmp_path):
    raw = synthetic_bars(1800)
    raw["timestamp"] = raw.timestamp.dt.tz_localize(None)
    localized = normalize(raw.iloc[:100], "America/New_York")
    assert localized.timestamp.iloc[0].hour == 5
    destination = tmp_path / "bars.parquet"
    save_dataset(raw, destination, Market(), {"source": "synthetic", "synthetic": True})
    spec = SearchSpec.model_validate(
        {
            "datasets": [{"name": "demo", "path": str(destination)}],
            "timeframes": [60],
            "strategies": ["momentum"],
            "seeds": [42],
            "folds": 2,
            "config": {"features": {"window": 4}, "split": {"purge_bars": 2}},
        }
    )
    prepared = prepare(spec, tmp_path / "prepared")
    item = prepared["markets"]["demo:60"]
    development = pd.read_parquet(item["development_path"])
    assert development.bar_end.max() <= pd.Timestamp(prepared["holdout_start"])
    directory = tmp_path / "candidate"
    directory.mkdir()
    result = run_candidate(
        {
            "spec": spec.model_dump(),
            "params": {"strategy": "momentum", "direction": "long_only", "policy.lookback_minutes": 240},
            "market": item,
            "directory": str(directory),
            "device": "cpu",
            "threads": 1,
            "start": prepared["start"],
            "holdout_start": prepared["holdout_start"],
        }
    )
    assert len(result["folds"]) == 2
    assert all(pd.Timestamp(f["end"]) < pd.Timestamp(prepared["holdout_start"]) for f in result["folds"])
    metadata = json.loads(destination.with_suffix(".metadata.json").read_text())
    assert metadata["quality"]["missing_bars"] == 0
    with pytest.raises(ValueError, match="fixed assumptions"):
        SearchSpec.model_validate(
            {"datasets": [{"name": "x", "path": "x"}], "parameters": {"execution.commission": [0, 0.01]}}
        )
    config = candidate_config(
        Config(), Market().model_dump(), {"direction": "long_only", "training.gamma": 0.9}, 17
    )
    assert config.training.gamma == 0.9 and config.training.seed == 17


def test_search_resume_interrupted_trial_and_freeze(tmp_path):
    import optuna
    import yaml

    from tensortrade_lab.experiments import finalize, search

    destination = tmp_path / "source.parquet"
    save_dataset(synthetic_bars(1200), destination, Market(), {"source": "synthetic", "synthetic": True})
    spec = {
        "datasets": [{"name": "test", "path": str(destination)}],
        "timeframes": [60],
        "strategies": ["momentum"],
        "directions": ["long_only"],
        "parameters": {"policy.threshold": [0.001]},
        "trials": 1,
        "seeds": [17],
        "folds": 2,
        "config": {
            "features": {"window": 2},
            "split": {"purge_bars": 1},
            "compute": {"workers": 1, "device": "cpu"},
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    output = tmp_path / "study"
    search(spec_path, output)
    storage = f"sqlite:///{output / 'study.sqlite3'}"
    study = optuna.load_study(study_name="research", storage=storage)
    assert len(study.trials) == 1
    search(spec_path, output)
    assert len(study.trials) == 1
    trial = study.ask()
    for name, value in study.best_trial.params.items():
        trial.suggest_categorical(name, [value])
    search(spec_path, output, max_trials=2)
    assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) == 2
    assert len(json.loads((output / "interrupted.json").read_text())) == 1
    result = finalize(output)
    assert json.loads((output / "finalist/manifest.json").read_text())["synthetic"] is True
    assert result == finalize(output)
    with pytest.raises(ValueError, match="frozen"):
        search(spec_path, output, max_trials=3)
