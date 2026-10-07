import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tensortrade_lab.config import Config, Market
from tensortrade_lab.data import synthetic_bars, validate_bars
from tensortrade_lab.environment import ResearchEnv
from tensortrade_lab.features import feature_columns
from tensortrade_lab.focused import build_frame, config_for, partitions, read_spec
from tensortrade_lab.market_data import expected_grid
from tensortrade_lab.objectives import aggregate, learning_gain, selection_score
from tensortrade_lab.training import fit_ppo


@pytest.fixture(scope="module")
def etf_data():
    market = Market(
        calendar="XNYS", bar_minutes=1440, periods_per_year=252, price_adjustment="split_adjusted"
    )
    grid = expected_grid("2003-01-01", "2015-12-31", market)
    frames = []
    for seed in [17, 83]:
        bars = synthetic_bars(len(grid), seed=seed)
        bars["timestamp"] = grid.timestamp
        frames.append(validate_bars(bars, market))
    return frames


def test_etf_causality_and_paired_ablation(etf_data):
    spec = read_spec("configs/focused_etf.yaml")
    config = config_for(spec, "SPY", {}, 17, 100000)
    own, peer = etf_data
    original = build_frame(own, peer, config, "cross")
    ablated = build_frame(own, peer, config, "own")
    assert original.timestamp.equals(ablated.timestamp)
    assert not any(c.startswith("context_peer") for c in feature_columns(ablated))
    pd.testing.assert_frame_equal(original[feature_columns(ablated)], ablated[feature_columns(ablated)])
    altered = peer.copy()
    boundary = peer.timestamp.iloc[1000]
    altered.loc[1000:, ["open", "high", "low", "close"]] *= 1.8
    future = build_frame(own, altered, config, "cross")
    pd.testing.assert_frame_equal(
        original[original.timestamp < boundary], future[future.timestamp < boundary]
    )
    delayed = peer.copy()
    delayed.loc[1000, "bar_end"] += pd.Timedelta(minutes=1)
    delayed_frame = build_frame(own, delayed, config, "cross")
    assert boundary not in set(delayed_frame.timestamp)
    assert (original.peer_available_at <= original.bar_end).all()


def test_selection_requires_excess_and_risk_feasibility():
    config = Config(objective="excess_return", selection_drawdown_limit=0.3)
    benchmark = {"total_return": 0.4}
    cash = {"total_return": 0, "max_drawdown": 0}
    winner = {"total_return": 0.5, "max_drawdown": 0.29}
    risky = {"total_return": 0.8, "max_drawdown": 0.31}
    assert selection_score(winner, benchmark, config) > selection_score(cash, benchmark, config)
    assert selection_score(risky, benchmark, config) < selection_score(cash, benchmark, config)
    rows = [{"excess_return": 0.1, "max_drawdown": 0.2}, {"excess_return": -0.01, "max_drawdown": 0.1}]
    assert not aggregate(rows, 0.3, 2 / 3)["eligible"]
    assert learning_gain(
        [{"timesteps": 20000, "score": -0.1}, {"timesteps": 60000, "score": -0.09}], 100000
    ) == pytest.approx(0.01)


@pytest.mark.parametrize("margin", [False, True])
def test_fast_observer_preserves_settlement_and_rewards(frame, scaler, config, margin):
    if margin:
        config = config.model_copy(
            update={
                "market": config.market.model_copy(
                    update={"calendar": "XNYS", "price_adjustment": "split_adjusted"}
                )
            }
        )
    slow = ResearchEnv(frame, scaler.transform(frame), config)
    fast = ResearchEnv(frame, scaler.transform(frame), config, training=True)
    np.testing.assert_array_equal(slow.reset(seed=17)[0], fast.reset(seed=17)[0])
    for index in range(len(frame) - config.window):
        action = [5, 0, 3, 1, 4, 2][index % 6]
        a, b = slow.step(action), fast.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    assert slow.state.fills == fast.state.fills
    slow.close()
    fast.close()


def test_random_training_episodes_end_and_reset(frame, scaler, config):
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"random_episode_bars": 20})}
    )
    env = ResearchEnv(frame, scaler.transform(frame), config, training=True)
    starts = []
    for seed in [1, 2, 3]:
        env.reset(seed=seed)
        starts.append(env.state.index)
        for i in range(20):
            _, _, done, _, _ = env.step(5)
            assert done == (i == 19)
        assert env.state.units == 0
    assert len(set(starts)) > 1
    env.close()


def test_ppo_resume_preserves_cumulative_budget(tmp_path, frame, scaler, config):
    raw = config.model_dump()
    raw["training"].update(
        total_timesteps=64,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        eval_every=32,
        hidden_sizes=[16],
        patience=100,
    )
    raw["compute"].update(device="cpu", vector_envs=1)
    config = Config.model_validate(raw)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    train, tune = frame.iloc[:100].reset_index(drop=True), frame.iloc[100:].reset_index(drop=True)
    _, actual = fit_ppo(train, tune, scaler, config, first)
    assert actual == 64
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"total_timesteps": 128})}
    )
    _, actual = fit_ppo(train, tune, scaler, config, second, resume_from=first)
    assert actual == 128
    assert [r["timesteps"] for r in json.loads((second / "validation.json").read_text())] == [32, 64, 96, 128]
    _, actual = fit_ppo(train, tune, scaler, config, second, resume_from=second)
    assert actual == 128


def test_fold_dates_and_training_purge(etf_data):
    spec = read_spec("configs/focused_etf.yaml")
    config = config_for(spec, "SPY", {}, 17, 100000)
    frame = build_frame(*etf_data, config, "cross")
    previous = None
    for fold in spec["folds"]:
        train, tune, scoring, start = partitions(frame, fold, config)
        boundary = pd.Timestamp(fold["training_end"], tz="UTC")
        assert len(train) == sum(frame.bar_end < boundary) - 5
        assert train.bar_end.iloc[-1] < tune.timestamp.iloc[0]
        assert scoring.timestamp.iloc[start + 1] >= pd.Timestamp(fold["tuning_end"], tz="UTC")
        if previous is not None:
            assert scoring.timestamp.iloc[start + 1] > previous
        previous = scoring.bar_end.iloc[-1]
    assert previous < pd.Timestamp(spec["holdout_start"], tz="UTC")


def test_registered_spec_cannot_change(tmp_path):
    from tensortrade_lab.artifacts import digest
    from tensortrade_lab.focused import run

    source = Path("configs/focused_etf.yaml")
    (tmp_path / "preregistration.json").write_text(json.dumps({"spec_sha256": digest(source)}))
    altered = tmp_path / "altered.yaml"
    altered.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="preregistered"):
        run(altered, tmp_path)


def test_full_protocol_on_synthetic_data(tmp_path, monkeypatch, etf_data):
    import yaml

    from tensortrade_lab import focused
    from tensortrade_lab.artifacts import digest, load_bundle, write_json

    spec = read_spec("configs/focused_etf.yaml")
    spec.update(seeds=[17], stage_budgets=[64], bootstrap_samples=20)
    # The separate partition test covers all folds; this exercises complete orchestration.
    spec["folds"] = spec["folds"][:1]
    spec["ppo_candidates"] = spec["ppo_candidates"][:1]
    spec["rule_candidates"] = spec["rule_candidates"][:1]
    spec["config"]["training"].update(
        total_timesteps=64, n_steps=32, batch_size=16, eval_every=32, n_epochs=1, hidden_sizes=[16]
    )
    for asset, data in zip(spec["assets"], etf_data, strict=True):
        source = tmp_path / f"{asset}.parquet"
        data.to_parquet(source, index=False)
        spec["sources"][asset] = str(source)
        market = dict(spec["config"]["market"], symbol=asset)
        write_json(
            source.with_suffix(".metadata.json"),
            {"market": market, "sha256": digest(source), "quality": {"synthetic": True}},
        )
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(
        focused, "parallel_jobs", lambda function, jobs, config: [function(job, "cpu", 1) for job in jobs]
    )
    output = focused.run(spec_path, tmp_path / "result")
    selected = json.loads((output / "selection.json").read_text())
    final = json.loads((output / "final_results.json").read_text())
    assert set(final["assets"]) == {"SPY", "QQQ"}
    assert (output / "selection.json").stat().st_mtime <= (
        output / "finalists/SPY/result.json"
    ).stat().st_mtime
    for asset in spec["assets"]:
        development = pd.read_parquet(output / "development" / f"{asset}.parquet")
        assert development.bar_end.max() < pd.Timestamp("2014-01-01", tz="UTC")
        assert not selected["assets"][asset]["deployment_approved"]
        assert "double_costs_and_delay" in final["assets"][asset]["stress_tests"]
        _, _, _, manifest = load_bundle(output / "finalists" / asset)
        assert manifest["feature_builder"] == "etf"
    assert focused.run(spec_path, output) == output
    from tensortrade_lab.etf_audit import audit_job

    before = digest(output / "selection.json")
    diagnostic = audit_job(
        (str(output), "SPY", "synthetic_regime", "2014-01-01", "2016-01-01", spec["sources"]), "cpu", 1
    )
    assert diagnostic["previously_seen"]
    assert digest(output / "selection.json") == before
    assert diagnostic["metrics"]["agent"]["start"] == final["assets"]["SPY"]["metrics"]["agent"]["start"]
