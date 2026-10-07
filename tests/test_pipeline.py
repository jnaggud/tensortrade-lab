import json

import pandas as pd
import pytest

from tensortrade_lab.artifacts import load_bundle
from tensortrade_lab.config import Config
from tensortrade_lab.data import synthetic_bars
from tensortrade_lab.evaluation import rollout
from tensortrade_lab.features import engineer
from tensortrade_lab.paper import backtest, replay
from tensortrade_lab.training import train, walk_forward


def tiny_config():
    raw = Config().model_dump()
    raw["features"]["window"] = 4
    raw["split"]["purge_bars"] = 4
    raw["training"].update(
        total_timesteps=64, n_steps=32, batch_size=16, n_epochs=1, eval_every=32, hidden_sizes=[16]
    )
    return Config.model_validate(raw)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment")
    source = root / "bars.csv"
    all_bars = synthetic_bars(650)
    all_bars.iloc[:600].to_csv(source, index=False)
    config = tiny_config()
    run = train(source, config, root / "runs", synthetic=True)
    return run, source, config, all_bars


def test_checkpoint_roundtrip_and_recorded_test(experiment):
    run, source, config, _ = experiment
    model, scaler, loaded_config, manifest = load_bundle(run)
    assert loaded_config == config
    frame = engineer(pd.read_csv(source).assign(timestamp=lambda x: pd.to_datetime(x.timestamp, utc=True)))
    start, stop = manifest["partitions"]["test"]
    recomputed, _, _ = rollout(frame.iloc[start:stop].reset_index(drop=True), scaler, config, model)
    recorded = json.loads((run / "test/metrics.json").read_text())["agent"]
    assert recomputed == recorded
    assert "SYNTHETIC DATA" in (run / "report.html").read_text()
    assert manifest["actual_timesteps"] == 64


def test_paper_idempotence_and_append_only(experiment, tmp_path):
    run, source, _, all_bars = experiment
    paper_source = tmp_path / "paper.csv"
    paper_source.write_bytes(source.read_bytes())
    state = tmp_path / "account.json"
    first = replay(run, paper_source, state)
    second = replay(run, paper_source, state)
    assert first["account"] == second["account"]
    assert second["new_fills"] == 0
    old_fills = pd.read_csv(state.with_suffix(".fills.csv"))
    all_bars.to_csv(paper_source, index=False)
    third = replay(run, paper_source, state)
    assert third["rows"] == second["rows"] + 50
    new_fills = pd.read_csv(state.with_suffix(".fills.csv"))
    pd.testing.assert_frame_equal(old_fills, new_fills.iloc[: len(old_fills)])
    changed = all_bars.copy()
    changed.loc[100, "volume"] *= 2
    changed.to_csv(paper_source, index=False)
    with pytest.raises(ValueError, match="Historical candles changed"):
        replay(run, paper_source, state)


def test_backtest_rejects_model_selection_data(experiment, tmp_path):
    run, source, _, all_bars = experiment
    output = backtest(run, source, tmp_path)
    assert (output / "metrics.json").is_file()
    past = tmp_path / "past.csv"
    all_bars.iloc[:300].to_csv(past, index=False)
    with pytest.raises(ValueError, match="no candles after"):
        backtest(run, past, tmp_path)


def test_walk_forward_has_disjoint_test_periods(experiment, tmp_path):
    _, source, config, _ = experiment
    run = walk_forward(source, config, tmp_path, folds=2, synthetic=True)
    folds = pd.read_csv(run / "folds.csv")
    assert len(folds) == 2
    assert folds.end.iloc[0] < folds.start.iloc[1]
    for path in run.glob("fold-*/manifest.json"):
        manifest = json.loads(path.read_text())
        assert manifest["partitions"]["train"][1] < manifest["partitions"]["validation"][0]
        assert manifest["partitions"]["validation"][1] < manifest["partitions"]["test"][0]


def test_paper_rejects_edits_to_dropped_warmup_rows(experiment, tmp_path):
    run, source, _, _ = experiment
    paper_source = tmp_path / "paper.csv"
    paper_source.write_bytes(source.read_bytes())
    state = tmp_path / "account.json"
    replay(run, paper_source, state)
    bars = pd.read_csv(paper_source)
    bars.loc[0, "high"] *= 2
    bars.to_csv(paper_source, index=False)
    with pytest.raises(ValueError, match="Historical candles changed"):
        replay(run, paper_source, state)


def test_new_holdout_can_start_after_validation(experiment, tmp_path):
    run, _, _, all_bars = experiment
    source = tmp_path / "new.csv"
    future = all_bars.copy()
    future.timestamp += pd.Timedelta(days=365)
    future.to_csv(source, index=False)
    output = backtest(run, source, tmp_path)
    results = json.loads((output / "metrics.json").read_text())
    assert results["agent"]["start"] > "2020-12-01"
