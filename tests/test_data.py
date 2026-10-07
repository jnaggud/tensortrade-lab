import numpy as np
import pandas as pd
import pytest

from tensortrade_lab.config import Config, Market
from tensortrade_lab.data import synthetic_bars, validate_bars
from tensortrade_lab.evaluation import chronological_split
from tensortrade_lab.features import FEATURE_NAMES, Scaler, engineer


def test_causal_features_and_scaling():
    original = synthetic_bars(300)
    changed = original.copy()
    changed.loc[200:, ["open", "high", "low", "close"]] *= 3
    a, b = engineer(original), engineer(changed)
    pd.testing.assert_frame_equal(a.iloc[:152], b.iloc[:152])
    scaler_a, scaler_b = Scaler.fit(a.iloc[:100]), Scaler.fit(b.iloc[:100])
    assert scaler_a == scaler_b
    np.testing.assert_allclose(scaler_a.transform(a)[:152], scaler_b.transform(b)[:152])
    assert np.isfinite(a[FEATURE_NAMES].to_numpy()).all()


@pytest.mark.parametrize(
    "corruption", ["duplicate", "order", "nan", "negative", "bounds", "gap", "numeric_time"]
)
def test_reject_invalid_candles(corruption):
    frame = synthetic_bars(200)
    if corruption == "duplicate":
        frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"]
    elif corruption == "order":
        frame = frame.iloc[::-1]
    elif corruption == "nan":
        frame.loc[3, "close"] = np.nan
    elif corruption == "negative":
        frame.loc[3, "volume"] = -1
    elif corruption == "bounds":
        frame.loc[3, "high"] = 1
    elif corruption == "gap":
        frame = frame.drop(index=3)
    else:
        frame["timestamp"] = np.arange(len(frame))
    with pytest.raises(ValueError):
        validate_bars(frame, Market())


def test_split_purges_and_leaves_test():
    config = Config()
    splits = chronological_split(2000, config)
    assert splits.train[1] + config.split.purge_bars == splits.validation[0]
    assert splits.validation[1] + config.split.purge_bars == splits.test[0]
    assert splits.test[1] == 2000
    with pytest.raises(ValueError):
        chronological_split(100, config)


def test_config_rejects_typo_and_nonfinite():
    with pytest.raises(ValueError):
        Config.model_validate({"risk": {"max_postion": 2}})
    with pytest.raises(ValueError):
        Config.model_validate({"execution": {"commission": float("nan")}})
