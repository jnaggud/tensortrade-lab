import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault(
    "MPLCONFIGDIR",
    "/private/tmp/tensortrade-mpl" if os.path.exists("/private/tmp") else "/tmp/tensortrade-mpl",
)

import pytest

from tensortrade_lab.config import Config
from tensortrade_lab.data import synthetic_bars
from tensortrade_lab.features import Scaler, engineer


def pytest_addoption(parser):
    parser.addoption(
        "--run-data-tests",
        action="store_true",
        default=False,
        help="Run reconciliation checks against separately supplied source exports.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-data-tests"):
        return
    skip = pytest.mark.skip(reason="Requires local source exports; opt in with --run-data-tests")
    for item in items:
        if "external_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def config():
    raw = Config().model_dump()
    raw["features"]["window"] = 4
    raw["risk"].update(stop_loss=0.9, take_profit=10, max_drawdown=0.9)
    return Config.model_validate(raw)


@pytest.fixture
def frame():
    return engineer(synthetic_bars(250))


@pytest.fixture
def scaler(frame):
    return Scaler.fit(frame.iloc[:100])
