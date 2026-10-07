import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/paper151"))
from download_factors import parse_monthly
from factor_study import factor_window, fit_scores, signal


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    months = pd.period_range("2010-01", "2026-08", freq="M")
    f = pd.DataFrame(
        rng.normal(0, 0.03, (len(months), 4)), index=months, columns=["Mkt-RF", "SMB", "HML", "RF"]
    )
    f.RF = 0.001
    slopes = rng.normal(0, 1, (3, 9))
    alpha = np.linspace(-0.01, 0.01, 9)
    y = (
        f.iloc[:, :3].to_numpy() @ slopes
        + alpha
        + f.RF.to_numpy()[:, None]
        + rng.normal(0, 0.005, (len(f), 9))
    )
    monthly = pd.DataFrame(y, index=months, columns=[str(i) for i in range(9)])
    factors = {"latest": f, **{str(y): f.loc[: f"{y}-07"].copy() for y in range(2016, 2026)}}
    return monthly, factors


def test_archive_release_cannot_be_used_at_previous_close(data):
    _, f = data
    _, a = factor_window(
        "archived", pd.Timestamp("2017-08-31 20:00", tz="UTC"), pd.Timestamp("2017-09-01", tz="UTC"), f
    )
    assert a["vintage"] == "2016"
    _, b = factor_window(
        "archived", pd.Timestamp("2017-09-29 20:00", tz="UTC"), pd.Timestamp("2017-10-02", tz="UTC"), f
    )
    assert b["vintage"] == "2017"


@pytest.mark.parametrize("mode", ["archived", "revised_annual", "revised_monthly"])
def test_future_months_and_unpublished_archives_cannot_change_signal(data, mode):
    monthly, f = data
    p = SimpleNamespace(symbols=list(map(str, range(11))))
    decision = pd.Timestamp("2017-02-28 21:00", tz="UTC")
    execution = pd.Timestamp("2017-03-01", tz="UTC")
    w, a = signal("ResidualMomentumLS", mode, decision, execution, p, monthly, f)
    future = monthly.copy()
    future.loc["2017-02":] *= 1000
    mutated = {k: v.copy() for k, v in f.items()}
    mutated["latest"].loc["2017-02":] *= -100
    for k in mutated:
        if k not in ("latest", "2016"):
            mutated[k] *= -100
    w2, b = signal("ResidualMomentumLS", mode, decision, execution, p, future, mutated)
    np.testing.assert_array_equal(w, w2)
    assert a == b
    assert abs(w.sum()) < 1e-10 and np.abs(w).sum() <= 1 + 1e-10


def test_residual_keeps_alpha_not_demeaned_fit_error(data):
    monthly, f = data
    table = f["latest"].iloc[:36]
    score, alpha, beta = fit_scores(monthly, table)
    y = monthly.loc[table.index].to_numpy() - table.RF.to_numpy()[:, None]
    expected = y[-12:] - table[["Mkt-RF", "SMB", "HML"]].to_numpy()[-12:] @ beta[1:]
    np.testing.assert_allclose(score, expected.mean(axis=0) / expected.std(axis=0, ddof=1))
    assert abs(alpha[-1]) > 0.001


def test_missing_factor_month_is_not_filled(data):
    _, f = data
    f["latest"] = f["latest"].drop(pd.Period("2016-12"))
    with pytest.raises(ValueError, match="Missing factor"):
        factor_window(
            "revised_monthly", pd.Timestamp("2017-02-28", tz="UTC"), pd.Timestamp("2017-03-01", tz="UTC"), f
        )


def test_csv_percent_conversion_and_missing_sentinel():
    f = parse_monthly("header\n201601, 1.5, -2.0, 3.0, 0.1\nAnnual factors\n2016,9,8,7,6")
    assert len(f) == 1 and f["Mkt-RF"][0] == 0.015 and f.SMB[0] == -0.02
    with pytest.raises(ValueError, match="Missing"):
        parse_monthly("201601,-99.99,1,1,1")
