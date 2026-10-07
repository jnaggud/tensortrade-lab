import copy

import numpy as np
import pandas as pd
import pytest

from tensortrade_lab.config import Market
from tensortrade_lab.market_data import expected_grid
from tensortrade_lab.portfolio import Ledger, make_panel, momentum_weights, rebalance_days
from tensortrade_lab.rotation import family_bootstrap, training_examples


@pytest.fixture
def panel():
    dates = expected_grid("2004-01-01", "2008-12-31", Market(calendar="XNYS", bar_minutes=1440))
    frames = {}
    for j, symbol in enumerate(["A", "B", "SPY"]):
        n = len(dates)
        close = 100 * np.exp(np.cumsum(np.sin(np.arange(n) / 17 + j) * 0.002 + 0.0001))
        frames[symbol] = dates.assign(open=close * 0.999, close=close, dividend=0.0)
    rates = pd.DataFrame(
        {"date": pd.date_range("2003-12-01", "2008-12-31", freq="B").astype(str), "discount_yield": 0.04}
    )
    return make_panel(frames, rates)


def test_self_financing_and_independent_fill_reconciliation(panel):
    p = copy.deepcopy(panel)
    p.dividend[302] = [1, 0.5, 0]
    ledger = Ledger(p, 300, 305, 0.001, 0.002)
    actions = [[0.6, 0.4, 0], None, [0.2, 0.7, 0], None, [0, 0, 0], [0.1, 0.3, 0.6]]
    cash, shares = 10000.0, np.zeros(3)
    cursor = 0
    for t, action in enumerate(actions, start=300):
        cash += (
            cash * p.cash_rate[t] * (p.timestamp[t].normalize() - p.timestamp[t - 1].normalize()).days / 365
        )
        cash += float(shares @ p.dividend[t])
        ledger.step(action)
        for fill in ledger.fills[cursor:]:
            j = p.symbols.index(fill["symbol"])
            cash -= fill["units"] * fill["price"] + fill["fee"]
            shares[j] += fill["units"]
        cursor = len(ledger.fills)
        assert ledger.cash == pytest.approx(cash, abs=1e-7)
        assert ledger.equity == pytest.approx(cash + shares @ p.close[t], abs=1e-7)
        assert ledger.cash >= 0
    assert np.abs(shares).max() < 1e-10
    with pytest.raises(ValueError):
        ledger.step()


def test_dividends_exclude_entry_day_and_passive_reinvests(panel):
    p = copy.deepcopy(panel)
    p.cash_rate[:] = 0
    p.opening[:] = p.close[:] = 100
    p.dividend[300:303, 0] = 1
    ledger = Ledger(p, 300, 302, 0, 0)
    ledger.step([1, 0, 0], drip=True)
    assert ledger.dividends == 0
    ledger.step(drip=True)
    assert ledger.equity == pytest.approx(10100)
    ledger.step(drip=True)
    assert ledger.equity == pytest.approx(10201)
    assert ledger.dividends == pytest.approx(201)


def test_next_open_and_no_leverage(panel):
    p = copy.deepcopy(panel)
    p.opening[300, 0] = 200
    ledger = Ledger(p, 300, 302, 0, 0)
    ledger.step([1, 0, 0])
    assert ledger.shares[0] == pytest.approx((10000 + ledger.interest) / 200)
    for bad in ([1, 1, 0], [1, -0.1, 0], [np.nan, 0, 0], [1, 0]):
        with pytest.raises(ValueError):
            ledger.rebalance(bad, p.opening[301])


def test_feature_and_action_causality(panel):
    p = copy.deepcopy(panel)
    expected = momentum_weights(p, 400, 252, investable=2, top_k=1)
    p.total_index[401:] *= 20
    p.volatility[401:] *= 20
    np.testing.assert_array_equal(momentum_weights(p, 400, 252, investable=2, top_k=1), expected)
    # Prefixing data also removes all future label inputs.
    train = panel.prefix("2007-01-01")
    x, y, info = training_examples(train, 2, 21)
    changed = copy.deepcopy(panel)
    changed.opening[len(train.timestamp) :] *= 10
    x2, y2, _ = training_examples(changed.prefix("2007-01-01"), 2, 21)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    assert pd.Timestamp(info["label_end"]) < pd.Timestamp("2007-01-01", tz="UTC")


def test_labels_use_open_execution_and_entitlement(panel):
    p = copy.deepcopy(panel)
    p.opening[:] = 100
    p.dividend[:] = 0
    p.dividend[253, 0] = 100  # Entry at 253 is NOT entitled to this distribution.
    p.dividend[254, 0] = 2
    _, y, _ = training_examples(p, 2, 1)
    assert y[0] == pytest.approx(0.01)
    assert y[1] == pytest.approx(-0.01)


def test_holiday_schedule_and_truncation(panel):
    weekly = rebalance_days(panel, "weekly")
    days = panel.timestamp.strftime("%Y-%m-%d")
    assert weekly[np.flatnonzero(days == "2007-04-05")[0]]  # Good Friday is closed.
    partial = panel.prefix("2007-04-04")
    assert not rebalance_days(partial, "weekly")[-1]


def test_joint_correction_rejects_uniform_underperformance():
    spec = {"folds": [{"name": "one"}], "seed": 17, "bootstrap_samples": 100, "bootstrap_block": 20}
    baseline = {"one": {"timestamps": list(range(200)), "log_returns": [0.001] * 200}}
    rows = {
        f"c{i}": {"one": {"timestamps": list(range(200)), "log_returns": [0.0005] * 200}} for i in range(10)
    }
    stats = family_bootstrap(rows, baseline, spec)
    assert all(r["family_adjusted_p"] == 1 for r in stats.values())


def test_tensortrade_matches_direct_ledger(panel):
    from stable_baselines3.common.env_checker import check_env

    from tensortrade_lab.portfolio_env import PortfolioEnv

    mean, scale = np.zeros(10), np.ones(10)
    env = PortfolioEnv(panel, 300, 305, 2, "weekly", mean, scale)
    check_env(env, warn=True)
    env.reset(seed=17)
    direct = Ledger(panel, 300, 305)
    scheduled = rebalance_days(panel, "weekly")
    action = np.zeros(3, dtype=np.float32)
    while direct.index < direct.end:
        weights = [1 / 3, 1 / 3, 0] if direct.index == 299 or scheduled[direct.index] else None
        direct.step(weights)
        _, reward, done, _, _ = env.step(action)
        assert env.state.equity == pytest.approx(direct.equity, abs=0.001)
        assert reward == pytest.approx(direct.reward_value, abs=1e-7)
    assert done


def test_simulations_and_supervised_cutoff(panel, tmp_path):
    from tensortrade_lab.rotation import fit_ranker, simulate

    spec = {
        "universe": ["A", "B"],
        "benchmark": "SPY",
        "label_horizon": 21,
        "supervised_seed": 17,
        "commission": 0.0005,
        "slippage": 0.0005,
        "initial_cash": 10000.0,
        "skip_recent": 21,
        "top_k": 1,
    }
    fold = {"start": "2007-01-01", "end": "2008-01-01"}
    model, info = fit_ranker(panel, fold, spec)
    assert pd.Timestamp(info["label_end"]) < pd.Timestamp(fold["start"], tz="UTC")
    for family in ("momentum", "supervised", "spy", "passive", "cash"):
        candidate = {"family": family, "frequency": "monthly", "lookback": 126, "weighting": "equal"}
        ledger = simulate(panel, fold, candidate, spec, model, save=tmp_path / family)
        assert np.isfinite([r["equity"] for r in ledger.history]).all()
        assert ledger.shares.sum() == pytest.approx(0, abs=1e-8)
        decisions = pd.read_csv(tmp_path / family / "decisions.csv")
        assert (pd.to_datetime(decisions.decided_at) < pd.to_datetime(decisions.execute_at)).all()
        if family == "cash":
            assert not ledger.fills
            assert ledger.equity > 10000


def test_raw_feature_prefix_and_rate_lag():
    dates = expected_grid("2004-01-01", "2006-12-31", Market(calendar="XNYS", bar_minutes=1440))
    frames = {
        s: dates.assign(open=100 + np.arange(len(dates)), close=101 + np.arange(len(dates)), dividend=0.0)
        for s in ("A", "B")
    }
    rates = pd.DataFrame(
        {"date": pd.date_range("2003-12-01", "2006-12-31", freq="B").astype(str), "discount_yield": 0.04}
    )
    original = make_panel(frames, rates)
    future = frames["B"].timestamp.iloc[400]
    frames["B"].loc[400:, ["open", "close"]] *= 2
    rates.loc[pd.to_datetime(rates.date, utc=True) >= future.normalize(), "discount_yield"] = 0.10
    altered = make_panel(frames, rates)
    np.testing.assert_allclose(original.features[:400], altered.features[:400], equal_nan=True)
    np.testing.assert_array_equal(original.cash_rate[:401], altered.cash_rate[:401])
    frames["A"] = frames["A"].drop(index=350)
    with pytest.raises(ValueError, match="missing sessions"):
        make_panel(frames, rates)


def test_portfolio_ppo_can_train_and_reload(panel, tmp_path):
    from stable_baselines3 import PPO

    from tensortrade_lab.compute import configure_threads
    from tensortrade_lab.portfolio_env import PortfolioEnv

    configure_threads(1)
    env = PortfolioEnv(panel, 300, 340, 2, "weekly", np.zeros(10), np.ones(10))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=17,
        policy_kwargs={"net_arch": [16]},
        device="cpu",
    )
    model.learn(total_timesteps=64)
    model.save(tmp_path / "model")
    loaded = PPO.load(tmp_path / "model", device="cpu")
    observation, _ = env.reset(seed=17)
    action, _ = loaded.predict(observation, deterministic=True)
    observation, reward, _, _, info = env.step(action)
    assert np.isfinite(observation).all() and np.isfinite(reward)
    assert info["equity"] > 0
