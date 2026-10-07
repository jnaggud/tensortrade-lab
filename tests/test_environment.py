import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from tensortrade_lab.config import Config
from tensortrade_lab.environment import ResearchEnv
from tensortrade_lab.evaluation import rollout


def constant_market(frame, price=100):
    result = frame.copy()
    result[["open", "high", "low", "close"]] = price
    return result


def test_gym_api_and_deterministic_reset(frame, scaler, config):
    env = ResearchEnv(frame, scaler.transform(frame), config)
    check_env(env)
    first, _ = env.reset(seed=7)
    env.step(5)
    second, info = env.reset(seed=7)
    np.testing.assert_array_equal(first, second)
    assert info["cash"] == config.execution.initial_cash
    assert info["units"] == 0


def test_no_future_observations_and_next_open_fills(frame, scaler, config):
    a = constant_market(frame)
    b = a.copy()
    start = config.features.window - 1
    b.loc[start + 1, ["open", "high", "low", "close"]] = 200
    env_a = ResearchEnv(a, scaler.transform(a), config)
    env_b = ResearchEnv(b, scaler.transform(b), config)
    np.testing.assert_array_equal(env_a.reset()[0], env_b.reset()[0])
    env_a.step(5)
    env_b.step(5)
    assert env_b.state.fills[0]["price"] == pytest.approx(200 * 1.0005)
    assert env_a.state.units == pytest.approx(2 * env_b.state.units, abs=1e-7)


def test_round_trip_matches_commission_and_slippage(frame, scaler, config):
    bars = constant_market(frame)
    raw = config.model_dump()
    raw["risk"]["max_position"] = 1
    env = ResearchEnv(bars, scaler.transform(bars), Config.model_validate(raw))
    env.reset()
    env.step(5)
    env.step(1)
    c, s = config.execution.commission, config.execution.slippage_bps / 10000
    expected = config.execution.initial_cash * (1 - c) ** 2 * (1 - s) / (1 + s)
    assert env.state.equity == pytest.approx(expected, abs=1e-5)
    assert env.state.units == pytest.approx(0, abs=1e-7)
    assert len(env.state.broker.trades) == 2


def test_post_cost_allocation_respects_limit(frame, scaler, config):
    bars = constant_market(frame)
    env = ResearchEnv(bars, scaler.transform(bars), config)
    env.reset()
    for action in [5, 3, 4, 2, 5]:
        env.step(action)
        assert env.state.exposure == pytest.approx((action - 1) * config.risk.max_position / 4, abs=1e-6)
        assert env.state.cash >= 0


def test_drawdown_halts_at_next_open_and_keeps_horizon(frame, scaler, config):
    bars = constant_market(frame)
    raw = config.model_dump()
    raw["risk"]["max_drawdown"] = 0.1
    start = config.features.window - 1
    bars.loc[start + 2 :, ["open", "high", "low", "close"]] = 50
    env = ResearchEnv(bars, scaler.transform(bars), Config.model_validate(raw))
    env.reset()
    env.step(5)
    _, _, done, _, _ = env.step(0)
    assert env.state.halted
    assert env.state.units == pytest.approx(0, abs=1e-7)
    assert not done
    while not done:
        _, _, done, _, _ = env.step(5)
    assert env.state.index == len(bars) - 1
    assert env.state.units == pytest.approx(0, abs=1e-7)


def test_stop_loss_and_cooldown(frame, scaler, config):
    bars = constant_market(frame)
    raw = config.model_dump()
    raw["risk"].update(stop_loss=0.05, cooldown_bars=3)
    start = config.features.window - 1
    bars.loc[start + 2 :, ["open", "high", "low", "close"]] = 90
    env = ResearchEnv(bars, scaler.transform(bars), Config.model_validate(raw))
    env.reset()
    env.step(5)
    env.step(5)
    assert env.state.fills[-1]["reason"] == "stop_loss"
    for _ in range(3):
        env.step(5)
        assert env.state.units == pytest.approx(0, abs=1e-7)
    env.step(5)
    assert env.state.units > 0


def test_cash_and_buy_hold_share_horizon_and_pay_costs(frame, scaler, config):
    bars = constant_market(frame)
    cash, history, _ = rollout(bars, scaler, config, "cash")
    buy, buy_history, fills = rollout(bars, scaler, config, "buy_hold")
    assert cash["total_return"] == 0
    assert cash["sharpe"] == 0
    assert buy["total_return"] < 0
    assert len(fills) == 2
    assert len(history) == len(buy_history)
    assert buy_history.units.iloc[-1] == pytest.approx(0, abs=1e-7)


def test_paper_keeps_position_at_end(frame, scaler, config):
    bars = constant_market(frame)
    env = ResearchEnv(bars, scaler.transform(bars), config, liquidate_on_end=False)
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(5)
    assert env.state.units > 0
    assert not any(fill["reason"] == "end_of_sample" for fill in env.state.fills)
    with pytest.raises(RuntimeError, match="Episode is complete"):
        env.step(0)


def test_rejects_invalid_action(frame, scaler, config):
    env = ResearchEnv(frame, scaler.transform(frame), config)
    env.reset()
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(1.5)
