"""TensorTrade adapter for the exact same ledger used by portfolio research."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from tensortrade.env.default.renderers import EmptyRenderer
from tensortrade.env.generic import ActionScheme, Informer, Observer, RewardScheme, Stopper, TradingEnv

from .portfolio import Ledger, rebalance_days


class Allocation(ActionScheme):
    def __init__(self, state, investable, frequency):
        super().__init__()
        self.state, self.investable = state, investable
        self.scheduled = rebalance_days(state.panel, frequency)

    @property
    def action_space(self):
        return Box(-5, 5, (self.investable + 1,), dtype=np.float32)

    def perform(self, env, action):
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError("Invalid allocation action")
        weights = None
        if self.state.index == self.state.start - 1 or self.scheduled[self.state.index]:
            values = np.exp(action - np.max(action))
            values /= values.sum()  # Last component represents cash.
            weights = np.pad(values[:-1], (0, len(self.state.panel.symbols) - self.investable))
        self.state.step(weights)

    def reset(self):
        self.state.reset()


class PortfolioObserver(Observer):
    def __init__(self, state, investable, mean, scale):
        super().__init__()
        self.state, self.investable = state, investable
        self.mean, self.scale = mean, scale

    @property
    def observation_space(self):
        count = self.investable * self.state.panel.features.shape[2] + self.investable + 1
        return Box(-10, 10, (count,), dtype=np.float32)

    def observe(self, env):
        s = self.state
        features = (s.panel.features[s.index, : self.investable] - self.mean) / self.scale
        weights = s.shares[: self.investable] * s.panel.close[s.index, : self.investable] / s.equity
        return np.clip(np.r_[features.ravel(), weights, s.cash / s.equity], -10, 10).astype(np.float32)

    def reset(self, random_start=0):
        pass


class PortfolioReward(RewardScheme):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def reward(self, env):
        return self.state.reward_value


class PortfolioStopper(Stopper):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def stop(self, env):
        return self.state.index >= self.state.end


class PortfolioInfo(Informer):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def info(self, env):
        return {"equity": self.state.equity}


class PortfolioEnv(TradingEnv):
    def __init__(self, panel, start, end, investable, frequency, mean, scale, **costs):
        self.state = Ledger(panel, start, end, **costs)
        super().__init__(
            action_scheme=Allocation(self.state, investable, frequency),
            observer=PortfolioObserver(self.state, investable, mean, scale),
            reward_scheme=PortfolioReward(self.state),
            stopper=PortfolioStopper(self.state),
            informer=PortfolioInfo(self.state),
            renderer=EmptyRenderer(),
        )

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        return super().reset(seed=seed, options=options)
