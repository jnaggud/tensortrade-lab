import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Market(StrictConfig):
    symbol: str = Field(default="BTC", pattern=r"^[A-Z0-9]{1,12}$")
    currency: str = Field(default="USD", pattern=r"^[A-Z0-9]{1,12}$")
    bar_minutes: int = Field(default=60, ge=1)
    periods_per_year: int = Field(default=8760, ge=1)
    require_regular_bars: bool = True
    calendar: Literal["24/7", "XNYS"] = "24/7"
    price_adjustment: Literal["raw", "split_adjusted", "total_return"] = "raw"

    @model_validator(mode="after")
    def distinct_symbols(self):
        if self.symbol == self.currency:
            raise ValueError("Asset and currency must differ")
        return self


class Execution(StrictConfig):
    initial_cash: float = Field(default=10000, gt=0)
    commission: float = Field(default=0.001, ge=0, lt=0.1)
    slippage_bps: float = Field(default=5, ge=0, lt=1000)
    min_trade_value: float = Field(default=10, gt=0)
    rebalance_band: float = Field(default=0.05, ge=0, lt=1)
    delay_bars: int = Field(default=0, ge=0, le=20)


class Risk(StrictConfig):
    max_position: float = Field(default=0.75, gt=0, le=1)
    max_drawdown: float = Field(default=0.2, gt=0, le=1)
    stop_loss: float = Field(default=0.08, gt=0, le=1)
    take_profit: float = Field(default=0.2, gt=0)
    cooldown_bars: int = Field(default=4, ge=0)
    cooldown_minutes: int | None = Field(default=None, ge=0)
    direction: Literal["long_only", "long_short"] = "long_only"
    borrow_apr: float = Field(default=0.03, ge=0, le=5)
    borrow_available: bool = False
    maintenance_margin: float = Field(default=0.30, gt=0, lt=1)


class Features(StrictConfig):
    window: int = Field(default=24, ge=1, le=512)
    observation_minutes: int | None = Field(default=None, ge=1)
    reference_minutes: int = Field(default=60, ge=1)
    higher_timeframes: list[int] = Field(default_factory=list)


class Compute(StrictConfig):
    device: Literal["auto", "cpu", "mps", "cuda", "hybrid"] = "auto"
    workers: int = Field(default=0, ge=0)
    threads_per_worker: int = Field(default=1, ge=1)
    vector_envs: int = Field(default=0, ge=0)
    gpu_jobs: int = Field(default=1, ge=1)


class Split(StrictConfig):
    train_fraction: float = Field(default=0.6, gt=0, lt=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    purge_bars: int = Field(default=24, ge=1)

    @model_validator(mode="after")
    def leaves_test(self):
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Train and validation fractions must leave a test partition")
        return self


class Training(StrictConfig):
    total_timesteps: int = Field(default=100000, ge=1)
    eval_every: int = Field(default=5000, ge=1)
    patience: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.0003, gt=0, lt=1)
    n_steps: int = Field(default=1024, ge=2)
    batch_size: int = Field(default=128, ge=2)
    n_epochs: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip_range: float = Field(default=0.2, gt=0, lt=1)
    entropy_coefficient: float = Field(default=0.01, ge=0)
    hidden_sizes: list[int] = Field(default_factory=lambda: [64, 64], min_length=1)
    drawdown_penalty: float = Field(default=0.10, ge=0)
    turnover_penalty: float = Field(default=0.0001, ge=0)
    seed: int = Field(default=42, ge=0)
    random_episode_bars: int | None = Field(default=None, ge=20)

    @model_validator(mode="after")
    def batch_shape(self):
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size:
            raise ValueError("n_steps must be divisible by batch_size")
        if any(n < 1 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive")
        return self


class Config(StrictConfig):
    market: Market = Field(default_factory=Market)
    execution: Execution = Field(default_factory=Execution)
    risk: Risk = Field(default_factory=Risk)
    features: Features = Field(default_factory=Features)
    split: Split = Field(default_factory=Split)
    training: Training = Field(default_factory=Training)
    compute: Compute = Field(default_factory=Compute)
    objective: Literal["return_drawdown", "excess_return"] = "return_drawdown"
    selection_drawdown_limit: float = Field(default=0.30, gt=0, le=1)

    @property
    def window(self):
        return (
            max(1, math.ceil(self.features.observation_minutes / self.market.bar_minutes))
            if self.features.observation_minutes
            else self.features.window
        )

    @property
    def cooldown(self):
        return (
            math.ceil(self.risk.cooldown_minutes / self.market.bar_minutes)
            if self.risk.cooldown_minutes is not None
            else self.risk.cooldown_bars
        )


def load_config(path: str | Path | None = None) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text()) or {}) if path else Config()
