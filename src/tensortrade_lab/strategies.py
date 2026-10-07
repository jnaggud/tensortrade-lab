"""Causal rule baselines and supervised next-open-return forecasting."""

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import feature_columns
from .margin import target_action


@dataclass
class RulePolicy:
    strategy: str
    lookback: int
    threshold: float
    long_short: bool

    def action(self, bars, index):
        if index < self.lookback:
            return 1
        close = float(bars.indicator_close.iloc[index])
        past = bars.indicator_close.iloc[index - self.lookback : index]
        if self.strategy == "momentum":
            signal = close / float(past.iloc[0]) - 1
        elif self.strategy == "mean_reversion":
            signal = float(past.mean()) / close - 1
        elif self.strategy == "breakout":
            signal = (
                close / float(past.max()) - 1
                if close > past.max()
                else close / float(past.min()) - 1
                if close < past.min()
                else 0
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return target_action(float(np.sign(signal)) if abs(signal) > self.threshold else 0, self.long_short)


@dataclass
class SupervisedPolicy:
    estimator: object
    columns: list[str]
    threshold: float
    long_short: bool

    def prepare(self, bars):
        self.predictions = self.estimator.predict(bars[self.columns].to_numpy())

    def action(self, bars, index):
        signal = float(self.predictions[index])
        return target_action(float(np.sign(signal)) if abs(signal) > self.threshold else 0, self.long_short)


def fit_strategy(strategy, train, config, params):
    long_short = config.risk.direction == "long_short"
    threshold = float(params.get("threshold", 0.002))
    lookback = max(1, math.ceil(params.get("lookback_minutes", 1440) / config.market.bar_minutes))
    if strategy != "supervised":
        return RulePolicy(strategy, lookback, threshold, long_short)
    columns = feature_columns(train)
    horizon = max(1, math.ceil(params.get("forecast_minutes", 240) / config.market.bar_minutes))
    # Features at close t; the earliest tradable entry is open t+1.
    labels = train.close.shift(-horizon) / train.open.shift(-1) - 1
    valid = labels.notna()
    if valid.sum() < 50:
        raise ValueError("Insufficient training labels after purging the forward horizon")
    estimator = HistGradientBoostingRegressor(
        max_iter=int(params.get("trees", 100)),
        max_leaf_nodes=int(params.get("leaves", 15)),
        learning_rate=float(params.get("tree_learning_rate", 0.05)),
        l2_regularization=1.0,
        early_stopping=False,
        random_state=config.training.seed,
    )
    estimator.fit(train.loc[valid, columns].to_numpy(), labels[valid].to_numpy())
    return SupervisedPolicy(estimator, columns, threshold, long_short)
