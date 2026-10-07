"""Causal indicators; normalization is fitted exclusively on training rows."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

FEATURE_NAMES = [
    "return_1",
    "return_4",
    "return_12",
    "return_24",
    "return_48",
    "trend_12",
    "trend_48",
    "volatility_24",
    "rsi_14",
    "range",
    "body",
    "volume_ratio",
    "hour_sin",
    "hour_cos",
]


def feature_columns(frame):
    return FEATURE_NAMES + sorted(c for c in frame if c.startswith("context_"))


def engineer(bars: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    config = config or Config()
    period = lambda n: max(1, math.ceil(n * config.features.reference_minutes / config.market.bar_minutes))
    # Forward adjustment uses only splits already effective. Execution retains original prices.
    factor = (
        bars.get("split", pd.Series(1.0, index=bars.index)).cumprod()
        if config.market.price_adjustment == "raw"
        else 1.0
    )
    close = bars.close * factor
    out = bars.copy()
    out["indicator_close"] = close
    for n in (1, 4, 12, 24, 48):
        out[f"return_{n}"] = np.log(close / close.shift(period(n)))
    for n in (12, 48):
        out[f"trend_{n}"] = close / close.rolling(period(n)).mean() - 1
    out["volatility_24"] = np.log(close / close.shift(1)).rolling(max(2, period(24))).std(ddof=0)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period(14), adjust=False, min_periods=period(14)).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period(14), adjust=False, min_periods=period(14)).mean()
    out["rsi_14"] = (gain / (gain + loss).replace(0, np.nan)).fillna(0.5)
    out["range"] = (bars.high - bars.low) / bars.close
    out["body"] = (bars.close - bars.open) / bars.open
    mean_volume = bars.volume.rolling(period(24)).mean()
    out["volume_ratio"] = np.log1p(bars.volume) - np.log1p(mean_volume)
    hour = bars.timestamp.dt.hour + bars.timestamp.dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if config.features.higher_timeframes:
        from .market_data import resample_bars

        for minutes in sorted(set(config.features.higher_timeframes)):
            if minutes <= config.market.bar_minutes:
                raise ValueError("Context timeframes must exceed the decision timeframe")
            higher = resample_bars(bars, config.market, minutes)
            higher[f"context_{minutes}_return"] = (
                np.log(
                    (higher.close * higher.split.cumprod()) / (higher.close * higher.split.cumprod()).shift(1)
                )
                if config.market.price_adjustment == "raw"
                else np.log(higher.close / higher.close.shift(1))
            )
            columns = ["bar_end", f"context_{minutes}_return"]
            # Publish only after the entire higher-timeframe bar has closed.
            out = pd.merge_asof(
                out.sort_values("bar_end"),
                higher[columns].sort_values("bar_end"),
                on="bar_end",
                direction="backward",
            )
    # Drop warm-up rows. Never bfill: that would introduce future information.
    return out.dropna(subset=feature_columns(out)).reset_index(drop=True)


@dataclass
class Scaler:
    mean: list[float]
    scale: list[float]
    columns: list[str]

    @classmethod
    def fit(cls, train: pd.DataFrame):
        columns = feature_columns(train)
        x = train[columns].to_numpy(dtype=float)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("Training features must be finite and nonempty")
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1
        return cls(x.mean(axis=0).tolist(), scale.tolist(), columns)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if self.columns[: len(FEATURE_NAMES)] != FEATURE_NAMES or not set(self.columns).issubset(
            data.columns
        ):
            raise ValueError("Checkpoint feature schema differs from this application")
        x = (data[self.columns].to_numpy(float) - np.array(self.mean)) / np.array(self.scale)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite normalized features")
        return np.clip(x, -10, 10).astype(np.float32)
