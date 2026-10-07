"""Strict OHLCV ingestion and deterministic, explicitly synthetic demo data."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Market

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_bars(data: pd.DataFrame, market: Market) -> pd.DataFrame:
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    extras = [c for c in ("dividend", "split", "borrow_available") if c in data]
    bars = data[COLUMNS + extras].copy()
    # Numeric epochs are deliberately rejected; milliseconds vs seconds is ambiguous.
    if pd.api.types.is_numeric_dtype(bars.timestamp):
        raise ValueError("timestamp must contain ISO-8601 dates, not ambiguous numeric epochs")
    bars["timestamp"] = pd.to_datetime(bars.timestamp, utc=True, errors="raise")
    if bars.timestamp.isna().any() or bars.timestamp.duplicated().any():
        raise ValueError("Timestamps must be non-null and unique")
    if not bars.timestamp.is_monotonic_increasing:
        raise ValueError("Candles must be in increasing timestamp order")
    for col in COLUMNS[1:]:
        bars[col] = pd.to_numeric(bars[col], errors="raise").astype(float)
    if not np.isfinite(bars[COLUMNS[1:]].to_numpy()).all():
        raise ValueError("OHLCV contains NaN or infinite values")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any() or (bars.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if (
        (bars.high < bars[["open", "close", "low"]].max(axis=1))
        | (bars.low > bars[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Invalid OHLC bounds")
    if len(bars) < 100:
        raise ValueError("At least 100 candles are required")
    for name, default in (("dividend", 0.0), ("split", 1.0)):
        bars[name] = pd.to_numeric(bars[name], errors="raise") if name in bars else default
        if not np.isfinite(bars[name]).all() or (bars[name] < 0).any():
            raise ValueError(f"Invalid {name} values")
    bars["split"] = bars["split"].replace(0, 1)
    if "borrow_available" in bars:
        parsed = (
            bars.borrow_available.astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        if parsed.isna().any():
            raise ValueError("borrow_available must contain boolean values")
        bars["borrow_available"] = parsed
    if market.calendar == "XNYS":
        from .market_data import attach_calendar

        bars = attach_calendar(bars.reset_index(drop=True), market)
    elif market.require_regular_bars:
        expected = pd.Timedelta(minutes=market.bar_minutes)
        if not bars.timestamp.diff().iloc[1:].eq(expected).all():
            raise ValueError("Missing candles or unexpected interval; repair data or configure session gaps")
    if "bar_end" not in bars:
        bars["bar_end"] = bars.timestamp + pd.Timedelta(minutes=market.bar_minutes)
    return bars.reset_index(drop=True)


def load_bars(path: str | Path, market: Market) -> pd.DataFrame:
    path = Path(path)
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return validate_bars(data, market)


def synthetic_bars(rows: int = 4000, seed: int = 42, bar_minutes: int = 60) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("At least 100 synthetic candles are required")
    rng = np.random.default_rng(seed)
    regime = np.sin(np.arange(rows) / 170) * 0.0006
    returns = regime + rng.normal(0, 0.007, rows)
    close = 30000 * np.exp(np.cumsum(returns))
    opening = np.r_[30000, close[:-1]] * np.exp(rng.normal(0, 0.001, rows))
    spread = rng.uniform(0.0005, 0.008, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq=f"{bar_minutes}min", tz="UTC"),
            "open": opening,
            "high": np.maximum(opening, close) * (1 + spread),
            "low": np.minimum(opening, close) * (1 - spread),
            "close": close,
            "volume": rng.lognormal(5, 0.6, rows),
        }
    )
