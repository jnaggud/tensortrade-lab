"""Exchange-session grids and complete-bar resampling. All timestamps denote bar opens."""

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import Market


@lru_cache(maxsize=64)
def schedule(start: str, end: str):
    import exchange_calendars as xc

    return xc.get_calendar("XNYS", start=start, end=end).schedule.loc[start:end]


def expected_grid(start, end, market: Market):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if market.calendar == "24/7":
        opens = pd.date_range(start.floor(f"{market.bar_minutes}min"), end, freq=f"{market.bar_minutes}min")
        return pd.DataFrame({"timestamp": opens, "bar_end": opens + pd.Timedelta(minutes=market.bar_minutes)})
    sessions = schedule(str(start.date()), str(end.date()))
    frames = []
    for row in sessions.itertuples():
        if market.bar_minutes == 1440:
            opens = pd.DatetimeIndex([row.open])
        else:
            opens = pd.date_range(row.open, row.close, freq=f"{market.bar_minutes}min", inclusive="left")
        ends = pd.DatetimeIndex([min(t + pd.Timedelta(minutes=market.bar_minutes), row.close) for t in opens])
        frames.append(pd.DataFrame({"timestamp": opens, "bar_end": ends}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "bar_end"])


def attach_calendar(bars, market):
    grid = expected_grid(bars.timestamp.iloc[0], bars.timestamp.iloc[-1], market)
    # Daily CSVs can use midnight UTC or local midnight as the session date.
    bars = bars.copy()
    if market.calendar == "XNYS" and market.bar_minutes == 1440:
        lookup = {str(t.date()): t for t in grid.timestamp}
        bars["timestamp"] = [lookup.get(str(t.date()), t) for t in bars.timestamp]
    valid = grid.set_index("timestamp")
    if not bars.timestamp.isin(valid.index).all():
        raise ValueError("Candles fall outside the configured exchange session or bar alignment")
    inside = valid.loc[(valid.index >= bars.timestamp.iloc[0]) & (valid.index <= bars.timestamp.iloc[-1])]
    if market.require_regular_bars and not bars.timestamp.equals(pd.Series(inside.index, name="timestamp")):
        raise ValueError("Missing candles within exchange sessions")
    bars["bar_end"] = bars.timestamp.map(valid.bar_end)
    return bars


def resample_bars(bars, source_market: Market, target_minutes: int):
    if target_minutes < source_market.bar_minutes or target_minutes % source_market.bar_minutes:
        raise ValueError("Can only aggregate integer multiples of the source timeframe; never upsample")
    if target_minutes == source_market.bar_minutes:
        return bars.copy()
    if target_minutes > 1440:
        raise ValueError("Maximum supported target timeframe is one day")
    target = source_market.model_copy(update={"bar_minutes": target_minutes})
    grid = expected_grid(bars.timestamp.iloc[0], bars.timestamp.iloc[-1], target)
    source = attach_calendar(bars, source_market)
    # Assign both actual and expected source bars to session-anchored target buckets.
    # Matching timestamps/counts rejects partial buckets without fabricating candles.
    actual = pd.merge_asof(
        source.drop(columns="bar_end"),
        grid.rename(columns={"timestamp": "bucket"}),
        left_on="timestamp",
        right_on="bucket",
    )
    expected = expected_grid(source.timestamp.iloc[0], source.timestamp.iloc[-1], source_market)
    expected = pd.merge_asof(
        expected.drop(columns="bar_end"),
        grid.rename(columns={"timestamp": "bucket"}),
        left_on="timestamp",
        right_on="bucket",
    )
    expected = expected[expected.timestamp < expected.bar_end]
    counts = expected.groupby("bucket").size()
    actual = actual[actual.timestamp < actual.bar_end]
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "bar_end": "last",
        "timestamp": "count",
    }
    for name, op in (("dividend", "sum"), ("split", "prod"), ("borrow_available", "first")):
        if name in actual:
            agg[name] = op
    result = actual.groupby("bucket").agg(agg)
    result = result[result.timestamp == counts.reindex(result.index)]
    # expected_grid includes the full first/last stock session, but crypto grids
    # need an explicit duration check at truncated edges.
    complete_count = np.ceil(
        (result.bar_end - result.index) / pd.Timedelta(minutes=source_market.bar_minutes)
    ).astype(int)
    result = result[result.timestamp == complete_count]
    result = result.drop(columns="timestamp").reset_index().rename(columns={"bucket": "timestamp"})
    if result.empty:
        raise ValueError("No complete target bars")
    return result


def trading_periods_per_year(market):
    return (
        365 * 1440 / market.bar_minutes
        if market.calendar == "24/7"
        else 252 * (1 if market.bar_minutes == 1440 else np.ceil(390 / market.bar_minutes))
    )
