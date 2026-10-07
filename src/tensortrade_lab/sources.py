"""Local archive and Yahoo imports with immutable provenance and explicit gap reports."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .artifacts import digest, write_json
from .config import Market
from .data import validate_bars
from .market_data import expected_grid, trading_periods_per_year


def normalize(frame, timezone="UTC"):
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        raise TypeError("Import one ticker at a time")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index.name = "timestamp"
        frame = frame.reset_index()
    frame = frame.rename(
        columns={
            "date": "timestamp",
            "datetime": "timestamp",
            "time": "timestamp",
            "dividends": "dividend",
            "stock splits": "split",
        }
    )
    if "timestamp" not in frame:
        raise ValueError("No date/timestamp column or datetime index")
    times = pd.to_datetime(frame.timestamp, errors="raise")
    if times.dt.tz is None:
        times = times.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    frame["timestamp"] = times.dt.tz_convert("UTC")
    return frame.sort_values("timestamp").reset_index(drop=True)


def quality(bars, market):
    grid = expected_grid(bars.timestamp.iloc[0], bars.timestamp.iloc[-1], market)
    grid = grid[(grid.timestamp >= bars.timestamp.iloc[0]) & (grid.timestamp <= bars.timestamp.iloc[-1])]
    missing = grid.loc[~grid.timestamp.isin(bars.timestamp), "timestamp"]
    return {
        "rows": len(bars),
        "start": bars.timestamp.iloc[0].isoformat(),
        "end": bars.timestamp.iloc[-1].isoformat(),
        "missing_bars": len(missing),
        "missing_examples": [x.isoformat() for x in missing.iloc[:10]],
        "expected_bars": len(grid),
        "coverage": (len(grid) - len(missing)) / max(1, len(grid)),
        "unexpected_timestamps": int((~bars.timestamp.isin(grid.timestamp)).sum()),
    }


def save_dataset(frame, destination, market, provenance, timezone="UTC", completed_only=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bars = validate_bars(
        normalize(frame, timezone), market.model_copy(update={"require_regular_bars": False})
    )
    if completed_only:
        bars = bars[bars.bar_end <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)
    if len(bars) < 100:
        raise ValueError("Fewer than 100 completed candles")
    metadata = {
        "schema_version": 1,
        "imported_at": datetime.now(UTC).isoformat(),
        "market": market.model_dump(),
        "quality": quality(bars, market),
        **provenance,
    }
    if destination.exists():
        raise ValueError(f"Dataset already exists; use a new name to preserve snapshots: {destination}")
    bars.to_parquet(destination, index=False)
    metadata["sha256"] = digest(destination)
    write_json(destination.with_suffix(".metadata.json"), metadata)
    return metadata


def import_local(source, destination, market, timezone="UTC", ticker=None, interval=None):
    source = Path(source).resolve()
    if source.suffix in (".db", ".sqlite"):
        if not ticker or not interval:
            raise ValueError("SQLite price_data import requires ticker and interval")
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as connection:
            frame = pd.read_sql_query(
                "SELECT date AS timestamp,open,high,low,close,volume FROM price_data WHERE ticker=? AND interval=? ORDER BY date",
                connection,
                params=(ticker, interval),
            )
    else:
        frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    return save_dataset(
        frame,
        destination,
        market,
        {
            "source": "local_archive",
            "source_path": str(source),
            "source_sha256": digest(source),
            "source_ticker": ticker,
            "adjustment_note": "Price adjustment and timezone supplied by importer; verify against original vendor.",
        },
        timezone,
    )


def download_yahoo(ticker, destination, interval="1d", start=None, end=None, period=None, calendar="XNYS"):
    import yfinance as yf

    mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "1d": 1440}
    if interval not in mapping:
        raise ValueError(f"Supported intervals: {list(mapping)}")
    if start and period:
        raise ValueError("Choose start/end or period")
    if (
        interval == "15m"
        and start
        and pd.Timestamp(start, tz="UTC") < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60)
    ):
        raise ValueError(
            "Yahoo 15-minute history is limited to roughly the last 60 days; use a local archive for older history"
        )
    cache = Path(destination).parent / ".yfinance-cache"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    options = (
        {"start": start, "end": end}
        if start
        else {"period": period or ("60d" if mapping[interval] < 1440 else "10y")}
    )
    frame = yf.Ticker(ticker).history(
        interval=interval,
        auto_adjust=False,
        actions=True,
        prepost=False,
        repair=False,
        raise_errors=True,
        **options,
    )
    if frame.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}; check interval, coverage and connectivity")
    # Yahoo OHLC already incorporates splits; dividends remain explicit cash events.
    market = Market(
        symbol="".join(c for c in ticker if c.isalnum()).upper(),
        bar_minutes=mapping[interval],
        calendar=calendar,
        price_adjustment="split_adjusted",
        require_regular_bars=False,
    )
    market = market.model_copy(update={"periods_per_year": int(trading_periods_per_year(market))})
    return save_dataset(
        frame,
        destination,
        market,
        {
            "source": "yfinance",
            "source_ticker": ticker,
            "request": {"interval": interval, **options},
            "auto_adjust": False,
            "note": "Research data; revised history and provider coverage limits may apply.",
        },
        completed_only=True,
    )


def catalog(directory="data"):
    records = []
    for sidecar in sorted(Path(directory).rglob("*.metadata.json")):
        record = json.loads(sidecar.read_text())
        if "quality" in record:
            records.append({"path": str(sidecar.with_suffix("").with_suffix(".parquet").resolve()), **record})
    return records
