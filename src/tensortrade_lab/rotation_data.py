"""Immutable research snapshots for the registered ETF rotation universe."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .artifacts import digest, write_json
from .sources import download_yahoo


def acquire(spec_path):
    import yfinance as yf

    spec = yaml.safe_load(Path(spec_path).read_text())
    root = Path(spec["data_directory"])
    root.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(root / ".yfinance-cache"))

    def fetch(symbol):
        path = root / f"{symbol}.parquet"
        if path.exists():
            meta = json.loads(path.with_suffix(".metadata.json").read_text())
            if meta["sha256"] != digest(path):
                raise ValueError(f"Changed snapshot: {symbol}")
            return symbol, meta["quality"]["rows"]
        meta = download_yahoo(symbol, path, start=spec["download_start"], end=spec["download_end"])
        return symbol, meta["quality"]["rows"]

    # Network I/O is bounded independently of the CPU research pool.
    # Initialize Yahoo's shared cookie/timezone database before concurrent reads.
    print(f"Downloaded {fetch(spec['benchmark'])}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, s) for s in spec["universe"]]):
            print(f"Downloaded {future.result()}", flush=True)
    path = root / "cash_yield.parquet"
    if not path.exists():
        frame = yf.Ticker("^IRX").history(
            start=spec["download_start"], end=spec["download_end"], auto_adjust=False, raise_errors=True
        )
        rate = frame.Close.astype(float) / 100
        if frame.empty or not np.isfinite(rate).all() or ((rate < -0.05) | (rate > 0.5)).any():
            raise ValueError("Invalid bill-yield history")
        pd.DataFrame(
            {"date": frame.index.strftime("%Y-%m-%d"), "discount_yield": rate.to_numpy()}
        ).to_parquet(path, index=False)
        write_json(
            path.with_suffix(".metadata.json"),
            {
                "source": "yfinance",
                "symbol": "^IRX",
                "sha256": digest(path),
                "request": {"start": spec["download_start"], "end": spec["download_end"]},
                "note": "13-week bank discount yield; approximate 91-day conversion, lagged before accrual.",
            },
        )
    return root
