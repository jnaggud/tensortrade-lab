"""Point-in-time ETF context, published only after the peer candle closes."""

import numpy as np
import pandas as pd

from .features import engineer, feature_columns


def economic_index(bars):
    # Inputs here are split-adjusted; dividends are known on their effective date.
    close = bars.close
    returns = (close + bars.get("dividend", 0)) / close.shift(1)
    return returns.fillna(1).cumprod()


def etf_frame(own, peer, config, feature_set="cross"):
    out = engineer(own, config)
    values = own[["timestamp", "bar_end"]].copy()
    price = economic_index(own)
    daily = np.log(price / price.shift(1))
    for n in (20, 60, 120):
        values[f"context_own_return_{n}"] = np.log(price / price.shift(n))
        values[f"context_own_trend_{n}"] = price / price.rolling(n).mean() - 1
    values["context_own_volatility"] = daily.rolling(20).std(ddof=0) * np.sqrt(252)
    values["context_own_downside"] = daily.clip(upper=0).pow(2).rolling(20).mean().pow(0.5) * np.sqrt(252)
    values["context_own_vol_regime"] = values.context_own_volatility / (
        daily.rolling(120).std(ddof=0) * np.sqrt(252)
    ).replace(0, np.nan)
    out = out.merge(values.drop(columns="bar_end"), on="timestamp", validate="one_to_one")
    if feature_set == "cross":
        other = peer[["bar_end"]].copy()
        peer_price = economic_index(peer)
        for n in (20, 60, 120):
            other[f"context_peer_return_{n}"] = np.log(peer_price / peer_price.shift(n))
        other["context_peer_volatility"] = np.log(peer_price / peer_price.shift(1)).rolling(20).std(
            ddof=0
        ) * np.sqrt(252)
        other["peer_available_at"] = peer.bar_end
        out = pd.merge_asof(
            out.sort_values("bar_end"),
            other.sort_values("bar_end"),
            on="bar_end",
            direction="backward",
            tolerance=pd.Timedelta(hours=8),
        )
        for n in (20, 60, 120):
            out[f"context_relative_{n}"] = out[f"context_own_return_{n}"] - out[f"context_peer_return_{n}"]
        if (out.peer_available_at > out.bar_end).any():
            raise ValueError("Peer data would arrive after decision time")
    return out.dropna(subset=feature_columns(out)).reset_index(drop=True)
