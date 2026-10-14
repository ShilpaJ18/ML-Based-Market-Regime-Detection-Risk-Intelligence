"""Feature engineering. All features use only past/current data (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

ROLL_VOL_WINDOW = 20
DD_WINDOW = 60


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi.where(avg_gain.notna())


def rolling_max_drawdown(close: pd.Series, window: int = DD_WINDOW) -> pd.Series:
    """Worst drawdown-from-rolling-peak observed in the trailing `window` days (<= 0)."""
    peak = close.rolling(window, min_periods=window).max()
    dd = close / peak - 1.0
    return dd.rolling(window, min_periods=window).min()


def build_features_one(px: pd.DataFrame) -> pd.DataFrame:
    """px: one ticker, columns trade_date/open/high/low/close/volume."""
    px = px.sort_values("trade_date").reset_index(drop=True)
    close, volume = px["close"], px["volume"]
    out = pd.DataFrame({"trade_date": px["trade_date"]})
    out["daily_return"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)
    out["volatility_20d"] = out["daily_return"].rolling(ROLL_VOL_WINDOW).std() * np.sqrt(config.TRADING_DAYS)
    out["ma_20"] = close.rolling(20).mean()
    out["ma_50"] = close.rolling(50).mean()
    out["ma20_ratio"] = close / out["ma_20"] - 1.0
    out["ma50_ratio"] = close / out["ma_50"] - 1.0
    out["volume_change"] = volume / volume.rolling(20).mean().replace(0.0, np.nan) - 1.0
    out["rsi_14"] = compute_rsi(close, 14)
    out["max_drawdown_60d"] = rolling_max_drawdown(close, DD_WINDOW)
    out["drawdown"] = close / close.cummax() - 1.0
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=config.MODEL_FEATURES).reset_index(drop=True)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Long-format features for all tickers: ticker, trade_date, FEATURE_STORE_COLUMNS."""
    frames = []
    for ticker, g in prices.groupby("ticker", sort=False):
        f = build_features_one(g)
        f.insert(0, "ticker", ticker)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["ticker", "trade_date"] + config.FEATURE_STORE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[["ticker", "trade_date"] + config.FEATURE_STORE_COLUMNS]
