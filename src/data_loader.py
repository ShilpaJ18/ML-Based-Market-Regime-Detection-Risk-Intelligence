"""Download and clean daily OHLCV data from Yahoo Finance (via yfinance).

Public API
----------
parse_tickers(text)               -> (valid_tickers, invalid_tickers)
load_prices(tickers, years)       -> (prices_long, errors, warnings)

`prices_long` has columns: ticker, trade_date, open, high, low, close, volume.
Prices are split/dividend adjusted (auto_adjust=True).
If a live download fails for a ticker, the last successfully cached CSV copy
(data/raw/prices_raw.csv) is used for that ticker and a warning is returned.
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd

from . import config

PRICE_COLUMNS = ["ticker", "trade_date", "open", "high", "low", "close", "volume"]
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")
RAW_CACHE = config.RAW_DIR / "prices_raw.csv"


def parse_tickers(text) -> tuple:
    """Split a comma/space separated string (or list) into valid and invalid tickers."""
    if isinstance(text, str):
        items = re.split(r"[,\s;]+", text)
    else:
        items = list(text)
    valid, invalid = [], []
    for item in items:
        t = str(item).strip().upper()
        if not t:
            continue
        if _TICKER_RE.match(t):
            if t not in valid:
                valid.append(t)
        elif t not in invalid:
            invalid.append(t)
    return valid, invalid


def _download_one(ticker: str, start: date, end: date, retries: int = 3) -> pd.DataFrame:
    """Download one ticker with retries. Raises RuntimeError on failure."""
    import yfinance as yf  # imported lazily so the rest of the package works offline

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(ticker).history(
                start=start.isoformat(), end=end.isoformat(), auto_adjust=True, actions=False
            )
            if raw is not None and not raw.empty:
                return raw
            last_error = "empty response (invalid ticker, delisted, or no data in range)"
        except Exception as exc:  # network errors, rate limits, JSON errors ...
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * attempt)
    raise RuntimeError(last_error or "unknown error")


def _standardise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df["trade_date"] = idx.normalize()
    df["ticker"] = ticker
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise RuntimeError(f"response is missing columns: {missing}")
    return df[PRICE_COLUMNS].reset_index(drop=True)


def clean_prices(df: pd.DataFrame) -> tuple:
    """Remove duplicates / missing / non-positive prices. Returns (clean_df, list_of_warnings)."""
    warns = []
    out = df.copy()
    n0 = len(out)
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1) & (out["volume"] >= 0)]
    out = out.drop_duplicates(subset=["ticker", "trade_date"], keep="last")
    out = out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = out[col].astype(float)
    dropped = n0 - len(out)
    if dropped:
        warns.append(f"Removed {dropped} rows with missing, duplicate or invalid values.")
    return out, warns


def _read_cache() -> pd.DataFrame:
    if not RAW_CACHE.exists():
        return pd.DataFrame(columns=PRICE_COLUMNS)
    try:
        df = pd.read_csv(RAW_CACHE, parse_dates=["trade_date"])
        return df[PRICE_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=PRICE_COLUMNS)


def _write_cache(df: pd.DataFrame) -> None:
    try:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        old = _read_cache()
        keep = old[~old["ticker"].isin(df["ticker"].unique())]
        pd.concat([keep, df], ignore_index=True).to_csv(RAW_CACHE, index=False)
    except OSError:
        pass  # caching is best-effort


def load_prices(
    tickers,
    years: int = config.DEFAULT_YEARS,
    progress: Optional[Callable[[str, float], None]] = None,
) -> tuple:
    """Download `years` of daily data for each ticker.

    Returns (prices_long, errors, warnings). `errors` maps ticker -> message for
    tickers that could not be loaded at all.
    """
    years = int(max(config.MIN_YEARS, min(config.MAX_YEARS, years)))
    valid, invalid = parse_tickers(tickers)
    errors, warnings = {}, []
    for t in invalid:
        errors[t] = "Invalid ticker format."
    if not valid:
        return pd.DataFrame(columns=PRICE_COLUMNS), errors or {"-": "No tickers given."}, warnings

    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=int(365.25 * years) + 1)
    cache = _read_cache()
    frames = []
    for i, t in enumerate(valid):
        if progress:
            progress(f"Downloading {t} ({i + 1}/{len(valid)})", i / len(valid))
        try:
            frames.append(_standardise(_download_one(t, start, end), t))
        except Exception as exc:
            cached = cache[cache["ticker"] == t]
            cached = cached[cached["trade_date"] >= pd.Timestamp(start)]
            if len(cached) > 0:
                frames.append(cached)
                warnings.append(
                    f"{t}: live download failed ({exc}); using cached copy "
                    f"ending {cached['trade_date'].max().date()}."
                )
            else:
                errors[t] = f"Download failed: {exc}"

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS), errors, warnings

    prices, clean_warns = clean_prices(pd.concat(frames, ignore_index=True))
    warnings.extend(clean_warns)
    _write_cache(prices)

    for t, g in prices.groupby("ticker"):
        age = (pd.Timestamp.today().normalize() - g["trade_date"].max()).days
        if age > 10:
            warnings.append(f"{t}: latest data point is {age} days old.")
        if len(g) < config.MIN_MODEL_ROWS:
            warnings.append(f"{t}: only {len(g)} rows downloaded - may be too short for modelling.")
    if progress:
        progress("Download complete", 1.0)
    return prices, errors, warnings
