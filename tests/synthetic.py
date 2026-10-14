"""SYNTHETIC data generator - used ONLY by tests / offline smoke checks, never by the app.

Creates a regime-switching geometric random walk so the pipeline can be verified
without internet access. These numbers are fake and must never be shown as market data.
"""
import numpy as np
import pandas as pd


def synthetic_prices(tickers=("NVDA", "AAPL", "MSFT", "AMD", "QQQ"), years=5, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(252 * years))
    n = len(dates)
    # shared 3-state Markov chain: calm / normal / stressed
    trans = np.array([[0.97, 0.025, 0.005], [0.03, 0.95, 0.02], [0.02, 0.08, 0.90]])
    vol = np.array([0.008, 0.015, 0.032])
    drift = np.array([0.0007, 0.0003, -0.0008])
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        state[i] = rng.choice(3, p=trans[state[i - 1]])
    frames = []
    for j, t in enumerate(tickers):
        beta = 0.8 + 0.25 * j
        r = drift[state] * beta + vol[state] * beta * rng.standard_normal(n)
        close = 100 * np.exp(np.cumsum(r))
        openp = close * (1 + 0.002 * rng.standard_normal(n))
        high = np.maximum(openp, close) * (1 + np.abs(0.004 * rng.standard_normal(n)))
        low = np.minimum(openp, close) * (1 - np.abs(0.004 * rng.standard_normal(n)))
        volume = rng.lognormal(16, 0.3, n) * (1 + 0.5 * (state == 2))
        frames.append(pd.DataFrame({"ticker": t, "trade_date": dates, "open": openp, "high": high,
                                    "low": low, "close": close, "volume": volume}))
    return pd.concat(frames, ignore_index=True)
