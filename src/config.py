"""Central configuration: paths, defaults, feature names, colours.

Every other module imports names from here so that feature names, model names
and column lists stay consistent across the whole project.
"""
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"

DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "AMD", "QQQ"]
DEFAULT_YEARS = 5
MIN_YEARS, MAX_YEARS = 1, 10
MIN_MODEL_ROWS = 250          # minimum usable rows per ticker to fit models

TRADING_DAYS = 252
RANDOM_STATE = 42
DEFAULT_N_REGIMES = 3
K_RANGE = list(range(2, 7))   # candidate numbers of regimes for model selection
DEFAULT_TRAIN_FRAC = 0.70
STABILITY_SEEDS = [0, 1, 2, 3, 4]

MODEL_NAMES = ["KMeans", "GMM"]
MODEL_LABELS = {"KMeans": "K-Means", "GMM": "Gaussian Mixture (GMM)"}

# Features fed to the clustering models (all stationary / scale-free).
MODEL_FEATURES = [
    "daily_return",
    "return_5d",
    "volatility_20d",
    "ma20_ratio",
    "ma50_ratio",
    "volume_change",
    "rsi_14",
    "max_drawdown_60d",
]
# Every engineered column stored in the `features` table (besides ticker/trade_date).
FEATURE_STORE_COLUMNS = [
    "daily_return",
    "return_5d",
    "volatility_20d",
    "ma_20",
    "ma_50",
    "ma20_ratio",
    "ma50_ratio",
    "volume_change",
    "rsi_14",
    "max_drawdown_60d",
    "drawdown",
]
FEATURE_DESCRIPTIONS = {
    "daily_return": "1-day simple return",
    "return_5d": "5-day simple return",
    "volatility_20d": "20-day rolling std of daily returns, annualised",
    "ma20_ratio": "close / 20-day moving average - 1",
    "ma50_ratio": "close / 50-day moving average - 1",
    "volume_change": "volume / 20-day average volume - 1",
    "rsi_14": "14-day RSI (Wilder smoothing)",
    "max_drawdown_60d": "worst drawdown from a rolling peak over 60 days",
}

REGIME_PALETTE = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#b5179e"]
ACCENT = "#3a86ff"

DISCLAIMER = (
    "Regimes are *descriptive* labels of historical market conditions found by "
    "unsupervised clustering. This project does not predict future prices and does "
    "not provide trading signals or investment advice."
)


def regime_names(k: int) -> list:
    """Names for k regimes, ordered from lowest to highest volatility."""
    table = {
        2: ["Calm", "Turbulent"],
        3: ["Calm", "Moderate", "Turbulent"],
        4: ["Calm", "Moderate", "Elevated", "Turbulent"],
        5: ["Very Calm", "Calm", "Moderate", "Elevated", "Turbulent"],
        6: ["Very Calm", "Calm", "Moderate", "Elevated", "Turbulent", "Extreme"],
    }
    return table.get(k, [f"Regime {i}" for i in range(k)])


def regime_color_map(k: int) -> dict:
    """regime name -> colour, evenly spread over the palette (calm=green ... turbulent=red)."""
    idx = np.linspace(0, len(REGIME_PALETTE) - 1, k).round().astype(int)
    return {name: REGIME_PALETTE[i] for name, i in zip(regime_names(k), idx)}
