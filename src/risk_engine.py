"""Risk metrics computed from actual return series.

Conventions
-----------
* Returns are simple daily returns.
* VaR and Expected Shortfall are reported as POSITIVE loss fractions
  (0.03 = a 3% one-day loss), historical (empirical) method, no distribution assumption.
* Annualisation uses 252 trading days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TD = config.TRADING_DAYS


def annualized_volatility(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(TD)) if len(r) > 1 else np.nan


def cumulative_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod() - 1.0


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min()) if len(r.dropna()) else np.nan


def historical_var(r: pd.Series, level: float = 0.95) -> float:
    r = r.dropna()
    return float(-np.quantile(r, 1.0 - level)) if len(r) else np.nan


def expected_shortfall(r: pd.Series, level: float = 0.95) -> float:
    r = r.dropna()
    if not len(r):
        return np.nan
    cutoff = np.quantile(r, 1.0 - level)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else np.nan


def rolling_volatility(r: pd.Series, window: int = 20) -> pd.Series:
    return r.rolling(window).std() * np.sqrt(TD)


def summarize_returns(r: pd.Series) -> dict:
    """Headline risk numbers for one return series (index = dates)."""
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ["n_days", "cumulative_return", "annualized_return",
                                    "annualized_volatility", "max_drawdown", "var_95", "var_99",
                                    "es_95", "es_99", "worst_day", "best_day", "return_to_risk"]}
    cum = float((1.0 + r).prod() - 1.0)
    ann_ret = float((1.0 + cum) ** (TD / n) - 1.0) if cum > -1 else np.nan
    vol = annualized_volatility(r)
    return {
        "n_days": n,
        "cumulative_return": cum,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "max_drawdown": max_drawdown(r),
        "var_95": historical_var(r, 0.95),
        "var_99": historical_var(r, 0.99),
        "es_95": expected_shortfall(r, 0.95),
        "es_99": expected_shortfall(r, 0.99),
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
        "return_to_risk": float(ann_ret / vol) if vol and vol > 0 else np.nan,
    }


def returns_from_prices(prices_one: pd.DataFrame) -> pd.Series:
    s = prices_one.sort_values("trade_date").set_index("trade_date")["close"]
    return s.pct_change().dropna()


def risk_by_ticker(prices: pd.DataFrame) -> pd.DataFrame:
    """Risk table with one row per ticker (column list matches the `risk_metrics` table)."""
    rows = []
    for ticker, g in prices.groupby("ticker"):
        r = returns_from_prices(g)
        if len(r) < 2:
            continue
        s = summarize_returns(r)
        rows.append({"ticker": ticker, "start_date": r.index.min(), "end_date": r.index.max(), **s})
    cols = ["ticker", "start_date", "end_date", "n_days", "cumulative_return", "annualized_return",
            "annualized_volatility", "max_drawdown", "var_95", "var_99", "es_95", "es_99",
            "worst_day", "best_day", "return_to_risk"]
    return pd.DataFrame(rows, columns=cols)


def risk_by_regime(ticker: str, model_name: str, returns: pd.Series, pred: pd.DataFrame) -> pd.DataFrame:
    """Risk inside each regime. `returns` indexed by date; pred has trade_date, regime_id, regime_name.

    max_drawdown here is computed on the compounded returns of that regime's days only.
    """
    p = pred.set_index("trade_date")[["regime_id", "regime_name"]]
    joined = p.join(returns.rename("ret"), how="inner").dropna()
    rows = []
    for (rid, rname), g in joined.groupby(["regime_id", "regime_name"]):
        r = g["ret"]
        rows.append({
            "ticker": ticker, "model_name": model_name, "regime_id": int(rid), "regime_name": rname,
            "n_days": len(r), "mean_daily_return": float(r.mean()),
            "annualized_volatility": annualized_volatility(r), "max_drawdown": max_drawdown(r),
            "var_95": historical_var(r, 0.95), "var_99": historical_var(r, 0.99),
            "es_95": expected_shortfall(r, 0.95), "es_99": expected_shortfall(r, 0.99),
            "worst_day": float(r.min()),
        })
    cols = ["ticker", "model_name", "regime_id", "regime_name", "n_days", "mean_daily_return",
            "annualized_volatility", "max_drawdown", "var_95", "var_99", "es_95", "es_99", "worst_day"]
    return pd.DataFrame(rows, columns=cols)


def high_risk_periods(ticker: str, returns: pd.Series, window: int = 20,
                      quantile: float = 0.90, max_gap: int = 5) -> pd.DataFrame:
    """Periods where rolling annualised volatility is above its `quantile` (full-sample, descriptive).

    Flagged days separated by <= max_gap trading days are merged into one period.
    """
    cols = ["ticker", "period_start", "period_end", "n_days", "peak_volatility", "avg_volatility",
            "vol_threshold", "period_return", "worst_day"]
    rv = rolling_volatility(returns, window).dropna()
    if len(rv) < 10:
        return pd.DataFrame(columns=cols)
    thr = float(rv.quantile(quantile))
    flagged = rv[rv >= thr].index
    if len(flagged) == 0:
        return pd.DataFrame(columns=cols)
    pos = {d: i for i, d in enumerate(rv.index)}
    groups, current = [], [flagged[0]]
    for d in flagged[1:]:
        if pos[d] - pos[current[-1]] <= max_gap + 1:
            current.append(d)
        else:
            groups.append(current)
            current = [d]
    groups.append(current)
    rows = []
    for grp in groups:
        s, e = grp[0], grp[-1]
        seg_r = returns.loc[s:e]
        rows.append({
            "ticker": ticker, "period_start": s, "period_end": e, "n_days": len(seg_r),
            "peak_volatility": float(rv.loc[s:e].max()), "avg_volatility": float(rv.loc[s:e].mean()),
            "vol_threshold": thr, "period_return": float((1 + seg_r).prod() - 1),
            "worst_day": float(seg_r.min()),
        })
    return pd.DataFrame(rows, columns=cols)
