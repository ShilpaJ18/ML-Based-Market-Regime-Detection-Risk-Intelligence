"""Data-backed insights: every sentence is computed from the pipeline result (no canned claims)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_insights(result, model_name: str, top_n_worst: int = 20) -> list:
    out = []
    k = result.n_regimes
    names = config.regime_names(k)
    hi_name, lo_name = names[-1], names[0]
    pred = result.regime_predictions
    pred = pred[pred["model_name"] == model_name]

    for t in result.tickers:
        p = pred[pred["ticker"] == t]
        f = result.features[result.features["ticker"] == t][["trade_date", "daily_return"]]
        rr = result.regime_risk_metrics
        rr = rr[(rr["ticker"] == t) & (rr["model_name"] == model_name)].set_index("regime_name")
        if p.empty or hi_name not in rr.index or lo_name not in rr.index:
            continue
        share_days = (p["regime_name"] == hi_name).mean()
        worst = f.nsmallest(top_n_worst, "daily_return").merge(p[["trade_date", "regime_name"]], on="trade_date")
        share_worst = (worst["regime_name"] == hi_name).mean() if len(worst) else np.nan
        vol_ratio = rr.loc[hi_name, "annualized_volatility"] / rr.loc[lo_name, "annualized_volatility"]
        es_ratio = rr.loc[hi_name, "es_95"] / rr.loc[lo_name, "es_95"] if rr.loc[lo_name, "es_95"] > 0 else np.nan
        out.append(
            f"**{t}** ({config.MODEL_LABELS[model_name]}): the {hi_name} regime covers {share_days:.0%} of days "
            f"but contains {share_worst:.0%} of the {len(worst)} worst daily returns; its annualised volatility is "
            f"{vol_ratio:.1f}x the {lo_name} regime and its 95% Expected Shortfall is {es_ratio:.1f}x."
        )

    rm_ = result.risk_metrics
    if len(rm_) > 1:
        a = rm_.loc[rm_["annualized_volatility"].idxmax()]
        b = rm_.loc[rm_["max_drawdown"].idxmin()]
        c = rm_.loc[rm_["es_99"].idxmax()]
        out.append(f"Highest annualised volatility: **{a['ticker']}** ({a['annualized_volatility']:.1%}). "
                   f"Deepest max drawdown: **{b['ticker']}** ({b['max_drawdown']:.1%}). "
                   f"Largest 99% Expected Shortfall: **{c['ticker']}** ({c['es_99']:.2%} in one day).")

    ev = result.model_evaluation
    if len(ev):
        best = ev.groupby("model_name")[["silhouette_test", "stability_ari_mean", "train_test_tv_distance"]].mean()
        if len(best) == 2:
            win = best["silhouette_test"].idxmax()
            out.append(
                f"Averaged over tickers, **{config.MODEL_LABELS[win]}** has the higher out-of-sample (test) silhouette "
                f"({best.loc[win, 'silhouette_test']:.2f} vs {best['silhouette_test'].drop(win).iloc[0]:.2f}). "
                f"Mean seed stability (ARI): K-Means {best.loc['KMeans', 'stability_ari_mean']:.2f}, "
                f"GMM {best.loc['GMM', 'stability_ari_mean']:.2f}."
            )
        drift = ev.groupby("model_name")["train_test_tv_distance"].mean()
        out.append("Average train-vs-test shift in regime frequencies (total-variation distance): "
                   + ", ".join(f"{config.MODEL_LABELS[m]} {v:.2f}" for m, v in drift.items())
                   + ". Large values mean the test period spent its time in different regimes than the training period.")
    return out
