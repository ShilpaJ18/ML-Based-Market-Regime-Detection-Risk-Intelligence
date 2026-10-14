"""End-to-end pipeline: download -> features -> regime models -> risk -> (optional) MySQL.

    result = run_pipeline(["NVDA", "AAPL"], years=5)
    result.tables()          # dict of DataFrames, one per MySQL table
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config, data_loader, features as feat, regime_models as rm, risk_engine as risk

RESULT_FILE = config.PROCESSED_DIR / "pipeline_result.pkl"


class PipelineError(Exception):
    """Raised when the pipeline cannot produce any result (message is user-friendly)."""


@dataclass
class PipelineResult:
    prices: pd.DataFrame
    features: pd.DataFrame
    regime_predictions: pd.DataFrame
    regime_stats: pd.DataFrame
    regime_durations: pd.DataFrame
    regime_transitions: pd.DataFrame
    risk_metrics: pd.DataFrame
    regime_risk_metrics: pd.DataFrame
    high_risk_periods: pd.DataFrame
    model_evaluation: pd.DataFrame
    model_selection: pd.DataFrame
    episodes: pd.DataFrame          # every contiguous regime run (not stored in MySQL)
    pca: pd.DataFrame               # ticker, trade_date, pc1, pc2
    pca_variance: pd.DataFrame      # ticker, pc1_var, pc2_var
    meta: dict = field(default_factory=dict)

    def tables(self) -> dict:
        return {name: getattr(self, name) for name in
                ["prices", "features", "regime_predictions", "regime_stats", "regime_durations",
                 "regime_transitions", "risk_metrics", "regime_risk_metrics", "high_risk_periods",
                 "model_evaluation", "model_selection"]}

    @property
    def tickers(self) -> list:
        return list(self.meta.get("tickers", []))

    @property
    def n_regimes(self) -> int:
        return int(self.meta.get("n_regimes", config.DEFAULT_N_REGIMES))


def _analyse_ticker(ticker: str, px: pd.DataFrame, ft: pd.DataFrame, k: int, train_frac: float) -> dict:
    """Fit both models for one ticker and compute all derived tables."""
    ft = ft.sort_values("trade_date").reset_index(drop=True)
    returns = risk.returns_from_prices(px)
    out = {k_: [] for k_ in ["pred", "stats", "dur", "trans", "risk_reg", "eval", "episodes"]}
    split = rm.chronological_split_index(len(ft), train_frac)
    train_end = ft["trade_date"].iloc[split - 1]
    fitted_ref = None

    for model_name in config.MODEL_NAMES:
        fitted = rm.fit_regime_model(ft, model_name, k, train_frac)
        fitted_ref = fitted_ref or fitted
        labels = fitted.predict(ft)
        pred = pd.DataFrame({
            "ticker": ticker, "trade_date": ft["trade_date"], "model_name": model_name,
            "n_clusters": k, "regime_id": labels,
            "regime_name": [config.regime_names(k)[i] for i in labels],
            "dataset_split": np.where(np.arange(len(ft)) < split, "train", "test"),
        })
        out["pred"].append(pred)

        stats = rm.cluster_statistics(ft, pred)
        stats.insert(0, "model_name", model_name)
        stats.insert(0, "ticker", ticker)
        out["stats"].append(stats)

        episodes = rm.compute_episodes(pred)
        ep_store = episodes.copy()
        ep_store.insert(0, "model_name", model_name)
        ep_store.insert(0, "ticker", ticker)
        out["episodes"].append(ep_store)
        dur = rm.duration_summary(episodes)
        dur.insert(0, "model_name", model_name)
        dur.insert(0, "ticker", ticker)
        out["dur"].append(dur)

        out["trans"].append(rm.transitions_long(ticker, model_name, pred, k))
        out["risk_reg"].append(risk.risk_by_regime(ticker, model_name, returns, pred))

        m = rm.metrics_for(fitted, ft)
        ari_mean, ari_min = rm.stability_ari(ft, model_name, k, train_frac, labels)
        out["eval"].append({
            "ticker": ticker, "model_name": model_name, "n_clusters": k,
            "train_rows": split, "test_rows": len(ft) - split, "train_end_date": train_end,
            **m, "stability_ari_mean": ari_mean, "stability_ari_min": ari_min,
            "train_test_tv_distance": rm.distribution_tv_distance(labels[:split], labels[split:], k),
        })

    coords, var = rm.pca_projection(ft, fitted_ref)
    coords.insert(0, "ticker", ticker)
    out["pca"] = coords
    out["pca_var"] = {"ticker": ticker, "pc1_var": float(var[0]), "pc2_var": float(var[1])}
    out["selection"] = rm.selection_sweep(ticker, ft, train_frac)
    out["high_risk"] = risk.high_risk_periods(ticker, returns)
    return out


def run_pipeline(
    tickers=None,
    years: int = config.DEFAULT_YEARS,
    n_regimes: int = config.DEFAULT_N_REGIMES,
    train_frac: float = config.DEFAULT_TRAIN_FRAC,
    prices: Optional[pd.DataFrame] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> PipelineResult:
    """Run everything. Pass `prices` to retrain on already downloaded data (no download)."""
    if not 2 <= n_regimes <= 6:
        raise PipelineError("Number of regimes must be between 2 and 6.")
    if not 0.5 <= train_frac <= 0.9:
        raise PipelineError("Training fraction must be between 0.5 and 0.9.")
    say = progress or (lambda msg, frac: None)
    errors, warnings = {}, []

    if prices is None:
        tickers = tickers or config.DEFAULT_TICKERS
        prices, errors, warnings = data_loader.load_prices(tickers, years, progress=say)
    else:
        prices, warns = data_loader.clean_prices(prices)
        warnings.extend(warns)
        if tickers:
            valid, _ = data_loader.parse_tickers(tickers)
            prices = prices[prices["ticker"].isin(valid)]
    if prices.empty:
        detail = "; ".join(f"{t}: {m}" for t, m in errors.items()) or "no data"
        raise PipelineError(f"No price data could be loaded ({detail}).")

    say("Engineering features", 0.1)
    features_all = feat.build_features(prices)

    parts = {k_: [] for k_ in ["pred", "stats", "dur", "trans", "risk_reg", "eval", "episodes",
                               "pca", "selection", "high_risk", "pca_var"]}
    usable = []
    tickers_in_data = list(prices["ticker"].unique())
    for i, ticker in enumerate(tickers_in_data):
        say(f"Fitting K-Means & GMM for {ticker}", 0.15 + 0.7 * i / max(len(tickers_in_data), 1))
        px = prices[prices["ticker"] == ticker]
        ft = features_all[features_all["ticker"] == ticker]
        if len(ft) < config.MIN_MODEL_ROWS:
            errors[ticker] = f"Only {len(ft)} usable rows after feature warm-up (need {config.MIN_MODEL_ROWS})."
            continue
        try:
            res = _analyse_ticker(ticker, px, ft, n_regimes, train_frac)
        except Exception as exc:
            errors[ticker] = f"Modelling failed: {type(exc).__name__}: {exc}"
            continue
        usable.append(ticker)
        for key in ["pred", "stats", "dur", "trans", "risk_reg", "episodes"]:
            parts[key].extend(res[key])
        parts["eval"].extend(res["eval"])
        for key in ["pca", "selection", "high_risk"]:
            parts[key].append(res[key])
        parts["pca_var"].append(res["pca_var"])

    if not usable:
        detail = "; ".join(f"{t}: {m}" for t, m in errors.items())
        raise PipelineError(f"No ticker could be modelled ({detail}).")

    say("Computing risk metrics", 0.9)
    prices = prices[prices["ticker"].isin(usable)].reset_index(drop=True)
    features_all = features_all[features_all["ticker"].isin(usable)].reset_index(drop=True)

    def cat(frames):
        frames = [f for f in frames if f is not None and len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    result = PipelineResult(
        prices=prices, features=features_all,
        regime_predictions=cat(parts["pred"]), regime_stats=cat(parts["stats"]),
        regime_durations=cat(parts["dur"]), regime_transitions=cat(parts["trans"]),
        risk_metrics=risk.risk_by_ticker(prices), regime_risk_metrics=cat(parts["risk_reg"]),
        high_risk_periods=cat(parts["high_risk"]), model_evaluation=pd.DataFrame(parts["eval"]),
        model_selection=cat(parts["selection"]), episodes=cat(parts["episodes"]),
        pca=cat(parts["pca"]), pca_variance=pd.DataFrame(parts["pca_var"]),
        meta={
            "tickers": usable, "years": years, "n_regimes": n_regimes, "train_frac": train_frac,
            "data_start": prices["trade_date"].min(), "data_end": prices["trade_date"].max(),
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "errors": errors, "warnings": warnings, "db_saved": False, "db_message": "Not saved to MySQL.",
        },
    )
    say("Done", 1.0)
    return result


def save_result_to_db(result: PipelineResult) -> str:
    """Save to MySQL. Updates result.meta; never raises (returns a status message)."""
    from . import database
    try:
        msg = database.save_result(result.tables())
        result.meta.update(db_saved=True, db_message=msg)
    except database.DatabaseError as exc:
        result.meta.update(db_saved=False, db_message=str(exc))
    return result.meta["db_message"]


def save_local(result: PipelineResult) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "wb") as fh:
        pickle.dump(result, fh)


def load_local() -> Optional[PipelineResult]:
    """Load the last saved result (None if missing or unreadable)."""
    if not RESULT_FILE.exists():
        return None
    try:
        with open(RESULT_FILE, "rb") as fh:
            obj = pickle.load(fh)
        return obj if isinstance(obj, PipelineResult) else None
    except Exception:
        return None
