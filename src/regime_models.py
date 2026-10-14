"""K-Means and Gaussian Mixture regime models, evaluation and regime statistics.

Leakage control
---------------
* The data is split chronologically: the first `train_frac` of rows is the
  training set, the remaining rows are the test set (no shuffling).
* The StandardScaler and the clustering model are fitted on the training rows only.
  Test rows are only *transformed / assigned* with the already fitted objects.
* All engineered features use trailing windows only.

Regimes are re-numbered by the mean volatility of their cluster centre
(0 = lowest volatility ... k-1 = highest) so labels are comparable across runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config

VOL_IDX = config.MODEL_FEATURES.index("volatility_20d")
STATS_COLUMNS = ["mean_" + f for f in config.MODEL_FEATURES]


# --------------------------------------------------------------------------- fitting
def make_model(model_name: str, k: int, seed: int = config.RANDOM_STATE):
    if model_name == "KMeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed)
    if model_name == "GMM":
        return GaussianMixture(
            n_components=k, covariance_type="full", n_init=5, reg_covar=1e-4, random_state=seed
        )
    raise ValueError(f"Unknown model '{model_name}'. Use one of {config.MODEL_NAMES}.")


def _centres(model) -> np.ndarray:
    return model.cluster_centers_ if hasattr(model, "cluster_centers_") else model.means_


@dataclass
class FittedRegimeModel:
    model_name: str
    k: int
    scaler: StandardScaler
    model: object
    rank: np.ndarray        # raw cluster id -> ordered regime id (0 = calmest)
    split_index: int

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[config.MODEL_FEATURES].values)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.rank[self.model.predict(self.transform(df))]


def chronological_split_index(n_rows: int, train_frac: float) -> int:
    idx = int(n_rows * train_frac)
    return max(1, min(n_rows - 1, idx))


def fit_regime_model(df: pd.DataFrame, model_name: str, k: int, train_frac: float,
                     seed: int = config.RANDOM_STATE) -> FittedRegimeModel:
    """Fit scaler + model on the chronological training slice of `df` (sorted by date)."""
    split = chronological_split_index(len(df), train_frac)
    x_train = df[config.MODEL_FEATURES].values[:split]
    scaler = StandardScaler().fit(x_train)
    model = make_model(model_name, k, seed).fit(scaler.transform(x_train))
    centre_vol = _centres(model)[:, VOL_IDX] * scaler.scale_[VOL_IDX] + scaler.mean_[VOL_IDX]
    order = np.argsort(centre_vol)
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    return FittedRegimeModel(model_name, k, scaler, model, rank, split)


# ------------------------------------------------------------------------ evaluation
def within_cluster_ss(x: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in np.unique(labels):
        pts = x[labels == c]
        total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total


def _safe(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return np.nan


def metrics_for(fitted: FittedRegimeModel, df: pd.DataFrame) -> dict:
    """Silhouette / inertia / BIC / AIC etc. All scores are computed in scaled feature space."""
    s = fitted.split_index
    xs = fitted.transform(df)
    labels = fitted.predict(df)
    xtr, xte, ltr, lte = xs[:s], xs[s:], labels[:s], labels[s:]
    is_gmm = fitted.model_name == "GMM"
    return {
        "silhouette_train": _safe(silhouette_score, xtr, ltr) if len(np.unique(ltr)) > 1 else np.nan,
        "silhouette_test": _safe(silhouette_score, xte, lte) if len(np.unique(lte)) > 1 and len(xte) > 2 else np.nan,
        "inertia_train": within_cluster_ss(xtr, ltr),
        "davies_bouldin_train": _safe(davies_bouldin_score, xtr, ltr) if len(np.unique(ltr)) > 1 else np.nan,
        "bic": _safe(fitted.model.bic, xtr) if is_gmm else np.nan,
        "aic": _safe(fitted.model.aic, xtr) if is_gmm else np.nan,
        "test_avg_loglik": _safe(fitted.model.score, xte) if is_gmm else np.nan,
    }


def distribution_tv_distance(labels_train: np.ndarray, labels_test: np.ndarray, k: int) -> float:
    """Total-variation distance between regime frequencies in train vs test (0 = identical)."""
    p = np.bincount(labels_train, minlength=k) / max(len(labels_train), 1)
    q = np.bincount(labels_test, minlength=k) / max(len(labels_test), 1)
    return float(0.5 * np.abs(p - q).sum())


def stability_ari(df: pd.DataFrame, model_name: str, k: int, train_frac: float,
                  ref_labels: np.ndarray, seeds=config.STABILITY_SEEDS) -> tuple:
    """Adjusted Rand Index between the reference labelling and labellings from other random seeds."""
    scores = []
    for sd in seeds:
        if sd == config.RANDOM_STATE:
            continue
        try:
            lab = fit_regime_model(df, model_name, k, train_frac, seed=sd).predict(df)
            scores.append(adjusted_rand_score(ref_labels, lab))
        except Exception:
            continue
    if not scores:
        return np.nan, np.nan
    return float(np.mean(scores)), float(np.min(scores))


def selection_sweep(ticker: str, df: pd.DataFrame, train_frac: float, k_values=config.K_RANGE) -> pd.DataFrame:
    """Fit both models for every candidate k (used for model selection plots)."""
    rows = []
    for model_name in config.MODEL_NAMES:
        for k in k_values:
            try:
                m = metrics_for(fit_regime_model(df, model_name, k, train_frac), df)
            except Exception:
                continue
            rows.append({"ticker": ticker, "model_name": model_name, "n_clusters": k,
                         **{c: m[c] for c in ["silhouette_train", "silhouette_test", "inertia_train", "bic", "aic"]}})
    cols = ["ticker", "model_name", "n_clusters", "silhouette_train", "silhouette_test",
            "inertia_train", "bic", "aic"]
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------ regime descriptive stats
def compute_episodes(pred: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of the same regime. pred needs trade_date, regime_id, regime_name."""
    if pred.empty:
        return pd.DataFrame(columns=["regime_id", "regime_name", "start", "end", "days"])
    p = pred.sort_values("trade_date").reset_index(drop=True)
    run_id = (p["regime_id"] != p["regime_id"].shift()).cumsum()
    ep = p.groupby(run_id).agg(
        regime_id=("regime_id", "first"), regime_name=("regime_name", "first"),
        start=("trade_date", "first"), end=("trade_date", "last"), days=("trade_date", "size"),
    ).reset_index(drop=True)
    return ep


def duration_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=["regime_id", "regime_name", "n_episodes", "mean_duration",
                                     "median_duration", "max_duration"])
    g = episodes.groupby(["regime_id", "regime_name"])["days"]
    out = g.agg(n_episodes="size", mean_duration="mean", median_duration="median", max_duration="max")
    return out.reset_index()


def frequency_table(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame(columns=["regime_id", "regime_name", "n_days", "frequency"])
    out = pred.groupby(["regime_id", "regime_name"]).size().rename("n_days").reset_index()
    out["frequency"] = out["n_days"] / out["n_days"].sum()
    return out


def transition_matrices(pred: pd.DataFrame, k: int) -> tuple:
    """Returns (counts, probabilities) DataFrames indexed/columned by regime name."""
    names = config.regime_names(k)
    counts = np.zeros((k, k), dtype=int)
    ids = pred.sort_values("trade_date")["regime_id"].values
    for a, b in zip(ids[:-1], ids[1:]):
        counts[int(a), int(b)] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sum, out=np.zeros((k, k)), where=row_sum > 0)
    return (pd.DataFrame(counts, index=names, columns=names),
            pd.DataFrame(probs, index=names, columns=names))


def transitions_long(ticker: str, model_name: str, pred: pd.DataFrame, k: int) -> pd.DataFrame:
    names = config.regime_names(k)
    counts, probs = transition_matrices(pred, k)
    rows = []
    for i in range(k):
        for j in range(k):
            rows.append({"ticker": ticker, "model_name": model_name, "from_regime_id": i,
                         "to_regime_id": j, "from_regime_name": names[i], "to_regime_name": names[j],
                         "transition_count": int(counts.iloc[i, j]), "probability": float(probs.iloc[i, j])})
    return pd.DataFrame(rows)


def cluster_statistics(features: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Mean of every model feature per regime. features/pred need ticker?, trade_date."""
    m = features[["trade_date"] + config.MODEL_FEATURES].merge(
        pred[["trade_date", "regime_id", "regime_name"]], on="trade_date", how="inner")
    if m.empty:
        return pd.DataFrame(columns=["regime_id", "regime_name", "n_days", "frequency"] + STATS_COLUMNS)
    g = m.groupby(["regime_id", "regime_name"])
    out = g[config.MODEL_FEATURES].mean().add_prefix("mean_")
    out.insert(0, "n_days", g.size())
    out.insert(1, "frequency", out["n_days"] / out["n_days"].sum())
    return out.reset_index()


def pca_projection(df: pd.DataFrame, fitted: FittedRegimeModel) -> tuple:
    """2-D PCA of the scaled features (PCA fitted on the training slice). Returns (coords, variance)."""
    xs = fitted.transform(df)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE).fit(xs[: fitted.split_index])
    z = pca.transform(xs)
    coords = pd.DataFrame({"trade_date": df["trade_date"].values, "pc1": z[:, 0], "pc2": z[:, 1]})
    return coords, pca.explained_variance_ratio_
