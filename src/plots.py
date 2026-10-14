"""Plotly figure builders shared by the Streamlit dashboards and the Jupyter notebook.

Every function takes real DataFrames/Series and returns a plotly Figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config, risk_engine as risk

UP, DOWN, NEUTRAL = "#2a9d8f", "#e76f51", "#8d99ae"
LINE_COLORS = ["#3a86ff", "#fb5607", "#8338ec", "#06d6a0", "#ffbe0b", "#ef476f", "#118ab2", "#6a994e"]


def style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef0f4")
    return fig


def _vline(fig, x, text, row=None):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color="#555", width=1.2, dash="dash"))
    fig.add_annotation(x=x, y=1.0, xref="x", yref="paper", text=text, showarrow=False,
                       yanchor="bottom", font=dict(size=11, color="#555"))


# ------------------------------------------------------------------------ overview
def price_panel(px_df: pd.DataFrame, ft_df: pd.DataFrame, kind: str = "Candlestick",
                ma20: bool = True, ma50: bool = True, title: str | None = None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.58, 0.20, 0.22])
    if kind == "Candlestick":
        fig.add_trace(go.Candlestick(x=px_df["trade_date"], open=px_df["open"], high=px_df["high"],
                                     low=px_df["low"], close=px_df["close"], name="Price",
                                     increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=px_df["trade_date"], y=px_df["close"], name="Close",
                                 line=dict(color=config.ACCENT, width=1.8)), row=1, col=1)
    if ma20 and "ma_20" in ft_df:
        fig.add_trace(go.Scatter(x=ft_df["trade_date"], y=ft_df["ma_20"], name="MA 20",
                                 line=dict(color="#ffbe0b", width=1.3)), row=1, col=1)
    if ma50 and "ma_50" in ft_df:
        fig.add_trace(go.Scatter(x=ft_df["trade_date"], y=ft_df["ma_50"], name="MA 50",
                                 line=dict(color="#8338ec", width=1.3)), row=1, col=1)
    up = px_df["close"] >= px_df["open"]
    fig.add_trace(go.Bar(x=px_df["trade_date"], y=px_df["volume"], name="Volume", showlegend=False,
                         marker_color=np.where(up, UP, DOWN), opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=ft_df["trade_date"], y=ft_df["rsi_14"], name="RSI 14",
                             line=dict(color="#3a86ff", width=1.4)), row=3, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line_dash="dot", line_color=NEUTRAL, row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return style(fig, title, height=680)


def line_returns(ft_df: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=ft_df["trade_date"], y=ft_df[column],
                           marker_color=np.where(ft_df[column] >= 0, UP, DOWN)))
    fig.update_yaxes(tickformat=".1%")
    return style(fig, title, height=300)


def cumulative_performance(close_wide: pd.DataFrame) -> go.Figure:
    cum = close_wide / close_wide.bfill().iloc[0] - 1.0
    fig = go.Figure()
    for i, c in enumerate(cum.columns):
        fig.add_trace(go.Scatter(x=cum.index, y=cum[c], name=c, line=dict(color=LINE_COLORS[i % 8], width=1.8)))
    fig.update_yaxes(tickformat=".0%", title_text="Cumulative return")
    return style(fig, "Cumulative performance (from start of selected range)", height=400)


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    r = returns.dropna()
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    tbl = monthly.unstack()
    tbl.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in tbl.columns]
    tbl.index.name = "Year"
    return tbl


def monthly_heatmap(returns: pd.Series, title: str = "Monthly returns") -> go.Figure:
    tbl = monthly_returns_table(returns)
    z = tbl.values * 100
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(z=z, x=list(tbl.columns), y=[str(y) for y in tbl.index],
                               colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim,
                               text=np.where(np.isnan(z), "", np.round(z, 1).astype(str)),
                               texttemplate="%{text}%", colorbar=dict(title="%")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return style(fig, title, height=max(260, 60 * len(tbl) + 120))


def correlation_heatmap(returns_wide: pd.DataFrame) -> go.Figure:
    corr = returns_wide.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(hovermode="closest")
    return style(fig, "Correlation of daily returns", height=400)


# ------------------------------------------------------------------------- regimes
def regime_timeline(pred: pd.DataFrame, color_map: dict, split_date=None, title: str | None = None) -> go.Figure:
    order = list(color_map.keys())
    fig = px.scatter(pred, x="trade_date", y="regime_name", color="regime_name",
                     color_discrete_map=color_map, category_orders={"regime_name": order[::-1]})
    fig.update_traces(marker=dict(symbol="square", size=7))
    fig.update_layout(showlegend=True, yaxis_title=None, xaxis_title=None, hovermode="closest")
    if split_date is not None:
        _vline(fig, split_date, "Train | Test")
    return style(fig, title or "Regime timeline", height=300)


def price_by_regime(px_df: pd.DataFrame, pred: pd.DataFrame, color_map: dict) -> go.Figure:
    m = px_df[["trade_date", "close"]].merge(pred[["trade_date", "regime_name"]], on="trade_date")
    fig = go.Figure(go.Scatter(x=m["trade_date"], y=m["close"], mode="lines", name="Close",
                               line=dict(color="#c8ccd4", width=1), hoverinfo="skip"))
    for name, color in color_map.items():
        g = m[m["regime_name"] == name]
        fig.add_trace(go.Scatter(x=g["trade_date"], y=g["close"], mode="markers", name=name,
                                 marker=dict(color=color, size=5)))
    fig.update_yaxes(title_text="Adjusted close")
    return style(fig, "Price coloured by regime", height=400)


def frequency_bar(freq: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = px.bar(freq, x="regime_name", y="frequency", color="regime_name", color_discrete_map=color_map,
                 text=freq["frequency"].map(lambda v: f"{v:.1%}"), category_orders={"regime_name": list(color_map)})
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Share of days", hovermode="closest")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "Regime frequency", height=340)


def duration_bar(dur: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dur["regime_name"], y=dur["mean_duration"], name="Mean", marker_color="#3a86ff"))
    fig.add_trace(go.Bar(x=dur["regime_name"], y=dur["median_duration"], name="Median", marker_color="#8ecae6"))
    fig.add_trace(go.Bar(x=dur["regime_name"], y=dur["max_duration"], name="Longest", marker_color="#023047"))
    fig.update_layout(barmode="group", yaxis_title="Trading days", hovermode="closest")
    fig.update_xaxes(categoryorder="array", categoryarray=list(color_map))
    return style(fig, "Regime duration (consecutive trading days)", height=340)


def transition_heatmap(probs: pd.DataFrame) -> go.Figure:
    fig = px.imshow(probs, text_auto=".1%", color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto")
    fig.update_layout(xaxis_title="To regime (next day)", yaxis_title="From regime", hovermode="closest")
    return style(fig, "Transition probabilities (historical frequencies)", height=360)


def pca_scatter(pca_df: pd.DataFrame, pred: pd.DataFrame, color_map: dict, variance=None) -> go.Figure:
    m = pca_df.merge(pred[["trade_date", "regime_name", "dataset_split"]], on="trade_date")
    fig = px.scatter(m, x="pc1", y="pc2", color="regime_name", symbol="dataset_split",
                     color_discrete_map=color_map, symbol_map={"train": "circle", "test": "diamond"},
                     category_orders={"regime_name": list(color_map)}, opacity=0.75, hover_data=["trade_date"])
    fig.update_traces(marker=dict(size=7))
    sub = f" (PC1 {variance[0]:.0%}, PC2 {variance[1]:.0%} of variance)" if variance is not None else ""
    fig.update_layout(hovermode="closest")
    return style(fig, "PCA projection of scaled features" + sub, height=460)


# ---------------------------------------------------------------------------- risk
def rolling_vol_fig(returns_by_ticker: dict, window: int = 20, periods: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (t, r) in enumerate(returns_by_ticker.items()):
        rv = risk.rolling_volatility(r, window)
        fig.add_trace(go.Scatter(x=rv.index, y=rv, name=t, line=dict(color=LINE_COLORS[i % 8], width=1.6)))
    if periods is not None and len(periods):
        for _, p in periods.iterrows():
            fig.add_shape(type="rect", x0=p["period_start"], x1=p["period_end"], y0=0, y1=1, xref="x",
                          yref="paper", fillcolor="rgba(231,111,81,0.15)", line_width=0, layer="below")
    fig.update_yaxes(tickformat=".0%", title_text=f"{window}-day annualised volatility")
    return style(fig, "Rolling volatility" + (" (shaded = high-risk periods)" if periods is not None else ""), 400)


def drawdown_fig(returns_by_ticker: dict) -> go.Figure:
    fig = go.Figure()
    for i, (t, r) in enumerate(returns_by_ticker.items()):
        dd = risk.drawdown_series(r)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=t, line=dict(color=LINE_COLORS[i % 8], width=1.4),
                                 fill="tozeroy" if len(returns_by_ticker) == 1 else None))
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown from peak")
    return style(fig, "Drawdowns (underwater chart)", 380)


def var_es_bar(risk_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label, color in [("var_95", "VaR 95%", "#8ecae6"), ("es_95", "ES 95%", "#219ebc"),
                              ("var_99", "VaR 99%", "#f4a261"), ("es_99", "ES 99%", "#e76f51")]:
        fig.add_trace(go.Bar(x=risk_df["ticker"], y=risk_df[col], name=label, marker_color=color))
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_yaxes(tickformat=".1%", title_text="1-day loss")
    return style(fig, "Historical VaR & Expected Shortfall by ticker", 380)


def regime_risk_bar(rr: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Annualised volatility", "VaR 95% (1-day)", "ES 95% (1-day)"))
    for j, col in enumerate(["annualized_volatility", "var_95", "es_95"], start=1):
        fig.add_trace(go.Bar(x=rr["regime_name"], y=rr[col], marker_color=[color_map.get(n, NEUTRAL) for n in rr["regime_name"]],
                             showlegend=False, text=rr[col].map(lambda v: f"{v:.1%}")), row=1, col=j)
        fig.update_yaxes(tickformat=".0%", row=1, col=j)
        fig.update_xaxes(categoryorder="array", categoryarray=list(color_map), row=1, col=j)
    fig.update_layout(hovermode="closest")
    return style(fig, "Risk by regime", 360)


def return_hist(r: pd.Series, title: str = "Distribution of daily returns") -> go.Figure:
    r = r.dropna()
    fig = go.Figure(go.Histogram(x=r, nbinsx=60, marker_color="#8ecae6", opacity=0.9))
    for lvl, col in [(0.95, "#f4a261"), (0.99, "#e76f51")]:
        v = -risk.historical_var(r, lvl)
        fig.add_shape(type="line", x0=v, x1=v, y0=0, y1=1, xref="x", yref="paper", line=dict(color=col, width=2, dash="dash"))
        fig.add_annotation(x=v, y=1, xref="x", yref="paper", text=f"VaR {int(lvl * 100)}%", showarrow=False,
                           yanchor="bottom", font=dict(color=col))
    fig.update_xaxes(tickformat=".1%")
    fig.update_layout(hovermode="closest", yaxis_title="Days")
    return style(fig, title, 360)


def return_box_by_regime(df: pd.DataFrame, color_map: dict) -> go.Figure:
    """df: regime_name, ret."""
    fig = px.box(df, x="regime_name", y="ret", color="regime_name", color_discrete_map=color_map,
                 category_orders={"regime_name": list(color_map)}, points="outliers")
    fig.update_yaxes(tickformat=".1%", title_text="Daily return")
    fig.update_layout(showlegend=False, xaxis_title=None, hovermode="closest")
    return style(fig, "Daily return distribution by regime", 360)


# --------------------------------------------------------------------- evaluation
def selection_curves(sel: pd.DataFrame) -> go.Figure:
    """sel: model_selection rows for one ticker."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Silhouette (higher is better)",
                                                        "Inertia / WCSS (look for the elbow)", "GMM BIC & AIC (lower is better)"))
    colors = {"KMeans": "#3a86ff", "GMM": "#fb5607"}
    for m in config.MODEL_NAMES:
        g = sel[sel["model_name"] == m].sort_values("n_clusters")
        fig.add_trace(go.Scatter(x=g["n_clusters"], y=g["silhouette_train"], name=f"{m} train", mode="lines+markers",
                                 line=dict(color=colors[m])), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["n_clusters"], y=g["silhouette_test"], name=f"{m} test", mode="lines+markers",
                                 line=dict(color=colors[m], dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["n_clusters"], y=g["inertia_train"], name=f"{m} inertia", mode="lines+markers",
                                 line=dict(color=colors[m]), showlegend=False), row=1, col=2)
    g = sel[sel["model_name"] == "GMM"].sort_values("n_clusters")
    fig.add_trace(go.Scatter(x=g["n_clusters"], y=g["bic"], name="BIC", mode="lines+markers", line=dict(color="#8338ec")), row=1, col=3)
    fig.add_trace(go.Scatter(x=g["n_clusters"], y=g["aic"], name="AIC", mode="lines+markers", line=dict(color="#06d6a0")), row=1, col=3)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="Number of regimes (k)", dtick=1, row=1, col=c)
    return style(fig, None, 380)


def cluster_distribution(pred: pd.DataFrame, color_map: dict) -> go.Figure:
    """pred: rows for one ticker, both models."""
    d = pred.groupby(["model_name", "regime_name"]).size().rename("n").reset_index()
    d["share"] = d["n"] / d.groupby("model_name")["n"].transform("sum")
    fig = px.bar(d, x="model_name", y="share", color="regime_name", color_discrete_map=color_map,
                 category_orders={"regime_name": list(color_map)}, text=d["share"].map(lambda v: f"{v:.0%}"))
    fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title="Share of days", hovermode="closest")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "Cluster distribution by model", 360)


def stability_bar(ev: pd.DataFrame) -> go.Figure:
    fig = px.bar(ev, x="ticker", y="stability_ari_mean", color="model_name", barmode="group",
                 color_discrete_map={"KMeans": "#3a86ff", "GMM": "#fb5607"}, text=ev["stability_ari_mean"].map(lambda v: f"{v:.2f}"))
    fig.update_layout(hovermode="closest", xaxis_title=None, yaxis_title="Mean ARI across random seeds")
    fig.update_yaxes(range=[0, 1.05])
    return style(fig, "Cluster stability (1.0 = identical regimes for every random seed)", 360)
