"""Dashboard 2 - Market Regime Detection."""
import pandas as pd
import streamlit as st

from src import config, plots, regime_models as rm
from .common import card, csv_button, empty_note, filter_dates, pct


def render(ctx):
    res = ctx.result
    k = res.n_regimes
    cmap = config.regime_color_map(k)
    st.subheader(f"Market Regime Detection - {ctx.ticker} - {config.MODEL_LABELS[ctx.model]}")
    st.caption(config.DISCLAIMER)

    pred_all = res.regime_predictions
    pred_all = pred_all[(pred_all["ticker"] == ctx.ticker) & (pred_all["model_name"] == ctx.model)]
    pred = filter_dates(pred_all, ctx.start, ctx.end).sort_values("trade_date")
    if pred.empty:
        return empty_note()
    px_t = ticker_prices(res, ctx)
    ft_t = filter_dates(res.features[res.features["ticker"] == ctx.ticker], ctx.start, ctx.end)

    episodes = rm.compute_episodes(pred)
    last = episodes.iloc[-1]
    freq = rm.frequency_table(pred)
    dur = rm.duration_summary(episodes)
    c = st.columns(4)
    card(c[0], "Latest regime", str(last["regime_name"]), f"Classification of {pred['trade_date'].iloc[-1].date()} (historical label, not a forecast)")
    card(c[1], "Days in current run", int(last["days"]))
    card(c[2], "Regime changes", len(episodes) - 1, "Number of switches in the selected range")
    card(c[3], "Training rows", int((pred_all["dataset_split"] == "train").sum()),
         "Models are fitted on the first part of the history only")

    test_rows = pred_all[pred_all["dataset_split"] == "test"]
    split_date = test_rows["trade_date"].min() if len(test_rows) else None
    if split_date is not None and not (pred["trade_date"].min() <= split_date <= pred["trade_date"].max()):
        split_date = None
    st.plotly_chart(plots.regime_timeline(pred, cmap, split_date), width="stretch")
    st.plotly_chart(plots.price_by_regime(px_t, pred, cmap), width="stretch")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(plots.frequency_bar(freq, cmap), width="stretch")
    with b:
        st.plotly_chart(plots.duration_bar(dur, cmap), width="stretch")

    a, b = st.columns(2)
    with a:
        _, probs = rm.transition_matrices(pred, k)
        st.plotly_chart(plots.transition_heatmap(probs), width="stretch")
        stay = pd.Series({n: probs.loc[n, n] for n in probs.index}, name="Probability of staying next day")
        st.caption("Self-transition: " + ", ".join(f"{n} {v:.0%}" for n, v in stay.items()))
    with b:
        pca = res.pca[res.pca["ticker"] == ctx.ticker]
        pca = filter_dates(pca, ctx.start, ctx.end)
        var_row = res.pca_variance[res.pca_variance["ticker"] == ctx.ticker]
        var = (var_row.iloc[0]["pc1_var"], var_row.iloc[0]["pc2_var"]) if len(var_row) else None
        st.plotly_chart(plots.pca_scatter(pca, pred, cmap, var), width="stretch")

    st.markdown("##### Cluster statistics (mean feature values per regime)")
    stats = rm.cluster_statistics(ft_t, pred)
    show = stats.copy()
    for col in show.columns:
        if col.startswith("mean_") and col not in ("mean_rsi_14",):
            show[col] = show[col].round(4)
    show["frequency"] = show["frequency"].map(lambda v: f"{v:.1%}")
    st.dataframe(show.round(3), width="stretch", hide_index=True)

    st.markdown("##### Regime durations")
    st.dataframe(dur.round(1), width="stretch", hide_index=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        csv_button(pred, "Regime labels (CSV)", f"{ctx.ticker}_{ctx.model}_regimes.csv", "dl_rg_pred")
    with d2:
        csv_button(stats, "Cluster stats (CSV)", f"{ctx.ticker}_{ctx.model}_cluster_stats.csv", "dl_rg_stats")
    with d3:
        csv_button(episodes, "Regime episodes (CSV)", f"{ctx.ticker}_{ctx.model}_episodes.csv", "dl_rg_ep")


def ticker_prices(res, ctx):
    p = res.prices[res.prices["ticker"] == ctx.ticker].sort_values("trade_date")
    return filter_dates(p, ctx.start, ctx.end)
