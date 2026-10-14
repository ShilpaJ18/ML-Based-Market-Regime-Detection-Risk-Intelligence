"""Dashboard 4 - Model Evaluation & SQL Analytics."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, database, insights, plots
from .common import csv_button, empty_note

_JOIN = "regime_predictions p JOIN features f ON f.ticker = p.ticker AND f.trade_date = p.trade_date"

PRESETS = {
    "Regime summary for the selected ticker": f"""SELECT p.regime_name, COUNT(*) AS days,
       ROUND(AVG(f.daily_return) * 100, 3) AS avg_daily_return_pct,
       ROUND(AVG(f.volatility_20d) * 100, 1) AS avg_ann_vol_pct,
       ROUND(MIN(f.daily_return) * 100, 2) AS worst_day_pct,
       ROUND(MAX(f.daily_return) * 100, 2) AS best_day_pct
FROM {_JOIN}
WHERE p.ticker = :ticker AND p.model_name = :model
GROUP BY p.regime_id, p.regime_name
ORDER BY p.regime_id""",
    "Regime mix for every ticker": """SELECT p.ticker, p.regime_name, COUNT(*) AS days,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY p.ticker), 1) AS pct_of_days
FROM regime_predictions p
WHERE p.model_name = :model
GROUP BY p.ticker, p.regime_id, p.regime_name
ORDER BY p.ticker, p.regime_id""",
    "Regime transitions (window function LAG)": """SELECT prev_regime, regime_name AS next_regime, COUNT(*) AS transitions
FROM (SELECT regime_name,
             LAG(regime_name) OVER (PARTITION BY ticker ORDER BY trade_date) AS prev_regime
      FROM regime_predictions
      WHERE ticker = :ticker AND model_name = :model) t
WHERE prev_regime IS NOT NULL
GROUP BY prev_regime, regime_name
ORDER BY transitions DESC""",
    "Latest regime of every ticker": """SELECT p.ticker, p.trade_date, p.regime_name, p.dataset_split
FROM regime_predictions p
JOIN (SELECT ticker, MAX(trade_date) AS last_date
      FROM regime_predictions WHERE model_name = :model GROUP BY ticker) m
  ON m.ticker = p.ticker AND m.last_date = p.trade_date
WHERE p.model_name = :model
ORDER BY p.ticker""",
    "20 worst days and their regime": f"""SELECT f.trade_date, ROUND(f.daily_return * 100, 2) AS return_pct,
       ROUND(f.volatility_20d * 100, 1) AS ann_vol_pct, p.regime_name
FROM {_JOIN}
WHERE p.ticker = :ticker AND p.model_name = :model
ORDER BY f.daily_return ASC
LIMIT 20""",
    "Regime mix by year": """SELECT {year} AS yr, p.regime_name, COUNT(*) AS days
FROM regime_predictions p
WHERE p.ticker = :ticker AND p.model_name = :model
GROUP BY {year}, p.regime_id, p.regime_name
ORDER BY yr, p.regime_id""",
    "RSI extremes by regime": f"""SELECT p.regime_name, COUNT(*) AS days,
       SUM(CASE WHEN f.rsi_14 > 70 THEN 1 ELSE 0 END) AS rsi_above_70_days,
       SUM(CASE WHEN f.rsi_14 < 30 THEN 1 ELSE 0 END) AS rsi_below_30_days
FROM {_JOIN}
WHERE p.ticker = :ticker AND p.model_name = :model
GROUP BY p.regime_id, p.regime_name
ORDER BY p.regime_id""",
    "Risk metrics by ticker": """SELECT ticker, ROUND(annualized_volatility * 100, 1) AS ann_vol_pct,
       ROUND(max_drawdown * 100, 1) AS max_drawdown_pct, ROUND(var_95 * 100, 2) AS var_95_pct,
       ROUND(var_99 * 100, 2) AS var_99_pct, ROUND(es_95 * 100, 2) AS es_95_pct,
       ROUND(cumulative_return * 100, 1) AS cumulative_return_pct
FROM risk_metrics
ORDER BY annualized_volatility DESC""",
    "Risk by regime for every ticker": """SELECT ticker, regime_name, n_days, ROUND(annualized_volatility * 100, 1) AS ann_vol_pct,
       ROUND(var_95 * 100, 2) AS var_95_pct, ROUND(es_95 * 100, 2) AS es_95_pct
FROM regime_risk_metrics
WHERE model_name = :model
ORDER BY ticker, regime_id""",
    "Model evaluation summary": """SELECT ticker, model_name, n_clusters, ROUND(silhouette_train, 3) AS sil_train,
       ROUND(silhouette_test, 3) AS sil_test, ROUND(stability_ari_mean, 3) AS stability_ari,
       ROUND(train_test_tv_distance, 3) AS train_test_shift
FROM model_evaluation
ORDER BY ticker, model_name""",
}


def _year_expr(engine) -> str:
    return "YEAR(p.trade_date)" if engine.dialect.name == "mysql" else "CAST(strftime('%Y', p.trade_date) AS INTEGER)"


def _literalise(sql: str, ticker: str, model: str, engine) -> str:
    """Inline the (validated) ticker/model so the editable query is self-contained."""
    return (sql.replace("{year}", _year_expr(engine)).replace(":ticker", f"'{ticker}'")
            .replace(":model", f"'{model}'"))


def render(ctx):
    res = ctx.result
    st.subheader("Model Evaluation & SQL Analytics")
    cmap = config.regime_color_map(res.n_regimes)
    ev = res.model_evaluation
    sel = res.model_selection
    pred = res.regime_predictions

    # ------------------------------------------------------------- model comparison
    st.markdown(f"##### Model comparison (k = {res.n_regimes}, train = first {res.meta['train_frac']:.0%} of history)")
    table = ev.copy()
    table["train_end_date"] = pd.to_datetime(table["train_end_date"]).dt.date
    table["model_name"] = table["model_name"].map(config.MODEL_LABELS)
    show_cols = ["ticker", "model_name", "silhouette_train", "silhouette_test", "inertia_train",
                 "davies_bouldin_train", "bic", "aic", "test_avg_loglik", "stability_ari_mean",
                 "train_test_tv_distance", "train_rows", "test_rows", "train_end_date"]
    st.dataframe(table[show_cols].round(3), width="stretch", hide_index=True)
    with st.expander("How to read these metrics"):
        st.markdown(
            "- **Silhouette** (-1..1): cohesion vs separation of clusters in scaled feature space; higher is better. "
            "*train* is in-sample, *test* uses the later, unseen period.\n"
            "- **Inertia (WCSS)**: within-cluster sum of squares; falls as k grows, so look for an *elbow*.\n"
            "- **Davies-Bouldin**: lower is better.\n"
            "- **BIC / AIC** (GMM only): likelihood penalised for model size; lower is better. "
            "**test_avg_loglik** is the average log-likelihood of unseen test rows (higher is better).\n"
            "- **stability_ari_mean**: Adjusted Rand Index between regimes from the reference run and runs with other "
            "random seeds (1.0 = identical).\n"
            "- **train_test_tv_distance**: how different the regime frequencies are in the test vs training period "
            "(0 = same mix).")
    csv_button(ev, "Model evaluation (CSV)", "model_evaluation.csv", "dl_ev")

    # --------------------------------------------------------------- k selection
    st.markdown(f"##### Choosing the number of regimes - {ctx.ticker}")
    s_t = sel[sel["ticker"] == ctx.ticker]
    if s_t.empty:
        empty_note()
    else:
        st.plotly_chart(plots.selection_curves(s_t), width="stretch")
        csv_button(s_t, "Selection sweep (CSV)", f"{ctx.ticker}_model_selection.csv", "dl_sel")

    a, b = st.columns(2)
    with a:
        p_t = pred[pred["ticker"] == ctx.ticker]
        st.plotly_chart(plots.cluster_distribution(p_t, cmap), width="stretch")
    with b:
        st.plotly_chart(plots.stability_bar(ev), width="stretch")

    st.markdown("##### Regime frequency: training vs test period")
    split_tbl = (pred[pred["ticker"] == ctx.ticker].groupby(["model_name", "dataset_split", "regime_name"]).size()
                 .rename("days").reset_index())
    split_tbl["share"] = split_tbl["days"] / split_tbl.groupby(["model_name", "dataset_split"])["days"].transform("sum")
    fig = px.bar(split_tbl, x="dataset_split", y="share", color="regime_name", facet_col="model_name",
                 color_discrete_map=cmap, category_orders={"regime_name": list(cmap), "dataset_split": ["train", "test"]})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(plots.style(fig, None, 320), width="stretch")

    # ------------------------------------------------------------- stored stats
    st.markdown(f"##### Regime statistics - {ctx.ticker}, {config.MODEL_LABELS[ctx.model]}")
    rs = res.regime_stats
    rs = rs[(rs["ticker"] == ctx.ticker) & (rs["model_name"] == ctx.model)].drop(columns=["ticker", "model_name"])
    st.dataframe(rs.round(4), width="stretch", hide_index=True)
    csv_button(rs, "Regime statistics (CSV)", f"{ctx.ticker}_{ctx.model}_regime_stats.csv", "dl_rs")

    # ----------------------------------------------------------------- insights
    st.markdown("##### Data-backed insights")
    st.caption("Every sentence below is computed from the current results; nothing is hard-coded.")
    for line in insights.generate_insights(res, ctx.model):
        st.markdown(f"- {line}")

    # --------------------------------------------------------------------- SQL
    st.markdown("##### SQL analytics")
    engine, backend = ctx.sql_engine, ctx.sql_backend
    st.caption(f"Backend: **{backend}**. Read-only: a single SELECT/WITH statement is allowed.")
    if backend.startswith("In-memory"):
        st.info("MySQL is not connected, so the same tables are loaded into an in-memory SQLite database "
                "for these queries. Connect MySQL (see README) to run them against your database.")
    preset = st.selectbox("Preset query", list(PRESETS), key="sql_preset")
    sql_default = _literalise(PRESETS[preset], ctx.ticker, ctx.model, engine)
    sql = st.text_area("SQL (editable)", sql_default, height=220, key=f"sql_{preset}_{ctx.ticker}_{ctx.model}_{backend}")
    if st.button("Run query", key="sql_run", type="primary"):
        try:
            out = database.run_select(engine, sql)
            st.session_state["sql_out"] = out
        except database.DatabaseError as exc:
            st.session_state["sql_out"] = None
            st.error(str(exc))
    out = st.session_state.get("sql_out")
    if out is not None:
        st.write(f"{len(out):,} rows")
        st.dataframe(out, width="stretch", hide_index=True)
        csv_button(out, "Query result (CSV)", "sql_result.csv", "dl_sql")
        num_cols = out.select_dtypes("number").columns.tolist()
        txt_cols = [c for c in out.columns if c not in num_cols]
        if 0 < len(out) <= 40 and num_cols and txt_cols:
            y = st.selectbox("Chart column", num_cols, key="sql_y")
            color = txt_cols[1] if len(txt_cols) > 1 else None
            fig = px.bar(out, x=txt_cols[0], y=y, color=color, barmode="group")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(plots.style(fig, None, 340), width="stretch")

    if backend.startswith("MySQL"):
        with st.expander("MySQL table row counts"):
            st.dataframe(database.table_counts(engine), width="stretch", hide_index=True)
