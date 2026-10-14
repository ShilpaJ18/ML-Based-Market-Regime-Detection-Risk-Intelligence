"""Dashboard 3 - Risk Intelligence."""
import pandas as pd
import streamlit as st

from src import config, plots, risk_engine as risk
from .common import card, csv_button, empty_note, filter_dates, pct


def _returns(res, ticker, start, end) -> pd.Series:
    p = res.prices[res.prices["ticker"] == ticker].sort_values("trade_date")
    p = filter_dates(p, start, end)
    return p.set_index("trade_date")["close"].pct_change().dropna()


def render(ctx):
    res = ctx.result
    cmap = config.regime_color_map(res.n_regimes)
    st.subheader(f"Risk Intelligence - {ctx.ticker}")
    st.caption("VaR / Expected Shortfall are historical (empirical) one-day losses; they describe the past "
               "sample and are not a guarantee about future losses.")
    r = _returns(res, ctx.ticker, ctx.start, ctx.end)
    if len(r) < 30:
        return empty_note("Need at least 30 trading days in the selected range.")
    s = risk.summarize_returns(r)
    c = st.columns(6)
    card(c[0], "Ann. volatility", pct(s["annualized_volatility"]))
    card(c[1], "Max drawdown", pct(s["max_drawdown"]))
    card(c[2], "VaR 95% (1d)", pct(s["var_95"], 2), "Loss exceeded on ~5% of days")
    card(c[3], "VaR 99% (1d)", pct(s["var_99"], 2), "Loss exceeded on ~1% of days")
    card(c[4], "ES 95% (1d)", pct(s["es_95"], 2), "Average loss on the worst 5% of days")
    card(c[5], "Cumulative return", pct(s["cumulative_return"]))

    st.markdown("##### Volatility and drawdowns")
    w1, w2 = st.columns([1, 1])
    window = w1.slider("Rolling window (days)", 10, 120, 20, 5, key="rk_win")
    q = w2.slider("High-risk threshold (volatility percentile)", 0.80, 0.99, 0.90, 0.01, key="rk_q")
    compare = [t for t in ctx.compare if t in res.tickers] or [ctx.ticker]
    rets = {t: _returns(res, t, ctx.start, ctx.end) for t in compare}
    periods = risk.high_risk_periods(ctx.ticker, r, window=window, quantile=q)
    st.plotly_chart(plots.rolling_vol_fig(rets, window, periods if ctx.ticker in rets else None), width="stretch")
    st.plotly_chart(plots.drawdown_fig(rets), width="stretch")

    st.markdown("##### Risk comparison by ticker")
    rows = []
    for t in compare:
        rr = rets[t]
        if len(rr) > 30:
            rows.append({"ticker": t, **risk.summarize_returns(rr)})
    cmp_df = pd.DataFrame(rows)
    if len(cmp_df):
        a, b = st.columns([3, 2])
        with a:
            st.plotly_chart(plots.var_es_bar(cmp_df), width="stretch")
        with b:
            tbl = cmp_df[["ticker", "annualized_volatility", "max_drawdown", "var_95", "var_99", "es_95", "es_99",
                          "cumulative_return", "return_to_risk"]].round(4)
            st.dataframe(tbl, width="stretch", hide_index=True)

    st.markdown(f"##### Risk by regime ({config.MODEL_LABELS[ctx.model]})")
    pred = res.regime_predictions
    pred = filter_dates(pred[(pred["ticker"] == ctx.ticker) & (pred["model_name"] == ctx.model)], ctx.start, ctx.end)
    rr = risk.risk_by_regime(ctx.ticker, ctx.model, r, pred)
    if len(rr):
        st.plotly_chart(plots.regime_risk_bar(rr, cmap), width="stretch")
        st.dataframe(rr.drop(columns=["ticker", "model_name"]).round(4), width="stretch", hide_index=True)
        st.caption("Max drawdown per regime is computed on the compounded returns of that regime's days only.")
    else:
        st.info("No regime labels in the selected date range.")

    st.markdown("##### Return distributions")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(plots.return_hist(r, f"{ctx.ticker} daily returns"), width="stretch")
    with b:
        if len(pred):
            joined = pred.set_index("trade_date")[["regime_name"]].join(r.rename("ret"), how="inner").reset_index(drop=True)
            st.plotly_chart(plots.return_box_by_regime(joined, cmap), width="stretch")

    st.markdown("##### High-risk periods")
    if len(periods):
        shown = periods.drop(columns=["ticker"]).copy()
        shown["period_start"] = shown["period_start"].dt.date
        shown["period_end"] = shown["period_end"].dt.date
        st.caption(f"Days when the {window}-day annualised volatility was in the top {1 - q:.0%} of its history "
                   f"(threshold {periods['vol_threshold'].iloc[0]:.1%}); gaps of up to 5 days are merged.")
        st.dataframe(shown.sort_values("peak_volatility", ascending=False).round(4), width="stretch", hide_index=True)
    else:
        st.info("No high-risk periods in this range.")

    d1, d2, d3 = st.columns(3)
    with d1:
        csv_button(cmp_df, "Ticker risk table (CSV)", "risk_by_ticker.csv", "dl_rk_t")
    with d2:
        csv_button(rr, "Regime risk table (CSV)", f"{ctx.ticker}_{ctx.model}_risk_by_regime.csv", "dl_rk_r")
    with d3:
        csv_button(periods, "High-risk periods (CSV)", f"{ctx.ticker}_high_risk_periods.csv", "dl_rk_p")
