"""Dashboard 1 - Market Overview."""
import pandas as pd
import streamlit as st

from src import plots, risk_engine as risk
from .common import card, csv_button, empty_note, filter_dates, num, pct, ticker_slice


def render(ctx):
    res = ctx.result
    st.subheader(f"Market Overview - {ctx.ticker}")
    px_t = filter_dates(ticker_slice(res.prices, ctx.ticker), ctx.start, ctx.end)
    ft_t = filter_dates(ticker_slice(res.features, ctx.ticker), ctx.start, ctx.end)
    if px_t.empty or ft_t.empty:
        return empty_note()

    r = px_t.set_index("trade_date")["close"].pct_change().dropna()
    s = risk.summarize_returns(r)
    c = st.columns(5)
    card(c[0], "Last close", f"${px_t['close'].iloc[-1]:,.2f}", f"As of {px_t['trade_date'].iloc[-1].date()}")
    card(c[1], "Return (range)", pct(s["cumulative_return"]), "Cumulative return over the selected dates")
    card(c[2], "Ann. volatility", pct(s["annualized_volatility"]))
    card(c[3], "RSI (14)", num(ft_t["rsi_14"].iloc[-1], 1), "Above 70 is often called overbought, below 30 oversold")
    card(c[4], "Max drawdown", pct(s["max_drawdown"]))

    left, right = st.columns([3, 1])
    kind = left.radio("Chart type", ["Candlestick", "Line"], horizontal=True, key="ov_kind")
    ma20 = right.checkbox("MA 20", True, key="ov_ma20")
    ma50 = right.checkbox("MA 50", True, key="ov_ma50")
    st.plotly_chart(plots.price_panel(px_t, ft_t, kind, ma20, ma50, f"{ctx.ticker} price, volume and RSI"),
                    width="stretch")

    st.markdown("##### Returns")
    which = st.radio("Series", ["daily_return", "return_5d"], horizontal=True, key="ov_ret",
                     format_func=lambda x: "Daily return" if x == "daily_return" else "5-day return")
    st.plotly_chart(plots.line_returns(ft_t, which), width="stretch")

    st.markdown("##### Cumulative performance")
    wide = res.prices.pivot(index="trade_date", columns="ticker", values="close").sort_index()
    wide = filter_dates(wide.reset_index(), ctx.start, ctx.end).set_index("trade_date")
    chosen = [t for t in ctx.compare if t in wide.columns] or list(wide.columns)
    st.plotly_chart(plots.cumulative_performance(wide[chosen].dropna(how="all")), width="stretch")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(plots.monthly_heatmap(r, f"{ctx.ticker} monthly returns (%)"), width="stretch")
    with b:
        rets = wide[chosen].pct_change().dropna(how="all")
        if rets.shape[1] >= 2:
            st.plotly_chart(plots.correlation_heatmap(rets), width="stretch")
        else:
            st.info("Select at least two tickers (sidebar -> Compare tickers) to see correlations.")

    d1, d2, d3 = st.columns(3)
    with d1:
        csv_button(px_t, "Prices (CSV)", f"{ctx.ticker}_prices.csv", "dl_ov_px")
    with d2:
        csv_button(ft_t, "Features (CSV)", f"{ctx.ticker}_features.csv", "dl_ov_ft")
    with d3:
        csv_button(plots.monthly_returns_table(r).reset_index(), "Monthly returns (CSV)",
                   f"{ctx.ticker}_monthly_returns.csv", "dl_ov_m")
