"""Small helpers shared by the four dashboards."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def filter_dates(df: pd.DataFrame, start, end, col: str = "trade_date") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[(df[col] >= pd.Timestamp(start)) & (df[col] <= pd.Timestamp(end))]


def csv_button(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(f"⬇ {label}", df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key)


def card(col, label: str, value, help_text: str | None = None, delta=None) -> None:
    col.metric(label, value, delta=delta, help=help_text)


def pct(v, digits: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{digits}%}"


def num(v, digits: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:,.{digits}f}"


def ticker_slice(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return df[df["ticker"] == ticker].sort_values("trade_date") if "trade_date" in df else df[df["ticker"] == ticker]


def empty_note(msg: str = "No data for the selected ticker / date range.") -> None:
    st.info(msg)
