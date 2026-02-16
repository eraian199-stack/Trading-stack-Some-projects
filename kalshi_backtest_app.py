from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


def normalize_price_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.dropna().empty:
        return clean
    max_val = clean.max()
    if max_val > 1:
        clean = clean / 100
    return clean


def fee_series(price: pd.Series, contracts: int, rate: float) -> pd.Series:
    raw = rate * contracts * price * (1 - price)
    return np.ceil(raw * 100) / 100


def build_ask_series(
    df: pd.DataFrame,
    ask_col: Optional[str],
    bid_col: Optional[str],
) -> pd.Series:
    if ask_col:
        return normalize_price_series(df[ask_col])
    if bid_col:
        bid = normalize_price_series(df[bid_col])
        return 1 - bid
    return pd.Series(dtype=float)


st.set_page_config(page_title="Kalshi Backtest", layout="wide")
st.title("Kalshi Negative-Spread Backtest")
st.caption("Upload a CSV of prices to backtest the fee-aware negative-spread rule.")

with st.sidebar:
    st.header("Parameters")
    contracts = st.number_input("Contracts per trade", min_value=1, value=1, step=1)
    fee_rate = st.number_input("Taker fee rate", min_value=0.0, value=0.07, step=0.001)
    safety_margin = st.number_input("Safety margin", min_value=0.5, max_value=1.0, value=0.99, step=0.01)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
sample = st.button("Use sample data")

if uploaded:
    df = pd.read_csv(uploaded)
elif sample:
    sample_csv = io.StringIO(
        "timestamp,yes_bid,yes_ask,no_bid,no_ask\n"
        "2024-01-01T00:00:00Z,0.48,0.49,0.50,0.51\n"
        "2024-01-01T00:01:00Z,0.49,0.50,0.49,0.50\n"
        "2024-01-01T00:02:00Z,0.47,0.48,0.51,0.52\n"
    )
    df = pd.read_csv(sample_csv)
else:
    st.info("Upload a CSV to begin or click 'Use sample data'.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)

columns = ["(none)"] + list(df.columns)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ts_col = st.selectbox("Timestamp column", columns, index=0)
with col2:
    yes_ask_col = st.selectbox("YES ask column", columns, index=0)
with col3:
    no_ask_col = st.selectbox("NO ask column", columns, index=0)
with col4:
    yes_bid_col = st.selectbox("YES bid column", columns, index=0)
with col5:
    no_bid_col = st.selectbox("NO bid column", columns, index=0)

yes_ask = build_ask_series(df, None if yes_ask_col == "(none)" else yes_ask_col,
                          None if no_bid_col == "(none)" else no_bid_col)
no_ask = build_ask_series(df, None if no_ask_col == "(none)" else no_ask_col,
                         None if yes_bid_col == "(none)" else yes_bid_col)

if yes_ask.empty or no_ask.empty:
    st.error("Provide YES/NO asks, or provide the opposite-side bids to derive asks.")
    st.stop()

fee_yes = fee_series(yes_ask, contracts, fee_rate)
fee_no = fee_series(no_ask, contracts, fee_rate)
cost = yes_ask + no_ask + fee_yes + fee_no
trigger = cost < safety_margin
profit = (1 - cost) * contracts

result = pd.DataFrame(
    {
        "yes_ask": yes_ask,
        "no_ask": no_ask,
        "fee_yes": fee_yes,
        "fee_no": fee_no,
        "cost": cost,
        "trigger": trigger,
        "profit": profit,
    }
)

if ts_col != "(none)":
    result.insert(0, "timestamp", df[ts_col])

trades = result[result["trigger"]].copy()

st.subheader("Summary")
total_trades = int(trades.shape[0])
total_profit = float(trades["profit"].sum()) if total_trades else 0.0
total_cost = float(trades["cost"].sum()) if total_trades else 0.0
roi = (total_profit / total_cost * 100) if total_cost else 0.0

metric_cols = st.columns(4)
metric_cols[0].metric("Triggers", f"{total_trades}")
metric_cols[1].metric("Total Profit ($)", f"{total_profit:.2f}")
metric_cols[2].metric("Avg Cost ($)", f"{(total_cost / total_trades):.4f}" if total_trades else "0.0000")
metric_cols[3].metric("ROI (%)", f"{roi:.2f}")

st.subheader("Trigger Trades")
st.dataframe(trades, use_container_width=True)

st.subheader("Cost Over Time")
st.line_chart(result["cost"])
