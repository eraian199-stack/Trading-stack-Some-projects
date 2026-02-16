import os
import sys
from datetime import datetime, date, time, timedelta

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
    except Exception:
        try:
            from streamlit.runtime.scriptrunner_utils import get_script_run_ctx
        except Exception:
            get_script_run_ctx = None

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "TSLA",
    "AMD", "META", "AMZN", "NFLX", "F",
]
MARKET_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "ARKK"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RTH_START = time(9, 30)
RTH_END = time(16, 0)
MONEYNESS_OPTIONS = ["Custom", "ITM (Higher Delta)", "ATM (Balanced)", "OTM (Convex)"]
MONEYNESS_PRESETS = {
    "ITM (Higher Delta)": {"delta": 0.65, "premium_pct": 5.0, "theta_pct": 6.0, "spread_pct": 3.0},
    "ATM (Balanced)": {"delta": 0.45, "premium_pct": 2.5, "theta_pct": 8.0, "spread_pct": 4.0},
    "OTM (Convex)": {"delta": 0.25, "premium_pct": 1.0, "theta_pct": 12.0, "spread_pct": 6.0},
}
MONEYNESS_DELTA_RANGES = {
    "ITM (Higher Delta)": "0.55-0.75",
    "ATM (Balanced)": "0.35-0.55",
    "OTM (Convex)": "0.20-0.35",
}
MONEYNESS_NOTES = {
    "ITM (Higher Delta)": "More delta, lower theta drag; use when edge is modest or chop risk is high.",
    "ATM (Balanced)": "Balanced delta/theta; default for clean trends with normal liquidity.",
    "OTM (Convex)": "Cheaper and convex but high theta; use only when edge is strong and trend is clear.",
}

# --- Data Fetching Function ---
def get_intraday_data(ticker, interval, period="1d", show_errors=True, include_prepost=False):
    try:
        # Fetch intraday data; session VWAP is calculated per day downstream.
        download_kwargs = dict(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if include_prepost:
            download_kwargs["prepost"] = True
        try:
            df = yf.download(multi_level_index=False, **download_kwargs)
        except TypeError:
            if "prepost" in download_kwargs:
                download_kwargs.pop("prepost", None)
            df = yf.download(**download_kwargs)
        
        if df.empty:
            if show_errors:
                st.error(f"No intraday data found for {ticker}. Market might be closed or ticker invalid.")
            return None

        # Data Cleaning (Flatten MultiIndex if exists)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] for col in df.columns]
            
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
            else:
                close_cols = [col for col in df.columns if str(col).endswith("Close")]
                if len(close_cols) == 1:
                    df = df.rename(columns={close_cols[0]: "Close"})
                else:
                    if show_errors:
                        st.error("Downloaded data is missing a Close column.")
                    return None

        close_prices = df["Close"]
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]
        close_prices = pd.to_numeric(close_prices, errors="coerce")
        df = df.copy()
        df["Close"] = close_prices

        return df
    except Exception as e:
        if show_errors:
            st.error(f"Error fetching data: {e}")
        return None


def add_session_vwap(df):
    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    date_index = df.index.date
    cum_vol = df["Volume"].groupby(date_index).cumsum()
    cum_pv = (tp * df["Volume"]).groupby(date_index).cumsum()
    df["VWAP"] = cum_pv / cum_vol
    return df


def _add_indicators_core(df, atr_window, vol_window, vwap_z_window, vwap_slope_window, adx_window):
    df = add_session_vwap(df)
    df_len = len(df)

    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    if df_len >= atr_window:
        atr_indicator = AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=atr_window
        )
        df["ATR"] = atr_indicator.average_true_range()
    else:
        df["ATR"] = pd.Series([pd.NA] * df_len, index=df.index)
    df["ATR_Pct"] = (df["ATR"] / df["Close"].replace(0, pd.NA)) * 100
    df["Vol_SMA"] = df["Volume"].rolling(vol_window).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA"]
    df["VWAP_STD"] = (df["Close"] - df["VWAP"]).rolling(vwap_z_window).std()
    df["VWAP_Z"] = (df["Close"] - df["VWAP"]) / df["VWAP_STD"].where(df["VWAP_STD"] != 0)
    df["VWAP_Slope"] = df["VWAP"].diff().rolling(vwap_slope_window).mean()
    if df_len >= adx_window:
        adx_indicator = ADXIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], window=adx_window
        )
        df["ADX"] = adx_indicator.adx()
    else:
        df["ADX"] = pd.Series([pd.NA] * df_len, index=df.index)

    return df


def add_indicators(
    df,
    atr_window,
    vol_window,
    vwap_z_window,
    vwap_slope_window,
    adx_window,
    regular_hours_only=False,
    regular_start=RTH_START,
    regular_end=RTH_END,
):
    if df is None or df.empty:
        return df
    if not regular_hours_only:
        return _add_indicators_core(df.copy(), atr_window, vol_window, vwap_z_window, vwap_slope_window, adx_window)

    session_mask = compute_session_ok_series(df.index, regular_start, regular_end)
    session_df = df.loc[session_mask].copy()
    session_df = _add_indicators_core(
        session_df, atr_window, vol_window, vwap_z_window, vwap_slope_window, adx_window
    )

    df = df.copy()
    indicator_cols = [
        "VWAP",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "ATR",
        "ATR_Pct",
        "Vol_SMA",
        "Vol_Ratio",
        "VWAP_STD",
        "VWAP_Z",
        "VWAP_Slope",
        "ADX",
    ]
    for col in indicator_cols:
        df[col] = pd.NA
    df.loc[session_mask, indicator_cols] = session_df[indicator_cols]
    return df


def get_market_filter_series(market_data, index):
    if market_data is None or market_data.empty:
        neutral = pd.Series(True, index=index)
        return neutral, neutral, "UNAVAILABLE"

    bull = (market_data["Close"] > market_data["VWAP"]) & (
        market_data["MACD"] > market_data["MACD_Signal"]
    )
    bear = (market_data["Close"] < market_data["VWAP"]) & (
        market_data["MACD"] < market_data["MACD_Signal"]
    )

    if bool(bull.iloc[-1]):
        bias = "BULLISH"
    elif bool(bear.iloc[-1]):
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    bull_aligned = bull.reindex(index, method="ffill")
    bear_aligned = bear.reindex(index, method="ffill")

    market_ok_long = ~bear_aligned.fillna(False)
    market_ok_short = ~bull_aligned.fillna(False)
    return market_ok_long, market_ok_short, bias


def compute_signal_series(
    df,
    use_volume_filter,
    vol_mult,
    use_vwap_extension,
    vwap_z_max,
    use_vwap_slope,
    use_atr_filter,
    atr_pct_min,
    session_ok_series=None,
    regime_ok_series=None,
    htf_ok_long_series=None,
    htf_ok_short_series=None,
    market_ok_long_series=None,
    market_ok_short_series=None,
):
    base_long = (df["Close"] > df["VWAP"]) & (df["MACD"] > df["MACD_Signal"])
    base_short = (df["Close"] < df["VWAP"]) & (df["MACD"] < df["MACD_Signal"])

    if use_volume_filter:
        volume_ok = df["Vol_Ratio"].isna() | (df["Vol_Ratio"] >= vol_mult)
    else:
        volume_ok = pd.Series(True, index=df.index)

    if use_vwap_extension:
        vwap_ok = df["VWAP_Z"].isna() | (df["VWAP_Z"].abs() <= vwap_z_max)
    else:
        vwap_ok = pd.Series(True, index=df.index)

    if use_atr_filter:
        atr_ok = df["ATR_Pct"].isna() | (df["ATR_Pct"] >= atr_pct_min)
    else:
        atr_ok = pd.Series(True, index=df.index)

    if use_vwap_slope:
        slope_ok_long = df["VWAP_Slope"] > 0
        slope_ok_short = df["VWAP_Slope"] < 0
    else:
        slope_ok_long = pd.Series(True, index=df.index)
        slope_ok_short = pd.Series(True, index=df.index)

    if session_ok_series is None:
        session_ok_series = pd.Series(True, index=df.index)
    else:
        session_ok_series = session_ok_series.reindex(df.index, method="ffill").fillna(True)

    if regime_ok_series is None:
        regime_ok_series = pd.Series(True, index=df.index)
    else:
        regime_ok_series = regime_ok_series.reindex(df.index, method="ffill").fillna(True)

    if htf_ok_long_series is None:
        htf_ok_long_series = pd.Series(True, index=df.index)
    else:
        htf_ok_long_series = htf_ok_long_series.reindex(df.index, method="ffill").fillna(True)

    if htf_ok_short_series is None:
        htf_ok_short_series = pd.Series(True, index=df.index)
    else:
        htf_ok_short_series = htf_ok_short_series.reindex(df.index, method="ffill").fillna(True)

    if market_ok_long_series is None:
        market_ok_long_series = pd.Series(True, index=df.index)
    else:
        market_ok_long_series = market_ok_long_series.reindex(df.index, method="ffill").fillna(True)

    if market_ok_short_series is None:
        market_ok_short_series = pd.Series(True, index=df.index)
    else:
        market_ok_short_series = market_ok_short_series.reindex(df.index, method="ffill").fillna(True)

    long_ok = (
        base_long
        & volume_ok
        & vwap_ok
        & atr_ok
        & slope_ok_long
        & session_ok_series
        & regime_ok_series
        & htf_ok_long_series
        & market_ok_long_series
    )
    short_ok = (
        base_short
        & volume_ok
        & vwap_ok
        & atr_ok
        & slope_ok_short
        & session_ok_series
        & regime_ok_series
        & htf_ok_short_series
        & market_ok_short_series
    )
    return base_long, base_short, long_ok, short_ok, volume_ok, vwap_ok, atr_ok, slope_ok_long, slope_ok_short


def compute_session_ok_series(index, start_time, end_time):
    if start_time is None or end_time is None:
        return pd.Series(True, index=index)
    times = index.time
    ok = [(t >= start_time) and (t <= end_time) for t in times]
    return pd.Series(ok, index=index)


def get_timeframe_ok_series(higher_df, base_index):
    if higher_df is None or higher_df.empty:
        neutral = pd.Series(True, index=base_index)
        return neutral, neutral, "UNAVAILABLE"

    base_long = (higher_df["Close"] > higher_df["VWAP"]) & (
        higher_df["MACD"] > higher_df["MACD_Signal"]
    )
    base_short = (higher_df["Close"] < higher_df["VWAP"]) & (
        higher_df["MACD"] < higher_df["MACD_Signal"]
    )

    if bool(base_long.iloc[-1]):
        bias = "BULLISH"
    elif bool(base_short.iloc[-1]):
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    ok_long = (~base_short).reindex(base_index, method="ffill").fillna(True)
    ok_short = (~base_long).reindex(base_index, method="ffill").fillna(True)
    return ok_long, ok_short, bias


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_expirations(ticker, cache_key):
    try:
        t = yf.Ticker(ticker)
        return t.options
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_option_chain(ticker, expiration, cache_key):
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiration)
        fetched_at = datetime.now().strftime("%H:%M:%S")
        return chain.calls, chain.puts, fetched_at
    except Exception:
        return None, None, None


def select_expiration(expirations, dte_choice):
    if not expirations:
        return None, None
    today = date.today()
    exp_dates = []
    for exp in expirations:
        try:
            exp_dates.append((exp, datetime.strptime(exp, "%Y-%m-%d").date()))
        except ValueError:
            continue
    if not exp_dates:
        return None, None

    def dte(exp_date):
        return (exp_date - today).days

    if dte_choice == "0DTE only":
        candidates = [e for e in exp_dates if dte(e[1]) == 0]
    elif dte_choice == "1-7 DTE":
        candidates = [e for e in exp_dates if 1 <= dte(e[1]) <= 7]
    else:
        candidates = [e for e in exp_dates if 0 <= dte(e[1]) <= 7]

    if not candidates:
        candidates = [e for e in exp_dates if dte(e[1]) >= 0]
    if not candidates:
        return None, None

    chosen = min(candidates, key=lambda e: dte(e[1]))
    return chosen[0], dte(chosen[1])


def compute_liquidity_side(df, underlying_price, min_oi, min_vol, max_spread_pct, max_spread_abs, atm_range_pct):
    if df is None or df.empty or underlying_price is None:
        return None
    df = df.copy()
    if "bid" not in df.columns or "ask" not in df.columns:
        return None
    if "openInterest" not in df.columns or "volume" not in df.columns:
        return None
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_abs"] = df["ask"] - df["bid"]
    df["spread_pct"] = df["spread_abs"] / df["mid"].replace(0, pd.NA)

    strike_low = underlying_price * (1 - atm_range_pct / 100.0)
    strike_high = underlying_price * (1 + atm_range_pct / 100.0)
    df = df[(df["strike"] >= strike_low) & (df["strike"] <= strike_high)]
    if df.empty:
        return None

    good = (
        (df["openInterest"] >= min_oi)
        & (df["volume"] >= min_vol)
        & (df["spread_abs"] <= max_spread_abs)
        & (df["spread_pct"] <= (max_spread_pct / 100.0))
    )

    pass_rate = float(good.mean()) if len(good) else 0.0
    summary = {
        "ok": bool(good.any()),
        "pass_rate": pass_rate,
        "median_spread_pct": float(df["spread_pct"].median(skipna=True)),
        "median_spread_abs": float(df["spread_abs"].median(skipna=True)),
        "max_oi": float(df["openInterest"].max(skipna=True)),
        "max_vol": float(df["volume"].max(skipna=True)),
        "count": int(len(df)),
    }
    return summary


def get_options_liquidity_snapshot(
    ticker,
    dte_choice,
    underlying_price,
    min_oi,
    min_vol,
    max_spread_pct,
    max_spread_abs,
    atm_range_pct,
    cache_key,
):
    expirations = fetch_option_expirations(ticker, cache_key)
    exp, dte = select_expiration(expirations, dte_choice)
    if exp is None:
        return None

    calls, puts, fetched_at = fetch_option_chain(ticker, exp, cache_key)
    if calls is None or puts is None:
        return None

    call_summary = compute_liquidity_side(
        calls,
        underlying_price,
        min_oi,
        min_vol,
        max_spread_pct,
        max_spread_abs,
        atm_range_pct,
    )
    put_summary = compute_liquidity_side(
        puts,
        underlying_price,
        min_oi,
        min_vol,
        max_spread_pct,
        max_spread_abs,
        atm_range_pct,
    )
    return {
        "expiration": exp,
        "dte": dte,
        "calls": call_summary,
        "puts": put_summary,
        "fetched_at": fetched_at,
    }


def get_moneyness_preset(profile, dte_choice):
    base = MONEYNESS_PRESETS.get(profile)
    if not base:
        return None
    preset = {
        "delta": base["delta"],
        "premium_pct": base["premium_pct"],
        "theta_pct": base["theta_pct"],
        "spread_pct": base["spread_pct"],
    }
    if dte_choice == "0DTE only":
        preset["theta_pct"] *= 1.6
        preset["spread_pct"] += 1.0
    elif dte_choice == "1-7 DTE":
        preset["theta_pct"] *= 0.8
        preset["spread_pct"] = max(preset["spread_pct"] - 0.5, 0.0)
    return preset


def apply_slippage(entry, exit_price, direction, slippage_bps):
    slip = slippage_bps / 10000.0
    if direction == "long":
        entry_adj = entry * (1 + slip)
        exit_adj = exit_price * (1 - slip)
        ret = (exit_adj - entry_adj) / entry_adj
    else:
        entry_adj = entry * (1 - slip)
        exit_adj = exit_price * (1 + slip)
        ret = (entry_adj - exit_adj) / entry_adj
    return ret, entry_adj, exit_adj


def compute_expectancy(values):
    if not values:
        return None
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    win_rate = len(wins) / len(values) if values else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


def estimate_bars_per_day(df, interval):
    if interval.endswith("m"):
        try:
            minutes = int(interval[:-1])
            if minutes > 0:
                return max(int(390 / minutes), 1)
        except ValueError:
            pass
    if interval.endswith("h"):
        try:
            hours = int(interval[:-1])
            if hours > 0:
                return max(int(6.5 / hours), 1)
        except ValueError:
            pass
    if df is not None and not df.empty:
        counts = df.groupby(df.index.date).size()
        if not counts.empty:
            return max(int(counts.median()), 1)
    return 1


def estimate_break_even_move(
    current_price,
    option_premium_pct,
    option_delta,
    theta_decay_pct,
    option_spread_pct,
    option_commission,
    slippage_bps,
    hold_bars,
    bars_per_day,
):
    if current_price is None or current_price <= 0:
        return None, None
    premium = current_price * (option_premium_pct / 100.0)
    if premium <= 0 or option_delta <= 0:
        return None, None
    slip = slippage_bps / 10000.0
    theta_cost = premium * (theta_decay_pct / 100.0) * (hold_bars / max(bars_per_day, 1))
    spread_penalty = premium * (option_spread_pct / 100.0)
    commission_penalty = option_commission
    numerator = (1 - slip) * theta_cost + spread_penalty + commission_penalty + (2 * premium * slip)
    denom = (1 - slip) * option_delta
    if denom <= 0:
        return None, None
    break_even_move = numerator / denom
    break_even_pct = (break_even_move / current_price) * 100
    return break_even_move, break_even_pct


def get_week_start_date(current_date, week_start_name):
    try:
        start_idx = WEEKDAY_NAMES.index(week_start_name)
    except ValueError:
        start_idx = 0
    current_idx = current_date.weekday()
    delta = (current_idx - start_idx) % 7
    return current_date - timedelta(days=delta)


def get_trade_log_path():
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    return os.path.join(base_dir, "trade_log.csv")


def load_trade_log(path):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    return df.to_dict("records")


def save_trade_log(path, records):
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def filter_trades_for_week(records, week_start_date):
    week_end = week_start_date + timedelta(days=6)
    week_records = []
    for record in records:
        ts_value = record.get("timestamp")
        if not ts_value:
            continue
        trade_dt = None
        try:
            trade_dt = datetime.strptime(ts_value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                trade_dt = datetime.fromisoformat(ts_value)
            except Exception:
                trade_dt = None
        if trade_dt is None:
            continue
        trade_date = trade_dt.date()
        if week_start_date <= trade_date <= week_end:
            week_records.append(record)
    return week_records


def render_app():
    # --- App Configuration ---
    st.set_page_config(page_title="Intraday VWAP Scalper", layout="wide")
    st.title("⚡ Intraday VWAP + MACD Scalper")
    st.markdown("""
**Strategy:** Institutional Trend Following (Day Trading).
* **Bullish:** Price breaks ABOVE VWAP line.
* **Bearish:** Price breaks BELOW VWAP line.
* **Confirm:** MACD must match the direction.
    """)

    # --- Sidebar Inputs ---
    st.sidebar.header("Configuration")
    quick_ticker = st.sidebar.selectbox("Quick Ticker", options=DEFAULT_TICKERS, index=0)
    custom_ticker = st.sidebar.text_input("Ticker Symbol", value=quick_ticker)
    ticker = custom_ticker.strip().upper() if custom_ticker.strip() else quick_ticker
    # Intraday requires short intervals. 5m is best for stability.
    interval = st.sidebar.selectbox("Interval", options=["1m", "5m", "15m"], index=1)

    st.sidebar.subheader("Signal Filters")
    use_volume_filter = st.sidebar.checkbox("Volume Filter", value=True)
    vol_window = int(st.sidebar.number_input("Volume SMA Window", value=20, min_value=2, step=1))
    vol_mult = float(st.sidebar.number_input("Min Volume Ratio", value=1.0, min_value=0.1, step=0.1))
    use_vwap_extension = st.sidebar.checkbox("Avoid Extended VWAP", value=True)
    vwap_z_window = int(st.sidebar.number_input("VWAP Z Window", value=20, min_value=5, step=1))
    vwap_z_max = float(st.sidebar.number_input("Max VWAP Z-Score", value=1.5, min_value=0.5, step=0.1))
    use_atr_filter = st.sidebar.checkbox("ATR Volatility Filter", value=True)
    atr_pct_min = float(
        st.sidebar.number_input("Min ATR % of Price", value=0.4, min_value=0.1, step=0.1)
    )
    use_vwap_slope = st.sidebar.checkbox("VWAP Slope Filter", value=False)
    vwap_slope_window = int(st.sidebar.number_input("VWAP Slope Window", value=10, min_value=2, step=1))
    show_vwap_bands = st.sidebar.checkbox("Show VWAP Bands (+/- 1 SD)", value=False)

    st.sidebar.subheader("Regime Filter")
    use_regime_filter = st.sidebar.checkbox("Use Regime Filter", value=True)
    adx_window = int(st.sidebar.number_input("ADX Window", value=14, min_value=5, step=1))
    adx_chop = float(st.sidebar.number_input("Chop Threshold (ADX)", value=15.0, min_value=5.0, step=1.0))
    adx_trend = float(st.sidebar.number_input("Trend Threshold (ADX)", value=25.0, min_value=10.0, step=1.0))
    trend_only_mode = st.sidebar.checkbox("Trend-only Mode (ADX >= Trend)", value=False)

    st.sidebar.subheader("Session Filter")
    use_session_filter = st.sidebar.checkbox("Use Session Filter", value=True)
    session_start = st.sidebar.time_input("Session Start", value=time(9, 45))
    session_end = st.sidebar.time_input("Session End", value=time(15, 30))

    st.sidebar.subheader("Extended Hours")
    include_prepost = st.sidebar.checkbox("Include Pre/Post Market Data", value=False)
    regular_hours_only = st.sidebar.checkbox(
        "Compute Indicators on Regular Session Only",
        value=True,
        disabled=not include_prepost,
    )

    st.sidebar.subheader("Higher Timeframe")
    higher_tf_options = ["Off", "5m", "15m", "30m", "60m"]
    default_higher = {"1m": "15m", "5m": "30m", "15m": "60m"}.get(interval, "Off")
    higher_tf_index = higher_tf_options.index(default_higher) if default_higher in higher_tf_options else 0
    higher_interval = st.sidebar.selectbox("Higher Timeframe Interval", options=higher_tf_options, index=higher_tf_index)
    use_higher_tf = higher_interval != "Off" and higher_interval != interval

    use_market_filter = st.sidebar.checkbox("Market Filter", value=False)
    if use_market_filter:
        market_quick = st.sidebar.selectbox("Market Filter Ticker", options=MARKET_TICKERS, index=0)
        market_custom = st.sidebar.text_input("Custom Market Ticker", value=market_quick)
        market_ticker = market_custom.strip().upper() if market_custom.strip() else market_quick
        if not market_ticker:
            st.sidebar.warning("Market filter ticker required.")
            st.stop()
    else:
        market_ticker = "SPY"

    st.sidebar.subheader("Risk Management")
    atr_window = int(st.sidebar.number_input("ATR Window", value=14, min_value=2, step=1))
    atr_stop_mult = float(st.sidebar.number_input("Stop ATR Multiplier", value=0.7, min_value=0.2, step=0.1))
    atr_target_mult = float(st.sidebar.number_input("Target ATR Multiplier", value=1.5, min_value=0.5, step=0.1))
    use_choppy_tighten = st.sidebar.checkbox("Tighten Stops in Choppy Regime", value=True)
    choppy_stop_factor = float(st.sidebar.number_input("Choppy Stop Factor", value=0.7, min_value=0.3, step=0.1))
    use_trailing_stop = st.sidebar.checkbox("Use Trailing Stop", value=True)
    trail_atr_mult = float(st.sidebar.number_input("Trailing ATR Multiplier", value=1.0, min_value=0.2, step=0.1))
    trail_lookback = int(st.sidebar.number_input("Trailing Lookback Bars", value=20, min_value=2, step=1))

    st.sidebar.subheader("Risk Budget")
    account_size = float(st.sidebar.number_input("Account Size ($)", value=10000.0, min_value=100.0, step=100.0))
    risk_pct = float(st.sidebar.number_input("Max Risk % / Trade", value=1.0, min_value=0.1, step=0.1))
    option_delta = float(st.sidebar.number_input("Assumed Option Delta", value=0.45, min_value=0.05, max_value=1.0, step=0.05))

    st.sidebar.subheader("Position (Optional)")
    position = st.sidebar.selectbox(
        "Current Position", options=["Flat", "Long Calls", "Long Puts"], index=0
    )

    st.sidebar.subheader("Trade Quality")
    min_signal_score = int(st.sidebar.number_input("Min Signal Score", value=4, min_value=0, step=1))
    st.sidebar.caption("Set to 0 to disable the score filter.")

    st.sidebar.subheader("Trade Budget (Weekly)")
    weekly_trade_limit = int(st.sidebar.number_input("Weekly Trade Limit", value=3, min_value=1, step=1))
    trade_week_start = st.sidebar.selectbox("Week Starts On", options=WEEKDAY_NAMES, index=0)
    enforce_trade_budget = st.sidebar.checkbox("Enforce Weekly Limit", value=True)
    high_selectivity = st.sidebar.checkbox("High-Selectivity Mode", value=True)
    min_score_ratio = float(
        st.sidebar.slider("Min Score Ratio (Selectivity)", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
    )
    require_edge = st.sidebar.checkbox("Require Edge vs Costs", value=True)
    edge_buffer_pct = float(st.sidebar.number_input("Edge Buffer (%)", value=20.0, min_value=0.0, step=5.0))
    persist_trade_log = st.sidebar.checkbox("Persist Trade Log", value=True)
    clear_trade_log = st.sidebar.button("Clear Trade Log")

    st.sidebar.subheader("Options Timing")
    dte_choice = st.sidebar.selectbox(
        "DTE Preference",
        options=["0DTE only", "1-7 DTE", "0-7 DTE"],
        index=1,
    )

    st.sidebar.subheader("Options Strategy (Moneyness)")
    moneyness_profile = st.sidebar.selectbox("Moneyness Profile", options=MONEYNESS_OPTIONS, index=2)
    use_moneyness_preset = st.sidebar.checkbox("Apply Moneyness Preset", value=True)
    if moneyness_profile != "Custom":
        delta_band = MONEYNESS_DELTA_RANGES.get(moneyness_profile, "n/a")
        st.sidebar.caption(f"Delta band: {delta_band}. {MONEYNESS_NOTES.get(moneyness_profile, '')}")

    st.sidebar.subheader("Options Liquidity")
    check_liquidity = st.sidebar.checkbox("Check Liquidity", value=False)
    enforce_liquidity = st.sidebar.checkbox("Enforce Liquidity Filter", value=False)
    min_open_interest = int(st.sidebar.number_input("Min Open Interest", value=100, min_value=0, step=50))
    min_option_volume = int(st.sidebar.number_input("Min Option Volume", value=100, min_value=0, step=50))
    max_spread_abs = float(st.sidebar.number_input("Max Spread ($)", value=0.10, min_value=0.01, step=0.01))
    max_spread_pct = float(st.sidebar.number_input("Max Spread (%)", value=5.0, min_value=1.0, step=1.0))
    atm_range_pct = float(st.sidebar.number_input("ATM Range (%)", value=5.0, min_value=1.0, step=1.0))
    if "options_cache_buster" not in st.session_state:
        st.session_state.options_cache_buster = 0
    if st.sidebar.button("Refresh Options Data"):
        st.session_state.options_cache_buster += 1

    st.sidebar.subheader("A+ Only Mode")
    use_a_plus_mode = st.sidebar.checkbox("Enable A+ Mode", value=True)
    a_plus_min_score_ratio = float(
        st.sidebar.slider("A+ Min Score Ratio", min_value=0.7, max_value=1.0, value=0.9, step=0.05)
    )
    a_plus_min_vol_ratio = float(
        st.sidebar.number_input("A+ Min Volume Ratio", value=1.2, min_value=0.5, step=0.1)
    )
    a_plus_min_atr_pct = float(
        st.sidebar.number_input("A+ Min ATR %", value=0.6, min_value=0.1, step=0.1)
    )
    a_plus_require_slope = st.sidebar.checkbox("A+ Require VWAP Slope", value=True)
    a_plus_require_htf = st.sidebar.checkbox("A+ Require Higher Timeframe", value=True)
    a_plus_require_market = st.sidebar.checkbox("A+ Require Market Alignment", value=False)
    a_plus_require_liquidity = st.sidebar.checkbox("A+ Require Liquidity PASS", value=True)
    if use_a_plus_mode and a_plus_require_htf and not use_higher_tf:
        st.sidebar.warning("A+ requires Higher Timeframe, but it is OFF.")
    if use_a_plus_mode and a_plus_require_market and not use_market_filter:
        st.sidebar.warning("A+ requires Market Filter, but it is OFF.")
    if use_a_plus_mode and a_plus_require_liquidity and not check_liquidity:
        st.sidebar.warning("A+ requires liquidity checks, but Options Liquidity is OFF.")

    st.sidebar.subheader("Backtest")
    if interval == "1m":
        stats_period_options = ["1d", "5d"]
        default_lookahead = 30
        default_stats_period = "5d"
    elif interval == "5m":
        stats_period_options = ["1d", "5d", "10d"]
        default_lookahead = 12
        default_stats_period = "10d"
    else:
        stats_period_options = ["1d", "5d", "10d", "20d"]
        default_lookahead = 8
        default_stats_period = "20d"
    stats_period_index = stats_period_options.index(default_stats_period)
    stats_period = st.sidebar.selectbox(
        "Backtest Period", options=stats_period_options, index=stats_period_index
    )
    run_year_backtest = st.sidebar.checkbox("Run 1Y Daily Backtest", value=False)
    lookahead_bars = int(
        st.sidebar.number_input("Lookahead Bars (Stats)", value=default_lookahead, min_value=1, step=1)
    )
    exit_max_bars = int(
        st.sidebar.number_input("Max Hold Bars (Exit Sim)", value=default_lookahead, min_value=1, step=1)
    )
    stats_use_setups = st.sidebar.checkbox("Use Base Setups if No Signals", value=True)
    slippage_bps = float(st.sidebar.number_input("Slippage (bps)", value=2.0, min_value=0.0, step=0.5))

    st.sidebar.subheader("Options Backtest (Approx)")
    run_options_backtest = st.sidebar.checkbox("Run Options Backtest", value=True)
    option_premium_pct = float(
        st.sidebar.number_input("Assumed Premium (% of price)", value=2.0, min_value=0.5, step=0.5)
    )
    theta_decay_pct = float(
        st.sidebar.number_input("Theta Decay (% per day)", value=10.0, min_value=0.0, step=1.0)
    )
    option_spread_pct = float(
        st.sidebar.number_input("Spread Penalty (% round trip)", value=5.0, min_value=0.0, step=0.5)
    )
    option_commission = float(
        st.sidebar.number_input("Commission ($ round trip)", value=0.0, min_value=0.0, step=0.1)
    )

    moneyness_preset = get_moneyness_preset(moneyness_profile, dte_choice) if use_moneyness_preset else None
    model_option_delta = option_delta
    model_option_premium_pct = option_premium_pct
    model_theta_decay_pct = theta_decay_pct
    model_option_spread_pct = option_spread_pct
    if moneyness_preset:
        model_option_delta = moneyness_preset["delta"]
        model_option_premium_pct = moneyness_preset["premium_pct"]
        model_theta_decay_pct = moneyness_preset["theta_pct"]
        model_option_spread_pct = moneyness_preset["spread_pct"]

    st.sidebar.warning("⚠️ **PDT RULE:** You can only make 3 day trades per week if under $25k!")

    trade_log_path = get_trade_log_path()
    if "trade_log" not in st.session_state:
        st.session_state.trade_log = []
    if persist_trade_log:
        st.session_state.trade_log = load_trade_log(trade_log_path)
    if clear_trade_log:
        st.session_state.trade_log = []
        if persist_trade_log:
            save_trade_log(trade_log_path, [])

    week_start_date = get_week_start_date(date.today(), trade_week_start)
    trades_this_week = filter_trades_for_week(st.session_state.trade_log, week_start_date)
    trades_this_week_count = len(trades_this_week)
    remaining_trades = max(0, weekly_trade_limit - trades_this_week_count)
    st.sidebar.caption(
        f"Trades this week: {trades_this_week_count} | Remaining: {remaining_trades} | Week start: {week_start_date}"
    )

    if not ticker:
        st.warning("Enter a ticker symbol to continue.")
        st.stop()

    # --- Main Execution ---
    if st.button("🔄 Refresh Data"):
        st.rerun()

    rth_indicators_only = include_prepost and regular_hours_only
    data = get_intraday_data(ticker, interval, period="1d", include_prepost=include_prepost)

    if data is not None and not data.empty:
        # 1. Calculate Indicators
        data = add_indicators(
            data,
            atr_window,
            vol_window,
            vwap_z_window,
            vwap_slope_window,
            adx_window,
            regular_hours_only=rth_indicators_only,
        )

        stats_data = None
        daily_stats_data = None
        if stats_period == "1d":
            stats_data = data
        else:
            stats_data = get_intraday_data(
                ticker, interval, period=stats_period, include_prepost=include_prepost
            )
            if stats_data is not None and not stats_data.empty:
                stats_data = add_indicators(
                    stats_data,
                    atr_window,
                    vol_window,
                    vwap_z_window,
                    vwap_slope_window,
                    adx_window,
                    regular_hours_only=rth_indicators_only,
                )

        if run_year_backtest:
            daily_stats_data = get_intraday_data(
                ticker, interval="1d", period="1y", include_prepost=False
            )
            if daily_stats_data is not None and not daily_stats_data.empty:
                daily_stats_data = add_indicators(
                    daily_stats_data,
                    atr_window,
                    vol_window,
                    vwap_z_window,
                    vwap_slope_window,
                    adx_window,
                    regular_hours_only=False,
                )

        market_data = None
        if use_market_filter:
            market_data = get_intraday_data(
                market_ticker,
                interval,
                period=stats_period,
                show_errors=False,
                include_prepost=include_prepost,
            )
            if market_data is not None and not market_data.empty:
                market_data = add_indicators(
                    market_data,
                    atr_window,
                    vol_window,
                    vwap_z_window,
                    vwap_slope_window,
                    adx_window,
                    regular_hours_only=rth_indicators_only,
                )

        higher_data = None
        higher_stats_data = None
        if use_higher_tf:
            higher_data = get_intraday_data(
                ticker, higher_interval, period="1d", include_prepost=include_prepost
            )
            if higher_data is not None and not higher_data.empty:
                higher_data = add_indicators(
                    higher_data,
                    atr_window,
                    vol_window,
                    vwap_z_window,
                    vwap_slope_window,
                    adx_window,
                    regular_hours_only=rth_indicators_only,
                )
            if stats_period == "1d":
                higher_stats_data = higher_data
            else:
                higher_stats_data = get_intraday_data(
                    ticker, higher_interval, period=stats_period, include_prepost=include_prepost
                )
                if higher_stats_data is not None and not higher_stats_data.empty:
                    higher_stats_data = add_indicators(
                        higher_stats_data,
                        atr_window,
                        vol_window,
                        vwap_z_window,
                        vwap_slope_window,
                        adx_window,
                        regular_hours_only=rth_indicators_only,
                    )

        # 2. Determine Current Trend
        market_bias = "OFF"
        if use_market_filter:
            market_ok_long_series, market_ok_short_series, market_bias = get_market_filter_series(
                market_data, data.index
            )
        else:
            market_ok_long_series = pd.Series(True, index=data.index)
            market_ok_short_series = pd.Series(True, index=data.index)

        session_ok_series = compute_session_ok_series(data.index, session_start, session_end)
        if not use_session_filter:
            session_ok_series = pd.Series(True, index=data.index)

        if use_regime_filter:
            if trend_only_mode:
                regime_ok_series = data["ADX"].fillna(-1) >= adx_trend
            else:
                regime_ok_series = data["ADX"].fillna(adx_chop + 1) > adx_chop
        else:
            regime_ok_series = pd.Series(True, index=data.index)

        if use_higher_tf:
            htf_ok_long_series, htf_ok_short_series, htf_bias = get_timeframe_ok_series(
                higher_data, data.index
            )
        else:
            htf_ok_long_series = pd.Series(True, index=data.index)
            htf_ok_short_series = pd.Series(True, index=data.index)
            htf_bias = "OFF"

        (
            base_long_series,
            base_short_series,
            long_ok_series,
            short_ok_series,
            volume_ok_series,
            vwap_z_ok_series,
            atr_ok_series,
            slope_ok_long_series,
            slope_ok_short_series,
        ) = compute_signal_series(
            data,
            use_volume_filter,
            vol_mult,
            use_vwap_extension,
            vwap_z_max,
            use_vwap_slope,
            use_atr_filter,
            atr_pct_min,
            session_ok_series,
            regime_ok_series,
            htf_ok_long_series,
            htf_ok_short_series,
            market_ok_long_series,
            market_ok_short_series,
        )

        current_price = float(data["Close"].iloc[-1])
        current_vwap = float(data["VWAP"].iloc[-1])
        current_macd = float(data["MACD"].iloc[-1])
        current_signal = float(data["MACD_Signal"].iloc[-1])
        current_volume = float(data["Volume"].iloc[-1])
        current_atr = data["ATR"].iloc[-1]
        current_atr_pct = data["ATR_Pct"].iloc[-1]

        macd_hist_delta_series = data["MACD_Hist"].diff()
        macd_hist_rising_series = macd_hist_delta_series > 0
        macd_hist_falling_series = macd_hist_delta_series < 0
        macd_hist_rising = bool(macd_hist_rising_series.iloc[-1])
        macd_hist_falling = bool(macd_hist_falling_series.iloc[-1])

        current_vol_ratio = None
        if pd.notna(data["Vol_Ratio"].iloc[-1]):
            current_vol_ratio = float(data["Vol_Ratio"].iloc[-1])

        current_vwap_z = data["VWAP_Z"].iloc[-1]
        if pd.notna(current_vwap_z):
            current_vwap_z = float(current_vwap_z)
        else:
            current_vwap_z = None

        current_vwap_slope = data["VWAP_Slope"].iloc[-1]
        if pd.notna(current_vwap_slope):
            current_vwap_slope = float(current_vwap_slope)
        else:
            current_vwap_slope = None

        current_adx = data["ADX"].iloc[-1]
        if pd.notna(current_adx):
            current_adx = float(current_adx)
        else:
            current_adx = None

        base_long = bool(base_long_series.iloc[-1])
        base_short = bool(base_short_series.iloc[-1])
        volume_ok = bool(volume_ok_series.iloc[-1])
        vwap_z_ok = bool(vwap_z_ok_series.iloc[-1])
        atr_ok = bool(atr_ok_series.iloc[-1])
        vwap_slope_ok_long = bool(slope_ok_long_series.iloc[-1])
        vwap_slope_ok_short = bool(slope_ok_short_series.iloc[-1])
        session_ok = bool(session_ok_series.iloc[-1])
        regime_ok = bool(regime_ok_series.iloc[-1])
        htf_ok_long = bool(htf_ok_long_series.iloc[-1])
        htf_ok_short = bool(htf_ok_short_series.iloc[-1])
        market_ok_long = bool(market_ok_long_series.iloc[-1])
        market_ok_short = bool(market_ok_short_series.iloc[-1])

        if current_adx is None:
            current_regime = "UNAVAILABLE"
        elif current_adx <= adx_chop:
            current_regime = "CHOPPY"
        elif current_adx >= adx_trend:
            current_regime = "TREND"
        else:
            current_regime = "NEUTRAL"

        include_market_score = use_market_filter and market_bias not in ["UNAVAILABLE"]
        score_max = 5
        if use_vwap_slope:
            score_max += 1
        if use_session_filter:
            score_max += 1
        if use_regime_filter:
            score_max += 1
        if use_higher_tf:
            score_max += 1
        if use_atr_filter:
            score_max += 1
        if include_market_score:
            score_max += 1

        long_score_series = (
            (data["Close"] > data["VWAP"]).astype(int)
            + (data["MACD"] > data["MACD_Signal"]).astype(int)
            + macd_hist_rising_series.astype(int)
            + volume_ok_series.astype(int)
            + vwap_z_ok_series.astype(int)
            + atr_ok_series.astype(int)
        )
        short_score_series = (
            (data["Close"] < data["VWAP"]).astype(int)
            + (data["MACD"] < data["MACD_Signal"]).astype(int)
            + macd_hist_falling_series.astype(int)
            + volume_ok_series.astype(int)
            + vwap_z_ok_series.astype(int)
            + atr_ok_series.astype(int)
        )
        if use_vwap_slope:
            long_score_series += slope_ok_long_series.astype(int)
            short_score_series += slope_ok_short_series.astype(int)
        if use_session_filter:
            long_score_series += session_ok_series.astype(int)
            short_score_series += session_ok_series.astype(int)
        if use_regime_filter:
            long_score_series += regime_ok_series.astype(int)
            short_score_series += regime_ok_series.astype(int)
        if use_higher_tf:
            long_score_series += htf_ok_long_series.astype(int)
            short_score_series += htf_ok_short_series.astype(int)
        if include_market_score:
            long_score_series += market_ok_long_series.astype(int)
            short_score_series += market_ok_short_series.astype(int)

        score_series = pd.concat([long_score_series, short_score_series], axis=1).max(axis=1)
        if min_signal_score > 0:
            score_ok_series = score_series >= min_signal_score
        else:
            score_ok_series = pd.Series(True, index=data.index)
        score_ratio_series = score_series / score_max if score_max else pd.Series(0, index=data.index)

        if use_a_plus_mode:
            a_plus_score_ok = score_ratio_series >= a_plus_min_score_ratio
            a_plus_vol_ok = data["Vol_Ratio"].isna() | (data["Vol_Ratio"] >= a_plus_min_vol_ratio)
            a_plus_atr_ok = data["ATR_Pct"].isna() | (data["ATR_Pct"] >= a_plus_min_atr_pct)
            a_plus_regime_ok = data["ADX"].fillna(-1) >= adx_trend

            a_plus_long_series = a_plus_score_ok & a_plus_vol_ok & a_plus_atr_ok & a_plus_regime_ok
            a_plus_short_series = a_plus_score_ok & a_plus_vol_ok & a_plus_atr_ok & a_plus_regime_ok
            if a_plus_require_slope:
                a_plus_long_series &= slope_ok_long_series
                a_plus_short_series &= slope_ok_short_series
            if a_plus_require_htf:
                if use_higher_tf:
                    a_plus_long_series &= htf_ok_long_series
                    a_plus_short_series &= htf_ok_short_series
                else:
                    a_plus_long_series &= False
                    a_plus_short_series &= False
            if a_plus_require_market:
                if use_market_filter:
                    a_plus_long_series &= market_ok_long_series
                    a_plus_short_series &= market_ok_short_series
                else:
                    a_plus_long_series &= False
                    a_plus_short_series &= False
        else:
            a_plus_long_series = pd.Series(True, index=data.index)
            a_plus_short_series = pd.Series(True, index=data.index)

        long_ok_series = long_ok_series & score_ok_series
        short_ok_series = short_ok_series & score_ok_series
        long_ok_series = long_ok_series & a_plus_long_series
        short_ok_series = short_ok_series & a_plus_short_series
        long_ok = bool(long_ok_series.iloc[-1])
        short_ok = bool(short_ok_series.iloc[-1])
        score_ok = bool(score_ok_series.iloc[-1])

        long_score = float(long_score_series.iloc[-1])
        short_score = float(short_score_series.iloc[-1])
        score_value = float(score_series.iloc[-1])
        score_direction = "LONG" if long_score > short_score else "SHORT" if short_score > long_score else "NEUTRAL"
        score_ratio = score_value / score_max if score_max else 0
        if score_ratio >= 0.8:
            signal_quality = "HIGH"
        elif score_ratio >= 0.6:
            signal_quality = "MEDIUM"
        else:
            signal_quality = "LOW"

        current_atr_value = float(current_atr) if pd.notna(current_atr) else None
        bars_per_day = estimate_bars_per_day(data, interval)
        expected_hold = max(int(exit_max_bars), 1)
        break_even_move, break_even_pct = estimate_break_even_move(
            current_price,
            model_option_premium_pct,
            model_option_delta,
            model_theta_decay_pct,
            model_option_spread_pct,
            option_commission,
            slippage_bps,
            expected_hold,
            bars_per_day,
        )
        expected_move = None
        edge_ok = True
        edge_ratio = None
        edge_state = "OFF"
        if require_edge:
            edge_state = "UNAVAILABLE"
            if current_atr_value is not None and break_even_move is not None:
                expected_move = atr_target_mult * current_atr_value
                required_move = break_even_move * (1 + edge_buffer_pct / 100.0)
                if required_move > 0:
                    edge_ratio = expected_move / required_move
                    edge_ok = expected_move >= required_move
                    edge_state = "PASS" if edge_ok else "FAIL"
                else:
                    edge_ok = False
                    edge_state = "FAIL"
            else:
                edge_ok = False
                edge_state = "UNAVAILABLE"

        # Logic Check
        trend = "NEUTRAL"
        color = "gray"

        # BULLISH: Price > VWAP AND MACD > Signal Line + Filters
        if long_ok:
            trend = "🚀 BULLISH MOMENTUM (Calls)"
            color = "green"
        # BEARISH: Price < VWAP AND MACD < Signal Line + Filters
        elif short_ok:
            trend = "🔻 BEARISH MOMENTUM (Puts)"
            color = "red"
        elif base_long:
            trend = "🟡 BULLISH SETUP (Filtered)"
            color = "orange"
        elif base_short:
            trend = "🟡 BEARISH SETUP (Filtered)"
            color = "orange"

        # --- Dashboard ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("VWAP Level", f"${current_vwap:.2f}", delta=f"{current_price - current_vwap:.2f}")
        col3.markdown(f"### Status: :{color}[{trend}]")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ATR", f"{current_atr:.2f}" if pd.notna(current_atr) else "n/a")
        k2.metric("Volume Ratio", f"{current_vol_ratio:.2f}x" if current_vol_ratio is not None else "n/a")
        k3.metric(
            "VWAP Z-Score",
            f"{current_vwap_z:.2f}" if current_vwap_z is not None and pd.notna(current_vwap_z) else "n/a",
        )
        k4.metric("Signal Score", f"{score_value}/{score_max}", delta=score_direction)
        if pd.notna(current_atr_pct):
            st.caption(f"ATR% of price: {float(current_atr_pct):.2f}%")

        if use_market_filter:
            st.caption(f"Market filter ({market_ticker}): {market_bias}")
        else:
            st.caption("Market filter: OFF")
        if use_higher_tf:
            st.caption(f"Higher timeframe ({higher_interval}): {htf_bias}")
        else:
            st.caption("Higher timeframe: OFF")
        if use_regime_filter:
            if current_adx is None:
                st.caption("Regime (ADX): UNAVAILABLE")
            else:
                mode_label = "Trend-only" if trend_only_mode else "Chop filter"
                st.caption(f"Regime (ADX): {current_regime} ({current_adx:.1f}) | {mode_label}")
        else:
            st.caption("Regime filter: OFF")
        if use_session_filter:
            session_state = "PASS" if session_ok else "FAIL"
            st.caption(f"Session filter: {session_state} ({session_start.strftime('%H:%M')}-{session_end.strftime('%H:%M')})")
        else:
            st.caption("Session filter: OFF")
        if use_vwap_slope and current_vwap_slope is not None:
            slope_label = "UP" if current_vwap_slope > 0 else "DOWN" if current_vwap_slope < 0 else "FLAT"
            st.caption(f"VWAP slope ({vwap_slope_window} bars): {slope_label}")

        action = "WAIT"
        action_detail = "No VWAP + MACD alignment on the latest bar."
        if long_ok:
            action = "BUY CALL"
            action_detail = "Price above VWAP, MACD bullish, filters pass."
        elif short_ok:
            action = "BUY PUT"
            action_detail = "Price below VWAP, MACD bearish, filters pass."
        elif base_long:
            action = "WATCH CALL SETUP"
            action_detail = "Bullish alignment present but filtered."
        elif base_short:
            action = "WATCH PUT SETUP"
            action_detail = "Bearish alignment present but filtered."

        if action.startswith("WATCH"):
            failures = []
            if min_signal_score > 0 and not score_ok:
                failures.append("score")
            if not volume_ok:
                failures.append("volume")
            if not vwap_z_ok:
                failures.append("extension")
            if use_atr_filter and not atr_ok:
                failures.append("volatility")
            if use_a_plus_mode:
                if score_ratio < a_plus_min_score_ratio:
                    failures.append("A+ score")
                if current_vol_ratio is None or current_vol_ratio < a_plus_min_vol_ratio:
                    failures.append("A+ volume")
                if current_atr_pct is None or pd.isna(current_atr_pct) or float(current_atr_pct) < a_plus_min_atr_pct:
                    failures.append("A+ ATR")
                if current_adx is None or current_adx < adx_trend:
                    failures.append("A+ trend")
                if a_plus_require_slope and (
                    (base_long and not vwap_slope_ok_long) or (base_short and not vwap_slope_ok_short)
                ):
                    failures.append("A+ slope")
                if a_plus_require_htf:
                    if not use_higher_tf:
                        failures.append("A+ HTF off")
                    elif (base_long and not htf_ok_long) or (base_short and not htf_ok_short):
                        failures.append("A+ HTF")
                if a_plus_require_market:
                    if not use_market_filter:
                        failures.append("A+ market off")
                    elif (base_long and not market_ok_long) or (base_short and not market_ok_short):
                        failures.append("A+ market")
                if a_plus_require_liquidity:
                    if not check_liquidity:
                        failures.append("A+ liquidity off")
            if use_vwap_slope and ((base_long and not vwap_slope_ok_long) or (base_short and not vwap_slope_ok_short)):
                failures.append("vwap slope")
            if use_session_filter and not session_ok:
                failures.append("session")
            if use_regime_filter and not regime_ok:
                failures.append("regime")
            if use_higher_tf and ((base_long and not htf_ok_long) or (base_short and not htf_ok_short)):
                failures.append("higher timeframe")
            if (base_long and not market_ok_long) or (base_short and not market_ok_short):
                failures.append("market filter")
            if failures:
                action_detail = "Alignment present but filtered: " + ", ".join(failures) + "."
            else:
                action_detail = "Alignment present but filters are neutral."

        decision = action
        decision_detail = action_detail
        if position == "Long Calls":
            if current_price < current_vwap or short_ok:
                decision = "EXIT CALLS"
                decision_detail = "Price below VWAP or bearish signal. Protect capital."
            else:
                decision = "HOLD CALLS"
                decision_detail = "Trend intact; manage with trailing stop."
        elif position == "Long Puts":
            if current_price > current_vwap or long_ok:
                decision = "EXIT PUTS"
                decision_detail = "Price above VWAP or bullish signal. Protect capital."
            else:
                decision = "HOLD PUTS"
                decision_detail = "Trend intact; manage with trailing stop."

        liquidity_state = "OFF"
        liquidity_ok_call = None
        liquidity_ok_put = None
        options_liquidity = None
        if check_liquidity:
            options_liquidity = get_options_liquidity_snapshot(
                ticker,
                dte_choice,
                current_price,
                min_open_interest,
                min_option_volume,
                max_spread_pct,
                max_spread_abs,
                atm_range_pct,
                st.session_state.options_cache_buster,
            )
            if options_liquidity is None:
                liquidity_state = "UNAVAILABLE"
            else:
                call_summary = options_liquidity.get("calls")
                put_summary = options_liquidity.get("puts")
                liquidity_ok_call = bool(call_summary and call_summary.get("ok"))
                liquidity_ok_put = bool(put_summary and put_summary.get("ok"))
                liquidity_state = "PASS" if (liquidity_ok_call or liquidity_ok_put) else "FAIL"

        if enforce_liquidity and check_liquidity and decision in ["BUY CALL", "BUY PUT"]:
            if decision == "BUY CALL" and liquidity_ok_call is False:
                decision = "WATCH CALL SETUP"
                decision_detail = "Options liquidity failed. Wait for tighter spreads and higher OI/volume."
            elif decision == "BUY PUT" and liquidity_ok_put is False:
                decision = "WATCH PUT SETUP"
                decision_detail = "Options liquidity failed. Wait for tighter spreads and higher OI/volume."
            elif decision in ["BUY CALL", "BUY PUT"] and (liquidity_ok_call is None or liquidity_ok_put is None):
                decision = "WAIT"
                decision_detail = "Options liquidity unavailable. Use caution or delay entry."

        if use_a_plus_mode and a_plus_require_liquidity and decision in ["BUY CALL", "BUY PUT"]:
            if not check_liquidity:
                decision = "WAIT"
                decision_detail = "A+ mode requires liquidity checks. Enable Options Liquidity."
            elif decision == "BUY CALL" and liquidity_ok_call is False:
                decision = "WAIT"
                decision_detail = "A+ mode blocked: call liquidity failed."
            elif decision == "BUY PUT" and liquidity_ok_put is False:
                decision = "WAIT"
                decision_detail = "A+ mode blocked: put liquidity failed."
            elif liquidity_ok_call is None or liquidity_ok_put is None:
                decision = "WAIT"
                decision_detail = "A+ mode blocked: liquidity unavailable."

        selectivity_reasons = []
        if decision in ["BUY CALL", "BUY PUT"] and high_selectivity:
            if score_ratio < min_score_ratio:
                selectivity_reasons.append("score")
            if current_regime != "TREND":
                selectivity_reasons.append("regime")
            if require_edge and not edge_ok:
                selectivity_reasons.append("edge")
            if selectivity_reasons:
                decision = "WAIT"
                decision_detail = "Selective mode blocked: " + ", ".join(selectivity_reasons) + "."

        if decision in ["BUY CALL", "BUY PUT"] and enforce_trade_budget and remaining_trades <= 0:
            decision = "WAIT"
            decision_detail = "Weekly trade limit reached. Stand down."

        trade_budget_state = "PASS" if remaining_trades > 0 else "FAIL"
        a_plus_state = "OFF"
        if use_a_plus_mode:
            a_plus_state = "PASS" if (a_plus_long_series.iloc[-1] or a_plus_short_series.iloc[-1]) else "FAIL"

        moneyness_reco = None
        moneyness_reason = None
        if decision in ["BUY CALL", "BUY PUT", "WATCH CALL SETUP", "WATCH PUT SETUP"]:
            moneyness_reco = "ATM (Balanced)"
            moneyness_reason = "Default to balanced delta/theta."
            if edge_ratio is not None:
                if edge_ratio >= 2.0 and score_ratio >= 0.85 and current_regime == "TREND":
                    moneyness_reco = "OTM (Convex)"
                    moneyness_reason = "Strong edge and trend; seek convexity."
                elif edge_ratio >= 1.2:
                    moneyness_reco = "ATM (Balanced)"
                    moneyness_reason = "Edge supports balanced exposure."
                elif edge_ratio >= 1.0:
                    moneyness_reco = "ITM (Higher Delta)"
                    moneyness_reason = "Edge is thin; higher delta helps."
                else:
                    moneyness_reco = "SKIP"
                    moneyness_reason = "Edge vs costs is too small."
            if current_regime == "CHOPPY" and moneyness_reco == "OTM (Convex)":
                moneyness_reco = "ITM (Higher Delta)"
                moneyness_reason = "Choppy regime favors higher delta."
            if dte_choice == "0DTE only" and moneyness_reco == "OTM (Convex)":
                moneyness_reco = "ATM (Balanced)"
                moneyness_reason = "0DTE favors ATM/ITM to reduce theta drag."

        st.subheader("Decision Now")
        if decision in ["BUY CALL", "HOLD CALLS"]:
            st.success(f"{decision} - {decision_detail}")
        elif decision in ["BUY PUT", "EXIT CALLS", "EXIT PUTS"]:
            st.error(f"{decision} - {decision_detail}")
        elif decision in ["WATCH CALL SETUP", "WATCH PUT SETUP"]:
            st.warning(f"{decision} - {decision_detail}")
        else:
            st.info(f"{decision} - {decision_detail}")

        st.subheader("Decision Engine (Optimal Strategy)")
        strategy_name = "Stand Aside"
        strategy_reason = "No high-quality edge detected."
        if decision in ["BUY CALL", "BUY PUT"]:
            if use_a_plus_mode and (a_plus_long_series.iloc[-1] or a_plus_short_series.iloc[-1]):
                strategy_name = "A+ Trend Continuation (Strict)"
                strategy_reason = "A+ filters passed; prioritize the highest-quality trend setups."
            elif current_regime == "TREND" and score_ratio >= 0.75:
                strategy_name = "Trend Continuation (VWAP + MACD)"
                strategy_reason = "Trend regime with strong score alignment."
            elif current_regime == "CHOPPY" and abs(current_vwap_z or 0) >= vwap_z_max:
                strategy_name = "Mean Reversion to VWAP"
                strategy_reason = "Choppy regime with extended VWAP Z-score."
            elif current_regime == "NEUTRAL" and use_atr_filter and atr_ok:
                strategy_name = "Volatility Expansion"
                strategy_reason = "Neutral regime but sufficient ATR expansion."
            else:
                strategy_name = "Selective Trend"
                strategy_reason = "Edge present but regime mixed."

        strat_steps = []
        if decision in ["BUY CALL", "BUY PUT"]:
            strat_steps = [
                f"Entry: {decision} on a candle close with VWAP + MACD alignment.",
                f"Risk: Use ATR stop {atr_stop_mult:.2f}x; tighten if choppy ({choppy_stop_factor:.2f}x).",
                f"Target: Aim for {atr_target_mult:.2f}x ATR; trail with {trail_atr_mult:.2f}x ATR after +1R.",
                "Filters: Confirm volume, VWAP extension, ATR%, and higher timeframe if enabled.",
                f"Contracts: {moneyness_reco or 'ATM/ITM'} with {dte_choice} preference.",
            ]
        else:
            strat_steps = [
                "No trade: wait for VWAP + MACD alignment with passing filters.",
                "If you must engage, use reduced size and tight stops.",
                "Focus on A+ setups only (score, ATR%, volume, trend).",
            ]

        st.markdown(
            "\n".join(
                [
                    f"- **Strategy:** {strategy_name}",
                    f"- **Why:** {strategy_reason}",
                ]
                + [f"- {step}" for step in strat_steps]
            )
        )

        if dte_choice == "0DTE only":
            dte_caption = "Options window: 0DTE only; prioritize liquidity and reduce size."
        elif dte_choice == "0-7 DTE":
            dte_caption = "Options window: 0-7 DTE; prefer 1-7 DTE unless liquidity is exceptional."
        else:
            dte_caption = "Options window: 1-7 DTE for stability; 0DTE optional only if you switch to it."
        st.caption(dte_caption)
        if dte_choice == "0DTE only" and not check_liquidity:
            st.warning("0DTE selected without liquidity checks. Enable liquidity filter or reduce size.")

        contract_suggestion = None
        if decision == "BUY CALL":
            if dte_choice == "0DTE only":
                contract_suggestion = "Suggested contract: 0DTE call."
            elif dte_choice == "0-7 DTE":
                contract_suggestion = "Suggested contract: 0DTE call or 1-7 DTE call (liquidity decides)."
            else:
                contract_suggestion = "Suggested contract: 1-7 DTE call."
        elif decision == "BUY PUT":
            if dte_choice == "0DTE only":
                contract_suggestion = "Suggested contract: 0DTE put."
            elif dte_choice == "0-7 DTE":
                contract_suggestion = "Suggested contract: 0DTE put or 1-7 DTE put (liquidity decides)."
            else:
                contract_suggestion = "Suggested contract: 1-7 DTE put."
        if contract_suggestion:
            st.caption(contract_suggestion)

        if moneyness_reco:
            if moneyness_reco == "SKIP":
                st.caption(f"Moneyness suggestion: SKIP. {moneyness_reason}")
            else:
                delta_band = MONEYNESS_DELTA_RANGES.get(moneyness_reco, "n/a")
                st.caption(
                    f"Moneyness suggestion: {moneyness_reco} (delta {delta_band}). {moneyness_reason}"
                )

        if moneyness_preset:
            st.caption(
                f"Moneyness preset active: {moneyness_profile} "
                f"(delta {model_option_delta:.2f}, premium {model_option_premium_pct:.1f}%, "
                f"theta {model_theta_decay_pct:.1f}%/day)."
            )

        st.caption(f"Signal quality: {signal_quality} ({score_value}/{score_max}).")
        if min_signal_score > 0 and min_signal_score > score_max:
            st.warning("Min Signal Score exceeds max score. No trades will qualify.")

        st.subheader("Weekly Trade Budget")
        t1, t2, t3 = st.columns(3)
        t1.metric("Trades This Week", f"{trades_this_week_count}")
        t2.metric("Remaining Trades", f"{remaining_trades}")
        t3.metric("Weekly Limit", f"{weekly_trade_limit}")
        if enforce_trade_budget:
            st.caption("Weekly trade limit enforced. Log trades to track usage.")
        if high_selectivity:
            st.caption(
                f"High-selectivity mode: score >= {min_score_ratio:.2f}, regime TREND, edge filter {edge_state}."
            )
        if use_a_plus_mode:
            st.caption(
                "A+ mode: requires high score, strong volume, higher ATR%, and TREND regime; "
                "optional slope/HTF/market alignment."
            )
        if require_edge and break_even_move is not None and expected_move is not None:
            st.caption(
                f"Edge check: expected move ${expected_move:.2f} vs required ${break_even_move * (1 + edge_buffer_pct / 100.0):.2f}."
            )
        if decision in ["BUY CALL", "BUY PUT"] and remaining_trades > 0:
            if st.button("Log This Trade"):
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "decision": decision,
                    "price": round(current_price, 2),
                    "score": round(score_value, 2),
                    "quality": signal_quality,
                    "regime": current_regime,
                    "dte": dte_choice,
                }
                st.session_state.trade_log.append(trade_record)
                if persist_trade_log:
                    save_trade_log(trade_log_path, st.session_state.trade_log)
                st.success("Trade logged.")
                st.rerun()
        elif decision in ["BUY CALL", "BUY PUT"] and remaining_trades <= 0:
            st.info("Weekly trade limit reached. No slots remaining.")

        if trades_this_week:
            trade_df = pd.DataFrame(trades_this_week)
            if "timestamp" in trade_df.columns:
                trade_df = trade_df.sort_values("timestamp", ascending=False)
            st.dataframe(trade_df.head(10))

        st.subheader("Decision Breakdown")
        if base_long or long_ok:
            signal_bias = "LONG"
        elif base_short or short_ok:
            signal_bias = "SHORT"
        else:
            signal_bias = "NONE"
        vwap_state = "LONG" if current_price > current_vwap else "SHORT" if current_price < current_vwap else "FLAT"
        macd_state = "LONG" if current_macd > current_signal else "SHORT" if current_macd < current_signal else "FLAT"
        macd_mom_state = "RISING" if macd_hist_rising else "FALLING" if macd_hist_falling else "FLAT"
        volume_state = "PASS" if volume_ok else "FAIL"
        vwap_ext_state = "PASS" if vwap_z_ok else "FAIL"
        atr_state = "PASS" if atr_ok else "FAIL"
        session_state = "PASS" if session_ok else "FAIL"
        regime_state = "PASS" if regime_ok else "FAIL"
        score_state = "PASS" if score_ok else "FAIL"

        if use_vwap_slope:
            if base_long or long_ok:
                slope_state = "PASS" if vwap_slope_ok_long else "FAIL"
            elif base_short or short_ok:
                slope_state = "PASS" if vwap_slope_ok_short else "FAIL"
            else:
                slope_state = "NEUTRAL"
        else:
            slope_state = "OFF"

        if use_higher_tf:
            if base_long or long_ok:
                htf_state = "PASS" if htf_ok_long else "FAIL"
            elif base_short or short_ok:
                htf_state = "PASS" if htf_ok_short else "FAIL"
            else:
                htf_state = "NEUTRAL"
        else:
            htf_state = "OFF"

        if not use_volume_filter:
            volume_state = "OFF"
        if not use_vwap_extension:
            vwap_ext_state = "OFF"
        if not use_atr_filter:
            atr_state = "OFF"
        if not use_session_filter:
            session_state = "OFF"
        if not use_regime_filter:
            regime_state = "OFF"
        if min_signal_score <= 0:
            score_state = "OFF"

        if use_market_filter:
            if base_long or long_ok:
                market_state = "PASS" if market_ok_long else "FAIL"
            elif base_short or short_ok:
                market_state = "PASS" if market_ok_short else "FAIL"
            else:
                market_state = "NEUTRAL"
        else:
            market_state = "OFF"

        if check_liquidity:
            if options_liquidity is None:
                liquidity_state = "UNAVAILABLE"
            else:
                if decision == "BUY CALL":
                    liquidity_state = "PASS" if liquidity_ok_call else "FAIL"
                elif decision == "BUY PUT":
                    liquidity_state = "PASS" if liquidity_ok_put else "FAIL"
                else:
                    liquidity_state = "PASS" if (liquidity_ok_call or liquidity_ok_put) else "FAIL"
        else:
            liquidity_state = "OFF"

        st.markdown(
            "\n".join(
                [
                    f"- Signal bias: {signal_bias}",
                    f"- VWAP alignment: {vwap_state}",
                    f"- MACD alignment: {macd_state}",
                    f"- MACD momentum: {macd_mom_state}",
                    f"- Score filter: {score_state}",
                    f"- Volume filter: {volume_state}",
                    f"- VWAP extension filter: {vwap_ext_state}",
                    f"- ATR volatility filter: {atr_state}",
                    f"- A+ mode: {a_plus_state}",
                    f"- VWAP slope filter: {slope_state}",
                    f"- Session filter: {session_state}",
                    f"- Regime filter: {regime_state}",
                    f"- Higher timeframe: {htf_state}",
                    f"- Market filter: {market_state}",
                    f"- Edge vs costs: {edge_state}",
                    f"- Trade budget: {trade_budget_state} (remaining {remaining_trades})",
                    f"- Options liquidity: {liquidity_state}",
                ]
            )
        )

        st.subheader("Recent Signals")
        signal_long_series = long_ok_series
        signal_short_series = short_ok_series
        signal_basis = "filtered signals"
        if not (signal_long_series | signal_short_series).any():
            signal_long_series = base_long_series
            signal_short_series = base_short_series
            signal_basis = "base setups"

        recent_signals = data[signal_long_series | signal_short_series].tail(3)
        if recent_signals.empty:
            st.info("No recent signals to display.")
        else:
            if score_max:
                score_ratio_series = score_series / score_max
            else:
                score_ratio_series = pd.Series(0, index=score_series.index)

            def quality_from_ratio(value):
                if value >= 0.8:
                    return "HIGH"
                if value >= 0.6:
                    return "MEDIUM"
                return "LOW"

            rows = []
            for idx in recent_signals.index:
                is_long = bool(signal_long_series.loc[idx])
                signal_label = "BUY CALL" if is_long else "BUY PUT"
                score_value_row = float(score_series.loc[idx]) if pd.notna(score_series.loc[idx]) else 0.0
                score_ratio_row = float(score_ratio_series.loc[idx]) if pd.notna(score_ratio_series.loc[idx]) else 0.0
                rows.append(
                    {
                        "Signal": signal_label,
                        "Price": float(recent_signals.loc[idx, "Close"]),
                        "VWAP": float(recent_signals.loc[idx, "VWAP"]),
                        "Score": f"{score_value_row:.0f}/{score_max}",
                        "Quality": quality_from_ratio(score_ratio_row),
                        "VolRatio": float(recent_signals.loc[idx, "Vol_Ratio"])
                        if pd.notna(recent_signals.loc[idx, "Vol_Ratio"])
                        else None,
                        "VWAP_Z": float(recent_signals.loc[idx, "VWAP_Z"])
                        if pd.notna(recent_signals.loc[idx, "VWAP_Z"])
                        else None,
                        "ATR%": float(recent_signals.loc[idx, "ATR_Pct"])
                        if pd.notna(recent_signals.loc[idx, "ATR_Pct"])
                        else None,
                    }
                )
            signal_table = pd.DataFrame(rows, index=recent_signals.index)
            st.dataframe(signal_table)
            st.caption(f"Showing last 3 {signal_basis}.")

        if check_liquidity:
            st.subheader("Options Liquidity Snapshot")
            if options_liquidity is None:
                st.info("Options liquidity unavailable. Consider disabling the filter or retry later.")
            else:
                exp = options_liquidity.get("expiration")
                dte = options_liquidity.get("dte")
                fetched_at = options_liquidity.get("fetched_at")
                time_note = f" | updated {fetched_at}" if fetched_at else ""
                st.caption(f"Expiration: {exp} (DTE: {dte}){time_note}")
                st.caption("Options chain cached up to ~60s. Use Refresh Options Data for latest.")

                call_summary = options_liquidity.get("calls")
                put_summary = options_liquidity.get("puts")

                c1, c2, c3 = st.columns(3)
                if call_summary:
                    c1.metric("Call Spread (median)", f"${call_summary['median_spread_abs']:.2f}")
                    c2.metric("Call OI (max)", f"{call_summary['max_oi']:.0f}")
                    c3.metric("Call Volume (max)", f"{call_summary['max_vol']:.0f}")
                    st.caption(
                        f"Call pass rate: {call_summary['pass_rate'] * 100:.1f}% across {call_summary['count']} contracts."
                    )
                else:
                    st.info("Call liquidity data unavailable for the selected filters.")

                p1, p2, p3 = st.columns(3)
                if put_summary:
                    p1.metric("Put Spread (median)", f"${put_summary['median_spread_abs']:.2f}")
                    p2.metric("Put OI (max)", f"{put_summary['max_oi']:.0f}")
                    p3.metric("Put Volume (max)", f"{put_summary['max_vol']:.0f}")
                    st.caption(
                        f"Put pass rate: {put_summary['pass_rate'] * 100:.1f}% across {put_summary['count']} contracts."
                    )
                else:
                    st.info("Put liquidity data unavailable for the selected filters.")

        st.subheader("Moneyness Playbook (ITM/ATM/OTM)")
        playbook_lines = [
            f"- ITM (Higher Delta): {MONEYNESS_NOTES['ITM (Higher Delta)']} (delta {MONEYNESS_DELTA_RANGES['ITM (Higher Delta)']}).",
            f"- ATM (Balanced): {MONEYNESS_NOTES['ATM (Balanced)']} (delta {MONEYNESS_DELTA_RANGES['ATM (Balanced)']}).",
            f"- OTM (Convex): {MONEYNESS_NOTES['OTM (Convex)']} (delta {MONEYNESS_DELTA_RANGES['OTM (Convex)']}).",
            "- 0DTE: lean ITM/ATM unless the edge ratio is very high and liquidity is strong.",
            "- 1-7 DTE: ATM/ITM is usually more stable; OTM is for strong momentum only.",
        ]
        st.markdown("\n".join(playbook_lines))

        st.subheader("Explainer (How to Act)")
        break_even_note = None
        if break_even_move is not None and break_even_pct is not None:
            break_even_note = (
                f"- Break-even move (approx): ${break_even_move:.2f} "
                f"({break_even_pct:.2f}%) for ~{expected_hold} bars hold."
            )
        explainer_lines = [
            "- Decision Now: BUY/HOLD/EXIT/WAIT is the primary action. Follow it unless risk rules conflict.",
            "- Signal quality: HIGH means most filters pass; MEDIUM/LOW means reduce size or wait.",
            "- Decision Breakdown: PASS = filter allows a trade; FAIL = stand down or tighten risk.",
            "- Risk Snapshot: Use the stop/target levels and trailing stop to manage loss and lock gains.",
            "- Options liquidity: PASS required for 0DTE; FAIL means skip or switch to 1-7 DTE.",
            "- Liquidity data: cached ~60s; use Refresh Options Data if you need the latest chain.",
            "- Options backtest: uses delta + theta + spread/commission penalties; overlap is one trade at a time.",
            "- Backtest Snapshot: sanity check only; tighten filters if hit rate is weak.",
            "- Weekly trade budget: limit new entries; log trades to track remaining slots.",
            "- Selectivity mode: requires high score, TREND regime, and edge vs costs if enabled.",
            "- Moneyness: ITM for thinner edge/chop, ATM for balanced trends, OTM only for strong momentum.",
            "- Trailing stop backtest: lower trail ATR/shorter lookback tightens exits; raise target to test trailing-only.",
            "- Pre/post market: useful for context, but use regular-session indicators to avoid VWAP distortion.",
        ]
        if break_even_note:
            explainer_lines.append(break_even_note)
        st.markdown("\n".join(explainer_lines))

        st.subheader("Risk Snapshot")
        signal_long_series = long_ok_series
        signal_short_series = short_ok_series
        signal_basis = "filtered signals"
        if not (signal_long_series | signal_short_series).any():
            signal_long_series = base_long_series
            signal_short_series = base_short_series
            signal_basis = "base setups"

        signal_any = signal_long_series | signal_short_series
        last_signal_index = None
        last_signal_pos = None
        last_signal_direction = None
        entry_price = current_price
        atr_for_risk = current_atr_value
        risk_basis = "Current bar alignment"

        if signal_any.any():
            last_signal_index = signal_any[signal_any].index[-1]
            last_signal_pos = data.index.get_indexer([last_signal_index])[0]
            last_signal_direction = "long" if bool(signal_long_series.loc[last_signal_index]) else "short"
            entry_price = float(data.loc[last_signal_index, "Close"])
            atr_at_entry = data.loc[last_signal_index, "ATR"]
            if pd.notna(atr_at_entry):
                atr_for_risk = float(atr_at_entry)
            risk_basis = f"Last {signal_basis} at {last_signal_index}"

        risk_direction = last_signal_direction
        if risk_direction is None:
            risk_direction = "long" if base_long else "short" if base_short else None

        if risk_direction and atr_for_risk is not None and atr_for_risk > 0:
            stop_mult = atr_stop_mult
            if use_choppy_tighten and current_regime == "CHOPPY":
                stop_mult = atr_stop_mult * choppy_stop_factor
            if risk_direction == "long":
                stop = entry_price - (stop_mult * atr_for_risk)
                target = entry_price + (atr_target_mult * atr_for_risk)
                risk = entry_price - stop
                reward = target - entry_price
            else:
                stop = entry_price + (stop_mult * atr_for_risk)
                target = entry_price - (atr_target_mult * atr_for_risk)
                risk = stop - entry_price
                reward = entry_price - target
            rr = (reward / risk) if risk > 0 else None

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Entry", f"${entry_price:.2f}")
            r2.metric("ATR", f"{atr_for_risk:.2f}")
            r3.metric("Stop", f"${stop:.2f}")
            r4.metric("Target", f"${target:.2f}")
            if rr is not None:
                st.caption(f"Estimated R:R = {rr:.2f} using ATR multipliers.")
            st.caption(f"Basis: {risk_basis}")
            if use_choppy_tighten and current_regime == "CHOPPY":
                st.caption(f"Choppy regime: stop tightened to {stop_mult:.2f}x ATR.")

            risk_budget = account_size * (risk_pct / 100.0)
            stop_distance = stop_mult * atr_for_risk
            risk_per_contract = stop_distance * model_option_delta * 100
            max_contracts = int(risk_budget // risk_per_contract) if risk_per_contract > 0 else None
            s1, s2, s3 = st.columns(3)
            s1.metric("Risk Budget", f"${risk_budget:.0f}")
            s2.metric(
                "Risk/Contract (est)",
                f"${risk_per_contract:.0f}" if risk_per_contract > 0 else "n/a",
            )
            s3.metric("Max Contracts (est)", f"{max_contracts}" if max_contracts is not None else "n/a")

            if use_trailing_stop and current_atr_value is not None and current_atr_value > 0:
                if last_signal_pos is not None:
                    trail_start = max(last_signal_pos, len(data) - trail_lookback)
                    trail_basis = f"since signal ({signal_basis})"
                else:
                    trail_start = max(len(data) - trail_lookback, 0)
                    trail_basis = f"last {trail_lookback} bars"

                if risk_direction == "long":
                    high_since = data["High"].iloc[trail_start:].max()
                    trail_stop = high_since - (trail_atr_mult * current_atr_value)
                    trail_ref = f"highest high {trail_basis} ({high_since:.2f})"
                else:
                    low_since = data["Low"].iloc[trail_start:].min()
                    trail_stop = low_since + (trail_atr_mult * current_atr_value)
                    trail_ref = f"lowest low {trail_basis} ({low_since:.2f})"

                if pd.notna(trail_stop):
                    st.caption(
                        f"Trailing stop suggestion: ${trail_stop:.2f} using "
                        f"{trail_atr_mult:.2f}x ATR from the {trail_ref}."
                    )
        else:
            st.info("No valid signal alignment or ATR data to compute risk levels.")

        # --- Plotting (Dual Axis) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # Plot 1: Price + VWAP
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)

        fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], 
                                 line=dict(color='orange', width=2), name='VWAP'), row=1, col=1)
        if show_vwap_bands:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['VWAP'] + data['VWAP_STD'],
                line=dict(color='orange', width=1, dash='dot'), name='VWAP +1 SD'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['VWAP'] - data['VWAP_STD'],
                line=dict(color='orange', width=1, dash='dot'), name='VWAP -1 SD'
            ), row=1, col=1)

        # Plot 2: MACD
        # Green histogram for positive, Red for negative
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]

        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color='blue', width=1), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'), row=2, col=1)

        # Layout Updates
        fig.update_layout(title=f"Intraday Analysis: {ticker} ({interval})", 
                          height=700, xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

        # --- Backtest Snapshot ---
        st.subheader("Backtest Snapshot")
        if stats_data is None or stats_data.empty:
            st.info("Backtest data unavailable for the selected period.")
        else:
            if use_market_filter:
                stats_market_ok_long, stats_market_ok_short, stats_market_bias = get_market_filter_series(
                    market_data, stats_data.index
                )
            else:
                stats_market_ok_long = None
                stats_market_ok_short = None
                stats_market_bias = "UNAVAILABLE"

            if use_session_filter:
                stats_session_ok = compute_session_ok_series(stats_data.index, session_start, session_end)
            else:
                stats_session_ok = None

            if use_regime_filter:
                if trend_only_mode:
                    stats_regime_ok = stats_data["ADX"].fillna(-1) >= adx_trend
                else:
                    stats_regime_ok = stats_data["ADX"].fillna(adx_chop + 1) > adx_chop
            else:
                stats_regime_ok = None

            if use_higher_tf:
                stats_htf_ok_long, stats_htf_ok_short, _ = get_timeframe_ok_series(
                    higher_stats_data, stats_data.index
                )
            else:
                stats_htf_ok_long = None
                stats_htf_ok_short = None

            (
                stats_base_long,
                stats_base_short,
                stats_long_ok,
                stats_short_ok,
                stats_volume_ok,
                stats_vwap_ok,
                stats_atr_ok,
                stats_slope_ok_long,
                stats_slope_ok_short,
            ) = compute_signal_series(
                stats_data,
                use_volume_filter,
                vol_mult,
                use_vwap_extension,
                vwap_z_max,
                use_vwap_slope,
                use_atr_filter,
                atr_pct_min,
                stats_session_ok,
                stats_regime_ok,
                stats_htf_ok_long,
                stats_htf_ok_short,
                stats_market_ok_long,
                stats_market_ok_short,
            )

            stats_macd_hist_delta = stats_data["MACD_Hist"].diff()
            stats_macd_rising = stats_macd_hist_delta > 0
            stats_macd_falling = stats_macd_hist_delta < 0
            stats_long_score = (
                (stats_data["Close"] > stats_data["VWAP"]).astype(int)
                + (stats_data["MACD"] > stats_data["MACD_Signal"]).astype(int)
                + stats_macd_rising.astype(int)
                + stats_volume_ok.astype(int)
                + stats_vwap_ok.astype(int)
                + stats_atr_ok.astype(int)
            )
            stats_short_score = (
                (stats_data["Close"] < stats_data["VWAP"]).astype(int)
                + (stats_data["MACD"] < stats_data["MACD_Signal"]).astype(int)
                + stats_macd_falling.astype(int)
                + stats_volume_ok.astype(int)
                + stats_vwap_ok.astype(int)
                + stats_atr_ok.astype(int)
            )
            if use_vwap_slope:
                stats_long_score += stats_slope_ok_long.astype(int)
                stats_short_score += stats_slope_ok_short.astype(int)
            if use_session_filter:
                stats_long_score += stats_session_ok.astype(int)
                stats_short_score += stats_session_ok.astype(int)
            if use_regime_filter:
                stats_long_score += stats_regime_ok.astype(int)
                stats_short_score += stats_regime_ok.astype(int)
            if use_higher_tf:
                stats_long_score += stats_htf_ok_long.astype(int)
                stats_short_score += stats_htf_ok_short.astype(int)
            if use_market_filter and stats_market_bias not in ["UNAVAILABLE"]:
                stats_long_score += stats_market_ok_long.astype(int)
                stats_short_score += stats_market_ok_short.astype(int)

            stats_score_max = 5
            if use_vwap_slope:
                stats_score_max += 1
            if use_session_filter:
                stats_score_max += 1
            if use_regime_filter:
                stats_score_max += 1
            if use_higher_tf:
                stats_score_max += 1
            if use_atr_filter:
                stats_score_max += 1
            if use_market_filter and stats_market_bias not in ["UNAVAILABLE"]:
                stats_score_max += 1

            stats_score_series = pd.concat([stats_long_score, stats_short_score], axis=1).max(axis=1)
            if min_signal_score > 0:
                stats_score_ok = stats_score_series >= min_signal_score
                stats_long_ok = stats_long_ok & stats_score_ok
                stats_short_ok = stats_short_ok & stats_score_ok

            if use_a_plus_mode:
                stats_score_ratio_series = stats_score_series / stats_score_max if stats_score_max else 0
                stats_vol_ok_strict = stats_data["Vol_Ratio"].isna() | (
                    stats_data["Vol_Ratio"] >= a_plus_min_vol_ratio
                )
                stats_atr_ok_strict = stats_data["ATR_Pct"].isna() | (
                    stats_data["ATR_Pct"] >= a_plus_min_atr_pct
                )
                stats_regime_strict = stats_data["ADX"].fillna(-1) >= adx_trend

                stats_long_plus = (
                    (stats_score_ratio_series >= a_plus_min_score_ratio)
                    & stats_vol_ok_strict
                    & stats_atr_ok_strict
                    & stats_regime_strict
                )
                stats_short_plus = stats_long_plus.copy()
                if a_plus_require_slope:
                    stats_long_plus &= stats_slope_ok_long
                    stats_short_plus &= stats_slope_ok_short
                if a_plus_require_htf:
                    if use_higher_tf:
                        stats_long_plus &= stats_htf_ok_long
                        stats_short_plus &= stats_htf_ok_short
                    else:
                        stats_long_plus &= False
                        stats_short_plus &= False
                if a_plus_require_market:
                    if use_market_filter:
                        stats_long_plus &= stats_market_ok_long
                        stats_short_plus &= stats_market_ok_short
                    else:
                        stats_long_plus &= False
                        stats_short_plus &= False

                stats_long_ok = stats_long_ok & stats_long_plus
                stats_short_ok = stats_short_ok & stats_short_plus

            stats_buy = stats_long_ok.fillna(False)
            stats_sell = stats_short_ok.fillna(False)
            stats_basis = "filtered signals"
            if stats_use_setups and (stats_buy | stats_sell).sum() == 0:
                stats_buy = stats_base_long.fillna(False)
                stats_sell = stats_base_short.fillna(False)
                stats_basis = "base setups"

            stats_mask = stats_buy | stats_sell
            signal_count = int(stats_mask.sum())
            long_count = int(stats_buy.sum())
            short_count = int(stats_sell.sum())

            positions = []
            directions = []
            for idx, (is_long, is_short) in enumerate(zip(stats_buy.tolist(), stats_sell.tolist())):
                if is_long or is_short:
                    positions.append(idx)
                    directions.append("long" if is_long else "short")

            forward_returns = []
            for pos, direction in zip(positions, directions):
                exit_pos = pos + lookahead_bars
                if exit_pos >= len(stats_data):
                    continue
                entry = stats_data["Close"].iloc[pos]
                exit_price = stats_data["Close"].iloc[exit_pos]
                if pd.isna(entry) or pd.isna(exit_price):
                    continue
                ret, _, _ = apply_slippage(entry, exit_price, direction, slippage_bps)
                forward_returns.append(ret)

            if signal_count == 0:
                st.info("No signals found for the backtest period.")
            elif not forward_returns:
                st.info("Not enough future bars to compute backtest stats. Increase the period.")
            else:
                hit_rate = sum(r > 0 for r in forward_returns) / len(forward_returns)
                avg_return = sum(forward_returns) / len(forward_returns)
                median_return = float(pd.Series(forward_returns).median())
                best_return = max(forward_returns)
                worst_return = min(forward_returns)
                expectancy = compute_expectancy(forward_returns)

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Signals", f"{signal_count}")
                b2.metric("Hit Rate", f"{hit_rate * 100:.1f}%")
                b3.metric(f"Avg Return ({lookahead_bars} bars)", f"{avg_return * 100:.2f}%")
                b4.metric("Median Return", f"{median_return * 100:.2f}%")
                st.caption(
                    f"Signals: {long_count} long / {short_count} short. Basis: {stats_basis}."
                )
                st.caption(
                    f"Best/Worst forward return: {best_return * 100:.2f}% / {worst_return * 100:.2f}%."
                )
                if expectancy:
                    pf = expectancy["profit_factor"]
                    pf_text = f"{pf:.2f}" if pf is not None else "n/a"
                    st.caption(
                        f"Expectancy per trade: {expectancy['expectancy'] * 100:.2f}% | "
                        f"Avg win/loss: {expectancy['avg_win'] * 100:.2f}% / {expectancy['avg_loss'] * 100:.2f}% | "
                        f"Profit factor: {pf_text}."
                    )
                st.caption(
                    "Backtest uses underlying price returns, not option P/L. "
                    "Stats are sensitive to interval and sample size."
                )
                st.caption(f"Slippage applied: {slippage_bps:.1f} bps per entry/exit.")

            # --- Daily Backtest (1Y) ---
            if run_year_backtest:
                st.subheader("Daily Backtest (1Y)")
                if daily_stats_data is None or daily_stats_data.empty:
                    st.info("Daily backtest data unavailable.")
                else:
                    (
                        daily_base_long,
                        daily_base_short,
                        daily_long_ok,
                        daily_short_ok,
                        _daily_volume_ok,
                        _daily_vwap_ok,
                        _daily_atr_ok,
                        _daily_slope_ok_long,
                        _daily_slope_ok_short,
                    ) = compute_signal_series(
                        daily_stats_data,
                        use_volume_filter,
                        vol_mult,
                        use_vwap_extension,
                        vwap_z_max,
                        use_vwap_slope,
                        use_atr_filter,
                        atr_pct_min,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )

                    daily_buy = daily_long_ok.fillna(False)
                    daily_sell = daily_short_ok.fillna(False)
                    daily_mask = daily_buy | daily_sell
                    daily_positions = []
                    daily_dirs = []
                    for idx, (is_long, is_short) in enumerate(
                        zip(daily_buy.tolist(), daily_sell.tolist())
                    ):
                        if is_long or is_short:
                            daily_positions.append(idx)
                            daily_dirs.append("long" if is_long else "short")

                    daily_returns = []
                    daily_lookahead = max(int(lookahead_bars), 1)
                    for pos, direction in zip(daily_positions, daily_dirs):
                        exit_pos = pos + daily_lookahead
                        if exit_pos >= len(daily_stats_data):
                            continue
                        entry = daily_stats_data["Close"].iloc[pos]
                        exit_price = daily_stats_data["Close"].iloc[exit_pos]
                        if pd.isna(entry) or pd.isna(exit_price):
                            continue
                        ret, _, _ = apply_slippage(entry, exit_price, direction, slippage_bps)
                        daily_returns.append(ret)

                    if not daily_returns:
                        st.info("Not enough daily bars to compute 1Y backtest stats.")
                    else:
                        d_hit = sum(r > 0 for r in daily_returns) / len(daily_returns)
                        d_avg = sum(daily_returns) / len(daily_returns)
                        d_med = float(pd.Series(daily_returns).median())
                        d_best = max(daily_returns)
                        d_worst = min(daily_returns)
                        d_exp = compute_expectancy(daily_returns)
                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("Signals", f"{len(daily_returns)}")
                        d2.metric("Hit Rate", f"{d_hit * 100:.1f}%")
                        d3.metric(f"Avg Return ({daily_lookahead} days)", f"{d_avg * 100:.2f}%")
                        d4.metric("Median Return", f"{d_med * 100:.2f}%")
                        st.caption(
                            f"Best/Worst forward return: {d_best * 100:.2f}% / {d_worst * 100:.2f}%."
                        )
                        if d_exp:
                            pf = d_exp["profit_factor"]
                            pf_text = f"{pf:.2f}" if pf is not None else "n/a"
                            st.caption(
                                f"Expectancy per trade: {d_exp['expectancy'] * 100:.2f}% | "
                                f"Avg win/loss: {d_exp['avg_win'] * 100:.2f}% / "
                                f"{d_exp['avg_loss'] * 100:.2f}% | Profit factor: {pf_text}."
                            )
                        st.caption(
                            "Daily backtest uses daily bars (1Y) and the same filters; "
                            "intraday dynamics will differ."
                        )

            # --- Options Backtest (Approx) ---
            st.subheader("Options Backtest (Approx)")
            if not run_options_backtest:
                st.info("Options backtest is disabled in the sidebar.")
            elif signal_count == 0:
                st.info("No signals available for options backtest.")
            else:
                bars_per_day = estimate_bars_per_day(stats_data, interval)
                option_trades = []
                i = 0
                stats_buy_list = stats_buy.tolist()
                stats_sell_list = stats_sell.tolist()
                while i < len(stats_data):
                    is_buy = stats_buy_list[i]
                    is_sell = stats_sell_list[i]
                    if not (is_buy or is_sell):
                        i += 1
                        continue

                    direction = "long" if is_buy else "short"
                    entry = stats_data["Close"].iloc[i]
                    atr_value = stats_data["ATR"].iloc[i]
                    if pd.isna(entry) or pd.isna(atr_value) or atr_value <= 0:
                        i += 1
                        continue

                    stop_mult = atr_stop_mult
                    if use_choppy_tighten:
                        adx_value = stats_data["ADX"].iloc[i]
                        if pd.notna(adx_value) and adx_value <= adx_chop:
                            stop_mult = atr_stop_mult * choppy_stop_factor

                    if direction == "long":
                        stop = entry - (stop_mult * atr_value)
                        target = entry + (atr_target_mult * atr_value)
                    else:
                        stop = entry + (stop_mult * atr_value)
                        target = entry - (atr_target_mult * atr_value)

                    exit_idx = min(i + exit_max_bars, len(stats_data) - 1)
                    exit_price = stats_data["Close"].iloc[exit_idx]
                    hold_bars = exit_idx - i
                    outcome = "timeout"

                    for step in range(1, exit_max_bars + 1):
                        idx = i + step
                        if idx >= len(stats_data):
                            break
                        high = stats_data["High"].iloc[idx]
                        low = stats_data["Low"].iloc[idx]
                        if pd.isna(high) or pd.isna(low):
                            continue
                        if direction == "long":
                            hit_stop = low <= stop
                            hit_target = high >= target
                        else:
                            hit_stop = high >= stop
                            hit_target = low <= target
                        if hit_stop or hit_target:
                            if hit_stop:
                                outcome = "stop"
                                exit_price = stop
                            else:
                                outcome = "target"
                                exit_price = target
                            exit_idx = idx
                            hold_bars = step
                            break

                    premium = entry * (model_option_premium_pct / 100.0)
                    if premium <= 0:
                        i = exit_idx + 1
                        continue

                    if direction == "long":
                        delta_move = model_option_delta * (exit_price - entry)
                    else:
                        delta_move = model_option_delta * (entry - exit_price)

                    theta_cost = premium * (model_theta_decay_pct / 100.0) * (hold_bars / max(bars_per_day, 1))
                    option_exit = max(premium + delta_move - theta_cost, 0)

                    slip = slippage_bps / 10000.0
                    entry_opt = premium * (1 + slip)
                    exit_opt = option_exit * (1 - slip)
                    spread_penalty = premium * (model_option_spread_pct / 100.0)
                    commission_penalty = option_commission

                    option_return = (exit_opt - entry_opt - spread_penalty - commission_penalty) / entry_opt

                    option_trades.append(
                        {
                            "direction": direction,
                            "return": option_return,
                            "hold": hold_bars,
                            "outcome": outcome,
                        }
                    )

                    i = exit_idx + 1

                if not option_trades:
                    st.info("No option trades could be simulated with current settings.")
                else:
                    option_returns = [t["return"] for t in option_trades]
                    opt_hit_rate = sum(r > 0 for r in option_returns) / len(option_returns)
                    opt_avg = sum(option_returns) / len(option_returns)
                    opt_median = float(pd.Series(option_returns).median())
                    opt_best = max(option_returns)
                    opt_worst = min(option_returns)
                    opt_avg_hold = sum(t["hold"] for t in option_trades) / len(option_trades)
                    opt_long = sum(t["direction"] == "long" for t in option_trades)
                    opt_short = sum(t["direction"] == "short" for t in option_trades)
                    opt_expectancy = compute_expectancy(option_returns)

                    o1, o2, o3, o4 = st.columns(4)
                    o1.metric("Trades", f"{len(option_trades)}")
                    o2.metric("Hit Rate", f"{opt_hit_rate * 100:.1f}%")
                    o3.metric("Avg Return", f"{opt_avg * 100:.2f}%")
                    o4.metric("Median Return", f"{opt_median * 100:.2f}%")
                    st.caption(
                        f"Trades: {opt_long} calls / {opt_short} puts | "
                        f"Avg hold: {opt_avg_hold:.1f} bars | "
                        f"Best/Worst: {opt_best * 100:.2f}% / {opt_worst * 100:.2f}%."
                    )
                    if opt_expectancy:
                        pf = opt_expectancy["profit_factor"]
                        pf_text = f"{pf:.2f}" if pf is not None else "n/a"
                        st.caption(
                            f"Expectancy per trade: {opt_expectancy['expectancy'] * 100:.2f}% | "
                            f"Avg win/loss: {opt_expectancy['avg_win'] * 100:.2f}% / "
                            f"{opt_expectancy['avg_loss'] * 100:.2f}% | Profit factor: {pf_text}."
                        )
                    st.caption(
                        f"Model: delta {model_option_delta:.2f}, premium {model_option_premium_pct:.1f}%, "
                        f"theta {model_theta_decay_pct:.1f}%/day, spread {model_option_spread_pct:.1f}%, "
                        f"commission ${option_commission:.2f} round trip."
                    )
                    if moneyness_preset:
                        st.caption(f"Moneyness preset applied: {moneyness_profile}.")

            # --- ATR Exit Simulation ---
            st.subheader("ATR Exit Simulation")
            if signal_count == 0:
                st.info("No trades available for exit simulation.")
            else:
                exit_results = []
                for pos, direction in zip(positions, directions):
                    entry = stats_data["Close"].iloc[pos]
                    atr_value = stats_data["ATR"].iloc[pos]
                    if pd.isna(entry) or pd.isna(atr_value) or atr_value <= 0:
                        continue
                    stop_mult = atr_stop_mult
                    if use_choppy_tighten:
                        adx_value = stats_data["ADX"].iloc[pos]
                        if pd.notna(adx_value) and adx_value <= adx_chop:
                            stop_mult = atr_stop_mult * choppy_stop_factor
                    if direction == "long":
                        stop = entry - (stop_mult * atr_value)
                        target = entry + (atr_target_mult * atr_value)
                        risk = entry - stop
                    else:
                        stop = entry + (stop_mult * atr_value)
                        target = entry - (atr_target_mult * atr_value)
                        risk = stop - entry
                    if risk <= 0:
                        continue

                    outcome = "timeout"
                    exit_price = stats_data["Close"].iloc[min(pos + exit_max_bars, len(stats_data) - 1)]
                    hold_bars = min(exit_max_bars, len(stats_data) - 1 - pos)

                    for step in range(1, exit_max_bars + 1):
                        idx = pos + step
                        if idx >= len(stats_data):
                            break
                        high = stats_data["High"].iloc[idx]
                        low = stats_data["Low"].iloc[idx]
                        if pd.isna(high) or pd.isna(low):
                            continue
                        if direction == "long":
                            hit_stop = low <= stop
                            hit_target = high >= target
                        else:
                            hit_stop = high >= stop
                            hit_target = low <= target
                        if hit_stop or hit_target:
                            if hit_stop:
                                outcome = "stop"
                                exit_price = stop
                            else:
                                outcome = "target"
                                exit_price = target
                            hold_bars = step
                            break

                    if pd.isna(exit_price):
                        continue
                    _, entry_adj, exit_adj = apply_slippage(entry, exit_price, direction, slippage_bps)
                    if direction == "long":
                        r_multiple = (exit_adj - entry_adj) / risk
                    else:
                        r_multiple = (entry_adj - exit_adj) / risk
                    exit_results.append({"outcome": outcome, "r": r_multiple, "hold": hold_bars})

                if not exit_results:
                    st.info("No trades available for exit simulation.")
                else:
                    total_trades = len(exit_results)
                    target_hits = sum(r["outcome"] == "target" for r in exit_results)
                    stop_hits = sum(r["outcome"] == "stop" for r in exit_results)
                    timeouts = sum(r["outcome"] == "timeout" for r in exit_results)
                    avg_r = sum(r["r"] for r in exit_results) / total_trades
                    avg_hold = sum(r["hold"] for r in exit_results) / total_trades
                    r_values = [r["r"] for r in exit_results]
                    r_expectancy = compute_expectancy(r_values)
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Trades", f"{total_trades}")
                    e2.metric("Target Hit", f"{(target_hits / total_trades) * 100:.1f}% ({target_hits})")
                    e3.metric("Stop Hit", f"{(stop_hits / total_trades) * 100:.1f}% ({stop_hits})")
                    e4.metric("Avg R", f"{avg_r:.2f}")
                    st.caption(
                        f"Timeouts: {timeouts} ({(timeouts / total_trades) * 100:.1f}%) | "
                        f"Avg hold: {avg_hold:.1f} bars."
                    )
                    if r_expectancy:
                        pf = r_expectancy["profit_factor"]
                        pf_text = f"{pf:.2f}" if pf is not None else "n/a"
                        st.caption(
                            f"Expectancy (R): {r_expectancy['expectancy']:.2f} | "
                            f"Avg win/loss (R): {r_expectancy['avg_win']:.2f} / {r_expectancy['avg_loss']:.2f} | "
                            f"Profit factor: {pf_text}."
                        )
                    st.caption(
                        "Exit sim uses intrabar high/low; if stop and target hit on the same bar, stop is assumed first."
                    )
                    if slippage_bps > 0:
                        st.caption(f"Exit sim includes {slippage_bps:.1f} bps slippage per entry/exit.")
                    if use_choppy_tighten:
                        st.caption(f"Choppy regime stops tightened by factor {choppy_stop_factor:.2f}.")

            # --- Trailing Stop Backtest (ATR) ---
            st.subheader("Trailing Stop Backtest (ATR)")
            if signal_count == 0:
                st.info("No trades available for trailing stop backtest.")
            elif not use_trailing_stop:
                st.info("Trailing stop is disabled in the sidebar.")
            else:
                trail_results = []
                lookback_window = max(int(trail_lookback), 1)
                for pos, direction in zip(positions, directions):
                    entry = stats_data["Close"].iloc[pos]
                    atr_value = stats_data["ATR"].iloc[pos]
                    if pd.isna(entry) or pd.isna(atr_value) or atr_value <= 0:
                        continue
                    stop_mult = atr_stop_mult
                    if use_choppy_tighten:
                        adx_value = stats_data["ADX"].iloc[pos]
                        if pd.notna(adx_value) and adx_value <= adx_chop:
                            stop_mult = atr_stop_mult * choppy_stop_factor
                    if direction == "long":
                        stop = entry - (stop_mult * atr_value)
                        target = entry + (atr_target_mult * atr_value)
                        trail_stop = stop
                        trail_window = [stats_data["High"].iloc[pos]]
                    else:
                        stop = entry + (stop_mult * atr_value)
                        target = entry - (atr_target_mult * atr_value)
                        trail_stop = stop
                        trail_window = [stats_data["Low"].iloc[pos]]

                    outcome = "timeout"
                    exit_price = stats_data["Close"].iloc[min(pos + exit_max_bars, len(stats_data) - 1)]
                    hold_bars = min(exit_max_bars, len(stats_data) - 1 - pos)

                    for step in range(1, exit_max_bars + 1):
                        idx = pos + step
                        if idx >= len(stats_data):
                            break
                        high = stats_data["High"].iloc[idx]
                        low = stats_data["Low"].iloc[idx]
                        if pd.isna(high) or pd.isna(low):
                            continue
                        if direction == "long":
                            trail_window.append(high)
                            if len(trail_window) > lookback_window:
                                trail_window.pop(0)
                            trail_ref = max(trail_window)
                            trail_stop = max(trail_stop, trail_ref - (trail_atr_mult * atr_value))
                            hit_stop = low <= trail_stop
                            hit_target = high >= target
                        else:
                            trail_window.append(low)
                            if len(trail_window) > lookback_window:
                                trail_window.pop(0)
                            trail_ref = min(trail_window)
                            trail_stop = min(trail_stop, trail_ref + (trail_atr_mult * atr_value))
                            hit_stop = high >= trail_stop
                            hit_target = low <= target
                        if hit_stop or hit_target:
                            if hit_stop:
                                outcome = "trail_stop"
                                exit_price = trail_stop
                            else:
                                outcome = "target"
                                exit_price = target
                            hold_bars = step
                            break

                    if pd.isna(exit_price):
                        continue
                    ret, _, _ = apply_slippage(entry, exit_price, direction, slippage_bps)
                    trail_results.append({"outcome": outcome, "return": ret, "hold": hold_bars})

                if not trail_results:
                    st.info("No trades available for trailing stop backtest.")
                else:
                    total_trades = len(trail_results)
                    hit_rate = sum(r["return"] > 0 for r in trail_results) / total_trades
                    avg_return = sum(r["return"] for r in trail_results) / total_trades
                    median_return = float(pd.Series([r["return"] for r in trail_results]).median())
                    best_return = max(r["return"] for r in trail_results)
                    worst_return = min(r["return"] for r in trail_results)
                    trail_expectancy = compute_expectancy([r["return"] for r in trail_results])
                    target_hits = sum(r["outcome"] == "target" for r in trail_results)
                    stop_hits = sum(r["outcome"] == "trail_stop" for r in trail_results)
                    timeouts = sum(r["outcome"] == "timeout" for r in trail_results)
                    avg_hold = sum(r["hold"] for r in trail_results) / total_trades

                    t1, t2, t3, t4 = st.columns(4)
                    t1.metric("Trades", f"{total_trades}")
                    t2.metric("Hit Rate", f"{hit_rate * 100:.1f}%")
                    t3.metric("Avg Return", f"{avg_return * 100:.2f}%")
                    t4.metric("Median Return", f"{median_return * 100:.2f}%")
                    st.caption(
                        f"Target hits: {target_hits} | Trail stops: {stop_hits} | "
                        f"Timeouts: {timeouts} | Avg hold: {avg_hold:.1f} bars."
                    )
                    if trail_expectancy:
                        pf = trail_expectancy["profit_factor"]
                        pf_text = f"{pf:.2f}" if pf is not None else "n/a"
                        st.caption(
                            f"Expectancy per trade: {trail_expectancy['expectancy'] * 100:.2f}% | "
                            f"Avg win/loss: {trail_expectancy['avg_win'] * 100:.2f}% / "
                            f"{trail_expectancy['avg_loss'] * 100:.2f}% | Profit factor: {pf_text}."
                        )
                    st.caption(
                        f"Trailing stop uses {trail_atr_mult:.2f}x ATR with a {lookback_window}-bar window; "
                        "if stop and target hit on the same bar, stop is assumed first."
                    )
                    st.caption(
                        f"Best/Worst return: {best_return * 100:.2f}% / {worst_return * 100:.2f}%."
                    )
                    if slippage_bps > 0:
                        st.caption(f"Trailing sim includes {slippage_bps:.1f} bps slippage per entry/exit.")

        # --- Trading Rules Section ---
        st.markdown("---")
        st.subheader("📋 Execution Rules (Strict!)")
        st.markdown(f"""
    1. **Entry:** Wait for a candle to **close** above/below the Orange VWAP line. Do not guess mid-candle.
    2. **Stop Loss:** If the price crosses back over the VWAP line against you, **SELL IMMEDIATELY**.
       * *Example:* You bought a Call because price > VWAP. 10 mins later, price drops below VWAP. Sell.
    3. **Profit Target:** Intraday options move fast. If you see +10% or +15% profit, sell half or all.
    4. **Trailing Stop:** After +1R, trail using ATR or VWAP to protect profits.
    """)
        st.subheader("Optimal Strategy (Quick Guide)")
        st.markdown("""
    - Trade only when VWAP + MACD align and Signal Score is 4+.
    - Prefer bars with volume ratio >= 1.0x and VWAP Z-Score inside your max extension.
    - Avoid the first 15-20 minutes; the cleanest VWAP trends often appear mid-morning and mid-afternoon.
    - If the market filter is bearish, be cautious with long scalps and reduce size.
    - Use ATR-based stops and targets; tighten when volatility expands quickly.
    - Enable VWAP slope or higher-timeframe filters in choppy tape; trail stops after +1R.
    - For 0DTE, require liquidity PASS and reduce size.
    """)
        st.subheader("Options Strategy (Detailed)")
        st.markdown("""
    1. Market bias: Trade in the same direction as the market filter when enabled; skip longs in a bearish tape.
    2. Entry timing: Wait for a candle close above/below VWAP with MACD confirmation; avoid the first 15-20 minutes.
    3. Signal quality: Target Signal Score 4+ and volume ratio >= 1.0x; avoid extended VWAP Z-Score.
    4. Confirmation: If higher timeframe is bearish, avoid calls; if bullish, avoid puts.
    5. Regime filter: If ADX says CHOPPY, wait for stronger confirmation or reduce size.
    6. Contract selection: 3-7 DTE for stability; 0DTE allowed if selected and liquidity is strong. Delta guide: ITM 0.55-0.75, ATM 0.35-0.55, OTM 0.20-0.35.
    7. Liquidity rules: Bid-ask spread <= $0.10 or <= 5% of premium, open interest >= 100, volume >= 100.
    8. Entry execution: Use limit orders near mid; do not chase if price runs away from VWAP.
    9. Stops: Exit if price closes back through VWAP or hits the ATR stop; honor hard stops quickly.
    10. Trailing stop: After +1R or strong trend continuation, trail using ATR from the highest high/lowest low.
    11. Targets: Scale out at +20% to +40% option premium, then trail with VWAP or ATR target.
    12. Time stop: If no progress after N bars (see Max Hold Bars), exit and wait for the next setup.
    13. Risk: Keep per-trade risk to 0.5% to 1% of account; stop for the day after ~2R loss.
    """)
        st.subheader("Other Quant Intraday Strategies (Optional)")
        st.markdown("""
    - **Opening Range Breakout (ORB):** Trade breaks of the first 15-30 minutes with volume confirmation; use VWAP as a filter.
    - **Mean Reversion to VWAP:** Fade extended VWAP Z-scores when ADX is low; use tight stops and small targets.
    - **Volatility Expansion:** Trade when ATR% spikes above a threshold and price breaks recent range.
    - **Trend Pullback:** Trade pullbacks to VWAP or 20-EMA in a trend; require higher timeframe alignment.
    - **Market-Neutral Filter:** Only trade when market filter and sector ETF align with your signal.
    - **Time-of-Day Edge:** Focus mid-morning or power hour; avoid first/last 10 minutes.
        """)

        st.subheader("Glossary (Indicators & Terms)")
        st.markdown("""
    - **ATR:** Average True Range, measures volatility; higher ATR = larger expected moves.
    - **ATR%:** ATR divided by price; normalizes volatility across tickers.
    - **VWAP:** Volume Weighted Average Price; intraday fair value anchor.
    - **VWAP Z-Score:** Distance from VWAP in standard deviations.
    - **MACD:** Momentum indicator (trend + momentum alignment).
    - **ADX:** Trend strength; low = choppy, high = trending.
    - **R (R-multiple):** Return in units of initial risk (stop distance).
    - **DTE:** Days to expiration; lower DTE = higher theta decay.
    - **Theta:** Time decay of option premium per day.
    - **Delta:** Option sensitivity to underlying price changes.
    - **Slippage:** Realistic price impact for entry/exit.
    - **Profit Factor:** Gross wins / gross losses; >1 is profitable.
    - **Expectancy:** Average return per trade after wins/losses.
        """)

    else:
        st.info("Market is likely closed or waiting for data. Try during trading hours (9:30 AM - 4:00 PM ET).")


def _running_with_streamlit():
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_with_streamlit():
    render_app()
elif __name__ == "__main__":
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(stcli.main())
