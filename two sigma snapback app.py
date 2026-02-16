#Two sigma snapback app
import sys
import itertools
from datetime import date, datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator


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

def ensure_streamlit():
    if get_script_run_ctx is None:
        return
    if get_script_run_ctx() is None:
        print("This Streamlit app must be run with Streamlit.")
        print("Run: streamlit run \"two sigma snapback app.py\"")
        sys.exit(0)

ensure_streamlit()

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
    "ATM (Balanced)": "Balanced delta/theta; default for clean swings with normal liquidity.",
    "OTM (Convex)": "Cheaper and convex but higher theta; use only when edge is strong.",
}

# --- App Configuration ---
st.set_page_config(page_title="2-Sigma Snapback Finder", layout="wide")
st.title("⚡ 2-Sigma Mean Reversion Hunter")
st.markdown("""
This tool identifies **Mean Reversion** setups for small accounts.
**Strategy:** Buy when price hits 2 Standard Deviations (Bollinger Bands) AND RSI confirms extremity.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="F").upper()
period = st.sidebar.selectbox("Data Period", options=["3mo", "6mo", "1y"], index=1)
interval = st.sidebar.selectbox("Time Interval", options=["1d", "1h"], index=0)
mode = st.sidebar.selectbox(
    "Mode",
    options=["Easy (1-2 week options)", "Advanced"],
    index=0,
)

if mode == "Easy (1-2 week options)":
    st.sidebar.caption("Easy mode uses looser thresholds and fewer filters.")
    bb_window = 20
    bb_std = 2.0
    rsi_window = 14
    rsi_oversold = 35
    rsi_overbought = 65
    use_confirmation = False
    rsi_recovery_long = 40
    rsi_recovery_short = 60
    use_trend_filter = False
    ema_window = 50
    use_vol_filter = False
    vol_window = 20
    vol_percentile = 80
    cooldown_bars = 0
    atr_window = 14
    atr_stop_mult = 1.5
    target_method = "Middle Band"
    atr_target_mult = 1.5
    trailing_atr_mult = 1.0
    use_trailing_exit = True
    show_setups = True
    use_regime_filter = False
    adx_window = 14
    adx_chop = 20.0
    adx_trend = 25.0
    regime_mode = "Prefer Chop"
    use_market_filter = False
    market_ticker = "SPY"
    market_ema_window = 50
    higher_interval = "Off"
    htf_ema_window = 50
    use_score_filter = False
    min_score_ratio = 0.75
    use_mr_filter = False
    autocorr_window = 20
    autocorr_max = 0.0
    dte_choice = "7-14 DTE"
    moneyness_profile = "ATM (Balanced)"
    use_moneyness_preset = True
    option_delta = 0.45
    option_premium_pct = 2.5
    theta_decay_pct = 8.0
    option_spread_pct = 4.0
    option_commission = 0.0
    slippage_bps = 2.0
    account_size = 10000.0
    risk_pct = 1.0
    require_edge = False
    edge_buffer_pct = 20.0
    check_liquidity = False
    enforce_liquidity = False
    min_open_interest = 100
    min_option_volume = 100
    max_spread_abs = 0.10
    max_spread_pct = 5.0
    atm_range_pct = 5.0
    st.sidebar.caption("Switch to Advanced for quant filters, options modeling, and liquidity checks.")
else:
    # Strategy Parameters
    st.sidebar.subheader("Strategy Parameters")
    bb_window = int(st.sidebar.number_input("Bollinger Window", value=20, min_value=2, step=1))
    bb_std = float(st.sidebar.number_input("Std Dev (Sigma)", value=2.0, min_value=0.5, step=0.5))
    rsi_window = int(st.sidebar.number_input("RSI Window", value=14, min_value=2, step=1))

    # Signal thresholds
    st.sidebar.subheader("Signal Logic")
    rsi_oversold = int(st.sidebar.number_input("RSI Oversold", value=30, min_value=1, max_value=50, step=1))
    rsi_overbought = int(st.sidebar.number_input("RSI Overbought", value=70, min_value=50, max_value=99, step=1))
    use_confirmation = st.sidebar.checkbox("Require Confirmation Bar", value=True)
    rsi_recovery_long = int(st.sidebar.number_input("RSI Recovery (Long)", value=35, min_value=1, max_value=70, step=1))
    rsi_recovery_short = int(st.sidebar.number_input("RSI Recovery (Short)", value=65, min_value=30, max_value=99, step=1))
    show_setups = st.sidebar.checkbox("Show Setup Markers", value=True)

    # Filters
    st.sidebar.subheader("Filters")
    use_trend_filter = st.sidebar.checkbox("Trend Filter (EMA)", value=True)
    ema_window = int(st.sidebar.number_input("EMA Window", value=50, min_value=2, step=1))
    use_vol_filter = st.sidebar.checkbox("Volatility Filter (Bandwidth Percentile)", value=True)
    vol_window = int(st.sidebar.number_input("Volatility Window", value=50, min_value=5, step=1))
    vol_percentile = int(st.sidebar.slider("Max Bandwidth Percentile", min_value=10, max_value=90, value=70, step=5))
    cooldown_bars = int(st.sidebar.number_input("Cooldown Bars", value=0, min_value=0, step=1))

    # Risk management
    st.sidebar.subheader("Risk Management")
    atr_window = int(st.sidebar.number_input("ATR Window", value=14, min_value=2, step=1))
    atr_stop_mult = float(st.sidebar.number_input("Stop ATR Multiplier", value=1.5, min_value=0.5, step=0.5))
    target_method = st.sidebar.selectbox("Target Method", options=["Middle Band", "ATR Multiple"], index=0)
    atr_target_mult = float(st.sidebar.number_input("Target ATR Multiplier", value=1.5, min_value=0.5, step=0.5))
    trailing_atr_mult = float(
        st.sidebar.number_input("Trailing ATR Multiplier", value=1.0, min_value=0.5, step=0.5)
    )
    use_trailing_exit = st.sidebar.checkbox("Use Trailing Stop in Exit Sim", value=True)

    st.sidebar.subheader("Regime Filter")
    use_regime_filter = st.sidebar.checkbox("Use Regime Filter (ADX)", value=True)
    adx_window = int(st.sidebar.number_input("ADX Window", value=14, min_value=5, step=1))
    adx_chop = float(st.sidebar.number_input("Chop Threshold (ADX)", value=20.0, min_value=5.0, step=1.0))
    adx_trend = float(st.sidebar.number_input("Trend Threshold (ADX)", value=25.0, min_value=10.0, step=1.0))
    regime_mode = st.sidebar.selectbox(
        "Regime Mode", options=["Prefer Chop", "Avoid Trend", "Trend Only"], index=0
    )

    st.sidebar.subheader("Market & Higher Timeframe")
    use_market_filter = st.sidebar.checkbox("Market Filter", value=False)
    if use_market_filter:
        market_ticker = st.sidebar.text_input("Market Ticker", value="SPY").upper()
    else:
        market_ticker = "SPY"
    market_ema_window = int(st.sidebar.number_input("Market EMA Window", value=50, min_value=2, step=1))
    higher_tf_options = ["Off", "1d", "1wk"]
    default_higher = "1wk" if interval == "1d" else "1d" if interval == "1h" else "Off"
    if default_higher not in higher_tf_options:
        default_higher = "Off"
    higher_interval = st.sidebar.selectbox(
        "Higher Timeframe Interval",
        options=higher_tf_options,
        index=higher_tf_options.index(default_higher),
    )
    htf_ema_window = int(st.sidebar.number_input("Higher TF EMA Window", value=50, min_value=2, step=1))

    st.sidebar.subheader("Signal Quality")
    use_score_filter = st.sidebar.checkbox("Use Score Filter", value=False)
    min_score_ratio = float(
        st.sidebar.slider("Min Score Ratio", min_value=0.5, max_value=1.0, value=0.75, step=0.05)
    )

    st.sidebar.subheader("Mean Reversion Filter")
    use_mr_filter = st.sidebar.checkbox("Use Autocorr Filter", value=True)
    autocorr_window = int(st.sidebar.number_input("Autocorr Window", value=20, min_value=5, step=1))
    autocorr_max = float(
        st.sidebar.number_input("Max Autocorr (Lag1)", value=0.0, min_value=-1.0, max_value=1.0, step=0.05)
    )

    st.sidebar.subheader("Options Model")
    dte_choice = st.sidebar.selectbox(
        "DTE Preference",
        options=["7-14 DTE", "1-7 DTE", "14-30 DTE"],
        index=0,
    )
    moneyness_profile = st.sidebar.selectbox("Moneyness Profile", options=MONEYNESS_OPTIONS, index=2)
    use_moneyness_preset = st.sidebar.checkbox("Apply Moneyness Preset", value=True)
    if moneyness_profile != "Custom":
        delta_band = MONEYNESS_DELTA_RANGES.get(moneyness_profile, "n/a")
        st.sidebar.caption(f"Delta band: {delta_band}. {MONEYNESS_NOTES.get(moneyness_profile, '')}")
    option_delta = float(st.sidebar.number_input("Assumed Option Delta", value=0.45, min_value=0.05, step=0.05))
    option_premium_pct = float(
        st.sidebar.number_input("Assumed Premium (% of price)", value=2.5, min_value=0.5, step=0.5)
    )
    theta_decay_pct = float(
        st.sidebar.number_input("Theta Decay (% per day)", value=8.0, min_value=0.0, step=1.0)
    )
    option_spread_pct = float(
        st.sidebar.number_input("Spread Penalty (% round trip)", value=4.0, min_value=0.0, step=0.5)
    )
    option_commission = float(
        st.sidebar.number_input("Commission ($ round trip)", value=0.0, min_value=0.0, step=0.1)
    )
    slippage_bps = float(st.sidebar.number_input("Slippage (bps)", value=2.0, min_value=0.0, step=0.5))

    st.sidebar.subheader("Risk Budget")
    account_size = float(st.sidebar.number_input("Account Size ($)", value=10000.0, min_value=100.0, step=100.0))
    risk_pct = float(st.sidebar.number_input("Max Risk % / Trade", value=1.0, min_value=0.1, step=0.1))
    require_edge = st.sidebar.checkbox("Require Edge vs Costs", value=True)
    edge_buffer_pct = float(st.sidebar.number_input("Edge Buffer (%)", value=20.0, min_value=0.0, step=5.0))

    st.sidebar.subheader("Options Liquidity")
    check_liquidity = st.sidebar.checkbox("Check Liquidity", value=False)
    enforce_liquidity = st.sidebar.checkbox("Enforce Liquidity Filter", value=False)
    min_open_interest = int(st.sidebar.number_input("Min Open Interest", value=100, min_value=0, step=50))
    min_option_volume = int(st.sidebar.number_input("Min Option Volume", value=100, min_value=0, step=50))
    max_spread_abs = float(st.sidebar.number_input("Max Spread ($)", value=0.10, min_value=0.01, step=0.01))
    max_spread_pct = float(st.sidebar.number_input("Max Spread (%)", value=5.0, min_value=1.0, step=1.0))
    atm_range_pct = float(st.sidebar.number_input("ATM Range (%)", value=5.0, min_value=1.0, step=1.0))

# Options data cache control
if "options_cache_buster" not in st.session_state:
    st.session_state.options_cache_buster = 0
if mode == "Advanced" and st.sidebar.button("Refresh Options Data"):
    st.session_state.options_cache_buster += 1

# Backtest settings
default_lookahead = 7 if interval == "1d" else 30
st.sidebar.subheader("Backtest")
lookahead_bars = int(
    st.sidebar.number_input("Lookahead Bars (Stats)", value=default_lookahead, min_value=1, step=1)
)
exit_max_bars = int(
    st.sidebar.number_input("Max Hold Bars (Exit Sim)", value=default_lookahead, min_value=1, step=1)
)
stats_use_setups = st.sidebar.checkbox("Use Setups When No Signals", value=True)

st.sidebar.subheader("Quant Research")
if mode == "Advanced":
    run_grid_search = st.sidebar.checkbox("Run Grid Search", value=False)
    run_walkforward = st.sidebar.checkbox("Run Walk-Forward Optimization", value=False)
    run_options_backtest = st.sidebar.checkbox("Run Options P&L Sim", value=True)
    optimize_metric = st.sidebar.selectbox(
        "Optimize Metric", options=["Avg R", "Win Rate", "Profit Factor"], index=0
    )
    min_trades_eval = int(st.sidebar.number_input("Min Trades for Eval", value=5, min_value=1, step=1))
    grid_use_setups = st.sidebar.checkbox(
        "Grid Uses Setups When No Signals", value=stats_use_setups
    )
    relax_grid_filters = st.sidebar.checkbox("Relax Grid Filters if Empty", value=True)
    max_grid_combos = int(st.sidebar.number_input("Max Grid Combos", value=100, min_value=20, step=20))
    show_top_n = int(st.sidebar.number_input("Show Top N", value=5, min_value=1, step=1))

    grid_bb_std = st.sidebar.multiselect(
        "Grid Sigma Values", options=[1.5, 1.8, 2.0, 2.2, 2.5], default=[bb_std]
    )
    grid_rsi_oversold = st.sidebar.multiselect(
        "Grid RSI Oversold", options=[25, 30, 35, 40], default=[rsi_oversold]
    )
    grid_rsi_overbought = st.sidebar.multiselect(
        "Grid RSI Overbought", options=[60, 65, 70, 75], default=[rsi_overbought]
    )
    grid_stop_mults = st.sidebar.multiselect(
        "Grid Stop ATR Mult", options=[1.0, 1.5, 2.0, 2.5], default=[atr_stop_mult]
    )
    grid_target_mults = st.sidebar.multiselect(
        "Grid Target ATR Mult", options=[1.0, 1.5, 2.0, 2.5], default=[atr_target_mult]
    )
    grid_trail_mults = st.sidebar.multiselect(
        "Grid Trail ATR Mult", options=[0.8, 1.0, 1.2, 1.5], default=[trailing_atr_mult]
    )
    grid_toggle_confirmation = st.sidebar.checkbox("Grid Confirmation Toggle", value=False)
    grid_toggle_trailing = st.sidebar.checkbox("Grid Trailing Toggle", value=False)

    walkforward_folds = int(st.sidebar.number_input("Walk-Forward Folds", value=3, min_value=1, step=1))
    walkforward_train_ratio = float(
        st.sidebar.slider("Walk-Forward Train Ratio", min_value=0.5, max_value=0.8, value=0.6, step=0.05)
    )
else:
    run_grid_search = False
    run_walkforward = False
    run_options_backtest = True
    optimize_metric = "Avg R"
    min_trades_eval = 5
    grid_use_setups = stats_use_setups
    relax_grid_filters = False
    max_grid_combos = 100
    show_top_n = 5
    grid_bb_std = [bb_std]
    grid_rsi_oversold = [rsi_oversold]
    grid_rsi_overbought = [rsi_overbought]
    grid_stop_mults = [atr_stop_mult]
    grid_target_mults = [atr_target_mult]
    grid_trail_mults = [trailing_atr_mult]
    grid_toggle_confirmation = False
    grid_toggle_trailing = False
    walkforward_folds = 1
    walkforward_train_ratio = 0.6

if not grid_bb_std:
    grid_bb_std = [bb_std]
if not grid_rsi_oversold:
    grid_rsi_oversold = [rsi_oversold]
if not grid_rsi_overbought:
    grid_rsi_overbought = [rsi_overbought]
if not grid_stop_mults:
    grid_stop_mults = [atr_stop_mult]
if not grid_target_mults:
    grid_target_mults = [atr_target_mult]
if not grid_trail_mults:
    grid_trail_mults = [trailing_atr_mult]

grid_confirmation_values = [use_confirmation] if not grid_toggle_confirmation else [True, False]
grid_trailing_values = [use_trailing_exit] if not grid_toggle_trailing else [True, False]

# --- Data Fetching Function ---
def get_data(ticker, period, interval):
    try:
        download_kwargs = dict(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        try:
            df = yf.download(multi_level_index=False, **download_kwargs)
        except TypeError:
            df = yf.download(**download_kwargs)
        
        if df.empty:
            st.error(f"No data found for {ticker}")
            return None

        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

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
                    st.error("Downloaded data is missing a Close column.")
                    return None
                
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


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
    if dte_choice == "1-7 DTE":
        preset["theta_pct"] *= 0.8
        preset["spread_pct"] = max(preset["spread_pct"] - 0.5, 0.0)
    elif dte_choice == "14-30 DTE":
        preset["theta_pct"] *= 0.6
        preset["spread_pct"] = max(preset["spread_pct"] - 1.0, 0.0)
    return preset


# Options model inputs (used for edge filter/backtests)
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_expirations(ticker, cache_key):
    try:
        t = yf.Ticker(ticker)
        return t.options
    except Exception:
        return []


@st.cache_data(ttl=120, show_spinner=False)
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

    if dte_choice == "1-7 DTE":
        candidates = [e for e in exp_dates if 1 <= dte(e[1]) <= 7]
    elif dte_choice == "7-14 DTE":
        candidates = [e for e in exp_dates if 7 <= dte(e[1]) <= 14]
    elif dte_choice == "14-30 DTE":
        candidates = [e for e in exp_dates if 14 <= dte(e[1]) <= 30]
    else:
        candidates = [e for e in exp_dates if dte(e[1]) >= 0]

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


def apply_cooldown_series(signal_series, cooldown):
    if cooldown <= 0:
        return signal_series
    filtered = signal_series.fillna(False).copy()
    last_signal_index = -cooldown - 1
    for idx, is_signal in enumerate(filtered.tolist()):
        if is_signal:
            if idx - last_signal_index <= cooldown:
                filtered.iloc[idx] = False
            else:
                last_signal_index = idx
    return filtered


def simulate_exit_trades(
    df,
    buy_signal,
    sell_signal,
    atr_stop_mult,
    atr_target_mult,
    trailing_atr_mult,
    use_trailing_exit,
    target_method,
    exit_max_bars,
):
    results = []
    signal_mask = buy_signal.fillna(False) | sell_signal.fillna(False)
    positions = [idx for idx, flag in enumerate(signal_mask.tolist()) if flag]
    directions = [
        "long" if bool(buy_signal.iloc[idx]) else "short"
        for idx in positions
    ]

    for pos, direction in zip(positions, directions):
        entry = df["Close"].iloc[pos]
        atr_value = df["atr"].iloc[pos]
        mid_band = df["bb_mid"].iloc[pos]
        if pd.isna(entry) or pd.isna(atr_value):
            continue

        if direction == "long":
            stop = entry - (atr_stop_mult * atr_value)
            if target_method == "Middle Band":
                target = mid_band if pd.notna(mid_band) else None
            else:
                target = entry + (atr_target_mult * atr_value)
        else:
            stop = entry + (atr_stop_mult * atr_value)
            if target_method == "Middle Band":
                target = mid_band if pd.notna(mid_band) else None
            else:
                target = entry - (atr_target_mult * atr_value)

        if target is None or pd.isna(stop) or pd.isna(target):
            continue
        if direction == "long" and target <= entry:
            continue
        if direction == "short" and target >= entry:
            continue

        risk = entry - stop if direction == "long" else stop - entry
        if pd.isna(risk) or risk <= 0:
            continue

        outcome = "timeout"
        exit_price = df["Close"].iloc[min(pos + exit_max_bars, len(df) - 1)]
        hold_bars = min(exit_max_bars, len(df) - 1 - pos)
        trail_stop = stop

        for step in range(1, exit_max_bars + 1):
            idx = pos + step
            if idx >= len(df):
                break
            high = df["High"].iloc[idx]
            low = df["Low"].iloc[idx]
            if pd.isna(high) or pd.isna(low):
                continue
            if use_trailing_exit:
                atr_bar = df["atr"].iloc[idx]
                if pd.isna(atr_bar) or atr_bar <= 0:
                    atr_bar = atr_value
                if direction == "long":
                    trail_candidate = high - (trailing_atr_mult * atr_bar)
                    if pd.notna(trail_candidate):
                        trail_stop = max(trail_stop, trail_candidate)
                else:
                    trail_candidate = low + (trailing_atr_mult * atr_bar)
                    if pd.notna(trail_candidate):
                        trail_stop = min(trail_stop, trail_candidate)
            effective_stop = (
                max(stop, trail_stop) if direction == "long" else min(stop, trail_stop)
            )
            if direction == "long":
                hit_stop = low <= effective_stop
                hit_target = high >= target
            else:
                hit_stop = high >= effective_stop
                hit_target = low <= target
            if hit_stop or hit_target:
                if hit_stop and hit_target:
                    outcome = "stop"
                    exit_price = effective_stop
                elif hit_stop:
                    outcome = "stop"
                    exit_price = effective_stop
                else:
                    outcome = "target"
                    exit_price = target
                hold_bars = step
                break

        if pd.isna(exit_price):
            continue
        if direction == "long":
            r_multiple = (exit_price - entry) / risk
        else:
            r_multiple = (entry - exit_price) / risk
        results.append(
            {
                "outcome": outcome,
                "r": r_multiple,
                "hold": hold_bars,
                "entry": float(entry),
                "exit_price": float(exit_price),
                "direction": direction,
            }
        )

    return results


# --- Main Execution ---
data = get_data(ticker, period, interval)

if data is not None and not data.empty:
    # 1. Calculate Indicators
    close_prices = data["Close"]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]
    close_prices = pd.to_numeric(close_prices, errors="coerce")
    data = data.copy()
    data["Close"] = close_prices
    
    # Bollinger Bands
    indicator_bb = BollingerBands(close=close_prices, window=bb_window, window_dev=bb_std)
    data["bb_mid"] = indicator_bb.bollinger_mavg()
    data["bb_stddev"] = (indicator_bb.bollinger_hband() - indicator_bb.bollinger_lband()) / (2 * bb_std)
    data["bb_high"] = data["bb_mid"] + (bb_std * data["bb_stddev"])
    data["bb_low"] = data["bb_mid"] - (bb_std * data["bb_stddev"])
    data["bb_width_base"] = (2 * data["bb_stddev"]) / data["bb_mid"].replace(0, pd.NA)
    vol_quantile = vol_percentile / 100.0
    data["bb_width_q_base"] = data["bb_width_base"].rolling(vol_window).quantile(vol_quantile)
    data["bb_width"] = bb_std * data["bb_width_base"]
    
    # RSI
    indicator_rsi = RSIIndicator(close=close_prices, window=rsi_window)
    data["rsi"] = indicator_rsi.rsi()

    # EMA Trend Filter
    data["ema"] = close_prices.ewm(span=ema_window, adjust=False).mean()

    # ATR for risk levels
    atr_indicator = AverageTrueRange(
        high=data["High"], low=data["Low"], close=data["Close"], window=atr_window
    )
    data["atr"] = atr_indicator.average_true_range()

    # ADX regime filter
    if len(data) >= adx_window:
        adx_indicator = ADXIndicator(
            high=data["High"], low=data["Low"], close=data["Close"], window=adx_window
        )
        data["adx"] = adx_indicator.adx()
    else:
        data["adx"] = pd.Series([pd.NA] * len(data), index=data.index)

    # Mean reversion regime (autocorrelation)
    data["ret"] = data["Close"].pct_change()
    if use_mr_filter:
        data["autocorr"] = data["ret"].rolling(autocorr_window).apply(
            lambda x: x.autocorr(lag=1), raw=False
        )
    else:
        data["autocorr"] = pd.Series([pd.NA] * len(data), index=data.index)

    # Volatility filter threshold
    data["bb_width_q"] = bb_std * data["bb_width_q_base"]

    # 2. Identify Signals
    # Setup: extreme + momentum
    data["setup_buy"] = (data["Close"] < data["bb_low"]) & (data["rsi"] < rsi_oversold)
    data["setup_sell"] = (data["Close"] > data["bb_high"]) & (data["rsi"] > rsi_overbought)

    if use_confirmation:
        confirm_buy = data["setup_buy"].shift(1) & (
            (data["Close"] > data["bb_low"]) | (data["rsi"] > rsi_recovery_long)
        )
        confirm_sell = data["setup_sell"].shift(1) & (
            (data["Close"] < data["bb_high"]) | (data["rsi"] < rsi_recovery_short)
        )
    else:
        confirm_buy = data["setup_buy"]
        confirm_sell = data["setup_sell"]

    trend_ok_long = (
        (data["Close"] > data["ema"]) if use_trend_filter else pd.Series(True, index=data.index)
    )
    trend_ok_short = (
        (data["Close"] < data["ema"]) if use_trend_filter else pd.Series(True, index=data.index)
    )
    vol_ok = (
        data["bb_width_q"].isna() | (data["bb_width"] <= data["bb_width_q"])
    ) if use_vol_filter else pd.Series(True, index=data.index)
    vol_ok_base = data["bb_width_q_base"].isna() | (data["bb_width_base"] <= data["bb_width_q_base"])

    if use_regime_filter:
        if regime_mode == "Prefer Chop":
            regime_ok = data["adx"].isna() | (data["adx"] <= adx_chop)
        elif regime_mode == "Trend Only":
            regime_ok = data["adx"].isna() | (data["adx"] >= adx_trend)
        else:
            regime_ok = data["adx"].isna() | (data["adx"] < adx_trend)
    else:
        regime_ok = pd.Series(True, index=data.index)

    if use_mr_filter:
        mr_ok = data["autocorr"].isna() | (data["autocorr"] <= autocorr_max)
    else:
        mr_ok = pd.Series(True, index=data.index)

    use_higher_tf = higher_interval != "Off" and higher_interval != interval
    htf_bias = "OFF"
    if use_higher_tf:
        higher_data = get_data(ticker, period, higher_interval)
        if higher_data is not None and not higher_data.empty:
            higher_close = higher_data["Close"]
            if isinstance(higher_close, pd.DataFrame):
                higher_close = higher_close.iloc[:, 0]
            higher_close = pd.to_numeric(higher_close, errors="coerce")
            higher_ema = higher_close.ewm(span=htf_ema_window, adjust=False).mean()
            htf_ok_long = (higher_close > higher_ema).reindex(data.index, method="ffill").fillna(True)
            htf_ok_short = (higher_close < higher_ema).reindex(data.index, method="ffill").fillna(True)
            if bool((higher_close > higher_ema).iloc[-1]):
                htf_bias = "BULLISH"
            elif bool((higher_close < higher_ema).iloc[-1]):
                htf_bias = "BEARISH"
            else:
                htf_bias = "NEUTRAL"
        else:
            htf_ok_long = pd.Series(True, index=data.index)
            htf_ok_short = pd.Series(True, index=data.index)
            htf_bias = "UNAVAILABLE"
    else:
        htf_ok_long = pd.Series(True, index=data.index)
        htf_ok_short = pd.Series(True, index=data.index)

    market_bias = "OFF"
    if use_market_filter and market_ticker:
        market_data = get_data(market_ticker, period, interval)
        if market_data is not None and not market_data.empty:
            market_close = market_data["Close"]
            if isinstance(market_close, pd.DataFrame):
                market_close = market_close.iloc[:, 0]
            market_close = pd.to_numeric(market_close, errors="coerce")
            market_ema = market_close.ewm(span=market_ema_window, adjust=False).mean()
            market_ok_long = (market_close > market_ema).reindex(data.index, method="ffill").fillna(True)
            market_ok_short = (market_close < market_ema).reindex(data.index, method="ffill").fillna(True)
            if bool((market_close > market_ema).iloc[-1]):
                market_bias = "BULLISH"
            elif bool((market_close < market_ema).iloc[-1]):
                market_bias = "BEARISH"
            else:
                market_bias = "NEUTRAL"
        else:
            market_ok_long = pd.Series(True, index=data.index)
            market_ok_short = pd.Series(True, index=data.index)
            market_bias = "UNAVAILABLE"
    else:
        market_ok_long = pd.Series(True, index=data.index)
        market_ok_short = pd.Series(True, index=data.index)

    bars_per_day = estimate_bars_per_day(data, interval)
    hold_bars = max(int(exit_max_bars), 1)
    slip = slippage_bps / 10000.0
    denom = (1 - slip) * model_option_delta
    if denom > 0:
        premium = data["Close"] * (model_option_premium_pct / 100.0)
        theta_cost = premium * (model_theta_decay_pct / 100.0) * (hold_bars / max(bars_per_day, 1))
        spread_penalty = premium * (model_option_spread_pct / 100.0)
        numerator = (1 - slip) * theta_cost + spread_penalty + option_commission + (2 * premium * slip)
        break_even_move_series = numerator / denom
        required_move_series = break_even_move_series * (1 + edge_buffer_pct / 100.0)
    else:
        break_even_move_series = pd.Series([pd.NA] * len(data), index=data.index)
        required_move_series = pd.Series([pd.NA] * len(data), index=data.index)

    if target_method == "Middle Band":
        target_long = data["bb_mid"]
        target_short = data["bb_mid"]
    else:
        target_long = data["Close"] + (atr_target_mult * data["atr"])
        target_short = data["Close"] - (atr_target_mult * data["atr"])

    expected_move_long = target_long - data["Close"]
    expected_move_short = data["Close"] - target_short
    edge_ok_long = expected_move_long >= required_move_series
    edge_ok_short = expected_move_short >= required_move_series
    edge_ok_long = edge_ok_long.fillna(False)
    edge_ok_short = edge_ok_short.fillna(False)

    raw_buy = confirm_buy & trend_ok_long & vol_ok & regime_ok & mr_ok & market_ok_long & htf_ok_long
    raw_sell = confirm_sell & trend_ok_short & vol_ok & regime_ok & mr_ok & market_ok_short & htf_ok_short
    if require_edge:
        raw_buy = raw_buy & edge_ok_long
        raw_sell = raw_sell & edge_ok_short

    score_max = 1
    if use_confirmation:
        score_max += 1
    if use_trend_filter:
        score_max += 1
    if use_vol_filter:
        score_max += 1
    if use_regime_filter:
        score_max += 1
    if use_mr_filter:
        score_max += 1
    if require_edge:
        score_max += 1
    if use_market_filter:
        score_max += 1
    if use_higher_tf:
        score_max += 1

    score_long_series = data["setup_buy"].astype(int)
    score_short_series = data["setup_sell"].astype(int)
    if use_confirmation:
        score_long_series += confirm_buy.fillna(False).astype(int)
        score_short_series += confirm_sell.fillna(False).astype(int)
    if use_trend_filter:
        score_long_series += trend_ok_long.astype(int)
        score_short_series += trend_ok_short.astype(int)
    if use_vol_filter:
        score_long_series += vol_ok.astype(int)
        score_short_series += vol_ok.astype(int)
    if use_regime_filter:
        score_long_series += regime_ok.astype(int)
        score_short_series += regime_ok.astype(int)
    if use_mr_filter:
        score_long_series += mr_ok.astype(int)
        score_short_series += mr_ok.astype(int)
    if require_edge:
        score_long_series += edge_ok_long.astype(int)
        score_short_series += edge_ok_short.astype(int)
    if use_market_filter:
        score_long_series += market_ok_long.astype(int)
        score_short_series += market_ok_short.astype(int)
    if use_higher_tf:
        score_long_series += htf_ok_long.astype(int)
        score_short_series += htf_ok_short.astype(int)

    score_ratio_long = score_long_series / score_max if score_max else pd.Series(0, index=data.index)
    score_ratio_short = score_short_series / score_max if score_max else pd.Series(0, index=data.index)
    score_ratio_series = pd.concat([score_ratio_long, score_ratio_short], axis=1).max(axis=1)
    score_value_series = pd.concat([score_long_series, score_short_series], axis=1).max(axis=1)
    if use_score_filter:
        raw_buy = raw_buy & (score_ratio_long >= min_score_ratio)
        raw_sell = raw_sell & (score_ratio_short >= min_score_ratio)

    raw_buy = raw_buy.fillna(False)
    raw_sell = raw_sell.fillna(False)

    if cooldown_bars > 0:
        any_signal = raw_buy | raw_sell
        allowed = apply_cooldown_series(any_signal, cooldown_bars)
        data["Buy_Signal"] = raw_buy & allowed
        data["Sell_Signal"] = raw_sell & allowed
    else:
        data["Buy_Signal"] = raw_buy
        data["Sell_Signal"] = raw_sell

    # Filter only rows where signals exist for the table
    signals = data[data["Buy_Signal"] | data["Sell_Signal"]].tail(10)

    # --- Dashboard Layout ---
    
    # Kpi Metrics
    latest = data.dropna(subset=["Close", "rsi", "bb_low", "bb_high"]).tail(1)
    if latest.empty:
        st.info("Not enough data to compute indicators for the selected period.")
        st.stop()
    latest = latest.iloc[0]

    current_price = float(latest["Close"])
    current_rsi = float(latest["rsi"])
    current_bb_low = float(latest["bb_low"])
    current_bb_high = float(latest["bb_high"])
    current_bb_mid = float(latest["bb_mid"])
    current_adx = float(latest["adx"]) if pd.notna(latest["adx"]) else None
    current_score_value = float(score_value_series.iloc[-1]) if not score_value_series.empty else 0.0
    current_score_ratio = float(score_ratio_series.iloc[-1]) if not score_ratio_series.empty else 0.0
    current_autocorr = float(latest["autocorr"]) if pd.notna(latest["autocorr"]) else None

    if current_adx is None:
        current_regime = "UNAVAILABLE"
    elif current_adx <= adx_chop:
        current_regime = "CHOPPY"
    elif current_adx >= adx_trend:
        current_regime = "TREND"
    else:
        current_regime = "NEUTRAL"

    if current_score_ratio >= 0.8:
        signal_quality = "HIGH"
    elif current_score_ratio >= 0.6:
        signal_quality = "MEDIUM"
    else:
        signal_quality = "LOW"

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Current RSI", f"{current_rsi:.2f}", delta_color="inverse")
    
    status = "NEUTRAL"
    if bool(latest["Buy_Signal"]):
        status = "🟢 CONFIRMED LONG"
    elif bool(latest["Sell_Signal"]):
        status = "🔴 CONFIRMED SHORT"
    elif bool(latest["setup_buy"]):
        status = "🟡 SETUP (Oversold)"
    elif bool(latest["setup_sell"]):
        status = "🟡 SETUP (Overbought)"
        
    col3.metric("Signal Status", status)
    regime_label = (
        f"{current_regime} ({current_adx:.1f})" if current_adx is not None else current_regime
    )
    autocorr_label = f"{current_autocorr:.2f}" if current_autocorr is not None else "n/a"
    st.caption(
        f"Signal score: {current_score_value:.0f}/{score_max} ({signal_quality}) | "
        f"Regime: {regime_label} | AutoCorr: {autocorr_label} | "
        f"Market: {market_bias} | Higher TF: {htf_bias}"
    )

    # Action callout for the latest valid bar
    action = "WAIT"
    action_detail = "No 2-sigma + RSI signal on the latest bar."
    if bool(latest["Buy_Signal"]):
        action = "BUY CALL"
        if use_confirmation:
            action_detail = (
                f"Confirmed long after oversold setup (<{rsi_oversold} RSI and below band), now recovering."
            )
        else:
            action_detail = (
                f"Setup: price below lower band and RSI {current_rsi:.2f} < {rsi_oversold}."
            )
    elif bool(latest["Sell_Signal"]):
        action = "BUY PUT"
        if use_confirmation:
            action_detail = (
                f"Confirmed short after overbought setup (>{rsi_overbought} RSI and above band), now reversing."
            )
        else:
            action_detail = (
                f"Setup: price above upper band and RSI {current_rsi:.2f} > {rsi_overbought}."
            )
    elif bool(latest["setup_buy"]):
        action = "WATCH SETUP"
        action_detail = "Oversold setup detected. Wait for confirmation or rebound."
    elif bool(latest["setup_sell"]):
        action = "WATCH SETUP"
        action_detail = "Overbought setup detected. Wait for confirmation or pullback."

    st.subheader("Action Now")
    if action == "BUY CALL":
        st.success(f"{action} — {action_detail}")
    elif action == "BUY PUT":
        st.error(f"{action} — {action_detail}")
    elif action == "WATCH SETUP":
        st.warning(f"{action} — {action_detail}")
    else:
        st.info(f"{action} — {action_detail}")
    st.caption(f"Options window: {dte_choice}. Prefer liquid strikes and defined risk.")

    last_signal = data[data["Buy_Signal"] | data["Sell_Signal"]].tail(1)
    if not last_signal.empty:
        last_row = last_signal.iloc[0]
        last_type = "BUY CALL" if bool(last_row["Buy_Signal"]) else "BUY PUT"
        last_time = last_signal.index[-1]
        st.caption(f"Last signal: {last_type} on {last_time}")
    else:
        st.caption("Last signal: none in the selected period.")
    signal_count = int((data["Buy_Signal"] | data["Sell_Signal"]).sum())
    setup_count = int((data["setup_buy"] | data["setup_sell"]).sum())
    st.caption(f"Signal count: {signal_count} | Setup count: {setup_count}")

    # --- Risk Snapshot ---
    def risk_levels(entry_price, atr_value, mid_band, direction):
        if atr_value is None or pd.isna(atr_value) or atr_value <= 0 or direction is None:
            return None, None, None
        if direction == "long":
            stop = entry_price - (atr_stop_mult * atr_value)
            if target_method == "Middle Band":
                target = mid_band if mid_band is not None and not pd.isna(mid_band) else None
            else:
                target = entry_price + (atr_target_mult * atr_value)
            risk = entry_price - stop
            reward = (target - entry_price) if target is not None else None
        else:
            stop = entry_price + (atr_stop_mult * atr_value)
            if target_method == "Middle Band":
                target = mid_band if mid_band is not None and not pd.isna(mid_band) else None
            else:
                target = entry_price - (atr_target_mult * atr_value)
            risk = stop - entry_price
            reward = (entry_price - target) if target is not None else None
        rr = (reward / risk) if (risk > 0 and reward is not None) else None
        return stop, target, rr

    st.subheader("Risk Snapshot")
    last_setup = data[data["setup_buy"] | data["setup_sell"]].tail(1)
    risk_row = None
    risk_label = None
    risk_pos = None
    risk_entry = None
    risk_target = None
    risk_direction = None
    risk_atr = None

    if bool(latest["Buy_Signal"]) or bool(latest["Sell_Signal"]):
        risk_row = latest
        risk_label = "Latest confirmed signal (this bar)"
        risk_pos = data.index.get_indexer([latest.name])[0]
    elif show_setups and (bool(latest["setup_buy"]) or bool(latest["setup_sell"])):
        risk_row = latest
        risk_label = "Latest setup (this bar)"
        risk_pos = data.index.get_indexer([latest.name])[0]
    elif not last_signal.empty:
        risk_row = last_signal.iloc[0]
        risk_label = f"Last confirmed signal ({last_signal.index[-1]})"
        risk_pos = data.index.get_indexer([last_signal.index[-1]])[0]
    elif show_setups and not last_setup.empty:
        risk_row = last_setup.iloc[0]
        risk_label = f"Last setup ({last_setup.index[-1]})"
        risk_pos = data.index.get_indexer([last_setup.index[-1]])[0]

    if risk_row is not None:
        is_long = bool(risk_row.get("Buy_Signal", False)) or bool(risk_row.get("setup_buy", False))
        is_short = bool(risk_row.get("Sell_Signal", False)) or bool(risk_row.get("setup_sell", False))
        direction = "long" if is_long else "short" if is_short else None
        entry_price = float(risk_row["Close"])
        atr_value = float(risk_row["atr"]) if pd.notna(risk_row["atr"]) else None
        mid_band = float(risk_row["bb_mid"]) if pd.notna(risk_row["bb_mid"]) else None
        stop, target, rr = risk_levels(entry_price, atr_value, mid_band, direction)
        risk_entry = entry_price
        risk_target = target
        risk_direction = direction
        risk_atr = atr_value
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Entry", f"${entry_price:.2f}")
        r2.metric("ATR", f"{atr_value:.2f}" if atr_value is not None else "n/a")
        r3.metric("Stop", f"${stop:.2f}" if stop is not None else "n/a")
        r4.metric("Target", f"${target:.2f}" if target is not None else "n/a")
        st.caption(
            f"Stop model: {atr_stop_mult:.2f}x ATR | "
            f"Trailing: {trailing_atr_mult:.2f}x ATR ({'ON' if use_trailing_exit else 'OFF'})."
        )
        if risk_label:
            st.caption(f"Basis: {risk_label}")
        if rr is not None:
            st.caption(f"Estimated R:R = {rr:.2f} using {target_method.lower()} target.")
        is_setup_only = (
            bool(risk_row.get("setup_buy", False)) or bool(risk_row.get("setup_sell", False))
        ) and not (bool(risk_row.get("Buy_Signal", False)) or bool(risk_row.get("Sell_Signal", False)))
        if is_setup_only:
            st.caption("Setup only; wait for confirmation if desired.")
        trail_stop = None
        trail_basis = None
        if risk_pos is not None and risk_pos >= 0 and atr_value is not None:
            atr_trail = data["atr"].iloc[-1]
            if pd.isna(atr_trail) or atr_trail <= 0:
                atr_trail = atr_value
            if direction == "long":
                highest_high = data["High"].iloc[risk_pos:].max()
                if pd.notna(highest_high):
                    trail_stop = highest_high - (trailing_atr_mult * atr_trail)
                    trail_basis = f"highest high since entry ({highest_high:.2f})"
            elif direction == "short":
                lowest_low = data["Low"].iloc[risk_pos:].min()
                if pd.notna(lowest_low):
                    trail_stop = lowest_low + (trailing_atr_mult * atr_trail)
                    trail_basis = f"lowest low since entry ({lowest_low:.2f})"
        if trail_stop is not None and trail_basis:
            st.caption(
                f"Trailing stop suggestion: ${trail_stop:.2f} using {trailing_atr_mult:.2f}x ATR "
                f"from the {trail_basis}."
            )
    else:
        st.info("No signals or setups available to compute risk levels.")

    # --- Options Readiness ---
    st.subheader("Options Readiness")
    base_entry = risk_entry if risk_entry is not None else current_price
    expected_hold = max(int(exit_max_bars), 1)
    break_even_move, break_even_pct = estimate_break_even_move(
        base_entry,
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
    if risk_entry is not None and risk_target is not None:
        expected_move = abs(risk_target - risk_entry)
    elif risk_atr is not None:
        expected_move = atr_target_mult * risk_atr

    edge_ratio = None
    edge_ok = None
    edge_state = "UNAVAILABLE"
    required_move = None
    if break_even_move is not None and expected_move is not None and break_even_move > 0:
        required_move = break_even_move * (1 + edge_buffer_pct / 100.0)
        edge_ratio = expected_move / required_move if required_move > 0 else None
        edge_ok = expected_move >= required_move
        edge_state = "PASS" if edge_ok else "FAIL"

    moneyness_reco = "ATM (Balanced)"
    moneyness_reason = "Default to balanced delta/theta."
    if edge_ratio is not None:
        if edge_ratio >= 2.0 and signal_quality == "HIGH":
            moneyness_reco = "OTM (Convex)"
            moneyness_reason = "Strong edge and high-quality setup."
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

    max_risk = account_size * (risk_pct / 100.0)
    est_premium = (base_entry * model_option_premium_pct / 100.0) if base_entry else None
    contract_cost = (est_premium * 100.0) if est_premium is not None else None
    max_contracts = int(max_risk // contract_cost) if contract_cost and contract_cost > 0 else None

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Break-even Move", f"${break_even_move:.2f}" if break_even_move is not None else "n/a")
    o2.metric("Break-even %", f"{break_even_pct:.2f}%" if break_even_pct is not None else "n/a")
    o3.metric("Expected Move", f"${expected_move:.2f}" if expected_move is not None else "n/a")
    o4.metric("Edge Status", edge_state)

    if est_premium is not None:
        st.caption(
            f"Risk budget: ${max_risk:.0f} | Est premium: ${est_premium:.2f} | "
            f"Max contracts: {max_contracts if max_contracts is not None else 'n/a'}"
        )
    st.caption(
        f"Model inputs: delta {model_option_delta:.2f}, premium {model_option_premium_pct:.1f}%, "
        f"theta/day {model_theta_decay_pct:.1f}%, spread {model_option_spread_pct:.1f}%."
    )
    if edge_ratio is not None and required_move is not None:
        st.caption(f"Edge ratio: {edge_ratio:.2f}x | Required move: ${required_move:.2f}")
    st.caption(f"Moneyness suggestion: {moneyness_reco}. {moneyness_reason}")

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
            st.caption(
                f"Liquidity ({options_liquidity.get('expiration')} | {options_liquidity.get('dte')} DTE "
                f"as of {options_liquidity.get('fetched_at')}): {liquidity_state}"
            )
    st.caption(f"Trade gate: edge {edge_state} | liquidity {liquidity_state} | quality {signal_quality}")
    if require_edge and edge_state == "FAIL":
        st.warning("Edge vs costs is too small. Consider skipping or using ITM contracts.")
    if enforce_liquidity and liquidity_state == "FAIL":
        st.warning("Liquidity filter failed. Consider skipping or widening your strike range.")

    # --- Decision Engine ---
    st.subheader("Decision Engine")
    latest_buy = bool(latest.get("Buy_Signal", False))
    latest_sell = bool(latest.get("Sell_Signal", False))
    latest_setup_buy = bool(latest.get("setup_buy", False))
    latest_setup_sell = bool(latest.get("setup_sell", False))

    decision_direction = None
    decision_state = "NONE"
    if latest_buy:
        decision_direction = "LONG"
        decision_state = "CONFIRMED"
    elif latest_sell:
        decision_direction = "SHORT"
        decision_state = "CONFIRMED"
    elif show_setups and (latest_setup_buy or latest_setup_sell):
        decision_direction = "LONG" if latest_setup_buy else "SHORT"
        decision_state = "SETUP"

    def status_label(is_on, ok):
        if not is_on:
            return "OFF"
        return "PASS" if ok else "FAIL"

    if decision_direction == "LONG":
        confirmation_ok = bool(confirm_buy.iloc[-1]) if use_confirmation else True
        trend_ok_dir = bool(trend_ok_long.iloc[-1]) if use_trend_filter else True
        market_ok_dir = bool(market_ok_long.iloc[-1]) if use_market_filter else True
        htf_ok_dir = bool(htf_ok_long.iloc[-1]) if use_higher_tf else True
        edge_ok_dir = bool(edge_ok_long.iloc[-1]) if require_edge else True
    elif decision_direction == "SHORT":
        confirmation_ok = bool(confirm_sell.iloc[-1]) if use_confirmation else True
        trend_ok_dir = bool(trend_ok_short.iloc[-1]) if use_trend_filter else True
        market_ok_dir = bool(market_ok_short.iloc[-1]) if use_market_filter else True
        htf_ok_dir = bool(htf_ok_short.iloc[-1]) if use_higher_tf else True
        edge_ok_dir = bool(edge_ok_short.iloc[-1]) if require_edge else True
    else:
        confirmation_ok = False
        trend_ok_dir = False
        market_ok_dir = False
        htf_ok_dir = False
        edge_ok_dir = False

    vol_ok_dir = bool(vol_ok.iloc[-1]) if use_vol_filter else True
    regime_ok_dir = bool(regime_ok.iloc[-1]) if use_regime_filter else True
    mr_ok_dir = bool(mr_ok.iloc[-1]) if use_mr_filter else True
    score_ok = (current_score_ratio >= min_score_ratio) if use_score_filter else True

    input_rows = [
        ("Signal", f"{decision_state} {decision_direction or ''}".strip()),
        ("Confirmation", status_label(use_confirmation, confirmation_ok)),
        ("Trend Filter", status_label(use_trend_filter, trend_ok_dir)),
        ("Volatility Filter", status_label(use_vol_filter, vol_ok_dir)),
        ("Regime Filter", status_label(use_regime_filter, regime_ok_dir)),
        ("Mean Reversion Filter", status_label(use_mr_filter, mr_ok_dir)),
        ("Market Filter", status_label(use_market_filter, market_ok_dir)),
        ("Higher TF Filter", status_label(use_higher_tf, htf_ok_dir)),
        ("Score Gate", status_label(use_score_filter, score_ok)),
        ("Edge vs Costs", status_label(require_edge, edge_ok_dir)),
        ("Liquidity", liquidity_state if check_liquidity else "OFF"),
    ]
    st.table(pd.DataFrame(input_rows, columns=["Input", "Status"]))

    filters_ok = all(
        [
            confirmation_ok if use_confirmation else True,
            trend_ok_dir if use_trend_filter else True,
            vol_ok_dir if use_vol_filter else True,
            regime_ok_dir if use_regime_filter else True,
            mr_ok_dir if use_mr_filter else True,
            market_ok_dir if use_market_filter else True,
            htf_ok_dir if use_higher_tf else True,
        ]
    )

    blockers = []
    if decision_state != "CONFIRMED":
        blockers.append("no confirmed signal")
    if decision_state == "CONFIRMED" and not filters_ok:
        blockers.append("filter failure")
    if require_edge and not edge_ok_dir:
        blockers.append("edge failed")
    if enforce_liquidity and liquidity_state in ("FAIL", "UNAVAILABLE"):
        blockers.append("liquidity failed")
    if use_score_filter and not score_ok:
        blockers.append("score below minimum")

    decision = "WAIT"
    decision_detail = "No confirmed signal."
    if decision_state == "SETUP":
        decision = "WATCH"
        decision_detail = "Setup only; wait for confirmation."
    elif decision_state == "CONFIRMED":
        if blockers:
            decision = "PASS"
            decision_detail = " / ".join(blockers)
        elif signal_quality == "LOW":
            decision = "PAPER ONLY"
            decision_detail = "Signal quality is low."
        else:
            decision = "TRADE"
            decision_detail = "All required checks pass."

    if decision_state == "CONFIRMED":
        signal_step_status = "PASS"
        signal_step_detail = "Confirmed on the latest bar."
    elif decision_state == "SETUP":
        signal_step_status = "SETUP"
        signal_step_detail = "Setup only; wait for confirmation."
    else:
        signal_step_status = "FAIL"
        signal_step_detail = "No signal on the latest bar."

    decision_steps = [
        ("1) Signal", signal_step_status, signal_step_detail),
        ("2) Filters", "PASS" if filters_ok else "FAIL", "Trend, vol, regime, MR, market, HTF."),
        ("3) Edge", status_label(require_edge, edge_ok_dir), "Break-even move vs target."),
        ("4) Liquidity", liquidity_state if check_liquidity else "OFF", "Options spreads/volume."),
        ("5) Size", "READY" if decision in ("TRADE", "PAPER ONLY") else "WAIT", "Risk budget + contracts."),
    ]

    st.markdown(f"**Decision:** {decision} — {decision_detail}")
    for label, status, detail in decision_steps:
        st.markdown(f"- {label}: **{status}** — {detail}")

    # --- Plotting with Plotly ---
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                open=data["Open"], high=data["High"],
                low=data["Low"], close=data["Close"], name="Price"))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data["bb_high"], 
                             line=dict(color='gray', width=1, dash='dash'), name="Upper Band (2σ)"))
    fig.add_trace(go.Scatter(x=data.index, y=data["bb_low"], 
                             line=dict(color='gray', width=1, dash='dash'), name="Lower Band (2σ)"))
    fig.add_trace(go.Scatter(x=data.index, y=data["bb_mid"], 
                             line=dict(color='orange', width=1), name="Moving Avg"))

    # Plot Signals
    buys = data[data["Buy_Signal"]]
    sells = data[data["Sell_Signal"]]
    if show_setups:
        setup_buys = data[data["setup_buy"]]
        setup_sells = data[data["setup_sell"]]

    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode='markers', 
                             marker=dict(symbol='triangle-up', size=12, color='green'), name="Buy Signal"))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode='markers', 
                             marker=dict(symbol='triangle-down', size=12, color='red'), name="Sell Signal"))
    if show_setups:
        fig.add_trace(go.Scatter(
            x=setup_buys.index, y=setup_buys["Close"], mode="markers",
            marker=dict(symbol="circle-open", size=8, color="green"), name="Buy Setup"
        ))
        fig.add_trace(go.Scatter(
            x=setup_sells.index, y=setup_sells["Close"], mode="markers",
            marker=dict(symbol="circle-open", size=8, color="red"), name="Sell Setup"
        ))

    fig.update_layout(title=f"{ticker} Price vs Bollinger Bands", xaxis_title="Date", yaxis_title="Price", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # --- Recent Signals Table ---
    st.subheader("Recent Signal Events")
    if not signals.empty:
        # Format for display
        display_signals = signals[["Close", "rsi", "bb_low", "bb_high", "bb_mid", "atr"]].copy()
        display_signals["Score"] = score_ratio_series.reindex(display_signals.index)
        display_signals["Type"] = signals.apply(lambda x: "BUY (Call)" if x["Buy_Signal"] else "SELL (Put)", axis=1)
        def compute_display_levels(row):
            direction = "long" if "BUY" in row["Type"] else "short"
            stop, target, rr = risk_levels(row["Close"], row["atr"], row["bb_mid"], direction)
            row["Stop"] = stop
            row["Target"] = target
            row["RR"] = rr
            return row

        display_signals = display_signals.apply(compute_display_levels, axis=1)
        st.dataframe(
            display_signals[["Type", "Score", "Close", "rsi", "Stop", "Target", "RR"]]
            .sort_index(ascending=False)
        )
    else:
        if show_setups:
            setup_signals = data[data["setup_buy"] | data["setup_sell"]].tail(10)
            if not setup_signals.empty:
                st.info("No confirmed signals. Showing recent setups.")
                display_setups = setup_signals[["Close", "rsi"]].copy()
                display_setups["Score"] = score_ratio_series.reindex(display_setups.index)
                display_setups["Type"] = setup_signals.apply(
                    lambda x: "SETUP BUY" if x["setup_buy"] else "SETUP SELL",
                    axis=1
                )
                st.dataframe(display_setups[["Type", "Score", "Close", "rsi"]].sort_index(ascending=False))
            else:
                st.info("No setups found in the selected period.")
        else:
            st.info("No extreme 2-Sigma signals found in the selected period.")

    # --- Backtest Snapshot ---
    st.subheader("Backtest Snapshot")
    stats_buy = data["Buy_Signal"].fillna(False)
    stats_sell = data["Sell_Signal"].fillna(False)
    stats_basis = "confirmed signals"
    if stats_use_setups and show_setups and (stats_buy | stats_sell).sum() == 0:
        stats_buy = data["setup_buy"].fillna(False)
        stats_sell = data["setup_sell"].fillna(False)
        stats_basis = "setups"

    stats_mask = stats_buy | stats_sell
    signal_count = int(stats_mask.sum())
    positions = []
    directions = []
    for idx, (is_buy, is_sell) in enumerate(zip(stats_buy.tolist(), stats_sell.tolist())):
        if is_buy or is_sell:
            positions.append(idx)
            directions.append("long" if is_buy else "short")

    forward_returns = []
    for pos, direction in zip(positions, directions):
        exit_pos = pos + lookahead_bars
        if exit_pos >= len(data):
            continue
        entry = data["Close"].iloc[pos]
        exit_price = data["Close"].iloc[exit_pos]
        if pd.isna(entry) or pd.isna(exit_price):
            continue
        if direction == "long":
            ret = (exit_price - entry) / entry
        else:
            ret = (entry - exit_price) / entry
        forward_returns.append(ret)

    if signal_count == 0:
        st.info("No signals available for backtest stats.")
    else:
        hit_rate = (
            sum(r > 0 for r in forward_returns) / len(forward_returns)
            if forward_returns
            else None
        )
        avg_return = (
            sum(forward_returns) / len(forward_returns)
            if forward_returns
            else None
        )
        wins = [r for r in forward_returns if r > 0]
        losses = [r for r in forward_returns if r <= 0]
        avg_win = sum(wins) / len(wins) if wins else None
        avg_loss = sum(losses) / len(losses) if losses else None
        if hit_rate is not None and avg_win is not None:
            if not losses:
                ev = avg_win
            elif avg_loss is not None:
                ev = (hit_rate * avg_win) + ((1 - hit_rate) * avg_loss)
            else:
                ev = None
        else:
            ev = None
        if wins and losses:
            profit_factor = sum(wins) / abs(sum(losses))
        elif wins and not losses:
            profit_factor = float("inf")
        else:
            profit_factor = None
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Signals", f"{signal_count}")
        b2.metric("Hit Rate", f"{hit_rate * 100:.1f}%" if hit_rate is not None else "n/a")
        b3.metric(
            f"Avg Return ({lookahead_bars} bars)",
            f"{avg_return * 100:.2f}%" if avg_return is not None else "n/a",
        )
        b4.metric("EV / Trade", f"{ev * 100:.2f}%" if ev is not None else "n/a")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Win", f"{avg_win * 100:.2f}%" if avg_win is not None else "n/a")
        c2.metric("Avg Loss", f"{avg_loss * 100:.2f}%" if avg_loss is not None else "n/a")
        if profit_factor == float("inf"):
            c3.metric("Profit Factor", "∞")
        else:
            c3.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor is not None else "n/a")
        st.caption(f"Basis: {stats_basis}.")
        if forward_returns and len(forward_returns) < signal_count:
            st.caption("Some signals were excluded due to insufficient future bars.")
        if not forward_returns:
            st.caption("Not enough future bars for lookahead stats; try a longer period.")

    # --- ATR Exit Simulation ---
    st.subheader("ATR Exit Simulation")
    exit_results = simulate_exit_trades(
        data,
        stats_buy,
        stats_sell,
        atr_stop_mult,
        atr_target_mult,
        trailing_atr_mult,
        use_trailing_exit,
        target_method,
        exit_max_bars,
    )

    if not exit_results:
        st.info("No trades available for exit simulation.")
    else:
        total_trades = len(exit_results)
        target_hits = sum(r["outcome"] == "target" for r in exit_results)
        stop_hits = sum(r["outcome"] == "stop" for r in exit_results)
        timeouts = sum(r["outcome"] == "timeout" for r in exit_results)
        avg_r = sum(r["r"] for r in exit_results) / total_trades
        avg_hold = sum(r["hold"] for r in exit_results) / total_trades
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Trades", f"{total_trades}")
        e2.metric("Target Hit", f"{(target_hits / total_trades) * 100:.1f}% ({target_hits})")
        e3.metric("Stop Hit", f"{(stop_hits / total_trades) * 100:.1f}% ({stop_hits})")
        e4.metric("Avg R", f"{avg_r:.2f}")
        st.caption(
            f"Timeouts: {timeouts} ({(timeouts / total_trades) * 100:.1f}%) | "
            f"Avg hold: {avg_hold:.1f} bars."
        )
        st.caption(
            f"Trailing stop: {'ON' if use_trailing_exit else 'OFF'} "
            f"({trailing_atr_mult:.2f}x ATR)."
        )
        st.caption("Exit sim uses intrabar high/low; if stop and target hit on the same bar, stop is assumed first.")

    # --- Options P&L Simulation ---
    st.subheader("Options P&L Simulation")
    if not run_options_backtest:
        st.info("Enable Options P&L Simulation in the sidebar to estimate returns.")
    elif not exit_results:
        st.info("No trades available for options P&L simulation.")
    else:
        option_returns = []
        option_pnls = []
        for trade in exit_results:
            entry = trade.get("entry")
            exit_price = trade.get("exit_price")
            direction = trade.get("direction")
            hold_bars = trade.get("hold", 0)
            if entry is None or exit_price is None or direction is None:
                continue
            premium = entry * (model_option_premium_pct / 100.0)
            if premium <= 0:
                continue
            underlying_move = (exit_price - entry) if direction == "long" else (entry - exit_price)
            theta_cost = premium * (model_theta_decay_pct / 100.0) * (hold_bars / max(bars_per_day, 1))
            spread_penalty = premium * (model_option_spread_pct / 100.0)
            slip = slippage_bps / 10000.0
            slippage_cost = premium * 2 * slip
            pnl = (model_option_delta * underlying_move) - theta_cost - spread_penalty - option_commission - slippage_cost
            pnl = max(pnl, -premium)
            option_pnls.append(pnl)
            option_returns.append(pnl / premium)

        if not option_returns:
            st.info("Options P&L simulation has insufficient data.")
        else:
            avg_ret = sum(option_returns) / len(option_returns)
            win_rate = sum(r > 0 for r in option_returns) / len(option_returns)
            median_ret = float(pd.Series(option_returns).median())
            avg_pnl = sum(option_pnls) / len(option_pnls)
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Avg Option Return", f"{avg_ret * 100:.1f}%")
            o2.metric("Win Rate", f"{win_rate * 100:.1f}%")
            o3.metric("Median Return", f"{median_ret * 100:.1f}%")
            o4.metric("Avg $/contract", f"${avg_pnl:.2f}")
            st.caption(
                "P&L model uses constant delta + theta decay; results are approximate."
            )

    # --- Quant Research ---
    st.subheader("Quant Research")
    if not run_grid_search and not run_walkforward:
        st.info("Enable Grid Search or Walk-Forward Optimization in the sidebar to run quant research.")
    else:
        if grid_use_setups:
            st.caption("Grid search will fall back to setups when no confirmed signals exist.")
        param_grid = []
        for combo in itertools.product(
            grid_bb_std,
            grid_rsi_oversold,
            grid_rsi_overbought,
            grid_confirmation_values,
            grid_stop_mults,
            grid_target_mults,
            grid_trail_mults,
            grid_trailing_values,
        ):
            if combo[1] >= combo[2]:
                continue
            param_grid.append(combo)

        if not param_grid:
            st.info("Grid has no valid parameter combinations.")
        else:
            if len(param_grid) > max_grid_combos:
                param_grid = param_grid[:max_grid_combos]
                st.caption(f"Grid truncated to {max_grid_combos} combinations.")

            def evaluate_param_set(
                params,
                mask=None,
                min_trades_override=None,
                ignore_filters=False,
            ):
                (
                    bb_std_val,
                    rsi_os,
                    rsi_ob,
                    confirm_flag,
                    stop_mult,
                    target_mult,
                    trail_mult,
                    trail_flag,
                ) = params
                bb_low = data["bb_mid"] - (bb_std_val * data["bb_stddev"])
                bb_high = data["bb_mid"] + (bb_std_val * data["bb_stddev"])
                setup_buy = (data["Close"] < bb_low) & (data["rsi"] < rsi_os)
                setup_sell = (data["Close"] > bb_high) & (data["rsi"] > rsi_ob)

                if confirm_flag:
                    confirm_buy = setup_buy.shift(1) & (
                        (data["Close"] > bb_low) | (data["rsi"] > rsi_recovery_long)
                    )
                    confirm_sell = setup_sell.shift(1) & (
                        (data["Close"] < bb_high) | (data["rsi"] < rsi_recovery_short)
                    )
                else:
                    confirm_buy = setup_buy
                    confirm_sell = setup_sell

                confirm_buy = confirm_buy.fillna(False)
                confirm_sell = confirm_sell.fillna(False)
                setup_buy = setup_buy.fillna(False)
                setup_sell = setup_sell.fillna(False)

                buy = confirm_buy
                sell = confirm_sell
                if grid_use_setups and (buy | sell).sum() == 0:
                    buy = setup_buy
                    sell = setup_sell
                if not ignore_filters:
                    if use_trend_filter:
                        buy &= trend_ok_long
                        sell &= trend_ok_short
                    if use_vol_filter:
                        buy &= vol_ok_base
                        sell &= vol_ok_base
                    if use_regime_filter:
                        buy &= regime_ok
                        sell &= regime_ok
                    if use_mr_filter:
                        buy &= mr_ok
                        sell &= mr_ok
                    if use_market_filter:
                        buy &= market_ok_long
                        sell &= market_ok_short
                    if use_higher_tf:
                        buy &= htf_ok_long
                        sell &= htf_ok_short

                if require_edge and not ignore_filters:
                    if target_method == "Middle Band":
                        expected_long = (data["bb_mid"] - data["Close"]).clip(lower=0)
                        expected_short = (data["Close"] - data["bb_mid"]).clip(lower=0)
                    else:
                        expected_long = target_mult * data["atr"]
                        expected_short = target_mult * data["atr"]
                    edge_ok_long_param = (expected_long >= required_move_series).fillna(False)
                    edge_ok_short_param = (expected_short >= required_move_series).fillna(False)
                    buy &= edge_ok_long_param
                    sell &= edge_ok_short_param

                if not ignore_filters:
                    score_max_param = 1
                    if confirm_flag:
                        score_max_param += 1
                    if use_trend_filter:
                        score_max_param += 1
                    if use_vol_filter:
                        score_max_param += 1
                    if use_regime_filter:
                        score_max_param += 1
                    if use_mr_filter:
                        score_max_param += 1
                    if require_edge:
                        score_max_param += 1
                    if use_market_filter:
                        score_max_param += 1
                    if use_higher_tf:
                        score_max_param += 1

                    score_long = setup_buy.astype(int)
                    score_short = setup_sell.astype(int)
                    if confirm_flag:
                        score_long += confirm_buy.astype(int)
                        score_short += confirm_sell.astype(int)
                    if use_trend_filter:
                        score_long += trend_ok_long.astype(int)
                        score_short += trend_ok_short.astype(int)
                    if use_vol_filter:
                        score_long += vol_ok_base.astype(int)
                        score_short += vol_ok_base.astype(int)
                    if use_regime_filter:
                        score_long += regime_ok.astype(int)
                        score_short += regime_ok.astype(int)
                    if use_mr_filter:
                        score_long += mr_ok.astype(int)
                        score_short += mr_ok.astype(int)
                    if require_edge:
                        score_long += edge_ok_long_param.astype(int)
                        score_short += edge_ok_short_param.astype(int)
                    if use_market_filter:
                        score_long += market_ok_long.astype(int)
                        score_short += market_ok_short.astype(int)
                    if use_higher_tf:
                        score_long += htf_ok_long.astype(int)
                        score_short += htf_ok_short.astype(int)

                    score_ratio_long = score_long / score_max_param if score_max_param else pd.Series(0, index=data.index)
                    score_ratio_short = score_short / score_max_param if score_max_param else pd.Series(0, index=data.index)

                    if use_score_filter:
                        buy &= score_ratio_long >= min_score_ratio
                        sell &= score_ratio_short >= min_score_ratio

                if mask is not None:
                    buy &= mask
                    sell &= mask

                if cooldown_bars > 0 and not ignore_filters:
                    allowed = apply_cooldown_series(buy | sell, cooldown_bars)
                    buy &= allowed
                    sell &= allowed

                exit_trades = simulate_exit_trades(
                    data,
                    buy,
                    sell,
                    stop_mult,
                    target_mult,
                    trail_mult,
                    trail_flag,
                    target_method,
                    exit_max_bars,
                )
                trades = len(exit_trades)
                min_trades = min_trades_eval if min_trades_override is None else min_trades_override
                if trades < min_trades:
                    return None

                r_values = [t["r"] for t in exit_trades]
                avg_r = sum(r_values) / trades
                win_rate = sum(r > 0 for r in r_values) / trades
                pos_sum = sum(r for r in r_values if r > 0)
                neg_sum = sum(r for r in r_values if r < 0)
                profit_factor = pos_sum / abs(neg_sum) if neg_sum < 0 else None
                avg_hold = sum(t["hold"] for t in exit_trades) / trades

                if optimize_metric == "Win Rate":
                    metric_value = win_rate
                elif optimize_metric == "Profit Factor":
                    metric_value = profit_factor if profit_factor is not None else -1.0
                else:
                    metric_value = avg_r

                params_label = (
                    f"σ={bb_std_val:.2f} RSI={rsi_os}/{rsi_ob} "
                    f"conf={'Y' if confirm_flag else 'N'} "
                    f"stop={stop_mult:.2f} target={target_mult:.2f} "
                    f"trail={trail_mult:.2f} ({'Y' if trail_flag else 'N'})"
                )
                return {
                    "params": params,
                    "label": params_label,
                    "trades": trades,
                    "avg_r": avg_r,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "avg_hold": avg_hold,
                    "metric": metric_value,
                    "ignore_filters": ignore_filters,
                }

            if run_grid_search:
                def collect_results(min_trades_override=None, ignore_filters=False):
                    collected = []
                    for params in param_grid:
                        metrics = evaluate_param_set(
                            params,
                            min_trades_override=min_trades_override,
                            ignore_filters=ignore_filters,
                        )
                        if metrics:
                            collected.append(metrics)
                    return collected

                results_note = None
                relaxed_filters = False
                results = collect_results()
                if not results:
                    results = collect_results(min_trades_override=1)
                    if results:
                        results_note = (
                            f"No parameter sets met Min Trades ({min_trades_eval}). "
                            "Showing results with Min Trades = 1."
                        )
                if not results and relax_grid_filters:
                    results = collect_results(ignore_filters=True)
                    if results:
                        results_note = "No results with filters. Showing relaxed grid (filters off)."
                        relaxed_filters = True
                    else:
                        results = collect_results(min_trades_override=1, ignore_filters=True)
                        if results:
                            results_note = (
                                "No results with filters. Showing relaxed grid (filters off) "
                                "with Min Trades = 1."
                            )
                            relaxed_filters = True

                if not results:
                    st.info("Grid search found no parameter sets with enough trades.")
                else:
                    if results_note:
                        st.warning(results_note)
                    results_sorted = sorted(results, key=lambda x: x["metric"], reverse=True)
                    top = results_sorted[:show_top_n]
                    grid_df = pd.DataFrame(top)
                    grid_df = grid_df.rename(
                        columns={
                            "label": "Params",
                            "avg_r": "Avg R",
                            "win_rate": "Win Rate",
                            "profit_factor": "Profit Factor",
                            "avg_hold": "Avg Hold",
                            "trades": "Trades",
                        }
                    )
                    st.dataframe(
                        grid_df[["Params", "Trades", "Avg R", "Win Rate", "Profit Factor", "Avg Hold"]]
                    )
                    trades_values = grid_df["Trades"].tolist() if "Trades" in grid_df else []
                    best_trades = max(trades_values) if trades_values else 0
                    if best_trades >= max(min_trades_eval, 10):
                        st.success("Confidence: High (adequate sample size).")
                    elif best_trades >= max(3, min_trades_eval):
                        st.warning("Confidence: Medium (sample size is limited).")
                    else:
                        st.warning(
                            "Confidence: Low (very small sample). Consider 6mo–1y data or looser filters."
                        )
                    if best_trades < max(3, min_trades_eval):
                        st.info(
                            "Tip: Increase Data Period to 6mo–1y (or longer) to improve sample size "
                            "before trusting the grid results."
                        )

                    if data["adx"].notna().any():
                        regime_masks = {
                            "Chop": data["adx"] <= adx_chop,
                            "Trend": data["adx"] >= adx_trend,
                            "Neutral": (data["adx"] > adx_chop) & (data["adx"] < adx_trend),
                        }
                        preset_rows = []
                        for regime_name, regime_mask in regime_masks.items():
                            best_regime = None
                            for params in param_grid:
                                metrics = evaluate_param_set(
                                    params,
                                    mask=regime_mask,
                                    ignore_filters=relaxed_filters,
                                )
                                if not metrics:
                                    continue
                                if best_regime is None or metrics["metric"] > best_regime["metric"]:
                                    best_regime = metrics
                            if best_regime:
                                preset_rows.append(
                                    {
                                        "Regime": regime_name,
                                        "Params": best_regime["label"],
                                        "Avg R": best_regime["avg_r"],
                                        "Win Rate": best_regime["win_rate"],
                                        "Trades": best_regime["trades"],
                                    }
                                )
                        if preset_rows:
                            st.caption("Regime-specific presets (best by selected metric):")
                            st.dataframe(pd.DataFrame(preset_rows))

            if run_walkforward:
                n = len(data)
                if n < 20:
                    st.info("Not enough data for walk-forward optimization.")
                else:
                    test_size = max(int(n * (1 - walkforward_train_ratio) / walkforward_folds), 1)
                    train_size = max(int(n * walkforward_train_ratio), 1)
                    wf_rows = []
                    total_weighted_r = 0.0
                    total_trades = 0
                    used_fallback = False
                    used_relaxed = False

                    for fold in range(walkforward_folds):
                        train_end = train_size + (fold * test_size)
                        test_end = train_end + test_size
                        if test_end > n:
                            break
                        train_mask = pd.Series(False, index=data.index)
                        test_mask = pd.Series(False, index=data.index)
                        train_mask.iloc[:train_end] = True
                        test_mask.iloc[train_end:test_end] = True

                        best_train = None
                        for params in param_grid:
                            metrics = evaluate_param_set(params, mask=train_mask)
                            if not metrics:
                                continue
                            if best_train is None or metrics["metric"] > best_train["metric"]:
                                best_train = metrics

                        if best_train is None:
                            for params in param_grid:
                                metrics = evaluate_param_set(params, mask=train_mask, min_trades_override=1)
                                if not metrics:
                                    continue
                                if best_train is None or metrics["metric"] > best_train["metric"]:
                                    best_train = metrics
                            if best_train is not None:
                                used_fallback = True

                        if best_train is None and relax_grid_filters:
                            for params in param_grid:
                                metrics = evaluate_param_set(params, mask=train_mask, ignore_filters=True)
                                if not metrics:
                                    continue
                                if best_train is None or metrics["metric"] > best_train["metric"]:
                                    best_train = metrics
                            if best_train is None:
                                for params in param_grid:
                                    metrics = evaluate_param_set(
                                        params,
                                        mask=train_mask,
                                        min_trades_override=1,
                                        ignore_filters=True,
                                    )
                                    if not metrics:
                                        continue
                                    if best_train is None or metrics["metric"] > best_train["metric"]:
                                        best_train = metrics
                                if best_train is not None:
                                    used_fallback = True
                                    used_relaxed = True
                            else:
                                used_relaxed = True

                        if best_train is None:
                            continue

                        test_metrics = evaluate_param_set(
                            best_train["params"],
                            mask=test_mask,
                            ignore_filters=best_train.get("ignore_filters", False),
                        )
                        if not test_metrics:
                            continue

                        total_weighted_r += test_metrics["avg_r"] * test_metrics["trades"]
                        total_trades += test_metrics["trades"]

                        wf_rows.append(
                            {
                                "Fold": fold + 1,
                                "Train End": data.index[train_end - 1],
                                "Test Start": data.index[train_end],
                                "Test End": data.index[test_end - 1],
                                "Params": best_train["label"],
                                "Test Trades": test_metrics["trades"],
                                "Test Avg R": test_metrics["avg_r"],
                                "Test Win Rate": test_metrics["win_rate"],
                            }
                        )

                    if not wf_rows:
                        st.info("Walk-forward produced no valid folds.")
                    else:
                        st.dataframe(pd.DataFrame(wf_rows))
                        if total_trades > 0:
                            st.caption(
                                f"Weighted Test Avg R: {total_weighted_r / total_trades:.2f} "
                                f"across {total_trades} trades."
                            )
                        if used_fallback:
                            st.caption("Walk-forward used Min Trades = 1 for training due to low trade counts.")
                        if used_relaxed:
                            st.caption("Walk-forward relaxed filters to find trades; treat results as exploratory.")

    # --- Optimal Strategy Guide ---
    st.subheader("Optimal Strategy (Quick Guide)")
    st.markdown("""
    - Use **Easy mode** for 1-2 week options when you want more frequent setups and faster decisions.
    - Use **Advanced mode** with confirmation + trend/vol filters when you want fewer, higher-quality signals.
    - Prefer **choppy regimes** for mean reversion; avoid strong trends unless the higher timeframe agrees.
    - Use the **autocorr filter** to trade only when short-term returns mean-revert (negative autocorr).
    - Skip trades when volatility is extreme (bandwidth filter fails) or when market/HTF filters disagree.
    - Favor setups that rebound back inside the bands and improve RSI; use the **Risk Snapshot** for stops/targets.
    - Use **Options Readiness** to confirm edge vs costs, liquidity, and position sizing.
    - If signals are rare, relax RSI thresholds (e.g., 35/65) or switch to a longer period (6mo-1y).
    """)

    st.subheader("How to Use (Step-by-Step)")
    st.markdown("""
    1. **Pick timeframe:** For swing trades, start with **1d** and **6mo-1y** data.
    2. **Choose mode:** Start in **Easy** for more setups; switch to **Advanced** for higher quality.
    3. **Look for signals:** 
       - **Setup** = price at/extreme bands + RSI extreme. 
       - **Confirmed** = setup + recovery bar (if confirmation on).
    4. **Check Risk Snapshot:** Use the **ATR stop**, **target**, and **trailing stop** as your plan.
    5. **Validate edge:** In **Options Readiness**, only trade when **Edge = PASS** and liquidity is OK.
    6. **Review stats:** If **Backtest Snapshot** is negative, tighten filters or widen the period.
    """)

    st.subheader("Parameter Cheatsheet (Quick Defaults)")
    st.markdown("""
    - **RSI Oversold/Overbought:** 30/70 (stricter), 35/65 (more setups)
    - **Sigma (BB Std Dev):** 2.0 (default), 1.75-2.25 depending on volatility
    - **Confirmation:** ON for quality, OFF for more signals
    - **EMA Trend Filter:** 50 EMA; turn ON to avoid fighting trends
    - **Volatility Filter:** Bandwidth percentile 60-80 to avoid extreme volatility
    - **Regime (ADX):** Chop <= 20, Trend >= 25; prefer chop for mean reversion
    - **Autocorr Filter:** Max autocorr <= 0 favors mean reversion
    - **Score Filter:** 0.7-0.85 for quality gating
    - **Stops/Targets:** 1.0-2.0 ATR stop, target mid-band or 1-2 ATR
    - **Trailing Stop:** 1.0 ATR is a solid default
    - **DTE:** 7-14 DTE for swing; use ATM/ITM if edge is thin
    """)

    # --- User Instructions ---
    st.markdown("---")
    st.markdown(f"""
    ### ⚡ How to execute this on Robinhood for ${10}:
    1. Look for a **Green Triangle** (Buy Signal) above.
    2. Go to Robinhood and search **{ticker}**.
    3. Click **Trade** -> **Trade Options**.
    4. Select an expiration date around **{dte_choice}**.
    5. Look for a **Call** (if Green Signal) or **Put** (if Red Signal) priced around **$0.05 - $0.10**.
    6. **Check Liquidity:** Ensure there is "Open Interest" of at least 100 contracts.
    """)
    st.caption('Pairs trading is now a separate app. Run: streamlit run "cointegration_app.py"')
