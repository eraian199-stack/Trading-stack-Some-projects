# Cointegration pairs trading app
import sys
import itertools
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

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

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint

    HAS_STATSMODELS = True
except Exception:
    sm = None
    adfuller = None
    coint = None
    HAS_STATSMODELS = False


def ensure_streamlit():
    if get_script_run_ctx is None:
        return
    if get_script_run_ctx() is None:
        print("This Streamlit app must be run with Streamlit.")
        print('Run: streamlit run "cointegration_app.py"')
        sys.exit(0)


ensure_streamlit()

st.set_page_config(page_title="Cointegration Pairs Trading", layout="wide")
st.title("Cointegration Pairs Trading - Test Harness")

st.markdown(
    "Market neutral pairs trading looks for two assets that move together. "
    "When their spread diverges, you short the winner and buy the loser, "
    "betting the relationship mean-reverts."
)
st.caption("Educational only; not financial advice. Verify any rules with your broker/regulator.")
if not HAS_STATSMODELS:
    st.warning("statsmodels not installed; cointegration/ADF p-values will be unavailable.")

constraint_left, constraint_right = st.columns(2)
constraint_left.info(
    "PDT Rule: With under $25,000, you cannot day trade stocks more than 3 times "
    "in a 5-day rolling period (unless you use a cash account, which has settlement delays)."
)
constraint_right.warning(
    "NY Regulations: Many \"funding\" prop firms that use CFDs are blocked in New York. "
    "Futures prop firms (like Topstep or Apex) are typically available, but confirm eligibility."
)

with st.expander("Phase 1: Python Test (Backtesting)"):
    st.markdown(
        """
        - Find a cointegrated pair (e.g., GLD vs GDX or KO vs PEP).
        - Z-Score > 2: spread is 2 standard deviations above normal (sell the spread).
        - Z-Score < -2: spread is 2 standard deviations below normal (buy the spread).
        - Z-Score = 0: spread has mean-reverted (exit).
        """
    )

with st.expander("Phase 2: Paper Trading (Forward Test)"):
    st.markdown(
        """
        1. Open a broker with paper trading (e.g., Webull or Interactive Brokers).
        2. Monitor the Z-Score once per day at market close.
        3. If Z-Score crosses your thresholds, execute the trade in a paper account.
        4. Exit when the Z-Score returns to 0.
        """
    )


@st.cache_data(ttl=600, show_spinner=False)
def get_pair_data(tickers, start, end, interval):
    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            if "Close" in level0:
                data = data["Close"]
            elif "Adj Close" in level0:
                data = data["Adj Close"]
            else:
                return None

        if isinstance(data, pd.Series):
            data = data.to_frame(name=str(tickers[0]).upper())

        data = data.copy()
        data.columns = [str(col).upper() for col in data.columns]
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_universe_prices(tickers, start, end, interval):
    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            if "Close" in level0:
                data = data["Close"]
            elif "Adj Close" in level0:
                data = data["Adj Close"]
            else:
                return None
        if isinstance(data, pd.Series):
            data = data.to_frame(name=str(tickers[0]).upper())
        data = data.copy()
        data.columns = [str(col).upper() for col in data.columns]
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data
    except Exception:
        return None


def estimate_hedge_ratio(y, x):
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    mask = y.notna() & x.notna()
    if mask.sum() < 2:
        return None, None
    y = y[mask]
    x = x[mask]
    if HAS_STATSMODELS and sm is not None:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        params = model.params
        beta = params.iloc[1] if len(params) > 1 else None
        r2 = float(model.rsquared) if hasattr(model, "rsquared") else None
        return beta, r2
    try:
        coeffs = np.polyfit(x.values, y.values, 1)
    except Exception:
        return None, None
    beta = float(coeffs[0]) if len(coeffs) > 0 else None
    corr = np.corrcoef(x.values, y.values)[0, 1]
    r2 = float(corr ** 2) if not np.isnan(corr) else None
    return beta, r2


def compute_coint_pvalue(y, x):
    if not HAS_STATSMODELS or coint is None:
        return None
    try:
        _score, pvalue, _crit = coint(y, x)
        return float(pvalue)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def compute_rolling_stats(series_1, series_2, window, do_coint):
    betas = []
    pvals = []
    idx = []
    total = len(series_1)
    if total < window:
        return None, None
    for end in range(window, total + 1):
        y = series_1.iloc[end - window : end]
        x = series_2.iloc[end - window : end]
        beta, _r2 = estimate_hedge_ratio(y, x)
        betas.append(beta)
        if do_coint:
            pvals.append(compute_coint_pvalue(y, x))
        idx.append(series_1.index[end - 1])
    beta_series = pd.Series(betas, index=idx)
    pval_series = pd.Series(pvals, index=idx) if do_coint else None
    return beta_series, pval_series


def compute_adf_pvalue(series):
    if not HAS_STATSMODELS or adfuller is None:
        return None
    try:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 20:
            return None
        result = adfuller(series, autolag="AIC")
        return float(result[1])
    except Exception:
        return None


def compute_half_life(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 20:
        return None
    lagged = series.shift(1)
    delta = series - lagged
    df = pd.concat([delta, lagged], axis=1).dropna()
    if df.empty:
        return None
    y = df.iloc[:, 0]
    x = df.iloc[:, 1]
    if HAS_STATSMODELS and sm is not None:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1] if len(model.params) > 1 else None
    else:
        try:
            beta = np.polyfit(x.values, y.values, 1)[0]
        except Exception:
            beta = None
    if beta is None or beta >= 0:
        return None
    half_life = -np.log(2) / beta
    return float(half_life)


def compute_hurst_exponent(series, max_lag=20):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < max_lag + 2:
        return None
    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diff = series.diff(lag).dropna()
        if diff.empty:
            return None
        std = diff.std()
        if std <= 0 or np.isnan(std):
            return None
        tau.append(np.sqrt(std))
    try:
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    except Exception:
        return None
    hurst = poly[0] * 2.0
    return float(hurst)


def estimate_bars_per_day(index, interval):
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
    if isinstance(index, pd.DatetimeIndex):
        counts = pd.Series(index.date).value_counts()
        if not counts.empty:
            return max(int(counts.median()), 1)
    return 1


def simulate_pairs_backtest(
    z_score,
    price_1,
    price_2,
    hedge_ratio,
    entry_upper,
    entry_lower,
    exit_z=0.0,
    stop_z=None,
    max_hold_bars=None,
    confirm_bars=1,
    slippage_bps=0.0,
    commission_per_share=0.0,
    borrow_annual_pct=0.0,
    bars_per_year=252,
):
    trades = []
    if hedge_ratio is None:
        return trades
    hr = abs(hedge_ratio)
    in_trade = False
    direction = None
    entry_idx = None
    entry_p1 = None
    entry_p2 = None
    entry_z = None

    confirm_bars = max(int(confirm_bars), 1)
    above = z_score >= entry_upper
    below = z_score <= entry_lower
    if confirm_bars > 1:
        above = above.rolling(confirm_bars).sum() == confirm_bars
        below = below.rolling(confirm_bars).sum() == confirm_bars

    for idx in range(len(z_score)):
        z_val = z_score.iloc[idx]
        if pd.isna(z_val):
            continue

        if not in_trade:
            if above.iloc[idx]:
                direction = "short"
            elif below.iloc[idx]:
                direction = "long"
            else:
                continue
            in_trade = True
            entry_idx = idx
            entry_p1 = price_1.iloc[idx]
            entry_p2 = price_2.iloc[idx]
            entry_z = z_val
            continue

        hold = idx - entry_idx
        exit_reason = None
        if stop_z is not None and abs(z_val) >= stop_z:
            exit_reason = "stop"
        if exit_reason is None:
            if direction == "long" and z_val >= exit_z:
                exit_reason = "mean"
            elif direction == "short" and z_val <= exit_z:
                exit_reason = "mean"
        if exit_reason is None and max_hold_bars is not None and hold >= max_hold_bars:
            exit_reason = "timeout"

        if exit_reason is None:
            continue

        exit_p1 = price_1.iloc[idx]
        exit_p2 = price_2.iloc[idx]
        if pd.isna(entry_p1) or pd.isna(entry_p2) or pd.isna(exit_p1) or pd.isna(exit_p2):
            in_trade = False
            continue

        if direction == "long":
            pnl = (exit_p1 - entry_p1) - hr * (exit_p2 - entry_p2)
            short_notional = hr * entry_p2
        else:
            pnl = (entry_p1 - exit_p1) - hr * (entry_p2 - exit_p2)
            short_notional = entry_p1
        gross = entry_p1 + hr * entry_p2
        slip_cost = gross * (slippage_bps / 10000.0) * 2
        commission_cost = (1 + hr) * commission_per_share * 2
        borrow_cost = short_notional * (borrow_annual_pct / 100.0) * (hold / max(bars_per_year, 1))
        total_cost = slip_cost + commission_cost + borrow_cost
        pnl_net = pnl - total_cost
        ret_gross = pnl / gross if gross else None
        ret_net = pnl_net / gross if gross else None

        trades.append(
            {
                "entry_index": entry_idx,
                "exit_index": idx,
                "direction": direction,
                "entry_z": float(entry_z),
                "exit_z": float(z_val),
                "hold_bars": int(hold),
                "pnl": float(pnl),
                "pnl_net": float(pnl_net),
                "costs": float(total_cost),
                "return": float(ret_gross) if ret_gross is not None else None,
                "return_net": float(ret_net) if ret_net is not None else None,
                "exit_reason": exit_reason,
            }
        )
        in_trade = False

    return trades


def summarize_trades(trades, use_net):
    if not trades:
        return None
    trade_df = pd.DataFrame(trades)
    ret_col = "return_net" if use_net else "return"
    returns = trade_df[ret_col].dropna()
    if returns.empty:
        return None
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / len(returns)
    avg_return = returns.mean()
    avg_win = wins.mean() if not wins.empty else None
    avg_loss = losses.mean() if not losses.empty else None
    ev_return = avg_return
    if avg_win is not None and avg_loss is not None:
        ev_return = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else None
    avg_hold = trade_df["hold_bars"].mean()
    avg_pnl = trade_df["pnl_net"].mean() if use_net else trade_df["pnl"].mean()
    return {
        "trades": len(returns),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "ev_return": ev_return,
        "profit_factor": profit_factor,
        "avg_hold": avg_hold,
        "avg_pnl": avg_pnl,
        "df": trade_df,
    }


def walk_forward_backtest(
    bt_df,
    hedge_ratio,
    entry_upper,
    entry_lower,
    exit_z,
    stop_z,
    max_hold_bars,
    confirm_bars,
    folds,
    train_ratio,
    min_trades_per_fold,
    use_net,
    slippage_bps,
    commission_per_share,
    borrow_annual_pct,
    bars_per_year,
):
    n = len(bt_df)
    if folds < 2 or n < folds * 30:
        return None, None
    fold_size = n // folds
    rows = []
    oos_trades = []
    for fold in range(folds):
        start = fold * fold_size
        end = n if fold == folds - 1 else (fold + 1) * fold_size
        if end - start < 20:
            continue
        train_end = start + int((end - start) * train_ratio)
        train_df = bt_df.iloc[start:train_end]
        test_df = bt_df.iloc[train_end:end]
        if train_df.empty or test_df.empty:
            continue

        train_trades = simulate_pairs_backtest(
            train_df["z"],
            train_df["p1"],
            train_df["p2"],
            hedge_ratio,
            entry_upper,
            entry_lower,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold_bars=max_hold_bars,
            confirm_bars=confirm_bars,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            borrow_annual_pct=borrow_annual_pct,
            bars_per_year=bars_per_year,
        )
        test_trades = simulate_pairs_backtest(
            test_df["z"],
            test_df["p1"],
            test_df["p2"],
            hedge_ratio,
            entry_upper,
            entry_lower,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold_bars=max_hold_bars,
            confirm_bars=confirm_bars,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            borrow_annual_pct=borrow_annual_pct,
            bars_per_year=bars_per_year,
        )
        train_stats = summarize_trades(train_trades, use_net)
        test_stats = summarize_trades(test_trades, use_net)

        train_trades_count = train_stats["trades"] if train_stats else 0
        test_trades_count = test_stats["trades"] if test_stats else 0
        train_ev = train_stats["ev_return"] if train_stats else None
        test_ev = test_stats["ev_return"] if test_stats else None

        rows.append(
            {
                "Fold": fold + 1,
                "Train Trades": train_trades_count,
                "Train EV": train_ev,
                "Test Trades": test_trades_count,
                "Test EV": test_ev,
            }
        )
        if test_stats and test_trades_count >= min_trades_per_fold:
            oos_trades.extend(test_trades)

    if not rows:
        return None, None
    oos_stats = summarize_trades(oos_trades, use_net) if oos_trades else None
    return pd.DataFrame(rows), oos_stats


default_pair_end = date.today()
default_pair_start = default_pair_end - timedelta(days=730)

with st.form("pair_trading_form"):
    col_a, col_b, col_c = st.columns(3)
    pair_asset_1 = col_a.text_input("Asset 1", value="GLD", key="pair_asset_1").upper()
    pair_asset_2 = col_b.text_input("Asset 2", value="GDX", key="pair_asset_2").upper()
    pair_interval = col_c.selectbox("Interval", options=["1d", "1h"], index=0, key="pair_interval")

    pair_dates = st.date_input(
        "Date Range",
        value=(default_pair_start, default_pair_end),
        key="pair_dates",
    )
    col_d, col_e, col_f = st.columns(3)
    pair_z_upper = col_d.number_input(
        "Z-Score Upper Threshold", value=2.0, step=0.1, key="pair_z_upper"
    )
    pair_z_lower = col_e.number_input(
        "Z-Score Lower Threshold", value=-2.0, step=0.1, key="pair_z_lower"
    )
    pair_use_log = col_f.checkbox("Use log prices", value=False, key="pair_use_log")
    pair_show_table = st.checkbox("Show latest Z-Score table", value=True, key="pair_show_table")

    st.subheader("Decision Engine Settings")
    col_g, col_h, col_i = st.columns(3)
    require_coint = col_g.checkbox("Require Cointegration", value=True)
    coint_p_max = col_h.number_input(
        "Max Cointegration p-value", value=0.05, min_value=0.001, max_value=0.2, step=0.005
    )
    require_adf = col_i.checkbox("Require ADF Stationarity", value=True)

    col_j, col_k, col_l = st.columns(3)
    adf_p_max = col_j.number_input(
        "Max ADF p-value", value=0.05, min_value=0.001, max_value=0.2, step=0.005
    )
    require_hurst = col_k.checkbox("Require Hurst < 0.5", value=True)
    hurst_max = col_l.number_input(
        "Max Hurst", value=0.5, min_value=0.1, max_value=0.9, step=0.05
    )

    col_m, col_n, col_o = st.columns(3)
    require_half_life = col_m.checkbox("Require Half-Life", value=True)
    min_half_life = col_n.number_input(
        "Min Half-Life (bars)", value=1.0, min_value=0.0, step=0.5
    )
    max_half_life = col_o.number_input(
        "Max Half-Life (bars)", value=60.0, min_value=1.0, step=1.0
    )

    col_p, col_q, col_r = st.columns(3)
    require_r2 = col_p.checkbox("Require Min R^2", value=False)
    min_r2 = col_q.number_input("Min R^2", value=0.2, min_value=0.0, max_value=1.0, step=0.05)
    setup_ratio = col_r.number_input(
        "Setup Alert Ratio", value=0.8, min_value=0.5, max_value=0.95, step=0.05
    )

    col_s, col_t, col_u = st.columns(3)
    use_rolling_z = col_s.checkbox("Use Rolling Z-Score", value=True)
    z_window = col_t.number_input("Z-Score Window (bars)", value=60, min_value=20, step=10)
    stop_z = col_u.number_input("Stop Z-Score", value=3.0, min_value=2.0, step=0.1)
    signal_confirm_bars = st.number_input(
        "Signal Confirmation Bars", value=1, min_value=1, step=1
    )

    st.subheader("Rolling Stability Checks")
    col_ra, col_rb, col_rc = st.columns(3)
    use_rolling_stability = col_ra.checkbox("Enable Rolling Stability", value=True)
    rolling_window = col_rb.number_input(
        "Rolling Window (bars)", value=120, min_value=40, step=20
    )
    max_beta_change_pct = col_rc.number_input(
        "Max Beta Drift (%)", value=30.0, min_value=5.0, step=5.0
    )
    col_rd, col_re, col_rf = st.columns(3)
    require_rolling_coint = col_rd.checkbox("Require Rolling Cointegration", value=True)
    rolling_coint_p_max = col_re.number_input(
        "Rolling Coint p max", value=0.1, min_value=0.01, max_value=0.3, step=0.01
    )
    min_coint_pass_rate = col_rf.number_input(
        "Min Rolling Pass Rate", value=0.6, min_value=0.1, max_value=1.0, step=0.05
    )

    st.subheader("Regime Filters")
    col_rg, col_rh, col_ri = st.columns(3)
    require_corr = col_rg.checkbox("Require Rolling Correlation", value=True)
    min_corr = col_rh.number_input("Min Correlation", value=0.6, min_value=0.0, max_value=1.0, step=0.05)
    regime_window = col_ri.number_input("Regime Window (bars)", value=60, min_value=20, step=10)
    col_rj, col_rk, col_rl = st.columns(3)
    require_spread_vol = col_rj.checkbox("Require Spread Vol Stability", value=True)
    max_spread_vol_ratio = col_rk.number_input(
        "Max Spread Vol Ratio", value=2.0, min_value=1.0, step=0.1
    )
    long_vol_window = col_rl.number_input("Long Vol Window (bars)", value=240, min_value=40, step=20)

    st.subheader("Backtest / EV")
    col_aa, col_ab, col_ac = st.columns(3)
    run_backtest = col_aa.checkbox("Run Backtest + EV", value=True)
    exit_z = col_ab.number_input("Exit Z-Score", value=0.0, step=0.1)
    max_hold_bars = col_ac.number_input("Max Hold Bars", value=60, min_value=5, step=5)
    use_stop_z = st.checkbox("Use Stop Z-Score", value=True)

    st.subheader("Costs (Net EV)")
    col_ad, col_ae, col_af = st.columns(3)
    include_costs = col_ad.checkbox("Include Costs in EV", value=True)
    slippage_bps = col_ae.number_input("Slippage (bps, round trip)", value=2.0, min_value=0.0, step=0.5)
    commission_per_share = col_af.number_input(
        "Commission ($/share, per side)", value=0.0, min_value=0.0, step=0.01
    )
    borrow_annual_pct = st.number_input(
        "Short Borrow Rate (% annual)", value=2.0, min_value=0.0, step=0.5
    )

    st.subheader("Walk-Forward (Out-of-Sample)")
    col_ag, col_ah, col_ai = st.columns(3)
    run_walkforward = col_ag.checkbox("Run Walk-Forward OOS", value=True)
    walk_folds = col_ah.number_input("Folds", value=4, min_value=2, step=1)
    walk_train_ratio = col_ai.number_input(
        "Train Ratio", value=0.7, min_value=0.5, max_value=0.9, step=0.05
    )
    col_aj, col_ak, col_al = st.columns(3)
    min_trades_per_fold = col_aj.number_input(
        "Min Trades / Fold", value=3, min_value=1, step=1
    )
    require_oos_positive = col_ak.checkbox("Require OOS EV > 0", value=True)
    require_all_folds_positive = col_al.checkbox("Require All Folds EV > 0", value=False)

    st.subheader("Sizing")
    col_v, col_w, col_x = st.columns(3)
    account_equity = col_v.number_input(
        "Account Equity ($)", value=10000.0, min_value=100.0, step=100.0
    )
    max_leverage = col_w.number_input(
        "Max Gross Leverage", value=1.0, min_value=0.5, step=0.1
    )
    gross_notional = col_x.number_input(
        "Gross Notional ($)", value=10000.0, min_value=100.0, step=100.0
    )
    round_shares = st.checkbox("Round shares to whole", value=True)

    pair_submitted = st.form_submit_button("Run Pairs Test")

confirm_bars = max(int(signal_confirm_bars), 1)

if pair_submitted:
    if not pair_asset_1 or not pair_asset_2:
        st.error("Please enter two tickers.")
    elif pair_asset_1 == pair_asset_2:
        st.error("Pick two different tickers.")
    elif pair_z_lower >= pair_z_upper:
        st.error("Lower Z-Score threshold must be below the upper threshold.")
    elif min_half_life > max_half_life:
        st.error("Min Half-Life must be less than or equal to Max Half-Life.")
    else:
        if isinstance(pair_dates, (list, tuple)) and len(pair_dates) == 2:
            pair_start, pair_end = pair_dates
        else:
            pair_start, pair_end = default_pair_start, default_pair_end

        if pair_start >= pair_end:
            st.error("End date must be after start date.")
        else:
            with st.spinner("Downloading pair data..."):
                pair_prices = get_pair_data(
                    [pair_asset_1, pair_asset_2],
                    pair_start,
                    pair_end,
                    pair_interval,
                )
            if pair_prices is None or pair_prices.empty:
                st.error("No data returned for that pair and date range.")
            else:
                missing = [
                    asset
                    for asset in (pair_asset_1, pair_asset_2)
                    if asset not in pair_prices.columns
                ]
                if missing:
                    st.error(f"Missing data for: {', '.join(missing)}")
                else:
                    pair_prices = pair_prices[[pair_asset_1, pair_asset_2]].dropna()
                    if pair_prices.empty:
                        st.error("No overlapping data between the two assets.")
                    else:
                        price_input = np.log(pair_prices) if pair_use_log else pair_prices
                        hedge_ratio, r2 = estimate_hedge_ratio(
                            price_input[pair_asset_1], price_input[pair_asset_2]
                        )
                        if hedge_ratio is None or np.isnan(hedge_ratio):
                            st.error("Unable to compute a hedge ratio for this pair.")
                        else:
                            spread = price_input[pair_asset_1] - (hedge_ratio * price_input[pair_asset_2])
                            if pair_use_log:
                                st.caption("Using log prices for hedge ratio and Z-Score calculations.")
                            spread_mean = float(spread.mean())
                            spread_std = float(spread.std(ddof=0))
                            if spread_std == 0 or np.isnan(spread_std):
                                st.error("Spread variance is zero; choose another pair or timeframe.")
                            else:
                                z_window_int = int(z_window)
                                if use_rolling_z:
                                    if len(spread) < z_window_int:
                                        st.warning(
                                            "Not enough bars for rolling Z-Score; using full-sample Z-Score."
                                        )
                                        z_score = (spread - spread_mean) / spread_std
                                    else:
                                        rolling_mean = spread.rolling(z_window_int).mean()
                                        rolling_std = spread.rolling(z_window_int).std(ddof=0)
                                        z_score = (spread - rolling_mean) / rolling_std
                                else:
                                    z_score = (spread - spread_mean) / spread_std

                                z_valid = z_score.dropna()
                                if z_valid.empty:
                                    st.error("Z-Score series has insufficient data.")
                                else:
                                    latest_z = float(z_valid.iloc[-1])
                                    coint_p = compute_coint_pvalue(
                                        price_input[pair_asset_1], price_input[pair_asset_2]
                                    )
                                    adf_p = compute_adf_pvalue(spread)
                                    half_life = compute_half_life(spread)
                                    hurst = compute_hurst_exponent(spread)
                                    bt_df = pd.DataFrame(
                                        {
                                            "z": z_score,
                                            "p1": pair_prices[pair_asset_1],
                                            "p2": pair_prices[pair_asset_2],
                                        }
                                    ).dropna()

                                    m1, m2, m3, m4 = st.columns(4)
                                    m1.metric("Hedge Ratio (beta)", f"{hedge_ratio:.3f}")
                                    m2.metric("Latest Z-Score", f"{latest_z:.2f}")
                                    m3.metric(
                                        "Cointegration p-value",
                                        f"{coint_p:.4f}" if coint_p is not None else "n/a",
                                    )
                                    m4.metric("R^2 (fit)", f"{r2:.2f}" if r2 is not None else "n/a")

                                    m5, m6, m7, m8 = st.columns(4)
                                    m5.metric(
                                        "ADF p-value", f"{adf_p:.4f}" if adf_p is not None else "n/a"
                                    )
                                    m6.metric(
                                        "Half-Life (bars)",
                                        f"{half_life:.1f}" if half_life is not None else "n/a",
                                    )
                                    m7.metric(
                                        "Hurst",
                                        f"{hurst:.2f}" if hurst is not None else "n/a",
                                    )
                                    m8.metric(
                                        "Spread Std",
                                        f"{spread_std:.4f}",
                                    )

                                    bars_per_day = estimate_bars_per_day(pair_prices.index, pair_interval)
                                    bars_per_year = max(bars_per_day * 252, 1)

                                    rolling_beta = None
                                    rolling_coint = None
                                    rolling_beta_last = None
                                    rolling_coint_last = None
                                    rolling_coint_pass_rate = None
                                    rolling_beta_ok = True
                                    rolling_coint_ok = True
                                    if use_rolling_stability:
                                        if len(price_input) < int(rolling_window):
                                            st.warning("Not enough data for rolling stability window.")
                                            rolling_beta_ok = False
                                            rolling_coint_ok = False if require_rolling_coint else True
                                        else:
                                            do_coint = bool(require_rolling_coint and HAS_STATSMODELS)
                                            rolling_beta, rolling_coint = compute_rolling_stats(
                                                price_input[pair_asset_1],
                                                price_input[pair_asset_2],
                                                int(rolling_window),
                                                do_coint,
                                            )
                                            if rolling_beta is not None and not rolling_beta.dropna().empty:
                                                rolling_beta_last = float(rolling_beta.dropna().iloc[-1])
                                                if hedge_ratio != 0:
                                                    drift = abs(rolling_beta_last - hedge_ratio) / abs(hedge_ratio)
                                                    rolling_beta_ok = drift <= (max_beta_change_pct / 100.0)
                                                else:
                                                    rolling_beta_ok = False
                                            else:
                                                rolling_beta_ok = False

                                            if require_rolling_coint:
                                                if not HAS_STATSMODELS or rolling_coint is None:
                                                    rolling_coint_ok = False
                                                else:
                                                    valid_p = rolling_coint.dropna()
                                                    if not valid_p.empty:
                                                        rolling_coint_last = float(valid_p.iloc[-1])
                                                        rolling_coint_pass_rate = float(
                                                            (valid_p <= rolling_coint_p_max).mean()
                                                        )
                                                        rolling_coint_ok = (
                                                            rolling_coint_last <= rolling_coint_p_max
                                                            and rolling_coint_pass_rate >= min_coint_pass_rate
                                                        )
                                                    else:
                                                        rolling_coint_ok = False

                                    rolling_corr_last = None
                                    rolling_corr_ok = True
                                    spread_vol_ratio_last = None
                                    spread_vol_ok = True
                                    if require_corr or require_spread_vol:
                                        if len(price_input) < max(int(regime_window), int(long_vol_window)):
                                            if require_corr:
                                                rolling_corr_ok = False
                                            if require_spread_vol:
                                                spread_vol_ok = False
                                        else:
                                            if require_corr:
                                                rolling_corr = price_input[pair_asset_1].rolling(
                                                    int(regime_window)
                                                ).corr(price_input[pair_asset_2])
                                                if rolling_corr.dropna().empty:
                                                    rolling_corr_ok = False
                                                else:
                                                    rolling_corr_last = float(rolling_corr.dropna().iloc[-1])
                                                    rolling_corr_ok = rolling_corr_last >= min_corr
                                            if require_spread_vol:
                                                short_std = spread.rolling(int(regime_window)).std(ddof=0)
                                                long_std = spread.rolling(int(long_vol_window)).std(ddof=0)
                                                ratio = short_std / long_std
                                                if ratio.dropna().empty:
                                                    spread_vol_ok = False
                                                else:
                                                    spread_vol_ratio_last = float(ratio.dropna().iloc[-1])
                                                    spread_vol_ok = spread_vol_ratio_last <= max_spread_vol_ratio

                                    use_net = bool(include_costs)
                                    wf_table = None
                                    oos_stats = None
                                    all_folds_ok = True
                                    if run_walkforward:
                                        if bt_df.empty:
                                            all_folds_ok = False
                                        else:
                                            wf_table, oos_stats = walk_forward_backtest(
                                                bt_df,
                                                hedge_ratio,
                                                pair_z_upper,
                                                pair_z_lower,
                                                exit_z,
                                                stop_z if use_stop_z else None,
                                                max_hold_bars,
                                                confirm_bars,
                                                int(walk_folds),
                                                float(walk_train_ratio),
                                                int(min_trades_per_fold),
                                                use_net,
                                                slippage_bps if include_costs else 0.0,
                                                commission_per_share if include_costs else 0.0,
                                                borrow_annual_pct if include_costs else 0.0,
                                                bars_per_year,
                                            )
                                            if wf_table is not None and not wf_table.empty:
                                                test_ev = wf_table["Test EV"].dropna()
                                                if not test_ev.empty:
                                                    all_folds_ok = bool((test_ev > 0).all())
                                            else:
                                                all_folds_ok = False
                                        if wf_table is None or wf_table.empty:
                                            all_folds_ok = False
                                    if hedge_ratio < 0:
                                        st.warning(
                                            "Negative hedge ratio detected; the pair may be negatively correlated. "
                                            "Confirm the relationship before trading."
                                        )

                                    hedge_ratio_abs = abs(hedge_ratio)
                                    st.caption(
                                        f"Hedge ratio suggests ~1 {pair_asset_1} share vs "
                                        f"{hedge_ratio_abs:.2f} {pair_asset_2} shares for a neutral spread."
                                    )

                                    above = z_score >= pair_z_upper
                                    below = z_score <= pair_z_lower
                                    if confirm_bars > 1:
                                        above = above.rolling(confirm_bars).sum() == confirm_bars
                                        below = below.rolling(confirm_bars).sum() == confirm_bars

                                    signal = "NONE"
                                    above_valid = above.dropna()
                                    below_valid = below.dropna()
                                    if not above_valid.empty and bool(above_valid.iloc[-1]):
                                        signal = "SHORT"
                                    elif not below_valid.empty and bool(below_valid.iloc[-1]):
                                        signal = "LONG"

                                    setup_flag = False
                                    setup_direction = None
                                    if signal == "NONE":
                                        upper_trigger = abs(pair_z_upper) * setup_ratio
                                        lower_trigger = abs(pair_z_lower) * setup_ratio
                                        if latest_z > 0 and latest_z >= upper_trigger:
                                            setup_flag = True
                                            setup_direction = "SHORT"
                                        elif latest_z < 0 and abs(latest_z) >= lower_trigger:
                                            setup_flag = True
                                            setup_direction = "LONG"

                                    if signal == "SHORT":
                                        st.error(
                                            f"Z-Score {latest_z:.2f} > {pair_z_upper:.2f}: "
                                            f"SHORT {pair_asset_1} / LONG {pair_asset_2}"
                                        )
                                    elif signal == "LONG":
                                        st.success(
                                            f"Z-Score {latest_z:.2f} < {pair_z_lower:.2f}: "
                                            f"LONG {pair_asset_1} / SHORT {pair_asset_2}"
                                        )
                                    elif setup_flag:
                                        st.warning(
                                            f"Setup alert: Z-Score {latest_z:.2f} nearing threshold."
                                        )
                                    else:
                                        st.info("Z-Score is inside thresholds; no trade signal.")
                                    st.caption("Exit when Z-Score returns to 0.")

                                    def status_label(required, ok, value):
                                        if not required:
                                            return "OFF"
                                        if value is None:
                                            return "UNAVAILABLE"
                                        return "PASS" if ok else "FAIL"

                                    coint_ok = (
                                        (coint_p is not None and coint_p <= coint_p_max)
                                        if require_coint
                                        else True
                                    )
                                    adf_ok = (
                                        (adf_p is not None and adf_p <= adf_p_max)
                                        if require_adf
                                        else True
                                    )
                                    half_life_ok = (
                                        (half_life is not None and min_half_life <= half_life <= max_half_life)
                                        if require_half_life
                                        else True
                                    )
                                    hurst_ok = (
                                        (hurst is not None and hurst <= hurst_max)
                                        if require_hurst
                                        else True
                                    )
                                    r2_ok = (
                                        (r2 is not None and r2 >= min_r2)
                                        if require_r2
                                        else True
                                    )

                                    price_1 = float(pair_prices[pair_asset_1].iloc[-1])
                                    price_2 = float(pair_prices[pair_asset_2].iloc[-1])
                                    base_qty = gross_notional / (price_1 + hedge_ratio_abs * price_2)
                                    qty_1 = base_qty
                                    qty_2 = hedge_ratio_abs * base_qty
                                    if round_shares:
                                        qty_1 = int(qty_1)
                                        qty_2 = int(qty_2)
                                    gross_used = (qty_1 * price_1) + (qty_2 * price_2)
                                    gross_limit = account_equity * max_leverage
                                    leverage_ok = gross_used <= gross_limit

                                    st.subheader("Decision Engine")
                                    signal_status = "PASS" if signal in ("LONG", "SHORT") else "SETUP" if setup_flag else "FAIL"
                                    oos_ev = oos_stats["ev_return"] if oos_stats else None
                                    oos_positive = oos_ev is not None and oos_ev > 0
                                    decision_rows = [
                                        (
                                            "Cointegration p-value",
                                            status_label(require_coint, coint_ok, coint_p),
                                            f"<= {coint_p_max:.3f}",
                                        ),
                                        (
                                            "ADF Stationarity",
                                            status_label(require_adf, adf_ok, adf_p),
                                            f"<= {adf_p_max:.3f}",
                                        ),
                                        (
                                            "Half-Life",
                                            status_label(require_half_life, half_life_ok, half_life),
                                            f"{min_half_life:.1f}-{max_half_life:.1f} bars",
                                        ),
                                        (
                                            "Hurst",
                                            status_label(require_hurst, hurst_ok, hurst),
                                            f"<= {hurst_max:.2f}",
                                        ),
                                        (
                                            "R^2 Fit",
                                            status_label(require_r2, r2_ok, r2),
                                            f">= {min_r2:.2f}",
                                        ),
                                        (
                                            "Rolling Beta Drift",
                                            status_label(use_rolling_stability, rolling_beta_ok, rolling_beta_last),
                                            f"<= {max_beta_change_pct:.0f}%",
                                        ),
                                        (
                                            "Rolling Coint",
                                            status_label(
                                                require_rolling_coint if use_rolling_stability else False,
                                                rolling_coint_ok,
                                                rolling_coint_last,
                                            ),
                                            f"p<={rolling_coint_p_max:.2f} & pass>={min_coint_pass_rate:.2f}",
                                        ),
                                        (
                                            "Rolling Corr",
                                            status_label(require_corr, rolling_corr_ok, rolling_corr_last),
                                            f">= {min_corr:.2f}",
                                        ),
                                        (
                                            "Spread Vol Ratio",
                                            status_label(require_spread_vol, spread_vol_ok, spread_vol_ratio_last),
                                            f"<= {max_spread_vol_ratio:.2f}",
                                        ),
                                        (
                                            "Walk-Forward OOS EV",
                                            status_label(run_walkforward and require_oos_positive, oos_positive, oos_ev),
                                            "> 0",
                                        ),
                                        ("Signal", signal_status, "Z-Score hit"),
                                        (
                                            "Leverage",
                                            "PASS" if leverage_ok else "FAIL",
                                            f"<= {gross_limit:.0f}",
                                        ),
                                    ]
                                    st.table(pd.DataFrame(decision_rows, columns=["Check", "Status", "Rule"]))
                                    if use_rolling_stability:
                                        if rolling_beta_last is not None:
                                            st.caption(
                                                f"Rolling beta last: {rolling_beta_last:.3f} "
                                                f"(max drift {max_beta_change_pct:.0f}%)."
                                            )
                                        if rolling_coint_pass_rate is not None:
                                            st.caption(
                                                f"Rolling coint pass rate: {rolling_coint_pass_rate:.2f}."
                                            )
                                    if require_corr and rolling_corr_last is not None:
                                        st.caption(f"Rolling corr last: {rolling_corr_last:.2f}.")
                                    if require_spread_vol and spread_vol_ratio_last is not None:
                                        st.caption(
                                            f"Spread vol ratio last: {spread_vol_ratio_last:.2f} "
                                            f"(max {max_spread_vol_ratio:.2f})."
                                        )
                                    if run_walkforward and wf_table is not None and not wf_table.empty:
                                        if oos_ev is not None:
                                            st.caption(f"OOS EV (net): {oos_ev * 100:.2f}%")
                                        else:
                                            st.caption("OOS EV (net): n/a")
                                    if use_stop_z and stop_z <= max(abs(pair_z_upper), abs(pair_z_lower)):
                                        st.warning("Stop Z-Score should be above the entry thresholds.")

                                    failures = []
                                    for label, ok, required in [
                                        ("cointegration", coint_ok, require_coint),
                                        ("ADF stationarity", adf_ok, require_adf),
                                        ("half-life", half_life_ok, require_half_life),
                                        ("hurst", hurst_ok, require_hurst),
                                        ("R^2", r2_ok, require_r2),
                                        ("rolling beta", rolling_beta_ok, use_rolling_stability),
                                        ("rolling coint", rolling_coint_ok, use_rolling_stability and require_rolling_coint),
                                        ("rolling corr", rolling_corr_ok, require_corr),
                                        ("spread vol", spread_vol_ok, require_spread_vol),
                                        ("walk-forward EV", oos_positive, run_walkforward and require_oos_positive),
                                    ]:
                                        if required and not ok:
                                            failures.append(label)
                                    if run_walkforward and require_all_folds_positive and not all_folds_ok:
                                        failures.append("OOS fold EV")
                                    if signal not in ("LONG", "SHORT"):
                                        failures.append("no signal")
                                    if not leverage_ok:
                                        failures.append("leverage")

                                    if signal in ("LONG", "SHORT") and not failures:
                                        decision = "TRADE"
                                        decision_detail = "All required checks pass."
                                    elif setup_flag:
                                        decision = "WATCH"
                                        decision_detail = "Setup only; wait for threshold."
                                    else:
                                        decision = "PASS" if failures else "WAIT"
                                        decision_detail = " / ".join(failures) if failures else "No signal."

                                    st.markdown(f"**Decision:** {decision} — {decision_detail}")
                                    st.subheader("Decision Steps (Trader Grade)")
                                    st.markdown(
                                        f"1) Data sanity: confirm overlapping data and no large gaps; "
                                        f"use {'rolling' if use_rolling_z else 'full-sample'} Z-Score."
                                    )
                                    st.markdown(
                                        f"2) Cointegration: p-value <= {coint_p_max:.3f} "
                                        f"({'required' if require_coint else 'optional'})."
                                    )
                                    st.markdown(
                                        f"3) Stationarity: ADF p <= {adf_p_max:.3f}, "
                                        f"Hurst <= {hurst_max:.2f}, half-life {min_half_life:.1f}-{max_half_life:.1f} bars."
                                    )
                                    st.markdown(
                                        "4) Stability: rolling beta drift within limit and rolling coint pass rate met."
                                    )
                                    st.markdown(
                                        "5) Regime: rolling correlation and spread volatility filters pass."
                                    )
                                    st.markdown(
                                        "6) OOS gate: walk-forward EV positive before live trades."
                                    )
                                    st.markdown(
                                        f"7) Signal: require {confirm_bars} bar(s) beyond thresholds "
                                        f"({pair_z_lower:.2f}, {pair_z_upper:.2f})."
                                    )
                                    st.markdown(
                                        f"8) Risk: exit at Z={exit_z:.2f}; "
                                        f"{('stop at |Z| >= ' + f'{stop_z:.2f}') if use_stop_z else 'no stop'}; "
                                        f"max hold {max_hold_bars} bars."
                                    )
                                    st.markdown(
                                        f"9) Size: hedge ratio sizing, keep gross <= {gross_limit:.0f} "
                                        f"and confirm borrow/fees on the short leg."
                                    )
                                    st.markdown(
                                        "10) Execute: enter both legs simultaneously; avoid legging risk."
                                    )
                                    st.markdown(
                                        "11) Monitor: track Z-Score daily; exit when mean reversion hits or stop triggers."
                                    )

                                    st.subheader("Sizing Plan")
                                    if signal == "LONG":
                                        leg1 = f"LONG {pair_asset_1}"
                                        leg2 = f"SHORT {pair_asset_2}"
                                    elif signal == "SHORT":
                                        leg1 = f"SHORT {pair_asset_1}"
                                        leg2 = f"LONG {pair_asset_2}"
                                    else:
                                        leg1 = f"{pair_asset_1}"
                                        leg2 = f"{pair_asset_2}"
                                    st.caption(
                                        f"Suggested shares (gross {gross_used:.0f}): {leg1} {qty_1:.2f}, "
                                        f"{leg2} {qty_2:.2f}"
                                    )
                                    if round_shares and (qty_1 == 0 or qty_2 == 0):
                                        st.warning("Position size rounds to zero; increase gross notional.")
                                    if not leverage_ok:
                                        st.warning(
                                            "Gross notional exceeds leverage limit; reduce gross notional."
                                        )
                                    if half_life is not None:
                                        st.caption(
                                            f"Expected mean-reversion horizon: ~{2 * half_life:.0f} to "
                                            f"{3 * half_life:.0f} bars."
                                        )

                                    # --- Backtest / Expected Value ---
                                    st.subheader("Backtest & Expected Value")
                                    if not run_backtest:
                                        st.info("Enable Backtest + EV in the settings to compute expectancy.")
                                    else:
                                        if bt_df.empty:
                                            st.info("Not enough data to run backtest.")
                                        else:
                                            trades = simulate_pairs_backtest(
                                                bt_df["z"],
                                                bt_df["p1"],
                                                bt_df["p2"],
                                                hedge_ratio,
                                                pair_z_upper,
                                                pair_z_lower,
                                                exit_z=exit_z,
                                                stop_z=stop_z if use_stop_z else None,
                                                max_hold_bars=max_hold_bars,
                                                confirm_bars=confirm_bars,
                                                slippage_bps=slippage_bps if include_costs else 0.0,
                                                commission_per_share=commission_per_share if include_costs else 0.0,
                                                borrow_annual_pct=borrow_annual_pct if include_costs else 0.0,
                                                bars_per_year=bars_per_year,
                                            )
                                            stats = summarize_trades(trades, use_net)
                                            if not stats:
                                                st.info("Backtest produced no usable trades.")
                                            else:
                                                ev_return = stats["ev_return"]
                                                avg_return = stats["avg_return"]
                                                win_rate = stats["win_rate"]
                                                avg_win = stats["avg_win"]
                                                avg_loss = stats["avg_loss"]
                                                profit_factor = stats["profit_factor"]
                                                avg_hold = stats["avg_hold"]
                                                avg_pnl = stats["avg_pnl"]
                                                ev_dollars = ev_return * gross_notional if ev_return is not None else None

                                                b1, b2, b3, b4 = st.columns(4)
                                                b1.metric("Trades", f"{stats['trades']}")
                                                b2.metric("Win Rate", f"{win_rate * 100:.1f}%")
                                                b3.metric("Avg Return / Trade", f"{avg_return * 100:.2f}%")
                                                b4.metric("EV / Trade", f"{ev_return * 100:.2f}%")

                                                c1, c2, c3, c4 = st.columns(4)
                                                c1.metric(
                                                    "Avg Win", f"{avg_win * 100:.2f}%" if avg_win is not None else "n/a"
                                                )
                                                c2.metric(
                                                    "Avg Loss",
                                                    f"{avg_loss * 100:.2f}%" if avg_loss is not None else "n/a",
                                                )
                                                if profit_factor is None:
                                                    c3.metric("Profit Factor", "n/a")
                                                else:
                                                    c3.metric("Profit Factor", f"{profit_factor:.2f}")
                                                c4.metric("Avg Hold (bars)", f"{avg_hold:.1f}")

                                                st.caption(
                                                    "EV formula: WinRate × AvgWin + (1 - WinRate) × AvgLoss."
                                                )
                                                if ev_dollars is not None:
                                                    st.caption(
                                                        f"Approx EV $/trade at gross {gross_notional:.0f}: "
                                                        f"${ev_dollars:.2f} (avg PnL: ${avg_pnl:.2f})."
                                                    )
                                                st.caption(
                                                    f"Backtest exits at Z={exit_z:.2f}, stop "
                                                    f"{'ON' if use_stop_z else 'OFF'} "
                                                    f"({stop_z:.2f}), max hold {max_hold_bars} bars."
                                                )
                                                st.caption(
                                                    f"Costs: {'ON' if include_costs else 'OFF'} | "
                                                    f"slippage {slippage_bps:.1f} bps | "
                                                    f"commission ${commission_per_share:.2f}/share | "
                                                    f"borrow {borrow_annual_pct:.1f}% annual."
                                                )
                                                st.dataframe(
                                                    stats["df"].tail(10)[
                                                        [
                                                            "direction",
                                                            "entry_z",
                                                            "exit_z",
                                                            "hold_bars",
                                                            "return_net" if include_costs else "return",
                                                            "exit_reason",
                                                        ]
                                                    ]
                                                )

                                        st.subheader("Walk-Forward OOS Results")
                                        if not run_walkforward:
                                            st.info("Enable Walk-Forward OOS to view out-of-sample results.")
                                        elif wf_table is None or wf_table.empty:
                                            st.info("Walk-forward did not produce enough folds.")
                                        else:
                                            st.dataframe(wf_table)
                                            if oos_stats:
                                                st.caption(
                                                    f"OOS EV ({'net' if include_costs else 'gross'}): "
                                                    f"{oos_stats['ev_return'] * 100:.2f}% | "
                                                    f"Trades: {oos_stats['trades']}"
                                                )
                                        fig = go.Figure()
                                        fig.add_trace(
                                            go.Scatter(
                                                x=z_score.index,
                                                y=z_score,
                                                name="Z-Score",
                                                line=dict(color="#1f77b4"),
                                            )
                                        )
                                        fig.add_trace(
                                            go.Scatter(
                                                x=z_score.index,
                                                y=[pair_z_upper] * len(z_score),
                                                name="Upper Threshold",
                                                line=dict(color="red", dash="dash"),
                                            )
                                        )
                                        fig.add_trace(
                                            go.Scatter(
                                                x=z_score.index,
                                                y=[pair_z_lower] * len(z_score),
                                                name="Lower Threshold",
                                                line=dict(color="green", dash="dash"),
                                            )
                                        )
                                        fig.add_trace(
                                            go.Scatter(
                                                x=z_score.index,
                                                y=[0] * len(z_score),
                                                name="Mean (0)",
                                                line=dict(color="black", dash="dot"),
                                            )
                                        )
                                        fig.update_layout(
                                            title=f"Z-Score: {pair_asset_1} vs {pair_asset_2}",
                                            xaxis_title="Date",
                                            yaxis_title="Z-Score",
                                            height=450,
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                        if pair_show_table:
                                            table = pd.DataFrame(
                                                {
                                                    pair_asset_1: pair_prices[pair_asset_1],
                                                    pair_asset_2: pair_prices[pair_asset_2],
                                                    "Spread": spread,
                                                    "Z-Score": z_score,
                                                }
                                            )
                                            table["Signal"] = np.where(
                                                z_score > pair_z_upper,
                                                f"SHORT {pair_asset_1} / LONG {pair_asset_2}",
                                                np.where(
                                                    z_score < pair_z_lower,
                                                    f"LONG {pair_asset_1} / SHORT {pair_asset_2}",
                                                    "",
                                                ),
                                            )
                                            st.dataframe(table.tail(15))

st.markdown("---")
st.header("Pair Scanner (Ranked by OOS EV + Stability)")
st.caption(
    "Scans a universe of tickers, ranks candidate pairs by out-of-sample EV and stability. "
    "Uses the same settings as the Decision Engine above."
)

with st.form("pair_scanner_form"):
    scanner_tickers = st.text_area(
        "Universe tickers (comma-separated)",
        value="GLD,GDX,SLV,USO,UNG,KO,PEP,XOM,CVX,JNJ,PFE",
    )
    col_scan_a, col_scan_b, col_scan_c = st.columns(3)
    max_pairs = col_scan_a.number_input("Max Pairs to Evaluate", value=50, min_value=10, step=10)
    top_n = col_scan_b.number_input("Top N Results", value=10, min_value=5, step=5)
    scanner_use_log = col_scan_c.checkbox("Use log prices", value=pair_use_log)
    run_scanner = st.form_submit_button("Run Scanner")

if run_scanner:
    tickers = [t.strip().upper() for t in scanner_tickers.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) < 2:
        st.error("Enter at least two tickers for scanning.")
    else:
        pair_start, pair_end = None, None
        if isinstance(pair_dates, (list, tuple)) and len(pair_dates) == 2:
            pair_start, pair_end = pair_dates
        if pair_start is None or pair_end is None:
            pair_start, pair_end = default_pair_start, default_pair_end
        if pair_start >= pair_end:
            st.error("End date must be after start date.")
        else:
            with st.spinner("Downloading universe data..."):
                universe_prices = get_universe_prices(tickers, pair_start, pair_end, pair_interval)
            if universe_prices is None or universe_prices.empty:
                st.error("No data returned for the tickers/date range.")
            else:
                pairs = list(itertools.combinations(tickers, 2))
                if len(pairs) > max_pairs:
                    st.warning(
                        f"Pair count {len(pairs)} exceeds max {max_pairs}; truncating."
                    )
                    pairs = pairs[: int(max_pairs)]

                results = []
                progress = st.progress(0)
                total = len(pairs)
                min_bars = max(
                    int(z_window),
                    int(rolling_window) if use_rolling_stability else 0,
                    int(long_vol_window) if require_spread_vol else 0,
                    int(regime_window) if require_corr else 0,
                    int(max_hold_bars) + 5,
                )
                bars_per_day = estimate_bars_per_day(universe_prices.index, pair_interval)
                bars_per_year = max(bars_per_day * 252, 1)

                for idx, (a, b) in enumerate(pairs):
                    progress.progress(min((idx + 1) / total, 1.0))
                    if a not in universe_prices.columns or b not in universe_prices.columns:
                        continue
                    df = universe_prices[[a, b]].dropna()
                    if len(df) < min_bars:
                        continue

                    price_input = np.log(df) if scanner_use_log else df
                    hedge_ratio, r2 = estimate_hedge_ratio(price_input[a], price_input[b])
                    if hedge_ratio is None or np.isnan(hedge_ratio):
                        continue
                    spread = price_input[a] - (hedge_ratio * price_input[b])
                    spread_std = float(spread.std(ddof=0))
                    if spread_std == 0 or np.isnan(spread_std):
                        continue
                    if use_rolling_z and len(spread) >= int(z_window):
                        rolling_mean = spread.rolling(int(z_window)).mean()
                        rolling_std = spread.rolling(int(z_window)).std(ddof=0)
                        z_score = (spread - rolling_mean) / rolling_std
                    else:
                        z_score = (spread - spread.mean()) / spread_std
                    z_valid = z_score.dropna()
                    if z_valid.empty:
                        continue
                    latest_z = float(z_valid.iloc[-1])

                    coint_p = compute_coint_pvalue(price_input[a], price_input[b])
                    adf_p = compute_adf_pvalue(spread)
                    half_life = compute_half_life(spread)
                    hurst = compute_hurst_exponent(spread)

                    rolling_beta_ok = True
                    rolling_coint_ok = True
                    rolling_beta_last = None
                    rolling_coint_pass_rate = None
                    if use_rolling_stability and len(price_input) >= int(rolling_window):
                        do_coint = bool(require_rolling_coint and HAS_STATSMODELS)
                        rolling_beta, rolling_coint = compute_rolling_stats(
                            price_input[a],
                            price_input[b],
                            int(rolling_window),
                            do_coint,
                        )
                        if rolling_beta is not None and not rolling_beta.dropna().empty:
                            rolling_beta_last = float(rolling_beta.dropna().iloc[-1])
                            if hedge_ratio != 0:
                                drift = abs(rolling_beta_last - hedge_ratio) / abs(hedge_ratio)
                                rolling_beta_ok = drift <= (max_beta_change_pct / 100.0)
                            else:
                                rolling_beta_ok = False
                        else:
                            rolling_beta_ok = False
                        if require_rolling_coint:
                            if rolling_coint is None or rolling_coint.dropna().empty:
                                rolling_coint_ok = False
                            else:
                                valid_p = rolling_coint.dropna()
                                rolling_coint_pass_rate = float(
                                    (valid_p <= rolling_coint_p_max).mean()
                                )
                                rolling_coint_ok = (
                                    float(valid_p.iloc[-1]) <= rolling_coint_p_max
                                    and rolling_coint_pass_rate >= min_coint_pass_rate
                                )
                    elif use_rolling_stability:
                        rolling_beta_ok = False
                        rolling_coint_ok = False if require_rolling_coint else True

                    rolling_corr_ok = True
                    rolling_corr_last = None
                    spread_vol_ok = True
                    spread_vol_ratio_last = None
                    if require_corr and len(price_input) >= int(regime_window):
                        rolling_corr = price_input[a].rolling(int(regime_window)).corr(price_input[b])
                        if not rolling_corr.dropna().empty:
                            rolling_corr_last = float(rolling_corr.dropna().iloc[-1])
                            rolling_corr_ok = rolling_corr_last >= min_corr
                        else:
                            rolling_corr_ok = False
                    elif require_corr:
                        rolling_corr_ok = False

                    if require_spread_vol and len(spread) >= int(long_vol_window):
                        short_std = spread.rolling(int(regime_window)).std(ddof=0)
                        long_std = spread.rolling(int(long_vol_window)).std(ddof=0)
                        ratio = short_std / long_std
                        if not ratio.dropna().empty:
                            spread_vol_ratio_last = float(ratio.dropna().iloc[-1])
                            spread_vol_ok = spread_vol_ratio_last <= max_spread_vol_ratio
                        else:
                            spread_vol_ok = False
                    elif require_spread_vol:
                        spread_vol_ok = False

                    confirm_bars = max(int(signal_confirm_bars), 1)
                    above = z_score >= pair_z_upper
                    below = z_score <= pair_z_lower
                    if confirm_bars > 1:
                        above = above.rolling(confirm_bars).sum() == confirm_bars
                        below = below.rolling(confirm_bars).sum() == confirm_bars
                    signal = "NONE"
                    if not above.dropna().empty and bool(above.dropna().iloc[-1]):
                        signal = "SHORT"
                    elif not below.dropna().empty and bool(below.dropna().iloc[-1]):
                        signal = "LONG"

                    use_net = bool(include_costs)
                    wf_table, oos_stats = walk_forward_backtest(
                        pd.DataFrame({"z": z_score, "p1": df[a], "p2": df[b]}).dropna(),
                        hedge_ratio,
                        pair_z_upper,
                        pair_z_lower,
                        exit_z,
                        stop_z if use_stop_z else None,
                        max_hold_bars,
                        confirm_bars,
                        int(walk_folds),
                        float(walk_train_ratio),
                        int(min_trades_per_fold),
                        use_net,
                        slippage_bps if include_costs else 0.0,
                        commission_per_share if include_costs else 0.0,
                        borrow_annual_pct if include_costs else 0.0,
                        bars_per_year,
                    )

                    oos_ev = oos_stats["ev_return"] if oos_stats else None
                    oos_trades = oos_stats["trades"] if oos_stats else 0
                    coint_ok = (coint_p is not None and coint_p <= coint_p_max) if require_coint else True
                    adf_ok = (adf_p is not None and adf_p <= adf_p_max) if require_adf else True
                    half_life_ok = (
                        (half_life is not None and min_half_life <= half_life <= max_half_life)
                        if require_half_life
                        else True
                    )
                    hurst_ok = (hurst is not None and hurst <= hurst_max) if require_hurst else True
                    r2_ok = (r2 is not None and r2 >= min_r2) if require_r2 else True
                    oos_ok = (oos_ev is not None and oos_ev > 0) if require_oos_positive else True
                    eligible = all(
                        [
                            coint_ok,
                            adf_ok,
                            half_life_ok,
                            hurst_ok,
                            r2_ok,
                            rolling_beta_ok if use_rolling_stability else True,
                            rolling_coint_ok if (use_rolling_stability and require_rolling_coint) else True,
                            rolling_corr_ok if require_corr else True,
                            spread_vol_ok if require_spread_vol else True,
                            oos_ok if run_walkforward else True,
                        ]
                    )

                    results.append(
                        {
                            "Pair": f"{a}-{b}",
                            "OOS EV %": oos_ev * 100 if oos_ev is not None else None,
                            "OOS Trades": oos_trades,
                            "Latest Z": latest_z,
                            "Signal": signal,
                            "Coint p": coint_p,
                            "ADF p": adf_p,
                            "Half-Life": half_life,
                            "Hurst": hurst,
                            "R^2": r2,
                            "Roll Beta Drift OK": rolling_beta_ok,
                            "Roll Coint OK": rolling_coint_ok,
                            "Roll Corr": rolling_corr_last,
                            "Spread Vol Ratio": spread_vol_ratio_last,
                            "Eligible": eligible,
                        }
                    )

                progress.empty()
                if not results:
                    st.info("Scanner produced no eligible pairs.")
                else:
                    res_df = pd.DataFrame(results)
                    res_df = res_df.sort_values(
                        by=["Eligible", "OOS EV %"],
                        ascending=[False, False],
                        na_position="last",
                    )
                    st.dataframe(res_df.head(int(top_n)))
