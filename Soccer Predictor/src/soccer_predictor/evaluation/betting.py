"""
Value-betting backtest and closing-line value (CLV).

We benchmark a model against the book because "good accuracy" and "makes money"
are different questions. A bet is placed only when the model thinks an outcome
has positive expected value beyond a safety margin:

    EV = model_prob * decimal_odds - 1 > edge_threshold

The original implementation had a bug: it placed a bet on EVERY qualifying
outcome within a match (so a match could carry up to three simultaneous bets).
A single match has exactly one result, so betting all three is incoherent
book-arbitrage fantasy. With `best_edge_only=True` (the default) we place AT MOST
ONE bet per match -- the single highest-EV qualifying outcome.

A POSITIVE backtest ROI does NOT prove real edge: it can be overfitting,
survivorship in the odds, or beating opening rather than closing prices. Treat
it as a hypothesis, never a guarantee (project non-negotiable #10). Closing-line
value (`closing_line_value`) is the more honest test -- beating the closing line
is the closest thing to proof of skill.

All odds are read from the canonical columns `home_odds`/`draw_odds`/`away_odds`
(and `closing_home_odds`/... for CLV); the realised outcome is `result`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import schemas

CLASSES: list[str] = schemas.RESULT_CLASSES  # ["H", "D", "A"]
_CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def max_drawdown(pnl_series) -> float:
    """Maximum peak-to-trough drop of a cumulative-PnL series.

    `pnl_series` is the sequence of per-bet profits (NOT already cumulative).
    Returns a non-negative number: the largest decline from a running peak of
    cumulative profit. 0.0 for an empty / monotonically rising series.
    """
    pnl = np.asarray(pnl_series, dtype=float)
    if pnl.size == 0:
        return 0.0
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = running_peak - equity
    return float(np.max(drawdowns))


# --------------------------------------------------------------------------- #
# Value-betting backtest
# --------------------------------------------------------------------------- #
def betting_backtest(
    probs,
    test_df: pd.DataFrame,
    edge_threshold: float = 0.05,
    stake: float = 1.0,
    kelly_fraction: float = 0.0,
    best_edge_only: bool = True,
) -> dict:
    """Flat-stake (or fractional-Kelly) value betting against the book.

    Bet an outcome only when ``model_prob * decimal_odds - 1 > edge_threshold``.
    With ``best_edge_only=True`` (default) at most ONE bet is placed per match --
    the highest-EV qualifying outcome -- which is the only coherent policy for a
    single-result event. With ``best_edge_only=False`` every qualifying outcome
    is backed independently (the legacy multi-bet behaviour, kept for analysis).

    Returns: n_bets, turnover, profit, roi, hit_rate, max_drawdown.
    """
    probs = np.asarray(probs, dtype=float)
    if "result" not in test_df.columns:
        test_df = schemas.add_result_column(test_df)

    odds = (
        test_df[schemas.ODDS_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    outcomes = test_df["result"].to_numpy()

    n_bets = 0
    staked = 0.0
    profit = 0.0
    wins = 0
    pnl: list[float] = []  # per-bet profit, in placement order, for drawdown

    for i in range(len(test_df)):
        o_i = outcomes[i]
        true_idx = _CLASS_INDEX.get(str(o_i), -1)

        # Collect every qualifying outcome (positive EV beyond the threshold).
        candidates: list[tuple[float, int, float, float]] = []  # (ev, c, odds, stake)
        for c in range(3):
            o = odds[i, c]
            if not np.isfinite(o) or o <= 1.0:
                continue
            p = probs[i, c]
            ev = p * o - 1.0
            if ev <= edge_threshold:
                continue
            if kelly_fraction > 0:
                b = o - 1.0
                k = max(0.0, (b * p - (1.0 - p)) / b) if b > 0 else 0.0
                s = stake * kelly_fraction * k
            else:
                s = stake
            if s <= 0:
                continue
            candidates.append((ev, c, o, s))

        if not candidates:
            continue

        if best_edge_only:
            # One bet per match: keep only the single highest-EV candidate.
            candidates = [max(candidates, key=lambda t: t[0])]

        for ev, c, o, s in candidates:
            n_bets += 1
            staked += s
            if true_idx == c:
                won = s * (o - 1.0)
                profit += won
                wins += 1
                pnl.append(won)
            else:
                profit -= s
                pnl.append(-s)

    roi = (profit / staked) if staked > 0 else 0.0
    return {
        "n_bets": int(n_bets),
        "turnover": round(float(staked), 2),
        "profit": round(float(profit), 2),
        "roi": round(float(roi), 4),
        "hit_rate": round(float(wins / n_bets), 4) if n_bets else 0.0,
        "max_drawdown": round(max_drawdown(pnl), 2),
    }


# --------------------------------------------------------------------------- #
# Closing-line value
# --------------------------------------------------------------------------- #
def closing_line_value(
    probs,
    test_df: pd.DataFrame,
    edge_threshold: float = 0.05,
) -> dict:
    """Closing-line value of the bets a model would place.

    For each bet (chosen exactly as in ``betting_backtest`` with
    ``best_edge_only=True``), compare the odds taken (opening / available
    ``home_odds``...) against the closing odds (``closing_*``). Beating the close
    -- i.e. taking a price higher than the closing price -- is the strongest
    cheap evidence of genuine edge.

    Returns mean log CLV (ln(odds_taken / closing_odds), positive is good) and
    the fraction of bets that beat the close. Requires both odds and closing odds
    to be present; bets without a closing price are skipped.
    """
    probs = np.asarray(probs, dtype=float)
    have_open = schemas.has_columns(test_df, schemas.ODDS_COLUMNS)
    have_close = schemas.has_columns(test_df, schemas.CLOSING_ODDS_COLUMNS)
    if not (have_open and have_close):
        return {"n_bets": 0, "mean_log_clv": float("nan"), "pct_beat_close": float("nan")}

    open_odds = (
        test_df[schemas.ODDS_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    close_odds = (
        test_df[schemas.CLOSING_ODDS_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    log_clvs: list[float] = []
    n_beat = 0
    for i in range(len(test_df)):
        # Pick the single best qualifying bet on the OPENING line (same policy as
        # betting_backtest with best_edge_only=True).
        best_ev = edge_threshold
        best_c = -1
        for c in range(3):
            o = open_odds[i, c]
            if not np.isfinite(o) or o <= 1.0:
                continue
            ev = probs[i, c] * o - 1.0
            if ev > best_ev:
                best_ev = ev
                best_c = c
        if best_c < 0:
            continue

        taken = open_odds[i, best_c]
        closing = close_odds[i, best_c]
        if not np.isfinite(closing) or closing <= 1.0:
            continue  # no usable closing price for this selection

        log_clv = float(np.log(taken / closing))
        log_clvs.append(log_clv)
        if taken > closing:
            n_beat += 1

    n = len(log_clvs)
    return {
        "n_bets": int(n),
        "mean_log_clv": round(float(np.mean(log_clvs)), 5) if n else float("nan"),
        "pct_beat_close": round(float(n_beat / n), 4) if n else float("nan"),
    }
