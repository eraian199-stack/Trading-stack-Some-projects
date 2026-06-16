"""
Market-implied probability features.

Bookmaker decimal odds carry an overround ("vig"): the raw inverse odds sum to
more than 1. De-vigging normalises them back to a probability simplex, giving the
market's implied P(home) / P(draw) / P(away). These are powerful features for a
classifier and the basis of the market baseline model -- but per the project
rules the market is a *benchmark first, a feature second*: we only attach these
columns where complete, valid odds exist, and leave NaN elsewhere (never invent
odds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import schemas

# Implied-probability columns added by add_market_features.
MARKET_FEATURE_COLUMNS: list[str] = [
    "mkt_imp_home",
    "mkt_imp_draw",
    "mkt_imp_away",
]


def implied_probabilities(odds_h: float, odds_d: float, odds_a: float) -> np.ndarray:
    """De-vigged [P(home), P(draw), P(away)] from decimal 1X2 odds.

    Returns an array of NaNs when any odd is missing or <= 1 (an invalid book).
    Normalisation is proportional ("basic"/multiplicative de-vig), which is the
    standard, assumption-light choice.
    """
    try:
        odds = np.array([float(odds_h), float(odds_d), float(odds_a)], dtype=float)
    except (TypeError, ValueError):
        return np.array([np.nan, np.nan, np.nan])
    if not np.all(np.isfinite(odds)) or np.any(odds <= 1.0):
        return np.array([np.nan, np.nan, np.nan])
    inv = 1.0 / odds
    total = inv.sum()
    if total <= 0:
        return np.array([np.nan, np.nan, np.nan])
    return inv / total


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mkt_imp_home/draw/away`` columns, NaN where odds are incomplete.

    Uses the canonical ``home_odds/draw_odds/away_odds`` triple. Rows without a
    full, valid book get NaN in all three columns (so an imputer / the model can
    handle the gap), matching how the market baseline filters such rows.
    """
    df = df.copy()
    out = np.full((len(df), 3), np.nan, dtype=float)
    if schemas.has_columns(df, schemas.ODDS_COLUMNS):
        odds = df[schemas.ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce")
        ok = odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
        ok_arr = ok.to_numpy()
        if ok_arr.any():
            inv = 1.0 / odds.to_numpy()[ok_arr]
            out[ok_arr] = inv / inv.sum(axis=1, keepdims=True)
    for j, col in enumerate(MARKET_FEATURE_COLUMNS):
        df[col] = out[:, j]
    return df
