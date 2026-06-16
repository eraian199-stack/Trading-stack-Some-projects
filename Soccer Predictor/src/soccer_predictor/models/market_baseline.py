"""
Market baseline model.

The de-vigged closing line is the benchmark that actually matters: if a fancy
model cannot beat the market on out-of-sample log loss, it has no betting edge
(only research / learning value). This model converts bookmaker 1X2 odds into
probabilities by removing the overround.

It is deliberately honest about coverage: a row that lacks all three valid odds
gets a row of NaN rather than invented probabilities. Callers select the rows
that actually have odds with ``schemas.rows_with_complete_odds`` and the ensemble
redistributes weight away from NaN components per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import schemas
from .base import BaseModel


class MarketBaseline(BaseModel):
    """De-vig bookmaker ``home_odds`` / ``draw_odds`` / ``away_odds`` into
    probabilities. ``fit`` is a no-op (there is nothing to learn)."""

    def fit(self, train_df: pd.DataFrame) -> "MarketBaseline":
        return self

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        n = len(test_df)
        out = np.full((n, 3), np.nan, dtype=float)
        if not schemas.has_columns(test_df, schemas.ODDS_COLUMNS):
            return out

        odds = test_df[schemas.ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce")
        odds_arr = odds.to_numpy(dtype=float)
        # A row is usable only when all three odds are present and > 1.0.
        valid = np.isfinite(odds_arr).all(axis=1) & (odds_arr > 1.0).all(axis=1)
        if valid.any():
            inv = 1.0 / odds_arr[valid]
            out[valid] = inv / inv.sum(axis=1, keepdims=True)
        return out
