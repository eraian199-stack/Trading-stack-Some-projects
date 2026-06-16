"""
Model ensemble.

A weighted average of several base models' 1X2 probabilities. The important
subtlety: some components legitimately produce NaN for some rows (the market
baseline emits NaN when a fixture has no odds), so the weights are renormalised
*per row* over the components that actually produced a finite prediction. A
fixture with no odds simply has the market's 0.25 weight redistributed across the
remaining models rather than poisoning the average.

``default_ensemble`` wires the canonical stack:
    Dixon-Coles        0.35   generative goals
    OutcomeClassifier  0.25   discriminative features
    MarketBaseline     0.25   de-vigged odds (the benchmark, used as a component)
    EloGoalsModel      0.15   robust Elo-driven scoreline floor
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel
from .dixon_coles import DixonColes
from .elo_goals import EloGoalsModel
from .market_baseline import MarketBaseline
from .outcome_classifier import OutcomeClassifier


class Ensemble(BaseModel):
    """Per-row weight-renormalising average of component models."""

    def __init__(self, components: list[tuple[BaseModel, float]]):
        if not components:
            raise ValueError("Ensemble needs at least one component.")
        self.components = list(components)

    def fit(self, train_df: pd.DataFrame) -> "Ensemble":
        for model, _weight in self.components:
            model.fit(train_df)
        return self

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        n = len(test_df)
        # Stack each component's (n, 3) prediction and its scalar weight.
        preds = np.empty((len(self.components), n, 3), dtype=float)
        weights = np.empty(len(self.components), dtype=float)
        for k, (model, weight) in enumerate(self.components):
            p = np.asarray(model.predict_proba(test_df), dtype=float)
            preds[k] = p
            weights[k] = float(weight)

        # A component contributes to a row only when that row is fully finite.
        finite = np.isfinite(preds).all(axis=2)  # (n_components, n)
        w = weights[:, None] * finite  # zero-out weight where NaN
        denom = w.sum(axis=0)  # (n,)

        # Replace NaNs with 0 so the masked weights kill them in the sum.
        clean = np.where(finite[:, :, None], preds, 0.0)
        weighted = (clean * w[:, :, None]).sum(axis=0)  # (n, 3)

        out = np.full((n, 3), np.nan, dtype=float)
        ok = denom > 0
        out[ok] = weighted[ok] / denom[ok, None]
        # Renormalise rows to sum to 1 (guards against tiny float drift).
        row_sums = out.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            out = np.where(row_sums > 0, out / row_sums, out)
        return out


def default_ensemble(form_window: int = 6) -> Ensemble:
    """The canonical 4-model ensemble (weights sum to 1.0)."""
    return Ensemble(
        [
            (DixonColes(), 0.35),
            (OutcomeClassifier(kind="gbm", form_window=form_window), 0.25),
            (MarketBaseline(), 0.25),
            (EloGoalsModel(), 0.15),
        ]
    )
