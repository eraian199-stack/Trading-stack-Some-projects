"""
Stacked ensemble (a "super learner").

The plain ``Ensemble`` averages base models with fixed weights. A stack instead
LEARNS how to combine them, by training a meta-model on the base models'
predictions. The only way to do this without leaking is to feed the meta-model
OUT-OF-FOLD base predictions: each training row's base features come from base
models fit ONLY on earlier rows. We generate those with an expanding-window,
time-ordered split (never random k-fold), exactly mirroring how the model is used
live. The base models are then refit on the full training set for prediction.

This is the disciplined ML upgrade: it can discover, e.g., "trust the market most,
lean on Dixon-Coles for scorelines, and let the GBM nudge where it has signal" --
but it is judged the same way as everything else (walk-forward, vs the market).
A regularised logistic meta-learner is the default; it is hard to overfit on a
handful of base-probability features, which is the point.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..data import schemas
from .base import BaseModel
from .dixon_coles import DixonColes
from .elo_goals import EloGoalsModel
from .market_baseline import MarketBaseline
from .outcome_classifier import OutcomeClassifier

CLASSES = schemas.RESULT_CLASSES


def default_base_factories(with_market: bool = True) -> dict[str, Callable[[], BaseModel]]:
    """The standard base learners for the stack."""
    bases: dict[str, Callable[[], BaseModel]] = {
        "dixon_coles": DixonColes,
        "elo": EloGoalsModel,
        "gbm": lambda: OutcomeClassifier(kind="gbm"),
    }
    if with_market:
        bases["market"] = MarketBaseline
    return bases


def _stack_features(base_probs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate base [P_H, P_D] columns (drop P_A -- redundant) into a matrix.

    NaN cells (e.g. the market baseline on a fixture with no odds) are imputed
    with the mean of the other bases' value for that class+row, so the meta-model
    always sees a complete row and a missing book never blanks a prediction.
    """
    names = sorted(base_probs)
    n = len(next(iter(base_probs.values())))
    # Per-class mean across bases (ignoring NaN) for imputation.
    stacked = np.stack([base_probs[k] for k in names], axis=0)  # (B, n, 3)
    with np.errstate(invalid="ignore"):
        class_mean = np.nanmean(stacked, axis=0)  # (n, 3)
    class_mean = np.where(np.isfinite(class_mean), class_mean, 1.0 / 3.0)
    cols = []
    for k in names:
        p = base_probs[k].copy()
        mask = ~np.isfinite(p)
        if mask.any():
            p[mask] = class_mean[mask]
        cols.append(p[:, :2])  # P_H, P_D (P_A = 1 - H - D)
    return np.hstack(cols)


class StackedEnsemble(BaseModel):
    """Meta-model over time-series out-of-fold base-model predictions."""

    def __init__(
        self,
        base_factories: dict[str, Callable[[], BaseModel]] | None = None,
        meta: str = "logreg",
        n_splits: int = 5,
        min_train_frac: float = 0.3,
        with_market: bool | None = None,
    ):
        self.base_factories = base_factories
        self.meta = meta
        self.n_splits = n_splits
        self.min_train_frac = min_train_frac
        self.with_market = with_market
        self.bases_: dict[str, BaseModel] = {}
        self.meta_model_ = None
        self._meta_backend = None
        self._scaler = None
        self._base_names: list[str] = []

    def _resolve_factories(self, df: pd.DataFrame) -> dict[str, Callable[[], BaseModel]]:
        if self.base_factories is not None:
            return self.base_factories
        use_market = self.with_market
        if use_market is None:
            use_market = schemas.has_usable_odds(df)
        return default_base_factories(with_market=use_market)

    def fit(self, train_df: pd.DataFrame) -> "StackedEnsemble":
        df = schemas.completed_matches(train_df)
        if "result" not in df.columns:
            df = schemas.add_result_column(df)
        df = df[df["result"].isin(CLASSES)].sort_values(
            "date", kind="mergesort"
        ).reset_index(drop=True)
        factories = self._resolve_factories(df)
        self._base_names = sorted(factories)

        # --- out-of-fold base predictions over an expanding time-series split ---
        n = len(df)
        start = int(n * self.min_train_frac)
        block = max(1, (n - start) // max(1, self.n_splits))
        oof_feats: list[np.ndarray] = []
        oof_y: list[np.ndarray] = []
        for k in range(self.n_splits):
            lo = start + k * block
            hi = n if k == self.n_splits - 1 else start + (k + 1) * block
            if lo >= n:
                break
            tr, te = df.iloc[:lo], df.iloc[lo:hi]
            if len(te) == 0 or len(tr) < 20:
                continue
            base_probs = {}
            for name, fac in factories.items():
                m = fac()
                m.fit(tr.copy())
                base_probs[name] = np.asarray(m.predict_proba(te.copy()), dtype=float)
            oof_feats.append(_stack_features(base_probs))
            oof_y.append(
                te["result"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
            )

        if not oof_feats:
            raise ValueError("StackedEnsemble needs more data to build OOF folds.")
        X = np.vstack(oof_feats)
        y = np.concatenate(oof_y)

        self._fit_meta(X, y)

        # --- refit base models on ALL training data for live prediction ---
        self.bases_ = {}
        for name, fac in factories.items():
            m = fac()
            m.fit(df.copy())
            self.bases_[name] = m
        return self

    def _fit_meta(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.meta == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.meta_model_ = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=1.0,
            )
            self.meta_model_.fit(X, y)
            self._meta_backend = "gbm"
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler().fit(X)
            self.meta_model_ = LogisticRegression(max_iter=2000, C=1.0)
            self.meta_model_.fit(self._scaler.transform(X), y)
            self._meta_backend = "logreg"

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.meta_model_ is None:
            raise RuntimeError("StackedEnsemble must be fit before predicting.")
        if len(test_df) == 0:
            return np.zeros((0, 3), dtype=float)
        base_probs = {
            name: np.asarray(m.predict_proba(test_df.copy()), dtype=float)
            for name, m in self.bases_.items()
        }
        X = _stack_features(base_probs)
        if self._meta_backend == "logreg":
            X = self._scaler.transform(X)
        proba = np.asarray(self.meta_model_.predict_proba(X), dtype=float)
        out = np.zeros((proba.shape[0], 3), dtype=float)
        for col, cls in enumerate(self.meta_model_.classes_):
            out[:, int(cls)] = proba[:, col]
        sums = out.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return out / sums

    def meta_weights(self) -> dict[str, float] | None:
        """For a logistic meta-model, the learned coefficient magnitude per base
        (a rough 'how much does the stack lean on each model' read). None for GBM."""
        if self._meta_backend != "logreg" or self.meta_model_ is None:
            return None
        coef = np.abs(self.meta_model_.coef_).sum(axis=0)  # over classes
        out: dict[str, float] = {}
        for i, name in enumerate(self._base_names):
            out[name] = float(coef[2 * i] + coef[2 * i + 1])
        total = sum(out.values()) or 1.0
        return {k: round(v / total, 3) for k, v in out.items()}
