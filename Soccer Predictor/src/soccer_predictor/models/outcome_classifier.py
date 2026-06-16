"""
Discriminative 1X2 outcome classifier.

Where the generative models (Dixon-Coles, xG-Poisson, Elo-goals) impose a goal
process, this model learns the H/D/A target directly from engineered, leakage-free
features built by ``features.feature_store``. It is more flexible with signal but
only predicts what it is trained on.

Backends:
    kind="gbm"     LightGBM if importable, else sklearn HistGradientBoosting
                   (same algorithm family).
    kind="logreg"  sklearn LogisticRegression with median-impute + standardize.

The exact feature set is chosen at fit time via
``feature_store.available_feature_columns`` so optional groups (xG, market,
rank, squad) are used only when present, and ``predict_proba`` always returns
columns aligned to [H, D, A].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import base
from .base import BaseModel
from ..features import feature_store

CLASSES = base.CLASSES  # ["H", "D", "A"]


class OutcomeClassifier(BaseModel):
    """Gradient-boosting or logistic-regression 1X2 classifier."""

    def __init__(
        self,
        kind: str = "gbm",
        build_feats: bool = True,
        form_window: int = 6,
        feature_cols: list[str] | None = None,
    ):
        """``kind``: 'gbm' (gradient boosting) or 'logreg' (logistic regression).
        ``build_feats``: run ``feature_store.build_features`` if features absent.
        ``feature_cols``: restrict to this subset of feature columns (intersected
        with what's available) -- used for feature-group ablations; None = all."""
        self.kind = kind
        self.build_feats = build_feats
        self.form_window = form_window
        self.feature_cols = feature_cols
        self.model = None
        self.imputer = None
        self.scaler = None
        self.backend: str | None = None
        self.cols: list[str] | None = None  # feature columns chosen at fit time

    # -- feature plumbing --------------------------------------------------
    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rebuild only when the base features are absent (the elo_home column is
        # a reliable sentinel for "feature_store has run on this frame").
        if self.build_feats and "elo_home" not in df.columns:
            return feature_store.build_features(df, form_window=self.form_window)
        return df

    # -- fit ---------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "OutcomeClassifier":
        train_df = self._ensure_features(train_df)
        if "result" not in train_df.columns:
            from ..data import schemas

            train_df = schemas.add_result_column(train_df)
        # Train only on rows with a real label and complete feature columns.
        available = feature_store.available_feature_columns(train_df)
        if self.feature_cols is not None:
            self.cols = [c for c in available if c in set(self.feature_cols)]
            if not self.cols:
                raise ValueError("feature_cols selected no available columns.")
        else:
            self.cols = available
        labelled = train_df[train_df["result"].isin(CLASSES)]
        X = labelled[self.cols].to_numpy(dtype=float)
        y = (
            labelled["result"]
            .map({c: i for i, c in enumerate(CLASSES)})
            .to_numpy()
        )

        if self.kind == "logreg":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self.imputer = SimpleImputer(strategy="median")
            Xi = self.imputer.fit_transform(X)
            self.scaler = StandardScaler().fit(Xi)
            Xi = self.scaler.transform(Xi)
            self.model = LogisticRegression(max_iter=2000, C=1.0)
            self.model.fit(Xi, y)
            self.backend = "logreg"
        else:
            try:
                from lightgbm import LGBMClassifier

                self.model = LGBMClassifier(
                    objective="multiclass",
                    num_class=3,
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=40,
                    verbose=-1,
                )
                self.backend = "lightgbm"
            except ImportError:
                from sklearn.ensemble import HistGradientBoostingClassifier

                self.model = HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.03,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                )
                self.backend = "hgb"
            # GBM backends tolerate NaNs natively (HGB) or via their own
            # handling (LightGBM), so no imputation here.
            self.model.fit(X, y)
        return self

    # -- predict -----------------------------------------------------------
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.cols is None:
            raise RuntimeError("OutcomeClassifier must be fit before predicting.")
        if len(test_df) == 0:
            return np.zeros((0, 3), dtype=float)
        test_df = self._ensure_features(test_df)
        if len(test_df) == 0:  # features step can drop unusable rows
            return np.zeros((0, 3), dtype=float)
        # Reindex to the exact training columns. Test data may legitimately lack
        # optional feature groups (xG-rolling, market-implied) that the training
        # frame had -- fill those with NaN so the column set always matches the
        # fitted model. GBM backends tolerate NaN natively; logreg imputes below.
        X = test_df.reindex(columns=self.cols).to_numpy(dtype=float)
        if self.backend == "logreg":
            X = self.scaler.transform(self.imputer.transform(X))
        proba = np.asarray(self.model.predict_proba(X), dtype=float)
        # Scatter the backend's columns into a fixed 3-wide [H, D, A] array by
        # class identity. A training slice can lack a class (e.g. an early
        # walk-forward fold with no draws -> classes_ == [0, 2]); filling the
        # absent class with 0 keeps the (n, 3) sum-to-1 contract instead of
        # raising on classes.index(missing).
        out = np.zeros((proba.shape[0], 3), dtype=float)
        for col, cls in enumerate(self.model.classes_):
            out[:, int(cls)] = proba[:, col]
        sums = out.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return out / sums
