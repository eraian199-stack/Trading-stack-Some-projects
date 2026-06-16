"""
Probability calibration wrapper.

A model can rank fixtures well yet still be over- or under-confident. This wraps
any ``BaseModel`` and corrects its probabilities on a *time-ordered* holdout (the
later slice of the training data), so the calibrator is fit on predictions the
base model did not train on -- and never on the future of what it calibrates.

Two methods:
    "isotonic"     per-class one-vs-rest IsotonicRegression, results renormalised
                   to a valid distribution. Flexible; needs a reasonably large
                   holdout to be stable.
    "temperature"  a single scalar T that sharpens / softens the whole vector in
                   logit space, chosen to minimise multiclass log loss. Robust on
                   small holdouts.
    "auto"         isotonic when the holdout is large enough, else temperature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from . import base
from .base import BaseModel

CLASSES = base.CLASSES  # ["H", "D", "A"]

# Minimum holdout rows before isotonic regression is trusted over temperature.
_ISOTONIC_MIN_HOLDOUT = 200


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature ``T`` to probability rows and renormalise.

    T == 1 leaves probabilities unchanged; T > 1 softens (less confident);
    0 < T < 1 sharpens. Operates in log space: p_i ** (1/T), then renormalise.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    T = float(max(T, 1e-6))
    logits = np.log(p) / T
    logits = logits - logits.max(axis=1, keepdims=True)  # stabilise exp
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Find the scalar temperature minimising multiclass log loss.

    ``outcomes`` is an array of "H"/"D"/"A". Returns T in [0.05, 20].
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    idx = np.array([CLASSES.index(o) for o in outcomes])
    rows = np.arange(len(idx))

    def neg_ll(T: float) -> float:
        scaled = temperature_scale(p, T)
        chosen = np.clip(scaled[rows, idx], 1e-12, 1.0)
        return float(-np.log(chosen).mean())

    res = minimize_scalar(neg_ll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


class CalibratedModel(BaseModel):
    """Wrap a base model and calibrate it on a time-ordered holdout."""

    def __init__(
        self, base: BaseModel, method: str = "auto", holdout_frac: float = 0.2
    ):
        self.base = base
        self.method = method
        self.holdout_frac = holdout_frac
        self.fitted_method: str | None = None
        self.temperature_: float = 1.0
        self.isotonics_: list | None = None

    # -- fit ---------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "CalibratedModel":
        df = train_df.copy()
        if "date" in df.columns:
            df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        n = len(df)
        n_holdout = int(round(n * self.holdout_frac))
        # Need both a training slice and a non-trivial holdout to calibrate.
        if n_holdout < 10 or n - n_holdout < 10:
            self.base.fit(df)
            self.fitted_method = "identity"
            self.temperature_ = 1.0
            self.isotonics_ = None
            return self

        train_slice = df.iloc[: n - n_holdout]
        holdout = df.iloc[n - n_holdout :]

        self.base.fit(train_slice)
        probs = np.asarray(self.base.predict_proba(holdout), dtype=float)

        if "result" not in holdout.columns:
            from ..data import schemas

            holdout = schemas.add_result_column(holdout)
        labels = holdout["result"].to_numpy()

        # Keep only rows with a real label AND finite base predictions.
        valid = np.array([str(x) in CLASSES for x in labels]) & np.isfinite(
            probs
        ).all(axis=1)
        probs = probs[valid]
        labels = labels[valid]

        method = self._choose_method(len(probs))
        if method == "isotonic":
            self._fit_isotonic(probs, labels)
            self.fitted_method = "isotonic"
        elif method == "temperature":
            self.temperature_ = fit_temperature(probs, labels) if len(probs) else 1.0
            self.fitted_method = "temperature"
        else:
            self.fitted_method = "identity"

        # Refit the base on the FULL training data so prediction uses all of it.
        self.base.fit(df)
        return self

    def _choose_method(self, n_holdout_valid: int) -> str:
        if n_holdout_valid < 10:
            return "identity"
        if self.method == "auto":
            return (
                "isotonic"
                if n_holdout_valid >= _ISOTONIC_MIN_HOLDOUT
                else "temperature"
            )
        return self.method

    def _fit_isotonic(self, probs: np.ndarray, labels: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression

        self.isotonics_ = []
        for c, cls in enumerate(CLASSES):
            y = (labels == cls).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[:, c], y)
            self.isotonics_.append(iso)

    # -- predict -----------------------------------------------------------
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        probs = np.asarray(self.base.predict_proba(test_df), dtype=float)
        if self.fitted_method == "temperature":
            out = self._apply_temperature(probs)
        elif self.fitted_method == "isotonic" and self.isotonics_ is not None:
            out = self._apply_isotonic(probs)
        else:
            out = probs
        # Renormalise finite rows; leave all-NaN rows as NaN (e.g. market gaps).
        row_sums = out.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            out = np.where(row_sums > 0, out / row_sums, out)
        return out

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        out = probs.copy()
        finite = np.isfinite(probs).all(axis=1)
        if finite.any():
            out[finite] = temperature_scale(probs[finite], self.temperature_)
        return out

    def _apply_isotonic(self, probs: np.ndarray) -> np.ndarray:
        out = probs.copy()
        finite = np.isfinite(probs).all(axis=1)
        if finite.any():
            cal = np.zeros((finite.sum(), 3), dtype=float)
            block = probs[finite]
            for c, iso in enumerate(self.isotonics_):
                cal[:, c] = iso.predict(block[:, c])
            out[finite] = cal
        return out
