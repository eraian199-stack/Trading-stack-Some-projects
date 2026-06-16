"""
Proper scoring metrics for 1X2 (Home/Draw/Away) probability forecasts.

This is where most football models are secretly broken, so the metrics get a lot
of care. We score with PROPER metrics, not just accuracy:

    * log loss  -- penalises confident wrong probabilities; the main metric.
    * RPS       -- Ranked Probability Score. THE metric for 1X2 because it
                   respects order (predicting Away when Home happens is worse
                   than predicting Draw). Lower is better. Classes are ordered
                   H < D < A (matching schemas.RESULT_CLASSES).
    * Brier     -- multiclass squared error.
    * accuracy  -- intuitive but weak; reported for context only.
    * ECE       -- expected calibration error: do the stated probabilities match
                   observed frequencies?

`probs` is an (n, 3) array with columns [P(H), P(D), P(A)]; `outcomes` is an
array of "H"/"D"/"A" labels (the canonical `result` column).
"""

from __future__ import annotations

import numpy as np

from ..data import schemas

# Class order H < D < A -- the order RPS depends on.
CLASSES: list[str] = schemas.RESULT_CLASSES  # ["H", "D", "A"]
_CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
_EPS = 1e-15


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _onehot(outcomes) -> np.ndarray:
    """One-hot encode an array of "H"/"D"/"A" labels into (n, 3)."""
    outcomes = np.asarray(outcomes)
    idx = np.array([_CLASS_INDEX[str(o)] for o in outcomes])
    oh = np.zeros((len(outcomes), 3), dtype=float)
    oh[np.arange(len(outcomes)), idx] = 1.0
    return oh


def _as_probs(probs) -> np.ndarray:
    return np.asarray(probs, dtype=float)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def log_loss(probs, outcomes) -> float:
    """Mean multiclass log loss (cross-entropy). Lower is better; 0 is perfect."""
    oh = _onehot(outcomes)
    p = np.clip(_as_probs(probs), _EPS, 1.0)
    return float(-np.mean(np.sum(oh * np.log(p), axis=1)))


def rps(probs, outcomes) -> float:
    """Mean Ranked Probability Score over ordered classes H < D < A.

    Squared error between the cumulative predicted and cumulative observed
    distributions, summed over the first (r-1)=2 thresholds. Lower is better.
    """
    oh = _onehot(outcomes)
    p = _as_probs(probs)
    cum_p = np.cumsum(p, axis=1)
    cum_o = np.cumsum(oh, axis=1)
    return float(np.mean(np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1)))


def brier(probs, outcomes) -> float:
    """Multiclass Brier score (mean squared error over the 3 class probs)."""
    oh = _onehot(outcomes)
    p = _as_probs(probs)
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def accuracy(probs, outcomes) -> float:
    """Top-1 accuracy. Intuitive but weak; reported for context only."""
    p = _as_probs(probs)
    pred = np.argmax(p, axis=1)
    true = np.array([_CLASS_INDEX[str(o)] for o in np.asarray(outcomes)])
    return float(np.mean(pred == true))


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """Pool all (match, class) predictions; compare confidence to frequency.

    Bins the flattened predicted probabilities and measures the gap between mean
    predicted probability and observed frequency within each bin, weighted by bin
    population. Lower is better; 0 means perfectly calibrated.
    """
    oh = _onehot(outcomes).ravel()
    p = _as_probs(probs).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, total = 0.0, len(p)
    if total == 0:
        return 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Last bin is closed on the right so p == 1.0 is counted.
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / total) * abs(p[m].mean() - oh[m].mean())
    return float(ece)


def all_metrics(probs, outcomes) -> dict:
    """Bundle of all 1X2 metrics. Keys: log_loss, rps, brier, accuracy, ece, n."""
    return {
        "log_loss": log_loss(probs, outcomes),
        "rps": rps(probs, outcomes),
        "brier": brier(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
        "ece": expected_calibration_error(probs, outcomes),
        "n": int(len(np.asarray(outcomes))),
    }


# --------------------------------------------------------------------------- #
# Binary metrics (over/under, BTTS, ...)
# --------------------------------------------------------------------------- #
def binary_log_loss(p, y) -> float:
    """Binary cross-entropy for a single market (e.g. over 2.5, BTTS yes)."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return 0.0
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_brier(p, y) -> float:
    """Binary Brier score (mean squared error of the probability)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))
