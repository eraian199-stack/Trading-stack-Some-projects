"""
Human-facing evaluation reports: leaderboards, formatted metric lines, and
calibration tables.

These functions turn the raw dicts/arrays produced by `metrics` and
`walk_forward` into things a person reads in a notebook, the CLI, or the
Streamlit app. They contain no scoring logic of their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

# Metric columns surfaced on the leaderboard, in display order. log_loss is the
# primary metric, so the table is sorted by it (ascending -- lower is better).
_LEADERBOARD_COLUMNS: list[str] = ["log_loss", "rps", "brier", "accuracy", "ece", "n"]


def leaderboard(results: dict[str, dict]) -> pd.DataFrame:
    """Tabulate {model_name: metrics_dict} into a frame sorted by log_loss.

    `results` maps a model label to a metrics dict (as returned by
    `metrics.all_metrics` / `walk_forward.evaluate_model`). The index is the model
    name; columns are the standard metrics. Sorted ascending on log_loss (best
    model first); models missing log_loss sink to the bottom.
    """
    if not results:
        return pd.DataFrame(columns=_LEADERBOARD_COLUMNS)

    rows = {}
    for name, m in results.items():
        rows[name] = {col: m.get(col, np.nan) for col in _LEADERBOARD_COLUMNS}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[_LEADERBOARD_COLUMNS]
    df.index.name = "model"
    return df.sort_values("log_loss", kind="mergesort", na_position="last")


def format_metrics(m: dict) -> str:
    """One-line human summary of a metrics dict.

    Robust to missing keys (e.g. an empty market comparison): only present,
    numeric values are rendered.
    """
    order = ["log_loss", "rps", "brier", "accuracy", "ece", "n"]
    parts: list[str] = []
    for key in order:
        if key not in m:
            continue
        val = m[key]
        if key == "n":
            parts.append(f"n={int(val)}")
        else:
            try:
                parts.append(f"{key}={float(val):.4f}")
            except (TypeError, ValueError):
                parts.append(f"{key}={val}")
    return "  ".join(parts)


def calibration_table(probs, outcomes, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: pool all (match, class) predictions into confidence bins.

    For each bin, report the count, mean predicted probability, observed
    frequency, and the calibration gap (|predicted - observed|). This is the
    per-bin breakdown behind `metrics.expected_calibration_error`. Empty bins are
    omitted.
    """
    oh = metrics._onehot(outcomes).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    records = []
    total = len(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        count = int(m.sum())
        if count == 0:
            continue
        mean_pred = float(p[m].mean())
        observed = float(oh[m].mean())
        records.append(
            {
                "bin_lo": round(float(lo), 3),
                "bin_hi": round(float(hi), 3),
                "count": count,
                "weight": round(count / total, 4) if total else 0.0,
                "mean_predicted": round(mean_pred, 4),
                "observed_freq": round(observed, 4),
                "gap": round(abs(mean_pred - observed), 4),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "bin_lo", "bin_hi", "count", "weight",
            "mean_predicted", "observed_freq", "gap",
        ],
    )
