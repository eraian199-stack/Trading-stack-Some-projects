"""
Rolling-window form helpers.

The feature store keeps a ``collections.deque`` of recent match records per team
(most recent ``form_window`` games). Each record is a small dict carrying at
least ``pts`` (points won), ``gf`` (goals for) and ``ga`` (goals against), plus
optional keys such as ``xgf`` / ``xga`` when xG is available. These helpers turn
such a deque into per-game means, returning NaN for an empty history so a team's
first appearance is honestly marked as "no information" rather than zero.

Pure functions, no leakage concerns of their own: leakage safety lives in the
feature store, which only ever feeds these the team's EARLIER matches.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def rolling_form(history: deque) -> tuple[float, float, float]:
    """Return (ppg, gf, ga) per-game means over the team's recent window.

    Empty history -> (nan, nan, nan).
    """
    if not history:
        return np.nan, np.nan, np.nan
    ppg = float(np.mean([h["pts"] for h in history]))
    gf = float(np.mean([h["gf"] for h in history]))
    ga = float(np.mean([h["ga"] for h in history]))
    return ppg, gf, ga


def rolling_mean(history: deque, key: str) -> float:
    """Mean of ``key`` across the records in ``history`` (nan if empty/missing).

    Records missing the key are skipped; if none have it, the result is NaN.
    """
    if not history:
        return np.nan
    values = [h[key] for h in history if key in h and h[key] is not None]
    if not values:
        return np.nan
    return float(np.mean(values))
