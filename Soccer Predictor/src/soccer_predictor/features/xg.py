"""
Rolling expected-goals (xG) features.

When a source supplies per-match xG (``home_xg`` / ``away_xg``), rolling xG-for /
xG-against per team is usually a stronger signal than rolling actual goals: xG is
a lower-variance estimate of a team's true scoring/conceding rate, so it
stabilises faster over a short window.

The feature store owns the no-leakage walk and the per-team history deques; this
module just declares the output column names and exposes a small helper to pull
the rolling xG-for / xG-against pair out of a team's history deque. (Note: 2026
removed FBref's free xG feed -- see docs/data_sources.md -- so these columns are
often simply absent, and everything degrades gracefully when so.)
"""

from __future__ import annotations

from collections import deque

from .rolling import rolling_mean

# Rolling xG columns added by the feature store when xG is present.
XG_ROLL_COLUMNS: list[str] = [
    "home_xg_for",
    "home_xg_against",
    "away_xg_for",
    "away_xg_against",
]


def rolling_xg(history: deque) -> tuple[float, float]:
    """Return (xg_for, xg_against) per-game means from a team's history deque.

    Records are expected to carry ``xgf`` / ``xga`` keys (as written by the
    feature store). Empty / xG-less history -> (nan, nan).
    """
    return rolling_mean(history, "xgf"), rolling_mean(history, "xga")
