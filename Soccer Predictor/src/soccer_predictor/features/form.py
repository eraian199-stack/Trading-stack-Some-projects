"""
Match-points helper.

A single, deliberately tiny module so the points convention (3 for a win, 1 for
a draw, 0 for a loss) lives in exactly one place and is reused by the feature
store when it appends a played match to a team's rolling history.
"""

from __future__ import annotations


def points_for(home_score: int, away_score: int) -> tuple[int, int]:
    """Return (home_points, away_points) for a finished match.

    Standard association-football scoring: win 3, draw 1, loss 0.
    """
    hs = int(home_score)
    asc = int(away_score)
    if hs > asc:
        return 3, 0
    if hs == asc:
        return 1, 1
    return 0, 3
