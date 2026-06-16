"""
Squad-strength differential features.

These are *static, pre-kickoff* attributes attached to each row by its source
(squad market value, an injury-burden score, FIFA ranking). Unlike form / Elo
they are not accumulated from match history, so there is no leakage concern as
long as the source values themselves are as-of the match date. We turn each pair
of side columns into a single signed difference (home minus away), which is the
form a model wants, and add a column ONLY when both of its source columns are
present (so the feature set stays honest about what data actually exists).
"""

from __future__ import annotations

import pandas as pd

# Differential columns added by add_squad_features (home - away).
SQUAD_FEATURE_COLUMNS: list[str] = [
    "squad_value_diff",
    "injury_score_diff",
    "fifa_rank_diff",
]

# Each differential and the (home, away) source columns it derives from.
_SQUAD_SOURCES: dict[str, tuple[str, str]] = {
    "squad_value_diff": ("home_squad_value", "away_squad_value"),
    "injury_score_diff": ("home_injury_score", "away_injury_score"),
    "fifa_rank_diff": ("home_fifa_rank", "away_fifa_rank"),
}


def add_squad_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add squad-strength differentials where their source columns exist.

    A differential column is created only when BOTH of its source columns are
    present; otherwise it is omitted entirely (downstream code checks presence).
    Each diff is home_value - away_value, coerced numerically (bad -> NaN).
    """
    df = df.copy()
    for diff_col, (home_col, away_col) in _SQUAD_SOURCES.items():
        if home_col in df.columns and away_col in df.columns:
            home = pd.to_numeric(df[home_col], errors="coerce")
            away = pd.to_numeric(df[away_col], errors="coerce")
            df[diff_col] = home - away
    return df


def available_squad_columns(df: pd.DataFrame) -> list[str]:
    """Which SQUAD_FEATURE_COLUMNS would be produced for this frame."""
    return [
        diff_col
        for diff_col, (home_col, away_col) in _SQUAD_SOURCES.items()
        if home_col in df.columns and away_col in df.columns
    ]
