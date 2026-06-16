"""
International-context features.

Tournament / national-team matches carry context that club matches do not:
whether the game is on neutral ground, and whether one side is the host nation
(host advantage is real even on a nominally "neutral" pitch). These are
pre-kickoff facts derived from the canonical schema, so there is no leakage.

- ``is_neutral``: 1.0 / 0.0 from the canonical ``neutral_site`` flag.
- ``is_host_home`` / ``is_host_away``: whether each side is playing in its own
  country, inferred from ``country`` (the match host nation) vs the team's
  ``home_country`` / ``away_country`` when those columns are supplied.

Columns are added only when the information to compute them is available, so the
international feature group simply does not appear for plain club frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.aliases import normalize_team_name

# International-context columns (added when computable).
INTL_FEATURE_COLUMNS: list[str] = [
    "is_neutral",
    "is_host_home",
    "is_host_away",
]


def _norm(value: object) -> str:
    return normalize_team_name(value)


def add_international_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add neutral / host-nation flags where the source supports them.

    ``is_neutral`` is added whenever ``neutral_site`` exists. The host flags are
    added only when a host ``country`` column is present together with at least
    one of the per-team country columns; an unknown side stays 0.0.
    """
    df = df.copy()

    if "neutral_site" in df.columns:
        df["is_neutral"] = df["neutral_site"].map(bool).astype(float)

    has_host = "country" in df.columns
    has_home_country = "home_country" in df.columns
    has_away_country = "away_country" in df.columns
    if has_host and (has_home_country or has_away_country):
        host = df["country"].map(_norm)
        if has_home_country:
            home_country = df["home_country"].map(_norm)
            same = (host != "") & (home_country != "") & (host == home_country)
            df["is_host_home"] = same.astype(float)
        else:
            df["is_host_home"] = np.zeros(len(df), dtype=float)
        if has_away_country:
            away_country = df["away_country"].map(_norm)
            same = (host != "") & (away_country != "") & (host == away_country)
            df["is_host_away"] = same.astype(float)
        else:
            df["is_host_away"] = np.zeros(len(df), dtype=float)

    return df


def available_international_columns(df: pd.DataFrame) -> list[str]:
    """Which INTL_FEATURE_COLUMNS would be produced for this frame."""
    cols: list[str] = []
    if "neutral_site" in df.columns:
        cols.append("is_neutral")
    if "country" in df.columns and (
        "home_country" in df.columns or "away_country" in df.columns
    ):
        cols.extend(["is_host_home", "is_host_away"])
    return cols
