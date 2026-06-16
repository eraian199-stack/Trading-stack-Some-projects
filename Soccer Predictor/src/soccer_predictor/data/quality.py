"""
Data-quality diagnostics.

Before training or simulating, it is worth knowing what you actually have: how
many matches, over what date range, how many teams, how complete the odds / xG
coverage is, whether the same fixture appears twice, and how many rows are still
missing scores. `quality_report` rolls these up into a plain dict suitable for
logging, the CLI's ``update-data`` summary, or the Streamlit data panel.

All functions are read-only and tolerate partially-populated frames (a frame
with no odds columns simply reports 0% coverage rather than raising).
"""

from __future__ import annotations

import pandas as pd

from . import schemas


# --------------------------------------------------------------------------- #
# Coverage helpers
# --------------------------------------------------------------------------- #
def odds_coverage(df: pd.DataFrame) -> float:
    """Fraction of rows with all three 1X2 odds present and > 1.0 (0..1)."""
    if len(df) == 0 or not schemas.has_columns(df, schemas.ODDS_COLUMNS):
        return 0.0
    odds = df[schemas.ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ok = odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
    return float(ok.mean())


def xg_coverage(df: pd.DataFrame) -> float:
    """Fraction of rows with both xG values present (0..1)."""
    if len(df) == 0 or not schemas.has_columns(df, schemas.XG_COLUMNS):
        return 0.0
    xg = df[schemas.XG_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return float(xg.notna().all(axis=1).mean())


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #
def find_duplicate_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that share a (date, home_team, away_team) key with another row.

    The canonical pipeline dedupes on this key, so a non-empty result here flags
    a source that double-lists matches (or a merge gone wrong). All members of
    each duplicate group are returned, sorted so they sit together.
    """
    keys = [c for c in ("date", "home_team", "away_team") if c in df.columns]
    if not keys or len(df) == 0:
        return df.iloc[0:0].copy()
    dup_mask = df.duplicated(subset=keys, keep=False)
    return df[dup_mask.to_numpy()].sort_values(keys, kind="mergesort").copy()


# --------------------------------------------------------------------------- #
# Roll-up report
# --------------------------------------------------------------------------- #
def quality_report(df: pd.DataFrame) -> dict:
    """Summarise a canonical match frame as a plain dict.

    Keys: ``n_matches, date_min, date_max, n_teams, pct_with_odds, pct_with_xg,
    n_duplicate_fixtures, n_missing_scores, competition_breakdown``. Robust to
    missing optional columns (coverage falls back to 0.0).
    """
    n_matches = int(len(df))

    if "date" in df.columns and n_matches:
        dates = pd.to_datetime(df["date"], errors="coerce")
        date_min = dates.min()
        date_max = dates.max()
    else:
        date_min = date_max = None

    teams: set[str] = set()
    for column in ("home_team", "away_team"):
        if column in df.columns:
            teams.update(t for t in df[column].astype(str) if t)

    if {"home_score", "away_score"}.issubset(df.columns) and n_matches:
        home = pd.to_numeric(df["home_score"], errors="coerce")
        away = pd.to_numeric(df["away_score"], errors="coerce")
        n_missing_scores = int((home.isna() | away.isna()).sum())
    else:
        n_missing_scores = n_matches

    if "competition" in df.columns and n_matches:
        breakdown = {
            str(k): int(v)
            for k, v in df["competition"].astype(str).value_counts().items()
        }
    else:
        breakdown = {}

    return {
        "n_matches": n_matches,
        "date_min": date_min,
        "date_max": date_max,
        "n_teams": len(teams),
        "pct_with_odds": odds_coverage(df),
        "pct_with_xg": xg_coverage(df),
        "n_duplicate_fixtures": int(len(find_duplicate_fixtures(df))),
        "n_missing_scores": n_missing_scores,
        "competition_breakdown": breakdown,
    }


__all__ = [
    "quality_report",
    "find_duplicate_fixtures",
    "odds_coverage",
    "xg_coverage",
]
