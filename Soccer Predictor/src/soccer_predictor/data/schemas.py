"""
Canonical data schemas.

There is ONE canonical match table used everywhere in this project -- club
leagues, club cups, international friendlies, qualifiers, and tournaments all
flow through the same columns. Loaders normalise their source format into this
schema; features, models, evaluation, and simulation all read it. Optional
columns degrade gracefully: anything missing is simply absent, and downstream
code checks for presence rather than assuming it.

This module is the single source of truth for column names and is imported by
nearly everything else, so it deliberately has no heavy dependencies.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Canonical columns
# --------------------------------------------------------------------------- #
# Required identity / result columns. Every canonical match frame has these.
REQUIRED_COLUMNS: list[str] = [
    "match_id",
    "date",
    "competition",
    "competition_type",
    "season",
    "stage",
    "round",
    "group",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral_site",
    "venue",
    "city",
    "country",
    "home_country",
    "away_country",
    "is_completed",
]

# Optional enrichment columns. Present only when a source supplies them.
OPTIONAL_COLUMNS: list[str] = [
    "home_xg",
    "away_xg",
    "home_odds",
    "draw_odds",
    "away_odds",
    "closing_home_odds",
    "closing_draw_odds",
    "closing_away_odds",
    "home_rest_days",
    "away_rest_days",
    "home_travel_km",
    "away_travel_km",
    "home_fifa_rank",
    "away_fifa_rank",
    "home_elo",
    "away_elo",
    "home_squad_value",
    "away_squad_value",
    "home_injury_score",
    "away_injury_score",
]

ALL_COLUMNS: list[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Bare minimum a loader must be able to produce to be usable at all.
MINIMUM_COLUMNS: list[str] = ["date", "home_team", "away_team"]

# Valid values for the competition_type discriminator.
COMPETITION_TYPES: tuple[str, ...] = (
    "club_league",
    "club_cup",
    "international_friendly",
    "qualifier",
    "tournament",
)

# Odds column triples (current and closing).
ODDS_COLUMNS: list[str] = ["home_odds", "draw_odds", "away_odds"]
CLOSING_ODDS_COLUMNS: list[str] = [
    "closing_home_odds",
    "closing_draw_odds",
    "closing_away_odds",
]
XG_COLUMNS: list[str] = ["home_xg", "away_xg"]

# 1X2 result classes, ordered Home < Draw < Away (RPS relies on this order).
RESULT_CLASSES: list[str] = ["H", "D", "A"]


# --------------------------------------------------------------------------- #
# Defaults for required columns when a source omits them
# --------------------------------------------------------------------------- #
def _default_for(column: str) -> object:
    if column == "neutral_site":
        return False
    if column == "is_completed":
        return False
    if column == "competition_type":
        return "club_league"
    if column in ("home_score", "away_score"):
        return pd.NA
    return ""


# --------------------------------------------------------------------------- #
# Construction / coercion
# --------------------------------------------------------------------------- #
def empty_match_frame() -> pd.DataFrame:
    """An empty frame with exactly the canonical columns in canonical order."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS})


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing REQUIRED columns with sensible defaults.

    Optional columns are left untouched (present if the source had them). The
    returned frame is ordered: required columns first (canonical order), then
    any optional/extra columns that were present, in their original order.
    """
    df = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = _default_for(column)
    ordered = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df[ordered]


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce canonical columns to their expected dtypes, in place on a copy.

    - date -> datetime64
    - scores / numeric optionals -> numeric (nullable where appropriate)
    - neutral_site / is_completed -> bool
    - team / text columns -> stripped strings
    Bad values become NaT / NaN rather than raising.
    """
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in ("home_score", "away_score"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_optionals = [
        "home_xg", "away_xg",
        "home_odds", "draw_odds", "away_odds",
        "closing_home_odds", "closing_draw_odds", "closing_away_odds",
        "home_rest_days", "away_rest_days",
        "home_travel_km", "away_travel_km",
        "home_fifa_rank", "away_fifa_rank",
        "home_elo", "away_elo",
        "home_squad_value", "away_squad_value",
        "home_injury_score", "away_injury_score",
    ]
    for column in numeric_optionals:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("neutral_site", "is_completed"):
        if column in df.columns:
            df[column] = df[column].map(coerce_bool).astype(bool)

    text_columns = [
        "match_id", "competition", "competition_type", "season", "stage",
        "round", "group", "home_team", "away_team", "venue", "city",
        "country", "home_country", "away_country",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].map(_clean_text)

    return df


def coerce_bool(value: object) -> bool:
    """Forgiving truthiness for neutral_site / is_completed style columns."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "neutral"}


def _clean_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


# --------------------------------------------------------------------------- #
# Derived columns
# --------------------------------------------------------------------------- #
def add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `result` column in {H, D, A} for completed matches.

    Rows without both scores get result == "" (treated as not-yet-played).
    This is the label every 1X2 model / metric consumes.
    """
    df = df.copy()
    home = pd.to_numeric(df.get("home_score"), errors="coerce")
    away = pd.to_numeric(df.get("away_score"), errors="coerce")
    result = np.full(len(df), "", dtype=object)
    valid = home.notna().to_numpy() & away.notna().to_numpy()
    hv = home.to_numpy()
    av = away.to_numpy()
    result[valid & (hv > av)] = "H"
    result[valid & (hv == av)] = "D"
    result[valid & (hv < av)] = "A"
    df["result"] = result
    return df


def mark_completed(df: pd.DataFrame) -> pd.DataFrame:
    """Set is_completed based on whether both scores are present."""
    df = df.copy()
    home = pd.to_numeric(df.get("home_score"), errors="coerce")
    away = pd.to_numeric(df.get("away_score"), errors="coerce")
    df["is_completed"] = (home.notna() & away.notna()).to_numpy()
    return df


def completed_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that have both scores -- the only rows usable for training/eval."""
    home = pd.to_numeric(df.get("home_score"), errors="coerce")
    away = pd.to_numeric(df.get("away_score"), errors="coerce")
    mask = home.notna() & away.notna()
    return df[mask.to_numpy()].copy()


# --------------------------------------------------------------------------- #
# Presence helpers (so downstream code degrades gracefully)
# --------------------------------------------------------------------------- #
def has_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return set(columns).issubset(df.columns)


def has_usable_odds(df: pd.DataFrame) -> bool:
    """True if at least one row has all three 1X2 odds present and > 1."""
    if not has_columns(df, ODDS_COLUMNS):
        return False
    odds = df[ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ok = odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
    return bool(ok.any())


def rows_with_complete_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of rows where all three 1X2 odds are present and > 1.

    This is the correct filter for the market baseline / betting: partial odds
    produce NaNs downstream, so they must be excluded rather than tolerated.
    """
    if not has_columns(df, ODDS_COLUMNS):
        return df.iloc[0:0].copy()
    odds = df[ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ok = odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
    return df[ok.to_numpy()].copy()


def has_usable_xg(df: pd.DataFrame) -> bool:
    if not has_columns(df, XG_COLUMNS):
        return False
    return bool(df[XG_COLUMNS].notna().all(axis=1).any())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class SchemaError(ValueError):
    """Raised when a frame cannot be coerced into the canonical schema."""


def validate_matches(df: pd.DataFrame, *, require_scores: bool = False) -> pd.DataFrame:
    """Validate and return a canonical match frame.

    - Ensures all REQUIRED columns exist (filling defaults).
    - Coerces dtypes.
    - Drops rows missing date / team names.
    - If require_scores, also drops rows without both scores.
    Raises SchemaError if the minimum identity columns are entirely absent.
    """
    missing_minimum = [c for c in MINIMUM_COLUMNS if c not in df.columns]
    if missing_minimum:
        raise SchemaError(
            f"Frame is missing minimum columns {missing_minimum}; "
            f"got columns {sorted(df.columns)}"
        )
    df = ensure_columns(df)
    df = coerce_types(df)
    df = df.dropna(subset=["date"])
    df = df[(df["home_team"] != "") & (df["away_team"] != "")]
    if require_scores:
        df = completed_matches(df)
    df = mark_completed(df)
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)
