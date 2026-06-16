"""
Normalisation helpers that turn a raw, source-specific frame into the canonical
match schema (see schemas.py).

`standardize_matches` is the one function every loader calls last: it renames
source columns onto canonical names, normalises team names, coerces dtypes,
dedupes, sorts, and marks completion. Loaders differ only in the column mapping
they pass in.
"""

from __future__ import annotations

import pandas as pd

from . import schemas
from .aliases import normalize_team_name


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns from source names -> canonical names.

    `mapping` is {source_column: canonical_column}. Source columns absent from
    the frame are skipped silently (sources vary in which fields they carry).
    """
    present = {src: dst for src, dst in mapping.items() if src in df.columns}
    return df.rename(columns=present)


def normalize_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ("home_team", "away_team"):
        if column in df.columns:
            df[column] = df[column].map(normalize_team_name)
    return df


def dedupe_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate fixtures keyed on (date, home_team, away_team).

    Keeps the first occurrence. Many sources double-list matches; the canonical
    table holds one row per fixture.
    """
    keys = [c for c in ("date", "home_team", "away_team") if c in df.columns]
    if not keys:
        return df
    return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def standardize_matches(
    df: pd.DataFrame,
    *,
    mapping: dict[str, str] | None = None,
    competition: str | None = None,
    competition_type: str | None = None,
    require_scores: bool = False,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Full pipeline: map columns -> normalise names -> validate -> dedupe.

    Returns a canonical, date-sorted match frame with a `result` column. Any
    optional columns the source provided (odds, xG, ranks ...) are preserved.
    """
    if mapping:
        df = apply_column_mapping(df, mapping)
    df = normalize_team_columns(df)
    if competition is not None and (
        "competition" not in df.columns or df["competition"].eq("").all()
    ):
        df = df.copy()
        df["competition"] = competition
    if competition_type is not None and (
        "competition_type" not in df.columns
        or df["competition_type"].isna().all()
    ):
        df = df.copy()
        df["competition_type"] = competition_type
    df = schemas.validate_matches(df, require_scores=require_scores)
    if dedupe:
        df = dedupe_fixtures(df)
    df = schemas.add_result_column(df)
    return df.reset_index(drop=True)
