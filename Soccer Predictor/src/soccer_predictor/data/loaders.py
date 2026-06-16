"""
Loaders: turn source files (canonical CSVs, football-data.co.uk club CSVs,
martj42 international results, group / fixture / odds configs) into the canonical
match schema.

Every loader's last step is `standardize_matches`, so they differ only in the
column mapping and a few source-specific quirks (date formats, encodings, odds
column preferences). A loader may be pointed at a single CSV file OR a directory
of CSVs (concatenated and re-sorted by date).

The functions here read; they never write or fetch. Network fetching with
caching lives in `sources.py` and simply calls these loaders on the cached file.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from . import schemas
from .aliases import normalize_team_name
from .normalizers import standardize_matches

# --------------------------------------------------------------------------- #
# football-data.co.uk odds column preferences
# --------------------------------------------------------------------------- #
# Preference order for the OPENING / main odds triple. We take the first triple
# fully present in the file. AvgH/D/A (market average) is the most robust; the
# bookmaker-specific triples are fallbacks.
_ODDS_CANDIDATES: list[tuple[str, str, str]] = [
    ("AvgH", "AvgD", "AvgA"),     # market average across books (most robust)
    ("B365H", "B365D", "B365A"),  # Bet365
    ("PSH", "PSD", "PSA"),        # Pinnacle
    ("BWH", "BWD", "BWA"),        # bwin
    ("MaxH", "MaxD", "MaxA"),     # market maximum
]

# Preference order for CLOSING odds, when football-data supplies them ("C" forms).
_CLOSING_ODDS_CANDIDATES: list[tuple[str, str, str]] = [
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("PSCH", "PSCD", "PSCA"),
    ("BWCH", "BWCD", "BWCA"),
    ("MaxCH", "MaxCD", "MaxCA"),
]


def _pick_triple(
    columns, candidates: list[tuple[str, str, str]]
) -> tuple[str, str, str] | None:
    cols = set(columns)
    for triple in candidates:
        if set(triple).issubset(cols):
            return triple
    return None


# --------------------------------------------------------------------------- #
# Generic directory-or-file helper
# --------------------------------------------------------------------------- #
def _iter_csv_paths(path: str) -> list[str]:
    """Return a sorted list of CSV paths for a file or directory argument."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory {path!r}")
        return files
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: {path!r}")
    return [path]


# --------------------------------------------------------------------------- #
# Canonical CSV loader
# --------------------------------------------------------------------------- #
def load_matches(path: str) -> pd.DataFrame:
    """Load a canonical CSV file or a directory of canonical CSVs.

    The input is expected to already use canonical (or close-to-canonical) column
    names; everything is passed through `standardize_matches`, which fills missing
    required columns, coerces dtypes, normalises team names, dedupes, sorts by
    date, and adds the `result` column.
    """
    paths = _iter_csv_paths(path)
    frames = [pd.read_csv(p) for p in paths]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return standardize_matches(df)


# --------------------------------------------------------------------------- #
# football-data.co.uk club loader
# --------------------------------------------------------------------------- #
def _read_football_data_csv(path: str) -> pd.DataFrame:
    """Parse one football-data.co.uk CSV into a (still-raw) canonical-ish frame."""
    raw = pd.read_csv(path, encoding="latin-1")
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"{path!r} is not a football-data.co.uk file: missing {sorted(missing)}"
        )

    out = pd.DataFrame()
    # football-data uses dd/mm/yy or dd/mm/yyyy; try the two known formats
    # explicitly (avoids pandas' slow, warning-prone per-element inference),
    # then fall back to dayfirst inference for any stragglers.
    raw_date = raw["Date"].astype(str).str.strip()
    parsed = pd.to_datetime(raw_date, format="%d/%m/%Y", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed_short = pd.to_datetime(raw_date[mask], format="%d/%m/%y", errors="coerce")
        parsed.loc[mask] = parsed_short
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            raw_date[mask], dayfirst=True, errors="coerce"
        )
    out["date"] = parsed
    out["home_team"] = raw["HomeTeam"]
    out["away_team"] = raw["AwayTeam"]
    out["home_score"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    out["away_score"] = pd.to_numeric(raw["FTAG"], errors="coerce")

    # Competition / season hints from the football-data "Div" column if present.
    if "Div" in raw.columns:
        out["competition"] = raw["Div"].astype("string").fillna("")
    if "Season" in raw.columns:
        out["season"] = raw["Season"].astype("string").fillna("")

    main = _pick_triple(raw.columns, _ODDS_CANDIDATES)
    if main is not None:
        h, d, a = main
        out["home_odds"] = pd.to_numeric(raw[h], errors="coerce")
        out["draw_odds"] = pd.to_numeric(raw[d], errors="coerce")
        out["away_odds"] = pd.to_numeric(raw[a], errors="coerce")

    closing = _pick_triple(raw.columns, _CLOSING_ODDS_CANDIDATES)
    if closing is not None:
        h, d, a = closing
        out["closing_home_odds"] = pd.to_numeric(raw[h], errors="coerce")
        out["closing_draw_odds"] = pd.to_numeric(raw[d], errors="coerce")
        out["closing_away_odds"] = pd.to_numeric(raw[a], errors="coerce")

    return out


def load_football_data(path: str) -> pd.DataFrame:
    """Load football-data.co.uk club CSV(s) into the canonical schema.

    Maps Date/HomeTeam/AwayTeam/FTHG/FTAG and the first available odds triple
    (Avg, then B365, PS, BW, Max) onto canonical columns; closing-style triples
    map to ``closing_*`` when present. ``competition_type`` is ``"club_league"``.
    A directory is concatenated and re-sorted by date.
    """
    paths = _iter_csv_paths(path)
    frames = [_read_football_data_csv(p) for p in paths]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return standardize_matches(df, competition_type="club_league")


# --------------------------------------------------------------------------- #
# martj42 international results loader
# --------------------------------------------------------------------------- #
def load_international_results(path: str) -> pd.DataFrame:
    """Load martj42-schema international results into the canonical schema.

    Source columns: ``date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral``. ``tournament`` maps to ``competition``,
    ``neutral`` to ``neutral_site``; ``competition_type`` is derived per-row from
    the competition name via ``sources.classify_competition_type``.
    """
    # Local import breaks the loaders <-> sources import cycle.
    from .sources import classify_competition_type

    paths = _iter_csv_paths(path)
    frames = [pd.read_csv(p) for p in paths]
    raw = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    mapping = {
        "tournament": "competition",
        "neutral": "neutral_site",
    }
    df = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns})

    # Derive competition_type per row from the (possibly newly-named) competition.
    if "competition" in df.columns:
        df = df.copy()
        df["competition_type"] = df["competition"].map(classify_competition_type)

    return standardize_matches(df)


# --------------------------------------------------------------------------- #
# Group config loader
# --------------------------------------------------------------------------- #
def load_groups(path: str) -> dict[str, list[str]]:
    """Load a ``group,team`` CSV into ``{GROUP: [team, ...]}``.

    Group labels are upper-cased; team names are normalised. Groups with fewer
    than two teams trigger a warning (not an error) -- group sizes vary across
    competitions (Euros, Copa, custom formats), so we do not hard-require four.
    """
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    if "group" not in cols or "team" not in cols:
        raise ValueError(
            f"{path!r} must have 'group' and 'team' columns; got {list(raw.columns)}"
        )

    groups: dict[str, list[str]] = {}
    for group_val, team_val in zip(raw[cols["group"]], raw[cols["team"]]):
        group = str(group_val).strip().upper()
        team = normalize_team_name(team_val)
        if not group or not team:
            continue
        groups.setdefault(group, [])
        if team not in groups[group]:
            groups[group].append(team)

    for group, teams in groups.items():
        if len(teams) < 2:
            import warnings

            warnings.warn(
                f"Group {group!r} has only {len(teams)} team(s); "
                "a group stage needs at least two.",
                stacklevel=2,
            )
    return groups


# --------------------------------------------------------------------------- #
# Fixtures loader (scores may be blank for upcoming matches)
# --------------------------------------------------------------------------- #
def load_fixtures(path: str) -> pd.DataFrame:
    """Load a fixtures CSV into the canonical schema.

    Scores may be blank/NaN for not-yet-played fixtures; those rows are kept and
    ``is_completed`` reflects whether both scores are present. At minimum the file
    needs ``home_team`` and ``away_team``; ``match_id, date, group, neutral,
    home_score, away_score`` are optional. A missing ``date`` is back-filled with
    a placeholder so the canonical date-sort/validation still works.
    """
    raw = pd.read_csv(path)
    mapping = {
        "tournament": "competition",
        "neutral": "neutral_site",
        "home": "home_team",
        "away": "away_team",
    }
    df = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns})

    if "home_team" not in df.columns or "away_team" not in df.columns:
        raise ValueError(
            f"{path!r} must have at least home_team and away_team columns; "
            f"got {list(raw.columns)}"
        )

    df = df.copy()
    # validate_matches drops rows with a missing date; fixtures legitimately may
    # have no date yet, so back-fill a placeholder rather than silently losing them.
    if "date" not in df.columns:
        df["date"] = pd.Timestamp("2100-01-01")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].fillna(pd.Timestamp("2100-01-01"))

    # Do not require scores: keep blank-score fixtures.
    out = standardize_matches(df, require_scores=False)
    return out


# --------------------------------------------------------------------------- #
# Odds-only / upcoming-fixtures-with-odds loader
# --------------------------------------------------------------------------- #
def load_odds(path: str) -> pd.DataFrame:
    """Load an upcoming-fixtures-with-odds CSV into the canonical schema.

    Tolerant of common odds column spellings (``odds_h/odds_d/odds_a``,
    ``home_odds/...``, ``h/d/a``) and closing variants, mapping them onto the
    canonical ``home_odds/draw_odds/away_odds`` (+ ``closing_*``). Scores may be
    absent (these are upcoming fixtures), so completion is inferred, not required.
    """
    raw = pd.read_csv(path)
    mapping = {
        # team / fixture aliases
        "home": "home_team",
        "away": "away_team",
        "tournament": "competition",
        "neutral": "neutral_site",
        # current 1X2 odds aliases
        "odds_h": "home_odds",
        "odds_d": "draw_odds",
        "odds_a": "away_odds",
        "home_win_odds": "home_odds",
        "draw_win_odds": "draw_odds",
        "away_win_odds": "away_odds",
        "h": "home_odds",
        "d": "draw_odds",
        "a": "away_odds",
        # closing odds aliases
        "closing_h": "closing_home_odds",
        "closing_d": "closing_draw_odds",
        "closing_a": "closing_away_odds",
        "close_home_odds": "closing_home_odds",
        "close_draw_odds": "closing_draw_odds",
        "close_away_odds": "closing_away_odds",
    }
    df = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns})

    if "home_team" not in df.columns or "away_team" not in df.columns:
        raise ValueError(
            f"{path!r} must have home_team and away_team columns; "
            f"got {list(raw.columns)}"
        )

    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.Timestamp("2100-01-01")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].fillna(pd.Timestamp("2100-01-01"))

    return standardize_matches(df, require_scores=False)


__all__ = [
    "load_matches",
    "load_football_data",
    "load_international_results",
    "load_groups",
    "load_fixtures",
    "load_odds",
]
