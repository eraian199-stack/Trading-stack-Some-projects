from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RESULT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

FIXTURE_COLUMNS = [
    "match_id",
    "date",
    "group",
    "home_team",
    "away_team",
    "neutral",
    "home_score",
    "away_score",
]


def normalize_team_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    aliases = {
        "USA": "United States",
        "USMNT": "United States",
        "Türkiye": "Turkey",
        "Turkiye": "Turkey",
        "Czech Republic": "Czechia",
        "Korea Republic": "South Korea",
        "IR Iran": "Iran",
        "DR Congo": "Congo DR",
        "Democratic Republic of the Congo": "Congo DR",
    }
    return aliases.get(text, text)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "neutral"}


def _read_csv(path_or_url: str | Path) -> pd.DataFrame:
    return pd.read_csv(path_or_url)


def standardize_results(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    missing = {"date", "home_team", "away_team", "home_score", "away_score"} - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"Results file is missing required columns: {sorted(missing)}")

    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            if column == "neutral":
                frame[column] = True
            else:
                frame[column] = ""

    frame = frame[RESULT_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame["neutral"] = frame["neutral"].map(_coerce_bool)
    frame = frame.dropna(subset=["date", "home_score", "away_score"])
    frame = frame[(frame["home_team"] != "") & (frame["away_team"] != "")]
    frame["home_score"] = frame["home_score"].astype(int)
    frame["away_score"] = frame["away_score"].astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def load_results(path_or_url: str | Path) -> pd.DataFrame:
    """Load historical international match results.

    Expected schema is compatible with common public international-results files:
    date, home_team, away_team, home_score, away_score, tournament, city,
    country, neutral.
    """

    return standardize_results(_read_csv(path_or_url))


def load_groups(path_or_url: str | Path) -> dict[str, list[str]]:
    frame = _read_csv(path_or_url)
    missing = {"group", "team"} - set(frame.columns)
    if missing:
        raise ValueError(f"Groups file is missing required columns: {sorted(missing)}")
    frame = frame[["group", "team"]].copy()
    frame["group"] = frame["group"].astype(str).str.strip().str.upper()
    frame["team"] = frame["team"].map(normalize_team_name)
    frame = frame[(frame["group"] != "") & (frame["team"] != "")]
    groups: dict[str, list[str]] = {}
    for group, rows in frame.groupby("group", sort=True):
        teams = rows["team"].tolist()
        if len(teams) != 4:
            raise ValueError(f"Group {group} has {len(teams)} teams; expected 4")
        groups[group] = teams
    return groups


def load_fixtures(path_or_url: str | Path) -> pd.DataFrame:
    frame = _read_csv(path_or_url)
    missing = {"home_team", "away_team"} - set(frame.columns)
    if missing:
        raise ValueError(f"Fixtures file is missing required columns: {sorted(missing)}")

    for column in FIXTURE_COLUMNS:
        if column not in frame.columns:
            if column == "neutral":
                frame[column] = True
            elif column in {"home_score", "away_score"}:
                frame[column] = pd.NA
            else:
                frame[column] = ""

    frame = frame[FIXTURE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["group"] = frame["group"].astype(str).str.strip().str.upper()
    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    frame["neutral"] = frame["neutral"].map(_coerce_bool)
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    return frame.sort_values(["date", "match_id"], na_position="last").reset_index(
        drop=True
    )


def groups_to_frame(groups: dict[str, Iterable[str]]) -> pd.DataFrame:
    rows = []
    for group, teams in sorted(groups.items()):
        for team in teams:
            rows.append({"group": group, "team": normalize_team_name(team)})
    return pd.DataFrame(rows)
