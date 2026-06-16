"""
Data layer tests: schema validation, loaders, dedupe, and the `result` column.

Everything runs off the synthetic generators and tiny in-memory CSVs written to
a temp dir, so the suite is fast and needs no network. These tests pin the
canonical column contract (home_score/away_score, result in {H,D,A},
neutral_site bool) that the rest of the package relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soccer_predictor.data import schemas, synthetic
from soccer_predictor.data.loaders import (
    load_fixtures,
    load_football_data,
    load_groups,
    load_international_results,
    load_matches,
)


# --------------------------------------------------------------------------- #
# Synthetic frames satisfy the canonical schema
# --------------------------------------------------------------------------- #
def test_generate_league_is_canonical():
    df = synthetic.generate_league(n_teams=8, n_seasons=2, seed=1)
    # Required columns all present.
    for col in schemas.REQUIRED_COLUMNS:
        assert col in df.columns, f"missing required column {col!r}"
    # Canonical (not legacy) score/odds names.
    assert {"home_score", "away_score"}.issubset(df.columns)
    assert {"home_odds", "draw_odds", "away_odds"}.issubset(df.columns)
    assert "home_goals" not in df.columns and "odds_h" not in df.columns
    # competition_type / neutral_site contract.
    assert (df["competition_type"] == "club_league").all()
    assert df["neutral_site"].dtype == bool
    assert not df["neutral_site"].any()  # club league is never neutral
    # Passes validation unchanged (idempotent canonical frame).
    validated = schemas.validate_matches(df)
    assert len(validated) == len(df)


def test_generate_international_history_canonical_and_mixed():
    df = synthetic.generate_international_history(n_teams=24, n_matches=300, seed=2)
    assert {"home_score", "away_score"}.issubset(df.columns)
    assert df["neutral_site"].dtype == bool
    # Mostly-neutral international football.
    assert df["neutral_site"].mean() > 0.5
    # Carries the qualifier label the contract calls out, plus friendlies.
    comps = set(df["competition"].unique())
    assert "FIFA World Cup qualification" in comps
    assert "Friendly" in comps
    # competition_type derived per row -> includes qualifier + friendly types.
    types = set(df["competition_type"].unique())
    assert types.issubset(set(schemas.COMPETITION_TYPES))
    assert "qualifier" in types


# --------------------------------------------------------------------------- #
# result column correctness
# --------------------------------------------------------------------------- #
def test_add_result_column_matches_scores():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "home_team": ["A", "B", "C"],
            "away_team": ["X", "Y", "Z"],
            "home_score": [2, 1, 0],
            "away_score": [0, 1, 3],
        }
    )
    out = schemas.add_result_column(df)
    assert list(out["result"]) == ["H", "D", "A"]


def test_result_blank_for_missing_scores():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "home_team": ["A", "B"],
            "away_team": ["X", "Y"],
            "home_score": [2, np.nan],
            "away_score": [0, np.nan],
        }
    )
    out = schemas.add_result_column(df)
    assert out.loc[0, "result"] == "H"
    assert out.loc[1, "result"] == ""  # not-yet-played -> blank label


def test_synthetic_result_is_consistent_with_scores():
    df = synthetic.generate_league(n_teams=6, n_seasons=2, seed=4)
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    expected = np.where(hs > as_, "H", np.where(hs == as_, "D", "A"))
    assert (df["result"].to_numpy() == expected).all()


# --------------------------------------------------------------------------- #
# validate_matches behaviour
# --------------------------------------------------------------------------- #
def test_validate_raises_without_minimum_columns():
    bad = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(schemas.SchemaError):
        schemas.validate_matches(bad)


def test_validate_drops_rows_missing_date_or_team():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", None, "2020-01-03"],
            "home_team": ["A", "B", ""],
            "away_team": ["X", "Y", "Z"],
            "home_score": [1, 0, 2],
            "away_score": [0, 0, 1],
        }
    )
    out = schemas.validate_matches(df)
    # Row 1 has no date, row 2 has an empty home_team -> only row 0 survives.
    assert len(out) == 1
    assert out.iloc[0]["home_team"] == "A"


def test_validate_sorts_by_date():
    df = pd.DataFrame(
        {
            "date": ["2020-03-01", "2020-01-01", "2020-02-01"],
            "home_team": ["A", "B", "C"],
            "away_team": ["X", "Y", "Z"],
            "home_score": [1, 1, 1],
            "away_score": [0, 0, 0],
        }
    )
    out = schemas.validate_matches(df)
    assert list(out["date"]) == sorted(out["date"])


# --------------------------------------------------------------------------- #
# Presence helpers
# --------------------------------------------------------------------------- #
def test_rows_with_complete_odds_filters_partial_and_bad():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["W", "X", "Y", "Z"],
            "home_odds": [2.0, np.nan, 1.0, 3.0],
            "draw_odds": [3.0, 3.0, 3.0, 3.5],
            "away_odds": [3.5, 3.5, 3.5, 2.2],
        }
    )
    keep = schemas.rows_with_complete_odds(df)
    # Row 1 (NaN home) and row 2 (home_odds==1.0, not > 1) are excluded.
    assert set(keep["home_team"]) == {"A", "D"}


# --------------------------------------------------------------------------- #
# Loaders (write tiny CSVs to a temp dir)
# --------------------------------------------------------------------------- #
def test_load_matches_roundtrips_synthetic(tmp_path):
    df = synthetic.generate_league(n_teams=6, n_seasons=1, seed=5)
    path = tmp_path / "matches.csv"
    df.to_csv(path, index=False)
    loaded = load_matches(str(path))
    assert {"home_score", "away_score", "result"}.issubset(loaded.columns)
    assert len(loaded) == len(df)
    assert set(loaded["result"].unique()).issubset({"H", "D", "A", ""})


def test_load_football_data_maps_columns(tmp_path):
    raw = pd.DataFrame(
        {
            "Div": ["E0", "E0"],
            "Date": ["12/08/23", "19/08/23"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Arsenal"],
            "FTHG": [2, 1],
            "FTAG": [1, 1],
            "AvgH": [1.8, 2.5],
            "AvgD": [3.5, 3.3],
            "AvgA": [4.2, 2.8],
        }
    )
    path = tmp_path / "E0.csv"
    raw.to_csv(path, index=False, encoding="latin-1")
    out = load_football_data(str(path))
    assert (out["competition_type"] == "club_league").all()
    assert {"home_odds", "draw_odds", "away_odds"}.issubset(out.columns)
    # dayfirst parsing: 12/08/23 -> 2023-08-12.
    first = out.sort_values("date").iloc[0]
    assert first["date"] == pd.Timestamp("2023-08-12")
    assert list(out.sort_values("date")["result"]) == ["H", "D"]


def test_load_international_results_maps_schema(tmp_path):
    raw = pd.DataFrame(
        {
            "date": ["2022-11-21", "2022-11-22"],
            "home_team": ["Qatar", "England"],
            "away_team": ["Ecuador", "Iran"],
            "home_score": [0, 6],
            "away_score": [2, 2],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "city": ["Al Khor", "Doha"],
            "country": ["Qatar", "Qatar"],
            "neutral": [False, True],
        }
    )
    path = tmp_path / "results.csv"
    raw.to_csv(path, index=False)
    out = load_international_results(str(path))
    assert "competition" in out.columns
    assert (out["competition"] == "FIFA World Cup").all()
    assert out["neutral_site"].dtype == bool
    # World Cup classifies as a tournament.
    assert (out["competition_type"] == "tournament").all()
    assert list(out.sort_values("date")["result"]) == ["A", "H"]


def test_load_groups_normalizes_and_uppercases(tmp_path):
    raw = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "team": ["USA", "Mexico", "Brazil", "Argentina"],
        }
    )
    path = tmp_path / "groups.csv"
    raw.to_csv(path, index=False)
    groups = load_groups(str(path))
    assert set(groups.keys()) == {"A", "B"}
    # USA alias -> "United States".
    assert "United States" in groups["A"]
    assert len(groups["A"]) == 2 and len(groups["B"]) == 2


def test_load_fixtures_keeps_blank_scores(tmp_path):
    raw = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_team": ["A", "B"],
            "away_team": ["X", "Y"],
            "home_score": [2, np.nan],
            "away_score": [0, np.nan],
        }
    )
    path = tmp_path / "fixtures.csv"
    raw.to_csv(path, index=False)
    out = load_fixtures(str(path))
    assert len(out) == 2  # the un-played fixture is retained
    completed = out["is_completed"].to_numpy()
    assert completed.sum() == 1  # exactly one has both scores


# --------------------------------------------------------------------------- #
# Dedupe (via standardize_matches, exercised through load_matches)
# --------------------------------------------------------------------------- #
def test_loader_dedupes_repeated_fixtures(tmp_path):
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "home_team": ["A", "A", "C"],  # first two are the same fixture
            "away_team": ["B", "B", "D"],
            "home_score": [1, 1, 2],
            "away_score": [0, 0, 2],
        }
    )
    path = tmp_path / "dupes.csv"
    raw.to_csv(path, index=False)
    out = load_matches(str(path))
    assert len(out) == 2  # one of the duplicate (date, A, B) rows dropped


# --------------------------------------------------------------------------- #
# Alias normalization (regression: '&' must resolve like 'and')
# --------------------------------------------------------------------------- #
def test_ampersand_team_name_normalizes():
    from soccer_predictor.data.aliases import normalize_team_name

    assert normalize_team_name("Bosnia & Herzegovina") == "Bosnia-Herzegovina"
    assert normalize_team_name("Bosnia and Herzegovina") == "Bosnia-Herzegovina"
    # idempotent on the canonical form
    assert normalize_team_name("Bosnia-Herzegovina") == "Bosnia-Herzegovina"
