"""
Offline tests for the live-data layer (odds_api + world_cup).

Network is mocked so these run anywhere: the Odds API JSON shape is fed through a
monkeypatched transport, and the World Cup group reconstruction is exercised on a
hand-built fixture frame.
"""

from __future__ import annotations

import pandas as pd

from soccer_predictor.data import odds_api, schemas, world_cup
from soccer_predictor.features.market import implied_probabilities


# --------------------------------------------------------------------------- #
# odds_api parsing
# --------------------------------------------------------------------------- #
def _fake_odds_payload():
    return [
        {
            "home_team": "Brazil",
            "away_team": "France",
            "commence_time": "2026-06-20T18:00:00Z",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Brazil", "price": 2.5},
                                {"name": "France", "price": 2.9},
                                {"name": "Draw", "price": 3.2},
                            ],
                        }
                    ]
                },
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Brazil", "price": 2.6},
                                {"name": "France", "price": 3.0},
                                {"name": "Draw", "price": 3.1},
                            ],
                        }
                    ]
                },
            ],
        }
    ]


def test_fetch_match_odds_parses_canonical_frame(monkeypatch):
    monkeypatch.setattr(odds_api, "_api_get", lambda *a, **k: _fake_odds_payload())
    df = odds_api.fetch_match_odds(aggregate="avg")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_team"] == "Brazil" and row["away_team"] == "France"
    # avg of (2.5, 2.6) = 2.55 ; draw avg of (3.2, 3.1) = 3.15
    assert abs(row["home_odds"] - 2.55) < 1e-6
    assert abs(row["draw_odds"] - 3.15) < 1e-6
    # de-vig must give a valid distribution
    p = implied_probabilities(row["home_odds"], row["draw_odds"], row["away_odds"])
    assert abs(float(p.sum()) - 1.0) < 1e-9


def test_fetch_match_odds_best_takes_max_price(monkeypatch):
    monkeypatch.setattr(odds_api, "_api_get", lambda *a, **k: _fake_odds_payload())
    df = odds_api.fetch_match_odds(aggregate="best")
    row = df.iloc[0]
    assert abs(row["home_odds"] - 2.6) < 1e-6  # max of (2.5, 2.6)
    assert abs(row["away_odds"] - 3.0) < 1e-6  # max of (2.9, 3.0)


def test_fetch_scores_parses_completed_only(monkeypatch):
    payload = [
        {
            "home_team": "Spain",
            "away_team": "Japan",
            "commence_time": "2026-06-15T16:00:00Z",
            "completed": True,
            "scores": [
                {"name": "Spain", "score": "2"},
                {"name": "Japan", "score": "1"},
            ],
        },
        {  # not completed -> excluded
            "home_team": "Italy",
            "away_team": "Ghana",
            "commence_time": "2026-06-16T16:00:00Z",
            "completed": False,
            "scores": None,
        },
    ]
    monkeypatch.setattr(odds_api, "_api_get", lambda *a, **k: payload)
    df = odds_api.fetch_scores()
    assert len(df) == 1
    assert df.iloc[0]["home_score"] == 2 and df.iloc[0]["away_score"] == 1
    assert bool(df.iloc[0]["is_completed"]) is True


def test_load_odds_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key-123")
    assert odds_api.load_odds_api_key() == "test-key-123"


# --------------------------------------------------------------------------- #
# world_cup group reconstruction
# --------------------------------------------------------------------------- #
def _two_group_history() -> pd.DataFrame:
    """Two WC groups (A: t1..t4, B: t5..t8), full round-robin within each."""
    from itertools import combinations

    rows = []
    groups = {"A": ["t1", "t2", "t3", "t4"], "B": ["t5", "t6", "t7", "t8"]}
    day = pd.Timestamp("2026-06-11")
    i = 0
    for teams in groups.values():
        for h, a in combinations(teams, 2):
            rows.append(
                {
                    "date": day + pd.Timedelta(days=i),
                    "home_team": h,
                    "away_team": a,
                    "home_score": 1,
                    "away_score": 0,
                    "competition": "FIFA World Cup",
                    "neutral": True,
                }
            )
            i += 1
    # plus an unrelated friendly that must be ignored
    rows.append(
        {
            "date": pd.Timestamp("2026-05-01"),
            "home_team": "t1",
            "away_team": "t8",
            "home_score": 0,
            "away_score": 0,
            "competition": "Friendly",
            "neutral": False,
        }
    )
    from soccer_predictor.data.normalizers import standardize_matches

    return standardize_matches(pd.DataFrame(rows))


def _wc_like_history(n_groups: int = 12) -> pd.DataFrame:
    """A full 48-team WC group stage (12 groups of 4) plus a stray friendly."""
    from itertools import combinations

    from soccer_predictor.data.normalizers import standardize_matches

    rows = []
    day = pd.Timestamp("2026-06-11")
    i = 0
    for g in range(n_groups):
        teams = [f"g{g}t{t}" for t in range(4)]
        for h, a in combinations(teams, 2):
            rows.append(
                {
                    "date": day + pd.Timedelta(days=i),
                    "home_team": h,
                    "away_team": a,
                    "home_score": 1,
                    "away_score": 0,
                    "competition": "FIFA World Cup",
                    "neutral": True,
                }
            )
            i += 1
    rows.append(
        {
            "date": pd.Timestamp("2026-05-01"),
            "home_team": "g0t0",
            "away_team": "g11t3",
            "home_score": 0,
            "away_score": 0,
            "competition": "Friendly",
            "neutral": False,
        }
    )
    return standardize_matches(pd.DataFrame(rows))


def test_reconstruct_groups_finds_disjoint_components():
    hist = _wc_like_history(12)
    comps = world_cup.reconstruct_groups(hist)
    assert len(comps) == 12
    assert sorted(len(c) for c in comps) == [4] * 12
    # the cross-group friendly must not have merged any components
    assert frozenset(["g0t0", "g0t1", "g0t2", "g0t3"]) in comps


def test_reconstruct_groups_raises_on_malformed_fixtures():
    """A contaminating cross-group match in the first 72 must fail loud."""
    import pytest

    hist = _two_group_history()  # only 2 groups -> not 12x4
    with pytest.raises(RuntimeError, match="12 groups of 4"):
        world_cup.reconstruct_groups(hist)


def test_build_fixtures_labels_groups_and_locks_scores(tmp_path):
    hist = _two_group_history()
    groups = {"A": ["t1", "t2", "t3", "t4"], "B": ["t5", "t6", "t7", "t8"]}
    out = tmp_path / "fix.csv"
    fixtures = world_cup.build_fixtures(hist, groups, out=out)
    assert len(fixtures) == 12  # two K4s = 12 group games
    assert set(fixtures["group"]) == {"A", "B"}
    # all games here have scores -> all locked in
    assert (fixtures["home_score"] != "").all()
    assert out.exists()
