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
    fixtures = world_cup.build_fixtures(hist, groups, out=out, use_store=False)
    assert len(fixtures) == 12  # two K4s = 12 group games
    assert set(fixtures["group"]) == {"A", "B"}
    # all games here have scores -> all locked in
    assert (fixtures["home_score"] != "").all()
    assert out.exists()


# --------------------------------------------------------------------------- #
# Durable results ledger
# --------------------------------------------------------------------------- #
def _wc_result(date, home, away, hs, as_):
    return {
        "date": date, "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "competition": "FIFA World Cup",
        "competition_type": "tournament", "neutral_site": True,
    }


def test_results_ledger_accumulates_and_persists(tmp_path):
    store = tmp_path / "ledger.csv"
    world_cup.update_results_store(
        extra_results=pd.DataFrame([_wc_result("2026-06-11", "t1", "t2", 2, 0)]),
        store=store,
    )
    assert store.exists()
    # A second game must not evict the first.
    led = world_cup.update_results_store(
        extra_results=pd.DataFrame([_wc_result("2026-06-12", "t3", "t4", 1, 1)]),
        store=store,
    )
    assert len(led) == 2
    assert len(world_cup.stored_results(store)) == 2
    # Monotonic: an empty refresh keeps every game already seen.
    assert len(world_cup.update_results_store(extra_results=None, store=store)) == 2


def test_completed_wc_results_dedupes_by_unordered_pair():
    # Same game reported by two feeds with home/away swapped -> one row.
    frame = pd.DataFrame([
        _wc_result("2026-06-11", "t1", "t2", 2, 0),
        _wc_result("2026-06-11", "t2", "t1", 0, 2),
    ])
    out = world_cup.completed_wc_results(extra_results=frame, use_store=False)
    assert len(out) == 1


def test_build_fixtures_order_insensitive_and_ledger(tmp_path):
    """A result whose sides are swapped vs the fixture still locks in, oriented
    to the fixture's own home/away, even when it comes only from the ledger."""
    hist = _two_group_history()
    # Blank one fixture so it has no history score and must be filled externally.
    m = (hist["home_team"] == "t1") & (hist["away_team"] == "t2")
    hist.loc[m, ["home_score", "away_score"]] = float("nan")
    groups = {"A": ["t1", "t2", "t3", "t4"], "B": ["t5", "t6", "t7", "t8"]}

    store = tmp_path / "ledger.csv"
    # Ledger lists the game SWAPPED relative to the fixture (t2 home, beat t1 3-1).
    world_cup.update_results_store(
        extra_results=pd.DataFrame([_wc_result("2026-06-11", "t2", "t1", 3, 1)]),
        store=store,
    )
    # No live feed at all -> the score can only come from the ledger.
    fixtures = world_cup.build_fixtures(
        hist, groups, extra_results=None, out=tmp_path / "f.csv", store=store
    )
    row = fixtures[(fixtures["home_team"] == "t1") & (fixtures["away_team"] == "t2")].iloc[0]
    # Oriented to the fixture: t1 (home) got 1, t2 (away) got 3.
    assert int(row["home_score"]) == 1 and int(row["away_score"]) == 3


def test_completed_wc_results_precedence_history_wins_over_extra():
    # martj42 (history) is authoritative: on a tie it beats the live odds feed.
    from soccer_predictor.data.normalizers import standardize_matches
    hist = standardize_matches(pd.DataFrame([_wc_result("2026-06-11", "t1", "t2", 2, 1)]))
    extra = pd.DataFrame([_wc_result("2026-06-11", "t1", "t2", 9, 9)])  # provisional/wrong
    out = world_cup.completed_wc_results(history=hist, extra_results=extra, use_store=False)
    row = out.iloc[0]
    assert int(row["home_score"]) == 2 and int(row["away_score"]) == 1


def test_completed_wc_results_scopes_to_valid_pairs():
    vp = world_cup.group_pair_keys({"A": ["t1", "t2", "t3", "t4"]})
    frame = pd.DataFrame([
        _wc_result("2026-06-11", "t1", "t2", 1, 0),   # group pair -> kept
        _wc_result("2026-07-10", "t1", "tX", 3, 0),   # not a group pair -> dropped
    ])
    out = world_cup.completed_wc_results(
        extra_results=frame, use_store=False, valid_pairs=vp
    )
    keys = {tuple(sorted((h, a))) for h, a in zip(out["home_team"], out["away_team"])}
    assert keys == {("t1", "t2")}


def test_ledger_group_end_blocks_same_pair_knockout_overwrite(tmp_path):
    """A final/SF rematch of two SAME-group teams shares the group pair key; the
    group-stage date window must stop it from overwriting the group result."""
    store = tmp_path / "ledger.csv"
    vp = world_cup.group_pair_keys({"A": ["t1", "t2", "t3", "t4"]})
    group_end = pd.Timestamp("2026-06-27")
    # Group game persisted.
    world_cup.update_results_store(
        extra_results=pd.DataFrame([_wc_result("2026-06-11", "t1", "t2", 2, 1)]),
        store=store, valid_pairs=vp, group_end=group_end,
    )
    # A July knockout rematch of the same pair must NOT enter or overwrite.
    led = world_cup.update_results_store(
        extra_results=pd.DataFrame([_wc_result("2026-07-19", "t2", "t1", 0, 0)]),
        store=store, valid_pairs=vp, group_end=group_end,
    )
    row = led[(led["home_team"] == "t1") & (led["away_team"] == "t2")].iloc[0]
    assert int(row["home_score"]) == 2 and int(row["away_score"]) == 1  # group result intact


def test_group_end_autoderives_from_full_fixtures_not_scored():
    """group_end must come from the WHOLE fixture list (incl. unplayed games), so
    a result played after the last SCORED game but within the group stage is kept
    -- not dropped because scored games happen to lag the schedule."""
    from itertools import combinations
    from soccer_predictor.data.normalizers import standardize_matches

    rows = []
    # Group A: all four-team round-robin PLAYED early (June 11-16).
    A = ["a1", "a2", "a3", "a4"]
    for i, (h, a) in enumerate(combinations(A, 2)):
        rows.append({**_wc_result(f"2026-06-1{1 + i}", h, a, 1, 0)})
    # Group B: scheduled LATER (up to June 25) but NOT yet played (no scores).
    B = ["b1", "b2", "b3", "b4"]
    for i, (h, a) in enumerate(combinations(B, 2)):
        rows.append({
            "date": f"2026-06-2{i}", "home_team": h, "away_team": a,
            "home_score": None, "away_score": None, "competition": "FIFA World Cup",
            "competition_type": "tournament", "neutral_site": True,
        })
    history = standardize_matches(pd.DataFrame(rows))
    # A live result for a group-B game on June 20 -- after the last SCORED date
    # (June 16) but well within the group stage (last fixture June 25).
    extra = pd.DataFrame([_wc_result("2026-06-20", "b1", "b2", 3, 1)])
    out = world_cup.completed_wc_results(history=history, extra_results=extra, use_store=False)
    keys = {tuple(sorted((h, a))) for h, a in zip(out["home_team"], out["away_team"])}
    assert ("b1", "b2") in keys  # kept: group_end came from the schedule, not scores


def test_update_results_store_quarantines_corrupt_ledger(tmp_path):
    """A corrupt ledger must be quarantined, never silently overwritten empty."""
    store = tmp_path / "ledger.csv"
    world_cup.update_results_store(
        extra_results=pd.DataFrame([_wc_result("2026-06-11", "t1", "t2", 2, 0)]),
        store=store,
    )
    store.write_text("totally,not\na,valid\nledger,,,\n\x00\x01garbage")  # corrupt it
    # Re-run with no feeds: the parseable subset is empty, but the bytes survive.
    world_cup.update_results_store(extra_results=None, store=store)
    bak = store.with_suffix(store.suffix + ".corrupt")
    assert bak.exists()  # original bytes preserved for recovery
    assert "garbage" in bak.read_text()
    # No stray temp file left behind.
    assert not store.with_suffix(store.suffix + ".tmp").exists()
