"""
Tournament / league simulation tests.

The headline invariants:
  * WC2026: 48 teams in the output, champion probabilities sum to ~1, every
    probability lives in [0, 1];
  * the group stage produces correctly-sized standings (one ranked row per team,
    ranks 1..n);
  * a single-elimination cup always crowns exactly one champion;
  * a knockout never returns an unresolved draw (every game has a winner).

Sim counts are tiny (n_simulations <= 8) so the suite is fast; the EloGoalsModel
is used because it never blows up on unseen teams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soccer_predictor.data import synthetic
from soccer_predictor.models.elo_goals import EloGoalsModel
from soccer_predictor.simulation import rules
from soccer_predictor.simulation.group_stage import simulate_group_stage
from soccer_predictor.simulation.knockout import (
    simulate_knockout,
    simulate_single_elimination,
)
from soccer_predictor.simulation.league import simulate_league
from soccer_predictor.simulation.tournament import (
    simulate_tournament,
    simulate_tournament_once,
)


@pytest.fixture(scope="module")
def fitted_model() -> EloGoalsModel:
    intl = synthetic.generate_international_history(
        n_teams=48, n_matches=700, seed=31
    )
    return EloGoalsModel().fit(intl)


@pytest.fixture(scope="module")
def wc_groups(fitted_model) -> dict[str, list[str]]:
    intl = synthetic.generate_international_history(
        n_teams=48, n_matches=700, seed=31
    )
    teams = sorted(set(intl["home_team"]) | set(intl["away_team"]))[:48]
    assert len(teams) == 48
    return {g: teams[i * 4 : (i + 1) * 4] for i, g in enumerate("ABCDEFGHIJKL")}


# --------------------------------------------------------------------------- #
# WC2026 full Monte Carlo
# --------------------------------------------------------------------------- #
def test_wc2026_has_48_teams_and_champion_sums_to_one(fitted_model, wc_groups):
    df = simulate_tournament(fitted_model, wc_groups, n_simulations=8, seed=1)
    assert len(df) == 48
    assert abs(float(df["champion_probability"].sum()) - 1.0) < 1e-9


def test_wc2026_all_probabilities_in_unit_interval(fitted_model, wc_groups):
    df = simulate_tournament(fitted_model, wc_groups, n_simulations=8, seed=2)
    prob_cols = [c for c in df.columns if c.endswith("_probability")]
    block = df[prob_cols].to_numpy(dtype=float)
    assert np.all(block >= 0.0) and np.all(block <= 1.0)
    # Every WC2026 stage has a probability column.
    for stage in rules.WC2026_STAGE_ORDER:
        assert f"{stage}_probability" in df.columns
    assert "win_group_probability" in df.columns


def test_wc2026_stage_probabilities_are_monotone(fitted_model, wc_groups):
    """A team cannot reach a later stage more often than an earlier one."""
    df = simulate_tournament(fitted_model, wc_groups, n_simulations=8, seed=3)
    order = list(rules.WC2026_STAGE_ORDER) + ["champion"]
    cols = [f"{s}_probability" for s in order]
    block = df[cols].to_numpy(dtype=float)
    # Each column is >= the next (within float tolerance).
    diffs = block[:, :-1] - block[:, 1:]
    assert np.all(diffs >= -1e-9)


# --------------------------------------------------------------------------- #
# Group stage shapes
# --------------------------------------------------------------------------- #
def test_group_stage_standings_have_right_team_counts(fitted_model, wc_groups):
    rng = np.random.default_rng(5)
    standings, slot_map, third_assignment, matches = simulate_group_stage(
        fitted_model,
        wc_groups,
        rng,
        n_best_third=8,
        neutral=True,
    )
    assert set(standings.keys()) == set(wc_groups.keys())
    for group, rows in standings.items():
        assert len(rows) == len(wc_groups[group])  # one row per team
        assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    # slot_map exposes 1A/2A for every group (advancing slots).
    for group in wc_groups:
        assert f"1{group}" in slot_map and f"2{group}" in slot_map
    # Best-third assignment fills the 8 best-third R32 slots.
    assert len(third_assignment) == 8
    assert all(g in wc_groups for g in third_assignment.values())
    assert matches  # group games were played


def test_group_stage_no_third_assignment_when_disabled(fitted_model, wc_groups):
    rng = np.random.default_rng(6)
    _, _, third_assignment, _ = simulate_group_stage(
        fitted_model, wc_groups, rng, n_best_third=0
    )
    assert third_assignment == {}


# --------------------------------------------------------------------------- #
# Knockout always yields exactly one resolved champion
# --------------------------------------------------------------------------- #
def test_tournament_once_knockout_has_no_draws(fitted_model, wc_groups):
    rng = np.random.default_rng(7)
    fmt = rules.world_cup_2026_format()
    result = simulate_tournament_once(fitted_model, fmt, wc_groups, rng)
    reached = result["reached"]
    # Exactly one champion.
    assert len(reached["champion"]) == 1
    champ = reached["champion"][0]
    assert champ  # non-empty team name
    # No knockout game ended in an unresolved draw.
    knockout_stages = set(fmt.stage_order)
    for m in result["matches"]:
        if m.stage in knockout_stages:
            assert m.winner, f"knockout match {m.match_id} has no winner"


def test_single_elimination_crowns_one_champion(fitted_model):
    rng = np.random.default_rng(8)
    teams = [f"Seed {i}" for i in range(8)]  # power of two
    reached, matches = simulate_single_elimination(fitted_model, teams, rng)
    assert len(reached["champion"]) == 1
    assert reached["champion"][0] in [t for t in teams]
    # final stage exists and every game resolved.
    assert "final" in reached
    for m in matches:
        assert m.winner


def test_single_elimination_rejects_non_power_of_two(fitted_model):
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        simulate_single_elimination(fitted_model, ["A", "B", "C"], rng)


# --------------------------------------------------------------------------- #
# League simulation
# --------------------------------------------------------------------------- #
def test_simulate_league_ranks_all_teams(fitted_model):
    rng = np.random.default_rng(10)
    teams = [f"Club {i}" for i in range(6)]
    standings = simulate_league(
        fitted_model, teams, rng, double_round_robin=False
    )
    assert len(standings) == len(teams)
    assert [r["rank"] for r in standings] == list(range(1, len(teams) + 1))
    # Each team plays (n-1) games in a single round-robin.
    for r in standings:
        assert r["played"] == len(teams) - 1


# --------------------------------------------------------------------------- #
# M4 regression: head-to-head block must keep fair_play in the chain (not noise).
# --------------------------------------------------------------------------- #
def test_rank_group_respects_fair_play_within_h2h_block():
    # Two teams identical on points/GD/GF and on head-to-head (they drew), but A
    # has a cleaner fair-play record. A must always finish above B.
    above = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        rows = [rules.new_standings_row("G", t) for t in ["A", "B"]]
        by = {r["team"]: r for r in rows}
        # identical records: each beat C and D by the same scores, drew each other
        for r in rows:
            r["played"], r["points"], r["gf"], r["ga"], r["gd"], r["wins"], r["draws"] = (
                3, 5, 4, 2, 2, 1, 2,
            )
        by["A"]["fair_play"] = 0   # no cards
        by["B"]["fair_play"] = 3   # worse discipline
        matches = [("A", "B", 1, 1)]  # head-to-head is a draw -> can't separate
        ranked = rules.rank_group(rows, matches, rules.FIFA_GROUP_TIEBREAKERS, rng)
        if ranked[0]["team"] == "A":
            above += 1
    assert above == 60, f"fair-play tiebreak ignored ({above}/60)"


# --------------------------------------------------------------------------- #
# M5 regression: two-leg ties are actually played as two legs and resolved.
# --------------------------------------------------------------------------- #
def test_two_leg_tie_aggregates_and_resolves(fitted_model):
    from soccer_predictor.simulation.match import simulate_two_leg_tie

    model = fitted_model
    rng = np.random.default_rng(0)
    teams = sorted(model.known_teams())[:2]
    seen_winner = set()
    for _ in range(50):
        m = simulate_two_leg_tie(
            model, teams[0], teams[1], rng,
            tie=rules.KnockoutTie(two_leg=True, away_goals=False),
        )
        assert m.two_leg is True
        # aggregate equals the sum of the two legs
        assert m.home_score == m.leg1_home + m.leg2_away
        assert m.away_score == m.leg1_away + m.leg2_home
        assert m.winner in (teams[0], teams[1])  # always resolved
        seen_winner.add(m.winner)
    assert len(seen_winner) >= 1


def test_two_leg_knockout_format_plays_two_legs(fitted_model):
    """A format with two_leg=True must route through the two-leg resolver."""
    from soccer_predictor.simulation.knockout import simulate_single_elimination

    model = fitted_model
    teams = sorted(model.known_teams())[:4]
    rng = np.random.default_rng(1)
    reached, matches = simulate_single_elimination(
        model, teams, rng, tie=rules.KnockoutTie(two_leg=True)
    )
    assert all(m.two_leg for m in matches)
    assert len(reached["champion"]) == 1
