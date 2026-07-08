"""
Knockout-bracket simulation.

`simulate_knockout` walks a `TournamentFormat`'s declared bracket in stage order,
resolving each `BracketMatch`'s source codes to concrete teams and playing the
tie with `simulate_match(..., knockout=True)` so draws are always broken. Source
codes follow the convention declared in `rules.BracketMatch`:

    "1A" / "2A" / "3A"   group slots          -> slot_map["1A"]
    "3A/B/C/D/F"         best-third placeholder -> slot_map["3" + third_assignment[match_id]]
    "M73"                a prior match's winner -> winners["M73"]
    "S1"                 a seed code (single-elim) -> slot_map["S1"]

The convenience `simulate_single_elimination` builds a seeded bracket and a
trivial slot map for a pure knockout cup (continental finals, club cups).
"""

from __future__ import annotations

import numpy as np

from ..data.aliases import normalize_team_name
from ..models.base import ScorelineModel
from . import rules
from .match import SimulatedMatch, simulate_match, simulate_two_leg_tie


def _resolve_source(
    src: str,
    slot_map: dict[str, str],
    third_assignment: dict[str, str],
    winners: dict[str, str],
    match_id: str,
) -> str:
    """Resolve one bracket source code to a concrete team name."""
    # Best-third placeholder, e.g. "3A/B/C/D/F".
    if src.startswith("3") and "/" in src:
        group = third_assignment.get(match_id)
        if group is None:
            raise KeyError(
                f"No third-place assignment for bracket match {match_id!r}"
            )
        return slot_map[f"3{group}"]
    # Prior match winner, e.g. "M73".
    if src in winners:
        return winners[src]
    # Group slot ("1A"/"2A"/"3A") or seed code ("S1") -> slot_map lookup.
    return slot_map[src]


def simulate_knockout(
    model: ScorelineModel,
    fmt: rules.TournamentFormat,
    slot_map: dict[str, str],
    third_assignment: dict[str, str],
    rng: np.random.Generator,
    known_results: dict[frozenset, tuple] | None = None,
) -> tuple[dict[str, list[str]], list[SimulatedMatch]]:
    """Play `fmt.bracket` to completion.

    Returns ``(reached, matches)`` where ``reached`` maps each knockout stage in
    ``fmt.stage_order`` to the list of teams that *played in* that stage, plus a
    final ``"champion"`` key holding the single winner. ``matches`` is every
    bracket game played.

    Knockout matches are neutral (international tournament) and always resolved:
    `simulate_match(..., knockout=True, tie=fmt.knockout)` adds ET / penalties so
    no draw survives.

    ``known_results`` locks games that have ACTUALLY been played:
    ``{frozenset({team_a, team_b}): (winner, winner_goals, loser_goals)}``. A
    bracket match whose two resolved teams are in ``known_results`` uses the real
    outcome instead of a fresh sim, so eliminated teams drop to 0 and the survivors'
    probabilities condition on what already happened.
    """
    winners: dict[str, str] = {}
    reached: dict[str, list[str]] = {stage: [] for stage in fmt.stage_order}
    reached["champion"] = []
    matches: list[SimulatedMatch] = []
    base_ctx = {"neutral_site": True}

    # Index the bracket by stage to play first-round-to-final in order.
    by_stage: dict[str, list[rules.BracketMatch]] = {}
    for bm in fmt.bracket:
        by_stage.setdefault(bm.stage, []).append(bm)

    final_stage = fmt.stage_order[-1] if fmt.stage_order else None

    for stage in fmt.stage_order:
        for bm in by_stage.get(stage, []):
            home = _resolve_source(
                bm.home_src, slot_map, third_assignment, winners, bm.match_id
            )
            away = _resolve_source(
                bm.away_src, slot_map, third_assignment, winners, bm.match_id
            )
            reached[stage].append(home)
            reached[stage].append(away)
            actual = known_results.get(frozenset((home, away))) if known_results else None
            if actual is not None:
                # Lock in a game already played: use the real winner + score.
                win, wg, lg = actual
                win = win if win in (home, away) else home
                hs, as_ = (wg, lg) if home == win else (lg, wg)
                match = SimulatedMatch(
                    match_id=bm.match_id, home_team=home, away_team=away,
                    home_score=hs, away_score=as_, winner=win, stage=stage,
                )
            elif fmt.knockout.two_leg:
                # Home-and-away tie (e.g. Champions League): each side hosts a
                # leg, decided on aggregate (+ away goals / ET / penalties).
                match = simulate_two_leg_tie(
                    model, home, away, rng,
                    tie=fmt.knockout, match_id=bm.match_id, stage=stage,
                )
            else:
                match = simulate_match(
                    model, home, away, rng,
                    context=base_ctx, knockout=True, tie=fmt.knockout,
                    match_id=bm.match_id, stage=stage,
                )
            winners[bm.match_id] = match.winner
            matches.append(match)
            if stage == final_stage:
                reached["champion"].append(match.winner)

    return reached, matches


def simulate_single_elimination(
    model: ScorelineModel,
    seeded_teams: list[str],
    rng: np.random.Generator,
    tie: rules.KnockoutTie | None = None,
) -> tuple[dict[str, list[str]], list[SimulatedMatch]]:
    """Run a pure seeded single-elimination cup over `seeded_teams`.

    `seeded_teams` is ordered best-seed-first; the bracket pairs 1-vs-n,
    2-vs-(n-1), ... (a power-of-two count is required). Returns the same
    ``(reached, matches)`` shape as `simulate_knockout`.
    """
    teams = [normalize_team_name(t) for t in seeded_teams]
    bracket, stage_order = rules.single_elimination_bracket(len(teams))
    slot_map = {f"S{i + 1}": teams[i] for i in range(len(teams))}
    fmt = rules.TournamentFormat(
        name="single_elimination",
        kind="knockout",
        knockout=tie if tie is not None else rules.KnockoutTie(),
        bracket=bracket,
        stage_order=stage_order,
    )
    return simulate_knockout(model, fmt, slot_map, {}, rng)
