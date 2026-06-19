"""
Single-match simulation.

`simulate_match` draws one fixture from a `ScorelineModel`. League and group
games stop after 90 minutes; knockout games that finish level go to extra time
(extra Poisson goals scaled to a 30-minute period) and, if still level, a
near-random penalty shootout (open-play edge shrunk toward 50/50).

The engine touches the model ONLY through its public scoreline interface
(`sample_score`, `expected_goals`, `win_probability_no_draw`) -- it never
hardcodes model internals (a project non-negotiable). International tournament
matches are neutral; pass that fact via `context["neutral_site"] = True` and the
model drops its home-advantage term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..models.base import ScorelineModel
from . import rules


@dataclass
class SimulatedMatch:
    """The outcome of one simulated fixture.

    `winner` is the canonical name of the side that advances (after ET/penalties
    in a knockout) or the 90-minute winner; it is "" for a genuine draw (only
    possible in league/group games, never in a resolved knockout).
    """

    match_id: str = ""
    home_team: str = ""
    away_team: str = ""
    home_score: int = 0
    away_score: int = 0
    winner: str = ""
    stage: str = ""
    extra_time: bool = False
    penalties: bool = False
    # Two-leg ties: home_score/away_score carry the AGGREGATE; these record the
    # per-leg scoreline (leg 1 at home_team's ground, leg 2 at away_team's).
    two_leg: bool = False
    leg1_home: int = 0
    leg1_away: int = 0
    leg2_home: int = 0
    leg2_away: int = 0


def simulate_match(
    model: ScorelineModel,
    home: str,
    away: str,
    rng: np.random.Generator,
    context: Mapping[str, Any] | None = None,
    knockout: bool = False,
    tie: rules.KnockoutTie | None = None,
    match_id: str = "",
    stage: str = "",
) -> SimulatedMatch:
    """Simulate one match and return a `SimulatedMatch`.

    Regulation: 90 minutes are drawn via `model.sample_score`. For a knockout
    game that is level after 90', if `tie.extra_time` is set we add a Poisson
    draw per side with rate `expected_goals * tie.et_fraction` (30 ET minutes are
    roughly a third of a 90-minute scoring rate). If it is still level and
    `tie.penalties` is set, the winner is decided by a PENALTY SHOOTOUT modelled
    as near-random: the open-play win probability is shrunk toward 0.5 by
    `tie.shootout_skill` (default 0.25, so a strong favourite wins a shootout only
    a little above 50%, matching the empirical near-coin-flip). The recorded
    scoreline is unchanged.
    """
    tie = tie if tie is not None else rules.KnockoutTie()
    ctx = dict(context) if context is not None else {}
    if knockout:
        ctx["knockout"] = True

    home_score, away_score = model.sample_score(home, away, rng, ctx)
    extra_time = False
    penalties = False

    if knockout and home_score == away_score:
        if tie.extra_time:
            extra_time = True
            eg_home, eg_away = model.expected_goals(home, away, ctx)
            home_score += int(rng.poisson(max(eg_home, 1e-4) * tie.et_fraction))
            away_score += int(rng.poisson(max(eg_away, 1e-4) * tie.et_fraction))

        if home_score == away_score:
            if tie.penalties:
                penalties = True
                # Shootouts are ~coin-flips: shrink the open-play edge toward 0.5
                # by `tie.shootout_skill` (0 = pure 50/50, 1 = full win prob).
                p_raw = model.win_probability_no_draw(home, away, ctx)
                p_home = 0.5 + tie.shootout_skill * (p_raw - 0.5)
                winner = home if rng.random() < p_home else away
            else:
                # No resolution configured: fall back to a fair coin so a
                # knockout never returns an unresolved draw.
                winner = home if rng.random() < 0.5 else away
            return SimulatedMatch(
                match_id=match_id,
                home_team=home,
                away_team=away,
                home_score=int(home_score),
                away_score=int(away_score),
                winner=winner,
                stage=stage,
                extra_time=extra_time,
                penalties=penalties,
            )

    if home_score > away_score:
        winner = home
    elif away_score > home_score:
        winner = away
    else:
        winner = ""  # genuine draw (league/group only)

    return SimulatedMatch(
        match_id=match_id,
        home_team=home,
        away_team=away,
        home_score=int(home_score),
        away_score=int(away_score),
        winner=winner,
        stage=stage,
        extra_time=extra_time,
        penalties=penalties,
    )


def simulate_two_leg_tie(
    model: ScorelineModel,
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    tie: rules.KnockoutTie | None = None,
    match_id: str = "",
    stage: str = "",
) -> SimulatedMatch:
    """Resolve a two-legged knockout tie (e.g. a Champions League round).

    Leg 1 is hosted by ``team_a``, leg 2 by ``team_b`` -- each leg uses home
    advantage (NOT neutral). The winner is decided on aggregate; if
    ``tie.away_goals`` is set, away goals break an aggregate tie before extra
    time. If still level, extra time (added to leg 2) and then penalties resolve
    it. The result's ``home_score`` / ``away_score`` carry the AGGREGATE
    (team_a, team_b); per-leg scores are on the leg fields.
    """
    tie = tie if tie is not None else rules.KnockoutTie(two_leg=True)
    leg_ctx = {"neutral_site": False, "knockout": True}

    # Leg 1 at team_a's ground; leg 2 at team_b's ground.
    l1_home, l1_away = model.sample_score(team_a, team_b, rng, leg_ctx)
    l2_home, l2_away = model.sample_score(team_b, team_a, rng, leg_ctx)

    agg_a = l1_home + l2_away  # team_a goals over both legs
    agg_b = l1_away + l2_home  # team_b goals over both legs
    extra_time = False
    penalties = False

    def _away_goals_winner() -> str:
        # team_a's away goals are scored at team_b's ground (leg 2 = l2_away);
        # team_b's away goals are scored at team_a's ground (leg 1 = l1_away).
        return team_a if l2_away > l1_away else team_b

    if agg_a > agg_b:
        winner = team_a
    elif agg_b > agg_a:
        winner = team_b
    elif tie.away_goals and l2_away != l1_away:
        winner = _away_goals_winner()
    else:
        if tie.extra_time:
            extra_time = True
            # ET played in leg 2 (team_b hosting team_a).
            eg_host, eg_visitor = model.expected_goals(team_b, team_a, leg_ctx)
            l2_home += int(rng.poisson(max(eg_host, 1e-4) * tie.et_fraction))
            l2_away += int(rng.poisson(max(eg_visitor, 1e-4) * tie.et_fraction))
            agg_a = l1_home + l2_away
            agg_b = l1_away + l2_home
        if agg_a > agg_b:
            winner = team_a
        elif agg_b > agg_a:
            winner = team_b
        elif tie.away_goals and l2_away != l1_away:
            winner = _away_goals_winner()
        else:
            penalties = True
            p_a = model.win_probability_no_draw(team_a, team_b, {"neutral_site": True})
            winner = team_a if rng.random() < p_a else team_b

    return SimulatedMatch(
        match_id=match_id,
        home_team=team_a,
        away_team=team_b,
        home_score=int(agg_a),
        away_score=int(agg_b),
        winner=winner,
        stage=stage,
        extra_time=extra_time,
        penalties=penalties,
        two_leg=True,
        leg1_home=int(l1_home),
        leg1_away=int(l1_away),
        leg2_home=int(l2_home),
        leg2_away=int(l2_away),
    )
