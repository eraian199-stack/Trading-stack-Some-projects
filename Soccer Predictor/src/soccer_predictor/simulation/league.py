"""
Single-table league simulation.

`simulate_league` plays a (double) round-robin among a set of teams, building a
standings table via `rules.new_standings_row` / `rules.apply_result` and ranking
it with the configured tie-breakers. Where the caller supplies fixtures with
actual scores those are used; missing scores are sampled from the model. This
backs both the league Monte Carlo and the "rest-of-season" projections in the
app.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data.aliases import normalize_team_name
from ..models.base import ScorelineModel
from . import rules
from .match import simulate_match


def simulate_league(
    model: ScorelineModel,
    teams: list[str],
    rng: np.random.Generator,
    double_round_robin: bool = True,
    fixtures: pd.DataFrame | None = None,
    tiebreakers: tuple[str, ...] = rules.LEAGUE_TIEBREAKERS,
    context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Simulate a league season and return ranked standings rows.

    Each row follows `rules.new_standings_row` plus a `rank` key. The default
    fixture list is a round-robin (`double_round_robin` -> home-and-away). If
    `fixtures` is given, completed rows (both scores present) use their actual
    score and pending rows are sampled; this lets the simulator project the rest
    of a season already in progress. `context` is merged into every match
    context (e.g. set `neutral_site=False` for a normal home/away league).
    """
    teams = [normalize_team_name(t) for t in teams]
    base_ctx: dict[str, Any] = dict(context) if context is not None else {}

    table: dict[str, dict[str, Any]] = {
        t: rules.new_standings_row("", t) for t in teams
    }
    team_set = set(teams)
    # Track within-league matches for any head-to-head tie-breakers.
    played: list[tuple[str, str, int, int]] = []

    fixture_rows = _league_fixtures(teams, double_round_robin, fixtures)
    for home, away, home_score, away_score in fixture_rows:
        if home not in team_set or away not in team_set:
            continue
        if home_score is None or away_score is None:
            sim = simulate_match(model, home, away, rng, context=base_ctx)
            hs, as_ = sim.home_score, sim.away_score
        else:
            hs, as_ = int(home_score), int(away_score)
        rules.apply_result(table[home], table[away], hs, as_)
        played.append((home, away, hs, as_))

    ranked = rules.rank_group(
        list(table.values()), played, tiebreakers=tiebreakers, rng=rng
    )
    return ranked


def _league_fixtures(
    teams: list[str],
    double_round_robin: bool,
    fixtures: pd.DataFrame | None,
) -> list[tuple[str, str, object, object]]:
    """Return (home, away, home_score|None, away_score|None) fixtures.

    Falls back to a generated round-robin when no fixture frame is supplied.
    """
    if fixtures is None:
        pairs = rules.round_robin_fixtures(teams, double=double_round_robin)
        return [(h, a, None, None) for h, a in pairs]

    out: list[tuple[str, str, object, object]] = []
    for row in fixtures.itertuples(index=False):
        home = normalize_team_name(getattr(row, "home_team", ""))
        away = normalize_team_name(getattr(row, "away_team", ""))
        if not home or not away:
            continue
        hs = getattr(row, "home_score", None)
        as_ = getattr(row, "away_score", None)
        if pd.isna(hs) or pd.isna(as_):
            hs, as_ = None, None
        out.append((home, away, hs, as_))
    return out
