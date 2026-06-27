"""
Group-stage simulation.

Generalises the World-Cup group logic from the legacy
`world_cup_predictor.tournament` module onto the new model interface and the
declarative tie-breakers / third-place allocator in `rules`. It supports any
number of groups of any size (Euros, Copa, custom configs -- not just 4-team
WC groups), and exposes the slot map / best-third assignment the knockout stage
needs.

The engine plays every within-group fixture (caller-supplied or a generated
round-robin), uses provided actual scores where present, ranks each group via
`rules.rank_group`, then -- when the format reserves best-third places -- selects
the best `n_best_third` third-placed teams and assigns their groups to the
bracket's best-third slots via `rules.wc2026_third_place_assignment`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data.aliases import normalize_team_name
from ..models.base import ScorelineModel
from . import rules
from .match import SimulatedMatch, simulate_match


def simulate_group_stage(
    model: ScorelineModel,
    groups: dict[str, list[str]],
    rng: np.random.Generator,
    fixtures: pd.DataFrame | None = None,
    tiebreakers: tuple[str, ...] = rules.FIFA_GROUP_TIEBREAKERS,
    n_best_third: int = 0,
    neutral: bool = True,
    force_third_assignment: dict[str, str] | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, str],
    dict[str, str],
    list[SimulatedMatch],
]:
    """Play and rank all groups.

    Returns ``(standings, slot_map, third_assignment, matches)`` where:

    - ``standings``: ``{group_letter: [ranked rows ...]}`` (rules row schema +
      ``rank``), best team first.
    - ``slot_map``: ``{"1A": team, "2A": team, "3A": team, ...}`` for every
      group letter and every advancing/eligible position.
    - ``third_assignment``: ``{r32_match_id: group_letter}`` mapping each
      best-third bracket slot to the group whose third-placed team fills it
      (via ``rules.wc2026_third_place_assignment``); empty unless
      ``n_best_third > 0``.
    - ``matches``: every simulated group fixture as a ``SimulatedMatch``.

    International tournament matches are neutral, so the default ``neutral=True``
    drops home advantage; set ``neutral=False`` for a home/away group stage.
    """
    norm_groups = {
        str(g).strip().upper(): [normalize_team_name(t) for t in teams]
        for g, teams in groups.items()
    }
    base_ctx: dict[str, Any] = {"neutral_site": bool(neutral)}

    # Blank standings tables keyed by group then team.
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for group, teams in norm_groups.items():
        table[group] = {t: rules.new_standings_row(group, t) for t in teams}

    # Per-group played fixtures (for head-to-head tie-breakers).
    group_played: dict[str, list[tuple[str, str, int, int]]] = {
        g: [] for g in norm_groups
    }

    matches: list[SimulatedMatch] = []
    fixture_rows = _group_fixtures(norm_groups, fixtures)
    for match_id, group, home, away, home_score, away_score in fixture_rows:
        if group not in table:
            continue
        if home not in table[group] or away not in table[group]:
            continue
        if home_score is None or away_score is None:
            sim = simulate_match(
                model, home, away, rng, context=base_ctx,
                match_id=match_id, stage="group",
            )
            hs, as_ = sim.home_score, sim.away_score
            winner = sim.winner
        else:
            hs, as_ = int(home_score), int(away_score)
            winner = home if hs > as_ else away if as_ > hs else ""
        rules.apply_result(table[group][home], table[group][away], hs, as_)
        group_played[group].append((home, away, hs, as_))
        matches.append(
            SimulatedMatch(
                match_id=match_id,
                home_team=home,
                away_team=away,
                home_score=hs,
                away_score=as_,
                winner=winner,
                stage="group",
            )
        )

    # Rank every group and build the slot map.
    standings: dict[str, list[dict[str, Any]]] = {}
    slot_map: dict[str, str] = {}
    for group in norm_groups:
        ranked = rules.rank_group(
            list(table[group].values()),
            group_played[group],
            tiebreakers=tiebreakers,
            rng=rng,
        )
        standings[group] = ranked
        for idx, row in enumerate(ranked, start=1):
            slot_map[f"{idx}{group}"] = row["team"]

    # Best-third assignment (only when the format reserves third places).
    third_assignment: dict[str, str] = {}
    if n_best_third > 0:
        third_rows = [
            standings[g][2] for g in standings if len(standings[g]) >= 3
        ]
        # Reuse rank_group's ordering over the pooled thirds (no h2h between
        # different groups, so pass an empty match list).
        ranked_thirds = rules.rank_group(
            third_rows, [], tiebreakers=tiebreakers, rng=rng
        )
        qualified_third_groups = [r["group"] for r in ranked_thirds[:n_best_third]]
        # Pin slots known from the actual published R32 fixtures, but only for
        # groups that actually qualified in THIS run -- so counterfactual sims
        # (a different set of thirds advancing) fall back to the solver, while the
        # realised/decided scenario uses the real bracket.
        fixed = None
        if force_third_assignment:
            qset = set(qualified_third_groups)
            fixed = {m: g for m, g in force_third_assignment.items() if g in qset}
        third_assignment = rules.wc2026_third_place_assignment(
            qualified_third_groups, fixed=fixed
        )

    return standings, slot_map, third_assignment, matches


def _group_fixtures(
    groups: dict[str, list[str]],
    fixtures: pd.DataFrame | None,
) -> list[tuple[str, str, str, str, object, object]]:
    """Yield (match_id, group, home, away, home_score|None, away_score|None).

    Generates a within-group round-robin when no fixtures are supplied; sorts a
    supplied fixture frame by date then match_id so completed games are applied
    in order.
    """
    if fixtures is None:
        rows: list[tuple[str, str, str, str, object, object]] = []
        for group in sorted(groups):
            teams = groups[group]
            for idx, (home, away) in enumerate(
                rules.combinations(teams, 2), start=1
            ):
                rows.append((f"{group}{idx}", group, home, away, None, None))
        return rows

    frame = fixtures.copy()
    if "date" in frame.columns and "match_id" in frame.columns:
        frame = frame.sort_values(["date", "match_id"], na_position="last")
    elif "date" in frame.columns:
        frame = frame.sort_values("date", na_position="last")

    out: list[tuple[str, str, str, str, object, object]] = []
    for i, row in enumerate(frame.itertuples(index=False)):
        group = str(getattr(row, "group", "")).strip().upper()
        home = normalize_team_name(getattr(row, "home_team", ""))
        away = normalize_team_name(getattr(row, "away_team", ""))
        match_id = str(getattr(row, "match_id", "") or f"{group}{i + 1}")
        hs = getattr(row, "home_score", None)
        as_ = getattr(row, "away_score", None)
        if pd.isna(hs) or pd.isna(as_):
            hs, as_ = None, None
        out.append((match_id, group, home, away, hs, as_))
    return out
