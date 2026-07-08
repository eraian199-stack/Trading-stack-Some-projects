"""
Full-tournament simulation.

Ties the group stage and the knockout bracket together. `simulate_tournament_once`
plays one whole tournament (group stage -> slot map / best-third assignment ->
bracket) and returns the raw artefacts. `simulate_tournament` is the public,
Monte-Carlo entry point that aggregates many runs into per-team stage-reach
probabilities; it defaults to the FIFA World Cup 2026 format so the CLI / app /
tests can call it with just a model and the group draw.

Tournament approximations (best-third allocation, omitted drawing-of-lots) live
on `rules` and are surfaced via `fmt.notes` -- assumptions are explicit, a
project non-negotiable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..models.base import ScorelineModel
from . import rules
from .group_stage import simulate_group_stage
from .knockout import simulate_knockout


def simulate_tournament_once(
    model: ScorelineModel,
    fmt: rules.TournamentFormat,
    groups: dict[str, list[str]],
    rng: np.random.Generator,
    fixtures: pd.DataFrame | None = None,
    force_third_assignment: dict[str, str] | None = None,
    known_results: dict[frozenset, tuple] | None = None,
) -> dict[str, Any]:
    """Play one full tournament and return its artefacts.

    Returns ``{standings, slot_map, third_assignment, reached, matches}`` where
    ``matches`` concatenates the group and knockout games. ``force_third_assignment``
    pins R32 best-third slots known from the real published bracket;
    ``known_results`` locks knockout games already played.
    """
    standings, slot_map, third_assignment, group_matches = simulate_group_stage(
        model,
        groups,
        rng,
        fixtures=fixtures,
        tiebreakers=fmt.group_tiebreakers,
        n_best_third=fmt.n_best_third,
        neutral=True,
        force_third_assignment=force_third_assignment,
    )
    reached, knockout_matches = simulate_knockout(
        model, fmt, slot_map, third_assignment, rng, known_results=known_results
    )
    return {
        "standings": standings,
        "slot_map": slot_map,
        "third_assignment": third_assignment,
        "reached": reached,
        "matches": group_matches + knockout_matches,
    }


def simulate_tournament(
    model: ScorelineModel,
    groups: dict[str, list[str]],
    fixtures: pd.DataFrame | None = None,
    n_simulations: int = 5000,
    seed: int = 7,
    fmt: rules.TournamentFormat | None = None,
    force_third_assignment: dict[str, str] | None = None,
    known_results: dict[frozenset, tuple] | None = None,
) -> pd.DataFrame:
    """Monte-Carlo a whole tournament and return per-team reach probabilities.

    Defaults to ``rules.world_cup_2026_format()`` when ``fmt`` is None. Delegates
    the run loop to ``monte_carlo.monte_carlo_tournament``. The returned frame has
    one row per team with ``team``, ``group``, a ``<stage>_probability`` column
    for every stage in ``fmt.stage_order``, ``champion_probability`` and
    ``win_group_probability``; rows are sorted by champion probability desc.

    The signature is contractual -- the CLI, app and tests depend on it.
    """
    from .monte_carlo import monte_carlo_tournament  # local: avoid import cycle

    fmt = fmt if fmt is not None else rules.world_cup_2026_format()
    return monte_carlo_tournament(
        model,
        fmt,
        groups,
        fixtures=fixtures,
        n_simulations=n_simulations,
        seed=seed,
        force_third_assignment=force_third_assignment,
        known_results=known_results,
    )
