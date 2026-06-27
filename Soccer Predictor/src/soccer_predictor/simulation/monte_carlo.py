"""
Monte-Carlo aggregation over many simulated runs.

`monte_carlo_tournament` runs `simulate_tournament_once` N times and tallies, per
team, how often it reached each knockout stage, won the title, and won its group;
it returns those tallies as probabilities. `monte_carlo_league` does the analogous
job for a single-table league (title / top-N / mean finishing position).

Both functions consume the model only through the simulation engines, which in
turn touch it only via its scoreline interface -- so swapping the model never
changes this aggregation layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..data.aliases import normalize_team_name
from ..models.base import ScorelineModel
from . import rules
from .league import simulate_league
from .tournament import simulate_tournament_once


def monte_carlo_tournament(
    model: ScorelineModel,
    fmt: rules.TournamentFormat,
    groups: dict[str, list[str]],
    fixtures: pd.DataFrame | None = None,
    n_simulations: int = 5000,
    seed: int = 7,
    progress: Callable[[int, int], None] | None = None,
    force_third_assignment: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate N tournament runs into per-team reach probabilities.

    The returned frame has one row per team (``team``, ``group``) plus a
    ``<stage>_probability`` column for each stage in ``fmt.stage_order``, a
    ``champion_probability`` column, and a ``win_group_probability`` column.
    Sorted by ``champion_probability`` descending. ``progress(i, n)`` is called
    after each run when supplied (for a CLI / Streamlit progress bar).
    """
    rng = np.random.default_rng(seed)
    n_simulations = int(n_simulations)

    # team -> group (canonical), preserving the draw's grouping.
    team_to_group: dict[str, str] = {}
    for group, teams in groups.items():
        glabel = str(group).strip().upper()
        for team in teams:
            team_to_group[normalize_team_name(team)] = glabel

    stages = list(fmt.stage_order)
    # Counters: per team, per stage / champion / group win.
    reach_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_win_counts: Counter[str] = Counter()

    for i in range(n_simulations):
        result = simulate_tournament_once(
            model, fmt, groups, rng, fixtures=fixtures,
            force_third_assignment=force_third_assignment,
        )
        reached = result["reached"]
        slot_map = result["slot_map"]

        # A team "reached" a stage if it appears among that stage's participants.
        for stage in stages:
            for team in set(reached.get(stage, [])):
                reach_counts[team][stage] += 1
        for team in reached.get("champion", []):
            reach_counts[team]["champion"] += 1

        # Group winners are the "1<group>" slots.
        for group in {g for g in team_to_group.values()}:
            winner = slot_map.get(f"1{group}")
            if winner:
                group_win_counts[winner] += 1

        if progress is not None:
            progress(i + 1, n_simulations)

    denom = float(n_simulations) if n_simulations > 0 else 1.0
    prob_columns = stages + ["champion"]
    rows: list[dict[str, Any]] = []
    for team in sorted(team_to_group):
        row: dict[str, Any] = {"team": team, "group": team_to_group[team]}
        for stage in prob_columns:
            row[f"{stage}_probability"] = reach_counts[team][stage] / denom
        row["win_group_probability"] = group_win_counts[team] / denom
        rows.append(row)

    frame = pd.DataFrame(rows)
    if "champion_probability" in frame.columns and not frame.empty:
        frame = frame.sort_values(
            "champion_probability", ascending=False
        ).reset_index(drop=True)
    return frame


def monte_carlo_league(
    model: ScorelineModel,
    teams: list[str],
    n_simulations: int = 2000,
    seed: int = 7,
    top_n: int = 4,
    double_round_robin: bool = True,
    fixtures: pd.DataFrame | None = None,
    tiebreakers: tuple[str, ...] = rules.LEAGUE_TIEBREAKERS,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Aggregate N league seasons into per-team title / top-N / position stats.

    Returns one row per team with ``team``, ``title_probability`` (finished 1st),
    ``top_<n>_probability`` (finished in the top ``top_n``), ``avg_points``,
    ``avg_rank``, and ``avg_position`` (same as avg_rank, kept for readability).
    Sorted by title probability descending.
    """
    rng = np.random.default_rng(seed)
    n_simulations = int(n_simulations)
    teams = [normalize_team_name(t) for t in teams]

    title_counts: Counter[str] = Counter()
    topn_counts: Counter[str] = Counter()
    points_sum: dict[str, float] = defaultdict(float)
    rank_sum: dict[str, float] = defaultdict(float)

    for i in range(n_simulations):
        standings = simulate_league(
            model,
            teams,
            rng,
            double_round_robin=double_round_robin,
            fixtures=fixtures,
            tiebreakers=tiebreakers,
        )
        for row in standings:
            team = row["team"]
            rank = row["rank"]
            rank_sum[team] += rank
            points_sum[team] += row["points"]
            if rank == 1:
                title_counts[team] += 1
            if rank <= top_n:
                topn_counts[team] += 1
        if progress is not None:
            progress(i + 1, n_simulations)

    denom = float(n_simulations) if n_simulations > 0 else 1.0
    rows: list[dict[str, Any]] = []
    for team in sorted(set(teams)):
        avg_rank = rank_sum[team] / denom
        rows.append(
            {
                "team": team,
                "title_probability": title_counts[team] / denom,
                f"top_{top_n}_probability": topn_counts[team] / denom,
                "avg_points": points_sum[team] / denom,
                "avg_rank": avg_rank,
                "avg_position": avg_rank,
            }
        )

    frame = pd.DataFrame(rows)
    if "title_probability" in frame.columns and not frame.empty:
        frame = frame.sort_values(
            "title_probability", ascending=False
        ).reset_index(drop=True)
    return frame
