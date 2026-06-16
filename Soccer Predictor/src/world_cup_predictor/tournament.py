from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .data import groups_to_frame, normalize_team_name
from .model import MatchPredictor


R32_MATCHES = [
    ("M73", "2A", "2B"),
    ("M74", "1E", "3A/B/C/D/F"),
    ("M75", "1F", "2C"),
    ("M76", "1C", "2F"),
    ("M77", "1I", "3C/D/F/G/H"),
    ("M78", "2E", "2I"),
    ("M79", "1A", "3C/E/F/H/I"),
    ("M80", "1L", "3E/H/I/J/K"),
    ("M81", "1D", "3B/E/F/I/J"),
    ("M82", "1G", "3A/E/H/I/J"),
    ("M83", "2K", "2L"),
    ("M84", "1H", "2J"),
    ("M85", "1B", "3E/F/G/I/J"),
    ("M86", "1J", "2H"),
    ("M87", "1K", "3D/E/I/J/L"),
    ("M88", "2D", "2G"),
]

NEXT_ROUNDS = {
    "round_of_16": [
        ("M89", "M73", "M75"),
        ("M90", "M74", "M77"),
        ("M91", "M76", "M78"),
        ("M92", "M79", "M80"),
        ("M93", "M83", "M84"),
        ("M94", "M81", "M82"),
        ("M95", "M86", "M88"),
        ("M96", "M85", "M87"),
    ],
    "quarterfinal": [
        ("M97", "M89", "M90"),
        ("M98", "M93", "M94"),
        ("M99", "M91", "M92"),
        ("M100", "M95", "M96"),
    ],
    "semifinal": [
        ("M101", "M97", "M98"),
        ("M102", "M99", "M100"),
    ],
    "final": [
        ("M104", "M101", "M102"),
    ],
}


@dataclass
class SimulatedMatch:
    match_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    winner: str
    stage: str


def _blank_table(groups: dict[str, list[str]]) -> dict[str, dict[str, dict[str, Any]]]:
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for group, teams in groups.items():
        table[group] = {}
        for team in teams:
            table[group][normalize_team_name(team)] = {
                "group": group,
                "team": normalize_team_name(team),
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "points": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "tie_noise": 0.0,
            }
    return table


def _apply_group_result(
    table: dict[str, dict[str, dict[str, Any]]],
    group: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> None:
    home = table[group][normalize_team_name(home_team)]
    away = table[group][normalize_team_name(away_team)]
    home["played"] += 1
    away["played"] += 1
    home["gf"] += home_score
    home["ga"] += away_score
    away["gf"] += away_score
    away["ga"] += home_score
    home["gd"] = home["gf"] - home["ga"]
    away["gd"] = away["gf"] - away["ga"]
    if home_score > away_score:
        home["wins"] += 1
        home["points"] += 3
        away["losses"] += 1
    elif away_score > home_score:
        away["wins"] += 1
        away["points"] += 3
        home["losses"] += 1
    else:
        home["draws"] += 1
        away["draws"] += 1
        home["points"] += 1
        away["points"] += 1


def _standings(
    table: dict[str, dict[str, dict[str, Any]]],
    rng: np.random.Generator,
) -> dict[str, list[dict[str, Any]]]:
    ranked: dict[str, list[dict[str, Any]]] = {}
    for group, team_rows in table.items():
        rows = []
        for row in team_rows.values():
            copied = dict(row)
            copied["tie_noise"] = float(rng.random())
            rows.append(copied)
        rows.sort(
            key=lambda r: (
                r["points"],
                r["gd"],
                r["gf"],
                r["wins"],
                r["tie_noise"],
            ),
            reverse=True,
        )
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx
        ranked[group] = rows
    return ranked


def _default_group_fixtures(groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for group, teams in sorted(groups.items()):
        for idx, (home, away) in enumerate(combinations(teams, 2), start=1):
            rows.append(
                {
                    "match_id": f"{group}{idx}",
                    "date": pd.NaT,
                    "group": group,
                    "home_team": home,
                    "away_team": away,
                    "neutral": True,
                    "home_score": np.nan,
                    "away_score": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _third_slot_rules() -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {}
    for match_id, _home_slot, away_slot in R32_MATCHES:
        if away_slot.startswith("3"):
            rules[match_id] = away_slot[1:].split("/")
    return rules


def _assign_third_place_groups(qualified_groups: list[str]) -> dict[str, str]:
    rules = _third_slot_rules()
    qualified = set(qualified_groups)
    ordered_slots = sorted(
        rules,
        key=lambda slot: len([group for group in rules[slot] if group in qualified]),
    )
    assignment: dict[str, str] = {}
    used: set[str] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(ordered_slots):
            return True
        slot = ordered_slots[pos]
        candidates = [group for group in rules[slot] if group in qualified and group not in used]
        for group in candidates:
            assignment[slot] = group
            used.add(group)
            if backtrack(pos + 1):
                return True
            used.remove(group)
            assignment.pop(slot, None)
        return False

    if not backtrack(0):
        remaining = [group for group in qualified_groups if group not in used]
        for slot in ordered_slots:
            if slot not in assignment and remaining:
                assignment[slot] = remaining.pop(0)
    return assignment


def _resolve_slot(
    slot: str,
    slot_map: dict[str, str],
    third_assignment: dict[str, str],
    match_id: str,
) -> str:
    if slot.startswith("3") and "/" in slot:
        group = third_assignment[match_id]
        return slot_map[f"3{group}"]
    return slot_map[slot]


def simulate_group_stage(
    predictor: MatchPredictor,
    groups: dict[str, list[str]],
    rng: np.random.Generator,
    fixtures: pd.DataFrame | None = None,
    state=None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, str], list[SimulatedMatch]]:
    if predictor.state is None:
        raise RuntimeError("Predictor is not fitted")
    active_state = state or predictor.state.clone()
    table = _blank_table(groups)
    group_fixtures = fixtures.copy() if fixtures is not None else _default_group_fixtures(groups)
    if "date" in group_fixtures.columns:
        group_fixtures = group_fixtures.sort_values(["date", "match_id"], na_position="last")

    matches: list[SimulatedMatch] = []
    for row in group_fixtures.itertuples(index=False):
        group = str(row.group).strip().upper()
        if group not in groups:
            continue
        home_team = normalize_team_name(row.home_team)
        away_team = normalize_team_name(row.away_team)
        if home_team not in table[group] or away_team not in table[group]:
            continue
        has_actual = not pd.isna(row.home_score) and not pd.isna(row.away_score)
        if has_actual:
            home_score = int(row.home_score)
            away_score = int(row.away_score)
            winner = home_team if home_score > away_score else away_team if away_score > home_score else ""
        else:
            home_score, away_score, winner = predictor.sample_score(
                home_team,
                away_team,
                rng,
                date=row.date if hasattr(row, "date") else None,
                tournament="FIFA World Cup",
                neutral=bool(row.neutral) if hasattr(row, "neutral") else True,
                state=active_state,
                knockout=False,
            )
        _apply_group_result(table, group, home_team, away_team, home_score, away_score)
        predictor.update_state_with_match(
            active_state,
            home_team,
            away_team,
            home_score,
            away_score,
            date=row.date if hasattr(row, "date") else None,
            tournament="FIFA World Cup",
            neutral=bool(row.neutral) if hasattr(row, "neutral") else True,
        )
        matches.append(
            SimulatedMatch(
                match_id=str(row.match_id),
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                winner=winner,
                stage="group",
            )
        )

    standings = _standings(table, rng)
    slot_map: dict[str, str] = {}
    third_rows = []
    for group, rows in standings.items():
        slot_map[f"1{group}"] = rows[0]["team"]
        slot_map[f"2{group}"] = rows[1]["team"]
        slot_map[f"3{group}"] = rows[2]["team"]
        third_rows.append(rows[2])

    third_rows.sort(
        key=lambda r: (r["points"], r["gd"], r["gf"], r["wins"], r["tie_noise"]),
        reverse=True,
    )
    qualified_third_groups = [row["group"] for row in third_rows[:8]]
    third_assignment = _assign_third_place_groups(qualified_third_groups)
    return standings, slot_map, third_assignment, matches


def _play_knockout_match(
    predictor: MatchPredictor,
    state,
    rng: np.random.Generator,
    match_id: str,
    home_team: str,
    away_team: str,
    stage: str,
) -> SimulatedMatch:
    home_score, away_score, winner = predictor.sample_score(
        home_team,
        away_team,
        rng,
        tournament="FIFA World Cup",
        neutral=True,
        state=state,
        knockout=True,
    )
    if not winner:
        winner = home_team if home_score > away_score else away_team
    predictor.update_state_with_match(
        state,
        home_team,
        away_team,
        home_score,
        away_score,
        tournament="FIFA World Cup",
        neutral=True,
    )
    return SimulatedMatch(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        winner=winner,
        stage=stage,
    )


def simulate_knockout_stage(
    predictor: MatchPredictor,
    slot_map: dict[str, str],
    third_assignment: dict[str, str],
    rng: np.random.Generator,
    state=None,
) -> tuple[dict[str, list[str]], list[SimulatedMatch]]:
    if predictor.state is None:
        raise RuntimeError("Predictor is not fitted")
    active_state = state or predictor.state.clone()
    winners: dict[str, str] = {}
    reached: dict[str, list[str]] = {
        "round_of_16": [],
        "quarterfinal": [],
        "semifinal": [],
        "final": [],
        "champion": [],
    }
    matches: list[SimulatedMatch] = []

    for match_id, home_slot, away_slot in R32_MATCHES:
        home_team = _resolve_slot(home_slot, slot_map, third_assignment, match_id)
        away_team = _resolve_slot(away_slot, slot_map, third_assignment, match_id)
        match = _play_knockout_match(
            predictor, active_state, rng, match_id, home_team, away_team, "round_of_32"
        )
        winners[match_id] = match.winner
        reached["round_of_16"].append(match.winner)
        matches.append(match)

    for stage, round_matches in NEXT_ROUNDS.items():
        for match_id, left_id, right_id in round_matches:
            match = _play_knockout_match(
                predictor,
                active_state,
                rng,
                match_id,
                winners[left_id],
                winners[right_id],
                stage,
            )
            winners[match_id] = match.winner
            matches.append(match)
            if stage == "round_of_16":
                reached["quarterfinal"].append(match.winner)
            elif stage == "quarterfinal":
                reached["semifinal"].append(match.winner)
            elif stage == "semifinal":
                reached["final"].append(match.winner)
            elif stage == "final":
                reached["champion"].append(match.winner)

    return reached, matches


def simulate_tournament(
    predictor: MatchPredictor,
    groups: dict[str, list[str]],
    fixtures: pd.DataFrame | None = None,
    n_simulations: int = 5000,
    seed: int | None = 7,
) -> pd.DataFrame:
    """Run Monte Carlo World Cup simulations.

    The group-stage qualification rule matches the 48-team format: top two in
    each group plus the best eight third-place teams. Third-place placement into
    the Round of 32 is approximated from FIFA's listed eligible slots unless a
    caller later replaces this module with the full Annex C allocation matrix.
    """

    if predictor.state is None:
        raise RuntimeError("Predictor is not fitted")
    rng = np.random.default_rng(seed)
    teams = groups_to_frame(groups)
    team_to_group = dict(zip(teams["team"], teams["group"], strict=False))
    counts: dict[str, Counter[str]] = defaultdict(Counter)

    for _ in range(int(n_simulations)):
        state = predictor.state.clone()
        standings, slot_map, third_assignment, _group_matches = simulate_group_stage(
            predictor, groups, rng, fixtures=fixtures, state=state
        )
        round_of_32_teams = set()
        for group, rows in standings.items():
            round_of_32_teams.add(rows[0]["team"])
            round_of_32_teams.add(rows[1]["team"])
        for group in set(third_assignment.values()):
            round_of_32_teams.add(slot_map[f"3{group}"])
        for team in round_of_32_teams:
            counts[team]["round_of_32"] += 1

        reached, _knockout_matches = simulate_knockout_stage(
            predictor, slot_map, third_assignment, rng, state=state
        )
        for stage, stage_teams in reached.items():
            for team in stage_teams:
                counts[team][stage] += 1

    rows = []
    for team in sorted(team_to_group):
        row = {
            "team": team,
            "group": team_to_group[team],
        }
        for stage in [
            "round_of_32",
            "round_of_16",
            "quarterfinal",
            "semifinal",
            "final",
            "champion",
        ]:
            row[f"{stage}_probability"] = counts[team][stage] / float(n_simulations)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("champion_probability", ascending=False).reset_index(drop=True)
