"""
Competition rules: standings, configurable tie-breakers, knockout-tie settings,
tournament-format definitions, and the FIFA World Cup 2026 bracket.

This module is pure data + pure functions (no model / no RNG state of its own
beyond what callers pass in). The simulation engines (group_stage, knockout,
league, tournament) build on it so that formats are *declared* here and merely
*executed* there. Tournament assumptions are made explicit (a project
non-negotiable): where the official rules are approximated, the format carries
`notes` / `approximate=True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import numpy as np

from ..data.aliases import normalize_team_name

# --------------------------------------------------------------------------- #
# Tie-breakers
# --------------------------------------------------------------------------- #
# Supported tie-break keys, applied in order until the tie is resolved.
TIEBREAKER_KEYS: tuple[str, ...] = (
    "points",
    "goal_difference",
    "goals_for",
    "wins",
    "head_to_head_points",
    "head_to_head_goal_difference",
    "head_to_head_goals_for",
    "fair_play",
    "random_draw",
)

# FIFA group-stage order (simplified but faithful in spirit). Fair play and the
# full drawing-of-lots procedure are approximated by random_draw at the tail.
FIFA_GROUP_TIEBREAKERS: tuple[str, ...] = (
    "points",
    "goal_difference",
    "goals_for",
    "head_to_head_points",
    "head_to_head_goal_difference",
    "head_to_head_goals_for",
    "fair_play",
    "random_draw",
)

# Generic league order (domestic leagues differ; this is a common default).
LEAGUE_TIEBREAKERS: tuple[str, ...] = (
    "points",
    "goal_difference",
    "goals_for",
    "wins",
    "random_draw",
)


def new_standings_row(group: str, team: str) -> dict[str, Any]:
    """A blank standings row for one team."""
    return {
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
        "fair_play": 0,  # cumulative disciplinary points (negative is better)
        "_noise": 0.0,
    }


def apply_result(
    home_row: dict[str, Any],
    away_row: dict[str, Any],
    home_score: int,
    away_score: int,
    *,
    points_win: int = 3,
    points_draw: int = 1,
) -> None:
    """Update two standings rows in place with one result."""
    home_row["played"] += 1
    away_row["played"] += 1
    home_row["gf"] += home_score
    home_row["ga"] += away_score
    away_row["gf"] += away_score
    away_row["ga"] += home_score
    home_row["gd"] = home_row["gf"] - home_row["ga"]
    away_row["gd"] = away_row["gf"] - away_row["ga"]
    if home_score > away_score:
        home_row["wins"] += 1
        home_row["points"] += points_win
        away_row["losses"] += 1
    elif away_score > home_score:
        away_row["wins"] += 1
        away_row["points"] += points_win
        home_row["losses"] += 1
    else:
        home_row["draws"] += 1
        away_row["draws"] += 1
        home_row["points"] += points_draw
        away_row["points"] += points_draw


def _head_to_head_stats(
    teams: set[str], matches: list[tuple[str, str, int, int]]
) -> dict[str, dict[str, int]]:
    """Mini-table among `teams` only, from the matches they played each other."""
    stats = {t: {"points": 0, "gd": 0, "gf": 0} for t in teams}
    for home, away, hs, ascore in matches:
        if home not in teams or away not in teams:
            continue
        stats[home]["gf"] += hs
        stats[away]["gf"] += ascore
        stats[home]["gd"] += hs - ascore
        stats[away]["gd"] += ascore - hs
        if hs > ascore:
            stats[home]["points"] += 3
        elif ascore > hs:
            stats[away]["points"] += 3
        else:
            stats[home]["points"] += 1
            stats[away]["points"] += 1
    return stats


def rank_group(
    rows: list[dict[str, Any]],
    matches: list[tuple[str, str, int, int]],
    tiebreakers: tuple[str, ...] = FIFA_GROUP_TIEBREAKERS,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Return rows sorted best-first using the configured tie-break order.

    `matches` is the list of (home, away, home_score, away_score) played within
    the group, needed for the head-to-head keys. `rng` supplies the random_draw
    tail-breaker; if None a fixed default generator is used.
    """
    rng = rng or np.random.default_rng(0)
    rows = [dict(r) for r in rows]
    for r in rows:
        r["_noise"] = float(rng.random())

    def sort_key(r: dict[str, Any]) -> tuple:
        key: list[float] = []
        for tb in tiebreakers:
            if tb == "points":
                key.append(r["points"])
            elif tb == "goal_difference":
                key.append(r["gd"])
            elif tb == "goals_for":
                key.append(r["gf"])
            elif tb == "wins":
                key.append(r["wins"])
            elif tb == "fair_play":
                key.append(-r.get("fair_play", 0))  # fewer cards is better
            elif tb == "random_draw":
                key.append(r["_noise"])
            # head_to_head_* handled below as a contextual pass
        return tuple(key)

    # Primary sort on non-h2h keys.
    rows.sort(key=sort_key, reverse=True)

    # Resolve ties that the primary keys could not, using head-to-head where the
    # config asks for it. We re-rank only within blocks tied on points/gd/gf.
    if "head_to_head_points" in tiebreakers or "head_to_head_goal_difference" in tiebreakers:
        rows = _apply_head_to_head(rows, matches, tiebreakers)

    for idx, r in enumerate(rows, start=1):
        r["rank"] = idx
    return rows


def _apply_head_to_head(
    rows: list[dict[str, Any]],
    matches: list[tuple[str, str, int, int]],
    tiebreakers: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Within blocks tied on (points, gd, gf), re-order by head-to-head."""
    out: list[dict[str, Any]] = []
    i = 0
    n = len(rows)
    while i < n:
        j = i + 1
        while (
            j < n
            and rows[j]["points"] == rows[i]["points"]
            and rows[j]["gd"] == rows[i]["gd"]
            and rows[j]["gf"] == rows[i]["gf"]
        ):
            j += 1
        block = rows[i:j]
        if len(block) > 1:
            teams = {r["team"] for r in block}
            h2h = _head_to_head_stats(teams, matches)

            def h2h_key(r: dict[str, Any]) -> tuple:
                # Continue the FIFA chain INSIDE the tied block: head-to-head
                # points -> GD -> goals-for, then fair play, then drawing of
                # lots. Previously this jumped straight to noise after the first
                # two h2h keys, discarding h2h goals-for and fair play.
                s = h2h[r["team"]]
                key: list[float] = []
                for tb in tiebreakers:
                    if tb == "head_to_head_points":
                        key.append(s["points"])
                    elif tb == "head_to_head_goal_difference":
                        key.append(s["gd"])
                    elif tb == "head_to_head_goals_for":
                        key.append(s["gf"])
                    elif tb == "fair_play":
                        key.append(-r.get("fair_play", 0))
                    elif tb == "random_draw":
                        key.append(r["_noise"])
                return tuple(key)

            block.sort(key=h2h_key, reverse=True)
        out.extend(block)
        i = j
    return out


# --------------------------------------------------------------------------- #
# Knockout-tie settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KnockoutTie:
    """How a knockout tie is resolved."""

    two_leg: bool = False
    away_goals: bool = False  # default OFF (abolished in most modern comps)
    extra_time: bool = True
    penalties: bool = True
    et_fraction: float = 1.0 / 3.0  # 30 ET minutes ~ 1/3 of a 90-minute rate
    # A penalty shootout is empirically close to a coin flip -- the favourite's
    # open-play edge barely carries over. `shootout_skill` is the FRACTION of that
    # edge retained: 0.0 = pure 50/50, 1.0 = full open-play win prob (unrealistic).
    # Default 0.25 keeps a slight favourite tilt while reflecting the near-randomness.
    shootout_skill: float = 0.25


# --------------------------------------------------------------------------- #
# Tournament format
# --------------------------------------------------------------------------- #
@dataclass
class BracketMatch:
    """One knockout slot. home_src / away_src are either group slot codes
    (e.g. '1A', '2B', '3ABCD' for a best-third) or prior match ids ('M73')."""

    match_id: str
    home_src: str
    away_src: str
    stage: str


@dataclass
class TournamentFormat:
    name: str
    kind: str = "groups_knockout"  # or "league" or "knockout"
    n_groups: int = 0
    teams_per_group: int = 0
    advance_per_group: int = 2
    n_best_third: int = 0
    group_tiebreakers: tuple[str, ...] = FIFA_GROUP_TIEBREAKERS
    knockout: KnockoutTie = field(default_factory=KnockoutTie)
    bracket: list[BracketMatch] = field(default_factory=list)
    # Ordered knockout stage names from first to final.
    stage_order: tuple[str, ...] = ()
    third_place_allocator: Callable[[list[str]], dict[str, str]] | None = None
    approximate: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- #
# FIFA World Cup 2026 bracket (48 teams, 12 groups, top 2 + best 8 thirds)
# --------------------------------------------------------------------------- #
# Round of 32. Away-slot codes beginning with '3' are best-third placeholders
# whose eligible groups are listed (e.g. '3A/B/C/D/F').
WC2026_R32: list[tuple[str, str, str]] = [
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

WC2026_NEXT_ROUNDS: dict[str, list[tuple[str, str, str]]] = {
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

WC2026_STAGE_ORDER: tuple[str, ...] = (
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
)


def wc2026_third_slot_rules() -> dict[str, list[str]]:
    """match_id -> eligible group letters for its best-third slot."""
    rules: dict[str, list[str]] = {}
    for match_id, _home, away in WC2026_R32:
        if away.startswith("3"):
            rules[match_id] = away[1:].split("/")
    return rules


def wc2026_third_place_assignment(qualified_groups: list[str]) -> dict[str, str]:
    """Assign the 8 best-third groups to R32 best-third slots.

    FIFA publishes a fixed allocation matrix (Annex C) keyed on *which* group
    letters qualify. This is a constraint-satisfaction approximation: it finds a
    valid assignment respecting each slot's eligible-group list. It is correct in
    that every assigned group is eligible for its slot and each group is used
    once; it is approximate in that it does not reproduce FIFA's exact published
    mapping for every one of the 495 combinations.
    """
    rules = wc2026_third_slot_rules()
    qualified = list(qualified_groups)
    qualified_set = set(qualified)
    ordered_slots = sorted(
        rules,
        key=lambda slot: len([g for g in rules[slot] if g in qualified_set]),
    )
    assignment: dict[str, str] = {}
    used: set[str] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(ordered_slots):
            return True
        slot = ordered_slots[pos]
        for g in rules[slot]:
            if g in qualified_set and g not in used:
                assignment[slot] = g
                used.add(g)
                if backtrack(pos + 1):
                    return True
                used.remove(g)
                assignment.pop(slot, None)
        return False

    if not backtrack(0):
        remaining = [g for g in qualified if g not in used]
        for slot in ordered_slots:
            if slot not in assignment and remaining:
                assignment[slot] = remaining.pop(0)
    return assignment


def world_cup_2026_format() -> TournamentFormat:
    """The 48-team, 12-group World Cup 2026 format."""
    bracket: list[BracketMatch] = [
        BracketMatch(m, h, a, "round_of_32") for m, h, a in WC2026_R32
    ]
    for stage, matches in WC2026_NEXT_ROUNDS.items():
        for m, h, a in matches:
            bracket.append(BracketMatch(m, h, a, stage))
    return TournamentFormat(
        name="world_cup_2026",
        kind="groups_knockout",
        n_groups=12,
        teams_per_group=4,
        advance_per_group=2,
        n_best_third=8,
        group_tiebreakers=FIFA_GROUP_TIEBREAKERS,
        knockout=KnockoutTie(two_leg=False, away_goals=False),
        bracket=bracket,
        stage_order=WC2026_STAGE_ORDER,
        third_place_allocator=wc2026_third_place_assignment,
        approximate=True,
        notes=(
            "Third-place allocation is a constraint-satisfaction approximation "
            "of FIFA Annex C; group tie-breakers omit the full drawing-of-lots "
            "procedure."
        ),
    )


# --------------------------------------------------------------------------- #
# FIFA World Cup legacy format (32 teams, 8 groups of 4; 1998-2022)
# --------------------------------------------------------------------------- #
# Standard 32-team Round of 16: group winners cross with runners-up.
WC_LEGACY_R16: list[tuple[str, str, str]] = [
    ("R49", "1A", "2B"),
    ("R50", "1C", "2D"),
    ("R51", "1E", "2F"),
    ("R52", "1G", "2H"),
    ("R53", "1B", "2A"),
    ("R54", "1D", "2C"),
    ("R55", "1F", "2E"),
    ("R56", "1H", "2G"),
]
WC_LEGACY_NEXT: dict[str, list[tuple[str, str, str]]] = {
    "quarterfinal": [
        ("R57", "R49", "R50"),
        ("R58", "R53", "R54"),
        ("R59", "R51", "R52"),
        ("R60", "R55", "R56"),
    ],
    "semifinal": [
        ("R61", "R57", "R59"),
        ("R62", "R58", "R60"),
    ],
    "final": [("R64", "R61", "R62")],
}
WC_LEGACY_STAGE_ORDER: tuple[str, ...] = (
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
)


def world_cup_legacy_format() -> TournamentFormat:
    """The 32-team, 8-group World Cup format used from 1998 to 2022.

    Top two per group advance straight to the Round of 16 (no third-place
    qualifiers). Used to backtest the match/simulation stack against the seven
    editions 1998-2022 (the "current regime" before the 48-team 2026 expansion).
    """
    bracket = [BracketMatch(m, h, a, "round_of_16") for m, h, a in WC_LEGACY_R16]
    for stage, matches in WC_LEGACY_NEXT.items():
        for m, h, a in matches:
            bracket.append(BracketMatch(m, h, a, stage))
    return TournamentFormat(
        name="world_cup_legacy",
        kind="groups_knockout",
        n_groups=8,
        teams_per_group=4,
        advance_per_group=2,
        n_best_third=0,
        group_tiebreakers=FIFA_GROUP_TIEBREAKERS,
        knockout=KnockoutTie(two_leg=False, away_goals=False),
        bracket=bracket,
        stage_order=WC_LEGACY_STAGE_ORDER,
        approximate=True,
        notes=(
            "32-team format (1998-2022): 8 groups of 4, top two advance to a "
            "16-team single-elimination bracket. Group tie-breakers omit the full "
            "drawing-of-lots procedure."
        ),
    )


# --------------------------------------------------------------------------- #
# Generic builders (Euros / Copa / custom groups, leagues, single-elim cups)
# --------------------------------------------------------------------------- #
def single_elimination_bracket(
    n_teams: int, stage_prefix: str = "K"
) -> tuple[list[BracketMatch], tuple[str, ...]]:
    """Seeded single-elimination bracket over `n_teams` (power of two).

    First-round slots are seed codes 'S1'..'Sn' paired 1-vs-n, 2-vs-(n-1) ...
    Returns (bracket, stage_order). Stage names are round_of_<k> ... final.
    """
    if n_teams < 2 or (n_teams & (n_teams - 1)) != 0:
        raise ValueError("single_elimination_bracket needs a power-of-two team count")
    stage_name = _stage_name_for(n_teams)
    bracket: list[BracketMatch] = []
    stage_order: list[str] = []
    # Round 1 from seeds.
    seeds = [f"S{i}" for i in range(1, n_teams + 1)]
    pairings = [(seeds[i], seeds[n_teams - 1 - i]) for i in range(n_teams // 2)]
    counter = 1
    prev_ids: list[str] = []
    stage_order.append(stage_name)
    for home, away in pairings:
        mid = f"{stage_prefix}{counter}"
        bracket.append(BracketMatch(mid, home, away, stage_name))
        prev_ids.append(mid)
        counter += 1
    # Subsequent rounds.
    current = prev_ids
    while len(current) > 1:
        stage_name = _stage_name_for(len(current))
        stage_order.append(stage_name)
        nxt: list[str] = []
        for i in range(0, len(current), 2):
            mid = f"{stage_prefix}{counter}"
            bracket.append(BracketMatch(mid, current[i], current[i + 1], stage_name))
            nxt.append(mid)
            counter += 1
        current = nxt
    return bracket, tuple(stage_order)


def _stage_name_for(n_teams_in_round: int) -> str:
    return {
        2: "final",
        4: "semifinal",
        8: "quarterfinal",
    }.get(n_teams_in_round, f"round_of_{n_teams_in_round}")


def round_robin_fixtures(
    teams: list[str], double: bool = False
) -> list[tuple[str, str]]:
    """All (home, away) pairings for a round-robin among `teams`."""
    pairs = [(h, a) for h, a in combinations(teams, 2)]
    if double:
        pairs += [(a, h) for h, a in combinations(teams, 2)]
    return pairs


def league_format(name: str, tiebreakers: tuple[str, ...] = LEAGUE_TIEBREAKERS) -> TournamentFormat:
    return TournamentFormat(
        name=name,
        kind="league",
        group_tiebreakers=tiebreakers,
        notes="Single-table round-robin league.",
    )


# Registry of named formats (builders take no args).
FORMATS: dict[str, Callable[[], TournamentFormat]] = {
    "world_cup_2026": world_cup_2026_format,
    "world_cup_legacy": world_cup_legacy_format,
}


def get_format(name: str) -> TournamentFormat:
    if name not in FORMATS:
        raise KeyError(
            f"Unknown format '{name}'. Known: {sorted(FORMATS)}. "
            "Custom group/knockout tournaments can be defined via CSV configs."
        )
    return FORMATS[name]()
