from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Deque

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "elo_diff",
    "abs_elo_diff",
    "home_elo",
    "away_elo",
    "home_form_ppg",
    "away_form_ppg",
    "form_ppg_diff",
    "home_gd_avg",
    "away_gd_avg",
    "gd_avg_diff",
    "home_gf_avg",
    "away_gf_avg",
    "home_ga_avg",
    "away_ga_avg",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "neutral",
    "home_field",
    "tournament_weight",
    "is_world_cup",
    "is_qualifier",
]


def tournament_weight(tournament: object) -> float:
    name = str(tournament or "").strip().lower()
    if "world cup" in name and ("qualification" in name or "qualifier" in name):
        return 40.0
    if "fifa world cup" in name or name == "world cup":
        return 60.0
    if any(key in name for key in ["euro", "copa america", "africa cup", "asian cup"]):
        return 50.0
    if "qualification" in name or "qualifier" in name:
        return 40.0
    if "nations league" in name:
        return 30.0
    if "friendly" in name:
        return 20.0
    return 30.0


def is_world_cup(tournament: object) -> int:
    name = str(tournament or "").strip().lower()
    return int("world cup" in name and "qualification" not in name)


def is_qualifier(tournament: object) -> int:
    name = str(tournament or "").strip().lower()
    return int("qualification" in name or "qualifier" in name)


def result_label(home_score: int | float, away_score: int | float) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def label_name(label: int) -> str:
    return {0: "home_win", 1: "draw", 2: "away_win"}[int(label)]


def goal_multiplier(goal_diff: int | float) -> float:
    margin = abs(float(goal_diff))
    if margin < 2:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


@dataclass
class TeamForm:
    maxlen: int = 10
    matches: Deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    last_date: pd.Timestamp | None = None

    def clone(self) -> "TeamForm":
        copied = TeamForm(maxlen=self.maxlen)
        copied.matches = deque((dict(row) for row in self.matches), maxlen=self.maxlen)
        copied.last_date = self.last_date
        return copied

    def add(self, date: pd.Timestamp | None, goals_for: int, goals_against: int) -> None:
        if goals_for > goals_against:
            points = 3
            win = 1
        elif goals_for == goals_against:
            points = 1
            win = 0
        else:
            points = 0
            win = 0
        self.matches.append(
            {
                "points": float(points),
                "goals_for": float(goals_for),
                "goals_against": float(goals_against),
                "goal_diff": float(goals_for - goals_against),
                "win": float(win),
            }
        )
        if date is not None and not pd.isna(date):
            self.last_date = pd.Timestamp(date)

    def _avg(self, key: str, default: float) -> float:
        if not self.matches:
            return default
        return float(np.mean([row[key] for row in self.matches]))

    @property
    def ppg(self) -> float:
        return self._avg("points", 1.0)

    @property
    def goal_diff(self) -> float:
        return self._avg("goal_diff", 0.0)

    @property
    def goals_for(self) -> float:
        return self._avg("goals_for", 1.1)

    @property
    def goals_against(self) -> float:
        return self._avg("goals_against", 1.1)

    def rest_days(self, date: pd.Timestamp | None) -> float:
        if date is None or pd.isna(date) or self.last_date is None:
            return 21.0
        days = (pd.Timestamp(date) - self.last_date).days
        return float(np.clip(days, 0, 60))


@dataclass
class RatingState:
    default_elo: float = 1500.0
    home_advantage: float = 65.0
    form_window: int = 10
    ratings: dict[str, float] = field(default_factory=dict)
    # Plain dict (NOT defaultdict with a local lambda, which would make RatingState
    # -- and therefore MatchPredictor.save() -- unpicklable). _form() get-or-creates.
    forms: dict[str, TeamForm] = field(init=False)

    def __post_init__(self) -> None:
        self.forms = {}

    def _form(self, team: str) -> TeamForm:
        """Get the team's form, creating an empty one on first reference."""
        form = self.forms.get(team)
        if form is None:
            form = TeamForm(maxlen=self.form_window)
            self.forms[team] = form
        return form

    def clone(self) -> "RatingState":
        copied = RatingState(
            default_elo=self.default_elo,
            home_advantage=self.home_advantage,
            form_window=self.form_window,
        )
        copied.ratings = dict(self.ratings)
        copied.forms = {team: form.clone() for team, form in self.forms.items()}
        return copied

    def rating(self, team: str) -> float:
        return float(self.ratings.get(team, self.default_elo))

    def expected_home_result(self, home_team: str, away_team: str, neutral: bool) -> float:
        home_rating = self.rating(home_team) + (0.0 if neutral else self.home_advantage)
        away_rating = self.rating(away_team)
        return float(1.0 / (1.0 + 10.0 ** (-(home_rating - away_rating) / 400.0)))

    def features(
        self,
        home_team: str,
        away_team: str,
        date: pd.Timestamp | None,
        tournament: str,
        neutral: bool = True,
    ) -> dict[str, float]:
        home_elo = self.rating(home_team)
        away_elo = self.rating(away_team)
        home_form = self._form(home_team)
        away_form = self._form(away_team)
        elo_diff = home_elo - away_elo
        home_rest = home_form.rest_days(date)
        away_rest = away_form.rest_days(date)
        return {
            "elo_diff": elo_diff,
            "abs_elo_diff": abs(elo_diff),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_form_ppg": home_form.ppg,
            "away_form_ppg": away_form.ppg,
            "form_ppg_diff": home_form.ppg - away_form.ppg,
            "home_gd_avg": home_form.goal_diff,
            "away_gd_avg": away_form.goal_diff,
            "gd_avg_diff": home_form.goal_diff - away_form.goal_diff,
            "home_gf_avg": home_form.goals_for,
            "away_gf_avg": away_form.goals_for,
            "home_ga_avg": home_form.goals_against,
            "away_ga_avg": away_form.goals_against,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_days_diff": home_rest - away_rest,
            "neutral": float(bool(neutral)),
            "home_field": float(not bool(neutral)),
            "tournament_weight": tournament_weight(tournament),
            "is_world_cup": float(is_world_cup(tournament)),
            "is_qualifier": float(is_qualifier(tournament)),
        }

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        date: pd.Timestamp | None,
        tournament: str,
        neutral: bool = True,
    ) -> None:
        expected = self.expected_home_result(home_team, away_team, neutral)
        if home_score > away_score:
            actual = 1.0
        elif home_score == away_score:
            actual = 0.5
        else:
            actual = 0.0
        k = tournament_weight(tournament)
        margin = goal_multiplier(home_score - away_score)
        delta = k * margin * (actual - expected)
        self.ratings[home_team] = self.rating(home_team) + delta
        self.ratings[away_team] = self.rating(away_team) - delta
        self._form(home_team).add(date, int(home_score), int(away_score))
        self._form(away_team).add(date, int(away_score), int(home_score))


def build_training_frame(results: pd.DataFrame) -> tuple[pd.DataFrame, RatingState]:
    rows: list[dict[str, object]] = []
    state = RatingState()
    ordered = results.sort_values("date").reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        features = state.features(
            home_team=row.home_team,
            away_team=row.away_team,
            date=row.date,
            tournament=row.tournament,
            neutral=bool(row.neutral),
        )
        label = result_label(row.home_score, row.away_score)
        rows.append(
            {
                **features,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_score": int(row.home_score),
                "away_score": int(row.away_score),
                "label": label,
                "label_name": label_name(label),
            }
        )
        state.update_match(
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=int(row.home_score),
            away_score=int(row.away_score),
            date=row.date,
            tournament=row.tournament,
            neutral=bool(row.neutral),
        )
    return pd.DataFrame(rows), state
