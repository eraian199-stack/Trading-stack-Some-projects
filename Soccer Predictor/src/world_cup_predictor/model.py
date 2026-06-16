from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from .data import load_results, normalize_team_name, standardize_results
from .ratings import FEATURE_COLUMNS, RatingState, build_training_frame


OUTCOME_COLUMNS = ["home_win", "draw", "away_win"]


def _as_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURE_COLUMNS].astype(float).to_numpy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def _poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
    lam = float(np.clip(lam, 0.05, 6.0))
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = np.exp(-lam)
    for goal in range(1, max_goals + 1):
        probs[goal] = probs[goal - 1] * lam / goal
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class SoftmaxOutcomeModel:
    learning_rate: float = 0.035
    l2: float = 0.015
    epochs: int = 900
    weights_: np.ndarray | None = None
    standardizer: Standardizer = field(default_factory=Standardizer)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftmaxOutcomeModel":
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two outcome classes to train a match model")
        x_scaled = self.standardizer.fit_transform(x)
        x_bias = np.c_[np.ones(len(x_scaled)), x_scaled]
        y_hot = _one_hot(y.astype(int), 3)
        weights = np.zeros((x_bias.shape[1], 3), dtype=float)
        for _ in range(self.epochs):
            probs = _softmax(x_bias @ weights)
            grad = (x_bias.T @ (probs - y_hot)) / len(x_bias)
            grad[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * grad
        self.weights_ = weights
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Outcome model is not fitted")
        x_scaled = self.standardizer.transform(x)
        x_bias = np.c_[np.ones(len(x_scaled)), x_scaled]
        return _softmax(x_bias @ self.weights_)


@dataclass
class PoissonGoalModel:
    learning_rate: float = 0.01
    l2: float = 0.01
    epochs: int = 700
    weights_: np.ndarray | None = None
    standardizer: Standardizer = field(default_factory=Standardizer)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PoissonGoalModel":
        x_scaled = self.standardizer.fit_transform(x)
        x_bias = np.c_[np.ones(len(x_scaled)), x_scaled]
        weights = np.zeros(x_bias.shape[1], dtype=float)
        weights[0] = float(np.log(np.clip(np.mean(y), 0.05, 5.0)))
        for _ in range(self.epochs):
            linear = np.clip(x_bias @ weights, -4.0, 2.4)
            mu = np.exp(linear)
            grad = (x_bias.T @ (mu - y)) / len(x_bias)
            grad[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * grad
        self.weights_ = weights
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Goal model is not fitted")
        x_scaled = self.standardizer.transform(x)
        x_bias = np.c_[np.ones(len(x_scaled)), x_scaled]
        return np.exp(np.clip(x_bias @ self.weights_, -4.0, 2.4))


@dataclass(frozen=True)
class MatchPrediction:
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float
    most_likely_score: str
    scoreline_probabilities: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
            "expected_home_goals": self.expected_home_goals,
            "expected_away_goals": self.expected_away_goals,
            "most_likely_score": self.most_likely_score,
        }


@dataclass
class MatchPredictor:
    outcome_model: SoftmaxOutcomeModel = field(default_factory=SoftmaxOutcomeModel)
    home_goal_model: PoissonGoalModel = field(default_factory=PoissonGoalModel)
    away_goal_model: PoissonGoalModel = field(default_factory=PoissonGoalModel)
    state: RatingState | None = None
    training_frame: pd.DataFrame | None = None
    training_summary: dict[str, Any] | None = None
    max_goals: int = 8

    def fit(self, results: pd.DataFrame | str | Path) -> "MatchPredictor":
        if not isinstance(results, pd.DataFrame):
            results = load_results(results)
        else:
            results = standardize_results(results)
        frame, state = build_training_frame(results)
        if len(frame) < 20:
            raise ValueError("Need at least 20 historical matches for a usable model")
        x = _as_matrix(frame)
        y = frame["label"].to_numpy(dtype=int)
        self.outcome_model.fit(x, y)
        self.home_goal_model.fit(x, frame["home_score"].to_numpy(dtype=float))
        self.away_goal_model.fit(x, frame["away_score"].to_numpy(dtype=float))
        self.state = state
        self.training_frame = frame
        self.training_summary = {
            "matches": int(len(frame)),
            "teams": int(len(set(frame["home_team"]) | set(frame["away_team"]))),
            "first_match": str(pd.Timestamp(frame["date"].min()).date()),
            "last_match": str(pd.Timestamp(frame["date"].max()).date()),
        }
        return self

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "MatchPredictor":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, MatchPredictor):
            raise TypeError("Pickle did not contain a MatchPredictor")
        return loaded

    def feature_frame(
        self,
        home_team: str,
        away_team: str,
        date: str | pd.Timestamp | None = None,
        tournament: str = "FIFA World Cup",
        neutral: bool = True,
        state: RatingState | None = None,
    ) -> pd.DataFrame:
        active_state = state or self.state
        if active_state is None:
            raise RuntimeError("Predictor is not fitted")
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)
        ts = pd.to_datetime(date, errors="coerce") if date is not None else None
        row = active_state.features(home_team, away_team, ts, tournament, neutral)
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        date: str | pd.Timestamp | None = None,
        tournament: str = "FIFA World Cup",
        neutral: bool = True,
        state: RatingState | None = None,
    ) -> MatchPrediction:
        features = self.feature_frame(
            home_team=home_team,
            away_team=away_team,
            date=date,
            tournament=tournament,
            neutral=neutral,
            state=state,
        )
        x = _as_matrix(features)
        probs = self.outcome_model.predict_proba(x)[0]
        home_lambda = float(self.home_goal_model.predict(x)[0])
        away_lambda = float(self.away_goal_model.predict(x)[0])
        matrix = self.score_matrix(home_lambda, away_lambda, probs)
        expected_home = float(
            sum(home * matrix[home, away] for home in range(matrix.shape[0]) for away in range(matrix.shape[1]))
        )
        expected_away = float(
            sum(away * matrix[home, away] for home in range(matrix.shape[0]) for away in range(matrix.shape[1]))
        )
        flat_idx = int(np.argmax(matrix))
        home_goals, away_goals = np.unravel_index(flat_idx, matrix.shape)
        top_scores = self.top_scorelines(matrix, limit=8)
        return MatchPrediction(
            home_team=normalize_team_name(home_team),
            away_team=normalize_team_name(away_team),
            home_win=float(probs[0]),
            draw=float(probs[1]),
            away_win=float(probs[2]),
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
            most_likely_score=f"{home_goals}-{away_goals}",
            scoreline_probabilities=top_scores,
        )

    def score_matrix(
        self,
        home_lambda: float,
        away_lambda: float,
        outcome_probs: np.ndarray,
    ) -> np.ndarray:
        home_pmf = _poisson_pmf(home_lambda, self.max_goals)
        away_pmf = _poisson_pmf(away_lambda, self.max_goals)
        matrix = np.outer(home_pmf, away_pmf)

        home_mask = np.fromfunction(lambda h, a: h > a, matrix.shape)
        draw_mask = np.fromfunction(lambda h, a: h == a, matrix.shape)
        away_mask = np.fromfunction(lambda h, a: h < a, matrix.shape)
        masks = [home_mask, draw_mask, away_mask]
        poisson_outcome = np.array([matrix[mask].sum() for mask in masks])
        tilted = matrix.copy()
        for idx, mask in enumerate(masks):
            if poisson_outcome[idx] > 1e-10:
                tilted[mask] *= outcome_probs[idx] / poisson_outcome[idx]
        tilted /= tilted.sum()
        return tilted

    @staticmethod
    def top_scorelines(matrix: np.ndarray, limit: int = 8) -> dict[str, float]:
        rows = []
        for home in range(matrix.shape[0]):
            for away in range(matrix.shape[1]):
                rows.append((float(matrix[home, away]), home, away))
        rows.sort(reverse=True)
        return {f"{home}-{away}": prob for prob, home, away in rows[:limit]}

    def sample_score(
        self,
        home_team: str,
        away_team: str,
        rng: np.random.Generator,
        date: str | pd.Timestamp | None = None,
        tournament: str = "FIFA World Cup",
        neutral: bool = True,
        state: RatingState | None = None,
        knockout: bool = False,
    ) -> tuple[int, int, str]:
        prediction = self.predict_match(
            home_team=home_team,
            away_team=away_team,
            date=date,
            tournament=tournament,
            neutral=neutral,
            state=state,
        )
        features = self.feature_frame(home_team, away_team, date, tournament, neutral, state)
        x = _as_matrix(features)
        probs = self.outcome_model.predict_proba(x)[0]
        matrix = self.score_matrix(
            float(self.home_goal_model.predict(x)[0]),
            float(self.away_goal_model.predict(x)[0]),
            probs,
        )
        idx = int(rng.choice(matrix.size, p=matrix.ravel()))
        home_goals, away_goals = np.unravel_index(idx, matrix.shape)
        winner = ""
        if knockout and home_goals == away_goals:
            no_draw = probs[[0, 2]]
            p_home = 0.5 if no_draw.sum() <= 0 else float(no_draw[0] / no_draw.sum())
            winner = prediction.home_team if rng.random() < p_home else prediction.away_team
        elif home_goals > away_goals:
            winner = prediction.home_team
        elif away_goals > home_goals:
            winner = prediction.away_team
        return int(home_goals), int(away_goals), winner

    def update_state_with_match(
        self,
        state: RatingState,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        date: str | pd.Timestamp | None = None,
        tournament: str = "FIFA World Cup",
        neutral: bool = True,
    ) -> None:
        ts = pd.to_datetime(date, errors="coerce") if date is not None else None
        state.update_match(
            normalize_team_name(home_team),
            normalize_team_name(away_team),
            int(home_score),
            int(away_score),
            ts,
            tournament,
            neutral,
        )


def evaluate_holdout(
    results: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
) -> dict[str, float]:
    cutoff = pd.Timestamp(cutoff_date)
    train = results[results["date"] < cutoff]
    test = results[results["date"] >= cutoff]
    if len(train) < 20 or len(test) < 1:
        raise ValueError("Holdout split needs at least 20 train matches and 1 test match")
    predictor = MatchPredictor().fit(train)
    rows = []
    state = predictor.state.clone()
    for row in test.sort_values("date").itertuples(index=False):
        pred = predictor.predict_match(
            row.home_team,
            row.away_team,
            row.date,
            row.tournament,
            bool(row.neutral),
            state=state,
        )
        actual = 0 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 2
        rows.append(
            {
                "actual": actual,
                "home_win": pred.home_win,
                "draw": pred.draw,
                "away_win": pred.away_win,
            }
        )
        predictor.update_state_with_match(
            state,
            row.home_team,
            row.away_team,
            int(row.home_score),
            int(row.away_score),
            row.date,
            row.tournament,
            bool(row.neutral),
        )
    scored = pd.DataFrame(rows)
    probs = scored[OUTCOME_COLUMNS].to_numpy()
    actual = scored["actual"].to_numpy(dtype=int)
    picked = probs.argmax(axis=1)
    eps = 1e-12
    log_loss = -np.mean(np.log(np.clip(probs[np.arange(len(actual)), actual], eps, 1.0)))
    brier = float(np.mean(np.sum((_one_hot(actual, 3) - probs) ** 2, axis=1)))
    return {
        "matches": float(len(scored)),
        "accuracy": float(np.mean(picked == actual)),
        "log_loss": float(log_loss),
        "brier": brier,
    }
