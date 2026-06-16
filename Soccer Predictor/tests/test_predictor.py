from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from world_cup_predictor.model import MatchPredictor
from world_cup_predictor.tournament import simulate_tournament


def _teams() -> list[str]:
    return [f"Team {group}{idx}" for group in "ABCDEFGHIJKL" for idx in range(1, 5)]


def _groups() -> dict[str, list[str]]:
    return {
        group: [f"Team {group}{idx}" for idx in range(1, 5)]
        for group in "ABCDEFGHIJKL"
    }


def _results() -> pd.DataFrame:
    teams = _teams()
    strengths = {team: 1800 - idx * 12 for idx, team in enumerate(teams)}
    rows = []
    start = date(2023, 1, 1)
    for idx in range(140):
        home = teams[idx % len(teams)]
        away = teams[(idx * 7 + 11) % len(teams)]
        if home == away:
            away = teams[(idx * 7 + 12) % len(teams)]
        diff = strengths[home] - strengths[away]
        if idx % 7 == 0:
            home_score, away_score = 1, 1
        elif diff >= 0:
            home_score = 2 + int(diff > 250)
            away_score = int(idx % 5 == 0)
        else:
            home_score = int(idx % 6 == 0)
            away_score = 2 + int(diff < -250)
        rows.append(
            {
                "date": start + timedelta(days=idx * 5),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "Friendly" if idx % 4 else "FIFA World Cup qualification",
                "city": "Test City",
                "country": "Testland",
                "neutral": True,
            }
        )
    return pd.DataFrame(rows)


def test_match_prediction_probabilities_sum_to_one() -> None:
    predictor = MatchPredictor().fit(_results())
    prediction = predictor.predict_match("Team A1", "Team B2")

    total = prediction.home_win + prediction.draw + prediction.away_win
    assert abs(total - 1.0) < 1e-9
    assert prediction.expected_home_goals >= 0
    assert prediction.expected_away_goals >= 0
    assert prediction.scoreline_probabilities


def test_tournament_simulation_returns_champion_distribution() -> None:
    predictor = MatchPredictor().fit(_results())
    sims = simulate_tournament(predictor, _groups(), n_simulations=3, seed=42)

    assert len(sims) == 48
    assert abs(sims["champion_probability"].sum() - 1.0) < 1e-9
    assert sims["round_of_32_probability"].between(0, 1).all()


def test_fitted_match_predictor_pickles_and_round_trips(tmp_path) -> None:
    """Regression: RatingState.forms must stay picklable (no local-lambda
    defaultdict), so MatchPredictor.save()/load() works."""
    predictor = MatchPredictor().fit(_results())
    path = tmp_path / "predictor.pkl"
    predictor.save(path)  # would raise AttributeError if forms used a lambda
    restored = MatchPredictor.load(path)

    before = predictor.predict_match("Team A1", "Team B2")
    after = restored.predict_match("Team A1", "Team B2")
    assert abs(before.home_win - after.home_win) < 1e-12
    assert abs(before.draw - after.draw) < 1e-12
    assert abs(before.away_win - after.away_win) < 1e-12
