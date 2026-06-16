"""
Evaluation-layer tests: proper scoring metrics, walk-forward shapes, and the
value-betting policy.

Headline invariants:
  * perfect probabilities give ~0 log loss / RPS / Brier and 100% accuracy;
  * walk_forward returns aligned (probs, outcomes, test_df) shapes from an
    expanding-window schedule;
  * betting with best_edge_only=True places AT MOST ONE bet per match.

Folds and sim counts are kept tiny so the suite is fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soccer_predictor.data import synthetic
from soccer_predictor.evaluation import metrics
from soccer_predictor.evaluation.betting import betting_backtest, max_drawdown
from soccer_predictor.evaluation.walk_forward import (
    evaluate_model,
    walk_forward,
    walk_forward_goals_markets,
)
from soccer_predictor.models.dixon_coles import DixonColes


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_perfect_probabilities_score_near_zero():
    outcomes = np.array(["H", "D", "A", "H", "A"])
    # One-hot "perfect" predictions.
    idx = {"H": 0, "D": 1, "A": 2}
    probs = np.zeros((len(outcomes), 3))
    for i, o in enumerate(outcomes):
        probs[i, idx[o]] = 1.0
    assert metrics.log_loss(probs, outcomes) < 1e-9
    assert metrics.rps(probs, outcomes) < 1e-12
    assert metrics.brier(probs, outcomes) < 1e-12
    assert metrics.accuracy(probs, outcomes) == 1.0


def test_uniform_log_loss_is_ln3():
    outcomes = np.array(["H", "D", "A"])
    probs = np.full((3, 3), 1.0 / 3.0)
    assert abs(metrics.log_loss(probs, outcomes) - np.log(3)) < 1e-9


def test_rps_respects_class_order():
    """Predicting Away when Home occurred (distance 2) is worse than predicting
    Draw (distance 1) under RPS."""
    outcomes = np.array(["H"])
    p_draw_guess = np.array([[0.0, 1.0, 0.0]])  # adjacent miss
    p_away_guess = np.array([[0.0, 0.0, 1.0]])  # far miss
    assert metrics.rps(p_draw_guess, outcomes) < metrics.rps(p_away_guess, outcomes)


def test_all_metrics_keys():
    outcomes = np.array(["H", "A"])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    m = metrics.all_metrics(probs, outcomes)
    assert set(m.keys()) == {"log_loss", "rps", "brier", "accuracy", "ece", "n"}
    assert m["n"] == 2


# --------------------------------------------------------------------------- #
# walk_forward shapes
# --------------------------------------------------------------------------- #
def test_walk_forward_shapes_align():
    df = synthetic.generate_league(n_teams=8, n_seasons=3, seed=41)
    probs, outcomes, test_df = walk_forward(
        DixonColes, df, min_train_frac=0.6, n_folds=3
    )
    assert probs.ndim == 2 and probs.shape[1] == 3
    assert probs.shape[0] == len(outcomes) == len(test_df)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    # Outcomes are valid 1X2 labels.
    assert set(np.unique(outcomes)).issubset({"H", "D", "A"})


def test_evaluate_model_returns_metric_bundle():
    df = synthetic.generate_league(n_teams=8, n_seasons=3, seed=42)
    out = evaluate_model(DixonColes, df, min_train_frac=0.6, n_folds=3)
    assert set(out.keys()) == {"log_loss", "rps", "brier", "accuracy", "ece", "n"}
    assert out["n"] > 0
    assert out["log_loss"] > 0  # a real model is not perfect


def test_walk_forward_goals_markets_shapes():
    df = synthetic.generate_league(n_teams=8, n_seasons=3, seed=43)
    out = walk_forward_goals_markets(
        DixonColes, df, min_train_frac=0.6, n_folds=2, ou_line=2.5
    )
    assert "over_2.5" in out and "btts" in out
    for market in out.values():
        assert market["n"] > 0
        assert 0.0 <= market["base_rate"] <= 1.0
        assert market["log_loss"] >= 0.0


# --------------------------------------------------------------------------- #
# Betting policy: best_edge_only places <= 1 bet per match
# --------------------------------------------------------------------------- #
def _all_three_value_frame() -> tuple[np.ndarray, pd.DataFrame]:
    """Construct fixtures where ALL three outcomes clear the edge threshold, so
    a naive policy would place 3 bets/match and best_edge_only must place 1."""
    n = 5
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * n) + pd.to_timedelta(range(n), "D"),
            "home_team": [f"H{i}" for i in range(n)],
            "away_team": [f"A{i}" for i in range(n)],
            "home_score": [1, 0, 2, 1, 0],
            "away_score": [0, 1, 2, 2, 0],
            # Generous odds so every model prob * odds - 1 clears the threshold.
            "home_odds": [4.0] * n,
            "draw_odds": [4.0] * n,
            "away_odds": [4.0] * n,
        }
    )
    # Model assigns 1/3 to each; 0.3333*4 - 1 = 0.333 > edge 0.05 for all three.
    probs = np.full((n, 3), 1.0 / 3.0)
    return probs, df


def test_betting_best_edge_only_at_most_one_bet_per_match():
    probs, df = _all_three_value_frame()
    res = betting_backtest(
        probs, df, edge_threshold=0.05, best_edge_only=True
    )
    assert res["n_bets"] <= len(df)


def test_betting_multi_bet_mode_exceeds_one_per_match():
    """With best_edge_only=False the legacy policy backs every qualifying
    outcome, so n_bets exceeds the match count (the bug best_edge_only fixes)."""
    probs, df = _all_three_value_frame()
    multi = betting_backtest(
        probs, df, edge_threshold=0.05, best_edge_only=False
    )
    single = betting_backtest(
        probs, df, edge_threshold=0.05, best_edge_only=True
    )
    assert multi["n_bets"] > single["n_bets"]
    assert single["n_bets"] <= len(df)


def test_betting_returns_expected_keys():
    probs, df = _all_three_value_frame()
    res = betting_backtest(probs, df)
    assert set(res.keys()) == {
        "n_bets",
        "turnover",
        "profit",
        "roi",
        "hit_rate",
        "max_drawdown",
    }


def test_betting_no_value_places_no_bets():
    n = 4
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * n),
            "home_team": [f"H{i}" for i in range(n)],
            "away_team": [f"A{i}" for i in range(n)],
            "home_score": [1] * n,
            "away_score": [0] * n,
            # Tight odds: 0.3333 * 2.0 - 1 = -0.33 < edge -> no bet.
            "home_odds": [2.0] * n,
            "draw_odds": [2.0] * n,
            "away_odds": [2.0] * n,
        }
    )
    probs = np.full((n, 3), 1.0 / 3.0)
    res = betting_backtest(probs, df, edge_threshold=0.05)
    assert res["n_bets"] == 0
    assert res["roi"] == 0.0


def test_max_drawdown_basic():
    # equity path: +1, +1 (peak 2), -3 (trough -1) -> max drawdown = 3.
    pnl = [1.0, 1.0, -3.0, 1.0]
    assert abs(max_drawdown(pnl) - 3.0) < 1e-9
    assert max_drawdown([]) == 0.0
    assert max_drawdown([1.0, 1.0, 1.0]) == 0.0  # monotonic rise


# --------------------------------------------------------------------------- #
# H1 regression: walk_forward must pre-build features so the classifier does NOT
# see reset Elo/form/played at predict time.
# --------------------------------------------------------------------------- #
def test_walk_forward_does_not_reset_classifier_features():
    from soccer_predictor.models.outcome_classifier import OutcomeClassifier

    df = synthetic.generate_league(n_teams=14, n_seasons=5, seed=3)
    _p, _o, test_df = walk_forward(lambda: OutcomeClassifier(kind="gbm"), df, n_folds=5)
    # Pre-built features ride along on the pooled test frame...
    assert "elo_home" in test_df.columns and "home_played" in test_df.columns
    # ...and they reflect full accumulated history, not a per-slice reset (~4).
    assert float(test_df["home_played"].max()) > 20.0


def test_walk_forward_handles_class_missing_in_a_fold():
    """H4: an early fold can train on a class-incomplete slice without crashing."""
    from soccer_predictor.models.outcome_classifier import OutcomeClassifier

    df = synthetic.generate_league(n_teams=10, n_seasons=4, seed=9)
    probs, outcomes, _ = walk_forward(
        lambda: OutcomeClassifier(kind="gbm"), df, min_train_frac=0.3, n_folds=8
    )
    assert probs.shape[1] == 3
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
