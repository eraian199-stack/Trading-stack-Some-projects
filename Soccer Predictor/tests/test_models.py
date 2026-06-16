"""
Model contract tests.

These pin the invariants every model must satisfy:
  * predict_proba rows sum to 1 (over [H, D, A]);
  * a ScorelineModel's score_matrix is square and sums to 1;
  * over_under respects the line: P(over 0.5) > P(over 4.5);
  * DixonColes recovers strong > weak ordering on synthetic data;
  * MarketBaseline de-vigs to a valid distribution and emits NaN without odds;
  * CalibratedModel preserves output shape and row-sums.

All models are fit on small synthetic frames so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soccer_predictor.data import synthetic
from soccer_predictor.models.base import collapse_1x2, score_matrix_from_rates
from soccer_predictor.models.calibration import CalibratedModel
from soccer_predictor.models.dixon_coles import DixonColes
from soccer_predictor.models.elo_goals import EloGoalsModel
from soccer_predictor.models.market_baseline import MarketBaseline
from soccer_predictor.models.poisson_xg import PoissonXG


@pytest.fixture(scope="module")
def league_df() -> pd.DataFrame:
    # Small but enough teams/games for a stable DC fit.
    return synthetic.generate_league(n_teams=8, n_seasons=3, seed=21)


def _is_distribution(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float)
    return bool(
        np.all(arr >= -1e-9)
        and np.all(arr <= 1 + 1e-9)
        and np.allclose(arr.sum(), 1.0, atol=1e-6)
    )


# --------------------------------------------------------------------------- #
# Score-matrix math (the shared foundation)
# --------------------------------------------------------------------------- #
def test_score_matrix_from_rates_square_and_normalised():
    mat = score_matrix_from_rates(1.4, 1.1, max_goals=8, rho=-0.05)
    assert mat.shape == (9, 9)
    assert np.all(mat >= 0.0)
    assert abs(float(mat.sum()) - 1.0) < 1e-9


def test_collapse_1x2_sums_to_one():
    mat = score_matrix_from_rates(1.6, 0.9, max_goals=10)
    p = collapse_1x2(mat)
    assert _is_distribution(p)


# --------------------------------------------------------------------------- #
# DixonColes
# --------------------------------------------------------------------------- #
def test_dixon_coles_predict_proba_sums_to_one(league_df):
    model = DixonColes().fit(league_df)
    test = league_df.head(20)
    proba = model.predict_proba(test)
    assert proba.shape == (len(test), 3)
    sums = proba.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_dixon_coles_score_matrix_normalised(league_df):
    model = DixonColes().fit(league_df)
    teams = sorted(set(league_df["home_team"]))
    mat = model.score_matrix(teams[0], teams[1])
    assert mat.shape[0] == mat.shape[1]
    assert abs(float(mat.sum()) - 1.0) < 1e-6


def test_dixon_coles_recovers_strength_ordering(league_df):
    """The team with the highest attack-minus-defence should beat the weakest at
    a neutral venue (no home-advantage confound)."""
    model = DixonColes().fit(league_df)
    strength = model.attack - model.defence
    strong = model.teams[int(np.argmax(strength))]
    weak = model.teams[int(np.argmin(strength))]
    ctx = {"neutral_site": True}
    p_strong_home = model.outcome_probs(strong, weak, ctx)  # [H, D, A]
    p_weak_home = model.outcome_probs(weak, strong, ctx)
    # Strong side wins more often than the weak side regardless of who is "home".
    assert p_strong_home[0] > p_strong_home[2]
    assert p_weak_home[2] > p_weak_home[0]


def test_dixon_coles_neutral_drops_home_advantage(league_df):
    model = DixonColes().fit(league_df)
    teams = sorted(set(league_df["home_team"]))
    h, a = teams[0], teams[1]
    p_home = model.outcome_probs(h, a, {"neutral_site": False})
    p_neutral = model.outcome_probs(h, a, {"neutral_site": True})
    # Dropping home advantage cannot increase the home-win probability.
    assert p_neutral[0] <= p_home[0] + 1e-9


def test_dixon_coles_unseen_team_is_league_average(league_df):
    model = DixonColes().fit(league_df)
    known = model.teams[0]
    # An unseen team should produce a valid, finite distribution (attack=def=0).
    p = model.outcome_probs("Totally Unseen FC", known, {"neutral_site": True})
    assert _is_distribution(p)


# --------------------------------------------------------------------------- #
# over_under respects the line (the bug this contract guards against)
# --------------------------------------------------------------------------- #
def test_over_under_respects_the_line(league_df):
    model = DixonColes().fit(league_df)
    teams = sorted(set(league_df["home_team"]))
    h, a = teams[0], teams[1]
    over_half, _ = model.over_under(h, a, line=0.5)
    over_high, _ = model.over_under(h, a, line=4.5)
    # P(total > 0.5) must strictly exceed P(total > 4.5).
    assert over_half > over_high
    # And both are valid probabilities.
    assert 0.0 <= over_high <= over_half <= 1.0


def test_btts_is_valid_probability(league_df):
    model = DixonColes().fit(league_df)
    teams = sorted(set(league_df["home_team"]))
    yes, no = model.both_teams_to_score(teams[0], teams[1])
    assert abs((yes + no) - 1.0) < 1e-9
    assert 0.0 <= yes <= 1.0


# --------------------------------------------------------------------------- #
# MarketBaseline
# --------------------------------------------------------------------------- #
def test_market_baseline_devig_sums_to_one(league_df):
    model = MarketBaseline().fit(league_df)
    proba = model.predict_proba(league_df.head(15))
    sums = proba.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)
    assert np.all(proba >= 0.0)


def test_market_baseline_nan_without_odds():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "home_team": ["A", "B"],
            "away_team": ["X", "Y"],
            "home_odds": [2.0, np.nan],
            "draw_odds": [3.3, 3.3],
            "away_odds": [3.5, 3.5],
        }
    )
    proba = MarketBaseline().predict_proba(df)
    # Row with full odds is a distribution; row with a missing odd is all-NaN.
    assert abs(proba[0].sum() - 1.0) < 1e-9
    assert np.isnan(proba[1]).all()


# --------------------------------------------------------------------------- #
# PoissonXG
# --------------------------------------------------------------------------- #
def test_poisson_xg_predict_proba_sums_to_one(league_df):
    model = PoissonXG().fit(league_df)
    proba = model.predict_proba(league_df.head(12))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_poisson_xg_requires_xg_columns():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [1],
            "away_score": [0],
        }
    )
    with pytest.raises(ValueError):
        PoissonXG().fit(df)


# --------------------------------------------------------------------------- #
# EloGoalsModel (the robust default)
# --------------------------------------------------------------------------- #
def test_elo_goals_predict_proba_sums_to_one(league_df):
    model = EloGoalsModel().fit(league_df)
    proba = model.predict_proba(league_df.head(12))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_elo_goals_score_matrix_normalised(league_df):
    model = EloGoalsModel().fit(league_df)
    teams = sorted(set(league_df["home_team"]))
    mat = model.score_matrix(teams[0], teams[1])
    assert mat.shape[0] == mat.shape[1]
    assert abs(float(mat.sum()) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# CalibratedModel preserves shape & row sums
# --------------------------------------------------------------------------- #
def test_calibrated_model_preserves_shape_and_sums(league_df):
    model = CalibratedModel(DixonColes(), method="temperature", holdout_frac=0.3)
    model.fit(league_df)
    test = league_df.head(20)
    proba = model.predict_proba(test)
    assert proba.shape == (len(test), 3)
    finite = np.isfinite(proba).all(axis=1)
    assert finite.any()
    np.testing.assert_allclose(proba[finite].sum(axis=1), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# OutcomeClassifier robustness (regression: missing optional cols / empty input)
# --------------------------------------------------------------------------- #
def test_outcome_classifier_predict_robust_to_missing_optional_columns(league_df):
    """Trained WITH odds/xG, asked to predict on a frame lacking them.

    The optional feature groups (market-implied, xG-rolling) must be reindexed
    to NaN rather than raising a KeyError.
    """
    from soccer_predictor.models.outcome_classifier import OutcomeClassifier

    clf = OutcomeClassifier(kind="gbm").fit(league_df)
    stripped = league_df.drop(columns=["home_odds", "draw_odds", "away_odds"]).tail(15)
    proba = clf.predict_proba(stripped)
    assert proba.shape == (len(stripped), 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_outcome_classifier_predict_on_empty_frame(league_df):
    from soccer_predictor.models.outcome_classifier import OutcomeClassifier

    clf = OutcomeClassifier(kind="gbm").fit(league_df)
    proba = clf.predict_proba(league_df.iloc[:0])
    assert proba.shape == (0, 3)


# --------------------------------------------------------------------------- #
# Picklability (regression: rating engines must not use local-lambda defaultdicts)
# --------------------------------------------------------------------------- #
def test_elo_models_pickle_round_trip(league_df):
    import pickle

    from soccer_predictor.features.elo import EloModel
    from soccer_predictor.models.elo_goals import EloGoalsModel

    elo = EloModel()
    for r in league_df.head(150).itertuples(index=False):
        elo.update(r.home_team, r.away_team, r.home_score, r.away_score)
    elo2 = pickle.loads(pickle.dumps(elo))  # would raise without the plain-dict fix
    assert elo2.rating("Team_00") == elo.rating("Team_00")

    eg = EloGoalsModel().fit(league_df)
    eg2 = pickle.loads(pickle.dumps(eg))
    ctx = {"neutral_site": True}
    np.testing.assert_allclose(
        eg.outcome_probs("Team_00", "Team_01", ctx),
        eg2.outcome_probs("Team_00", "Team_01", ctx),
    )


def test_fitted_default_ensemble_pickles(league_df):
    import pickle

    from soccer_predictor.models.ensemble import default_ensemble

    ens = default_ensemble().fit(league_df)
    ens2 = pickle.loads(pickle.dumps(ens))  # Elo + classifier + market + calibration
    test = league_df.tail(20)
    np.testing.assert_allclose(
        ens.predict_proba(test), ens2.predict_proba(test), equal_nan=True
    )
