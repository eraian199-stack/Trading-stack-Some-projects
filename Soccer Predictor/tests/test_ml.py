"""
Tests for the ML suite: the stacked ensemble and the disciplined ML report
(no-skill floor, overfit gap, beat-the-market verdict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import soccer_predictor as sp
from soccer_predictor.evaluation import ml_report
from soccer_predictor.models.stacking import StackedEnsemble


@pytest.fixture(scope="module")
def league_df() -> pd.DataFrame:
    # Small but enough for OOF folds; carries odds + xG (synthetic). Kept compact
    # so the (Dixon-Coles-heavy) stack tests stay fast.
    return sp.generate_league(n_teams=10, n_seasons=4, seed=11)


# --------------------------------------------------------------------------- #
# StackedEnsemble
# --------------------------------------------------------------------------- #
def test_stacked_ensemble_predicts_valid_distribution(league_df):
    n = len(league_df)
    st = StackedEnsemble(n_splits=2).fit(league_df.iloc[: n - 40])
    proba = st.predict_proba(league_df.iloc[n - 40 :])
    assert proba.shape == (40, 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_stacked_ensemble_empty_input(league_df):
    st = StackedEnsemble(n_splits=2).fit(league_df.iloc[:300])
    assert st.predict_proba(league_df.iloc[:0]).shape == (0, 3)


def test_stack_meta_weights_sum_to_one(league_df):
    st = StackedEnsemble(meta="logreg", n_splits=2).fit(league_df.iloc[:300])
    w = st.meta_weights()
    assert w is not None
    # weights are rounded to 3 dp, so allow rounding slack
    assert abs(sum(w.values()) - 1.0) < 0.01
    assert set(w).issuperset({"dixon_coles", "elo", "gbm"})


def test_stack_works_without_odds(league_df):
    # Drop odds -> the market base must be omitted, stack still works.
    no_odds = league_df.drop(columns=["home_odds", "draw_odds", "away_odds"])
    st = StackedEnsemble(n_splits=2).fit(no_odds.iloc[:300])
    proba = st.predict_proba(no_odds.iloc[300:340])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert "market" not in st.bases_


# --------------------------------------------------------------------------- #
# ML report diagnostics
# --------------------------------------------------------------------------- #
def test_no_skill_log_loss_matches_prior_entropy():
    # 50% H, 25% D, 25% A -> prior log loss = -(.5ln.5 + .25ln.25 + .25ln.25)
    outcomes = np.array(["H"] * 50 + ["D"] * 25 + ["A"] * 25, dtype=object)
    expected = -(0.5 * np.log(0.5) + 0.25 * np.log(0.25) + 0.25 * np.log(0.25))
    assert abs(ml_report.no_skill_log_loss(outcomes) - expected) < 1e-9


def test_beats_market_keys_and_market_is_hard(league_df):
    # The synthetic market is near-truth, so a model should NOT significantly beat it.
    probs, outcomes, test_df = sp.walk_forward(
        lambda: sp.DixonColes(), league_df, n_folds=3
    )
    bm = ml_report.beats_market(probs, test_df, n_boot=300, seed=0)
    assert set(bm) >= {"delta_mean", "ci_low", "ci_high", "model_beats_market", "n"}
    assert bm["n"] > 0
    assert bm["model_beats_market"] is False  # can't beat the (near-truth) book


def test_overfit_gap_runs(league_df):
    gap = ml_report.overfit_gap(lambda: sp.EloGoalsModel(), league_df, n_folds=3)
    assert set(gap) == {"in_sample", "oos", "gap"}
    assert np.isfinite(gap["oos"])


def test_ml_comparison_leaderboard(league_df):
    factories = {
        "market": sp.MarketBaseline,
        "elo": sp.EloGoalsModel,
        "stack": lambda: StackedEnsemble(n_splits=2),
    }
    lb = ml_report.ml_comparison(
        league_df, factories=factories, n_folds=3, include_overfit=False
    )
    assert {"market", "elo", "stack"}.issubset(set(lb.index))
    assert "log_loss" in lb.columns and "vs_market" in lb.columns
    # the market row is its own benchmark
    assert lb.loc["market", "vs_market"] == 0.0
