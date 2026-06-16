"""
ML evaluation with the discipline this project insists on.

Accuracy/log-loss leaderboards are easy to fool yourself with, so the ML suite is
judged on three things beyond the headline metrics:

1. **No-skill baseline.** The class-prior log loss. Any model must beat it; a
   model scoring far BELOW it on shuffled/limited data would signal leakage.
2. **Overfit gap.** In-sample vs out-of-sample log loss per fold. A big gap means
   the model is memorising, not generalising.
3. **Beat-the-market test.** The only bar that means money: does the model beat
   the de-vigged book OUT OF SAMPLE on the rows the book can price? Reported as a
   paired per-match log-loss delta with a bootstrap 95% CI. A free model that
   *significantly* beats the closing line is suspicious (leakage / stale odds),
   not a gift -- the report flags it either way.

Everything is walk-forward and time-ordered (never random k-fold).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..data import schemas
from . import metrics
from .walk_forward import _ordered_completed, _with_features, walk_forward

CLASSES = schemas.RESULT_CLASSES
_EPS = 1e-15


def no_skill_log_loss(outcomes: np.ndarray) -> float:
    """Log loss of always predicting the class base rates (the no-skill floor)."""
    idx = np.array([CLASSES.index(o) for o in outcomes])
    counts = np.bincount(idx, minlength=3).astype(float)
    priors = counts / counts.sum()
    return float(-np.mean(np.log(np.clip(priors[idx], _EPS, 1.0))))


def _per_match_log_loss(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    idx = np.array([CLASSES.index(o) for o in outcomes])
    chosen = np.clip(probs[np.arange(len(idx)), idx], _EPS, 1.0)
    return -np.log(chosen)


def beats_market(
    model_probs: np.ndarray,
    test_df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired per-match log-loss delta (model - market) on odds-complete rows.

    Negative delta => the model beats the de-vigged book. Returns the mean delta,
    a bootstrap 95% CI, the share of matches the model wins, and a verdict flag.
    Empty/odds-less inputs yield NaNs.
    """
    from ..models.market_baseline import MarketBaseline

    if len(test_df) == 0 or not schemas.has_usable_odds(test_df):
        return {"delta_mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "pct_model_better": float("nan"),
                "n": 0, "model_beats_market": False}
    with_odds = schemas.rows_with_complete_odds(test_df)
    keep = test_df.index.isin(with_odds.index)
    sub = test_df.loc[keep]
    mp = model_probs[keep]
    outcomes = sub["result"].to_numpy()

    market = MarketBaseline().fit(with_odds.copy())
    market_probs = np.asarray(market.predict_proba(with_odds.copy()), dtype=float)
    finite = np.isfinite(market_probs).all(axis=1) & np.isfinite(mp).all(axis=1)
    mp, market_probs, outcomes = mp[finite], market_probs[finite], outcomes[finite]
    if len(outcomes) == 0:
        return {"delta_mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "pct_model_better": float("nan"),
                "n": 0, "model_beats_market": False}

    delta = _per_match_log_loss(mp, outcomes) - _per_match_log_loss(market_probs, outcomes)
    rng = np.random.default_rng(seed)
    boot = np.array([
        delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "delta_mean": round(float(delta.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "pct_model_better": round(float((delta < 0).mean()), 3),
        "n": int(len(delta)),
        # "beats" only if the WHOLE CI is below zero (significantly better OOS).
        "model_beats_market": bool(hi < 0),
    }


def overfit_gap(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    min_train_frac: float = 0.5,
    n_folds: int = 6,
) -> dict:
    """Mean in-sample vs out-of-sample log loss across walk-forward folds.

    ``gap = oos - in_sample``; a large positive gap means the model fits the
    training data far better than the future -- the classic overfit signature.
    """
    df = _with_features(_ordered_completed(df))
    n = len(df)
    start = int(n * min_train_frac)
    block = max(1, (n - start) // max(1, n_folds))
    in_lls, oos_lls = [], []
    for k in range(n_folds):
        lo = start + k * block
        hi = n if k == n_folds - 1 else start + (k + 1) * block
        if lo >= n:
            break
        tr, te = df.iloc[:lo], df.iloc[lo:hi]
        if len(te) == 0 or len(tr) < 20:
            continue
        model = model_factory()
        model.fit(tr.copy())
        tr_lab = tr[tr["result"].isin(CLASSES)]
        in_p = np.asarray(model.predict_proba(tr_lab.copy()), dtype=float)
        oos_p = np.asarray(model.predict_proba(te.copy()), dtype=float)
        in_lls.append(metrics.log_loss(in_p, tr_lab["result"].to_numpy()))
        oos_lls.append(metrics.log_loss(oos_p, te["result"].to_numpy()))
    if not oos_lls:
        return {"in_sample": float("nan"), "oos": float("nan"), "gap": float("nan")}
    in_m, oos_m = float(np.mean(in_lls)), float(np.mean(oos_lls))
    return {"in_sample": round(in_m, 4), "oos": round(oos_m, 4),
            "gap": round(oos_m - in_m, 4)}


def ml_comparison(
    df: pd.DataFrame,
    factories: dict[str, Callable[[], object]] | None = None,
    min_train_frac: float = 0.5,
    n_folds: int = 6,
    include_overfit: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Walk-forward leaderboard with overfit gap + beat-the-market verdict.

    Columns: log_loss, rps, brier, ece, n, no_skill (the floor), vs_market
    (delta_mean), vs_market_ci, beats_market, overfit_gap. Sorted by log loss.
    Includes the StackedEnsemble by default. The market row is the benchmark.
    """
    if factories is None:
        from ..models import (
            DixonColes, EloGoalsModel, MarketBaseline, OutcomeClassifier,
            default_ensemble,
        )
        from ..models.stacking import StackedEnsemble

        factories = {
            "market": MarketBaseline,
            "elo": EloGoalsModel,
            "dixon_coles": DixonColes,
            "logreg": lambda: OutcomeClassifier(kind="logreg"),
            "gbm": lambda: OutcomeClassifier(kind="gbm"),
            "ensemble": default_ensemble,
            "stack": StackedEnsemble,
        }

    rows = {}
    for name, fac in factories.items():
        probs, outcomes, test_df = walk_forward(
            fac, df, min_train_frac=min_train_frac, n_folds=n_folds
        )
        if len(outcomes) == 0:
            continue
        m = metrics.all_metrics(probs, outcomes)
        m["no_skill"] = round(no_skill_log_loss(outcomes), 4)
        if name != "market":
            bm = beats_market(probs, test_df, seed=seed)
            m["vs_market"] = bm["delta_mean"]
            m["vs_market_ci"] = f"[{bm['ci_low']}, {bm['ci_high']}]"
            m["beats_market"] = bm["model_beats_market"]
        else:
            m["vs_market"] = 0.0
            m["vs_market_ci"] = "-"
            m["beats_market"] = False
        if include_overfit:
            m["overfit_gap"] = overfit_gap(fac, df, min_train_frac, n_folds)["gap"]
        rows[name] = m

    df_out = pd.DataFrame(rows).T
    order = [c for c in ["log_loss", "rps", "brier", "ece", "no_skill",
                         "vs_market", "vs_market_ci", "beats_market",
                         "overfit_gap", "n"] if c in df_out.columns]
    return df_out[order].sort_values("log_loss")
