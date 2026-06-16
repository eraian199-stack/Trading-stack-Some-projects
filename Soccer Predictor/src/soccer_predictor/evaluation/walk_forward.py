"""
Time-aware (walk-forward) model evaluation.

TIME-AWARE validation only. We use an expanding-window walk-forward: train on
the past, predict the next block, step forward, repeat. Random k-fold leaks the
future into training and produces fantasy accuracy -- never use it here.

Three entry points:
    walk_forward                 -- pooled 1X2 probs/outcomes/test rows.
    walk_forward_goals_markets   -- over/under + BTTS log loss / Brier.
    evaluate_model               -- walk_forward + all_metrics in one call.
    compare_to_market            -- model vs the de-vigged book on the SAME rows
                                    that have complete odds.

Everything reads the canonical schema: `home_score`/`away_score`, the `result`
label in {H, D, A}, and `home_odds`/`draw_odds`/`away_odds`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..data import schemas
from . import metrics


def _ordered_completed(df: pd.DataFrame) -> pd.DataFrame:
    """Date-sorted, completed-only frame with a `result` column."""
    df = schemas.completed_matches(df)
    if "result" not in df.columns:
        df = schemas.add_result_column(df)
    # Keep only rows with a usable 1X2 label.
    df = df[df["result"].isin(schemas.RESULT_CLASSES)]
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


def _with_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe features ONCE over the full ordered frame.

    Critical for the OutcomeClassifier/Ensemble path: if features were rebuilt
    per fold on the isolated test slice, every per-team running state (Elo,
    rolling form, rest days, games played) would reset to base/empty over the
    slice -- severe train/serve skew. Building once over the whole ordered frame
    gives each row features computed from only EARLIER rows (no leakage), and the
    classifier's `_ensure_features` then sees the `elo_home` sentinel and reuses
    these columns instead of rebuilding. Scoreline models ignore the extra
    columns, so this is safe for every model.
    """
    if "elo_home" in df.columns:
        return df
    try:
        from ..features.feature_store import build_features
    except Exception:  # pragma: no cover - features is a core module
        return df
    return build_features(df)


def _fold_bounds(n: int, min_train_frac: float, n_folds: int):
    """Yield (train_hi, test_lo, test_hi) for an expanding-window schedule."""
    start = int(n * min_train_frac)
    block = max(1, (n - start) // max(1, n_folds))
    for k in range(n_folds):
        test_lo = start + k * block
        test_hi = n if k == n_folds - 1 else start + (k + 1) * block
        if test_lo >= n:
            break
        yield test_lo, test_lo, test_hi


# --------------------------------------------------------------------------- #
# 1X2 walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    min_train_frac: float = 0.5,
    n_folds: int = 6,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Expanding-window 1X2 evaluation.

    `model_factory` is a zero-arg callable returning a FRESH, unfitted model each
    fold (so no information leaks across folds). Returns
    (pooled_probs (n, 3), pooled_outcomes ("H"/"D"/"A"), pooled_test_df).
    """
    df = _with_features(_ordered_completed(df))
    n = len(df)

    all_probs: list[np.ndarray] = []
    all_outcomes: list[np.ndarray] = []
    test_slices: list[pd.DataFrame] = []

    for train_hi, test_lo, test_hi in _fold_bounds(n, min_train_frac, n_folds):
        train = df.iloc[:train_hi]
        test = df.iloc[test_lo:test_hi]
        if len(test) == 0 or len(train) == 0:
            continue
        model = model_factory()
        model.fit(train.copy())
        probs = np.asarray(model.predict_proba(test.copy()), dtype=float)
        all_probs.append(probs)
        all_outcomes.append(test["result"].to_numpy())
        test_slices.append(test)

    if not all_probs:
        return (
            np.zeros((0, 3), dtype=float),
            np.array([], dtype=object),
            df.iloc[0:0].copy(),
        )

    probs = np.vstack(all_probs)
    outcomes = np.concatenate(all_outcomes)
    test_df = pd.concat(test_slices, ignore_index=True)
    return probs, outcomes, test_df


# --------------------------------------------------------------------------- #
# Goals-market walk-forward (over/under, BTTS)
# --------------------------------------------------------------------------- #
def walk_forward_goals_markets(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    min_train_frac: float = 0.5,
    n_folds: int = 6,
    ou_line: float = 2.5,
) -> dict:
    """Walk-forward evaluation for over/under and BTTS.

    The model must expose `over_under(home, away, line)` and
    `both_teams_to_score(home, away)` -- ScorelineModel subclasses do. Uses the
    canonical `home_score`/`away_score` columns for the realised outcome.
    Returns a dict of metrics for each market.
    """
    df = _ordered_completed(df)
    n = len(df)

    ou_pred: list[float] = []
    ou_true: list[float] = []
    btts_pred: list[float] = []
    btts_true: list[float] = []

    for train_hi, test_lo, test_hi in _fold_bounds(n, min_train_frac, n_folds):
        train = df.iloc[:train_hi]
        test = df.iloc[test_lo:test_hi]
        if len(test) == 0 or len(train) == 0:
            continue
        model = model_factory()
        model.fit(train.copy())
        for r in test.itertuples(index=False):
            over, _ = model.over_under(r.home_team, r.away_team, ou_line)
            yes, _ = model.both_teams_to_score(r.home_team, r.away_team)
            hs = float(r.home_score)
            as_ = float(r.away_score)
            ou_pred.append(float(over))
            ou_true.append(1.0 if (hs + as_) > ou_line else 0.0)
            btts_pred.append(float(yes))
            btts_true.append(1.0 if (hs > 0 and as_ > 0) else 0.0)

    return {
        f"over_{ou_line}": {
            "log_loss": metrics.binary_log_loss(ou_pred, ou_true),
            "brier": metrics.binary_brier(ou_pred, ou_true),
            "base_rate": round(float(np.mean(ou_true)), 3) if ou_true else 0.0,
            "n": len(ou_true),
        },
        "btts": {
            "log_loss": metrics.binary_log_loss(btts_pred, btts_true),
            "brier": metrics.binary_brier(btts_pred, btts_true),
            "base_rate": round(float(np.mean(btts_true)), 3) if btts_true else 0.0,
            "n": len(btts_true),
        },
    }


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def evaluate_model(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    min_train_frac: float = 0.5,
    n_folds: int = 6,
) -> dict:
    """Run walk_forward and return all_metrics over the pooled predictions."""
    probs, outcomes, _ = walk_forward(
        model_factory, df, min_train_frac=min_train_frac, n_folds=n_folds
    )
    if len(outcomes) == 0:
        return {"log_loss": float("nan"), "rps": float("nan"),
                "brier": float("nan"), "accuracy": float("nan"),
                "ece": float("nan"), "n": 0}
    return metrics.all_metrics(probs, outcomes)


def compare_to_market(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    min_train_frac: float = 0.5,
    n_folds: int = 6,
) -> dict:
    """Compare a model to the de-vigged market baseline.

    Both are scored on the SAME rows -- only test rows with complete 1X2 odds, so
    the comparison is apples-to-apples. Returns
    {model, market, model_minus_market_log_loss}; a NEGATIVE difference means the
    model beats the closing/opening line on log loss.
    """
    # Imported lazily: the market_baseline module lives in a sibling subpackage
    # and we keep this module importable even before it is on disk.
    from ..models.market_baseline import MarketBaseline

    probs, outcomes, test_df = walk_forward(
        model_factory, df, min_train_frac=min_train_frac, n_folds=n_folds
    )
    if len(test_df) == 0:
        return {"model": {}, "market": {}, "model_minus_market_log_loss": float("nan")}

    # Restrict to test rows the market can actually price.
    with_odds = schemas.rows_with_complete_odds(test_df)
    if len(with_odds) == 0:
        return {
            "model": metrics.all_metrics(probs, outcomes),
            "market": {},
            "model_minus_market_log_loss": float("nan"),
        }

    # Align model probs to the odds-complete subset by position. rows_with_complete_odds
    # preserves the original index, so use it to select the matching model rows.
    keep = test_df.index.isin(with_odds.index)
    model_probs = probs[keep]
    sub_outcomes = test_df.loc[keep, "result"].to_numpy()

    market = MarketBaseline()
    market.fit(with_odds.copy())
    market_probs = np.asarray(market.predict_proba(with_odds.copy()), dtype=float)

    # Guard against any all-NaN rows the baseline could not price.
    finite = np.isfinite(market_probs).all(axis=1)
    model_metrics = metrics.all_metrics(model_probs[finite], sub_outcomes[finite])
    market_metrics = metrics.all_metrics(market_probs[finite], sub_outcomes[finite])

    return {
        "model": model_metrics,
        "market": market_metrics,
        "model_minus_market_log_loss": (
            model_metrics["log_loss"] - market_metrics["log_loss"]
        ),
    }
