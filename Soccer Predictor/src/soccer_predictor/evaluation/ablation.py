"""
Ablation harness: does each variable actually earn its place, out of sample?

The discipline of this project is that a variable stays ON only if it improves
OUT-OF-SAMPLE predictions, not in-sample fit. This module runs that test
systematically and reusably -- the same logic applies to World Cup backtests now
and to clubs / Champions League / league winners later.

Two evaluators:
- ``wc_ablation`` -- run each model variant through the past-World-Cup match-level
  backtest (``backtest_world_cups``) and compare to a named baseline. This is the
  one that answers "do our probabilities/Elo translate to good predictions for
  previous World Cups?".
- ``walk_forward_ablation`` -- run each variant through a generic walk-forward
  (any competition / club league) and compare.

Both return a tidy table with the variant's metrics, the delta vs the baseline,
and a ``helps`` flag (True only if it beats the baseline log loss by more than a
tolerance -- i.e. the variable did something real, not noise).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import metrics
from .walk_forward import walk_forward
from .wc_backtest import backtest_world_cups


def ablation_table(
    results: dict[str, dict],
    baseline: str,
    tol: float = 0.001,
    primary: str = "log_loss",
) -> pd.DataFrame:
    """Turn {variant: metrics} into a delta-vs-baseline table.

    ``helps`` is True when the variant lowers the primary metric (log loss) below
    the baseline by more than ``tol`` -- a real out-of-sample improvement, not
    noise. The baseline row is the reference (helps = False).
    """
    df = pd.DataFrame(results).T
    if baseline not in df.index:
        raise KeyError(f"baseline {baseline!r} not in variants {list(df.index)}")
    base_primary = df.loc[baseline, primary]
    df[f"d_{primary}"] = (df[primary] - base_primary).round(4)
    if "rps" in df.columns:
        df["d_rps"] = (df["rps"] - df.loc[baseline, "rps"]).round(4)
    df["helps"] = df[f"d_{primary}"] < -tol
    df.loc[baseline, "helps"] = False
    cols = [c for c in [primary, f"d_{primary}", "rps", "d_rps", "brier", "ece",
                        "helps", "n"] if c in df.columns]
    return df[cols].sort_values(primary)


def wc_ablation(
    history: pd.DataFrame,
    variants: dict[str, Callable[[], object]],
    baseline: str = "baseline",
    min_year: int = 2006,
    update_within: bool = True,
    tol: float = 0.001,
) -> pd.DataFrame:
    """Ablation on the past-World-Cup match-level backtest.

    ``variants`` maps a name to a fresh-model factory; one MUST be ``baseline``.
    Returns the delta-vs-baseline table (lower log loss / RPS is better).
    """
    results: dict[str, dict] = {}
    for name, factory in variants.items():
        res = backtest_world_cups(
            history, factory, min_year=min_year, update_within=update_within
        )
        results[name] = res["pooled"] or {
            "log_loss": float("nan"), "rps": float("nan"), "brier": float("nan"),
            "ece": float("nan"), "n": 0,
        }
    return ablation_table(results, baseline, tol=tol)


def walk_forward_ablation(
    df: pd.DataFrame,
    variants: dict[str, Callable[[], object]],
    baseline: str = "baseline",
    min_train_frac: float = 0.5,
    n_folds: int = 6,
    tol: float = 0.001,
) -> pd.DataFrame:
    """Ablation on a generic walk-forward (clubs / any competition)."""
    results: dict[str, dict] = {}
    for name, factory in variants.items():
        probs, outcomes, _ = walk_forward(
            factory, df, min_train_frac=min_train_frac, n_folds=n_folds
        )
        if len(outcomes) == 0:
            results[name] = {"log_loss": float("nan"), "rps": float("nan"),
                            "brier": float("nan"), "ece": float("nan"), "n": 0}
        else:
            results[name] = metrics.all_metrics(probs, outcomes)
    return ablation_table(results, baseline, tol=tol)


# --------------------------------------------------------------------------- #
# Predefined variant sets
# --------------------------------------------------------------------------- #
def national_model_variants() -> dict[str, Callable[[], object]]:
    """National-team EloGoalsModel component ablation (the WC engine)."""
    from ..models.elo_goals import EloGoalsModel

    return {
        "baseline": EloGoalsModel,  # plain margin-weighted Elo
        "importance": lambda: EloGoalsModel(use_importance=True),
        "form_150": lambda: EloGoalsModel(form_weight=150.0),
        "form_300": lambda: EloGoalsModel(form_weight=300.0),
        "pedigree": lambda: EloGoalsModel(pedigree_weight=0.5),
        "form+pedigree": lambda: EloGoalsModel(form_weight=250.0, pedigree_weight=0.5),
    }


def elo_hyperparam_variants() -> dict[str, Callable[[], object]]:
    """Elo K / home-advantage / scoreline-shape ablation."""
    from ..models.elo_goals import EloGoalsModel

    return {
        "baseline": EloGoalsModel,
        "k10": lambda: EloGoalsModel(elo_kwargs={"k": 10.0}),
        "k30": lambda: EloGoalsModel(elo_kwargs={"k": 30.0}),
        "homeadv0": lambda: EloGoalsModel(home_advantage_override=0.0),
        "homeadv100": lambda: EloGoalsModel(home_advantage_override=100.0),
        "rho0": lambda: EloGoalsModel(rho=0.0),
        "slope0.6": lambda: EloGoalsModel(supremacy_slope=0.6),
        "slope0.9": lambda: EloGoalsModel(supremacy_slope=0.9),
    }


def club_feature_variants() -> dict[str, Callable[[], object]]:
    """Feature-group ablation for the club OutcomeClassifier.

    The baseline uses ALL available features. Each variant drops exactly ONE group
    (keeping everything else, including the market-implied features) so the delta
    cleanly isolates that group's marginal value. A positive d_log_loss means the
    dropped group was useful. ``feature_cols`` is intersected with what's actually
    available, so passing a superset is safe across leagues with/without xG.
    """
    from ..features import feature_store
    from ..models.outcome_classifier import OutcomeClassifier

    base = list(feature_store.FEATURE_COLUMNS)
    market = list(feature_store.MARKET_FEATURE_COLUMNS)
    xg = list(getattr(feature_store, "XG_FEATURE_COLUMNS", []))
    full = base + market + xg

    elo_only = ["elo_home", "elo_away", "elo_diff", "neutral_site"]
    drop_form = [c for c in full if "form" not in c and "_gf" not in c and "_ga" not in c]
    drop_rest = [c for c in full if "rest" not in c]
    drop_market = [c for c in full if c not in set(market)]
    drop_xg = [c for c in full if c not in set(xg)]

    def clf(cols):
        return lambda: OutcomeClassifier(kind="logreg", feature_cols=cols)

    return {
        "baseline": lambda: OutcomeClassifier(kind="logreg"),  # all available
        "drop_form": clf(drop_form),
        "drop_rest": clf(drop_rest),
        "drop_market": clf(drop_market),
        "drop_xg": clf(drop_xg),
        "elo_only": clf(elo_only),
    }
