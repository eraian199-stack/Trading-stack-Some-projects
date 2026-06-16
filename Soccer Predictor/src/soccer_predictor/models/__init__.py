"""Models subpackage: 1X2 / scoreline models, ensemble, and calibration.

Public model surface (all build against ``models.base``):
    MarketBaseline      de-vigged bookmaker odds (the benchmark)
    DixonColes          time-weighted MLE generative goal model
    PoissonXG / XGPoisson   xG-based Poisson scoreline model
    EloGoalsModel       robust Elo-driven scoreline model (simulator default)
    OutcomeClassifier   discriminative GBM / logistic 1X2 classifier
    Ensemble / default_ensemble   per-row weight-renormalising blend
    CalibratedModel     isotonic / temperature calibration wrapper
"""

from .base import BaseModel, ScorelineModel
from .calibration import CalibratedModel, fit_temperature, temperature_scale
from .dixon_coles import DixonColes
from .elo_goals import EloGoalsModel
from .ensemble import Ensemble, default_ensemble
from .market_baseline import MarketBaseline
from .outcome_classifier import OutcomeClassifier
from .poisson_xg import PoissonXG, XGPoisson

__all__ = [
    "BaseModel",
    "ScorelineModel",
    "MarketBaseline",
    "DixonColes",
    "PoissonXG",
    "XGPoisson",
    "EloGoalsModel",
    "OutcomeClassifier",
    "Ensemble",
    "default_ensemble",
    "CalibratedModel",
    "temperature_scale",
    "fit_temperature",
]
