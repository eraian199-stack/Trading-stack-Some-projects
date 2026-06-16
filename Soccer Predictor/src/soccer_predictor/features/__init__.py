"""
No-leakage feature engineering.

The central entry point is :func:`feature_store.build_features`, which walks a
canonical match frame in date order and attaches only pre-kickoff information.
Individual feature families live in their own modules (elo, rolling, form,
rest_travel, market, xg, squad, international) and are composed by the store.
"""

from .elo import EloModel, goal_diff_multiplier
from .feature_store import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    XG_FEATURE_COLUMNS,
    available_feature_columns,
    build_features,
)
from .international import INTL_FEATURE_COLUMNS, add_international_features
from .market import add_market_features, implied_probabilities
from .squad import SQUAD_FEATURE_COLUMNS, add_squad_features

__all__ = [
    "EloModel",
    "goal_diff_multiplier",
    "build_features",
    "available_feature_columns",
    "FEATURE_COLUMNS",
    "XG_FEATURE_COLUMNS",
    "MARKET_FEATURE_COLUMNS",
    "SQUAD_FEATURE_COLUMNS",
    "INTL_FEATURE_COLUMNS",
    "implied_probabilities",
    "add_market_features",
    "add_squad_features",
    "add_international_features",
]
