"""
soccer_predictor: a unified club + international + tournament soccer predictor.

Three layers, deliberately separated (see docs/framework.md):

1. Match prediction  -- given two teams + context, a probability distribution
   over outcomes and scorelines (1X2, exact score, xG, O/U, BTTS, fair odds).
2. Tournament / season simulation -- group stages, league tables, knockout
   brackets, and Monte Carlo winner / advancement probabilities (World Cup 2026
   format built in).
3. Research / betting evaluation -- walk-forward validation, log loss / RPS /
   Brier / calibration, a market benchmark, and betting backtests.

Core rule: the match model only answers "what happens in THIS fixture"; the
simulator only answers "given a format, who advances / wins". They never mix.

Quick start:

    import soccer_predictor as sp
    df = sp.generate_league()                      # synthetic, no downloads
    probs, outcomes, _ = sp.walk_forward(lambda: sp.DixonColes(), df)
    print(sp.all_metrics(probs, outcomes))
"""

from __future__ import annotations

# -- data -------------------------------------------------------------------
from .data import schemas
from .data.aliases import normalize_team_name
from .data.normalizers import standardize_matches
from .data.loaders import (
    load_matches,
    load_football_data,
    load_international_results,
    load_groups,
    load_fixtures,
)
from .data.sources import (
    INTERNATIONAL_RESULTS_URL,
    classify_competition_type,
    fetch_international_results,
    fetch_football_data,
)
from .data.synthetic import generate_league, generate_international_history
from .data.quality import quality_report
from .data import odds_api, world_cup
from .data.odds_api import (
    fetch_match_odds,
    fetch_scores,
    list_sports,
    load_odds_api_key,
    WORLD_CUP_SPORT,
)

# -- features ---------------------------------------------------------------
from .features.feature_store import (
    build_features,
    available_feature_columns,
    FEATURE_COLUMNS,
)
from .features.elo import EloModel

# -- models -----------------------------------------------------------------
from .models.base import BaseModel, ScorelineModel, predict_markets
from .models.market_baseline import MarketBaseline
from .models.dixon_coles import DixonColes
from .models.poisson_xg import PoissonXG, XGPoisson
from .models.elo_goals import EloGoalsModel
from .models.outcome_classifier import OutcomeClassifier
from .models.ensemble import Ensemble, default_ensemble
from .models.calibration import CalibratedModel
from .models.market_anchor import MarketAnchoredModel
from .models.stacking import StackedEnsemble, default_base_factories

# -- evaluation -------------------------------------------------------------
from .evaluation.metrics import (
    log_loss, rps, brier, accuracy, expected_calibration_error, all_metrics,
)
from .evaluation.walk_forward import (
    walk_forward, walk_forward_goals_markets, evaluate_model, compare_to_market,
)
from .evaluation.betting import betting_backtest, closing_line_value, max_drawdown
from .evaluation.reports import leaderboard, format_metrics, calibration_table
from .evaluation.wc_backtest import (
    backtest_world_cups,
    compare_models_on_world_cups,
    backtest_champions,
    backtest_tournament,
    actual_stage_reach,
    identify_world_cups,
)
from .evaluation.ml_report import (
    ml_comparison,
    beats_market,
    overfit_gap,
    no_skill_log_loss,
)
from .evaluation.ablation import (
    ablation_table,
    wc_ablation,
    walk_forward_ablation,
    national_model_variants,
    elo_hyperparam_variants,
    club_feature_variants,
)

# -- simulation -------------------------------------------------------------
from .simulation import rules
from .simulation.rules import (
    world_cup_2026_format,
    world_cup_legacy_format,
    TournamentFormat,
    KnockoutTie,
)
from .simulation.match import simulate_match, SimulatedMatch
from .simulation.league import simulate_league
from .simulation.group_stage import simulate_group_stage
from .simulation.knockout import simulate_knockout, simulate_single_elimination
from .simulation.tournament import simulate_tournament, simulate_tournament_once
from .simulation.monte_carlo import monte_carlo_tournament, monte_carlo_league

__version__ = "0.2.0"

__all__ = [
    # data
    "schemas", "normalize_team_name", "standardize_matches",
    "load_matches", "load_football_data", "load_international_results",
    "load_groups", "load_fixtures",
    "INTERNATIONAL_RESULTS_URL", "classify_competition_type",
    "fetch_international_results", "fetch_football_data",
    "generate_league", "generate_international_history", "quality_report",
    "odds_api", "world_cup", "fetch_match_odds", "fetch_scores", "list_sports",
    "load_odds_api_key", "WORLD_CUP_SPORT",
    # features
    "build_features", "available_feature_columns", "FEATURE_COLUMNS", "EloModel",
    # models
    "BaseModel", "ScorelineModel", "predict_markets",
    "MarketBaseline", "DixonColes", "PoissonXG", "XGPoisson", "EloGoalsModel",
    "OutcomeClassifier", "Ensemble", "default_ensemble", "CalibratedModel",
    "MarketAnchoredModel", "StackedEnsemble", "default_base_factories",
    # evaluation
    "log_loss", "rps", "brier", "accuracy", "expected_calibration_error",
    "all_metrics", "walk_forward", "walk_forward_goals_markets",
    "evaluate_model", "compare_to_market",
    "betting_backtest", "closing_line_value", "max_drawdown",
    "leaderboard", "format_metrics", "calibration_table",
    "backtest_world_cups", "compare_models_on_world_cups", "backtest_champions",
    "backtest_tournament", "actual_stage_reach",
    "identify_world_cups",
    "ml_comparison", "beats_market", "overfit_gap", "no_skill_log_loss",
    "ablation_table", "wc_ablation", "walk_forward_ablation",
    "national_model_variants", "elo_hyperparam_variants", "club_feature_variants",
    # simulation
    "rules", "world_cup_2026_format", "world_cup_legacy_format",
    "TournamentFormat", "KnockoutTie",
    "simulate_match", "SimulatedMatch", "simulate_league",
    "simulate_group_stage", "simulate_knockout", "simulate_single_elimination",
    "simulate_tournament", "simulate_tournament_once",
    "monte_carlo_tournament", "monte_carlo_league",
    "__version__",
]
