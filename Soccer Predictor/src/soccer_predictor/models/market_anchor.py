"""
Market-anchored scoreline model (optional overlay).

The de-vigged betting line is the single strongest estimate of a match's true
probabilities -- far sharper than any free-data model. This wraps ANY scoreline
model and, for fixtures where market odds are available, blends the model's 1X2
toward the de-vigged market, then re-derives a consistent score matrix (so every
downstream market -- O/U, BTTS, correct score -- stays coherent).

It is deliberately OPTIONAL and degrades cleanly:
  - fixtures WITH odds (near-term group games) get anchored;
  - fixtures WITHOUT odds (hypothetical future knockout matchups whose teams are
    not yet known to the book) fall back to the pure base model.

So a tournament Monte Carlo can anchor the games the market can price and let the
base model carry the rest. ``weight`` in [0, 1] is the pull toward the market
(0 = ignore market, 1 = trust the market entirely on priced fixtures).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..data.aliases import normalize_team_name
from ..features.market import implied_probabilities
from .base import ScorelineModel, collapse_1x2, tilt_matrix_to_outcome


class MarketAnchoredModel(ScorelineModel):
    """Blend a base scoreline model toward the de-vigged market where priced."""

    def __init__(
        self,
        base_model: ScorelineModel,
        weight: float = 0.5,
        odds: pd.DataFrame | dict | None = None,
        odds_provider: Callable[[str, str], tuple[float, float, float] | None] | None = None,
    ):
        self.base = base_model
        self.weight = float(np.clip(weight, 0.0, 1.0))
        self.max_goals = getattr(base_model, "max_goals", 10)
        self.odds_provider = odds_provider
        self._odds: dict[tuple[str, str], tuple[float, float, float]] = {}
        if odds is not None:
            self.set_odds(odds)

    # -- odds wiring -------------------------------------------------------
    def set_odds(self, odds: pd.DataFrame | dict) -> "MarketAnchoredModel":
        """Load a fixture->odds table (a canonical odds DataFrame or a dict).

        DataFrame needs home_team/away_team/home_odds/draw_odds/away_odds. A dict
        is keyed by ``(home, away)`` -> ``(home_odds, draw_odds, away_odds)``.
        """
        self._odds = {}
        if isinstance(odds, pd.DataFrame):
            for r in odds.itertuples(index=False):
                try:
                    key = (normalize_team_name(r.home_team), normalize_team_name(r.away_team))
                    self._odds[key] = (float(r.home_odds), float(r.draw_odds), float(r.away_odds))
                except (AttributeError, TypeError, ValueError):
                    continue
        elif isinstance(odds, dict):
            for (h, a), v in odds.items():
                self._odds[(normalize_team_name(h), normalize_team_name(a))] = (
                    float(v[0]), float(v[1]), float(v[2])
                )
        return self

    def _market_1x2(self, home: str, away: str) -> np.ndarray | None:
        """De-vigged [H, D, A] for this fixture, or None if not priced.

        Tries the (home, away) listing first, then the reversed listing with the
        home/away odds swapped (the book may list the tie the other way round).
        """
        h, a = normalize_team_name(home), normalize_team_name(away)
        if (h, a) in self._odds:
            oh, od, oa = self._odds[(h, a)]
        elif (a, h) in self._odds:
            oa, od, oh = self._odds[(a, h)]  # swap: their home is our away
        elif self.odds_provider is not None:
            got = self.odds_provider(h, a)
            if not got:
                return None
            oh, od, oa = got
        else:
            return None
        if not all(np.isfinite([oh, od, oa])) or min(oh, od, oa) <= 1.0:
            return None
        return implied_probabilities(oh, od, oa)

    # -- model interface ---------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "MarketAnchoredModel":
        self.base.fit(train_df)
        return self

    def known_teams(self) -> set[str]:
        return self.base.known_teams() if hasattr(self.base, "known_teams") else set()

    def _score_matrix(
        self, home: str, away: str, context: Mapping[str, Any] | None
    ) -> np.ndarray:
        mat = self.base.score_matrix(home, away, context)
        if self.weight <= 0.0:
            return mat
        market = self._market_1x2(home, away)
        if market is None:  # unpriced fixture -> pure base model
            return mat
        model_1x2 = collapse_1x2(mat)
        blended = (1.0 - self.weight) * model_1x2 + self.weight * market
        return tilt_matrix_to_outcome(mat, blended)
