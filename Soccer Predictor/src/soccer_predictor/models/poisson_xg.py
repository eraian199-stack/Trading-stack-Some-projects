"""
xG-based Poisson model.

Team attack / defence strengths are estimated from *expected* goals rather than
actual goals, turned into Poisson scoring rates, and collapsed into a full score
matrix (so every market falls out). xG is a lower-variance estimate of a team's
true scoring / conceding rate than goals, so over short, recent windows this
often beats a goals-based model. Strengths are shrunk toward the league mean to
stay stable on small samples; unseen teams default to league-average (1.0).

A ``ScorelineModel`` subclass: it implements only ``_score_matrix`` and inherits
``predict_proba`` / ``over_under`` / ``both_teams_to_score`` from the base.
Honours ``context["neutral_site"]`` by dropping the home-advantage factor on
neutral ground. ``XGPoisson`` is kept as an alias of ``PoissonXG``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import base
from .base import ScorelineModel
from ..data import schemas


class PoissonXG(ScorelineModel):
    """Attack / defence strengths from rolling xG, shrunk toward the league mean."""

    def __init__(self, max_goals: int = 10, shrink: float = 6.0):
        self.max_goals = max_goals
        self.shrink = shrink  # pseudo-count pulling strengths toward 1.0
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.league_mean: float = 1.3
        self.home_adv: float = 1.15

    def known_teams(self) -> set[str]:
        """Teams seen during fit (unseen teams fall back to strength 1.0)."""
        return set(self.attack)

    # -- fit ---------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "PoissonXG":
        if not schemas.has_columns(train_df, schemas.XG_COLUMNS):
            raise ValueError(
                "PoissonXG needs home_xg / away_xg columns. Provide a frame with "
                "expected-goals data (e.g. via the xG ingest / merge step)."
            )
        df = train_df.dropna(subset=["home_xg", "away_xg"]).copy()
        if len(df) == 0:
            raise ValueError(
                "PoissonXG needs rows with non-null home_xg / away_xg. "
                "Did the xG merge match any rows?"
            )
        df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
        df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")
        df = df.dropna(subset=["home_xg", "away_xg"])

        all_xg = pd.concat([df["home_xg"], df["away_xg"]])
        self.league_mean = float(all_xg.mean())
        self.home_adv = float(
            df["home_xg"].mean() / max(df["away_xg"].mean(), 1e-6)
        )

        scored: dict[str, list[float]] = {}  # team -> xG created
        conceded: dict[str, list[float]] = {}  # team -> xG conceded
        for r in df.itertuples(index=False):
            scored.setdefault(r.home_team, []).append(float(r.home_xg))
            conceded.setdefault(r.home_team, []).append(float(r.away_xg))
            scored.setdefault(r.away_team, []).append(float(r.away_xg))
            conceded.setdefault(r.away_team, []).append(float(r.home_xg))

        lm = self.league_mean if self.league_mean > 1e-6 else 1.3
        k = self.shrink
        for team in scored:
            n_s = len(scored[team])
            n_c = len(conceded[team])
            raw_att = float(np.mean(scored[team])) / lm
            raw_def = float(np.mean(conceded[team])) / lm
            # Shrink toward 1.0 by k pseudo-games.
            self.attack[team] = (n_s * raw_att + k * 1.0) / (n_s + k)
            self.defence[team] = (n_c * raw_def + k * 1.0) / (n_c + k)
        return self

    # -- rates -------------------------------------------------------------
    def _rates(
        self, home: str, away: str, context: Mapping[str, Any] | None
    ) -> tuple[float, float]:
        ah = self.attack.get(home, 1.0)
        dh = self.defence.get(home, 1.0)
        aa = self.attack.get(away, 1.0)
        da = self.defence.get(away, 1.0)
        # On neutral ground neither side gets the home-advantage tilt.
        adv = 1.0 if (context or {}).get("neutral_site") else self.home_adv
        adv = max(adv, 1e-6)
        lam = self.league_mean * ah * da * np.sqrt(adv)
        mu = self.league_mean * aa * dh / np.sqrt(adv)
        return float(lam), float(mu)

    # -- subclass hook -----------------------------------------------------
    def _score_matrix(
        self, home: str, away: str, context: Mapping[str, Any] | None
    ) -> np.ndarray:
        lam, mu = self._rates(home, away, context)
        return base.score_matrix_from_rates(lam, mu, self.max_goals, rho=0.0)


# Alias kept for the uploaded-framework name.
XGPoisson = PoissonXG
