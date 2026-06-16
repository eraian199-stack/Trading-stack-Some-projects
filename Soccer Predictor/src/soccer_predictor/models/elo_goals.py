"""
Elo-driven scoreline model (the robust minimum-viable default).

This is the model the tournament simulator falls back to when nothing richer is
supplied. It is deliberately simple and never blows up on unseen teams, so it can
score any fixture in any bracket:

    1. An internal ``EloModel`` is fit over the training frame in date order
       (pre-kickoff ratings only, home advantage dropped on neutral ground).
    2. For a fixture, the Elo win expectation gives a *home-vs-away* split, which
       is spread into a 1X2 target by carving out a draw mass that shrinks as the
       rating gap widens (close games draw more often than mismatches).
    3. A two-Poisson score matrix is built around a league-average goal sum,
       tilted toward the stronger side, then reweighted with
       ``base.tilt_matrix_to_outcome`` so its 1X2 margins match the Elo target.

A ``ScorelineModel`` subclass: implements only ``_score_matrix`` and inherits all
the market math. The resulting matrix always sums to 1 and reflects the Elo
strength difference (stronger team gets more mass below the diagonal).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import base
from .base import ScorelineModel

# Optional dependency on the features package; degrade to a local fallback so the
# model stays importable and usable even before features.elo is written.
try:  # pragma: no cover - import wiring
    from ..features.elo import EloModel, competition_importance
except ImportError:  # pragma: no cover - fallback path
    EloModel = None  # type: ignore[assignment]

    def competition_importance(competition: object) -> float:  # type: ignore[misc]
        return 1.0


class _FallbackElo:
    """Minimal World-Football-Elo engine matching the features.elo contract.

    Only used when ``features.elo`` is not yet importable. Mirrors its public
    surface: expected_home / update / rating, all honouring a ``neutral`` flag.
    """

    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 65.0,
        base_rating: float = 1500.0,
        scale: float = 400.0,
    ):
        self.k = k
        self.home_advantage = home_advantage
        self.base_rating = base_rating
        self.scale = scale
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    def expected_home(self, home: str, away: str, neutral: bool = False) -> float:
        adv = 0.0 if neutral else self.home_advantage
        diff = (self.rating(home) + adv) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / self.scale))

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        weight: float = 1.0,
    ) -> None:
        exp_home = self.expected_home(home, away, neutral=neutral)
        if home_score > away_score:
            actual = 1.0
        elif home_score == away_score:
            actual = 0.5
        else:
            actual = 0.0
        gd = abs(int(home_score) - int(away_score))
        if gd <= 1:
            mult = 1.0
        elif gd == 2:
            mult = 1.5
        elif gd == 3:
            mult = 1.75
        else:
            mult = 1.75 + (gd - 3) / 8.0
        delta = self.k * mult * float(weight) * (actual - exp_home)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta


class EloGoalsModel(ScorelineModel):
    """Score matrix derived from Elo win expectation + a league goal level."""

    def __init__(
        self,
        elo_kwargs: dict | None = None,
        max_goals: int = 10,
        base_goal_sum: float = 2.6,
        supremacy_slope: float = 0.75,
        max_supremacy: float = 2.3,
        rho: float = -0.05,
        *,
        use_importance: bool = False,
        form_weight: float = 0.0,
        form_window_days: float = 730.0,
        form_decay_days: float = 365.0,
        pedigree_weight: float = 0.0,
        pedigree_k: float = 6.0,
        squad_strength: dict[str, float] | None = None,
        squad_weight: float = 0.0,
        home_advantage_override: float | None = None,
    ):
        """Map a national-team *effective rating* onto a goal supremacy and read
        every market off the resulting Dixon-Coles score matrix.

        The base signal is a margin- and (optionally) importance-weighted Elo --
        which already blends long-run pedigree with recent results. On top of it,
        several OPTIONAL components (all default OFF, all backtest-tunable) add
        orthogonal information to the effective rating, in Elo points:

        - ``form_weight`` x recent FORM: trailing-window, importance- and time-
          decayed over/under-performance vs the team's Elo expectation (the
          "two-year run-in"). Captures hot/cold form distinct from level.
        - ``pedigree_weight`` x PEDIGREE: a slow (small-K) Elo minus the fast Elo,
          i.e. mean-reversion toward long-run strength. (Expected to add little --
          Elo already encodes pedigree; the backtest confirms.)
        - ``squad_weight`` x SQUAD strength: an external per-team rating (e.g.
          FotMob player ratings / market value), centred to a zero-mean Elo
          adjustment. CURRENT-tournament only -- not available historically, so it
          cannot be backtested on past World Cups.

        ``supremacy_slope`` / ``max_supremacy`` / ``rho`` shape the score matrix
        (see the H2 fix: deriving 1X2 from the goal distribution floors draws and
        longshots even on extreme mismatches).

        EMPIRICAL NOTE: backtested match-by-match on World Cups 2006-2022, NONE of
        the optional components improved out-of-sample log loss -- plain margin-
        weighted Elo was best, recent form slightly hurt, and pedigree/importance-
        weighting hurt more. Hence every component defaults OFF. They remain
        available (and squad strength is a live-only overlay that cannot be
        backtested), but the parsimonious default is the one the evidence supports.
        """
        self.elo_kwargs = elo_kwargs or {}
        self.max_goals = max_goals
        self.base_goal_sum = base_goal_sum
        self.supremacy_slope = supremacy_slope
        self.max_supremacy = max_supremacy
        self.rho = rho
        self.use_importance = use_importance
        self.form_weight = form_weight
        self.form_window_days = form_window_days
        self.form_decay_days = form_decay_days
        self.pedigree_weight = pedigree_weight
        self.pedigree_k = pedigree_k
        self.squad_weight = squad_weight
        self.squad_strength = self._center_squad(squad_strength)
        self.home_advantage_override = home_advantage_override
        self.elo = self._new_elo()
        self.elo_slow = None
        self.form: dict[str, float] = {}

    @staticmethod
    def _center_squad(squad: dict[str, float] | None) -> dict[str, float]:
        if not squad:
            return {}
        from ..data.aliases import normalize_team_name

        vals = list(squad.values())
        mean = sum(vals) / len(vals)
        return {normalize_team_name(k): float(v) - mean for k, v in squad.items()}

    def known_teams(self) -> set[str]:
        """Teams seen during fit (for the betting-edge known-team guard)."""
        return set(self.elo.ratings)

    def _new_elo(self, k_override: float | None = None):
        cls = EloModel if EloModel is not None else _FallbackElo
        kwargs = dict(self.elo_kwargs)
        if k_override is not None:
            kwargs["k"] = k_override
        return cls(**kwargs)

    @property
    def _home_adv(self) -> float:
        if self.home_advantage_override is not None:
            return float(self.home_advantage_override)
        return float(getattr(self.elo, "home_advantage", 65.0))

    @property
    def _scale(self) -> float:
        return float(getattr(self.elo, "scale", 400.0))

    # -- fit ---------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "EloGoalsModel":
        self.elo = self._new_elo()
        self.elo_slow = self._new_elo(self.pedigree_k) if self.pedigree_weight else None
        df = train_df.dropna(subset=["home_score", "away_score"]).copy()
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

        track_form = self.form_weight != 0.0
        residuals: dict[str, list[tuple[pd.Timestamp, float, float]]] = {}
        last_date = df["date"].max() if len(df) else None

        for row in df.itertuples(index=False):
            neutral = bool(getattr(row, "neutral_site", False))
            weight = (
                competition_importance(getattr(row, "competition", ""))
                if self.use_importance
                else 1.0
            )
            if track_form:
                # Residual vs the CURRENT (pre-update) Elo expectation = form signal.
                exp_home = self.elo.expected_home(
                    row.home_team, row.away_team, neutral=neutral
                )
                actual_home = (
                    1.0 if row.home_score > row.away_score
                    else 0.5 if row.home_score == row.away_score else 0.0
                )
                res_home = actual_home - float(exp_home)
                from ..data.aliases import normalize_team_name

                residuals.setdefault(normalize_team_name(row.home_team), []).append(
                    (row.date, res_home, weight)
                )
                residuals.setdefault(normalize_team_name(row.away_team), []).append(
                    (row.date, -res_home, weight)
                )
            self.elo.update(
                row.home_team, row.away_team,
                int(row.home_score), int(row.away_score),
                neutral=neutral, weight=weight,
            )
            if self.elo_slow is not None:
                self.elo_slow.update(
                    row.home_team, row.away_team,
                    int(row.home_score), int(row.away_score),
                    neutral=neutral, weight=weight,
                )

        if track_form and last_date is not None:
            self.form = self._summarise_form(residuals, last_date)
        return self

    def _summarise_form(
        self, residuals: dict, last_date: pd.Timestamp
    ) -> dict[str, float]:
        """Trailing-window, importance- and time-decayed mean residual per team."""
        out: dict[str, float] = {}
        cutoff = last_date - pd.Timedelta(days=self.form_window_days)
        for team, recs in residuals.items():
            num = den = 0.0
            for date, res, imp in recs:
                if date < cutoff:
                    continue
                age = (last_date - date).days
                w = imp * np.exp(-age / max(self.form_decay_days, 1.0))
                num += w * res
                den += w
            out[team] = (num / den) if den > 0 else 0.0
        return out

    def observe(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        competition: str = "",
    ) -> None:
        """Update the Elo state with one new result (for within-tournament
        backtesting / live updating). Form is fixed at fit time, so it is not
        recomputed here -- only the level ratings move."""
        weight = competition_importance(competition) if self.use_importance else 1.0
        self.elo.update(home, away, int(home_score), int(away_score),
                        neutral=neutral, weight=weight)
        if self.elo_slow is not None:
            self.elo_slow.update(home, away, int(home_score), int(away_score),
                                 neutral=neutral, weight=weight)

    # -- effective rating (Elo + optional components, in Elo points) -------
    def effective_rating(self, team: str) -> float:
        from ..data.aliases import normalize_team_name

        t = normalize_team_name(team)
        rating = self.elo.rating(t)
        if self.form_weight and t in self.form:
            rating += self.form_weight * self.form[t]
        if self.pedigree_weight and self.elo_slow is not None:
            rating += self.pedigree_weight * (self.elo_slow.rating(t) - self.elo.rating(t))
        if self.squad_weight and t in self.squad_strength:
            rating += self.squad_weight * self.squad_strength[t]
        return rating

    # -- expected goal rates from the effective rating --------------------
    def _rates(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Expected (home, away) goal rates from the effective-rating difference.

        The win expectation ``e`` is turned into a goal *supremacy* via the logit,
        scaled and capped, then split across the league goal sum -- the cap keeps
        the underdog's rate non-trivial so longshot wins / draws stay floored.
        """
        adv = 0.0 if neutral else self._home_adv
        diff = (self.effective_rating(home) + adv) - self.effective_rating(away)
        e = 1.0 / (1.0 + 10 ** (-diff / self._scale))
        e = min(max(e, 1e-6), 1.0 - 1e-6)
        supremacy = self.supremacy_slope * float(np.log(e / (1.0 - e)))
        supremacy = float(np.clip(supremacy, -self.max_supremacy, self.max_supremacy))
        lam = max((self.base_goal_sum + supremacy) / 2.0, 0.05)
        mu = max((self.base_goal_sum - supremacy) / 2.0, 0.05)
        return lam, mu

    # -- subclass hook -----------------------------------------------------
    def _score_matrix(
        self, home: str, away: str, context: Mapping[str, Any] | None
    ) -> np.ndarray:
        neutral = bool((context or {}).get("neutral_site"))
        lam, mu = self._rates(home, away, neutral)
        # 1X2, draw and longshot masses all fall out of this distribution -- no
        # artificial draw carving, so nothing collapses to zero on a mismatch.
        return base.score_matrix_from_rates(lam, mu, self.max_goals, rho=self.rho)
