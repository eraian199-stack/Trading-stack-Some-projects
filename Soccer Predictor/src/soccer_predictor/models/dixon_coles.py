"""
Dixon-Coles generative goal model.

Each team carries an attack and a defence parameter; goals are modelled as
Poisson with the 1997 Dixon-Coles low-score dependence correction (rho) and an
optional exponential time-decay so recent matches count more. One MLE fit yields
EVERY market at once -- 1X2, over/under, BTTS, correct score -- because the
score-line probability matrix is the single source of truth.

This is a ``ScorelineModel`` subclass: it implements only ``_score_matrix`` and
inherits ``predict_proba`` / ``over_under`` / ``both_teams_to_score`` from the
base, so the over/under line argument is always respected. Honours
``context["neutral_site"]`` by dropping the home-advantage term on neutral ground
(international tournament matches).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from . import base
from .base import ScorelineModel
from .market_baseline import MarketBaseline


def _dc_tau(
    hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> np.ndarray:
    """Low-score dependence correction from Dixon & Coles (1997)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau = np.where(m00, 1 - lam * mu * rho, tau)
    tau = np.where(m01, 1 + lam * rho, tau)
    tau = np.where(m10, 1 + mu * rho, tau)
    tau = np.where(m11, 1 - rho, tau)
    return tau


class DixonColes(ScorelineModel):
    """Time-weighted Dixon-Coles maximum-likelihood goal model."""

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        """``xi`` is the time-decay rate per day; 0 disables decay. ``xi`` ~
        0.0018 gives a half-life of roughly one year."""
        self.xi = xi
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.idx: dict[str, int] = {}
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.home_adv: float = 0.0
        self.rho: float = 0.0

    def known_teams(self) -> set[str]:
        """Teams seen during fit (unseen teams fall back to league average)."""
        return set(self.idx)

    # -- fit ---------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "DixonColes":
        df = train_df.dropna(subset=["home_score", "away_score"]).copy()
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = df["home_team"].map(self.idx).to_numpy()
        a = df["away_team"].map(self.idx).to_numpy()
        hg = pd.to_numeric(df["home_score"], errors="coerce").to_numpy(dtype=float)
        ag = pd.to_numeric(df["away_score"], errors="coerce").to_numpy(dtype=float)

        # Exponential time-decay weights relative to the most recent match.
        t = df["date"].to_numpy(dtype="datetime64[D]")
        age_days = (t.max() - t) / np.timedelta64(1, "D")
        weights = np.exp(-self.xi * age_days.astype(float))

        # Params: [attack(n), defence(n), home_adv, rho].
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [0.0]])

        def neg_ll(params: np.ndarray) -> float:
            attack = params[:n]
            defence = params[n : 2 * n]
            home_adv = params[-2]
            rho = params[-1]
            lam = np.exp(home_adv + attack[h] - defence[a])
            mu = np.exp(attack[a] - defence[h])
            core = (
                hg * np.log(lam) - lam - gammaln(hg + 1)
                + ag * np.log(mu) - mu - gammaln(ag + 1)
            )
            tau = np.maximum(_dc_tau(hg, ag, lam, mu, rho), 1e-10)
            ll = weights * (core + np.log(tau))
            return -ll.sum()

        # Identifiability: pin the sum of attack strengths to zero.
        constraints = ({"type": "eq", "fun": lambda p: p[:n].sum()},)
        bounds = [(-3, 3)] * (2 * n) + [(-1, 2), (-0.2, 0.2)]
        res = minimize(
            neg_ll,
            x0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-7},
        )

        p = res.x
        self.attack = p[:n]
        self.defence = p[n : 2 * n]
        self.home_adv = float(p[-2])
        self.rho = float(p[-1])
        return self

    # -- rates -------------------------------------------------------------
    def _rates(
        self, home: str, away: str, context: Mapping[str, Any] | None
    ) -> tuple[float, float]:
        """Expected goals for a fixture. Unseen teams default to league-average
        (attack = defence = 0). Neutral venues drop the home-advantage term."""
        ah = self.attack[self.idx[home]] if home in self.idx else 0.0
        aa = self.attack[self.idx[away]] if away in self.idx else 0.0
        dh = self.defence[self.idx[home]] if home in self.idx else 0.0
        da = self.defence[self.idx[away]] if away in self.idx else 0.0
        home_adv = 0.0 if (context or {}).get("neutral_site") else self.home_adv
        lam = float(np.exp(home_adv + ah - da))
        mu = float(np.exp(aa - dh))
        return lam, mu

    # -- subclass hook -----------------------------------------------------
    def _score_matrix(
        self, home: str, away: str, context: Mapping[str, Any] | None
    ) -> np.ndarray:
        if self.attack is None:
            raise RuntimeError("DixonColes must be fit before scoring.")
        lam, mu = self._rates(home, away, context)
        return base.score_matrix_from_rates(lam, mu, self.max_goals, self.rho)

    # -- market baseline passthrough --------------------------------------
    def predict_market_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Convenience: the de-vigged market probabilities for the same rows."""
        return MarketBaseline().predict_proba(test_df)
