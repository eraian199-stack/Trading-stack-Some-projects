"""
Model contracts and score-matrix math.

This module is the linchpin every model and the simulator build against. Two
contracts:

    BaseModel        fit(train_df) -> self ; predict_proba(test_df) -> (n, 3)
                     where columns are [P(home win), P(draw), P(away win)].

    ScorelineModel   additionally exposes score_matrix(home, away, context)
                     -> (G+1, G+1) array M where M[i, j] = P(home scores i,
                     away scores j). Every 1X2 / over-under / BTTS / correct-score
                     market is derived from this one matrix, so a subclass only
                     implements `_score_matrix`; everything else is provided.

The simulator consumes ScorelineModel ONLY through score_matrix / expected_goals
/ win_probability_no_draw, so it never hardcodes model logic (a project
non-negotiable).

`context` is an optional dict carrying pre-kickoff facts a model may use:
    neutral_site (bool), competition (str), competition_type (str), date,
    knockout (bool). Models ignore keys they don't care about. Use
    `row_context(row)` to build one from a match-frame row.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import schemas
from ..data.aliases import normalize_team_name

CLASSES: list[str] = schemas.RESULT_CLASSES  # ["H", "D", "A"]


# --------------------------------------------------------------------------- #
# Score-matrix math (shared by every generative model)
# --------------------------------------------------------------------------- #
def poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
    """PMF over 0..max_goals for a Poisson(lam), renormalised to sum to 1."""
    lam = float(np.clip(lam, 1e-4, 25.0))
    g = np.arange(max_goals + 1)
    log_pmf = g * np.log(lam) - lam - _gammaln(g + 1.0)
    pmf = np.exp(log_pmf)
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def _gammaln(x: np.ndarray) -> np.ndarray:
    """log-gamma without a hard SciPy dependency (Lanczos approximation)."""
    try:  # SciPy is a core dependency, but stay importable if it's absent.
        from scipy.special import gammaln as _sp_gammaln

        return _sp_gammaln(x)
    except Exception:  # pragma: no cover - fallback path
        g = 7.0
        c = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
        ]
        x = np.asarray(x, dtype=float) - 1.0
        a = np.full_like(x, c[0])
        t = x + g + 0.5
        for i in range(1, len(c)):
            a = a + c[i] / (x + i)
        return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


def score_matrix_from_rates(
    home_rate: float,
    away_rate: float,
    max_goals: int = 10,
    rho: float = 0.0,
) -> np.ndarray:
    """Independent-Poisson score matrix with the Dixon-Coles low-score tweak.

    rho applies the 1997 Dixon-Coles correction to the four low-score cells
    (0-0, 0-1, 1-0, 1-1), which fixes independent Poisson's tendency to
    under-rate those. rho == 0 disables it. The matrix is renormalised to 1.

    If rho is large relative to the rates, a correction factor can go negative
    (e.g. ``1 - lam*mu*rho < 0``). Rather than silently clip that cell to 0 --
    which distorts the low-score distribution -- we detect it and skip the DC
    correction for this fixture, falling back to independent Poisson (still a
    valid distribution). Realistic fitted rho (small, negative) never trips this.
    """
    home_pmf = poisson_pmf(home_rate, max_goals)
    away_pmf = poisson_pmf(away_rate, max_goals)
    mat = np.outer(home_pmf, away_pmf)
    if rho:
        lam = float(np.clip(home_rate, 1e-4, 25.0))
        mu = float(np.clip(away_rate, 1e-4, 25.0))
        f00 = 1.0 - lam * mu * rho
        f01 = 1.0 + lam * rho
        f10 = 1.0 + mu * rho
        f11 = 1.0 - rho
        if min(f00, f01, f10, f11) > 0.0:  # correction valid for these rates
            mat[0, 0] *= f00
            mat[0, 1] *= f01
            mat[1, 0] *= f10
            mat[1, 1] *= f11
        # else: leave the independent-Poisson cells untouched (no silent zeroing)
    mat = np.clip(mat, 0.0, None)
    total = mat.sum()
    return mat / total if total > 0 else mat


def collapse_1x2(mat: np.ndarray) -> np.ndarray:
    """[P(home win), P(draw), P(away win)] from a score matrix."""
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    out = np.array([p_home, p_draw, p_away], dtype=float)
    s = out.sum()
    return out / s if s > 0 else out


def markets_from_matrix(mat: np.ndarray, ou_line: float = 2.5) -> dict[str, float]:
    """All standard derived markets from a score matrix."""
    p = collapse_1x2(mat)
    totals = np.add.outer(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    over = float(mat[totals > ou_line].sum())
    btts = float(mat[1:, 1:].sum())
    return {
        "home_win": float(p[0]),
        "draw": float(p[1]),
        "away_win": float(p[2]),
        f"over_{ou_line}": over,
        f"under_{ou_line}": 1.0 - over,
        "btts_yes": btts,
        "btts_no": 1.0 - btts,
    }


def tilt_matrix_to_outcome(mat: np.ndarray, target_1x2: Sequence[float]) -> np.ndarray:
    """Reweight a score matrix so its 1X2 margins match `target_1x2`.

    Used by rating-based models: a Poisson matrix gives sensible scorelines, but
    we want the win/draw/away masses to match an Elo-derived 1X2 estimate.
    Preserves the conditional score distribution within each outcome region.
    """
    target = np.asarray(target_1x2, dtype=float)
    target = target / target.sum() if target.sum() > 0 else target
    home_mask = np.fromfunction(lambda h, a: h > a, mat.shape)
    draw_mask = np.fromfunction(lambda h, a: h == a, mat.shape)
    away_mask = np.fromfunction(lambda h, a: h < a, mat.shape)
    masks = [home_mask, draw_mask, away_mask]
    current = np.array([mat[m].sum() for m in masks])
    tilted = mat.copy()
    for idx, mask in enumerate(masks):
        if current[idx] > 1e-12:
            tilted[mask] *= target[idx] / current[idx]
    total = tilted.sum()
    return tilted / total if total > 0 else tilted


# --------------------------------------------------------------------------- #
# Context helpers
# --------------------------------------------------------------------------- #
def row_context(row: Any) -> dict[str, Any]:
    """Build a model context dict from a match-frame row (Series or namedtuple)."""

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(row, Mapping):
            return row.get(name, default)
        return getattr(row, name, default)

    neutral = _get("neutral_site", False)
    try:
        neutral = bool(neutral) if not pd.isna(neutral) else False
    except (TypeError, ValueError):
        neutral = bool(neutral)
    return {
        "neutral_site": neutral,
        "competition": _get("competition", ""),
        "competition_type": _get("competition_type", ""),
        "date": _get("date", None),
        "knockout": bool(_get("knockout", False)),
    }


# --------------------------------------------------------------------------- #
# Base contracts
# --------------------------------------------------------------------------- #
class BaseModel:
    """Minimal model contract: fit + predict_proba (n, 3) over [H, D, A]."""

    def fit(self, train_df: pd.DataFrame) -> "BaseModel":  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class ScorelineModel(BaseModel):
    """Generative model exposing a full score matrix.

    Subclasses implement `_score_matrix(home, away, context) -> np.ndarray`
    (raw, already normalised). Everything else -- predict_proba, over/under,
    BTTS, expected goals, sampling, penalty resolution -- is derived here.
    """

    max_goals: int = 10

    # -- subclass hook -----------------------------------------------------
    def _score_matrix(
        self, home: str, away: str, context: Mapping[str, Any] | None
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # -- public score matrix ----------------------------------------------
    def score_matrix(
        self,
        home: str,
        away: str,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        mat = self._score_matrix(
            normalize_team_name(home), normalize_team_name(away), context or {}
        )
        mat = np.clip(np.asarray(mat, dtype=float), 0.0, None)
        total = mat.sum()
        return mat / total if total > 0 else mat

    # -- derived markets ---------------------------------------------------
    def outcome_probs(
        self, home: str, away: str, context: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        return collapse_1x2(self.score_matrix(home, away, context))

    def over_under(
        self,
        home: str,
        away: str,
        line: float = 2.5,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[float, float]:
        mat = self.score_matrix(home, away, context)
        totals = np.add.outer(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
        over = float(mat[totals > line].sum())
        return over, 1.0 - over

    def both_teams_to_score(
        self, home: str, away: str, context: Mapping[str, Any] | None = None
    ) -> tuple[float, float]:
        mat = self.score_matrix(home, away, context)
        yes = float(mat[1:, 1:].sum())
        return yes, 1.0 - yes

    def expected_goals(
        self, home: str, away: str, context: Mapping[str, Any] | None = None
    ) -> tuple[float, float]:
        mat = self.score_matrix(home, away, context)
        goals = np.arange(mat.shape[0])
        eg_home = float((goals[:, None] * mat).sum())
        eg_away = float((goals[None, :] * mat).sum())
        return eg_home, eg_away

    def top_scorelines(
        self,
        home: str,
        away: str,
        k: int = 8,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        mat = self.score_matrix(home, away, context)
        flat = [
            (float(mat[i, j]), i, j)
            for i in range(mat.shape[0])
            for j in range(mat.shape[1])
        ]
        flat.sort(reverse=True)
        return {f"{i}-{j}": p for p, i, j in flat[:k]}

    def most_likely_score(
        self, home: str, away: str, context: Mapping[str, Any] | None = None
    ) -> str:
        mat = self.score_matrix(home, away, context)
        i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
        return f"{int(i)}-{int(j)}"

    # -- sampling (for the Monte Carlo simulator) --------------------------
    def sample_score(
        self,
        home: str,
        away: str,
        rng: np.random.Generator,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[int, int]:
        mat = self.score_matrix(home, away, context)
        flat = mat.ravel()
        flat = flat / flat.sum()
        idx = int(rng.choice(flat.size, p=flat))
        i, j = np.unravel_index(idx, mat.shape)
        return int(i), int(j)

    def win_probability_no_draw(
        self, home: str, away: str, context: Mapping[str, Any] | None = None
    ) -> float:
        """P(home wins | not a draw) -- used to resolve penalty shootouts."""
        p = self.outcome_probs(home, away, context)
        denom = p[0] + p[2]
        return float(p[0] / denom) if denom > 1e-12 else 0.5

    # -- default predict_proba via the matrix ------------------------------
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(test_df), 3), dtype=float)
        for i, row in enumerate(test_df.itertuples(index=False)):
            ctx = row_context(row)
            out[i] = self.outcome_probs(row.home_team, row.away_team, ctx)
        sums = out.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return out / sums


# --------------------------------------------------------------------------- #
# Convenience: all markets + fair odds for one fixture (any ScorelineModel)
# --------------------------------------------------------------------------- #
def predict_markets(
    model: ScorelineModel,
    home: str,
    away: str,
    context: Mapping[str, Any] | None = None,
    ou_line: float = 2.5,
    top_k: int = 8,
) -> dict[str, Any]:
    """Full prediction bundle for one fixture: 1X2, xG, O/U, BTTS, top scores,
    most-likely score, and fair decimal odds (1 / probability)."""
    mat = model.score_matrix(home, away, context)
    markets = markets_from_matrix(mat, ou_line)
    eg_home, eg_away = model.expected_goals(home, away, context)
    p_home, p_draw, p_away = markets["home_win"], markets["draw"], markets["away_win"]

    def _fair(p: float) -> float:
        return float(1.0 / p) if p > 1e-9 else float("inf")

    i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
    return {
        "home_team": normalize_team_name(home),
        "away_team": normalize_team_name(away),
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
        "expected_home_goals": eg_home,
        "expected_away_goals": eg_away,
        "most_likely_score": f"{int(i)}-{int(j)}",
        f"over_{ou_line}": markets[f"over_{ou_line}"],
        f"under_{ou_line}": markets[f"under_{ou_line}"],
        "btts_yes": markets["btts_yes"],
        "btts_no": markets["btts_no"],
        "fair_home_odds": _fair(p_home),
        "fair_draw_odds": _fair(p_draw),
        "fair_away_odds": _fair(p_away),
        "scoreline_probabilities": model.top_scorelines(home, away, top_k, context),
    }
