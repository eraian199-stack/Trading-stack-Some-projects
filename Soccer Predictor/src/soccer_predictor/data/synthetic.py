"""
Synthetic match generators.

These exist so the whole pipeline runs end-to-end with zero downloads, and so
behaviour can be sanity-checked against a KNOWN ground truth. Each team has a
latent attack and defence strength; goals are Poisson with a log-linear mean --
exactly the structure the Dixon-Coles / xG-Poisson models assume -- so a correct
model should recover the true skill ordering and produce well-calibrated
probabilities.

Two generators are provided:

- `generate_league` -- a multi-season club league (double round-robin) WITH
  realistic vig'd bookmaker odds and synthetic xG. The odds are built from the
  TRUE outcome probabilities plus a little noise and an overround, mirroring the
  real closing line (a strong, slightly-biased estimate of the truth). This makes
  the betting backtest honest: beating this synthetic "market" is hard, just like
  the real one.

- `generate_international_history` -- a pool of national teams playing a mix of
  friendlies and World Cup qualifiers on (mostly) neutral ground, for exercising
  the tournament simulator end to end.

Both emit CANONICAL columns (home_score/away_score, home_odds/draw_odds/away_odds,
home_xg/away_xg, neutral_site, competition_type, ...) and pass
`schemas.validate_matches`. The frames are clearly synthetic -- team names are
``Team_NN`` / ``Nation NN`` and competitions are obvious placeholders -- so demo
data is never mistaken for real data (a project non-negotiable).
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from . import schemas
from .normalizers import standardize_matches


# --------------------------------------------------------------------------- #
# True-probability / odds helpers (shared by both generators)
# --------------------------------------------------------------------------- #
def _true_probs(
    lam_h: float, lam_a: float, max_goals: int = 15
) -> tuple[float, float, float]:
    """Exact 1X2 probabilities for two independent Poisson scoring rates."""
    g = np.arange(max_goals + 1)
    # Poisson PMF without a hard scipy.stats dependency in the hot path.
    log_fact = np.cumsum(np.log(np.arange(1, max_goals + 1)))
    log_fact = np.concatenate([[0.0], log_fact])  # 0! = 1
    hp = np.exp(g * np.log(lam_h) - lam_h - log_fact)
    ap = np.exp(g * np.log(lam_a) - lam_a - log_fact)
    mat = np.outer(hp, ap)
    p_h = float(np.tril(mat, -1).sum())
    p_d = float(np.trace(mat))
    p_a = float(np.triu(mat, 1).sum())
    s = p_h + p_d + p_a
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_h / s, p_d / s, p_a / s


def _make_odds(
    p_h: float, p_d: float, p_a: float, vig: float, rng: np.random.Generator
) -> np.ndarray:
    """Vig'd decimal odds (home, draw, away) from true probabilities + noise.

    The bookmaker is good but not an oracle: a small multiplicative noise is
    applied, then an overround so the implied probabilities sum to ``1 + vig``.
    """
    noise = rng.normal(1.0, 0.05, size=3)
    p = np.array([p_h, p_d, p_a]) * noise
    p = np.clip(p, 1e-3, None)
    p = p / p.sum()
    implied = p * (1.0 + vig)  # overround
    odds = 1.0 / implied
    return np.round(odds, 3)


# --------------------------------------------------------------------------- #
# Club league
# --------------------------------------------------------------------------- #
def generate_league(
    n_teams: int = 20,
    n_seasons: int = 6,
    home_advantage: float = 0.30,
    vig: float = 0.05,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic multi-season club league as a canonical match frame.

    Each team has a latent attack/defence strength; goals are Poisson with a
    log-linear mean (home advantage added to the home rate). The frame carries
    realistic vig'd `home_odds/draw_odds/away_odds` and synthetic `home_xg/away_xg`.

    Returns a canonical, date-sorted frame (``competition_type="club_league"``,
    ``neutral_site=False``) that passes ``schemas.validate_matches`` and has a
    ``result`` column. Team names are obvious placeholders -- this is demo data.
    """
    rng = np.random.default_rng(seed)

    # Latent skills, centred so the league mean rate is fixed by `base`.
    attack = rng.normal(0.0, 0.35, size=n_teams)
    defence = rng.normal(0.0, 0.35, size=n_teams)
    attack -= attack.mean()
    defence -= defence.mean()
    base = 0.1  # baseline log-scoring rate

    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    fixtures = list(permutations(range(n_teams), 2))  # home/away double round-robin

    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2018-08-01")
    per_day = max(1, n_teams // 2)  # ~10 fixtures per match-day for a 20-team league

    for season in range(n_seasons):
        order = rng.permutation(len(fixtures))
        season_start = start + pd.Timedelta(days=season * 330)
        season_label = f"{2018 + season}/{2019 + season}"
        for slot, idx in enumerate(order):
            h, a = fixtures[idx]
            match_day = slot // per_day
            date = season_start + pd.Timedelta(days=match_day * 3)

            lam_h = float(np.exp(base + home_advantage + attack[h] - defence[a]))
            lam_a = float(np.exp(base + attack[a] - defence[h]))
            hg = int(rng.poisson(lam_h))
            ag = int(rng.poisson(lam_a))

            # Synthetic xG: a noisy reading of the TRUE scoring rate. In real life
            # xG is a better estimate of true rate than goals are -- this mirrors that.
            home_xg = max(0.0, lam_h * rng.normal(1.0, 0.15) + rng.normal(0, 0.1))
            away_xg = max(0.0, lam_a * rng.normal(1.0, 0.15) + rng.normal(0, 0.1))

            p_h, p_d, p_a = _true_probs(lam_h, lam_a)
            odds = _make_odds(p_h, p_d, p_a, vig, rng)

            rows.append(
                {
                    "match_id": f"SYN-L{season}-{slot}",
                    "date": date,
                    "competition": "Synthetic League",
                    "season": season_label,
                    "home_team": teams[h],
                    "away_team": teams[a],
                    "home_score": hg,
                    "away_score": ag,
                    "home_xg": round(home_xg, 3),
                    "away_xg": round(away_xg, 3),
                    "home_odds": float(odds[0]),
                    "draw_odds": float(odds[1]),
                    "away_odds": float(odds[2]),
                    "neutral_site": False,
                }
            )

    df = pd.DataFrame(rows)
    # Run through the canonical pipeline: fills required cols, coerces, dedupes,
    # sorts by date, and adds the `result` column.
    return standardize_matches(df, competition_type="club_league")


# --------------------------------------------------------------------------- #
# International history
# --------------------------------------------------------------------------- #
# A small spread of synthetic competitions, weighted toward friendlies and
# qualifiers (the day-to-day diet of international football). Names are obvious
# placeholders, except the qualifier label which classify_competition_type and
# the tournament demo expect to see verbatim.
_INTL_COMPETITIONS: tuple[tuple[str, float], ...] = (
    ("Friendly", 0.45),
    ("FIFA World Cup qualification", 0.40),
    ("Continental Championship qualification", 0.10),
    ("Synthetic Nations League", 0.05),
)


def generate_international_history(
    n_teams: int = 48,
    n_matches: int = 1500,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic international match history as a canonical frame.

    A pool of `n_teams` national teams (latent attack/defence strengths) plays
    `n_matches` games spread over several years: a mix of friendlies, World Cup
    qualifiers, and continental qualifiers, mostly on neutral ground. This is
    intended to feed the tournament simulator (ratings, group draws, etc.) end to
    end. The frame passes ``schemas.validate_matches`` and carries a ``result``
    column. Team names ("Nation 00" ...) make clear this is demo data.
    """
    rng = np.random.default_rng(seed)

    attack = rng.normal(0.0, 0.40, size=n_teams)
    defence = rng.normal(0.0, 0.40, size=n_teams)
    attack -= attack.mean()
    defence -= defence.mean()
    base = 0.05  # international scoring rates run a touch lower than club leagues

    teams = [f"Nation {i:02d}" for i in range(n_teams)]
    comp_names = [c for c, _ in _INTL_COMPETITIONS]
    comp_weights = np.array([w for _, w in _INTL_COMPETITIONS], dtype=float)
    comp_weights = comp_weights / comp_weights.sum()

    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2018-01-01")
    for m in range(n_matches):
        h, a = rng.choice(n_teams, size=2, replace=False)
        # International matches are mostly at a neutral venue (host of a window /
        # tournament), but qualifiers carry true home advantage roughly half the
        # time.
        neutral = bool(rng.random() < 0.65)
        comp = comp_names[int(rng.choice(len(comp_names), p=comp_weights))]

        ha = 0.0 if neutral else 0.25  # home advantage only when not neutral
        lam_h = float(np.exp(base + ha + attack[h] - defence[a]))
        lam_a = float(np.exp(base + attack[a] - defence[h]))
        hg = int(rng.poisson(lam_h))
        ag = int(rng.poisson(lam_a))

        # Spread matches roughly evenly across ~7 years on FIFA-window-ish dates.
        date = start + pd.Timedelta(days=int(m * (365 * 7 / max(1, n_matches))))

        rows.append(
            {
                "match_id": f"SYN-I{m}",
                "date": date,
                "competition": comp,
                "home_team": teams[int(h)],
                "away_team": teams[int(a)],
                "home_score": hg,
                "away_score": ag,
                "neutral_site": neutral,
                "city": "Synthetic City",
                "country": "Synthetica",
            }
        )

    df = pd.DataFrame(rows)
    # competition_type is derived per-row from the competition name below, so do
    # not force a single type here.
    from .sources import classify_competition_type

    df["competition_type"] = df["competition"].map(classify_competition_type)
    return standardize_matches(df)
