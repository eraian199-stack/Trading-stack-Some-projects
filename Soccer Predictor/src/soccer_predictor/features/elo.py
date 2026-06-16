"""
Elo rating engine tuned for football/soccer.

Key design choices (these matter a lot):
- Home advantage is added to the home team's rating when computing the expected
  result, NOT baked into the stored rating. On neutral ground (international
  tournaments) the advantage is dropped via the ``neutral`` argument.
- The margin of victory inflates the rating update (a 4-0 win shifts ratings
  more than a 1-0 win). This is the single biggest improvement over textbook
  Elo for football and is borrowed from the World Football Elo / 538 approach.
- Ratings are updated sequentially in date order. For every match we store the
  PRE-match rating of each team, which is what you feed into a model. Using the
  post-match rating would leak the result you are trying to predict.

Ported from the project's original ``elo.py`` with the ``neutral`` argument
added so home advantage is suppressed on neutral ground.
"""

from __future__ import annotations

from ..data.aliases import normalize_team_name


def goal_diff_multiplier(goal_diff: int) -> float:
    """Inflate the K-factor based on margin of victory.

    Mirrors the World Football Elo Ratings scheme:
      gd 1 -> 1.00, gd 2 -> 1.50, gd 3 -> 1.75, gd>=4 -> 1.75 + (gd-3)/8
    """
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


def competition_importance(competition: object) -> float:
    """Match-importance multiplier on the Elo K-factor (World-Football-Elo style).

    A World Cup result should move ratings more than a friendly. Keyed on the
    competition NAME (robust across feeds), in [0.5, 1.0]. Unknown competitions
    default to 0.7 (a generic competitive match). Used so the Elo "run-in" weights
    meaningful games (qualifiers, continental cups, Nations League) above friendlies.
    """
    name = str(competition or "").strip().lower()
    if not name:
        return 0.7
    if "world cup" in name and ("qualif" in name or "qualifier" in name):
        return 0.85
    if "world cup" in name:
        return 1.0
    if any(k in name for k in ("euro", "copa am", "copa á", "cup of nations",
                               "african cup", "asian cup", "gold cup", "nations league")):
        return 0.9 if ("qualif" not in name) else 0.8
    if "qualif" in name or "qualifier" in name:
        return 0.8
    if "friendly" in name:
        return 0.5
    return 0.7


class EloModel:
    """Sequential Elo rating engine (margin-of-victory weighted)."""

    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 65.0,
        base_rating: float = 1500.0,
        scale: float = 400.0,
    ):
        self.k = float(k)
        self.home_advantage = float(home_advantage)
        self.base_rating = float(base_rating)
        self.scale = float(scale)
        # Plain dict (NOT defaultdict with a lambda): a local-lambda factory makes
        # the whole model unpicklable. Unseen teams fall back to base_rating via
        # .get() on read; update() creates entries explicitly.
        self.ratings: dict[str, float] = {}

    def expected_home(self, home: str, away: str, neutral: bool = False) -> float:
        """Probability-like expected score for the home team in [0, 1].

        On neutral ground the home-advantage term is dropped.
        """
        home = normalize_team_name(home)
        away = normalize_team_name(away)
        advantage = 0.0 if neutral else self.home_advantage
        home_rating = self.ratings.get(home, self.base_rating)
        away_rating = self.ratings.get(away, self.base_rating)
        diff = (home_rating + advantage) - away_rating
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
        """Apply one result to both teams' ratings (in date order).

        ``weight`` scales the K-factor by match importance (e.g.
        ``competition_importance(competition)``) so a World Cup game moves ratings
        more than a friendly. Default 1.0 preserves the original behaviour.
        """
        home = normalize_team_name(home)
        away = normalize_team_name(away)
        exp_home = self.expected_home(home, away, neutral=neutral)
        if home_score > away_score:
            actual = 1.0
        elif home_score == away_score:
            actual = 0.5
        else:
            actual = 0.0
        mult = goal_diff_multiplier(int(home_score) - int(away_score))
        delta = self.k * mult * float(weight) * (actual - exp_home)
        self.ratings[home] = self.ratings.get(home, self.base_rating) + delta
        self.ratings[away] = self.ratings.get(away, self.base_rating) - delta

    def rating(self, team: str) -> float:
        """Current stored rating for a team (base_rating if unseen)."""
        return self.ratings.get(normalize_team_name(team), self.base_rating)
