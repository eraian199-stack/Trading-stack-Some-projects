"""
The central, no-leakage feature builder.

THE most important rule in football prediction: every feature for a match may
only use information available BEFORE kick-off. People fool themselves into 70%+
"accuracy" by leaking post-match info (computing a season average that includes
the match being predicted, or using random k-fold splits that mix the future
into training).

We enforce this by walking matches in strict date order. For each match we read
each team's history of EARLIER matches only and record those features; the
match's own outcome is appended to the running state AFTER its features are
written. The same discipline applies to Elo (pre-match ratings are stored, then
updated) and to rolling form / xG (computed over the team's prior games).

Base features (always produced) -- ``FEATURE_COLUMNS``:
    elo_home, elo_away, elo_diff           pre-match Elo (neutral-aware)
    home_form_pts, away_form_pts           points per game over last N matches
    home_gf, home_ga, away_gf, away_ga     rolling goals for/against per game
    home_rest_days, away_rest_days         days since each team last played
    home_played, away_played               matches in history (data sufficiency)
    neutral_site                           pre-kickoff neutral-ground flag

Optional groups are attached only when their source columns are present and
non-null in the input:
    XG_FEATURE_COLUMNS        rolling xG-for / xG-against (needs home_xg/away_xg)
    MARKET_FEATURE_COLUMNS    de-vigged market-implied probs (needs odds triple)
    SQUAD_FEATURE_COLUMNS     squad value / injury / FIFA-rank differentials
    INTL_FEATURE_COLUMNS      neutral / host-nation flags

Ported from the project's original ``features.py`` but using canonical column
names, neutral-aware Elo, and the optional column groups above.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from ..data import schemas
from .elo import EloModel
from .form import points_for
from .international import (
    INTL_FEATURE_COLUMNS,
    add_international_features,
    available_international_columns,
)
from .market import MARKET_FEATURE_COLUMNS, add_market_features
from .rolling import rolling_form, rolling_mean
from .squad import (
    SQUAD_FEATURE_COLUMNS,
    add_squad_features,
    available_squad_columns,
)
from .xg import XG_ROLL_COLUMNS

# Base feature set, always produced (CONTRACTS.md). Includes the neutral flag.
FEATURE_COLUMNS: list[str] = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "home_form_pts",
    "away_form_pts",
    "home_gf",
    "home_ga",
    "away_gf",
    "away_ga",
    "home_rest_days",
    "away_rest_days",
    "home_played",
    "away_played",
    "neutral_site",
]

# Re-exported optional groups (the rolling-xG names live in xg.XG_ROLL_COLUMNS).
XG_FEATURE_COLUMNS: list[str] = list(XG_ROLL_COLUMNS)
MARKET_FEATURE_COLUMNS = list(MARKET_FEATURE_COLUMNS)


def _has_xg(df: pd.DataFrame) -> bool:
    return (
        "home_xg" in df.columns
        and "away_xg" in df.columns
        and pd.to_numeric(df["home_xg"], errors="coerce").notna().any()
    )


def _has_market(df: pd.DataFrame) -> bool:
    return schemas.has_usable_odds(df)


def build_features(
    df: pd.DataFrame,
    form_window: int = 6,
    elo_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Return ``df`` with leakage-safe pre-kickoff features attached.

    Walks matches in date order, recording each team's pre-match Elo, rolling
    form (and xG, if present), rest days and games-played BEFORE revealing the
    match outcome. Elo updates honour ``neutral_site``. Adds a ``result`` column
    if absent. Optional feature groups are attached when their source columns are
    present and non-null. The input is not mutated; a sorted copy is returned.

    NO FEATURE USES FUTURE INFORMATION: appending later matches and rebuilding
    leaves any earlier row's features unchanged.
    """
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    if "result" not in df.columns:
        df = schemas.add_result_column(df)

    elo = EloModel(**(elo_kwargs or {}))
    has_xg = _has_xg(df)

    # Per-team running state (built only from already-seen matches).
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=form_window))
    last_played: dict[str, pd.Timestamp] = {}
    games_seen: dict[str, int] = defaultdict(int)

    base_cols = [c for c in FEATURE_COLUMNS if c != "neutral_site"]
    active_cols = base_cols + (XG_FEATURE_COLUMNS if has_xg else [])
    feats = {c: np.full(len(df), np.nan) for c in active_cols}
    neutral_flags = np.zeros(len(df), dtype=bool)

    # Pre-extract optional source arrays once (numpy-2/pandas-3 safe).
    neutral_col = (
        df["neutral_site"].map(schemas.coerce_bool).to_numpy()
        if "neutral_site" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    if has_xg:
        home_xg = pd.to_numeric(df["home_xg"], errors="coerce").to_numpy()
        away_xg = pd.to_numeric(df["away_xg"], errors="coerce").to_numpy()

    for i, row in enumerate(df.itertuples(index=False)):
        h = row.home_team
        a = row.away_team
        date = row.date
        neutral = bool(neutral_col[i])
        neutral_flags[i] = neutral

        # ----- pre-match features (no leakage) -----
        feats["elo_home"][i] = elo.rating(h)
        feats["elo_away"][i] = elo.rating(a)
        feats["elo_diff"][i] = elo.rating(h) - elo.rating(a)

        hp, hgf, hga = rolling_form(history[h])
        ap, agf, aga = rolling_form(history[a])
        feats["home_form_pts"][i] = hp
        feats["away_form_pts"][i] = ap
        feats["home_gf"][i] = hgf
        feats["home_ga"][i] = hga
        feats["away_gf"][i] = agf
        feats["away_ga"][i] = aga

        feats["home_rest_days"][i] = (
            float((date - last_played[h]).days) if h in last_played else np.nan
        )
        feats["away_rest_days"][i] = (
            float((date - last_played[a]).days) if a in last_played else np.nan
        )

        feats["home_played"][i] = games_seen[h]
        feats["away_played"][i] = games_seen[a]

        if has_xg:
            feats["home_xg_for"][i] = rolling_mean(history[h], "xgf")
            feats["home_xg_against"][i] = rolling_mean(history[h], "xga")
            feats["away_xg_for"][i] = rolling_mean(history[a], "xgf")
            feats["away_xg_against"][i] = rolling_mean(history[a], "xga")

        # ----- reveal the outcome and update running state -----
        hs = row.home_score
        ascore = row.away_score
        # Skip state update for not-yet-played fixtures (NaN scores), but their
        # pre-kickoff features above are still valid predictors.
        if pd.isna(hs) or pd.isna(ascore):
            continue
        hs = int(hs)
        ascore = int(ascore)

        h_pts, a_pts = points_for(hs, ascore)
        h_rec = {"pts": h_pts, "gf": hs, "ga": ascore}
        a_rec = {"pts": a_pts, "gf": ascore, "ga": hs}
        if has_xg:
            hxg = home_xg[i]
            axg = away_xg[i]
            if not np.isnan(hxg) and not np.isnan(axg):
                h_rec["xgf"], h_rec["xga"] = float(hxg), float(axg)
                a_rec["xgf"], a_rec["xga"] = float(axg), float(hxg)
        history[h].append(h_rec)
        history[a].append(a_rec)
        last_played[h] = date
        last_played[a] = date
        games_seen[h] += 1
        games_seen[a] += 1
        elo.update(h, a, hs, ascore, neutral=neutral)

    for c in active_cols:
        df[c] = feats[c]
    # neutral_site is itself a base feature (canonical bool).
    df["neutral_site"] = neutral_flags

    # ----- optional groups attached from source columns (no leakage) -----
    if _has_market(df):
        df = add_market_features(df)
    df = add_squad_features(df)
    df = add_international_features(df)

    return df


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    """Base features plus whichever optional groups apply to ``df``.

    Mirrors :func:`build_features`: a group's columns are listed only when its
    source columns are present (and, for xG/market, non-null), so a model can ask
    for exactly the columns that will actually exist after a build.
    """
    cols = list(FEATURE_COLUMNS)
    if _has_xg(df):
        cols += XG_FEATURE_COLUMNS
    if _has_market(df):
        cols += MARKET_FEATURE_COLUMNS
    cols += available_squad_columns(df)
    cols += available_international_columns(df)
    return cols
