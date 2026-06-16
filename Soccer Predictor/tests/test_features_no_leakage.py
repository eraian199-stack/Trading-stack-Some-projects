"""
The CRITICAL leakage test.

A match's features may only depend on EARLIER matches. The decisive check:
build features on a prefix of the data, then build them again on the prefix PLUS
future matches; every feature value for the earlier rows must be byte-for-byte
identical. If a later match leaked into an earlier row's features, the two builds
would disagree.

We also pin some sanity invariants the leakage-safe walk implies: home_played /
away_played are pre-match counts (the very first appearance of a team is 0), and
those counts grow monotonically as a team plays more games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soccer_predictor.data import synthetic
from soccer_predictor.features.feature_store import (
    FEATURE_COLUMNS,
    available_feature_columns,
    build_features,
)


def _feature_cols_in(df: pd.DataFrame) -> list[str]:
    """Numeric base/optional feature columns that actually exist on `df`."""
    return [c for c in available_feature_columns(df) if c in df.columns and c != "neutral_site"]


# --------------------------------------------------------------------------- #
# The leakage check
# --------------------------------------------------------------------------- #
def test_appending_future_rows_does_not_change_earlier_features():
    df = synthetic.generate_league(n_teams=8, n_seasons=2, seed=11)
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    cut = len(df) // 2
    prefix = df.iloc[:cut].copy()

    built_prefix = build_features(prefix, form_window=5)
    built_full = build_features(df, form_window=5)

    # build_features sorts by date; align both on (date, home, away) to compare.
    key = ["date", "home_team", "away_team"]
    cols = _feature_cols_in(built_prefix)

    a = built_prefix.set_index(key)[cols].sort_index()
    b = built_full.set_index(key)[cols].sort_index()
    # Compare only the rows present in the prefix build.
    b = b.loc[a.index]

    np.testing.assert_allclose(
        a.to_numpy(dtype=float),
        b.to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
        err_msg="future matches leaked into earlier rows' features",
    )


def test_leakage_check_holds_on_international_data():
    """Same invariant on neutral-heavy international data (neutral-aware Elo)."""
    df = synthetic.generate_international_history(n_teams=20, n_matches=300, seed=12)
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    cut = int(len(df) * 0.6)
    prefix = df.iloc[:cut].copy()

    built_prefix = build_features(prefix, form_window=6)
    built_full = build_features(df, form_window=6)

    key = ["date", "home_team", "away_team"]
    cols = _feature_cols_in(built_prefix)
    a = built_prefix.set_index(key)[cols].sort_index()
    b = built_full.set_index(key)[cols].sort_index().loc[a.index]

    np.testing.assert_allclose(
        a.to_numpy(dtype=float),
        b.to_numpy(dtype=float),
        equal_nan=True,
        rtol=0,
        atol=0,
    )


# --------------------------------------------------------------------------- #
# Base-feature presence + dtypes
# --------------------------------------------------------------------------- #
def test_base_feature_columns_present():
    df = synthetic.generate_league(n_teams=6, n_seasons=1, seed=13)
    built = build_features(df, form_window=4)
    for col in FEATURE_COLUMNS:
        assert col in built.columns, f"missing base feature {col!r}"
    # The synthetic league carries xG + odds, so optional groups attach too.
    cols = available_feature_columns(built)
    assert "home_xg_for" in cols  # xG rolling group present
    assert "mkt_imp_home" in cols  # market group present


# --------------------------------------------------------------------------- #
# played-count sanity (pre-match, monotonic)
# --------------------------------------------------------------------------- #
def test_played_counts_start_at_zero_and_are_pre_match():
    df = synthetic.generate_league(n_teams=6, n_seasons=1, seed=14)
    built = build_features(df, form_window=6)

    # The earliest match each team appears in must show a pre-match count of 0.
    seen: set[str] = set()
    for row in built.itertuples(index=False):
        if row.home_team not in seen:
            assert row.home_played == 0, (
                f"{row.home_team} first appears with home_played={row.home_played}"
            )
            seen.add(row.home_team)
        if row.away_team not in seen:
            assert row.away_played == 0
            seen.add(row.away_team)


def test_played_counts_monotonic_per_team():
    df = synthetic.generate_league(n_teams=6, n_seasons=1, seed=15)
    built = build_features(df, form_window=6).sort_values("date", kind="mergesort")

    last_seen: dict[str, float] = {}
    for row in built.itertuples(index=False):
        for team, played in (
            (row.home_team, row.home_played),
            (row.away_team, row.away_played),
        ):
            if team in last_seen:
                # A team's recorded pre-match games-played never decreases and
                # strictly increases between two of its own appearances.
                assert played > last_seen[team], (
                    f"{team} played count not increasing: "
                    f"{last_seen[team]} -> {played}"
                )
            last_seen[team] = played


def test_elo_diff_consistent_with_elo_columns():
    df = synthetic.generate_league(n_teams=6, n_seasons=1, seed=16)
    built = build_features(df, form_window=6)
    np.testing.assert_allclose(
        built["elo_diff"].to_numpy(dtype=float),
        (built["elo_home"] - built["elo_away"]).to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
    )
