"""
Tests for the national-team variables, market anchor, and WC backtest harness.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import soccer_predictor as sp
from soccer_predictor.data import players
from soccer_predictor.data.normalizers import standardize_matches
from soccer_predictor.evaluation import wc_backtest
from soccer_predictor.models.elo_goals import EloGoalsModel
from soccer_predictor.models.market_anchor import MarketAnchoredModel


@pytest.fixture(scope="module")
def league_df() -> pd.DataFrame:
    return sp.generate_league(n_teams=10, n_seasons=3, seed=4)


# --------------------------------------------------------------------------- #
# EloGoalsModel optional components
# --------------------------------------------------------------------------- #
def test_components_default_off_match_plain(league_df):
    """The default model must equal a fully-off-components model."""
    a = EloGoalsModel().fit(league_df)
    b = EloGoalsModel(
        use_importance=False, form_weight=0.0, pedigree_weight=0.0, squad_weight=0.0
    ).fit(league_df)
    ctx = {"neutral_site": True}
    np.testing.assert_allclose(
        a.outcome_probs("Team_00", "Team_01", ctx),
        b.outcome_probs("Team_00", "Team_01", ctx),
    )


def test_squad_overlay_shifts_effective_rating(league_df):
    # A big positive squad score for Team_00 should raise its effective rating.
    plain = EloGoalsModel().fit(league_df)
    boosted = EloGoalsModel(
        squad_strength={"Team_00": 500.0, "Team_01": -500.0}, squad_weight=1.0
    ).fit(league_df)
    assert boosted.effective_rating("Team_00") > plain.effective_rating("Team_00")
    # and its win probability vs Team_01 should rise
    ctx = {"neutral_site": True}
    assert (
        boosted.outcome_probs("Team_00", "Team_01", ctx)[0]
        > plain.outcome_probs("Team_00", "Team_01", ctx)[0]
    )


def test_form_component_changes_predictions(league_df):
    plain = EloGoalsModel().fit(league_df)
    formed = EloGoalsModel(form_weight=300.0).fit(league_df)
    ctx = {"neutral_site": True}
    # form is populated and (for at least some pair) moves the probabilities
    assert len(formed.form) > 0
    diffs = [
        abs(plain.outcome_probs(h, a, ctx)[0] - formed.outcome_probs(h, a, ctx)[0])
        for h, a in [("Team_00", "Team_01"), ("Team_02", "Team_03")]
    ]
    assert max(diffs) > 1e-4


# --------------------------------------------------------------------------- #
# Market anchor
# --------------------------------------------------------------------------- #
def test_market_anchor_blends_and_falls_back(league_df):
    base = EloGoalsModel().fit(league_df)
    odds = {("Team_00", "Team_01"): (1.5, 4.0, 6.0)}
    anch = MarketAnchoredModel(base, weight=0.6, odds=odds)
    ctx = {"neutral_site": True}
    p_base = base.outcome_probs("Team_00", "Team_01", ctx)
    p_anch = anch.outcome_probs("Team_00", "Team_01", ctx)
    from soccer_predictor.features.market import implied_probabilities

    mkt = implied_probabilities(1.5, 4.0, 6.0)
    # blended home prob sits strictly between base and market
    lo, hi = sorted([p_base[0], mkt[0]])
    assert lo - 1e-9 <= p_anch[0] <= hi + 1e-9
    assert abs(p_anch.sum() - 1.0) < 1e-9
    # an unpriced fixture is identical to the base model
    np.testing.assert_allclose(
        anch.outcome_probs("Team_02", "Team_03", ctx),
        base.outcome_probs("Team_02", "Team_03", ctx),
    )


def test_market_anchor_handles_reversed_listing(league_df):
    base = EloGoalsModel().fit(league_df)
    # odds stored as (Team_01, Team_00); query (Team_00, Team_01) must swap.
    anch = MarketAnchoredModel(base, weight=1.0, odds={("Team_01", "Team_00"): (2.0, 3.5, 3.6)})
    ctx = {"neutral_site": True}
    p = anch.outcome_probs("Team_00", "Team_01", ctx)
    from soccer_predictor.features.market import implied_probabilities

    mkt_listed = implied_probabilities(2.0, 3.5, 3.6)  # for Team_01 home
    # our Team_00 is their away -> our home prob ~ their away prob
    assert abs(p[0] - mkt_listed[2]) < 0.05


# --------------------------------------------------------------------------- #
# Legacy WC format
# --------------------------------------------------------------------------- #
def test_world_cup_legacy_format_shape():
    fmt = sp.world_cup_legacy_format()
    assert fmt.n_groups == 8 and fmt.teams_per_group == 4 and fmt.n_best_third == 0
    assert len(fmt.bracket) == 8 + 4 + 2 + 1  # R16 + QF + SF + final
    assert fmt.stage_order[-1] == "final"


# --------------------------------------------------------------------------- #
# WC backtest harness (synthetic edition)
# --------------------------------------------------------------------------- #
def _history_with_wc(year: int = 2018) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = [f"N{i:02d}" for i in range(16)]
    strength = {t: rng.normal() for t in teams}
    rows = []
    # prior history (friendlies + qualifiers) before the edition
    day = pd.Timestamp(f"{year - 4}-01-01")
    for k in range(900):
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(0.2 + strength[h] - strength[a])
        lam_a = np.exp(0.1 + strength[a] - strength[h])
        rows.append({
            "date": day + pd.Timedelta(days=k),
            "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(lam_h)), "away_score": int(rng.poisson(lam_a)),
            "competition": "Friendly" if k % 2 else "FIFA World Cup qualification",
            "neutral": False,
        })
    # the World Cup edition itself (a round-robin block, neutral)
    start = pd.Timestamp(f"{year}-06-14")
    for i, (h, a) in enumerate(combinations(teams[:8], 2)):
        lam_h = np.exp(0.0 + strength[h] - strength[a])
        lam_a = np.exp(0.0 + strength[a] - strength[h])
        rows.append({
            "date": start + pd.Timedelta(days=i),
            "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(lam_h)), "away_score": int(rng.poisson(lam_a)),
            "competition": "FIFA World Cup", "neutral": True,
        })
    return standardize_matches(pd.DataFrame(rows))


def test_identify_and_backtest_world_cups():
    h = _history_with_wc(2018)
    eds = wc_backtest.identify_world_cups(h)
    assert 2018 in eds and len(eds[2018]) > 0
    res = wc_backtest.backtest_world_cups(
        h, EloGoalsModel, years=[2018], min_year=2000, update_within=True
    )
    assert res["pooled"]["n"] > 0
    assert np.allclose(res["probs"].sum(axis=1), 1.0, atol=1e-6)
    assert 0.0 < res["pooled"]["log_loss"] < 3.0


def test_squad_value_asof_cutoff_and_coverage_gate(monkeypatch):
    """Squad value uses the last valuation ON/BEFORE kickoff (no leakage) and
    omits under-covered nations so they fall back to pure Elo."""
    from soccer_predictor.data import squad_value as sv

    before = int(pd.Timestamp("2017-01-01").timestamp() * 1000)
    after = int(pd.Timestamp("2019-01-01").timestamp() * 1000)  # after the 2018 WC

    def fake_players(year):
        rows = []
        for i in range(20):  # covered nation: 20 players, values 1..20 M as-of 2017
            rows.append({"citizenship": "Testland", "name": f"P{i}",
                         "market_value_history": [
                             {"x": before, "y": (i + 1) * 1_000_000},
                             {"x": after, "y": 999_000_000}]})  # future -> must be ignored
        for i in range(3):  # under-covered nation: 3 players -> excluded
            rows.append({"citizenship": "Smalland", "name": f"S{i}",
                         "market_value_history": [{"x": before, "y": 5_000_000}]})
        return rows

    monkeypatch.setattr(sv, "fetch_players", fake_players)
    df = sv.build_squad_value(2018, top_k=10, min_covered=15)
    teams = set(df["team"])
    assert "Testland" in teams and "Smalland" not in teams  # coverage gate
    row = df[df["team"] == "Testland"].iloc[0]
    # As-of 2018 each value = the 2017 point (the 2019 point is after kickoff);
    # top-10 of 1..20 M is 11..20 M, mean 15.5 M.
    assert abs(row["squad_value_eur"] - 15_500_000) < 1.0


def test_load_ability_csv_per_team_and_per_player(tmp_path):
    """The overlay accepts any rating site's export: a per-team CSV as-is, and a
    per-player CSV aggregated to the team's top-K mean (loose column detection)."""
    from soccer_predictor.data import squad_value as sv

    per_team = tmp_path / "team.csv"
    per_team.write_text("team,rating\nFrance,7.4\nBrazil,7.1\n")
    out = sv.load_ability_csv(str(per_team))
    assert abs(out["France"] - 7.4) < 1e-9 and abs(out["Brazil"] - 7.1) < 1e-9

    per_player = tmp_path / "players.csv"  # SofaScore-style: many rows per nation
    per_player.write_text(
        "player,team,rating\nA,France,8.0\nB,France,7.0\nC,France,6.0\nD,Brazil,9.0\n"
    )
    out2 = sv.load_ability_csv(str(per_player), top_k=2)  # France top-2 mean = 7.5
    assert abs(out2["France"] - 7.5) < 1e-9 and abs(out2["Brazil"] - 9.0) < 1e-9


def test_combine_ability_blends_only_where_present():
    """Blending z-scores each source, averages where they overlap, keeps singletons,
    and drops teams no source has (so they fall back to pure Elo)."""
    from soccer_predictor.data import squad_value as sv
    fifa = {"A": 90.0, "B": 80.0, "C": 70.0}          # 0-100 scale
    sofa = {"A": 7.5, "B": 6.5, "D": 9.0}             # 0-10 scale (different scale)
    out = sv.combine_ability([fifa, sofa])
    assert set(out) == {"A", "B", "C", "D"}           # union; nothing else
    # A is top of both -> highest; C only in fifa (its low z), D only in sofa (its top z)
    assert out["A"] > out["B"]
    assert out["D"] > out["C"]                          # D is sofa's best, C is fifa's worst
    # scale-invariant: scaling a source by 10x doesn't change the blend
    out2 = sv.combine_ability([fifa, {k: v * 10 for k, v in sofa.items()}])
    assert abs(out2["A"] - out["A"]) < 1e-9


def test_build_tm_ability_deages_as_of(monkeypatch):
    """Per-edition TM ability uses the as-of-kickoff valuation (no leakage) and
    de-ages it (a 33-yo great rated on ability, not his discounted price)."""
    from soccer_predictor.data import squad_value as sv
    from soccer_predictor.data.players import _age_value_retention

    before = int(pd.Timestamp("2017-01-01").timestamp() * 1000)
    after = int(pd.Timestamp("2019-01-01").timestamp() * 1000)  # after the 2018 WC
    monkeypatch.setattr(sv, "fetch_players", lambda year: [
        {"citizenship": "Testland", "name": f"P{i}", "market_value_history": [
            {"x": before, "y": 10_000_000, "age": "33"},
            {"x": after, "y": 99_000_000, "age": "35"}]}  # post-WC -> ignored
        for i in range(15)
    ])
    a = sv.build_tm_ability(2018, top_k=10, min_covered=15)
    row = a[a["team"] == "Testland"].iloc[0]
    assert abs(row["ability"] - 10_000_000 / _age_value_retention(33.0)) < 1.0


def test_fifa_ability_snapshot_pre_wc_and_aggregate(monkeypatch):
    """FIFA ability uses the rating snapshot on/before kickoff (no leakage) and
    aggregates each nation's top-K overall."""
    from soccer_predictor.data import squad_value as sv

    rows = [
        {"fifa_version": 18, "fifa_update_date": "2017-09-01", "short_name": "A",
         "overall": 90, "age": 27, "nationality_name": "Testland"},
        {"fifa_version": 18, "fifa_update_date": "2017-09-01", "short_name": "B",
         "overall": 80, "age": 27, "nationality_name": "Testland"},
        {"fifa_version": 18, "fifa_update_date": "2019-01-01", "short_name": "A",
         "overall": 99, "age": 29, "nationality_name": "Testland"},  # post-WC -> ignore
    ] + [{"fifa_version": 18, "fifa_update_date": "2017-09-01", "short_name": f"P{i}",
          "overall": 50, "age": 25, "nationality_name": "Testland"} for i in range(15)]
    monkeypatch.setattr(sv, "fetch_fifa_players", lambda: pd.DataFrame(rows))
    a = sv.build_fifa_ability(2018, top_k=2, min_covered=2)
    row = a[a["team"] == "Testland"].iloc[0]
    assert abs(row["ability"] - 85.0) < 1e-9  # top-2 of the 2017-09 snapshot = (90+80)/2


def test_build_current_ability_deages_veterans(monkeypatch):
    """The live ability overlay de-ages value so a 34-yo squad is rated on ability
    (value / age-retention), not its age-discounted price."""
    from soccer_predictor.data import squad_value as sv
    from soccer_predictor.data.players import _age_value_retention

    monkeypatch.setattr(sv, "_scraper_dir_listing", lambda: {"2022/players.json.gz": "x"})
    monkeypatch.setattr(sv, "fetch_players", lambda year: [
        {"citizenship": "Testland", "name": f"P{i}",
         "market_value_history": [{"x": 1, "y": (i + 1) * 1_000_000, "age": "34"}]}
        for i in range(16)
    ])
    df = sv.build_current_ability(top_k=10, min_covered=15)
    row = df[df["team"] == "Testland"].iloc[0]
    expected = (sum(range(7, 17)) * 1_000_000 / 10) / _age_value_retention(34.0)  # top-10 mean, de-aged
    assert abs(row["ability"] - expected) < 1.0
    assert int(row["source_year"]) == 2022


def test_actual_stage_reach_from_bracket():
    """Stage participation is derived from who PLAYED each round (penalty-proof),
    the third-place playoff is skipped, and the final is the last match."""
    fmt = sp.world_cup_legacy_format()
    rows = []
    # 48 group games (any teams) dated before the knockouts.
    base = pd.Timestamp("2018-06-01")
    for i in range(fmt.n_groups * 6):
        rows.append({"date": base + pd.Timedelta(days=i // 8),
                     "home_team": f"t{i % 32}", "away_team": f"t{(i + 1) % 32}"})
    def ko(day, pairs):
        for h, a in pairs:
            rows.append({"date": pd.Timestamp(day), "home_team": h, "away_team": a})
    ko("2018-06-20", [(f"t{2*i}", f"t{2*i+1}") for i in range(8)])      # R16: t0..t15
    ko("2018-06-24", [("t0","t2"), ("t4","t6"), ("t8","t10"), ("t12","t14")])  # QF
    ko("2018-06-28", [("t0","t4"), ("t8","t12")])                      # SF
    ko("2018-06-30", [("t4","t12")])                                   # 3rd place (skipped)
    ko("2018-07-01", [("t0","t8")])                                    # final
    reach = sp.actual_stage_reach(pd.DataFrame(rows), fmt, champion="t0")
    assert reach["round_of_16"] == {f"t{i}" for i in range(16)}
    assert reach["quarterfinal"] == {"t0", "t2", "t4", "t6", "t8", "t10", "t12", "t14"}
    assert reach["semifinal"] == {"t0", "t4", "t8", "t12"}
    assert reach["final"] == {"t0", "t8"}          # the LAST match, not the 3rd-place one
    assert reach["champion"] == {"t0"}


def test_compare_models_on_world_cups_leaderboard():
    h = _history_with_wc(2018)
    lb = wc_backtest.compare_models_on_world_cups(
        h,
        {"plain": EloGoalsModel, "form": lambda: EloGoalsModel(form_weight=200)},
        years=[2018],
        min_year=2000,
    )
    assert set(lb.index) == {"plain", "form"}
    assert "log_loss" in lb.columns


# --------------------------------------------------------------------------- #
# Player / squad strength ingestion
# --------------------------------------------------------------------------- #
def test_squad_csv_and_elo_adjustment(tmp_path):
    csv = tmp_path / "squad.csv"
    pd.DataFrame(
        {"team": ["Brazil", "Qatar", "Bosnia & Herzegovina"], "strength": [90.0, 50.0, 70.0]}
    ).to_csv(csv, index=False)
    strength = players.load_squad_strength_csv(csv)
    assert strength["Brazil"] == 90.0
    # '&' name resolves to the canonical alias
    assert "Bosnia-Herzegovina" in strength
    adj = players.to_elo_adjustment(strength, spread=60.0)
    assert abs(sum(adj.values())) < 1e-6  # zero-mean
    assert adj["Brazil"] > adj["Qatar"]  # stronger -> higher Elo points


# --------------------------------------------------------------------------- #
# Transfermarkt value parsing + log-scaled Elo adjustment
# --------------------------------------------------------------------------- #
def test_parse_transfermarkt_value():
    from soccer_predictor.data.players import _parse_tm_value

    assert abs(_parse_tm_value("€807.50m") - 807.5) < 1e-9
    assert abs(_parse_tm_value("€1.23bn") - 1230.0) < 1e-9
    assert _parse_tm_value("€-") is None
    assert _parse_tm_value("-") is None
    assert _parse_tm_value("") is None


def test_to_elo_adjustment_log_is_zero_mean_and_ordered():
    adj = players.to_elo_adjustment(
        {"A": 1500.0, "B": 200.0, "C": 50.0}, spread=60.0, log=True
    )
    assert abs(sum(adj.values())) < 1e-6  # zero-mean
    assert adj["A"] > adj["B"] > adj["C"]  # monotone in value
    # log dampens the giant outlier vs a linear scaling
    lin = players.to_elo_adjustment({"A": 1500.0, "B": 200.0, "C": 50.0}, log=False)
    assert adj["A"] < lin["A"]


# --------------------------------------------------------------------------- #
# Age-adjusted ability (value is age-discounted; de-age it toward quality)
# --------------------------------------------------------------------------- #
def test_age_value_retention_curve():
    from soccer_predictor.data.players import _age_value_retention as r

    assert abs(r(26) - 1.0) < 1e-9         # peak window: value ~ ability
    assert r(33) <= 0.70 and r(36) < 0.60  # veterans discounted by value
    assert r(40) <= 0.30                    # floored for the very old
    assert r(19) > 1.0                      # young carry a potential premium
    # de-aging recovers a veteran's ability above their market value...
    assert 15.0 / r(33) > 20.0
    # ...and trims a young player's potential-inflated value
    assert 80.0 / r(19) < 80.0


def test_age_adjustment_reranks_old_vs_young_squads():
    """An older squad worth the same as a younger one scores higher on ability."""
    from soccer_predictor.data.players import _age_value_retention as r

    old = sum(20.0 / r(a) for a in [33, 32, 34, 31])    # 4 vets, €20m each
    young = sum(20.0 / r(a) for a in [20, 21, 19, 22])  # 4 kids, €20m each
    assert old > young


# --------------------------------------------------------------------------- #
# League tiers (level of competition)
# --------------------------------------------------------------------------- #
def test_club_normalisation_and_tiering():
    from soccer_predictor.data.players import _norm_club, _club_tier

    assert _norm_club("FC Barcelona") == "barcelona"
    assert _norm_club("Atlético de Madrid") == "atletico madrid"
    assert _norm_club("Arsenal FC") == "arsenal"
    # top-5 club -> tier 1.0, matched
    assert _club_tier("Real Madrid") == (1.0, True)
    # strong non-top-5 leagues the user flagged sit clearly above the tail
    assert _club_tier("Ajax Amsterdam")[0] == 0.80      # Eredivisie
    assert _club_tier("Flamengo")[0] == 0.80            # Brazil
    assert _club_tier("Benfica")[0] == 0.80             # Primeira
    assert _club_tier("River Plate")[0] == 0.72         # Argentina
    # unknown club -> default tier, not matched
    assert _club_tier("Obscure Town FC") == (0.45, False)


def test_league_tiers_ordering():
    from soccer_predictor.data.players import LEAGUE_TIERS, DEFAULT_LEAGUE_TIER

    assert LEAGUE_TIERS["premier league"] == 1.0
    assert LEAGUE_TIERS["eredivisie"] > LEAGUE_TIERS["mls"] > DEFAULT_LEAGUE_TIER
    assert LEAGUE_TIERS["brazil serie a"] > DEFAULT_LEAGUE_TIER
