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


def test_remaining_ko_bracket_reconstructs_and_zeros_eliminated():
    """From the actual current-round matchups, remaining_ko_bracket rebuilds the
    remaining bracket over ONLY the still-alive teams, so monte_carlo_knockout gives
    probabilities to exactly those teams (everyone eliminated is absent -> 0)."""
    from soccer_predictor.data import world_cup
    from soccer_predictor.simulation import rules
    from soccer_predictor.simulation.tournament import simulate_tournament_once
    from soccer_predictor.simulation.monte_carlo import monte_carlo_knockout
    from soccer_predictor.models.elo_goals import EloGoalsModel
    groups = {chr(65 + i): [f"T{i}_{j}" for j in range(4)] for i in range(12)}
    fmt = rules.world_cup_2026_format()
    res = simulate_tournament_once(EloGoalsModel(), fmt, groups, np.random.default_rng(1))
    qf = [(m.home_team, m.away_team) for m in res["matches"] if m.stage == "quarterfinal"]
    assert len(qf) == 4
    built = world_cup.remaining_ko_bracket(res["slot_map"], res["third_assignment"], qf)
    assert built is not None
    fmt2, slot_map2 = built
    assert fmt2.stage_order == ("quarterfinal", "semifinal", "final")
    ko = monte_carlo_knockout(EloGoalsModel(), fmt2, slot_map2, n_simulations=200)
    assert set(ko["team"]) == set(slot_map2.values()) and len(set(slot_map2.values())) == 8
    assert abs(ko["champion_probability"].sum() - 1.0) < 1e-9
    # wrong number of fixtures -> not a clean round -> None
    assert world_cup.remaining_ko_bracket(res["slot_map"], res["third_assignment"], qf[:3]) is None


def test_known_results_lock_knockout_and_zero_eliminated():
    """Locking an actual knockout result makes the winner champion in 100% of runs
    and the loser 0% -- i.e. an eliminated team drops to zero."""
    from soccer_predictor.simulation import rules
    from soccer_predictor.simulation.tournament import simulate_tournament
    from soccer_predictor.models.elo_goals import EloGoalsModel
    fmt = rules.TournamentFormat(
        name="t", n_groups=2, teams_per_group=1, advance_per_group=1,
        bracket=[rules.BracketMatch("F", "1A", "1B", "final")], stage_order=("final",),
    )
    groups = {"A": ["Brazil"], "B": ["Argentina"]}
    known = {frozenset({"Brazil", "Argentina"}): ("Argentina", 2, 0)}
    sims = simulate_tournament(EloGoalsModel(), groups, n_simulations=50, fmt=fmt,
                               known_results=known)
    champ = dict(zip(sims["team"], sims["champion_probability"]))
    assert champ["Argentina"] == 1.0
    assert champ["Brazil"] == 0.0


def test_completed_ko_results_skips_group_and_draws():
    """Only decisive non-group WC games become locked knockout results."""
    from soccer_predictor.data import world_cup
    groups = {"A": ["France", "Sweden"], "B": ["Germany", "Spain"]}  # group pairs within
    scores = pd.DataFrame([
        {"home_team": "France", "away_team": "Germany", "home_score": 2, "away_score": 1,
         "date": "2026-07-01"},   # KO (cross-group) -> France wins
        {"home_team": "France", "away_team": "Sweden", "home_score": 1, "away_score": 0,
         "date": "2026-06-20"},   # group pair -> skipped
        {"home_team": "Spain", "away_team": "Brazil", "home_score": 1, "away_score": 1,
         "date": "2026-07-01"},   # draw (shootout) -> skipped
    ])
    empty = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score",
                                  "away_score", "competition"])
    res = world_cup.completed_ko_results(empty, scores, groups)
    assert res == {frozenset({"France", "Germany"}): ("France", 2, 1)}


def test_knockout_meeting_probabilities():
    """Pairwise KO-meeting probabilities are computed from the simulated bracket:
    the two group winners always contest the single final, so the final-stage
    meeting probabilities sum to 1 and every meet prob is a valid fraction."""
    from soccer_predictor.simulation import rules
    from soccer_predictor.simulation.monte_carlo import knockout_meeting_probabilities
    from soccer_predictor.models.elo_goals import EloGoalsModel
    fmt = rules.TournamentFormat(
        name="t", n_groups=2, teams_per_group=2, advance_per_group=1,
        bracket=[rules.BracketMatch("F", "1A", "1B", "final")],
        stage_order=("final",),
    )
    groups = {"A": ["Aa", "Ab"], "B": ["Ba", "Bb"]}
    df = knockout_meeting_probabilities(EloGoalsModel(), fmt, groups, n_simulations=200, seed=1)
    assert not df.empty
    assert {"team_a", "team_b", "meet_probability", "likeliest_round", "final_prob"} <= set(df.columns)
    assert df["meet_probability"].between(0.0, 1.0).all()
    assert abs(df["final_prob"].sum() - 1.0) < 1e-9   # exactly one final per simulation
    assert (df["likeliest_round"] == "final").all()


def test_third_place_assignment_honours_fixed_slots():
    """Pinned best-third slots (read off the real bracket) are kept exactly, and the
    rest of the assignment stays eligible + a bijection."""
    from soccer_predictor.simulation import rules
    qualified = ["A", "B", "C", "D", "E", "F", "G", "H"]
    elig = rules.wc2026_third_slot_rules()
    asn = rules.wc2026_third_place_assignment(qualified, fixed={"M77": "F"})
    assert asn["M77"] == "F"                                   # pin honoured
    assert all(g in elig[m] for m, g in asn.items())           # every slot eligible
    assert sorted(asn.values()) == sorted(qualified)           # each group used once
    # an ineligible pin is ignored, not blindly trusted
    asn2 = rules.wc2026_third_place_assignment(qualified, fixed={"M74": "G"})  # M74=A/B/C/D/F
    assert asn2["M74"] != "G"


def test_live_third_assignment_reads_actual_fixtures():
    """A real R32 fixture between a group winner (1I) and a third (3F) pins that
    match's best-third group to F; undecided/group fixtures are ignored."""
    from soccer_predictor.data import world_cup
    slot_map = {"1I": "France", "3F": "Sweden", "1A": "Mexico", "2B": "Croatia"}
    complete = {"I", "F", "A", "B"}
    pins = world_cup.live_third_assignment(
        slot_map, complete, [("France", "Sweden"), ("Mexico", "Croatia")])
    assert pins == {"M77": "F"}                                # M77 = 1I vs 3[C/D/F/G/H]
    # a team from an incomplete group is not trusted
    assert world_cup.live_third_assignment(slot_map, {"I"}, [("France", "Sweden")]) == {}


def test_wc_elo_factory_uncached_callers_cached():
    """Regression: _wc_elo_factory returns a closure, so it must NOT be wrapped by
    @st.cache_data (the closure can't be pickled when Streamlit writes the cache).
    Its callers, which return plain dicts, must stay cached."""
    from soccer_predictor.apps import streamlit_app as app
    assert not hasattr(app._wc_elo_factory, "clear")        # the factory is plain
    assert hasattr(app._wc_tournament_backtest, "clear")    # callers are cached
    assert hasattr(app._wc_past_simulation, "clear")
    model = app._wc_elo_factory(host_adv=50.0)(2018)        # builds a host-bumped Elo
    assert model.host_advantage == 50.0 and "Russia" in model.hosts


def test_host_advantage_bumps_only_the_host():
    """host_advantage adds Elo points to host teams' effective rating only, and
    flows into a higher predicted goal rate for the host."""
    from soccer_predictor.models.elo_goals import EloGoalsModel
    host = EloGoalsModel(hosts=["Brazil"], host_advantage=50.0)
    plain = EloGoalsModel(host_advantage=0.0)
    # unseen teams sit at base_rating; the host gets exactly +50, others unchanged
    assert host.effective_rating("Brazil") - plain.effective_rating("Brazil") == 50.0
    assert host.effective_rating("France") == plain.effective_rating("France")
    # and it raises the host's expected goal rate vs a neutral equal opponent
    lam_host, _ = host._rates("Brazil", "France", neutral=True)
    lam_plain, _ = plain._rates("Brazil", "France", neutral=True)
    assert lam_host > lam_plain


def test_fetch_outrights_devig(monkeypatch):
    """Outright odds are aggregated across books and de-vigged to sum to 1."""
    from soccer_predictor.data import odds_api
    fake = [{"bookmakers": [{"markets": [{"key": "outrights", "outcomes": [
        {"name": "France", "price": 2.0},   # raw implied 0.50
        {"name": "Brazil", "price": 4.0},   # raw implied 0.25
    ]}]}]}]
    monkeypatch.setattr(odds_api, "_api_get", lambda *a, **k: fake)
    m = odds_api.fetch_outrights(cache=False)
    assert abs(sum(m.values()) - 1.0) < 1e-9          # de-vigged
    assert abs(m["France"] - 2.0 / 3.0) < 1e-6        # 0.50 / 0.75
    assert abs(m["Brazil"] - 1.0 / 3.0) < 1e-6        # 0.25 / 0.75


def test_blend_champion_to_market_moves_toward_market_and_normalises():
    from soccer_predictor.apps.streamlit_app import _blend_champion_to_market
    sim = pd.DataFrame({
        "team": ["Brazil", "France", "Spain"],
        "champion_probability": [0.50, 0.20, 0.30],
    })
    market = {"Brazil": 0.10, "France": 0.40}  # Spain unpriced -> keeps its sim value
    out = _blend_champion_to_market(sim, market, weight=0.5)
    assert abs(out["champion_probability"].sum() - 1.0) < 1e-9
    probs = dict(zip(out["team"], out["champion_probability"]))
    assert probs["Brazil"] < 0.50      # pulled down toward 0.10
    assert probs["France"] > 0.20      # pulled up toward 0.40
    # market empty -> unchanged
    same = _blend_champion_to_market(sim, {}, weight=0.5)
    assert list(same["champion_probability"]) == [0.50, 0.20, 0.30]


def test_penalty_shootout_shrinks_toward_coin_flip():
    """A penalty shootout is near-random: the favourite's open-play edge is shrunk
    toward 0.5 by shootout_skill (0 = pure coin flip, 1 = full edge)."""
    from soccer_predictor.simulation.match import simulate_match
    from soccer_predictor.simulation import rules

    class _Stub:  # always 0-0 in regulation + ET, strong open-play favourite
        def sample_score(self, h, a, rng, ctx):
            return (0, 0)
        def expected_goals(self, h, a, ctx):
            return (0.0, 0.0)
        def win_probability_no_draw(self, h, a, ctx):
            return 0.8

    rng = np.random.default_rng(0)
    def fav_rate(skill, n=4000):
        tie = rules.KnockoutTie(extra_time=False, penalties=True, shootout_skill=skill)
        wins = sum(simulate_match(_Stub(), "Fav", "Dog", rng, knockout=True,
                                  tie=tie).winner == "Fav" for _ in range(n))
        return wins / n
    assert abs(fav_rate(0.0) - 0.5) < 0.04        # pure coin flip
    assert abs(fav_rate(1.0) - 0.8) < 0.04        # full open-play edge
    assert 0.53 < fav_rate(0.25) < 0.62           # default ~0.575: slight tilt only


def test_wc_start_weights_counts_starts_and_subs(monkeypatch):
    """Actual-tournament-XI weighting counts starts (1.0) + sub appearances (0.3)
    per player across the edition, keyed by (family, given-initial)."""
    from soccer_predictor.data import squad_value as sv
    df = pd.DataFrame([
        {"tournament_id": "WC-2018", "team_name": "Testland", "given_name": "Al",
         "family_name": "Pha", "starter": 1, "substitute": 0},
        {"tournament_id": "WC-2018", "team_name": "Testland", "given_name": "Al",
         "family_name": "Pha", "starter": 1, "substitute": 0},   # 2 starts
        {"tournament_id": "WC-2018", "team_name": "Testland", "given_name": "Be",
         "family_name": "Ta", "starter": 0, "substitute": 1},     # 1 sub
        {"tournament_id": "WC-2022", "team_name": "Testland", "given_name": "X",
         "family_name": "Y", "starter": 1, "substitute": 0},      # other edition -> excluded
    ])
    monkeypatch.setattr(sv, "fetch_wc_appearances", lambda: df)
    w = sv.wc_start_weights(2018, sub_weight=0.3)
    assert w["Testland"][("pha", "a")] == 2.0
    assert abs(w["Testland"][("ta", "b")] - 0.3) < 1e-9
    assert ("y", "x") not in w.get("Testland", {})  # 2022 row excluded


def test_xi_weighted_mean_emphasises_starting_xi():
    """Squad strength weights the projected XI (top 11 by rating) far above the
    bench, instead of an equal-weighted top-K where the 23rd man counts fully."""
    from soccer_predictor.data.squad_value import _xi_weighted_mean
    vals = [90.0] * 11 + [50.0] * 12          # strong XI, weak bench
    w = _xi_weighted_mean(vals, squad=23, xi=11)
    assert w > sum(vals) / len(vals)          # XI dominates -> pulled toward 90
    assert abs(w - (11 * 90 + 12 * 0.25 * 50) / (11 + 12 * 0.25)) < 1e-9  # ~81.4
    # squad within the XI band -> plain mean (every player a "starter")
    assert abs(_xi_weighted_mean([80.0, 70.0], squad=23, xi=11) - 75.0) < 1e-9


def test_age_retention_trims_young_and_boosts_old_symmetrically():
    """De-aging must cut both ways: young value is inflated by potential (ratio >1
    -> trimmed), veteran value is age-discounted (ratio <1 -> boosted), ~0.06/yr."""
    from soccer_predictor.data.players import _age_value_retention
    assert _age_value_retention(26) == 1.0
    assert _age_value_retention(18) > 1.3            # teenager: big potential premium trimmed
    assert _age_value_retention(34) < 0.7            # veteran: boosted
    assert abs(_age_value_retention(20) - (1.0 + 0.06 * 4)) < 1e-9   # 1.24
    assert abs(_age_value_retention(32) - (1.0 - 0.06 * 4)) < 1e-9   # 0.76 (symmetric)


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
