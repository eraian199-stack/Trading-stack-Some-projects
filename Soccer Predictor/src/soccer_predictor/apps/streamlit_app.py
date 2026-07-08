"""
Streamlit front-end for ``soccer_predictor``.

Four tabs, each a thin view over the already-written package internals (the app
NEVER reimplements modelling, evaluation or simulation logic -- it loads data,
fits a model, and renders what the package returns):

    Match Predictor      1X2 / O-U / BTTS / fair-odds bundle for one fixture.
    Tournament Simulator Monte-Carlo a groups+knockout tournament (default WC2026)
                         and chart per-team reach probabilities.
    Backtest Lab         walk-forward proper-scoring metrics + calibration table.
    Betting Edge Scanner model fair odds vs market odds vs edge, with a loud
                         "ROI is a hypothesis" / calibration caveat.

Data load and model fit are cached (``st.cache_data`` / ``st.cache_resource``).
When synthetic/demo data is in use it is flagged LOUDLY (a project
non-negotiable). A sidebar model selector chooses Elo / Dixon-Coles / xG /
Ensemble.

Run it via ``streamlit run app_unified.py`` at the project root (a thin launcher
that execs this module), or ``streamlit run src/soccer_predictor/apps/streamlit_app.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..data import loaders, odds_api, schemas, sources, synthetic, world_cup
from ..data.aliases import normalize_team_name
from ..evaluation import betting, metrics, reports, walk_forward, wc_backtest
from ..features.market import implied_probabilities
from ..models import DixonColes, EloGoalsModel, PoissonXG, default_ensemble
from ..models.base import ScorelineModel, predict_markets

# --------------------------------------------------------------------------- #
# Model registry (label -> zero-arg factory). All four are scoreline-capable so
# every tab can drive the simulator / market math.
# --------------------------------------------------------------------------- #
MODEL_FACTORIES = {
    "Elo": EloGoalsModel,
    "Dixon-Coles": DixonColes,
    "xG (Poisson)": PoissonXG,
    "Ensemble": default_ensemble,
}

# Models that can run on INTERNATIONAL / World Cup data. xG/Ensemble need club xG
# (internationals have none -> they error), so they are excluded. Dixon-Coles runs
# but is a CLUB model: on neutral-venue international data with 300+ sparse teams it
# is slow (~30s/fit) and miscalibrated (worse than the no-skill floor; degenerate
# tournament sims) -- offered only so you can SEE that. Elo is the engine here.
WC_MODELS = {"Elo": EloGoalsModel, "Dixon-Coles": DixonColes}

SAMPLE_BANNER = (
    "SAMPLE / SYNTHETIC DATA IN USE -- every number below is generated demo data "
    "and is **NOT real**. Load a real CSV in the sidebar before trusting any "
    "probability, edge, or simulation."
)


# --------------------------------------------------------------------------- #
# Cached data + model
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading match data...")
def load_data(source_kind: str, source_value: str, synthetic_kind: str) -> pd.DataFrame:
    """Load a canonical match frame for the chosen source.

    ``source_kind`` is "Synthetic" or "Local file/dir"; the result is cached on
    the (kind, value) key so re-renders do not re-read or re-download.
    """
    if source_kind == "International (live)":
        return sources.fetch_international_results()
    if source_kind == "Synthetic":
        if synthetic_kind == "International":
            return synthetic.generate_international_history()
        return synthetic.generate_league()
    # Local file or directory: auto-detect the loader.
    return _auto_load(source_value)


def _auto_load(path: str) -> pd.DataFrame:
    last_exc: Exception | None = None
    for fn in (
        loaders.load_matches,
        loaders.load_international_results,
        loaders.load_football_data,
    ):
        try:
            return fn(path)
        except (ValueError, schemas.SchemaError, KeyError, FileNotFoundError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not load {path!r}: {last_exc}")


@st.cache_resource(show_spinner="Fitting model...")
def fit_model(model_label: str, data_key: str, source_kind: str,
              source_value: str, synthetic_kind: str) -> ScorelineModel:
    """Fit the selected model on the (cached) training data.

    ``data_key`` exists only to bust the cache when the data source changes; the
    fitted model is cached on (model_label, data_key).
    """
    df = load_data(source_kind, source_value, synthetic_kind)
    train = schemas.completed_matches(df)
    model = MODEL_FACTORIES[model_label]()
    return model.fit(train.copy())


@st.cache_resource(show_spinner="Fitting Elo + squad-ability model...")
def _match_overlay_model(data_key: str, source_kind: str, source_value: str,
                         synthetic_kind: str, ability_w: float, sofa: bool):
    """Elo fit on the loaded data, tilted toward CURRENT squad ability (FIFA +
    age-adjusted Transfermarkt, optionally + league-adjusted SofaScore). Returns
    (model, source_label); falls back to plain Elo if no ability source is found."""
    df = load_data(source_kind, source_value, synthetic_kind)
    train = schemas.completed_matches(df)
    adj, label = _live_ability(sofa)
    if not adj:
        return MODEL_FACTORIES["Elo"]().fit(train.copy()), ""
    model = EloGoalsModel(squad_strength=adj, squad_weight=ability_w)
    return model.fit(train.copy()), label


def _data_key(source_kind: str, source_value: str, synthetic_kind: str) -> str:
    return f"{source_kind}|{source_value}|{synthetic_kind}"


# --------------------------------------------------------------------------- #
# Sidebar (data source + model selector)
# --------------------------------------------------------------------------- #
def _sidebar() -> dict:
    st.sidebar.header("Data")
    source_kind = st.sidebar.radio(
        "Source",
        ["International (live)", "Synthetic", "Local file/dir"],
        index=0,
        help="International (live) = real national-team results (martj42), fit on "
        "the full history — the World Cup setting. Synthetic = labelled demo data. "
        "Local = a canonical / football-data / martj42 CSV or directory.",
    )
    synthetic_kind = "League"
    source_value = ""
    if source_kind == "Synthetic":
        synthetic_kind = st.sidebar.radio(
            "Synthetic kind", ["League", "International"], index=1
        )
    elif source_kind == "Local file/dir":
        source_value = st.sidebar.text_input(
            "CSV file or directory", value="data/templates/club_league_template.csv"
        )

    st.sidebar.header("Model")
    if source_kind == "International (live)":
        # Dixon-Coles / xG / Ensemble are club models (DC's MLE over ~300 national
        # teams is impractical), so the national-team setting uses Elo, the engine
        # the World Cup backtests validated.
        model_options = ["Elo"]
        st.sidebar.caption(
            "National-team engine: **Elo** (margin-weighted, backtested on past "
            "World Cups). Dixon-Coles / xG suit club data — pick a club source to "
            "use them."
        )
    else:
        model_options = list(MODEL_FACTORIES)
    model_label = st.sidebar.selectbox("Model", model_options, index=0)

    st.sidebar.header("Simulation")
    sims = st.sidebar.slider("Simulations", 100, 20000, 5000, step=100)
    seed = int(st.sidebar.number_input("Seed", min_value=0, value=7, step=1))

    return {
        "source_kind": source_kind,
        "synthetic_kind": synthetic_kind,
        "source_value": source_value,
        "model_label": model_label,
        "sims": int(sims),
        "seed": seed,
        "is_synthetic": source_kind == "Synthetic",
    }


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def _team_index(teams: list[str], prefs: list[str], fallback: int) -> int:
    for p in prefs:
        if p in teams:
            return teams.index(p)
    return min(fallback, len(teams) - 1)


def _tab_match(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Match Predictor")
    intl = cfg["source_kind"] == "International (live)"
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    if len(teams) < 2:
        st.info("Need at least two teams in the data.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    home = c1.selectbox(
        "Home (listed) team", teams,
        index=_team_index(teams, ["Brazil", "Argentina", "Spain"], 0),
    )
    away = c2.selectbox(
        "Away team", teams,
        index=_team_index(teams, ["France", "England", "Germany"], 1),
    )
    # World Cup / international matches are neutral by default; clubs are not.
    neutral = c3.toggle("Neutral venue", value=intl)
    ou_line = c3.number_input("Over/under line", value=2.5, step=0.5)
    if home == away:
        st.info("Pick two different teams.")
        return

    # Optional squad-ability overlay (national teams only; Elo engine).
    overlay_on = sofa = False
    if intl and cfg["model_label"] == "Elo":
        overlay_on = st.toggle(
            "Squad-ability overlay (FIFA + Transfermarkt)", value=False,
            key="match_ability",
            help="Tilt Elo toward CURRENT squad ability, blended z-score-wise from "
            "EA-FC/FIFA overall + age-adjusted Transfermarkt. Backtested to beat "
            "plain Elo by a hair on past World Cups (see the backtest below).")
        if overlay_on:
            sofa = st.checkbox(
                "↳ also blend SofaScore club-season ratings (league-adjusted)",
                value=False, key="match_sofa",
                help="Adds league-adjusted SofaScore club-season ratings as a third "
                "source. Backtested to slightly dilute FIFA+TM — for experimentation.")
    ability_w = 0.5 if overlay_on else 0.0

    ov_label = ""
    if intl and cfg["model_label"] == "Elo" and overlay_on:
        model, ov_label = _match_overlay_model(
            _data_key(cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"]),
            cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"],
            ability_w, sofa)
    else:
        model = fit_model(
            cfg["model_label"],
            _data_key(cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"]),
            cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"],
        )
    # Capability check, NOT isinstance: a model cached by @st.cache_resource is built
    # from the class object as it was at cache time, so after a hot-reload
    # isinstance(cached_model, ScorelineModel) is False (the class identity differs)
    # even though the model is fine. hasattr survives reloads.
    if not hasattr(model, "score_matrix"):
        st.error("Selected model cannot produce a score matrix.")
        return
    if ov_label:
        st.caption(f"⚗️ Squad-ability overlay ON — sources: {ov_label}. "
                   "Toggle it off to compare.")

    bundle = predict_markets(
        model, home, away, context={"neutral_site": bool(neutral)}, ou_line=ou_line
    )

    probs = pd.DataFrame(
        {
            "Outcome": ["Home win", "Draw", "Away win"],
            "Probability": [bundle["home_win"], bundle["draw"], bundle["away_win"]],
        }
    )
    st.plotly_chart(
        px.bar(probs, x="Outcome", y="Probability", range_y=[0, 1], text_auto=".1%",
               title=f"{home} vs {away}"),
        use_container_width=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Expected goals",
        f"{bundle['expected_home_goals']:.2f} - {bundle['expected_away_goals']:.2f}",
    )
    m2.metric("Most likely score", bundle["most_likely_score"])
    m3.metric(
        f"Over {ou_line}",
        f"{bundle[f'over_{ou_line}']:.1%}",
    )

    f1, f2, f3 = st.columns(3)
    f1.metric("Fair home odds", f"{bundle['fair_home_odds']:.2f}")
    f2.metric("Fair draw odds", f"{bundle['fair_draw_odds']:.2f}")
    f3.metric("Fair away odds", f"{bundle['fair_away_odds']:.2f}")

    st.caption("Top scorelines")
    score_df = pd.DataFrame(
        [{"score": s, "probability": p}
         for s, p in bundle["scoreline_probabilities"].items()]
    )
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    # Live market comparison for this exact fixture (if a book is pricing it now).
    market = _market_for_fixture(home, away)
    st.divider()
    st.markdown("**Model vs the live market**")
    if market is None:
        st.caption(
            "No live market line for this fixture right now (the book only prices "
            "near-term games, and an Odds API key must be set). The fair odds above "
            "are the model's own."
        )
    else:
        model_p = [bundle["home_win"], bundle["draw"], bundle["away_win"]]
        labels = ["Home", "Draw", "Away"]
        rows = []
        for i, lab in enumerate(labels):
            edge = model_p[i] * market["odds"][i] - 1.0
            rows.append({
                "selection": lab,
                "model_prob": round(model_p[i], 3),
                "market_prob": round(float(market["market"][i]), 3),
                "best_odds": market["odds"][i],
                "edge": round(edge, 3),
            })
        mdf = pd.DataFrame(rows)
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        best = mdf.loc[mdf["edge"].idxmax()]
        suspect = (
            best["market_prob"] < 0.08 or best["best_odds"] > 12.0
            or abs(best["model_prob"] - best["market_prob"]) > 0.10
            or best["model_prob"] > 0.95
        )
        if best["edge"] > 0.05 and not suspect:
            st.success(f"Model sees value on **{best['selection']}** "
                       f"(edge {best['edge']:+.1%}). Hypothesis only — beat the "
                       f"closing line before trusting it.")
        elif best["edge"] > 0.05:
            st.warning(f"Largest 'edge' is on {best['selection']} ({best['edge']:+.1%}) "
                       "but it looks like model miscalibration (longshot / big "
                       "divergence), not real value.")
        else:
            st.caption("No qualifying edge — the model agrees with the book here.")

    # --- Backtest: how this predictor (and the overlay) scores out-of-sample ----
    if intl and cfg["model_label"] == "Elo":
        st.divider()
        with st.expander("📊 Backtest — how this predictor scores out-of-sample"):
            st.caption(
                "Every World Cup finals match since 1998, predicted pre-kickoff with "
                "the same Elo engine (fit only on results BEFORE each tournament — no "
                "leakage). 'Hit rate' = how often the highest-probability outcome was "
                "the actual result."
            )
            try:
                ml = _wc_matchlevel_backtest("All since 1998", "Elo")
                p, fl = ml["pooled"], ml["floor"]
                b1, b2, b3 = st.columns(3)
                b1.metric("Log loss", f"{p['log_loss']:.4f}",
                          f"{p['log_loss'] - fl['log_loss']:+.4f} vs floor",
                          delta_color="inverse")
                b2.metric("Hit rate (top pick correct)", f"{p['accuracy']:.1%}")
                b3.metric("Brier", f"{p['brier']:.4f}")
                st.caption(
                    f"No-skill floor (always predict the base rate): log loss "
                    f"{fl['log_loss']:.4f} over {ml['n']} matches — the gap is the skill."
                )
            except Exception as exc:
                st.warning(f"Backtest unavailable: {exc}")
            st.markdown(
                "**Does the squad-ability overlay help?** Plain Elo vs Elo+ability, "
                "match-level OOS on the clean editions (2018 & 2022)"
                + (" · +SofaScore (league-adjusted) blended" if sofa else "") + ":"
            )
            try:
                lb = _wc_ability_leaderboard(False, sofa).rename(
                    columns={"accuracy": "hit_rate"})
                st.dataframe(lb.round(4), use_container_width=True)
                if "elo" in lb.index:
                    d = float(lb["log_loss"].min() - lb.loc["elo", "log_loss"])
                    st.caption(
                        f"Best overlay vs plain Elo: {d:+.4f} log loss "
                        f"({'helps' if d < 0 else 'no improvement'}). Only two clean "
                        "editions (FIFA ratings start at 15) — underpowered, exploratory."
                    )
            except Exception as exc:
                st.warning(f"Ability comparison unavailable: {exc}")


def _market_for_fixture(home: str, away: str):
    """Live de-vigged market + best odds for a fixture, or None if not priced.

    Matches the fixture in either listing order (swapping home/away odds when the
    book lists the tie the other way round). Returns {market: [H,D,A] probs,
    odds: (oh,od,oa)} for the home/away as queried.
    """
    if odds_api.load_odds_api_key() is None:
        return None
    try:
        odds = _live_odds(odds_api.WORLD_CUP_SPORT, "best")
    except Exception:
        return None
    if odds.empty:
        return None
    h, a = normalize_team_name(home), normalize_team_name(away)
    for r in odds.itertuples(index=False):
        if r.home_team == h and r.away_team == a:
            oh, od, oa = r.home_odds, r.draw_odds, r.away_odds
            return {"market": implied_probabilities(oh, od, oa), "odds": (oh, od, oa)}
        if r.home_team == a and r.away_team == h:  # reversed listing -> swap H/A
            oa, od, oh = r.home_odds, r.draw_odds, r.away_odds
            return {"market": implied_probabilities(oh, od, oa), "odds": (oh, od, oa)}
    return None


def _tab_tournament(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Tournament Simulator")
    from ..simulation import rules
    from ..simulation.tournament import simulate_tournament

    fmt = rules.world_cup_2026_format()
    st.caption(f"Format: {fmt.name}.  Assumptions: {fmt.notes}")

    # Default to the real, verified WC 2026 draw + fixtures (with played results)
    # when they exist; otherwise the placeholder template. The dedicated
    # "🏆 World Cup 2026 (live)" tab does this end-to-end with the market anchor.
    import os as _os

    _real_groups = "data/world_cup_2026_groups.csv"
    _real_fixtures = "data/world_cup_2026_fixtures.csv"
    groups_path = st.text_input(
        "Groups CSV (group,team)",
        value=_real_groups if _os.path.exists(_real_groups)
        else "data/templates/world_cup_2026_groups_template.csv",
    )
    fixtures_path = st.text_input(
        "Fixtures CSV (optional, locks in played results)",
        value=_real_fixtures if _os.path.exists(_real_fixtures) else "",
    )

    if not st.button("Run tournament simulation", type="primary"):
        st.info("Set a group draw and press the button to Monte-Carlo the tournament.")
        return

    try:
        groups = loaders.load_groups(groups_path)
    except Exception as exc:
        st.error(f"Could not load groups: {exc}")
        return
    fixtures = None
    if fixtures_path:
        try:
            fixtures = loaders.load_fixtures(fixtures_path)
        except Exception as exc:
            st.error(f"Could not load fixtures: {exc}")
            return

    model = fit_model(
        cfg["model_label"],
        _data_key(cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"]),
        cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"],
    )

    with st.spinner(f"Simulating {cfg['sims']} tournaments..."):
        table = simulate_tournament(
            model, groups, fixtures=fixtures,
            n_simulations=cfg["sims"], seed=cfg["seed"], fmt=fmt,
        )

    st.dataframe(table, use_container_width=True, hide_index=True)
    if "champion_probability" in table.columns and not table.empty:
        top = table.head(16).copy()
        st.plotly_chart(
            px.bar(
                top, x="champion_probability", y="team", color="group",
                orientation="h", text="champion_probability",
                title="Title probability (top 16)",
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            use_container_width=True,
        )


WC_BACKTEST_EDITIONS = [
    "All since 1998", "2026", "2022", "2018", "2014", "2010", "2006", "2002", "1998",
]


@st.cache_data(show_spinner="Backtesting past World Cups...", ttl=3600)
def _wc_matchlevel_backtest(edition: str, model_label: str = "Elo") -> dict:
    """Match-level OOS backtest of a model on past World Cups (+ no-skill floor).

    Every finals match is predicted pre-kickoff by a model fit only on prior
    international results. The bar is the no-skill floor (always predict the H/D/A
    base rate) -- the gap between the two is the model's real skill.
    """
    hist = _live_history()
    years = None if edition == "All since 1998" else [int(edition)]
    res = wc_backtest.backtest_world_cups(
        hist, WC_MODELS[model_label], years=years, min_year=1998, update_within=True
    )
    out = np.asarray(res["outcomes"], dtype=object)
    floor: dict = {}
    if len(out):
        base = np.array([(out == k).mean() for k in ("H", "D", "A")])
        floor = metrics.all_metrics(np.tile(base, (len(out), 1)), out)
    per = pd.DataFrame(res["editions"]).T if res["editions"] else pd.DataFrame()
    return {"pooled": res["pooled"], "floor": floor, "per_edition": per, "n": int(len(out))}


def _wc_elo_factory(host_adv: float = 0.0, ability_by_year: dict | None = None,
                    ability_w: float = 0.0):
    """Edition-aware Elo factory: host home bump (per-edition WC_HOSTS) + optional
    squad-ability tilt. Used in the all-neutral SIMULATION backtest path. NOT a
    cached function -- it returns a closure (unpicklable); its callers are cached."""
    def fac(year):
        kw: dict = {}
        if host_adv > 0:
            kw.update(hosts=wc_backtest.WC_HOSTS.get(year, ()), host_advantage=host_adv)
        if ability_w > 0 and ability_by_year and year in ability_by_year:
            kw.update(squad_strength=ability_by_year.get(year), squad_weight=ability_w)
        return EloGoalsModel(**kw)
    return fac


@st.cache_data(show_spinner="Simulating past World Cups...", ttl=3600)
def _wc_tournament_backtest(edition: str, n_sims: int, model_label: str = "Elo",
                            host_adv: float = 0.0) -> dict:
    """Full tournament-simulation scorecard (champion rank + stage calibration)."""
    hist = _live_history()
    years = None if edition == "All since 1998" else [int(edition)]
    factory = WC_MODELS[model_label]
    if model_label == "Elo" and host_adv > 0:
        factory = _wc_elo_factory(host_adv=host_adv)
    return wc_backtest.backtest_tournament(
        hist, factory, years=years, min_year=1998, n_simulations=n_sims
    )


@st.cache_data(show_spinner="Loading FIFA squad-ability ratings...", ttl=3600)
def _wc_fifa_ability(sofa: bool = False) -> dict:
    """{year: Elo-point adjustments} from FIFA squad ability, optionally SUPPLEMENTED
    per edition by data/ability_overlay_<year>.csv (e.g. that season's SofaScore
    ratings), blended z-score-wise where both exist. ``sofa`` also blends the
    backtested SofaScore club-season ratings (data/sofascore_ratings_<year>.csv) --
    note these were measured to DILUTE the FIFA+TM blend; for experimentation."""
    from ..data import players, squad_value
    byyear = squad_value.build_fifa_ability_by_year(list(squad_value.WC_FIFA_VERSION))
    out = {}
    for y, fifa in byyear.items():
        sources = [fifa]
        try:  # age-adjusted Transfermarkt ability for that edition
            tdf = squad_value.build_tm_ability(y)
            td = dict(zip(tdf["team"], tdf["ability"]))
            if td:
                sources.append(td)
        except Exception:
            pass
        extra_csvs = [Path(f"data/ability_overlay_{y}.csv")]
        if sofa:
            extra_csvs.append(Path(f"data/sofascore_ratings_{y}.csv"))
        for csv in extra_csvs:
            if csv.exists():
                try:
                    extra = squad_value.load_ability_csv(str(csv))
                    if extra:
                        sources.append(extra)
                except Exception:
                    pass
        combined = squad_value.combine_ability(sources)
        out[y] = players.to_elo_adjustment(combined, spread=60.0, log=False)
    return out


@st.cache_data(show_spinner="Loading actual tournament starters...", ttl=3600)
def _wc_actual_xi_ability() -> dict:
    """{year: Elo-point adjustments} weighting each squad by who ACTUALLY started
    that World Cup (jfjelstul line-ups; strength from FIFA + age-adjusted TM + any
    per-player data/ability_overlay_<year>.csv). LEAKY -- diagnostic only."""
    from ..data import players, squad_value
    out = {}
    for y in squad_value.WC_FIFA_VERSION:
        if y in squad_value.FIFA_LEAKAGE_EDITIONS:
            continue
        try:
            df = squad_value.build_actual_xi_ability(
                y, csv_path=f"data/ability_overlay_{y}.csv")
            if len(df):
                out[y] = players.to_elo_adjustment(
                    dict(zip(df["team"], df["ability"])), spread=60.0, log=False)
        except Exception:
            continue
    return out


@st.cache_data(show_spinner="Backtesting squad ability vs Elo...", ttl=3600)
def _wc_ability_leaderboard(include_actual_xi: bool = False, sofa: bool = False) -> pd.DataFrame:
    """Match-level OOS: plain Elo vs Elo+ability on the clean editions. The
    rating-rank XI is leakage-free; the optional actual-tournament-XI variant
    (starred) is a leaky diagnostic. ``sofa`` blends in SofaScore club-season
    ratings (backtested to dilute -- for experimentation)."""
    from ..data import squad_value
    adj = _wc_fifa_ability(sofa)
    clean = [y for y in squad_value.WC_FIFA_VERSION if y not in squad_value.FIFA_LEAKAGE_EDITIONS]
    facs = {
        "elo": EloGoalsModel,
        "elo+ability(rank-XI)@0.5": lambda year: EloGoalsModel(
            squad_strength=adj.get(year), squad_weight=0.5),
        "elo+ability(rank-XI)@1.0": lambda year: EloGoalsModel(
            squad_strength=adj.get(year), squad_weight=1.0),
    }
    if include_actual_xi:
        axi = _wc_actual_xi_ability()
        facs["elo+ability(actual-XI)@0.5 *leaky"] = lambda year: EloGoalsModel(
            squad_strength=axi.get(year), squad_weight=0.5)
        facs["elo+ability(actual-XI)@1.0 *leaky"] = lambda year: EloGoalsModel(
            squad_strength=axi.get(year), squad_weight=1.0)
    return wc_backtest.compare_models_on_world_cups(
        _live_history(), facs, years=clean, min_year=min(clean)
    )


@st.cache_data(show_spinner="Simulating that World Cup...", ttl=3600)
def _wc_past_simulation(
    year: int, n_sims: int, ability_w: float = 0.0, model_label: str = "Elo",
    host_adv: float = 0.0, sofa: bool = False,
) -> dict | None:
    """Groups + full per-team win/advancement probabilities for one past edition.
    The FIFA squad-ability tilt and the host home bump only apply to the Elo engine
    (only Elo has those terms); other models are simulated as-is. ``sofa`` blends in
    SofaScore club-season ratings (backtested to dilute -- for experimentation)."""
    if model_label == "Elo" and (ability_w > 0 or host_adv > 0):
        adj = _wc_fifa_ability(sofa) if ability_w > 0 else None
        fac = _wc_elo_factory(host_adv=host_adv, ability_by_year=adj, ability_w=ability_w)
        return wc_backtest.simulate_past_world_cup(
            _live_history(), fac, year, n_simulations=n_sims)
    return wc_backtest.simulate_past_world_cup(
        _live_history(), WC_MODELS[model_label], year, n_simulations=n_sims
    )


_FINISH_ORDER = [
    ("champion", "🏆 Champion"), ("final", "Final"), ("semifinal", "Semifinal"),
    ("quarterfinal", "Quarterfinal"), ("round_of_16", "Round of 16"),
]


def _actual_finish(team: str, reached: dict | None) -> str:
    """Deepest stage a team actually reached that edition (for the results column)."""
    if not reached:
        return ""
    for key, label in _FINISH_ORDER:
        if team in reached.get(key, set()):
            return label
    return "Group stage"


def _render_past_wc_simulation(
    year: int, n_sims: int, ability_w: float = 0.0, model_label: str = "Elo",
    host_adv: float = 0.0, sofa: bool = False,
) -> None:
    """Groups + pre-tournament win/advancement probabilities for one past edition,
    shown next to what actually happened (the live tab's view, run on history)."""
    res = _wc_past_simulation(year, n_sims, ability_w, model_label, host_adv, sofa)
    if res is None:
        st.info("Couldn't reconstruct that edition's groups from the fixtures.")
        return
    if host_adv > 0 and year in wc_backtest.WC_HOSTS:
        hosts = ", ".join(wc_backtest.WC_HOSTS[year])
        st.caption(f"🏟️ Host bump +{host_adv:.0f} Elo applied to: {hosts}.")
    if ability_w > 0:
        from ..data import squad_value
        if year in squad_value.WC_FIFA_VERSION:
            leak = " (FIFA 15 post-dates the 2014 WC — leakage)" if year in squad_value.FIFA_LEAKAGE_EDITIONS else ""
            sf = " + SofaScore club-season ratings" if sofa else ""
            st.caption(f"⚗️ FIFA squad-ability overlay ON (weight {ability_w}){sf}{leak}.")
        else:
            st.info(f"No FIFA ability ratings for {year} (mirror covers 2018+); showing plain Elo.")
    groups, sims, reached = res["groups"], res["sims"], res["reached"]
    if res["champion"]:
        fav = sims.iloc[0]
        st.success(
            f"Actual champion: **{res['champion']}** — the model ranked them "
            f"#{res['champion_rank']} of {len(sims)} pre-tournament. Its favourite "
            f"was {fav['team']} ({fav['champion_probability']:.1%})."
        )
    st.markdown(f"**Groups — {year}** (reconstructed from the fixtures; labels arbitrary)")
    gdf = pd.DataFrame(
        [{"Group": g, "Teams": ", ".join(t)} for g, t in groups.items()]
    )
    st.dataframe(gdf, use_container_width=True, hide_index=True)

    disp = sims.copy()
    disp.insert(2, "actual_finish", [_actual_finish(t, reached) for t in disp["team"]])
    st.markdown(
        f"**Pre-tournament win probability — {year}** (model fit only on data "
        "before the tournament; `actual_finish` = how far the team really got)"
    )
    fig = px.bar(
        disp.head(16), x="champion_probability", y="team", color="group",
        orientation="h", text="champion_probability",
    ).update_layout(yaxis={"categoryorder": "total ascending"},
                    title=f"Champion probability — {year} (top 16)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button(
        "Download probabilities (CSV)", disp.to_csv(index=False),
        file_name=f"world_cup_{year}_backtest.csv", mime="text/csv", key="wc_bt_dl",
    )


def _tab_backtest(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Backtest Lab")

    # --- World Cup backtest (international, 1998+) -- the high-power test ------
    st.markdown("#### 🏆 World Cup backtest (international, 1998+)")
    st.caption(
        "The high-power test: every World Cup finals match since 1998 predicted "
        "pre-kickoff by the chosen model, fit only on prior results (no leakage). "
        "The bar is the no-skill floor (predict the base rate) -- the gap is real "
        "skill. Historical odds aren't free, so the market isn't shown."
    )
    e1, e2, e3 = st.columns([1.1, 1, 1])
    edition = e1.selectbox("Edition", WC_BACKTEST_EDITIONS, key="wc_bt_edition")
    model_label = e2.selectbox(
        "Model", list(WC_MODELS), key="wc_bt_model",
        help="Elo is the national-team engine. Dixon-Coles is a CLUB model — it runs "
        "on international data but is slow (~30s/fit) and miscalibrated on neutral "
        "venues, so it scores worse than the no-skill floor and gives degenerate "
        "sims; offered so you can see that. xG / Ensemble need club xG (internationals "
        "have none), so they are omitted.",
    )
    n_sims = int(e3.slider("Sim runs", 1000, 8000, 3000, step=1000, key="wc_bt_sims"))
    cc1, cc2 = st.columns(2)
    run_tourney = cc1.checkbox(
        "Also simulate the bracket (champion rank + calibration)",
        value=False, key="wc_bt_tourney",
        help="Monte-Carlo each completed tournament the way the live tab does — "
        "slower. Scores how well the simulated advancement probabilities matched "
        "reality.",
    )
    ability_on = cc2.toggle(
        "FIFA squad-ability overlay (experimental, Elo only)", value=False,
        key="wc_bt_ability",
        help="Squad ability as a pre-tournament overlay, blended z-score-wise from "
        "EA-FC/FIFA overall ratings + age-adjusted Transfermarkt ability (+ any "
        "data/ability_overlay_<year>.csv you add). Adds a plain-Elo vs Elo+ability "
        "head-to-head on the clean editions (2018 & 2022) and tilts the "
        "single-edition probabilities. Small OOS improvement, 2 clean editions "
        "(underpowered) — exploratory. Applies to Elo only.",
    )
    ability_w = 0.5 if (ability_on and model_label == "Elo") else 0.0
    compare_axi = False
    if ability_on and model_label == "Elo":
        compare_axi = st.checkbox(
            "↳ also compare ACTUAL tournament starters (leaky diagnostic)",
            value=False, key="wc_bt_axi",
            help="Adds an Elo+ability variant weighting each squad by who actually "
            "started/subbed that World Cup (jfjelstul line-ups; strength from FIFA + "
            "age-adjusted Transfermarkt + any per-player data/ability_overlay_<year>"
            ".csv). It LEAKS (uses in-tournament selections), so it's an upper-bound "
            "diagnostic for comparison, NOT a valid out-of-sample score.",
        )
    sofa_on = False
    if ability_on and model_label == "Elo":
        sofa_on = st.checkbox(
            "↳ also blend SofaScore club-season ratings (experimental)",
            value=False, key="wc_bt_sofa",
            help="Adds SofaScore club-season player ratings for the season BEFORE each "
            "World Cup (data/sofascore_ratings_<year>.csv, built by "
            "scripts/build_sofascore_overlays.py) as a third source in the ability "
            "blend. League-adjusted (a 7.5 in a weak league anchors below a 7.5 in a "
            "strong one). Backtested OOS it still slightly DILUTES the FIFA+TM blend "
            "(Elo+FIFA+TM @0.5 = 0.9902 vs +SofaScore @0.5 = 0.9921) — included so "
            "you can see that; leave off for the best overlay.",
        )
    host_on = st.toggle(
        "Host advantage in the bracket sim (backtested, Elo only)", value=False,
        key="wc_bt_host",
        help="Give each edition's host(s) a home bump in the all-neutral bracket "
        "simulation. (The match-level score above already honours host advantage "
        "via the data's neutral_site flag, so this affects only the bracket sim and "
        "the single-edition probabilities.) Backtested leakage-free on host games "
        "1998-2022: helps 5/7 editions (2014 Brazil & 2022 Qatar are "
        "counterexamples), pooled-optimal ~100 Elo.",
    )
    host_adv = 0.0
    if host_on and model_label == "Elo":
        host_adv = float(st.slider(
            "↳ host home bump (Elo pts)", 0, 120,
            int(wc_backtest.DEFAULT_HOST_ADVANTAGE), step=5, key="wc_bt_host_adv"))
    if model_label != "Elo":
        st.caption(
            "⚠️ Dixon-Coles is a club model run on neutral-venue international data — "
            "expect weak/erratic numbers (it's here to show why Elo is the engine for "
            "national teams), and it's slow (~30s/fit). Ability/host overlays are Elo-only."
        )
    if st.button("Run World Cup backtest", type="primary", key="wc_bt_run"):
        try:
            ml = _wc_matchlevel_backtest(edition, model_label)
        except Exception as exc:
            st.error(f"World Cup backtest failed: {exc}")
        else:
            if ml["n"] == 0:
                st.warning("No finals matches for that selection yet.")
            else:
                p, floor = ml["pooled"], ml["floor"]
                cols = st.columns(5)
                cols[0].metric(
                    "Log loss", f"{p['log_loss']:.4f}",
                    f"{p['log_loss'] - floor['log_loss']:+.4f} vs floor",
                    delta_color="inverse",
                )
                cols[1].metric("RPS", f"{p['rps']:.4f}")
                cols[2].metric("Brier", f"{p['brier']:.4f}")
                cols[3].metric("Accuracy", f"{p['accuracy']:.1%}")
                cols[4].metric("ECE", f"{p['ece']:.4f}")
                st.caption(
                    f"No-skill floor — always predict the base rate: log loss "
                    f"{floor['log_loss']:.4f} · RPS {floor['rps']:.4f} · Brier "
                    f"{floor['brier']:.4f}. The model beats it by "
                    f"{floor['log_loss'] - p['log_loss']:+.4f} log loss over "
                    f"{ml['n']} matches — that gap is the skill."
                )
                if ability_on and model_label == "Elo":
                    try:
                        lb = _wc_ability_leaderboard(compare_axi, sofa_on)
                        st.markdown(
                            "**Squad-ability vs plain Elo** — match-level OOS on the "
                            "clean editions (2018 & 2022), lower log loss = better"
                            + (" · +SofaScore club-season ratings blended" if sofa_on else "")
                            + (" · `*leaky` rows weight the ACTUAL tournament XI"
                               if compare_axi else "")
                        )
                        st.dataframe(lb.round(4), use_container_width=True)
                        if "elo" in lb.index:
                            d = float(lb["log_loss"].min() - lb.loc["elo", "log_loss"])
                            st.caption(
                                f"Best variant vs plain Elo: {d:+.4f} log loss "
                                f"({'helps' if d < 0 else 'no improvement'}). Only TWO "
                                "clean editions (FIFA starts at 15; 2014 = post-WC "
                                "leakage) — underpowered, exploratory."
                                + (" The `*leaky` actual-XI rows use in-tournament "
                                   "line-ups (selections/injuries revealed), so they "
                                   "flatter the model — a diagnostic upper bound, not a "
                                   "valid OOS number." if compare_axi else "")
                            )
                    except Exception as exc:
                        st.warning(f"Ability comparison unavailable: {exc}")
                if edition == "All since 1998":
                    if not ml["per_edition"].empty:
                        st.caption("Per edition")
                        st.dataframe(
                            ml["per_edition"][
                                ["n", "log_loss", "rps", "brier", "accuracy", "ece"]
                            ].round(4),
                            use_container_width=True,
                        )
                    if run_tourney:
                        res = _wc_tournament_backtest(edition, n_sims, model_label, host_adv)
                        eds = res.get("editions")
                        if eds is None or len(eds) == 0:
                            st.info("No completed tournament to simulate for that selection.")
                        else:
                            st.caption("Tournament simulation — champion the model ranked")
                            st.dataframe(eds, use_container_width=True, hide_index=True)
                            st.caption(
                                "Stage-reach calibration — Brier skill vs the base rate "
                                "(>0 means the advancement probabilities carry real info)"
                            )
                            st.dataframe(res["calibration"], use_container_width=True,
                                         hide_index=True)
                            cs = res["champion_skill"]
                            st.caption(
                                f"Champion log loss: model {cs['mean_neglogp_model']} vs "
                                f"uniform {cs['mean_neglogp_uniform']} over "
                                f"{cs['editions']} editions (lower = better)."
                            )
                elif edition == "2026":
                    st.info(
                        "2026 is in progress — open the **🏆 World Cup 2026 (live)** "
                        "tab for its groups and live, updating win probabilities."
                    )
                else:
                    _render_past_wc_simulation(
                        int(edition), n_sims, ability_w, model_label, host_adv, sofa_on)

    st.divider()

    # --- Walk-forward on the loaded data (club / synthetic) ------------------
    st.markdown("#### Walk-forward on the loaded data")
    st.caption(
        "Expanding-window walk-forward only -- random k-fold leaks the future "
        "and produces fantasy accuracy."
    )

    c1, c2 = st.columns(2)
    folds = int(c1.slider("Folds", 2, 10, 5))
    min_train = c2.slider("First training fraction", 0.3, 0.8, 0.5, step=0.05)

    if not st.button("Run walk-forward backtest", key="wf_run"):
        st.info("Press the button to run the time-aware backtest on the loaded data.")
        return

    factory = MODEL_FACTORIES[cfg["model_label"]]
    with st.spinner("Walking forward..."):
        probs, outcomes, test_df = walk_forward.walk_forward(
            factory, df, min_train_frac=min_train, n_folds=folds
        )

    if len(outcomes) == 0:
        st.warning("No test rows were produced; provide more data.")
        return

    m = metrics.all_metrics(probs, outcomes)
    cols = st.columns(5)
    cols[0].metric("Log loss", f"{m['log_loss']:.4f}")
    cols[1].metric("RPS", f"{m['rps']:.4f}")
    cols[2].metric("Brier", f"{m['brier']:.4f}")
    cols[3].metric("Accuracy", f"{m['accuracy']:.1%}")
    cols[4].metric("ECE", f"{m['ece']:.4f}")

    # Model vs the book (only on rows the book can price).
    if schemas.has_usable_odds(df):
        cmp = walk_forward.compare_to_market(
            factory, df, min_train_frac=min_train, n_folds=folds
        )
        delta = cmp.get("model_minus_market_log_loss", float("nan"))
        st.metric(
            "Model - Market log loss",
            f"{delta:+.4f}",
            help="Negative = the model beats the de-vigged book on log loss.",
        )

    st.caption("Calibration table (reliability of stated probabilities)")
    cal = reports.calibration_table(probs, outcomes)
    st.dataframe(cal, use_container_width=True, hide_index=True)
    if not cal.empty:
        st.plotly_chart(
            px.scatter(
                cal, x="mean_predicted", y="observed_freq", size="count",
                title="Reliability diagram (closer to the diagonal = better)",
            ).add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                        line=dict(dash="dot")),
            use_container_width=True,
        )


def _tab_betting(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Betting Edge Scanner")
    st.warning(
        "A POSITIVE edge/ROI is a HYPOTHESIS, not proof. It can be overfitting, "
        "survivorship in the odds, or beating opening rather than closing prices. "
        "Check calibration (Backtest Lab) and closing-line value before trusting any edge."
    )

    # 1) LIVE scan against The Odds API -- works for the World Cup regardless of
    #    whether the loaded historical data carries odds.
    model = fit_model(
        cfg["model_label"],
        _data_key(cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"]),
        cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"],
    )
    if hasattr(model, "score_matrix"):  # capability check (reload-proof vs isinstance)
        _live_odds_section(
            model, neutral=cfg["source_kind"] == "International (live)"
        )

    # 2) Historical value-betting backtest -- only when the loaded data has odds
    #    (club leagues / football-data). International history has none.
    st.divider()
    st.markdown("**Historical value-betting backtest (out-of-sample)**")
    if not schemas.has_usable_odds(df):
        st.info(
            "The loaded data has no historical 1X2 odds (e.g. international results "
            "have none free), so there is nothing to backtest here. Use the live "
            "scan above for the World Cup, or load a club league (football-data, "
            "which carries closing odds) to backtest value betting."
        )
        return

    c1, c2 = st.columns(2)
    edge = c1.slider("Edge threshold", 0.0, 0.30, 0.05, step=0.01)
    folds = int(c2.slider("Walk-forward folds", 2, 10, 5))
    if not st.button("Run historical betting backtest", type="primary"):
        st.caption("Press the button to backtest value betting on this data.")
        return

    factory = MODEL_FACTORIES[cfg["model_label"]]
    with st.spinner("Scanning for value bets out-of-sample..."):
        probs, outcomes, test_df = walk_forward.walk_forward(
            factory, df, n_folds=folds
        )
    if len(test_df) == 0:
        st.warning("No out-of-sample rows produced.")
        return

    result = betting.betting_backtest(
        probs, test_df, edge_threshold=edge, best_edge_only=True
    )
    clv = betting.closing_line_value(probs, test_df, edge_threshold=edge)

    cols = st.columns(4)
    cols[0].metric("Bets placed", result["n_bets"])
    cols[1].metric("ROI", f"{result['roi']:.2%}")
    cols[2].metric("Hit rate", f"{result['hit_rate']:.1%}")
    cols[3].metric("Max drawdown", f"{result['max_drawdown']:.1f}")
    if not np.isnan(clv.get("pct_beat_close", float("nan"))):
        st.metric(
            "Beat the close",
            f"{clv['pct_beat_close']:.1%}",
            help="Fraction of bets that took a price higher than the closing "
            "line -- the strongest cheap evidence of genuine edge.",
        )

    # Fair odds vs market odds vs edge, per out-of-sample fixture.
    with_odds_mask = schemas.rows_with_complete_odds(test_df).index
    keep = test_df.index.isin(with_odds_mask)
    sub = test_df.loc[keep].reset_index(drop=True)
    sub_probs = probs[keep]
    if len(sub) == 0:
        return

    odds = sub[schemas.ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy()
    labels = ["Home", "Draw", "Away"]
    records = []
    for i in range(len(sub)):
        for c in range(3):
            o = odds[i, c]
            if not np.isfinite(o) or o <= 1.0:
                continue
            p = float(sub_probs[i, c])
            records.append(
                {
                    "fixture": f"{sub.iloc[i]['home_team']} vs {sub.iloc[i]['away_team']}",
                    "selection": labels[c],
                    "model_prob": round(p, 4),
                    "fair_odds": round(1.0 / p, 3) if p > 1e-9 else float("inf"),
                    "market_odds": round(float(o), 3),
                    "edge": round(p * o - 1.0, 4),
                }
            )
    edge_df = pd.DataFrame(records)
    edge_df = edge_df[edge_df["edge"] > edge].sort_values("edge", ascending=False)
    st.caption(f"Qualifying value selections (edge > {edge:.0%})")
    st.dataframe(edge_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Live data (cached) -- World Cup 2026 + The Odds API
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Fetching real international results...", ttl=900)
def _live_history() -> pd.DataFrame:
    # Tournament-fresh TTL so newly-played games are picked up within hours, not
    # the 7-day off-season default.
    return sources.fetch_international_results(ttl_days=world_cup._WC_HISTORY_TTL_DAYS)


@st.cache_data(show_spinner="Fetching live results...", ttl=900)
def _live_scores() -> pd.DataFrame:
    """Live completed-match scores from the Odds API (martj42 lags), or empty."""
    try:
        return odds_api.fetch_scores()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner="Saving results...", ttl=900)
def _live_results_store() -> pd.DataFrame:
    """Fold today's group games (martj42 + Odds-API) into the durable ledger.

    Persisting here means a game stays locked in on every later run even after it
    rolls out of the Odds-API window or the martj42 cache, so the live tab never
    "loses" a result it has already shown. Scoped to the 72 group pairings.
    """
    return world_cup.update_results_store(
        _live_history(), _live_scores(), valid_pairs=world_cup.group_pair_keys()
    )


def _live_results_fingerprint() -> str:
    """A content hash of every group result played so far (pair -> score).

    Used as the model/sim cache key INSTEAD of a row count: a score correction or
    a net-zero feed transition (one game absorbed by martj42 as another rolls out
    of the Odds-API window) leaves the count unchanged but the content different,
    which would otherwise serve a stale model and simulation.
    """
    import hashlib

    played = _live_results_store()  # persisted + group-scoped ledger
    parts = sorted(
        f"{world_cup._pair_key(h, a)}={int(hs)}-{int(as_)}"
        for h, a, hs, as_ in zip(
            played["home_team"], played["away_team"],
            played["home_score"], played["away_score"],
        )
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


@st.cache_data(show_spinner="Assembling live training data...", ttl=900)
def _live_train(as_of: str = "") -> pd.DataFrame:
    """Completed history + every WC game played so far (so the model sees them).
    ``as_of`` (ISO date) rewinds the training to games on/before that day, for the
    "probabilities as of an earlier point" view."""
    _live_results_store()  # persist before the trainer reads the ledger
    df = world_cup.live_training_frame(use_odds_api_scores=True)
    if as_of:
        df = df[pd.to_datetime(df["date"], errors="coerce") <= pd.Timestamp(as_of)]
    return df


@st.cache_data(show_spinner="Building squad-ability overlay...", ttl=3600)
def _live_ability(sofa: bool = False) -> tuple[dict, str]:
    """Ability overlay for the live model: FIFA squad ability, SUPPLEMENTED by any
    rating-site CSVs you drop in (data/ability_overlay_*.csv -- SofaScore, etc.),
    blended z-score-wise. ``sofa`` also blends the backtested SofaScore club-season
    ratings (data/sofascore_ratings_2026.csv; measured to DILUTE FIFA+TM -- for
    experimentation). Returns (Elo-point adjustments, source label); empty if
    nothing is available (overlay then no-ops -> pure Elo)."""
    from ..data import players, squad_value
    sources: list[dict] = []
    labels: list[str] = []
    try:  # FIFA ability is the auto-fetched base
        df = squad_value.build_fifa_current_ability()
        d = dict(zip(df["team"], df["ability"]))
        if d:
            sources.append(d)
            labels.append(f"FIFA {int(df['source_year'].iloc[0])} ({len(d)})")
    except Exception:
        pass
    try:  # age-adjusted Transfermarkt ability (de-aged value) as another source
        tdf = squad_value.build_current_ability()
        td = dict(zip(tdf["team"], tdf["ability"]))
        if td:
            sources.append(td)
            labels.append(f"TM-deaged {int(tdf['source_year'].iloc[0])} ({len(td)})")
    except Exception:
        pass
    supplements = sorted(Path("data").glob("ability_overlay_*.csv"))
    if sofa:  # optional SofaScore club-season ratings (backtested to dilute FIFA+TM)
        sf = Path("data/sofascore_ratings_2026.csv")
        if sf.exists():
            supplements.append(sf)
    for p in supplements:
        try:
            d = squad_value.load_ability_csv(str(p))
            if d:
                sources.append(d)
                labels.append(
                    f"{p.stem.replace('ability_overlay_', '').replace('sofascore_ratings_', 'SofaScore ')} ({len(d)})")
        except Exception:
            continue
    if not sources:
        return {}, "no ability source available"
    combined = squad_value.combine_ability(sources)  # z-scored blend, only-where-present
    if not combined:
        return {}, "no usable ability rows"
    # `combined` is already z-scored; to_elo_adjustment re-normalises to spread.
    return players.to_elo_adjustment(combined, spread=60.0, log=False), " + ".join(labels)


@st.cache_resource(show_spinner="Fitting model on real history...")
def _live_model(
    model_label: str, fingerprint: str, ability_w: float = 0.0, host_adv: float = 0.0,
    sofa: bool = False, as_of: str = "",
) -> object:
    del fingerprint  # cache key only; new/changed results -> new key -> refit
    train = _live_train(as_of)  # as_of rewinds the fit for the time-machine view
    factory = MODEL_FACTORIES[model_label]
    # Only the Elo engine takes the squad overlay / host bump; both feed the
    # all-neutral simulation, where 2026's hosts (US/Canada/Mexico) get no edge
    # otherwise. Backtested: host advantage helps 5/7 past WCs (see WC_HOSTS).
    if isinstance(factory(), EloGoalsModel) and (ability_w > 0 or host_adv > 0):
        kwargs: dict = {}
        if ability_w > 0:
            adj, _ = _live_ability(sofa)
            if adj:
                kwargs.update(squad_strength=adj, squad_weight=ability_w)
        if host_adv > 0:
            kwargs.update(hosts=wc_backtest.WC_HOSTS.get(2026, ()), host_advantage=host_adv)
        if kwargs:
            return EloGoalsModel(**kwargs).fit(train.copy())
    return factory().fit(train.copy())


def _asof_conditioned(as_of: str):
    """(groups, fixtures, known_ko_results) locked to games on/before ``as_of``
    (all completed WC games when as_of is empty). Uses build_fixtures(out=None) so
    a rewound view never overwrites the shared live fixtures file."""
    groups = world_cup.load_groups()
    hist, sc = _live_history(), _live_scores()
    if as_of:
        cutoff = pd.Timestamp(as_of)
        hist = hist[pd.to_datetime(hist["date"], errors="coerce") <= cutoff]
        if sc is not None and len(sc):
            sc = sc[pd.to_datetime(sc["date"], errors="coerce") <= cutoff]
    fixtures = world_cup.build_fixtures(hist, groups, out=None, extra_results=sc)
    known = world_cup.completed_ko_results(hist, sc, groups)
    return groups, fixtures, known


@st.cache_data(show_spinner="Finding tournament checkpoints...", ttl=900)
def _asof_options(fingerprint: str) -> list:
    """(label, as_of) checkpoints for the time-machine dropdown: Now, each earlier
    completed WC matchday, the end of the group stage, and pre-tournament."""
    del fingerprint
    opts: list = [("Now (latest results)", "")]
    try:
        groups = world_cup.load_groups()
        grp = world_cup.group_pair_keys(groups)
        hist, sc = _live_history(), _live_scores()
        rows = []
        for f in (world_cup.wc2026_matches(hist), sc):
            if f is None or len(f) == 0:
                continue
            for r in f.itertuples(index=False):
                h, a = getattr(r, "home_team", ""), getattr(r, "away_team", "")
                if world_cup._has_score(getattr(r, "home_score", None), getattr(r, "away_score", None)):
                    dt = pd.to_datetime(getattr(r, "date", None), errors="coerce")
                    if pd.notna(dt):
                        rows.append((dt.normalize(), world_cup._pair_key(h, a) in grp))
        if not rows:
            return opts
        df = pd.DataFrame(rows, columns=["dt", "is_group"]).sort_values("dt")
        days = list(df["dt"].unique())
        group_end = df[df["is_group"]]["dt"].max() if df["is_group"].any() else None
        for dt in reversed(days[:-1]):  # every earlier matchday (latest == "Now")
            ds = pd.Timestamp(dt).strftime("%Y-%m-%d")
            n = int((df["dt"] <= dt).sum())
            tag = " · end of group stage" if group_end is not None and dt == group_end else ""
            opts.append((f"As of {ds} ({n} games in){tag}", ds))
        start = (pd.Timestamp(days[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        opts.append(("Before the tournament (pre-tournament projection)", start))
    except Exception:
        pass
    return opts


@st.cache_data(show_spinner="Reading actual R32 fixtures...", ttl=900)
def _live_r32_third_pins(fingerprint: str, as_of: str = "") -> dict:
    """{r32_match_id: group} best-third slots read off the ACTUAL published R32
    fixtures (live odds feed) + the locked, completed group standings. Empty until
    real KO fixtures appear; grows as groups finalise. So the simulated bracket
    matches reality instead of the solver's guess once the draw is known."""
    del fingerprint  # cache key only
    try:
        import numpy as np
        from ..simulation.group_stage import simulate_group_stage
        groups, fixtures, _ = _asof_conditioned(as_of)
        odds = _live_odds(odds_api.WORLD_CUP_SPORT, "avg")
        if odds.empty:
            return {}
        grp_pairs = world_cup.group_pair_keys(groups)
        ko = [(r.home_team, r.away_team) for r in odds.itertuples(index=False)
              if world_cup._pair_key(r.home_team, r.away_team) not in grp_pairs]
        if not ko:
            return {}
        # Completed groups' standings are model-independent, so a bare Elo suffices.
        _, slot_map, _, _ = simulate_group_stage(
            EloGoalsModel(), groups, np.random.default_rng(0),
            fixtures=fixtures, n_best_third=8)
        complete = world_cup.complete_groups(fixtures, groups)
        return world_cup.live_third_assignment(slot_map, complete, ko)
    except Exception:
        return {}


def _assemble_live_ko_frame(ko_df, groups: dict, slot_map: dict) -> pd.DataFrame:
    """Expand a remaining-bracket result into the full live frame: alive teams carry
    their reach/champion probs (earlier stages = 1, they got there), every eliminated
    team is 0, and win_group is the actual group result."""
    from ..data.aliases import normalize_team_name
    t2g = {normalize_team_name(t): str(g).strip().upper()
           for g, teams in groups.items() for t in teams}
    stage_cols = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]
    present = [s for s in stage_cols if f"{s}_probability" in ko_df.columns]
    earlier = stage_cols[:stage_cols.index(present[0])] if present else []
    ko_map = {normalize_team_name(r["team"]): r for _, r in ko_df.iterrows()}
    winners = {normalize_team_name(slot_map[f"1{g}"]) for g in set(t2g.values())
               if slot_map.get(f"1{g}")}
    rows = []
    for t, g in t2g.items():
        row = {"team": t, "group": g}
        r = ko_map.get(t)
        for s in stage_cols:
            row[f"{s}_probability"] = (1.0 if (r is not None and s in earlier)
                                       else float(r[f"{s}_probability"]) if (r is not None and f"{s}_probability" in ko_df.columns)
                                       else 0.0)
        row["champion_probability"] = float(r["champion_probability"]) if r is not None else 0.0
        row["win_group_probability"] = 1.0 if t in winners else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("champion_probability", ascending=False).reset_index(drop=True)


def _remaining_ko_frame(model, groups, fixtures, third_assignment: dict, n_sims: int):
    """If the tournament is in the knockout rounds, simulate ONLY the actual remaining
    bracket (from the live upcoming fixtures) so eliminated teams are absent (0).
    Returns the full live frame, or None if not cleanly in a KO round."""
    try:
        import numpy as np
        from ..simulation.group_stage import simulate_group_stage
        from ..simulation.monte_carlo import monte_carlo_knockout
        odds = _live_odds(odds_api.WORLD_CUP_SPORT, "avg")
        if odds.empty:
            return None
        grp = world_cup.group_pair_keys(groups)
        upcoming = [(r.home_team, r.away_team) for r in odds.itertuples(index=False)
                    if world_cup._pair_key(r.home_team, r.away_team) not in grp]
        if not upcoming:
            return None
        _, slot_map, _, _ = simulate_group_stage(
            EloGoalsModel(), groups, np.random.default_rng(0), fixtures=fixtures, n_best_third=8)
        built = world_cup.remaining_ko_bracket(slot_map, third_assignment, upcoming)
        if built is None:
            return None
        fmt2, slot_map2 = built
        ko = monte_carlo_knockout(model, fmt2, slot_map2, n_simulations=n_sims)
        return _assemble_live_ko_frame(ko, groups, slot_map)
    except Exception:
        return None


@st.cache_data(show_spinner="Simulating World Cup 2026...", ttl=900)
def _live_wc(
    model_label: str, n_sims: int, anchor: bool, fingerprint: str,
    ability_w: float = 0.0, host_adv: float = 0.0, sofa: bool = False,
    third_pins: tuple = (), as_of: str = "",
) -> pd.DataFrame:
    model = _live_model(model_label, fingerprint, ability_w, host_adv, sofa, as_of)
    groups, fixtures, known = _asof_conditioned(as_of)
    if anchor and not as_of:  # current market odds are anachronistic for a past cutoff
        from ..models.market_anchor import MarketAnchoredModel

        try:
            odds = _live_odds(odds_api.WORLD_CUP_SPORT, "avg")
            if not odds.empty:
                model = MarketAnchoredModel(model, weight=0.5, odds=odds)
        except Exception:
            pass  # no odds/key -> pure model
    # Once the knockouts are under way (Now), simulate the ACTUAL remaining bracket
    # from the live fixtures so every eliminated team is 0 -- robust to the feeds
    # only surfacing a rolling window of KO results.
    if not as_of:
        remaining = _remaining_ko_frame(model, groups, fixtures, dict(third_pins), n_sims)
        if remaining is not None:
            return remaining
    from ..simulation.tournament import simulate_tournament

    return simulate_tournament(
        model, groups, fixtures=fixtures, n_simulations=n_sims,
        known_results=known or None,
        force_third_assignment=dict(third_pins) or None,
    )


@st.cache_data(show_spinner="Computing projected knockout meetings...", ttl=900)
def _live_meetings(
    model_label: str, fingerprint: str, ability_w: float, host_adv: float, sofa: bool,
    third_pins: tuple, teams: tuple, as_of: str = "", n_sims: int = 3000,
) -> pd.DataFrame:
    """Pairwise P(meet in the KO stage) among `teams`, using the same model/overlays,
    the actual-fixture R32 pins, and the as-of conditioning (locked results)."""
    from ..simulation import rules
    from ..simulation.monte_carlo import knockout_meeting_probabilities
    model = _live_model(model_label, fingerprint, ability_w, host_adv, sofa, as_of)
    groups, fixtures, known = _asof_conditioned(as_of)
    return knockout_meeting_probabilities(
        model, rules.world_cup_2026_format(), groups, fixtures=fixtures,
        n_simulations=n_sims, teams=list(teams),
        force_third_assignment=dict(third_pins) or None,
        known_results=known or None,
    )


@st.cache_data(show_spinner="Fetching live odds...", ttl=900)
def _live_odds(sport: str, aggregate: str) -> pd.DataFrame:
    return odds_api.fetch_match_odds(sport=sport, aggregate=aggregate)


@st.cache_data(show_spinner="Fetching futures (winner) market...", ttl=1800)
def _live_outrights() -> dict:
    """De-vigged WC-winner (futures) market champion probabilities, or empty."""
    try:
        return odds_api.fetch_outrights()
    except Exception:
        return {}


def _blend_champion_to_market(sims: pd.DataFrame, market: dict, weight: float = 0.5):
    """Blend the sim's champion_probability toward the futures market, renormalised.

    champ_final = (1-w)*sim + w*market for teams the market prices; teams it does
    not price keep their sim value. The column is then renormalised to sum to 1.
    """
    if not market or "champion_probability" not in sims.columns:
        return sims
    out = sims.copy()
    blended = []
    for team, sim_p in zip(out["team"], out["champion_probability"]):
        mp = market.get(team)
        blended.append((1 - weight) * sim_p + weight * mp if mp is not None else sim_p)
    total = sum(blended)
    out["champion_probability"] = [b / total for b in blended] if total > 0 else blended
    return out.sort_values("champion_probability", ascending=False).reset_index(drop=True)


def _tab_world_cup() -> None:
    st.subheader("World Cup 2026 — live")
    st.caption(
        "Real groups (verified vs the actual fixtures); every game played so far is "
        "locked in from three sources — martj42 history, the live Odds-API feed, and "
        "a saved results ledger so nothing is lost once seen; the rest is "
        "Monte-Carlo'd from the model fit on real history through today."
    )
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    model_label = c1.selectbox(
        "Model", list(WC_MODELS), key="wc_model",
        help="Elo is the national-team engine. Dixon-Coles (club model) runs but is "
        "slow and miscalibrated on neutral international data; xG/Ensemble need club "
        "xG that internationals lack, so they're omitted here.",
    )
    n_sims = int(c2.slider("Simulations", 1000, 30000, 8000, step=1000, key="wc_sims"))
    anchor = c3.toggle(
        "Anchor to live market", value=True, key="wc_anchor",
        help="Blend near-term fixtures toward the de-vigged live betting line "
        "(the strongest signal). Future knockout matchups use the pure model.",
    )
    go = c4.button("Run", type="primary")
    ability_on = st.toggle(
        "Squad-ability overlay (experimental, NOT backtested)", value=False,
        key="wc_ability",
        help="Tilt the Elo engine toward squad ABILITY, blended z-score-wise from "
        "EA-FC/FIFA overall ratings + age-adjusted Transfermarkt ability "
        "(auto-fetched), SUPPLEMENTED by any rating-site exports you drop in as "
        "data/ability_overlay_*.csv (SofaScore / FotMob / WhoScored — browser-gated "
        "APIs, so export to CSV; per-team or per-player, any scale). Blended only "
        "where each source has data. Off by default and exploratory (the market "
        "anchor already prices ability); uncovered nations fall back to pure Elo; "
        "Elo model only.",
    )
    sofa_on = False
    if ability_on:
        sofa_on = st.checkbox(
            "↳ also blend SofaScore club-season ratings (experimental)",
            value=False, key="wc_sofa",
            help="Adds SofaScore 2025/26 club-season ratings "
            "(data/sofascore_ratings_2026.csv) as a third source in the ability "
            "blend, LEAGUE-ADJUSTED (a 7.5 in a weak league anchors below a 7.5 in a "
            "strong one). Backtested on past WCs it still slightly DILUTES the FIFA+TM "
            "blend, so it's here for experimentation only — leave off for the best overlay.",
        )
    futures_anchor = st.toggle(
        "Anchor champion odds to the futures market", value=True, key="wc_futures",
        help="Blend the simulated CHAMPION probabilities toward the de-vigged "
        "bookmaker WORLD-CUP-WINNER (outright) market — the single best title "
        "forecast available, and the market is the ceiling no model has beaten "
        "out-of-sample here. Affects the champion column only.",
    )
    host_on = st.toggle(
        "Host advantage for US / Canada / Mexico", value=True, key="wc_host",
        help="The simulation treats every game as neutral, but the hosts play on "
        "home soil. BACKTESTED on past World Cups (1998-2022): a host home bump "
        "cuts host-game log loss on 5 of 7 editions (2014 Brazil and 2022 Qatar "
        "are counterexamples). With three co-hosts across a continent the effect "
        "is diluted vs a single host, so a moderate +40 Elo is applied (vs ~65-100 "
        "for single hosts). Elo model only.",
    )
    host_adv = 40.0 if host_on else 0.0  # co-host-discounted (single hosts: ~65-100)
    if c4.button("🔄 Refresh live data"):
        # Force a fresh martj42 download (bypass the disk cache) so a just-played
        # game shows even if the cached feed predates it, then drop every cached
        # live artifact so the new result flows all the way through.
        try:
            sources.fetch_international_results(cache=False)
        except Exception:
            pass
        for fn in (_live_history, _live_scores, _live_results_store,
                   _live_train, _live_ability, _live_model, _live_wc, _live_odds,
                   _live_outrights, _live_r32_third_pins, _live_meetings,
                   _asof_options):
            try:
                fn.clear()
            except Exception:
                pass
        st.rerun()
    if not go:
        st.info("Pick a model and run. First run downloads + caches real data, then "
                "locks in the group games already played (incl. today's live results). "
                "Use 🔄 Refresh after a match finishes.")
        return
    ability_w = 0.5 if ability_on else 0.0  # modest; >0.5 hurt in the backtest
    sofa = sofa_on and ability_on  # SofaScore is a source within the ability blend
    try:
        fingerprint = _live_results_fingerprint()  # content key: refit on any change
        asof_opts = _asof_options(fingerprint)  # time-machine checkpoints
        asof_labels = [o[0] for o in asof_opts]
        pick = st.selectbox(
            "Probabilities as of", asof_labels, index=0, key="wc_asof",
            help="Rewind the tournament: recompute every probability conditioned ONLY "
            "on games played on/before the chosen checkpoint — the model is refit and "
            "the bracket re-simulated from there. 'Now' uses all results; pick an "
            "earlier day to see what the probabilities were before those games.",
        )
        as_of = asof_opts[asof_labels.index(pick)][1]
        played = int(
            world_cup.build_fixtures(
                _live_history(), world_cup.load_groups(), extra_results=_live_scores()
            ).pipe(lambda d: (d["home_score"] != "").sum())
        )
        host_adv_eff = host_adv if model_label == "Elo" else 0.0
        third_pins = _live_r32_third_pins(fingerprint, as_of)  # actual R32 slots
        sims = _live_wc(model_label, n_sims, anchor, fingerprint, ability_w,
                        host_adv_eff, sofa, tuple(sorted(third_pins.items())), as_of)
    except Exception as exc:
        st.error(f"Live World Cup pipeline failed: {exc}")
        return
    if as_of:
        st.info(f"⏪ Rewound — **{pick}**. Every probability below is conditioned only "
                "on games up to that point (model refit, bracket re-simulated); the "
                "live market anchor is off for past checkpoints.")
    st.caption(
        f"{played}/72 group games played and locked in (incl. live results)"
        + (" · anchored to live market odds" if anchor else " · pure model (no anchor)")
        + (f" · +{host_adv_eff:.0f} Elo host bump (US/CAN/MEX)" if host_adv_eff else "")
        + (f" · {len(third_pins)}/8 R32 best-third slots set from the actual draw"
           if third_pins else "")
    )
    if ability_on:
        _adj, _src = _live_ability(sofa)
        if not _adj:
            st.warning(f"Ability overlay unavailable — using pure model. ({_src})")
        elif model_label != "Elo":
            st.info("The ability overlay only affects the Elo model; pick Elo to use it.")
        else:
            st.caption(
                f"⚗️ Experimental ability overlay ON ({len(_adj)} teams, source: {_src}) "
                "— not backtested; this did not beat plain Elo on past World Cups."
            )
    if futures_anchor:
        market_champ = _live_outrights()
        if market_champ:
            sims = _blend_champion_to_market(sims, market_champ, weight=0.5)
            st.caption(
                f"🎯 Champion probabilities blended 50/50 with the live futures "
                f"(winner) market ({len(market_champ)} teams) — the market is the "
                "ceiling no model has beaten OOS here."
            )
        else:
            st.caption("Futures market unavailable — champion probabilities are pure model.")
    cols = st.columns(3)
    top = sims.iloc[0]
    cols[0].metric("Favourite", str(top["team"]), f"{top['champion_probability']:.1%}")
    cols[1].metric("Teams", f"{len(sims)}")
    cols[2].metric("Model", model_label + (" + market" if anchor else ""))
    show = sims.head(16)
    fig = px.bar(
        show, x="champion_probability", y="team", color="group", orientation="h",
        text="champion_probability",
    ).update_layout(yaxis={"categoryorder": "total ascending"},
                    title="Champion probability (top 16)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sims, use_container_width=True, hide_index=True)
    st.download_button(
        "Download full probabilities (CSV)", sims.to_csv(index=False),
        file_name="world_cup_2026_probabilities.csv", mime="text/csv",
    )

    # Bracket-path interaction: which contenders are projected to MEET, and when.
    if model_label == "Elo" and st.checkbox(
        "Show projected knockout meetings (bracket paths)", value=False, key="wc_meet",
        help="How often the title contenders are projected to face each other in the "
        "knockout stage and in which round. This is the bracket-path effect already "
        "priced into every probability above — two strong teams drawn into the same "
        "quarter collide early, so they can't both go deep.",
    ):
        try:
            top = sims.head(12)["team"].tolist()
            mdf = _live_meetings(
                model_label, fingerprint, ability_w, host_adv_eff, sofa,
                tuple(sorted(third_pins.items())), tuple(top), as_of)
            if mdf.empty:
                st.info("No projected meetings among the top teams.")
            else:
                view = mdf.head(15).copy()
                view["meet_probability"] = (view["meet_probability"] * 100).round(1)
                rounds = {"round_of_32": "R32", "round_of_16": "R16",
                          "quarterfinal": "QF", "semifinal": "SF", "final": "Final"}
                view["likeliest_round"] = view["likeliest_round"].map(lambda s: rounds.get(s, s))
                st.dataframe(
                    view[["team_a", "team_b", "meet_probability", "likeliest_round"]]
                    .rename(columns={"meet_probability": "meet %", "likeliest_round": "likeliest round"}),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Read: e.g. two contenders shown meeting mostly in the R16/QF are on "
                    "a collision path — the deeper-stage probabilities above already "
                    "account for it."
                )
        except Exception as exc:
            st.warning(f"Meeting projection unavailable: {exc}")


def _live_odds_section(model: ScorelineModel, neutral: bool) -> None:
    """Live market vs model edge for upcoming fixtures (The Odds API)."""
    st.markdown("**Live market edge (The Odds API)**")
    if odds_api.load_odds_api_key() is None:
        st.info(
            "No Odds API key found. Set ODDS_API_KEY or create ~/.odds_api_key to "
            "scan live fixtures. Get a free key at the-odds-api.com."
        )
        return
    sport = st.text_input("Odds API sport key", value=odds_api.WORLD_CUP_SPORT)
    if not st.button("Fetch live odds & scan", key="live_odds_btn"):
        return
    try:
        odds = _live_odds(sport, "best")
    except Exception as exc:
        st.error(f"Odds fetch failed: {exc}")
        return
    if odds.empty:
        st.warning(f"No upcoming {sport} games with odds right now.")
        return
    known = model.known_teams() if hasattr(model, "known_teams") else None
    rows = []
    for r in odds.itertuples(index=False):
        teams_known = known is None or (r.home_team in known and r.away_team in known)
        p = model.outcome_probs(r.home_team, r.away_team, {"neutral_site": neutral})
        mkt = implied_probabilities(r.home_odds, r.draw_odds, r.away_odds)
        market = [r.home_odds, r.draw_odds, r.away_odds]
        evs = [float(p[c]) * market[c] - 1.0 for c in range(3)]
        best = int(np.argmax(evs))
        suspect = (
            float(mkt[best]) < 0.08
            or market[best] > 12.0
            or abs(float(p[best]) - float(mkt[best])) > 0.10
            or float(p[best]) > 0.95
            or not teams_known
        )
        rows.append(
            {
                "fixture": f"{r.home_team} vs {r.away_team}",
                "pick": ["Home", "Draw", "Away"][best],
                "model_prob": round(float(p[best]), 3),
                "market_prob": round(float(mkt[best]), 3),
                "best_odds": market[best],
                "edge": round(evs[best], 3),
                "likely_miscalibration": suspect,
                "unmatched_team": not teams_known,
            }
        )
    edge_df = pd.DataFrame(rows).sort_values("edge", ascending=False)
    st.warning(
        "Large edges on longshots (likely_miscalibration=True) are almost always "
        "the model overrating the underdog, NOT value. Calibrate first."
    )
    st.dataframe(edge_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Soccer Predictor", layout="wide")
    st.title("Soccer Predictor")
    st.caption("Unified club + international + tournament soccer prediction.")

    cfg = _sidebar()

    try:
        df = load_data(cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"])
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    if cfg["is_synthetic"]:
        st.error(SAMPLE_BANNER, icon="🚨")

    # Data summary banner.
    completed = schemas.completed_matches(df)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Matches", f"{len(completed):,}")
    s2.metric("Teams", f"{len(teams):,}")
    s3.metric("Has odds", "Yes" if schemas.has_usable_odds(df) else "No")
    s4.metric("Has xG", "Yes" if schemas.has_usable_xg(df) else "No")

    tab_wc, tab_match, tab_tournament, tab_backtest, tab_betting = st.tabs(
        [
            "🏆 World Cup 2026 (live)",
            "Match Predictor",
            "Tournament Simulator",
            "Backtest Lab",
            "Betting Edge Scanner",
        ]
    )
    with tab_wc:
        _tab_world_cup()
    with tab_match:
        _tab_match(df, cfg)
    with tab_tournament:
        _tab_tournament(df, cfg)
    with tab_backtest:
        _tab_backtest(df, cfg)
    with tab_betting:
        _tab_betting(df, cfg)


# Entry point is app_unified.py, which calls main() explicitly on every Streamlit
# rerun. This module must therefore NOT auto-run main() when it is *imported* (that
# would render the app twice on the first run -> StreamlitDuplicateElementId, and
# importing it for tests would have side effects). It only runs main() if executed
# directly as a script (`python streamlit_app.py`), never on import.
if __name__ == "__main__":  # pragma: no cover
    main()
