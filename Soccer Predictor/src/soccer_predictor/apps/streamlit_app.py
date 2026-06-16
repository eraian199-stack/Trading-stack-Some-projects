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

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..data import loaders, odds_api, schemas, sources, synthetic, world_cup
from ..evaluation import betting, metrics, reports, walk_forward
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


def _data_key(source_kind: str, source_value: str, synthetic_kind: str) -> str:
    return f"{source_kind}|{source_value}|{synthetic_kind}"


# --------------------------------------------------------------------------- #
# Sidebar (data source + model selector)
# --------------------------------------------------------------------------- #
def _sidebar() -> dict:
    st.sidebar.header("Data")
    source_kind = st.sidebar.radio(
        "Source", ["Synthetic", "Local file/dir"], index=0,
        help="Synthetic = clearly-labelled demo data. Local = a canonical / "
        "football-data / martj42 international CSV or directory.",
    )
    synthetic_kind = "League"
    source_value = ""
    if source_kind == "Synthetic":
        synthetic_kind = st.sidebar.radio(
            "Synthetic kind", ["League", "International"], index=0
        )
    else:
        source_value = st.sidebar.text_input(
            "CSV file or directory", value="data/templates/club_league_template.csv"
        )

    st.sidebar.header("Model")
    model_label = st.sidebar.selectbox(
        "Model", list(MODEL_FACTORIES), index=1,
        help="Elo (robust default), Dixon-Coles (MLE goals), xG-Poisson "
        "(needs xG columns), or the 4-model Ensemble.",
    )

    st.sidebar.header("Simulation")
    sims = st.sidebar.slider("Simulations", 100, 20000, 2000, step=100)
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
def _tab_match(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Match Predictor")
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    if len(teams) < 2:
        st.info("Need at least two teams in the data.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    home = c1.selectbox("Home (listed) team", teams, index=0)
    away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    neutral = c3.toggle("Neutral venue", value=False)
    ou_line = c3.number_input("Over/under line", value=2.5, step=0.5)

    model = fit_model(
        cfg["model_label"],
        _data_key(cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"]),
        cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"],
    )
    if not isinstance(model, ScorelineModel):
        st.error("Selected model cannot produce a score matrix.")
        return

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


def _tab_tournament(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Tournament Simulator")
    from ..simulation import rules
    from ..simulation.tournament import simulate_tournament

    fmt = rules.world_cup_2026_format()
    st.caption(f"Format: {fmt.name}.  Assumptions: {fmt.notes}")

    groups_path = st.text_input(
        "Groups CSV (group,team)",
        value="data/templates/world_cup_2026_groups_template.csv",
    )
    fixtures_path = st.text_input("Fixtures CSV (optional)", value="")

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


def _tab_backtest(df: pd.DataFrame, cfg: dict) -> None:
    st.subheader("Backtest Lab")
    st.caption(
        "Expanding-window walk-forward only -- random k-fold leaks the future "
        "and produces fantasy accuracy."
    )

    c1, c2 = st.columns(2)
    folds = int(c1.slider("Folds", 2, 10, 5))
    min_train = c2.slider("First training fraction", 0.3, 0.8, 0.5, step=0.05)

    if not st.button("Run walk-forward backtest", type="primary"):
        st.info("Press the button to run the time-aware backtest.")
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
        "A POSITIVE ROI is a HYPOTHESIS, not proof of edge. It can be "
        "overfitting, survivorship in the odds, or beating opening rather than "
        "closing prices. Check the model's calibration (Backtest Lab) and "
        "closing-line value before trusting any edge."
    )

    if not schemas.has_usable_odds(df):
        st.info(
            "This data has no usable 1X2 odds, so there is nothing to scan "
            "against. Load a frame with home_odds / draw_odds / away_odds."
        )
        return

    c1, c2 = st.columns(2)
    edge = c1.slider("Edge threshold", 0.0, 0.30, 0.05, step=0.01)
    folds = int(c2.slider("Walk-forward folds", 2, 10, 5))

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

    # Live upcoming-fixture scan against The Odds API.
    st.divider()
    model = fit_model(
        cfg["model_label"],
        _data_key(cfg["source_kind"], cfg["source_value"], cfg["synthetic_kind"]),
        cfg["source_kind"],
        cfg["source_value"],
        cfg["synthetic_kind"],
    )
    if isinstance(model, ScorelineModel):
        _live_odds_section(model, neutral=True)


# --------------------------------------------------------------------------- #
# Live data (cached) -- World Cup 2026 + The Odds API
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Fetching real international results...", ttl=3600)
def _live_history() -> pd.DataFrame:
    return sources.fetch_international_results()


@st.cache_resource(show_spinner="Fitting model on real history...")
def _live_model(model_label: str, n_matches: int) -> object:
    del n_matches  # cache key only; forces a refit when the history grows
    factory = MODEL_FACTORIES[model_label]
    return factory().fit(schemas.completed_matches(_live_history()).copy())


@st.cache_data(show_spinner="Simulating World Cup 2026...", ttl=1800)
def _live_wc(model_label: str, n_sims: int, n_matches: int, anchor: bool) -> pd.DataFrame:
    del n_matches
    model = _live_model(model_label, _live_history_len())
    groups = world_cup.load_groups()
    world_cup.build_fixtures(_live_history(), groups)
    fixtures = loaders.load_fixtures(str(world_cup.FIXTURES_CSV))
    if anchor:
        from ..models.market_anchor import MarketAnchoredModel

        try:
            odds = _live_odds(odds_api.WORLD_CUP_SPORT, "avg")
            if not odds.empty:
                model = MarketAnchoredModel(model, weight=0.5, odds=odds)
        except Exception:
            pass  # no odds/key -> pure model
    from ..simulation.tournament import simulate_tournament

    return simulate_tournament(model, groups, fixtures=fixtures, n_simulations=n_sims)


def _live_history_len() -> int:
    return len(_live_history())


@st.cache_data(show_spinner="Fetching live odds...", ttl=900)
def _live_odds(sport: str, aggregate: str) -> pd.DataFrame:
    return odds_api.fetch_match_odds(sport=sport, aggregate=aggregate)


def _tab_world_cup() -> None:
    st.subheader("World Cup 2026 — live")
    st.caption(
        "Real groups (verified vs the actual fixtures), real results locked in as "
        "they are played, and the rest Monte-Carlo'd. Model is fit on real "
        "international history through today."
    )
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    model_label = c1.selectbox("Model", list(MODEL_FACTORIES), key="wc_model")
    n_sims = int(c2.slider("Simulations", 1000, 30000, 8000, step=1000, key="wc_sims"))
    anchor = c3.toggle(
        "Anchor to live market", value=True, key="wc_anchor",
        help="Blend near-term fixtures toward the de-vigged live betting line "
        "(the strongest signal). Future knockout matchups use the pure model.",
    )
    go = c4.button("Run", type="primary")
    if not go:
        st.info("Pick a model and run. First run downloads + caches real data, "
                "then locks in the group games already played.")
        return
    try:
        n = _live_history_len()
        played = int(
            world_cup.build_fixtures(_live_history(), world_cup.load_groups())
            .pipe(lambda d: (d["home_score"] != "").sum())
        )
        sims = _live_wc(model_label, n_sims, n, anchor)
    except Exception as exc:
        st.error(f"Live World Cup pipeline failed: {exc}")
        return
    st.caption(
        f"{played}/72 group games already played and locked in"
        + (" · anchored to live market odds" if anchor else " · pure model (no anchor)")
    )
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


def _running_under_streamlit() -> bool:
    """True only inside an actual ``streamlit run`` session.

    Lets a plain ``import`` of this module (e.g. in tests / smoke checks) succeed
    without trying to render a page outside a Streamlit runtime.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


# ``streamlit run <file>`` executes this script (as __main__ or as a module);
# either way we render only when a real Streamlit runtime is present, so a bare
# import for testing stays side-effect-free.
if __name__ == "__main__" or _running_under_streamlit():
    main()
