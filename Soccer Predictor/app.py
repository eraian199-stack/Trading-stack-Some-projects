from __future__ import annotations

from datetime import date as date_type
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from world_cup_predictor.data import load_fixtures, load_groups, load_results
from world_cup_predictor.model import MatchPredictor, evaluate_holdout
from world_cup_predictor.tournament import simulate_tournament


SAMPLE_RESULTS = ROOT / "data" / "sample_results.csv"
SAMPLE_GROUPS = ROOT / "data" / "sample_groups.csv"
DEFAULT_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("World Cup 2026 Predictor")


@st.cache_data(show_spinner=False)
def _load_results(source_kind: str, source_value: str) -> pd.DataFrame:
    if source_kind == "Public URL":
        return load_results(source_value)
    return load_results(Path(source_value))


@st.cache_resource(show_spinner=False)
def _fit_model(results_csv_key: str, source_kind: str, source_value: str) -> MatchPredictor:
    del results_csv_key
    results = _load_results(source_kind, source_value)
    return MatchPredictor().fit(results)


with st.sidebar:
    st.header("Data")
    source_kind = st.radio("Historical results", ["Local file", "Public URL"], index=0)
    if source_kind == "Public URL":
        results_source = st.text_input("Results URL", DEFAULT_RESULTS_URL)
    else:
        results_source = st.text_input("Results CSV", str(SAMPLE_RESULTS))
    groups_source = st.text_input("Groups CSV", str(SAMPLE_GROUPS))
    fixtures_source = st.text_input("Fixtures CSV", "")
    sims = st.slider("Simulations", min_value=250, max_value=20000, value=2500, step=250)
    seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=7, step=1)

try:
    results = _load_results(source_kind, results_source)
    predictor = _fit_model(str(hash(results_source)), source_kind, results_source)
except Exception as exc:  # pragma: no cover - Streamlit display path
    st.error(f"Could not load or train model: {exc}")
    st.stop()

summary = predictor.training_summary or {}
metric_cols = st.columns(4)
metric_cols[0].metric("Matches", f"{summary.get('matches', 0):,}")
metric_cols[1].metric("Teams", f"{summary.get('teams', 0):,}")
metric_cols[2].metric("First Match", summary.get("first_match", "n/a"))
metric_cols[3].metric("Last Match", summary.get("last_match", "n/a"))

if str(results_source) == str(SAMPLE_RESULTS):
    st.warning("Sample results are only for smoke testing. Use a full international-results file before trusting probabilities.")

tab_match, tab_tournament, tab_backtest = st.tabs(["Match", "Tournament", "Backtest"])

with tab_match:
    teams = sorted(set(results["home_team"]) | set(results["away_team"]))
    left, right, opts = st.columns([1, 1, 1])
    home = left.selectbox("Home/listed team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away_default = teams.index("France") if "France" in teams else min(1, len(teams) - 1)
    away = right.selectbox("Away team", teams, index=away_default)
    neutral = opts.toggle("Neutral venue", value=True)
    tournament = opts.selectbox(
        "Tournament",
        ["FIFA World Cup", "FIFA World Cup qualification", "Friendly", "Continental championship"],
    )
    date = opts.date_input("Match date", value=date_type(2026, 6, 15))
    if st.button("Predict Match", type="primary"):
        pred = predictor.predict_match(home, away, date=date, tournament=tournament, neutral=neutral)
        probs = pd.DataFrame(
            [
                {"Outcome": "Home/listed win", "Probability": pred.home_win},
                {"Outcome": "Draw", "Probability": pred.draw},
                {"Outcome": "Away win", "Probability": pred.away_win},
            ]
        )
        st.plotly_chart(
            px.bar(probs, x="Outcome", y="Probability", range_y=[0, 1], text_auto=".1%"),
            width="stretch",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected goals", f"{pred.expected_home_goals:.2f} - {pred.expected_away_goals:.2f}")
        c2.metric("Most likely score", pred.most_likely_score)
        c3.metric("No-draw edge", f"{pred.home_win - pred.away_win:+.1%}")
        st.dataframe(
            pd.DataFrame(
                [{"score": score, "probability": prob} for score, prob in pred.scoreline_probabilities.items()]
            ),
            width="stretch",
            hide_index=True,
        )

with tab_tournament:
    if not groups_source:
        st.info("Provide a groups CSV with columns: group, team.")
    else:
        try:
            groups = load_groups(groups_source)
            fixtures = load_fixtures(fixtures_source) if fixtures_source else None
        except Exception as exc:  # pragma: no cover - Streamlit display path
            st.error(f"Could not load tournament files: {exc}")
            st.stop()
        if st.button("Run Tournament Simulation", type="primary"):
            sims_frame = simulate_tournament(
                predictor,
                groups,
                fixtures=fixtures,
                n_simulations=int(sims),
                seed=int(seed),
            )
            st.dataframe(sims_frame, width="stretch", hide_index=True)
            top = sims_frame.head(16).copy()
            st.plotly_chart(
                px.bar(
                    top,
                    x="champion_probability",
                    y="team",
                    color="group",
                    orientation="h",
                    text="champion_probability",
                ).update_layout(yaxis={"categoryorder": "total ascending"}),
                width="stretch",
            )

with tab_backtest:
    cutoff = st.date_input("Holdout cutoff")
    if st.button("Run Holdout"):
        try:
            metrics = evaluate_holdout(results, pd.Timestamp(cutoff))
            st.dataframe(
                pd.DataFrame([metrics]),
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:  # pragma: no cover - Streamlit display path
            st.error(f"Backtest failed: {exc}")
