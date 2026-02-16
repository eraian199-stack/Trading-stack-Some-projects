#!/usr/bin/env python3
"""Streamlit UI for the Bangladesh real-estate probabilistic risk MVP."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from bd_real_estate_risk_mvp import SCRIPT_DIR, load_assumptions, run_simulations


DEFAULT_ASSUMPTIONS_PATH = SCRIPT_DIR / "bd_real_estate_assumptions.json"
DEFAULT_OUTDIR = SCRIPT_DIR / "bd_real_estate_outputs"

PCT_COLUMNS = [
    "irr_mean",
    "irr_p50",
    "irr_p05",
    "irr_p95",
    "downside_prob_irr_lt_0",
    "downside_prob_irr_lt_hurdle",
    "refinance_failure_prob",
    "covenant_breach_prob",
    "cvar_5_irr",
]


def format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PCT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x * 100.0:.2f}%" if pd.notna(x) else "nan")
    if "equity_multiple_mean" in out.columns:
        out["equity_multiple_mean"] = out["equity_multiple_mean"].map(
            lambda x: f"{x:.2f}x" if pd.notna(x) else "nan"
        )
    return out


def build_report_markdown(summaries: list[dict[str, Any]], sims: int, seed: int, hurdle: float) -> str:
    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        return "# Bangladesh Real Estate Risk MVP Report\n\nNo results generated."

    lines: list[str] = []
    lines.append("# Bangladesh Real Estate Risk MVP Report")
    lines.append("")
    lines.append(f"- Simulations per scenario: `{sims}`")
    lines.append(f"- Seed: `{seed}`")
    lines.append(f"- Hurdle IRR: `{hurdle:.2%}`")
    lines.append("")
    lines.append("## Scenario Summary")
    lines.append("")
    lines.append(
        "| Scenario | Mean IRR | P5 IRR | P50 IRR | P95 IRR | P(IRR<0) | "
        "P(IRR<hurdle) | Refi fail | Covenant breach | CVaR(5%) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, s in summary_df.iterrows():
        lines.append(
            "| {scenario} | {irr_mean:.2%} | {irr_p05:.2%} | {irr_p50:.2%} | {irr_p95:.2%} | {p0:.2%} | {ph:.2%} | {rf:.2%} | {cb:.2%} | {cvar:.2%} |".format(
                scenario=s["scenario"],
                irr_mean=s["irr_mean"],
                irr_p05=s["irr_p05"],
                irr_p50=s["irr_p50"],
                irr_p95=s["irr_p95"],
                p0=s["downside_prob_irr_lt_0"],
                ph=s["downside_prob_irr_lt_hurdle"],
                rf=s["refinance_failure_prob"],
                cb=s["covenant_breach_prob"],
                cvar=s["cvar_5_irr"],
            )
        )

    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `Refi fail` is the probability the senior debt cannot be refinanced at maturity.")
    lines.append("- `Covenant breach` tracks any DSCR/LTV breach during hold.")
    lines.append("- `CVaR(5%)` is the mean IRR in the worst 5% tail outcomes.")

    return "\n".join(lines)


def build_irr_density(sim_df: pd.DataFrame, bins: int = 60) -> pd.DataFrame:
    if sim_df.empty:
        return pd.DataFrame(columns=["irr_bin", "density", "scenario"])

    irr_series = sim_df["irr"].dropna()
    if irr_series.empty:
        return pd.DataFrame(columns=["irr_bin", "density", "scenario"])

    lo = float(irr_series.min())
    hi = float(irr_series.max())
    if np.isclose(lo, hi):
        hi = lo + 1e-6

    edges = np.linspace(lo, hi, bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    rows: list[dict[str, Any]] = []
    for scenario, grp in sim_df.groupby("scenario"):
        vals = grp["irr"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        hist, _ = np.histogram(vals, bins=edges, density=True)
        for x, y in zip(centers, hist):
            rows.append({"irr_bin": x, "density": y, "scenario": scenario})

    return pd.DataFrame(rows)


def read_assumptions_from_source(source_mode: str, uploaded_file: Any) -> dict[str, Any]:
    if source_mode == "Default file":
        return load_assumptions(DEFAULT_ASSUMPTIONS_PATH)

    if uploaded_file is None:
        raise ValueError("Upload a JSON assumptions file to continue.")

    payload = uploaded_file.getvalue().decode("utf-8")
    return json.loads(payload)


def main() -> None:
    if get_script_run_ctx(suppress_warning=True) is None:
        print(
            "This app must be launched with Streamlit.\n"
            "Run:\n"
            "/opt/anaconda3/bin/python -m streamlit run "
            "\"/Users/elhamraian/Library/CloudStorage/OneDrive-Personal/Documents/Work/Work/Personal App/bd_real_estate_streamlit.py\""
        )
        return

    st.set_page_config(page_title="Bangladesh Real Estate Risk MVP", layout="wide")
    st.title("Bangladesh Real Estate Risk MVP")
    st.caption(
        "Monte Carlo real-estate risk model with refinancing feasibility, covenant stress, and scenario analysis."
    )

    with st.sidebar:
        st.header("Run Settings")
        source_mode = st.radio("Assumptions source", ["Default file", "Upload JSON"], index=0)
        uploaded_file = None
        if source_mode == "Upload JSON":
            uploaded_file = st.file_uploader("Assumptions JSON", type=["json"])

        sims = int(st.number_input("Simulations per scenario", min_value=100, max_value=50000, value=2000, step=100))
        seed = int(st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1))

    try:
        loaded_assump = read_assumptions_from_source(source_mode, uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    source_text = json.dumps(loaded_assump, indent=2)
    source_fingerprint = json.dumps(loaded_assump, sort_keys=True)
    previous_source = st.session_state.get("source_fingerprint")
    if "assumptions_text" not in st.session_state or previous_source != source_fingerprint:
        st.session_state["assumptions_text"] = source_text
        st.session_state["source_fingerprint"] = source_fingerprint

    if st.sidebar.button("Reset JSON to source"):
        st.session_state["assumptions_text"] = source_text

    st.subheader("Assumptions JSON (editable)")
    st.text_area("Assumptions JSON", key="assumptions_text", height=420, label_visibility="collapsed")

    try:
        assumptions = json.loads(st.session_state["assumptions_text"])
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()

    if "scenarios" not in assumptions or not isinstance(assumptions["scenarios"], dict):
        st.error("Assumptions must include a `scenarios` object.")
        st.stop()

    scenario_names = list(assumptions["scenarios"].keys())
    with st.sidebar:
        selected_scenarios = st.multiselect(
            "Scenarios",
            options=scenario_names,
            default=scenario_names,
        )
        run_clicked = st.button("Run Simulation", type="primary")

    if run_clicked:
        if not selected_scenarios:
            st.error("Select at least one scenario.")
            st.stop()

        with st.spinner("Running Monte Carlo simulation..."):
            sim_rows, summaries = run_simulations(
                assump=assumptions,
                sims=sims,
                seed=seed,
                scenario_names=selected_scenarios,
            )

        st.session_state["sim_df"] = pd.DataFrame(sim_rows)
        st.session_state["summary_df"] = pd.DataFrame(summaries)

    sim_df = st.session_state.get("sim_df")
    summary_df = st.session_state.get("summary_df")

    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        st.info("Click `Run Simulation` in the sidebar to generate results.")
        st.stop()

    st.subheader("Scenario Summary")
    st.dataframe(format_percent_columns(summary_df), use_container_width=True)

    focus_scenario = st.selectbox("Focus scenario", options=summary_df["scenario"].tolist(), index=0)
    focus = summary_df[summary_df["scenario"] == focus_scenario].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean IRR", f"{focus['irr_mean'] * 100.0:.2f}%")
    c2.metric("P(IRR < 0)", f"{focus['downside_prob_irr_lt_0'] * 100.0:.2f}%")
    c3.metric("Refi Failure", f"{focus['refinance_failure_prob'] * 100.0:.2f}%")
    c4.metric("Covenant Breach", f"{focus['covenant_breach_prob'] * 100.0:.2f}%")

    st.subheader("IRR Density by Scenario")
    density_df = build_irr_density(sim_df)
    if density_df.empty:
        st.warning("No IRR data available for charting.")
    else:
        st.line_chart(density_df, x="irr_bin", y="density", color="scenario")

    st.subheader("Downloads")
    summary_csv = summary_df.to_csv(index=False).encode("utf-8")
    sim_csv = sim_df.to_csv(index=False).encode("utf-8")
    report_md = build_report_markdown(
        summaries=summary_df.to_dict(orient="records"),
        sims=sims,
        seed=seed,
        hurdle=float(assumptions.get("project", {}).get("hurdle_irr", 0.12)),
    ).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download scenario_summary.csv",
        data=summary_csv,
        file_name="scenario_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Download simulation_results.csv",
        data=sim_csv,
        file_name="simulation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d3.download_button(
        "Download report.md",
        data=report_md,
        file_name="report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Persist output files to disk"):
        outdir_str = st.text_input("Output folder", value=str(DEFAULT_OUTDIR))
        if st.button("Save files"):
            outdir = Path(outdir_str)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "scenario_summary.csv").write_bytes(summary_csv)
            (outdir / "simulation_results.csv").write_bytes(sim_csv)
            (outdir / "report.md").write_bytes(report_md)
            st.success(f"Saved to {outdir}")


if __name__ == "__main__":
    script_ctx = get_script_run_ctx(suppress_warning=True)
    relaunched = os.environ.get("BD_RE_STREAMLIT_RELAUNCHED") == "1"

    if script_ctx is None and not relaunched:
        # Running via `python file.py`: re-exec into Streamlit so the app opens immediately.
        os.environ["BD_RE_STREAMLIT_RELAUNCHED"] = "1"
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
                "--server.headless",
                "false",
            ],
        )
    else:
        main()
