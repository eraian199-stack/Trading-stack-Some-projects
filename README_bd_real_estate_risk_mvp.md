# Bangladesh Real-Estate Risk MVP

This MVP extends deterministic real-estate underwriting with probabilistic risk modeling:
- Monte Carlo paths for rent growth, cap rates, rates, inflation, and demand
- Layered debt (senior + mezz)
- Refinancing feasibility check at senior maturity (LTV + DSCR constrained)
- Covenant breach tracking (DSCR/LTV)
- Scenario presets for Bangladesh-style macro stress (`base`, `stress`, `severe`)

## Files

- `bd_real_estate_risk_mvp.py`: simulation engine + reporting
- `bd_real_estate_streamlit.py`: interactive Streamlit UI
- `bd_real_estate_assumptions.json`: editable base assumptions

## Run

```bash
python3 bd_real_estate_risk_mvp.py \
  --assumptions bd_real_estate_assumptions.json \
  --sims 3000 \
  --seed 42 \
  --outdir bd_real_estate_outputs
```

## Outputs

- `simulation_results.csv`: one row per simulated path
- `scenario_summary.csv`: key risk metrics by scenario
- `report.md`: human-readable summary table

## Streamlit UI

```bash
/opt/anaconda3/bin/python bd_real_estate_streamlit.py
```

Direct `python` execution now auto-launches Streamlit and opens the app in your browser.

Manual alternative:

```bash
/opt/anaconda3/bin/python -m streamlit run bd_real_estate_streamlit.py
```

From the UI you can:
- Edit assumptions JSON directly
- Select scenarios, simulation count, and seed
- Run simulations and inspect summary metrics/charts
- Download CSV/Markdown outputs

## Metrics produced

- IRR distribution (mean, p5, p50, p95)
- Downside probability (`P(IRR < 0)` and `P(IRR < hurdle)`)
- Refinance failure probability
- Covenant breach probability
- Tail risk (`CVaR(5%)`)

## Customize quickly

Edit `bd_real_estate_assumptions.json`:
- Project economics: cost, NOI, hold period, capex
- Capital stack: initial LTVs, spreads, maturity/amortization
- Refi constraints: `max_ltv`, `min_dscr`, `required_coverage_ratio`
- Scenario shocks for Bangladesh-specific base/stress/severe narratives
