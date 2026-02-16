# Bangladesh Real Estate Risk MVP Report

- Assumptions file: `/Users/elhamraian/Library/CloudStorage/OneDrive-Personal/Documents/Work/Work/Personal App/bd_real_estate_assumptions.json`
- Simulations per scenario: `100`
- Seed: `1`
- Hurdle IRR: `14.00%`

## Scenario Summary

| Scenario | Mean IRR | P5 IRR | P50 IRR | P95 IRR | P(IRR<0) | P(IRR<hurdle) | Refi fail | Covenant breach | CVaR(5%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 15.00% | -24.05% | 18.12% | 28.45% | 9.09% | 30.30% | 10.00% | 72.00% | -33.17% |
| stress | -14.46% | -60.33% | -2.57% | 14.38% | 50.57% | 91.95% | 52.00% | 94.00% | -72.29% |
| severe | -40.07% | -72.00% | -39.24% | 0.03% | 94.92% | 100.00% | 96.00% | 98.00% | -84.60% |

## Interpretation Guide

- `Refi fail` is the probability the senior debt cannot be refinanced at maturity under scenario conditions.
- `Covenant breach` tracks any DSCR/LTV breach during the hold period.
- `CVaR(5%)` is the average IRR within the worst 5% of outcomes (tail risk).