#!/usr/bin/env python3
"""Bangladesh real-estate probabilistic risk model (MVP).

This script runs Monte Carlo simulations for a development/income property
with layered capital (senior + mezzanine debt), refinancing checks, and
macro stress scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class DebtTranche:
    name: str
    principal: float
    spread: float
    amort_years: int | None
    maturity_year: int
    refinancable: bool


@dataclass
class SimulationResult:
    scenario: str
    irr: float
    equity_multiple: float
    refinance_failed: bool
    covenant_breach: bool
    min_dscr: float
    max_ltv: float
    exit_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bangladesh real-estate Monte Carlo risk simulation with "
            "refinancing feasibility and stress scenarios."
        )
    )
    parser.add_argument(
        "--assumptions",
        type=Path,
        default=SCRIPT_DIR / "bd_real_estate_assumptions.json",
        help="Path to assumptions JSON.",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=2000,
        help="Simulations per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "bd_real_estate_outputs",
        help="Output directory for CSV/Markdown results.",
    )
    return parser.parse_args()


def load_assumptions(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.is_absolute():
        candidate = SCRIPT_DIR / path
        if candidate.exists():
            path = candidate
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annuity_payment(principal: float, rate: float, years: int) -> float:
    if years <= 0:
        return principal
    if rate <= 1e-9:
        return principal / years
    denom = 1.0 - (1.0 + rate) ** (-years)
    return principal * rate / denom


def irr_bisection(cashflows: list[float], low: float = -0.95, high: float = 2.5) -> float:
    def npv(rate: float) -> float:
        return sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows))

    f_low = npv(low)
    f_high = npv(high)

    if math.isnan(f_low) or math.isnan(f_high):
        return float("nan")

    if f_low == 0:
        return low
    if f_high == 0:
        return high

    if f_low * f_high > 0:
        for upper in (5.0, 10.0, 25.0):
            f_upper = npv(upper)
            if f_low * f_upper <= 0:
                high = upper
                f_high = f_upper
                break
        else:
            return float("nan")

    for _ in range(120):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return (low + high) / 2.0


def initialize_capital_stack(assump: dict[str, Any]) -> tuple[list[DebtTranche], float]:
    project = assump["project"]
    debt_cfg = assump["debt"]
    total_cost = float(project["total_project_cost"])  # includes dev + acquisition

    senior_principal = total_cost * float(debt_cfg["senior"]["initial_ltv"])
    mezz_principal = total_cost * float(debt_cfg["mezz"]["initial_ltv"])
    debt_total = senior_principal + mezz_principal
    equity = total_cost - debt_total

    tranches = [
        DebtTranche(
            name="senior",
            principal=senior_principal,
            spread=float(debt_cfg["senior"]["spread_bps"]) / 10000.0,
            amort_years=int(debt_cfg["senior"].get("amort_years", 0)) or None,
            maturity_year=int(debt_cfg["senior"]["maturity_year"]),
            refinancable=True,
        ),
        DebtTranche(
            name="mezz",
            principal=mezz_principal,
            spread=float(debt_cfg["mezz"]["spread_bps"]) / 10000.0,
            amort_years=None,
            maturity_year=int(debt_cfg["mezz"]["maturity_year"]),
            refinancable=False,
        ),
    ]

    return tranches, equity


def clipped_normal(rng: np.random.Generator, mean: float, std: float, low: float, high: float) -> float:
    sample = rng.normal(mean, std)
    return float(min(high, max(low, sample)))


def simulate_one_path(
    rng: np.random.Generator,
    assump: dict[str, Any],
    scenario_name: str,
    scenario: dict[str, Any],
) -> SimulationResult:
    project = assump["project"]
    market = assump["market"]
    macro = assump["macro"]
    covenants = assump["covenants"]
    refi_cfg = assump["refinancing"]

    hold_years = int(project["hold_years"])
    annual_capex = float(project["annual_capex"])
    exit_txn_cost = float(project["exit_transaction_cost_pct"])
    distress_haircut = float(project["distress_sale_haircut_pct"])

    tranches, initial_equity = initialize_capital_stack(assump)
    outstanding = {t.name: t.principal for t in tranches}
    tranche_by_name = {t.name: t for t in tranches}

    noi = float(project["starting_noi"])
    short_rate = float(macro["base_short_rate"]) + float(scenario.get("rate_shift", 0.0))
    inflation = float(macro["base_inflation"]) + float(scenario.get("inflation_shift", 0.0))

    rent_shift = float(scenario.get("rent_growth_shift", 0.0))
    cap_shift = float(scenario.get("cap_rate_shift", 0.0))
    liq_haircut = float(scenario.get("liquidity_haircut", 0.0))
    shock_scale = float(scenario.get("shock_scale", 1.0))

    equity_cashflows: list[float] = [-initial_equity]
    refinance_failed = False
    covenant_breach = False
    min_dscr = float("inf")
    max_ltv = 0.0

    terminal_value = float(project["starting_noi"]) / (float(market["cap_rate_base"]) + cap_shift)

    for year in range(1, hold_years + 1):
        short_rate = max(
            0.0,
            short_rate
            + float(macro["short_rate_kappa"]) * (float(macro["short_rate_theta"]) - short_rate)
            + clipped_normal(
                rng,
                0.0,
                float(macro["short_rate_sigma"]) * shock_scale,
                -0.15,
                0.15,
            ),
        )

        inflation = max(
            -0.03,
            float(macro["inflation_mean_reversion"]) * inflation
            + (1.0 - float(macro["inflation_mean_reversion"]))
            * (float(macro["base_inflation"]) + float(scenario.get("inflation_shift", 0.0)))
            + clipped_normal(
                rng,
                0.0,
                float(macro["inflation_sigma"]) * shock_scale,
                -0.08,
                0.08,
            ),
        )

        demand_shock = clipped_normal(
            rng,
            float(market["demand_mean"]),
            float(market["demand_sigma"]) * shock_scale,
            -0.4,
            0.4,
        )

        rent_growth = (
            float(market["rent_growth_base"])
            + rent_shift
            + float(market["rent_inflation_beta"]) * inflation
            + float(market["rent_demand_beta"]) * demand_shock
            + clipped_normal(
                rng,
                0.0,
                float(market["rent_growth_sigma"]) * shock_scale,
                -0.2,
                0.2,
            )
        )

        opex_drag = float(market["opex_inflation_drag"]) * inflation
        noi = max(1.0, noi * (1.0 + rent_growth - opex_drag))

        cap_rate = max(
            0.03,
            float(market["cap_rate_base"])
            + cap_shift
            + float(market["cap_rate_rate_beta"]) * (short_rate - float(macro["base_short_rate"]))
            + clipped_normal(
                rng,
                0.0,
                float(market["cap_rate_sigma"]) * shock_scale,
                -0.06,
                0.06,
            ),
        )

        terminal_value = noi / cap_rate

        debt_service = 0.0
        covenant_service = 0.0
        for t in tranches:
            if outstanding[t.name] <= 1e-6:
                continue
            contract_rate = short_rate + t.spread
            contract_rate = max(0.01, contract_rate)

            if t.name == "mezz":
                interest = outstanding[t.name] * contract_rate
                principal_due = outstanding[t.name] if year == t.maturity_year else 0.0
            elif t.amort_years:
                years_remaining = max(1, t.amort_years - (year - 1))
                payment = annuity_payment(outstanding[t.name], contract_rate, years_remaining)
                interest = outstanding[t.name] * contract_rate
                principal_due = max(0.0, min(outstanding[t.name], payment - interest))
            else:
                interest = outstanding[t.name] * contract_rate
                principal_due = outstanding[t.name] if year == t.maturity_year else 0.0

            is_bullet_principal = (t.amort_years is None) and (principal_due > 0.0)
            settle_from_exit = is_bullet_principal and (year == hold_years)
            principal_paid = 0.0 if settle_from_exit else principal_due

            outstanding[t.name] = max(0.0, outstanding[t.name] - principal_paid)
            debt_service += interest + principal_paid
            covenant_service += interest + (principal_paid if t.amort_years else 0.0)

        dscr = noi / covenant_service if covenant_service > 1e-9 else 99.0
        total_debt = sum(outstanding.values())
        ltv = total_debt / terminal_value if terminal_value > 1e-9 else 99.0

        min_dscr = min(min_dscr, dscr)
        max_ltv = max(max_ltv, ltv)

        if dscr < float(covenants["min_dscr"]) or ltv > float(covenants["max_ltv"]):
            covenant_breach = True

        equity_cashflow = noi - debt_service - annual_capex

        senior_maturity = int(assump["debt"]["senior"]["maturity_year"])
        if year == senior_maturity and outstanding["senior"] > 1e-6:
            liquidity_factor = max(0.0, 1.0 - liq_haircut)
            refi_spread = float(refi_cfg["refi_spread_bps"]) / 10000.0
            refi_rate = max(0.01, short_rate + refi_spread)

            max_loan_ltv = terminal_value * float(refi_cfg["max_ltv"]) * liquidity_factor
            max_loan_dscr = noi / (float(refi_cfg["min_dscr"]) * refi_rate)
            max_refi_proceeds = max(0.0, min(max_loan_ltv, max_loan_dscr))

            old_senior = outstanding["senior"]
            required = old_senior * float(refi_cfg["required_coverage_ratio"])

            if max_refi_proceeds >= required:
                # Refinance rolls old senior debt into a new facility.
                # If constraints force smaller proceeds, equity covers the gap.
                new_senior = min(old_senior, max_refi_proceeds)
                equity_paydown = max(0.0, old_senior - new_senior)
                equity_cashflow -= equity_paydown
                outstanding["senior"] = new_senior
                tranche_by_name["senior"].spread = refi_spread
            else:
                refinance_failed = True
                distressed_value = terminal_value * (1.0 - distress_haircut)
                sale_cost = distressed_value * exit_txn_cost
                net_sale = distressed_value - sale_cost - sum(outstanding.values())
                equity_cashflow += net_sale
                equity_cashflows.append(equity_cashflow)
                break

        if year == hold_years and not refinance_failed:
            sale_cost = terminal_value * exit_txn_cost
            net_sale = terminal_value - sale_cost - sum(outstanding.values())
            equity_cashflow += net_sale

        equity_cashflows.append(equity_cashflow)

    irr = irr_bisection(equity_cashflows)
    inflows = sum(cf for cf in equity_cashflows[1:] if cf > 0)
    outflows = initial_equity + sum(-cf for cf in equity_cashflows[1:] if cf < 0)
    equity_multiple = inflows / max(1e-6, outflows)

    return SimulationResult(
        scenario=scenario_name,
        irr=irr,
        equity_multiple=equity_multiple,
        refinance_failed=refinance_failed,
        covenant_breach=covenant_breach,
        min_dscr=min_dscr,
        max_ltv=max_ltv,
        exit_value=terminal_value,
    )


def summarize_scenario(results: list[SimulationResult], hurdle: float) -> dict[str, float | str]:
    irr_values = np.array([r.irr for r in results], dtype=float)
    irr_values = irr_values[~np.isnan(irr_values)]
    em_values = np.array([r.equity_multiple for r in results], dtype=float)

    refi_fail = np.mean([1.0 if r.refinance_failed else 0.0 for r in results])
    cov_breach = np.mean([1.0 if r.covenant_breach else 0.0 for r in results])

    downside_zero = float(np.mean(irr_values < 0.0)) if irr_values.size else float("nan")
    downside_hurdle = float(np.mean(irr_values < hurdle)) if irr_values.size else float("nan")

    cvar_cutoff = np.quantile(irr_values, 0.05) if irr_values.size else float("nan")
    cvar_values = irr_values[irr_values <= cvar_cutoff] if irr_values.size else np.array([])
    cvar_5 = float(np.mean(cvar_values)) if cvar_values.size else float("nan")

    return {
        "scenario": results[0].scenario if results else "unknown",
        "sims": len(results),
        "irr_mean": float(np.mean(irr_values)) if irr_values.size else float("nan"),
        "irr_p50": float(np.quantile(irr_values, 0.50)) if irr_values.size else float("nan"),
        "irr_p05": float(np.quantile(irr_values, 0.05)) if irr_values.size else float("nan"),
        "irr_p95": float(np.quantile(irr_values, 0.95)) if irr_values.size else float("nan"),
        "equity_multiple_mean": float(np.mean(em_values)),
        "downside_prob_irr_lt_0": downside_zero,
        "downside_prob_irr_lt_hurdle": downside_hurdle,
        "refinance_failure_prob": float(refi_fail),
        "covenant_breach_prob": float(cov_breach),
        "cvar_5_irr": cvar_5,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_pct(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x * 100.0:.2f}%"


def write_report(
    path: Path,
    assumptions_path: Path,
    summaries: list[dict[str, Any]],
    sims: int,
    seed: int,
    hurdle: float,
) -> None:
    lines: list[str] = []
    lines.append("# Bangladesh Real Estate Risk MVP Report")
    lines.append("")
    lines.append(f"- Assumptions file: `{assumptions_path}`")
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

    for s in summaries:
        lines.append(
            "| {scenario} | {irr_mean} | {irr_p05} | {irr_p50} | {irr_p95} | {p0} | {ph} | {rf} | {cb} | {cvar} |".format(
                scenario=s["scenario"],
                irr_mean=format_pct(float(s["irr_mean"])),
                irr_p05=format_pct(float(s["irr_p05"])),
                irr_p50=format_pct(float(s["irr_p50"])),
                irr_p95=format_pct(float(s["irr_p95"])),
                p0=format_pct(float(s["downside_prob_irr_lt_0"])),
                ph=format_pct(float(s["downside_prob_irr_lt_hurdle"])),
                rf=format_pct(float(s["refinance_failure_prob"])),
                cb=format_pct(float(s["covenant_breach_prob"])),
                cvar=format_pct(float(s["cvar_5_irr"])),
            )
        )

    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `Refi fail` is the probability the senior debt cannot be refinanced at maturity under scenario conditions.")
    lines.append("- `Covenant breach` tracks any DSCR/LTV breach during the hold period.")
    lines.append("- `CVaR(5%)` is the average IRR within the worst 5% of outcomes (tail risk).")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_simulations(
    assump: dict[str, Any],
    sims: int,
    seed: int,
    scenario_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hurdle = float(assump["project"].get("hurdle_irr", 0.12))
    rng = np.random.default_rng(seed)
    sim_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    available = assump["scenarios"]
    names = scenario_names or list(available.keys())
    for scenario_name in names:
        if scenario_name not in available:
            continue
        scenario = available[scenario_name]
        scenario_results: list[SimulationResult] = []
        for _ in range(sims):
            scenario_results.append(simulate_one_path(rng, assump, scenario_name, scenario))
        summaries.append(summarize_scenario(scenario_results, hurdle))

        for r in scenario_results:
            sim_rows.append(
                {
                    "scenario": r.scenario,
                    "irr": r.irr,
                    "equity_multiple": r.equity_multiple,
                    "refinance_failed": int(r.refinance_failed),
                    "covenant_breach": int(r.covenant_breach),
                    "min_dscr": r.min_dscr,
                    "max_ltv": r.max_ltv,
                    "exit_value": r.exit_value,
                }
            )

    return sim_rows, summaries


def main() -> None:
    args = parse_args()
    assump = load_assumptions(args.assumptions)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sim_rows, summaries = run_simulations(assump, args.sims, args.seed)
    hurdle = float(assump["project"].get("hurdle_irr", 0.12))

    write_csv(outdir / "simulation_results.csv", sim_rows)
    write_csv(outdir / "scenario_summary.csv", summaries)
    write_report(
        outdir / "report.md",
        args.assumptions,
        summaries,
        args.sims,
        args.seed,
        hurdle,
    )

    print("Run complete")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Scenario summary: {(outdir / 'scenario_summary.csv').resolve()}")
    print(f"Report: {(outdir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
