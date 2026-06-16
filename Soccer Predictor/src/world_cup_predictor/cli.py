from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_fixtures, load_groups, load_results
from .model import MatchPredictor, evaluate_holdout
from .tournament import simulate_tournament


DEFAULT_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def _cmd_backtest(args: argparse.Namespace) -> None:
    results = load_results(args.results)
    metrics = evaluate_holdout(results, args.cutoff_date)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def _cmd_predict_match(args: argparse.Namespace) -> None:
    results = load_results(args.results)
    predictor = MatchPredictor().fit(results)
    pred = predictor.predict_match(
        args.home,
        args.away,
        date=args.date,
        tournament=args.tournament,
        neutral=not args.home_field,
    )
    print(pd.Series(pred.as_dict()).to_string())
    print("\nTop scorelines:")
    for score, prob in pred.scoreline_probabilities.items():
        print(f"{score}: {prob:.3%}")


def _cmd_simulate(args: argparse.Namespace) -> None:
    results = load_results(args.results)
    groups = load_groups(args.groups)
    fixtures = load_fixtures(args.fixtures) if args.fixtures else None
    predictor = MatchPredictor().fit(results)
    sims = simulate_tournament(
        predictor,
        groups,
        fixtures=fixtures,
        n_simulations=args.sims,
        seed=args.seed,
    )
    columns = [
        "team",
        "group",
        "round_of_32_probability",
        "round_of_16_probability",
        "quarterfinal_probability",
        "semifinal_probability",
        "final_probability",
        "champion_probability",
    ]
    print(sims[columns].head(args.top).to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sims.to_csv(out, index=False)
        print(f"\nWrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="world-cup-predictor",
        description="Train soccer match models and simulate World Cup 2026.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Evaluate a date-based holdout.")
    backtest.add_argument("--results", default=DEFAULT_RESULTS_URL)
    backtest.add_argument("--cutoff-date", required=True)
    backtest.set_defaults(func=_cmd_backtest)

    match = subparsers.add_parser("predict-match", help="Predict one match.")
    match.add_argument("--results", default=DEFAULT_RESULTS_URL)
    match.add_argument("--home", required=True)
    match.add_argument("--away", required=True)
    match.add_argument("--date")
    match.add_argument("--tournament", default="FIFA World Cup")
    match.add_argument("--home-field", action="store_true")
    match.set_defaults(func=_cmd_predict_match)

    simulate = subparsers.add_parser("simulate", help="Run World Cup Monte Carlo sims.")
    simulate.add_argument("--results", default=DEFAULT_RESULTS_URL)
    simulate.add_argument("--groups", required=True)
    simulate.add_argument("--fixtures")
    simulate.add_argument("--sims", type=int, default=5000)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--top", type=int, default=20)
    simulate.add_argument("--out")
    simulate.set_defaults(func=_cmd_simulate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
