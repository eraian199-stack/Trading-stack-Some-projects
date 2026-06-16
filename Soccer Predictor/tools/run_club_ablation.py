"""
Club ablation runner: for one league, test (a) which feature groups help the
classifier and (b) how every model stacks up against the market, out of sample.

Reusable for any football-data.co.uk league (EPL E0, La Liga SP1, Bundesliga D1,
Serie A I1, Ligue 1 F1, plus lower divisions). Prints a JSON blob so it can be
run in parallel and synthesised.

Usage:
    PYTHONPATH=src python tools/run_club_ablation.py --league E0 \
        --seasons 1819,1920,2021,2122,2223,2324 --folds 6
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

import soccer_predictor as sp  # noqa: E402
from soccer_predictor.data import schemas, sources  # noqa: E402
from soccer_predictor.evaluation import ablation, ml_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--seasons", default="1819,1920,2021,2122,2223,2324")
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()

    frames = []
    for code in [s.strip() for s in args.seasons.split(",") if s.strip()]:
        try:
            frames.append(sources.fetch_football_data(league=args.league, season=code))
        except Exception as exc:  # noqa: BLE001
            print(f"# season {code} failed: {exc}", file=sys.stderr)
    if not frames:
        print(json.dumps({"league": args.league, "error": "no seasons fetched"}))
        return 1
    df = schemas.validate_matches(pd.concat(frames, ignore_index=True))

    out: dict = {
        "league": args.league,
        "n_matches": int(len(df)),
        "date_range": f"{df['date'].min().date()}..{df['date'].max().date()}",
        "has_odds": bool(schemas.has_usable_odds(df)),
    }

    # (a) feature-group ablation for the classifier
    try:
        fa = ablation.walk_forward_ablation(
            df, ablation.club_feature_variants(), n_folds=args.folds
        )
        out["feature_ablation"] = fa.reset_index(names="variant").round(4).to_dict("records")
    except Exception as exc:  # noqa: BLE001
        out["feature_ablation_error"] = str(exc)

    # (b) full model leaderboard vs the market (with overfit + beat-market)
    try:
        lb = ml_report.ml_comparison(df, n_folds=args.folds, include_overfit=True)
        out["model_leaderboard"] = lb.reset_index(names="model").round(4).to_dict("records")
    except Exception as exc:  # noqa: BLE001
        out["model_leaderboard_error"] = str(exc)

    # (c) Elo-hyperparameter ablation -- does K=30 / slope=0.9 (which helped on the
    # WC backtest) GENERALISE to clubs, or was that backtest-overfitting?
    try:
        eh = ablation.walk_forward_ablation(
            df, ablation.elo_hyperparam_variants(), n_folds=args.folds
        )
        out["elo_hyperparam_ablation"] = eh.reset_index(names="variant").round(4).to_dict("records")
    except Exception as exc:  # noqa: BLE001
        out["elo_hyperparam_error"] = str(exc)

    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
