"""
Command-line interface for ``soccer_predictor``.

One ``argparse`` parser with seven subcommands, each a thin wrapper around the
already-written package internals (loaders, models, evaluation, simulation). The
CLI never reimplements modelling or simulation logic -- it only wires data into a
model and a model into an evaluator / simulator, then prints the result.

Subcommands (exact names, the contract the package depends on):

    train               fit a model on a data source and report a fit summary
    predict-match       full market bundle (1X2, O/U, BTTS, fair odds) for a tie
    backtest            walk-forward metrics + value-betting backtest
    simulate-league     Monte-Carlo a single-table league
    simulate-tournament Monte-Carlo a groups+knockout tournament (default WC2026)
    update-data         refresh the cached remote data sources
    evaluate-market     model vs the de-vigged book on odds-complete rows

Data sources are resolved uniformly via ``--data`` (a canonical / football-data /
international CSV or directory) or, when omitted, synthetic demo data -- which is
LOUDLY flagged as not real (a project non-negotiable). The default remote results
URL is ``sources.INTERNATIONAL_RESULTS_URL`` and ``--format`` defaults to
``world_cup_2026``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import loaders, odds_api, players, schemas, sources, synthetic, world_cup
from .evaluation import betting, walk_forward, wc_backtest
from .models import (
    DixonColes,
    EloGoalsModel,
    MarketBaseline,
    PoissonXG,
    default_ensemble,
)
from .models.base import ScorelineModel, predict_markets
from .models.market_anchor import MarketAnchoredModel
from .simulation import rules
from .simulation.monte_carlo import monte_carlo_league
from .simulation.tournament import simulate_tournament

# --------------------------------------------------------------------------- #
# Model registry (name -> zero-arg factory)
# --------------------------------------------------------------------------- #
# A factory rather than an instance so walk-forward gets a FRESH model per fold.
MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "elo": EloGoalsModel,
    "dixon-coles": DixonColes,
    "xg": PoissonXG,
    "market": MarketBaseline,
    "ensemble": default_ensemble,
}

# Models usable by the simulator must expose a full score matrix.
_SCORELINE_MODELS = {"elo", "dixon-coles", "xg", "ensemble"}

SAMPLE_DATA_WARNING = (
    "WARNING: using SYNTHETIC demo data -- these numbers are NOT real and must "
    "not be trusted. Supply --data <csv|dir> with real matches for any serious "
    "use."
)


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    """Resolve a match frame from the parsed args.

    Returns ``(frame, is_synthetic)``. ``--data`` points at a CSV file or a
    directory; ``--source`` selects the loader (auto / canonical / football-data
    / international). With no ``--data`` we fall back to synthetic demo data and
    flag it.
    """
    path = getattr(args, "data", None)
    if not path:
        kind = getattr(args, "synthetic", "league")
        if kind == "international":
            return synthetic.generate_international_history(), True
        return synthetic.generate_league(), True

    source = getattr(args, "source", "auto")
    loader = _resolve_loader(source, path)
    return loader(path), False


def _resolve_loader(source: str, path: str) -> Callable[[str], pd.DataFrame]:
    """Pick the loader for a ``--source`` choice (auto-detects when 'auto')."""
    if source == "canonical":
        return loaders.load_matches
    if source == "football-data":
        return loaders.load_football_data
    if source == "international":
        return loaders.load_international_results
    # auto: try canonical first, then international, then football-data.
    def _auto(p: str) -> pd.DataFrame:
        for fn in (
            loaders.load_matches,
            loaders.load_international_results,
            loaders.load_football_data,
        ):
            try:
                return fn(p)
            except (ValueError, schemas.SchemaError, KeyError):
                continue
        # Re-raise the canonical loader's error if nothing worked.
        return loaders.load_matches(p)

    return _auto


def _make_model(name: str) -> Any:
    if name not in MODEL_FACTORIES:
        raise SystemExit(
            f"Unknown model {name!r}. Choose from {sorted(MODEL_FACTORIES)}."
        )
    return MODEL_FACTORIES[name]()


def _require_scoreline(name: str) -> None:
    if name not in _SCORELINE_MODELS:
        raise SystemExit(
            f"Model {name!r} cannot drive a simulation (it is not a scoreline "
            f"model). Choose from {sorted(_SCORELINE_MODELS)}."
        )


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    """Print a result payload as JSON or as readable key: value lines."""
    if as_json:
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        _print_human(payload)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def _print_human(payload: dict[str, Any], indent: int = 0) -> None:
    pad = "  " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_human(value, indent + 1)
        elif isinstance(value, float):
            print(f"{pad}{key}: {value:.4f}")
        else:
            print(f"{pad}{key}: {value}")


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    """Fit a model and print a short training summary."""
    df, synthetic_data = _load_data(args)
    if synthetic_data:
        print(SAMPLE_DATA_WARNING, file=sys.stderr)

    train = schemas.completed_matches(df)
    model = _make_model(args.model)
    model.fit(train.copy())

    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    summary: dict[str, Any] = {
        "model": args.model,
        "synthetic_data": synthetic_data,
        "n_matches": int(len(train)),
        "n_teams": len(teams),
        "date_min": str(train["date"].min()) if len(train) else "",
        "date_max": str(train["date"].max()) if len(train) else "",
    }
    if isinstance(model, DixonColes) and model.attack is not None:
        summary["home_advantage"] = float(model.home_adv)
        summary["rho"] = float(model.rho)
    _emit(summary, args.json)
    return 0


def cmd_predict_match(args: argparse.Namespace) -> int:
    """Fit a scoreline model and print the full market bundle for one fixture."""
    _require_scoreline(args.model)
    df, synthetic_data = _load_data(args)
    if synthetic_data:
        print(SAMPLE_DATA_WARNING, file=sys.stderr)

    model = _make_model(args.model)
    model.fit(schemas.completed_matches(df).copy())
    if not isinstance(model, ScorelineModel):  # pragma: no cover - guarded above
        raise SystemExit(f"Model {args.model!r} is not a scoreline model.")

    context = {"neutral_site": bool(args.neutral)}
    bundle = predict_markets(
        model, args.home, args.away, context=context, ou_line=args.ou_line
    )
    bundle["model"] = args.model
    bundle["neutral_site"] = bool(args.neutral)
    _emit(bundle, args.json)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward evaluation plus a value-betting backtest."""
    df, synthetic_data = _load_data(args)
    if synthetic_data:
        print(SAMPLE_DATA_WARNING, file=sys.stderr)

    factory = MODEL_FACTORIES[args.model] if args.model in MODEL_FACTORIES else None
    if factory is None:
        raise SystemExit(
            f"Unknown model {args.model!r}. Choose from {sorted(MODEL_FACTORIES)}."
        )

    probs, outcomes, test_df = walk_forward.walk_forward(
        factory, df, min_train_frac=args.min_train_frac, n_folds=args.folds
    )
    from .evaluation import metrics

    payload: dict[str, Any] = {
        "model": args.model,
        "synthetic_data": synthetic_data,
        "n_test": int(len(outcomes)),
    }
    if len(outcomes):
        payload["metrics"] = metrics.all_metrics(probs, outcomes)
        bet = betting.betting_backtest(
            probs,
            test_df,
            edge_threshold=args.edge,
            stake=1.0,
            kelly_fraction=args.kelly,
            best_edge_only=True,
        )
        payload["betting"] = bet
        clv = betting.closing_line_value(probs, test_df, edge_threshold=args.edge)
        payload["closing_line_value"] = clv
        payload["betting_caveat"] = (
            "Positive ROI is a hypothesis, not proof of edge -- it can be "
            "overfitting or beating opening rather than closing prices."
        )
    else:
        payload["error"] = "No test rows produced; provide more data."
    _emit(payload, args.json)
    return 0


def cmd_evaluate_market(args: argparse.Namespace) -> int:
    """Compare a model against the de-vigged book on odds-complete rows."""
    df, synthetic_data = _load_data(args)
    if synthetic_data:
        print(SAMPLE_DATA_WARNING, file=sys.stderr)

    if not schemas.has_usable_odds(df):
        raise SystemExit(
            "This data has no usable 1X2 odds, so there is no market to compare "
            "against. Supply a frame with home_odds/draw_odds/away_odds."
        )

    factory = MODEL_FACTORIES.get(args.model)
    if factory is None:
        raise SystemExit(
            f"Unknown model {args.model!r}. Choose from {sorted(MODEL_FACTORIES)}."
        )

    result = walk_forward.compare_to_market(
        factory, df, min_train_frac=args.min_train_frac, n_folds=args.folds
    )
    payload: dict[str, Any] = {
        "model": args.model,
        "synthetic_data": synthetic_data,
        "model_metrics": result.get("model", {}),
        "market_metrics": result.get("market", {}),
        "model_minus_market_log_loss": result.get(
            "model_minus_market_log_loss", float("nan")
        ),
        "note": (
            "A negative model_minus_market_log_loss means the model beats the "
            "book on log loss on the rows the book can price."
        ),
    }
    _emit(payload, args.json)
    return 0


def cmd_simulate_league(args: argparse.Namespace) -> int:
    """Monte-Carlo a single-table league season."""
    _require_scoreline(args.model)
    df, synthetic_data = _load_data(args)
    if synthetic_data:
        print(SAMPLE_DATA_WARNING, file=sys.stderr)

    train = schemas.completed_matches(df)
    model = _make_model(args.model)
    model.fit(train.copy())

    # Teams come from an explicit --teams CSV column, or the data's own teams.
    if args.teams:
        teams = [t.strip() for t in args.teams.split(",") if t.strip()]
    else:
        teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    if len(teams) < 2:
        raise SystemExit("Need at least two teams to simulate a league.")

    table = monte_carlo_league(
        model,
        teams,
        n_simulations=args.sims,
        seed=args.seed,
        double_round_robin=not args.single_round_robin,
    )
    _print_table(table, args, head=args.top)
    return 0


def cmd_simulate_tournament(args: argparse.Namespace) -> int:
    """Monte-Carlo a groups+knockout tournament (default WC2026)."""
    _require_scoreline(args.model)
    df, synthetic_data = _load_data(args)
    if synthetic_data:
        print(SAMPLE_DATA_WARNING, file=sys.stderr)

    fmt = _resolve_format(args.format)

    if args.groups:
        groups = loaders.load_groups(args.groups)
    elif synthetic_data:
        groups = _synthetic_groups(df, fmt)
    else:
        raise SystemExit(
            "simulate-tournament needs a group draw: pass --groups <group,team CSV>."
        )

    fixtures = loaders.load_fixtures(args.fixtures) if args.fixtures else None

    train = schemas.completed_matches(df)
    model = _make_model(args.model)
    model.fit(train.copy())

    print(
        f"Simulating {fmt.name} ({args.sims} runs)..."
        + (f"\nAssumptions: {fmt.notes}" if fmt.notes else ""),
        file=sys.stderr,
    )
    table = simulate_tournament(
        model,
        groups,
        fixtures=fixtures,
        n_simulations=args.sims,
        seed=args.seed,
        fmt=fmt,
    )
    _print_table(table, args, head=args.top)
    return 0


def cmd_update_data(args: argparse.Namespace) -> int:
    """Refresh the cached remote data sources (re-runs reuse the cache)."""
    refreshed: dict[str, Any] = {}
    targets = args.targets or ["international"]

    for target in targets:
        try:
            if target == "international":
                path = sources.cached_get(
                    args.results_url, ttl_days=0.0 if args.force else 7.0
                )
                frame = loaders.load_international_results(path)
                refreshed["international"] = {
                    "cache_path": path,
                    "n_matches": int(len(frame)),
                }
            elif target == "football-data":
                frame = sources.fetch_football_data(
                    league=args.league, season=args.season, cache=not args.force
                )
                refreshed["football-data"] = {
                    "league": args.league,
                    "season": args.season,
                    "n_matches": int(len(frame)),
                }
            else:
                refreshed[target] = {"error": f"unknown target {target!r}"}
        except Exception as exc:  # informative degradation, never a bare traceback
            refreshed[target] = {"error": f"{type(exc).__name__}: {exc}"}

    _emit({"refreshed": refreshed}, args.json)
    return 0


def cmd_world_cup(args: argparse.Namespace) -> int:
    """Live FIFA World Cup 2026: fetch real data, lock in played games, simulate."""
    _require_scoreline(args.model)
    print(
        f"Fetching real data and simulating World Cup 2026 with the {args.model} "
        f"model ({args.sims} runs)...",
        file=sys.stderr,
    )

    # Form / squad overlays are Elo-only (the national-team model). The market
    # anchor wraps ANY scoreline model and is ON BY DEFAULT (the live line is the
    # strongest signal); pass --no-anchor to disable.
    overlays = args.form_weight or args.squad_csv
    if overlays and args.model != "elo":
        raise SystemExit(
            "--form-weight / --squad-csv are only supported with --model elo "
            "(the national-team model)."
        )

    # Build + fit the base model ourselves so we can attach overlays / the anchor.
    # Train on history + live Odds-API results (martj42 lags) unless disabled, so
    # the model reflects games played today.
    history = world_cup.live_training_frame(use_odds_api_scores=not args.no_live_scores)
    if args.model == "elo":
        squad = None
        if args.squad_csv:
            squad = players.to_elo_adjustment(
                players.load_squad_strength_csv(args.squad_csv), log=args.squad_log
            )
            print(f"Loaded squad strength for {len(squad)} teams "
                  f"(CURRENT-only; not backtestable).", file=sys.stderr)
        model = EloGoalsModel(form_weight=args.form_weight, squad_strength=squad,
                              squad_weight=args.squad_weight)
    else:
        model = _make_model(args.model)
    model.fit(history.copy())

    if not args.no_anchor:
        try:
            odds = odds_api.fetch_match_odds(aggregate="avg")
            model = MarketAnchoredModel(model, weight=args.anchor_weight, odds=odds)
            print(f"Anchored to live market odds for {len(odds)} fixtures "
                  f"(weight={args.anchor_weight}); unpriced future matchups use the "
                  f"pure model. Pass --no-anchor to disable.", file=sys.stderr)
        except odds_api.OddsApiError as exc:
            print(f"Anchor unavailable, running pure model ({exc}).", file=sys.stderr)

    table = world_cup.simulate(
        model=model,
        model_name=args.model,
        n_simulations=args.sims,
        seed=args.seed,
        refresh_groups_data=args.refresh_groups,
        use_odds_api_scores=not args.no_live_scores,
    )
    _print_table(table, args, head=args.top)
    return 0


def cmd_fetch_ratings(args: argparse.Namespace) -> int:
    """Fetch a team,strength CSV for the squad-strength overlay (one command)."""
    if args.source == "transfermarkt":
        _msg = {
            "ability": "AGE-ADJUSTED squad ability (de-ages value toward player "
                       "quality; CURRENT-only)",
            "value": "raw squad MARKET VALUE (age-discounted; transfer worth, not "
                     "pure ability; CURRENT-only)",
            "league": "squad LEAGUE STRENGTH (value-weighted mean league tier; "
                      "level of competition; CURRENT-only)",
        }[args.metric]
        print(f"Fetching {_msg} from Transfermarkt...", file=sys.stderr)
        try:
            if args.metric == "league":
                # Only keep well-covered squads; low-match teams stay on pure Elo
                # (the universal fallback) rather than a default-tier guess.
                frame = players.fetch_transfermarkt_squad_metrics(
                    pages=args.pages, cache=not args.force
                )
                kept = frame[frame["club_match_rate"] >= args.min_match_rate]
                dropped = len(frame) - len(kept)
                if dropped:
                    print(f"Dropped {dropped} squads below club_match_rate "
                          f"{args.min_match_rate} (they will be ranked by Elo).",
                          file=sys.stderr)
                vals = dict(zip(kept["team"], kept["league_strength"]))
            elif args.metric == "ability":
                vals = players.fetch_transfermarkt_ability(
                    pages=args.pages, cache=not args.force
                )
            else:
                vals = players.fetch_transfermarkt_values(
                    pages=args.pages, cache=not args.force
                )
        except Exception as exc:  # noqa: BLE001 - degrade with a clear message
            raise SystemExit(f"Transfermarkt fetch failed: {exc}")
    else:
        raise SystemExit(f"Unknown --source {args.source!r}.")
    df = pd.DataFrame(
        {"team": list(vals), "strength": list(vals.values())}
    ).sort_values("strength", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} teams to {args.out}. Use it with: "
          f"world-cup --squad-csv {args.out} --squad-weight 1.0", file=sys.stderr)
    print(df.head(args.top).to_csv(index=False))
    return 0


def cmd_backtest_world_cups(args: argparse.Namespace) -> int:
    """Match-level out-of-sample backtest of model variants on past World Cups."""
    history = sources.fetch_international_results()
    factories = {
        "elo": EloGoalsModel,
        "elo+importance": lambda: EloGoalsModel(use_importance=True),
        "elo+form": lambda: EloGoalsModel(form_weight=args.form_weight or 250.0),
        "elo+pedigree": lambda: EloGoalsModel(pedigree_weight=0.5),
    }
    if args.model != "all":
        factories = {args.model: factories.get(args.model, EloGoalsModel)}
    print(
        f"Backtesting on World Cups since {args.min_year} (match-level, "
        "out-of-sample). Historical market odds are not free, so only model "
        "variants are compared.",
        file=sys.stderr,
    )
    lb = wc_backtest.compare_models_on_world_cups(
        history, factories, min_year=args.min_year, update_within=not args.no_update
    )
    if args.tournament or args.champions:
        print(
            f"\nTournament-simulation backtest since {args.min_year} (the same "
            f"pipeline as the live World Cup tab: reconstruct groups, fit "
            f"pre-tournament, Monte-Carlo the bracket; {args.sims} sims).",
            file=sys.stderr,
        )
        res = wc_backtest.backtest_tournament(
            history, EloGoalsModel, min_year=args.min_year, n_simulations=args.sims
        )
        print("Per edition (champion the model would have predicted):", file=sys.stderr)
        print(res["editions"].to_csv(index=False))
        print("Stage-reach calibration (Brier skill vs base rate; >0 = real skill):",
              file=sys.stderr)
        print(res["calibration"].to_csv(index=False))
        cs = res["champion_skill"]
        print(
            f"Champion log-loss: model {cs['mean_neglogp_model']} vs uniform "
            f"{cs['mean_neglogp_uniform']} over {cs['editions']} editions "
            f"(lower = better; below uniform = real skill).",
            file=sys.stderr,
        )
    if getattr(args, "squad", False):
        try:
            from .data import players, squad_value
            squad_value.build_squad_value_by_year()  # fetch + cache (CC0 Transfermarkt)
            byyear = squad_value.load_squad_value_by_year()
        except Exception as exc:  # network / data unavailable -> skip gracefully
            print(f"\nSquad-value ablation skipped ({exc}).", file=sys.stderr)
            byyear = {}
        if byyear:
            adj = {y: players.to_elo_adjustment(byyear[y], spread=60.0, log=True)
                   for y in byyear}
            years = sorted(byyear)  # editions the CC0 data covers (2014/2018/2022)
            sq = {
                "elo": EloGoalsModel,
                "elo+squad@0.5": lambda year: EloGoalsModel(
                    squad_strength=adj.get(year), squad_weight=0.5),
                "elo+squad@1.0": lambda year: EloGoalsModel(
                    squad_strength=adj.get(year), squad_weight=1.0),
            }
            print(
                f"\nSquad-value overlay ablation on {years} (point-in-time "
                "Transfermarkt squad value, CC0; under-covered nations fall back to "
                "pure Elo). Does talent-via-value beat plain Elo out-of-sample?",
                file=sys.stderr,
            )
            sqlb = wc_backtest.compare_models_on_world_cups(history, sq, years=years)
            print(sqlb.round(4).to_csv())
    if args.out:
        lb.to_csv(args.out)
        print(f"Wrote leaderboard to {args.out}", file=sys.stderr)
    print(lb.round(4).to_csv())
    return 0


def cmd_ml_report(args: argparse.Namespace) -> int:
    """Disciplined ML leaderboard: every model vs the market, out of sample.

    Reports log loss / RPS / Brier / ECE, the no-skill floor, the overfit gap,
    and a beat-the-market verdict (bootstrap CI on the per-match log-loss delta).
    """
    from .evaluation import ml_report

    if args.seasons:
        codes = [s.strip() for s in args.seasons.split(",") if s.strip()]
        frames = []
        for code in codes:
            try:
                frames.append(sources.fetch_football_data(league=args.league, season=code))
            except Exception as exc:  # noqa: BLE001
                print(f"season {code} failed: {exc}", file=sys.stderr)
        if not frames:
            raise SystemExit("No football-data seasons fetched.")
        df = pd.concat(frames, ignore_index=True)
        df = schemas.validate_matches(df)
        synthetic_data = False
    else:
        df, synthetic_data = _load_data(args)
    if synthetic_data:
        print(SAMPLE_DATA_WARNING, file=sys.stderr)
    if not schemas.has_usable_odds(df):
        print("NOTE: this data has no usable odds, so the beat-the-market columns "
              "will be blank (e.g. international history has no free closing odds).",
              file=sys.stderr)

    print(f"Running ML comparison on {len(df)} matches "
          f"({df['date'].min().date()}->{df['date'].max().date()}), "
          f"{args.folds} walk-forward folds...", file=sys.stderr)
    lb = ml_report.ml_comparison(
        df, min_train_frac=args.min_train_frac, n_folds=args.folds,
        include_overfit=not args.no_overfit,
    )
    if args.out:
        lb.to_csv(args.out)
        print(f"Wrote leaderboard to {args.out}", file=sys.stderr)
    print(lb.to_csv())
    print("Reading it: 'market' is the benchmark; vs_market<0 means the model "
          "beat the book OOS; beats_market=True only if the bootstrap CI is wholly "
          "below 0 (a free model that does is suspicious, not a gift). overfit_gap "
          "= OOS minus in-sample log loss (big = memorising).", file=sys.stderr)
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Does each variable earn its place out of sample? Delta-vs-baseline table."""
    from .evaluation import ablation

    if args.target in ("wc-model", "wc-hyperparams"):
        history = sources.fetch_international_results()
        variants = (
            ablation.national_model_variants() if args.target == "wc-model"
            else ablation.elo_hyperparam_variants()
        )
        print(f"WC ablation [{args.target}] on past World Cups since {args.min_year} "
              "(out-of-sample, match-level). 'helps'=lowers log loss vs baseline.",
              file=sys.stderr)
        tbl = ablation.wc_ablation(
            history, variants, min_year=args.min_year, update_within=not args.no_update
        )
    elif args.target == "club-features":
        codes = [s.strip() for s in (args.seasons or "1819,1920,2021,2122,2223,2324").split(",")]
        frames = []
        for code in codes:
            try:
                frames.append(sources.fetch_football_data(league=args.league, season=code))
            except Exception as exc:  # noqa: BLE001
                print(f"season {code} failed: {exc}", file=sys.stderr)
        if not frames:
            raise SystemExit("No football-data seasons fetched.")
        df = schemas.validate_matches(pd.concat(frames, ignore_index=True))
        print(f"Club feature-group ablation on {args.league} ({len(df)} matches), "
              f"{args.folds} walk-forward folds.", file=sys.stderr)
        tbl = ablation.walk_forward_ablation(
            df, ablation.club_feature_variants(), n_folds=args.folds
        )
    else:
        raise SystemExit(f"Unknown --target {args.target!r}.")
    if args.out:
        tbl.to_csv(args.out)
        print(f"Wrote {args.out}", file=sys.stderr)
    print(tbl.to_csv())
    return 0


def cmd_edges(args: argparse.Namespace) -> int:
    """Betting Edge Scanner: model fair odds vs the best live market price."""
    _require_scoreline(args.model)
    # Training data: --data if given, else real international history.
    if args.data:
        df = _resolve_loader(args.source, args.data)(args.data)
    else:
        df = sources.fetch_international_results()

    model = _make_model(args.model)
    model.fit(schemas.completed_matches(df).copy())

    try:
        odds = odds_api.fetch_match_odds(
            sport=args.sport, aggregate="best", ttl_hours=args.max_age
        )
    except odds_api.OddsApiError as exc:
        raise SystemExit(str(exc)) from exc
    if odds.empty:
        print(f"No upcoming {args.sport} games with odds right now.", file=sys.stderr)
        return 0

    # The /odds endpoint also returns in-play games whose prices move every
    # minute; skip already-kicked-off fixtures unless the user asks to keep them.
    if not args.include_live and "date" in odds.columns:
        now = pd.Timestamp.utcnow().tz_localize(None)
        odds = odds[odds["date"] >= now]
        if odds.empty:
            print("All games have already kicked off; pass --include-live to "
                  "scan them anyway.", file=sys.stderr)
            return 0

    from .features.market import implied_probabilities

    known = model.known_teams() if hasattr(model, "known_teams") else None
    unmatched: set[str] = set()
    rows = []
    ctx = {"neutral_site": bool(args.neutral)}
    for r in odds.itertuples(index=False):
        # Known-team guard: an Odds-API name the fitted model never saw would
        # silently get a generic base rating and masquerade as value. Flag it.
        teams_known = known is None or (r.home_team in known and r.away_team in known)
        if known is not None:
            if r.home_team not in known:
                unmatched.add(r.home_team)
            if r.away_team not in known:
                unmatched.add(r.away_team)
        p = model.outcome_probs(r.home_team, r.away_team, ctx)
        market = {"home": r.home_odds, "draw": r.draw_odds, "away": r.away_odds}
        probs = {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])}
        mkt_p = implied_probabilities(r.home_odds, r.draw_odds, r.away_odds)
        mkt = {"home": float(mkt_p[0]), "draw": float(mkt_p[1]), "away": float(mkt_p[2])}
        evs = {k: probs[k] * market[k] - 1.0 for k in market}
        best = max(evs, key=evs.get)
        # A reported edge is suspect (likely model miscalibration, not value) when:
        #  - the pick is a heavy longshot the market prices below ~8%, OR
        #  - the model diverges implausibly far from the market on the pick
        #    (>0.10 absolute) or is near-certain (>0.95), OR
        #  - either team was not in the training data (generic fallback rating).
        suspect = (
            mkt[best] < 0.08
            or market[best] > 12.0
            or abs(probs[best] - mkt[best]) > 0.10
            or probs[best] > 0.95
            or not teams_known
        )
        rows.append(
            {
                "date": pd.Timestamp(r.date).strftime("%Y-%m-%d %H:%M"),
                "match": f"{r.home_team} v {r.away_team}",
                "best_bet": best,
                "model_p": round(probs[best], 3),
                "market_p": round(mkt[best], 3),
                "best_odds": market[best],
                "edge": round(evs[best], 3),
                "value": (evs[best] > args.edge) and not suspect,
                "likely_miscalibration": suspect,
                "unmatched_team": not teams_known,
            }
        )
    table = pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
    if unmatched:
        print(
            "WARNING: these Odds-API team names were NOT in the training data and "
            f"got generic fallback ratings (predictions unreliable): {sorted(unmatched)}. "
            "Add them to data/aliases.py.",
            file=sys.stderr,
        )
    print(
        "Edge = model_prob * best_decimal_odds - 1. value=True only when the edge "
        "clears the threshold AND the pick is not flagged likely_miscalibration "
        "(longshot the market prices <~8%, implausible model-vs-market divergence, "
        "or an unmatched team). Positive edge is a HYPOTHESIS; the closing line is "
        "the real bar. Calibrate the model (CalibratedModel) before trusting any edge.",
        file=sys.stderr,
    )
    _print_table(table, args, head=args.top)
    return 0


# --------------------------------------------------------------------------- #
# Shared subcommand utilities
# --------------------------------------------------------------------------- #
def _resolve_format(name: str) -> rules.TournamentFormat:
    """Map a ``--format`` string onto a TournamentFormat (default WC2026)."""
    try:
        return rules.get_format(name)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def _synthetic_groups(
    df: pd.DataFrame, fmt: rules.TournamentFormat
) -> dict[str, list[str]]:
    """Build a placeholder group draw from the synthetic teams.

    Only used in the demo (synthetic) path so simulate-tournament runs end to end
    with no external files. Fills ``n_groups`` of ``teams_per_group`` teams.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n_groups = fmt.n_groups or 12
    per_group = fmt.teams_per_group or 4
    needed = n_groups * per_group
    if len(teams) < needed:
        # Reuse teams cyclically so the demo still fills every slot.
        teams = (teams * ((needed // max(1, len(teams))) + 1))[:needed]
    else:
        teams = teams[:needed]
    letters = [chr(ord("A") + i) for i in range(n_groups)]
    groups: dict[str, list[str]] = {}
    for gi, letter in enumerate(letters):
        groups[letter] = teams[gi * per_group : (gi + 1) * per_group]
    return groups


def _print_table(table: pd.DataFrame, args: argparse.Namespace, head: int) -> None:
    """Print a result frame as CSV (default), JSON, or a head() preview."""
    if getattr(args, "out", None):
        table.to_csv(args.out, index=False)
        print(f"Wrote {len(table)} rows to {args.out}", file=sys.stderr)
    view = table.head(head) if head and head > 0 else table
    if args.json:
        print(view.to_json(orient="records", indent=2))
    else:
        print(view.to_csv(index=False))


# --------------------------------------------------------------------------- #
# Parser construction
# --------------------------------------------------------------------------- #
def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data",
        default=None,
        help="Path to a canonical / football-data / international CSV or a "
        "directory of CSVs. Omit to use synthetic demo data.",
    )
    p.add_argument(
        "--source",
        choices=["auto", "canonical", "football-data", "international"],
        default="auto",
        help="Which loader to use for --data (default: auto-detect).",
    )
    p.add_argument(
        "--synthetic",
        choices=["league", "international"],
        default="league",
        help="Which synthetic generator to use when --data is omitted.",
    )
    p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON output."
    )


def _add_model_arg(p: argparse.ArgumentParser, default: str = "ensemble") -> None:
    p.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORIES),
        default=default,
        help=f"Model to use (default: {default}).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser with all seven subcommands."""
    parser = argparse.ArgumentParser(
        prog="soccer-predictor",
        description=(
            "Unified club + international + tournament soccer prediction. "
            "Subcommands fit models, predict fixtures, backtest, and simulate "
            "leagues/tournaments. With no --data, clearly-labelled SYNTHETIC "
            "demo data is used."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---------------------------------------------------------------
    p_train = sub.add_parser("train", help="Fit a model and print a fit summary.")
    _add_data_args(p_train)
    _add_model_arg(p_train)
    p_train.set_defaults(func=cmd_train)

    # --- predict-match -------------------------------------------------------
    p_pred = sub.add_parser(
        "predict-match",
        help="Full market bundle (1X2, O/U, BTTS, fair odds) for one fixture.",
    )
    _add_data_args(p_pred)
    _add_model_arg(p_pred, default="dixon-coles")
    p_pred.add_argument("--home", required=True, help="Home (or listed) team.")
    p_pred.add_argument("--away", required=True, help="Away team.")
    p_pred.add_argument(
        "--neutral", action="store_true", help="Neutral venue (drops home edge)."
    )
    p_pred.add_argument(
        "--ou-line", type=float, default=2.5, help="Over/under goals line."
    )
    p_pred.set_defaults(func=cmd_predict_match)

    # --- backtest ------------------------------------------------------------
    p_bt = sub.add_parser(
        "backtest",
        help="Walk-forward metrics plus a value-betting backtest.",
    )
    _add_data_args(p_bt)
    _add_model_arg(p_bt)
    p_bt.add_argument("--folds", type=int, default=6, help="Walk-forward folds.")
    p_bt.add_argument(
        "--min-train-frac",
        type=float,
        default=0.5,
        help="Fraction of data in the first training window.",
    )
    p_bt.add_argument(
        "--edge", type=float, default=0.05, help="Value-bet edge threshold."
    )
    p_bt.add_argument(
        "--kelly",
        type=float,
        default=0.0,
        help="Fractional Kelly multiplier (0 = flat stake).",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # --- simulate-league -----------------------------------------------------
    p_lg = sub.add_parser(
        "simulate-league", help="Monte-Carlo a single-table league season."
    )
    _add_data_args(p_lg)
    _add_model_arg(p_lg, default="dixon-coles")
    p_lg.add_argument(
        "--teams",
        default=None,
        help="Comma-separated team list (default: all teams in the data).",
    )
    p_lg.add_argument("--sims", type=int, default=2000, help="Number of seasons.")
    p_lg.add_argument("--seed", type=int, default=7, help="RNG seed.")
    p_lg.add_argument(
        "--single-round-robin",
        action="store_true",
        help="Single round-robin (default is home-and-away).",
    )
    p_lg.add_argument("--top", type=int, default=0, help="Show only the top N rows.")
    p_lg.add_argument("--out", default=None, help="Write the full table to this CSV.")
    p_lg.set_defaults(func=cmd_simulate_league)

    # --- simulate-tournament -------------------------------------------------
    p_tn = sub.add_parser(
        "simulate-tournament",
        help="Monte-Carlo a groups+knockout tournament (default world_cup_2026).",
    )
    _add_data_args(p_tn)
    _add_model_arg(p_tn, default="elo")
    p_tn.add_argument(
        "--groups",
        default=None,
        help="group,team CSV draw (required unless using synthetic data).",
    )
    p_tn.add_argument(
        "--fixtures",
        default=None,
        help="Optional fixtures CSV (uses played scores where present).",
    )
    p_tn.add_argument(
        "--format",
        default="world_cup_2026",
        help="Tournament format name (default: world_cup_2026).",
    )
    p_tn.add_argument("--sims", type=int, default=5000, help="Number of runs.")
    p_tn.add_argument("--seed", type=int, default=7, help="RNG seed.")
    p_tn.add_argument("--top", type=int, default=0, help="Show only the top N rows.")
    p_tn.add_argument("--out", default=None, help="Write the full table to this CSV.")
    p_tn.set_defaults(func=cmd_simulate_tournament)

    # --- world-cup (live WC 2026 convenience) --------------------------------
    p_wc = sub.add_parser(
        "world-cup",
        help="Live FIFA World Cup 2026: fetch real data, lock in played games, simulate.",
    )
    _add_model_arg(p_wc, default="elo")
    p_wc.add_argument("--sims", type=int, default=10000, help="Number of runs.")
    p_wc.add_argument("--seed", type=int, default=7, help="RNG seed.")
    p_wc.add_argument(
        "--refresh-groups",
        action="store_true",
        help="Re-fetch the official group draw from Wikipedia (verified vs fixtures).",
    )
    p_wc.add_argument(
        "--no-live-scores",
        action="store_true",
        help="Do NOT top up results from the Odds API /scores endpoint "
        "(by default live results are merged, since martj42 lags).",
    )
    p_wc.add_argument(
        "--no-anchor",
        action="store_true",
        help="Disable the live market anchor (it is ON by default).",
    )
    p_wc.add_argument(
        "--anchor-weight", type=float, default=0.5,
        help="Pull toward the de-vigged market for priced fixtures (0..1).",
    )
    p_wc.add_argument(
        "--form-weight", type=float, default=0.0,
        help="Recent-form Elo-point weight (0=off; backtests show it doesn't help).",
    )
    p_wc.add_argument(
        "--squad-csv", default=None,
        help="team,strength CSV for a CURRENT-only squad overlay (elo model). "
        "Generate one with `soccer-predictor fetch-ratings`.",
    )
    p_wc.add_argument(
        "--squad-weight", type=float, default=1.0,
        help="Weight on the squad-strength overlay when --squad-csv is given.",
    )
    p_wc.add_argument(
        "--squad-log", action="store_true", default=True,
        help="Log-transform squad values before z-scoring (right for market value).",
    )
    p_wc.add_argument(
        "--no-squad-log", action="store_false", dest="squad_log",
        help="Do NOT log-transform squad strengths (use for linear scores).",
    )
    p_wc.add_argument("--top", type=int, default=16, help="Show only the top N rows.")
    p_wc.add_argument("--out", default=None, help="Write the full table to this CSV.")
    p_wc.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_wc.set_defaults(func=cmd_world_cup)

    # --- fetch-ratings (squad-strength CSV) ----------------------------------
    p_fr = sub.add_parser(
        "fetch-ratings",
        help="Fetch a team,strength CSV (squad market value) for the squad overlay.",
    )
    p_fr.add_argument(
        "--source", choices=["transfermarkt"], default="transfermarkt",
        help="Strength source.",
    )
    p_fr.add_argument(
        "--metric", choices=["ability", "value", "league"], default="ability",
        help="'ability' = age-adjusted value (closer to player quality); "
        "'value' = raw market value (age-discounted); 'league' = value-weighted "
        "mean league tier (level of competition). Default: ability.",
    )
    p_fr.add_argument("--pages", type=int, default=3, help="Transfermarkt pages (25 teams each).")
    p_fr.add_argument(
        "--min-match-rate", type=float, default=0.5,
        help="For --metric league: drop squads whose club coverage is below this "
        "(they fall back to pure Elo). Default 0.5.",
    )
    p_fr.add_argument(
        "--out", default="data/squad_strength.csv", help="Output CSV path."
    )
    p_fr.add_argument("--force", action="store_true", help="Bypass the cache.")
    p_fr.add_argument("--top", type=int, default=15, help="Preview the top N rows.")
    p_fr.set_defaults(func=cmd_fetch_ratings)

    # --- backtest-world-cups -------------------------------------------------
    p_bw = sub.add_parser(
        "backtest-world-cups",
        help="Out-of-sample backtest of the model on past WCs (match-level + "
             "full tournament simulation), default back to 1998.",
    )
    p_bw.add_argument(
        "--model",
        choices=["all", "elo", "elo+importance", "elo+form", "elo+pedigree"],
        default="all",
        help="Variant to backtest (default: all).",
    )
    p_bw.add_argument(
        "--min-year", type=int, default=1998,
        help="Earliest WC edition to include (default 1998; the 32-team regime).",
    )
    p_bw.add_argument(
        "--form-weight", type=float, default=250.0,
        help="Form weight used for the elo+form variant.",
    )
    p_bw.add_argument(
        "--no-update", action="store_true",
        help="Do NOT update ratings within each tournament (fit-once).",
    )
    p_bw.add_argument(
        "--tournament", action="store_true",
        help="Also run the full tournament-simulation backtest (champion rank + "
             "stage-reach calibration) -- the same pipeline as the live WC tab.",
    )
    p_bw.add_argument(
        "--champions", action="store_true",
        help="Alias for --tournament (back-compat).",
    )
    p_bw.add_argument(
        "--squad", action="store_true",
        help="Also run the squad-VALUE overlay ablation (point-in-time "
             "Transfermarkt value, 2014/2018/2022) vs plain Elo. Backtested "
             "result: it does NOT beat Elo out-of-sample (kept off).",
    )
    p_bw.add_argument("--sims", type=int, default=4000, help="Tournament-backtest runs.")
    p_bw.add_argument("--out", default=None, help="Write the leaderboard CSV here.")
    p_bw.set_defaults(func=cmd_backtest_world_cups)

    # --- ml-report (disciplined ML leaderboard vs the market) ----------------
    p_ml = sub.add_parser(
        "ml-report",
        help="ML leaderboard (incl. stacked ensemble) vs the market, OOS, with "
        "overfit + beat-the-market diagnostics.",
    )
    _add_data_args(p_ml)
    p_ml.add_argument(
        "--league", default="E0",
        help="football-data league code when using --seasons (default E0 = EPL).",
    )
    p_ml.add_argument(
        "--seasons", default=None,
        help="Comma-separated football-data season codes (e.g. 1819,1920,2021,"
        "2122,2223,2324). Fetches + concatenates them (these carry closing odds).",
    )
    p_ml.add_argument("--folds", type=int, default=6, help="Walk-forward folds.")
    p_ml.add_argument(
        "--min-train-frac", type=float, default=0.5,
        help="Fraction of data in the first training window.",
    )
    p_ml.add_argument(
        "--no-overfit", action="store_true",
        help="Skip the (slower) in-sample-vs-OOS overfit-gap pass.",
    )
    p_ml.add_argument("--out", default=None, help="Write the leaderboard CSV here.")
    p_ml.set_defaults(func=cmd_ml_report)

    # --- ablation (does each variable earn its place OOS?) --------------------
    p_ab = sub.add_parser(
        "ablation",
        help="Ablation: which variables improve out-of-sample predictions "
        "(World Cup backtest or club walk-forward).",
    )
    p_ab.add_argument(
        "--target",
        choices=["wc-model", "wc-hyperparams", "club-features"],
        default="wc-model",
        help="wc-model = EloGoals components; wc-hyperparams = Elo K/home-adv/shape; "
        "club-features = classifier feature groups on a club league.",
    )
    p_ab.add_argument("--min-year", type=int, default=1998, help="Earliest WC edition.")
    p_ab.add_argument("--no-update", action="store_true", help="Fit-once (no within-tournament update).")
    p_ab.add_argument("--league", default="E0", help="Club league code (club-features).")
    p_ab.add_argument("--seasons", default=None, help="Club season codes (club-features).")
    p_ab.add_argument("--folds", type=int, default=6, help="Walk-forward folds (club-features).")
    p_ab.add_argument("--out", default=None, help="Write the table CSV here.")
    p_ab.set_defaults(func=cmd_ablation)

    # --- edges (live Betting Edge Scanner) -----------------------------------
    p_ed = sub.add_parser(
        "edges",
        help="Scan live market odds for model edge (The Odds API).",
    )
    # Elo is the default: it fits in one fast pass over full international
    # history, whereas Dixon-Coles MLE over hundreds of national teams is slow.
    _add_model_arg(p_ed, default="elo")
    p_ed.add_argument(
        "--sport",
        default=odds_api.WORLD_CUP_SPORT,
        help=f"Odds API sport key (default: {odds_api.WORLD_CUP_SPORT}).",
    )
    p_ed.add_argument(
        "--data",
        default=None,
        help="Training data CSV/dir (default: real international history).",
    )
    p_ed.add_argument(
        "--source",
        choices=["auto", "canonical", "football-data", "international"],
        default="auto",
        help="Loader for --data (default: auto-detect).",
    )
    p_ed.add_argument(
        "--neutral", action="store_true", default=True,
        help="Treat fixtures as neutral venue (default for tournaments).",
    )
    p_ed.add_argument(
        "--edge", type=float, default=0.05, help="Edge threshold to flag value."
    )
    p_ed.add_argument(
        "--max-age",
        type=float,
        default=0.5,
        help="Max cached-odds age in hours before re-fetching (default 0.5).",
    )
    p_ed.add_argument(
        "--include-live",
        action="store_true",
        help="Include already-kicked-off (in-play) games (default: skip them).",
    )
    p_ed.add_argument("--top", type=int, default=0, help="Show only the top N rows.")
    p_ed.add_argument("--out", default=None, help="Write the full table to this CSV.")
    p_ed.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_ed.set_defaults(func=cmd_edges)

    # --- update-data ---------------------------------------------------------
    p_up = sub.add_parser(
        "update-data", help="Refresh cached remote data sources."
    )
    p_up.add_argument(
        "targets",
        nargs="*",
        choices=["international", "football-data"],
        help="Which sources to refresh (default: international).",
    )
    p_up.add_argument(
        "--results-url",
        default=sources.INTERNATIONAL_RESULTS_URL,
        help="International results URL.",
    )
    p_up.add_argument("--league", default="E0", help="football-data league code.")
    p_up.add_argument("--season", default="2324", help="football-data season code.")
    p_up.add_argument(
        "--force", action="store_true", help="Ignore the cache and re-download."
    )
    p_up.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_up.set_defaults(func=cmd_update_data)

    # --- evaluate-market -----------------------------------------------------
    p_mk = sub.add_parser(
        "evaluate-market",
        help="Compare a model to the de-vigged book on odds-complete rows.",
    )
    _add_data_args(p_mk)
    _add_model_arg(p_mk)
    p_mk.add_argument("--folds", type=int, default=6, help="Walk-forward folds.")
    p_mk.add_argument(
        "--min-train-frac",
        type=float,
        default=0.5,
        help="Fraction of data in the first training window.",
    )
    p_mk.set_defaults(func=cmd_evaluate_market)

    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the chosen subcommand. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
