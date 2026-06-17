# Soccer Predictor

A unified soccer prediction system that handles **club**, **international**, and
**tournament** prediction from one codebase. The headline use case is a
**live FIFA World Cup 2026** model, but the same engine simulates the Euros,
Copa América, the Champions League, and domestic leagues, and runs club-level
betting research.

It is not magic and should not be treated as certainty. It is an honest research
and simulation toolkit: leakage-safe features, time-aware validation, proper
probabilistic scoring, a market benchmark, and explicit tournament assumptions.

## Live World Cup 2026

The flagship feature simulates the rest of the World Cup conditioned on the games
already played, and **updates as results come in**:

- Real groups (the actual 72 group-stage fixtures, cross-checked against the
  official draw) and a margin-weighted Elo→Dixon-Coles scoreline model fit on the
  full international history (since 1872).
- **Live results ingestion.** Games that have been played are locked in from three
  sources — the martj42 history feed, the live [Odds API](https://the-odds-api.com)
  `/scores` feed, and a durable on-disk results ledger — so a result like
  "France 3-1 Senegal" both fixes that group game and updates the fitted model.
- **Durable ledger.** Because martj42 caches for days and the Odds-API window only
  spans ~3 days, every completed group game is also persisted to
  `data/world_cup_2026_results.csv` (git-ignored, per-machine). Once a game is
  seen it stays locked in even after it rolls out of both live feeds.
- **Market anchor (on by default).** Near-term fixtures are blended toward the
  de-vigged live betting line — the single strongest short-horizon signal — while
  future knockout matchups use the pure model.

Run it in the browser (the **🏆 World Cup 2026 (live)** tab) or from the CLI:

```bash
# App (5 tabs; the World Cup tab has a "🔄 Refresh live data" button)
python -m streamlit run app_unified.py

# CLI — champion / advancement probabilities, anchored to the live market
PYTHONPATH=src python -m soccer_predictor.cli world-cup --sims 10000 \
  --out outputs/world_cup_2026_probabilities.csv
```

A free [Odds API](https://the-odds-api.com) key enables live odds + scores. Put it
in `~/.odds_api_key` (or set `ODDS_API_KEY`). Without a key the model still runs;
it just can't ingest live results or anchor to the market. Every paid/quota'd call
is cached to disk so re-runs don't burn the free tier.

## Three layers (kept strictly separate)

1. **Match Prediction Engine** — given two teams + context (date, venue,
   competition, neutral), a probability distribution over outcomes and scores:
   1X2, exact scoreline, expected goals, over/under, both-teams-to-score, fair
   odds, and edge vs the market.
2. **Tournament / Season Simulation Engine** — group stages, league tables,
   knockout brackets, best-third-place rules, two-leg ties, extra time and
   penalties, and Monte Carlo winner / advancement probabilities. World Cup 2026
   (12 groups of 4, top two + best eight thirds, Round of 32 → final) is built
   in; custom CSV-defined tournaments are supported.
3. **Research / Betting Evaluation Engine** — expanding walk-forward validation,
   log loss / RPS / Brier / calibration (ECE), market-baseline comparison,
   betting backtests with closing-line value, variable ablation, and ensembles.

**Core rule:** the match model only answers *"what happens in this fixture?"*;
the simulator only answers *"given a format, who advances / wins?"* They never mix
responsibilities — every simulator takes a model object and never hardcodes model
logic.

## Model stack

| Model | Type | Use |
|---|---|---|
| `MarketBaseline` | de-vigged odds | the benchmark every model must beat out of sample |
| `EloGoalsModel` | margin-weighted Elo → goal supremacy → Dixon-Coles matrix | the national-team engine; **default for the simulator** |
| `DixonColes` | generative goal model (MLE + time decay + low-score correction) | full scoreline matrix → every market |
| `PoissonXG` | generative model on xG | better scoring-rate estimate where xG exists (club) |
| `OutcomeClassifier` | logistic / gradient boosting on engineered features | flexible signal; LightGBM if installed else sklearn HGB |
| `Ensemble` | weighted blend | DC + classifier + market + Elo, weights redistributed when odds are missing |
| `MarketAnchoredModel` | wrapper | blends any base model toward the de-vigged live line on priced fixtures |
| `StackedEnsemble` | super-learner (time-series OOF) | best free *calibrated* estimator; never beats the closing line out of sample |
| `CalibratedModel` | isotonic / temperature wrapper | calibrate any model on a time-ordered holdout |

Every model exposes `fit(df)` / `predict_proba(df) -> (n, 3)` over `[H, D, A]`.
Every scoreline model also exposes `score_matrix(home, away, context)`.

## Install

```bash
python3 -m pip install -e .          # core: numpy, pandas, scipy, scikit-learn, plotly, streamlit
python3 -m pip install -e ".[boost]" # optional LightGBM
python3 -m pip install -e ".[xg]"    # optional Understat xG ingestion
```

Installing adds a `soccer-predictor` console script. You can also run without
installing via `PYTHONPATH=src python -m soccer_predictor.cli ...`.

## Quick start (Python, no downloads)

```python
import soccer_predictor as sp

df = sp.generate_league()                                  # synthetic, labelled NOT REAL
probs, outcomes, test = sp.walk_forward(lambda: sp.DixonColes(), df)
print(sp.all_metrics(probs, outcomes))                     # log_loss, rps, brier, accuracy, ece
print(sp.compare_to_market(lambda: sp.DixonColes(), df))   # model vs the book
```

## CLI

13 subcommands (`soccer-predictor <cmd> --help` for flags):

```
train                  fit + save a model
predict-match          one fixture: 1X2, xG, scorelines, O/U, BTTS, fair odds
backtest               walk-forward metrics
simulate-league        domestic-league Monte Carlo table
simulate-tournament    group + knockout Monte Carlo (custom CSV formats)
world-cup              LIVE World Cup 2026 (real data + live results + market anchor)
edges                  live model-vs-market value scan for upcoming fixtures
fetch-ratings          build a team,strength squad-overlay CSV (Transfermarkt)
backtest-world-cups    match- and tournament-level backtest on past World Cups
ml-report              market floor, overfit gap, beats-market bootstrap CI
ablation               does each added variable actually help, out of sample?
update-data            refresh cached sources
evaluate-market        model vs de-vigged closing odds
```

Examples:

```bash
# Live World Cup, pure model (no market anchor), top 16
PYTHONPATH=src python -m soccer_predictor.cli world-cup --no-anchor --top 16

# One fixture
PYTHONPATH=src python -m soccer_predictor.cli predict-match \
  --home "Brazil" --away "France" --date 2026-06-15 --competition "FIFA World Cup"

# Did the optional national-team variables earn their place?
PYTHONPATH=src python -m soccer_predictor.cli ablation
```

## Streamlit app

```bash
python -m streamlit run app_unified.py   # the unified app (5 tabs)
python -m streamlit run app.py           # the original World Cup-only app (still works)
```

Five tabs: **🏆 World Cup 2026 (live)**, **Match Predictor**, **Tournament
Simulator**, **Backtest Lab**, and **Betting Edge Scanner** (model fair odds vs
market vs edge, with calibration warnings).

## How the live data flows

```
martj42 international results ─┐
  (full history; ~6h cache       │
   during the tournament)        ├─► results ledger (data/world_cup_2026_results.csv)
Odds API /scores (last ~3 days) ─┘     monotonic: a game, once seen, is never lost
                                        │
                  ┌─────────────────────┴─────────────────────┐
       locked-in group fixtures                   model training frame
       (build_fixtures, order-insensitive)        (history + games martj42 lacks,
                  │                                 no double-count)
                  └────────────► Monte-Carlo the rest ◄────────┘
                                  (+ optional live-market anchor)
```

Results are matched to fixtures order-insensitively (World Cup games are at
neutral venues, so feeds disagree on who is "home") and the ledger is scoped to
the 72 group pairings and the group-stage date window, so a later knockout result
can never overwrite a group game.

## Data sources

- **International results:** `martj42/international_results` (every men's
  international since 1872). Fetched + cached automatically.
- **Live odds + scores:** [The Odds API](https://the-odds-api.com) (free tier
  ~500 req/month, cached). Powers live results, the market baseline, the market
  anchor, and the edge scanner.
- **Club results + odds:** football-data.co.uk (one CSV per league per season).
- **Elo:** ClubElo (club) / the built-in margin-weighted Elo (international).
- **Squad strength (optional overlay):** Transfermarkt national-team squad value /
  age-adjusted ability / league tier, via `fetch-ratings` → `world-cup --squad-csv`.
- **xG (club only):** Understat via `understatapi`. **Heads-up:** FBref removed xG
  on 2026-01-20; Understat (top-5 European leagues + RFPL) is the practical free
  source as of mid-2026. xG does **not** cover national teams.

See [docs/data_sources.md](docs/data_sources.md) for the full free-vs-paid map.

## Honest limitations

- Bundled sample / synthetic CSVs are **smoke-test data, not real**. The app
  warns loudly when they are in use.
- Backtests on past World Cups show that **none** of the optional national-team
  variables (recent form, historical pedigree, competition-importance weighting)
  reliably help out of sample — they default OFF. Plain margin-weighted Elo plus
  the live market is the trustworthy core.
- The World Cup 2026 third-place allocation is a constraint-satisfaction
  **approximation** of FIFA's Annex C matrix (every assignment is valid; it does
  not reproduce the exact published mapping for all cases). Such assumptions are
  carried on `TournamentFormat.notes` and labelled approximate. Group tie-breakers
  approximate the full FIFA drawing-of-lots procedure.
- Squad-strength overlays are **current-only** (no free historical squads) and
  therefore not backtestable; they overlap the market anchor, which already prices
  ability. Treat them as a minor tilt.
- A model is only as fresh as its data; without xG / lineups / injuries / odds it
  cannot see them.
- **Positive betting ROI is a hypothesis, not proof.** Beating the *closing* line
  out of sample is the real bar, and nothing here clears it consistently. Not
  financial advice.

## Docs

- [docs/framework.md](docs/framework.md) — architecture
- [docs/world_cup_2026.md](docs/world_cup_2026.md) — the World Cup pipeline
- [docs/data_sources.md](docs/data_sources.md) — sources, club vs international
- [docs/model_notes.md](docs/model_notes.md) — models, ensemble, calibration, caveats
- [docs/CONTRACTS.md](docs/CONTRACTS.md) — internal module API contracts

## Project layout

```
src/soccer_predictor/
  data/        schemas, loaders, normalizers, aliases, sources, odds_api,
               players, world_cup, quality, synthetic
  features/    elo, rolling, form, rest_travel, market, xg, squad, international
  models/      base, market_baseline, dixon_coles, poisson_xg, elo_goals,
               outcome_classifier, ensemble, market_anchor, stacking, calibration
  evaluation/  metrics, walk_forward, betting, ml_report, ablation, wc_backtest
  simulation/  rules, match, league, group_stage, knockout, tournament, monte_carlo
  apps/        streamlit_app.py
  cli.py
src/world_cup_predictor/   legacy World Cup-only package (kept working)
app_unified.py             Streamlit entry point (5-tab unified app)
tests/                     pytest suite incl. a no-leakage guard
```
