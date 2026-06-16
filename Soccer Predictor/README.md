# Soccer Predictor

A unified soccer prediction system that handles **club**, **international**, and
**tournament** prediction from one codebase. The headline use case is **FIFA
World Cup 2026**, but the same engine simulates the Euros, Copa América, the
Champions League, and domestic leagues, and runs club-level betting research.

It is not magic and should not be treated as certainty. It is an honest research
and simulation toolkit: leakage-safe features, time-aware validation, proper
probabilistic scoring, a market benchmark, and explicit tournament assumptions.

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
   betting backtests with closing-line value, and model ensembles.

**Core rule:** the match model only answers *"what happens in this fixture?"*;
the simulator only answers *"given a format, who advances / wins?"* They never mix
responsibilities — every simulator takes a model object and never hardcodes model
logic.

## Model stack

| Model | Type | Use |
|---|---|---|
| `MarketBaseline` | de-vigged odds | the benchmark every model must beat out of sample |
| `EloGoalsModel` | Elo → Poisson scoreline | robust minimum-viable model; **default for the simulator** |
| `DixonColes` | generative goal model (MLE + time decay + low-score correction) | full scoreline matrix → every market |
| `PoissonXG` / `XGPoisson` | generative model on xG | better scoring-rate estimate where xG exists (club) |
| `OutcomeClassifier` | logistic / gradient boosting on engineered features | flexible signal; LightGBM if installed else sklearn HGB |
| `Ensemble` | weighted blend | DC 0.35 + classifier 0.25 + market 0.25 + Elo 0.15, weights redistributed when odds are missing |
| `CalibratedModel` | isotonic / temperature wrapper | calibrate any model on a time-ordered holdout |

Every model exposes `fit(df)` / `predict_proba(df) -> (n, 3)` over `[H, D, A]`.
Every scoreline model also exposes `score_matrix(home, away, context)`.

## Install

```bash
python3 -m pip install -e .          # core: numpy, pandas, scipy, scikit-learn, plotly, streamlit
python3 -m pip install -e ".[boost]" # optional LightGBM
python3 -m pip install -e ".[xg]"    # optional Understat xG ingestion
```

Or run without installing via `PYTHONPATH=src`.

## Quick start (Python, no downloads)

```python
import soccer_predictor as sp

df = sp.generate_league()                                  # synthetic, labelled NOT REAL
probs, outcomes, test = sp.walk_forward(lambda: sp.DixonColes(), df)
print(sp.all_metrics(probs, outcomes))                     # log_loss, rps, brier, accuracy, ece
print(sp.compare_to_market(lambda: sp.DixonColes(), df))   # model vs the book
```

## World Cup 2026

```bash
# 1. Real historical results (cached after first download)
#    martj42/international_results is the backbone source.

# 2. Group + fixture CSVs (templates provided; verify against FIFA before a serious run)
python3 tools/refresh_wc2026_groups_from_wikipedia.py --out data/world_cup_2026_groups.csv

# 3. Simulate
PYTHONPATH=src python3 -m soccer_predictor.cli simulate-tournament \
  --format world_cup_2026 \
  --groups data/world_cup_2026_groups.csv \
  --fixtures data/world_cup_2026_fixtures.csv \
  --sims 50000 \
  --out outputs/world_cup_2026_probabilities.csv
```

```python
import soccer_predictor as sp
hist  = sp.fetch_international_results()        # cached
model = sp.EloGoalsModel().fit(hist)           # or DixonColes / Ensemble
groups = sp.load_groups("data/world_cup_2026_groups.csv")
sims = sp.simulate_tournament(model, groups, n_simulations=50000)
print(sims.sort_values("champion_probability", ascending=False).head(12))
```

If a fixture's `home_score`/`away_score` are filled in, the simulator treats it
as already played and only simulates the rest — so you can update odds live as the
tournament unfolds.

## CLI

```
soccer-predictor train               # fit + save a model
soccer-predictor predict-match       # one fixture: 1X2, xG, scorelines, O/U, BTTS, fair odds
soccer-predictor backtest            # walk-forward metrics
soccer-predictor simulate-league     # domestic-league Monte Carlo table
soccer-predictor simulate-tournament # group + knockout Monte Carlo (World Cup 2026 default)
soccer-predictor update-data         # refresh cached sources
soccer-predictor evaluate-market     # model vs de-vigged closing odds
```

Example:

```bash
PYTHONPATH=src python3 -m soccer_predictor.cli predict-match \
  --home "Brazil" --away "France" --date 2026-06-15 --competition "FIFA World Cup"
```

## Streamlit app

```bash
streamlit run app_unified.py      # the new unified app (Match / Tournament / Backtest / Betting Edge)
streamlit run app.py              # the original World Cup-only app (still works)
```

Four views: **Match Predictor**, **Tournament Simulator**, **Backtest Lab**, and
**Betting Edge Scanner** (model fair odds vs market vs edge, with calibration
warnings).

## Data sources

- **International results:** `martj42/international_results` (every men's
  international since 1872). Fetched + cached automatically.
- **Club results + odds:** football-data.co.uk (one CSV per league per season).
- **Elo:** ClubElo (club) / the built-in margin-weighted Elo (international).
- **xG (club only):** Understat via `understatapi`. **Heads-up:** FBref removed xG
  on 2026-01-20; Understat (top-5 European leagues + RFPL) is the practical free
  source as of mid-2026. xG does **not** cover national teams.
- Verify against FIFA before a serious World Cup run: the official
  scores-and-fixtures page, the men's ranking, and the bracket/third-place rules.

See [docs/data_sources.md](docs/data_sources.md) for the full free-vs-paid map.

## Honest limitations

- Bundled sample / synthetic CSVs are **smoke-test data, not real**. The app
  warns loudly when they are in use.
- The World Cup 2026 third-place allocation is a constraint-satisfaction
  **approximation** of FIFA's Annex C matrix (every assignment is valid; it does
  not reproduce the exact published mapping for all 495 cases). All such
  assumptions are carried on `TournamentFormat.notes` and labelled approximate.
- Group tie-breakers approximate the full FIFA drawing-of-lots procedure.
- A model is only as fresh as its data; without xG / lineups / injuries / odds it
  cannot see them.
- **Positive betting ROI is a hypothesis, not proof.** Beating the *closing* line
  out of sample is the real bar, and most models never clear it. Nothing here is
  financial advice.

## Docs

- [docs/framework.md](docs/framework.md) — architecture
- [docs/data_sources.md](docs/data_sources.md) — sources, club vs international
- [docs/model_notes.md](docs/model_notes.md) — models, ensemble, calibration, caveats
- [docs/CONTRACTS.md](docs/CONTRACTS.md) — internal module API contracts

## Project layout

```
src/soccer_predictor/
  data/        schemas, loaders, normalizers, aliases, sources, quality, synthetic
  features/    elo, rolling, form, rest_travel, market, xg, squad, international, feature_store
  models/      base, market_baseline, dixon_coles, poisson_xg, elo_goals, outcome_classifier, ensemble, calibration
  evaluation/  metrics, walk_forward, betting, reports
  simulation/  rules, match, league, group_stage, knockout, tournament, monte_carlo
  apps/        streamlit_app.py
  cli.py
src/world_cup_predictor/   legacy World Cup-only package (kept working)
tests/                     pytest suite incl. a no-leakage guard
```
