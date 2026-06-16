# Framework Overview

This document describes the architecture of the `soccer_predictor` package: one
canonical data model, three processing layers, and a module map. It is a map of
the design, not an API reference — for the binding interface contract see
[`CONTRACTS.md`](CONTRACTS.md). For where data comes from see
[`data_sources.md`](data_sources.md); for the modelling choices see
[`model_notes.md`](model_notes.md).

The unifying idea: **club leagues, club cups, international friendlies,
qualifiers, and tournaments all flow through one canonical match table.** Loaders
normalise every source into it, and features, models, evaluation, and simulation
all read it. Adding a new data source means writing a loader and (sometimes) a
name alias — nothing downstream changes.

---

## 1. The canonical data model

There is exactly one match schema, defined in `data/schemas.py`. Every row is one
fixture. Optional columns degrade gracefully: anything a source does not supply
is simply absent, and downstream code checks for presence rather than assuming
it.

**Required identity / result columns** (always present after `validate_matches`):

```
match_id, date, competition, competition_type, season, stage, round, group,
home_team, away_team, home_score, away_score, neutral_site, venue, city,
country, home_country, away_country, is_completed
```

**Optional enrichment columns** (present only when a source carries them):

```
home_xg, away_xg,
home_odds, draw_odds, away_odds,
closing_home_odds, closing_draw_odds, closing_away_odds,
home_rest_days, away_rest_days, home_travel_km, away_travel_km,
home_fifa_rank, away_fifa_rank, home_elo, away_elo,
home_squad_value, away_squad_value, home_injury_score, away_injury_score
```

**Naming is non-negotiable.** Scores are `home_score` / `away_score` (never
`home_goals`). Odds are `home_odds` / `draw_odds` / `away_odds` and their
`closing_*` counterparts (never `odds_h`). The 1X2 label column is `result` ∈
{`H`, `D`, `A`} (Home < Draw < Away, the order RPS relies on), added by
`schemas.add_result_column`. The neutral-ground flag is `neutral_site` (bool).

`competition_type` is the discriminator that lets one table serve both club and
international work. It is one of:

```
club_league, club_cup, international_friendly, qualifier, tournament
```

Key schema helpers downstream code leans on:

- `validate_matches(df, require_scores=False)` — fill required columns, coerce
  dtypes, drop rows without a date or team names, sort by date. The contract
  every loader's output must satisfy.
- `add_result_column(df)` — derive `result` for completed matches (blank string
  for not-yet-played fixtures, so the same table holds played and upcoming games).
- `completed_matches(df)` — rows with both scores; the only rows usable for
  training and evaluation.
- `rows_with_complete_odds(df)` — the correct filter for the market baseline and
  betting (partial odds become NaN downstream and must be excluded, not
  tolerated).
- `has_usable_odds` / `has_usable_xg` — presence gates so optional feature groups
  switch on only when their source columns exist and are non-null.

Team names are funnelled through `data/aliases.normalize_team_name` so the rest
of the system only ever sees one canonical spelling (e.g. "Korea Republic" →
"South Korea", "Man United" canonicalised across club sources). When a join
reports unmatched rows, the fix is almost always a new alias entry.

`data/normalizers.standardize_matches` is the one function every loader calls
last: it renames source columns onto canonical names, normalises team names,
coerces dtypes, dedupes on `(date, home_team, away_team)`, sorts, and adds
`result`. Loaders differ only in the column mapping they pass in.

---

## 2. The three layers

Data flows left to right. Each layer reads the canonical table and never reaches
back to a raw source format.

```
   ┌─────────────┐      ┌──────────────────┐      ┌───────────────────────┐
   │  DATA layer │  →   │ MODELLING layer  │  →   │  SIMULATION layer     │
   │  (ingest)   │      │ (features+models)│      │  (Monte Carlo)        │
   └─────────────┘      └──────────────────┘      └───────────────────────┘
   loaders/sources      features/feature_store     match/league/group_stage
   quality/synthetic    models/* (1X2 + scoreline) knockout/tournament
                        evaluation/* (the judge)    monte_carlo
```

### Layer 1 — Data (`data/`)

Turns the messy outside world into the canonical table.

- `loaders.py` — file/directory loaders: `load_matches`, `load_football_data`
  (football-data.co.uk club CSVs), `load_international_results` (martj42 schema),
  `load_groups`, `load_fixtures`, `load_odds`.
- `sources.py` — remote fetchers with a disk cache (`cached_get`,
  `fetch_international_results`, `fetch_football_data`, `fetch_clubelo`) and
  `classify_competition_type`. Re-runs must never re-fetch (cache policy lives in
  [`data_sources.md`](data_sources.md)). All fetchers degrade gracefully.
- `quality.py` — `quality_report`, duplicate-fixture detection, odds/xG coverage.
- `synthetic.py` — `generate_league` and `generate_international_history` produce
  fully-formed canonical frames (with odds and xG) so the whole pipeline can be
  exercised offline. **Synthetic output is labelled as not real.**

### Layer 2 — Modelling (`features/`, `models/`, `evaluation/`)

**Features** (`features/`) — `feature_store.build_features` is the central,
no-leakage builder. It walks matches in **date order**, attaches only
pre-kickoff features (Elo, rolling form, rest/travel, and — when their source
columns are present — rolling xG, de-vigged market-implied probabilities, FIFA
rank, squad value), and reveals a match's own outcome only **after** its features
are recorded. Optional feature groups switch on by column presence;
`available_feature_columns(df)` reports which are active. Sub-modules: `elo.py`
(World-Football-Elo with a `neutral` flag), `rolling.py` / `form.py`,
`rest_travel.py` (haversine + a starter city-coordinate table), `market.py`,
`xg.py`, `squad.py`, `international.py`.

**Models** (`models/`) — two contracts, both from `models/base.py`:

- `BaseModel`: `fit(train_df) -> self`, `predict_proba(test_df) -> (n, 3)` over
  `[H, D, A]`.
- `ScorelineModel(BaseModel)`: additionally a full score matrix `M[i, j] =
  P(home i, away j)`. **Subclasses implement only `_score_matrix`; predict_proba,
  over/under, BTTS, expected goals, sampling, and penalty resolution are all
  derived in the base.** This is deliberate: every market is collapsed from the
  one matrix, so the "over_under ignores the line" class of bug cannot occur.

  Concrete models: `market_baseline.MarketBaseline` (de-vigged book odds — the
  benchmark), `dixon_coles.DixonColes` (MLE with time-decay + low-score ρ),
  `poisson_xg.PoissonXG` (attack/defence from rolling xG),
  `outcome_classifier.OutcomeClassifier` (LightGBM if available, else sklearn
  `HistGradientBoostingClassifier` / logistic), `ensemble.EloGoalsModel` (the
  robust minimum-viable scoreline model and the simulator's default), and
  `ensemble.Ensemble` / `default_ensemble`. `calibration.CalibratedModel` wraps
  any model with time-ordered isotonic or temperature calibration.

  All scoreline models honour `context["neutral_site"]`: on neutral ground the
  home-advantage term is dropped (most World Cup matches are neutral).

**Evaluation** (`evaluation/`) — the judge. `metrics.py` (log loss, RPS, Brier,
accuracy, ECE), `walk_forward.py` (expanding-window, time-ordered backtest),
`betting.py` (value-betting backtest and closing-line value — **positive ROI is
a hypothesis, not proof**), `reports.py` (leaderboard, calibration table).

### Layer 3 — Simulation (`simulation/`)

A Monte Carlo engine that consumes **any model object** through the
`ScorelineModel` interface only (`sample_score`, `expected_goals`,
`win_probability_no_draw`). It never hardcodes model logic.

- `rules.py` — pure data + functions: standings, configurable tie-breakers,
  `KnockoutTie`, `TournamentFormat` / `BracketMatch`, and the full WC2026 bracket
  (`world_cup_2026_format`). Tournament assumptions are declared here and carried
  on `TournamentFormat.notes` / `approximate=True`.
- `match.py` — `simulate_match` (90 min via `sample_score`; extra time then
  penalties for drawn knockouts).
- `league.py`, `group_stage.py`, `knockout.py` — execute the format declared in
  `rules.py`. `group_stage` returns standings, a `slot_map` (`"1A"`, `"2A"`,
  `"3A"`…), a best-third assignment, and the played matches.
- `tournament.py` / `monte_carlo.py` — `simulate_tournament_once` runs the whole
  bracket once; `monte_carlo_tournament` / `simulate_tournament` aggregate N runs
  into per-team reach probabilities for every stage plus champion and group-win.

---

## 3. Entry points

- `cli.py` — argparse subcommands: `train`, `predict-match`, `backtest`,
  `simulate-league`, `simulate-tournament`, `update-data`, `evaluate-market`.
  Default results URL is `sources.INTERNATIONAL_RESULTS_URL`; `--format` defaults
  to `world_cup_2026`.
- `apps/streamlit_app.py` — four tabs (Match Predictor, Tournament Simulator,
  Backtest Lab, Betting Edge Scanner) with a model selector. It warns loudly when
  sample/synthetic data is in use.

---

## 4. Module map

```
src/soccer_predictor/
├── data/
│   ├── schemas.py          canonical columns + helpers (single source of truth)
│   ├── aliases.py          normalize_team_name (club + international)
│   ├── normalizers.py      standardize_matches (the loader funnel)
│   ├── loaders.py          file/directory loaders
│   ├── sources.py          remote fetchers + cache + classify_competition_type
│   ├── quality.py          quality_report, duplicate/coverage checks
│   └── synthetic.py        generate_league / generate_international_history
├── features/
│   ├── feature_store.py    build_features (central, no-leakage)
│   ├── elo.py  rolling.py  form.py  rest_travel.py
│   └── market.py  xg.py  squad.py  international.py
├── models/
│   ├── base.py             BaseModel, ScorelineModel, score-matrix math
│   ├── market_baseline.py  dixon_coles.py  poisson_xg.py
│   ├── outcome_classifier.py  ensemble.py  calibration.py
├── evaluation/
│   ├── metrics.py  walk_forward.py  betting.py  reports.py
├── simulation/
│   ├── rules.py            formats, tie-breakers, WC2026 bracket
│   ├── match.py  league.py  group_stage.py  knockout.py
│   ├── tournament.py  monte_carlo.py
├── apps/streamlit_app.py
└── cli.py
```

Supporting assets live outside the package: `docs/` (this overview,
`data_sources.md`, `model_notes.md`), `data/templates/` (CSV templates and
clearly-labelled samples), and `tests/` (fast pytest suite over synthetic data,
with `test_features_no_leakage.py` as the critical guard).

---

## 5. Design invariants (carried from `CONTRACTS.md`)

1. Canonical column names only — `home_score`/`away_score`,
   `home_odds`/`draw_odds`/`away_odds`, `result` ∈ {H,D,A}, `neutral_site`.
2. No feature may use future match information (walk in date order; reveal the
   outcome only after recording features).
3. Every scoreline model implements only `_score_matrix` and inherits the rest.
4. Every simulator takes a model object and never hardcodes model logic.
5. Market odds are a benchmark first, a feature second.
6. Optional data sources fail gracefully.
7. Tournament approximations are explicit (`TournamentFormat.notes`).
8. Sample/demo data is labelled as not real.
9. Positive betting ROI is a hypothesis, not proof.
