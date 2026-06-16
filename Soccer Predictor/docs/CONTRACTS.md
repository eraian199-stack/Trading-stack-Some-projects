# Implementation Contracts (read this before writing any module)

This is the binding interface spec for the `soccer_predictor` package. The
foundation files below are ALREADY written and are authoritative — read them, do
not modify them, and import from them:

- `src/soccer_predictor/data/schemas.py` — canonical columns + helpers
- `src/soccer_predictor/data/aliases.py` — `normalize_team_name`
- `src/soccer_predictor/data/normalizers.py` — `standardize_matches`
- `src/soccer_predictor/models/base.py` — `BaseModel`, `ScorelineModel`, score math
- `src/soccer_predictor/simulation/rules.py` — formats, tie-breakers, WC2026 bracket

## Global rules (NON-NEGOTIABLE)

1. **Canonical column names only.** Scores are `home_score` / `away_score` (NOT
   home_goals). Odds are `home_odds` / `draw_odds` / `away_odds` and
   `closing_home_odds` / ... The 1X2 label column is `result` ∈ {"H","D","A"}
   (add via `schemas.add_result_column`). Neutral flag is `neutral_site` (bool).
2. **No feature may use future match information.** Build features by walking
   matches in date order; a match's own outcome is revealed only AFTER its
   features are recorded.
3. Every model exposes `fit(train_df) -> self` and
   `predict_proba(test_df) -> np.ndarray` shape `(n, 3)`, columns `[H, D, A]`.
4. Every scoreline model subclasses `ScorelineModel` and implements
   `_score_matrix(home, away, context) -> np.ndarray` (M[i,j] = P(home i, away j)).
   Do NOT reimplement `predict_proba` / `over_under` / `both_teams_to_score` —
   inherit them (this is how the "over_under ignores the line" bug is avoided).
5. Every simulator accepts a model object; it must never hardcode model logic.
6. Market odds are a benchmark first, a feature second.
7. All optional data sources must fail gracefully (informative error or skip).
8. Tournament assumptions must be explicit (carried on `TournamentFormat.notes`).
9. Sample/demo data must be labelled as not real.
10. Positive betting ROI is a hypothesis, not proof.

## Environment / compatibility

- Target Python ≥3.10. Tests run under the venv at
  `/Users/elhamraian/Personal Trading apps/.venv/bin/python` (Python 3.13,
  numpy 2.x, pandas 3.x, scipy 1.17, scikit-learn 1.9). Write
  numpy-2/pandas-3-safe code: no `pd.DataFrame.append`, no `applymap`, no
  deprecated `Series.iteritems`; use `.map`, `.itertuples`, `np.asarray`.
- Relative imports within the package, e.g. `from ..data import schemas`,
  `from .base import ScorelineModel`, `from ..models.base import predict_markets`.
- `scipy` and `scikit-learn` are core deps (assume present). `lightgbm`,
  `understatapi`, `soccerdata` are OPTIONAL — guard their imports.
- Smoke-test your modules after writing, e.g.:
  `cd "<project>" && PYTHONPATH=src "<venv>/bin/python" -c "import soccer_predictor.models.dixon_coles"`

## `context` dict (passed to score_matrix / simulators)

Keys: `neutral_site` (bool), `competition` (str), `competition_type` (str),
`date`, `knockout` (bool). Build one from a row with `base.row_context(row)`.
Scoreline models SHOULD honour `neutral_site`: when True, drop the home-advantage
term (international tournament matches are neutral). Ignore keys you don't use.

---

# Module-by-module API (implement EXACTLY these public names)

## data/loaders.py
- `load_matches(path) -> pd.DataFrame` — canonical CSV file OR directory of CSVs;
  pass-through through `standardize_matches`. Returns canonical frame w/ `result`.
- `load_football_data(path) -> pd.DataFrame` — football-data.co.uk club CSV(s).
  Map `Date→date, HomeTeam→home_team, AwayTeam→away_team, FTHG→home_score,
  FTAG→away_score`. Pick the first available odds triple from
  `[(AvgH,AvgD,AvgA),(B365H,B365D,B365A),(PSH,PSD,PSA),(BWH,BWD,BWA),(MaxH,MaxD,MaxA)]`
  → `home_odds/draw_odds/away_odds`; if closing-style cols exist map to closing_*.
  `competition_type="club_league"`. Dates are dd/mm/yy (`dayfirst=True`),
  encoding `latin-1`. Directory → concat + sort.
- `load_international_results(path) -> pd.DataFrame` — martj42 schema
  (`date,home_team,away_team,home_score,away_score,tournament,city,country,neutral`).
  Map `tournament→competition, neutral→neutral_site`; set `competition_type` via
  `classify_competition_type(competition)`.
- `load_groups(path) -> dict[str, list[str]]` — CSV with columns `group,team`;
  upper-case group, normalize team names; validate ≥2 teams per group (warn, do
  not hard-require exactly 4 — Euros etc. vary).
- `load_fixtures(path) -> pd.DataFrame` — fixtures CSV → canonical (scores may be
  blank/NaN; keep them, mark is_completed accordingly). Columns at least
  `home_team,away_team`; optional `match_id,date,group,neutral,home_score,away_score`.
- `load_odds(path) -> pd.DataFrame` — optional upcoming-fixtures-with-odds loader.

## data/sources.py
- Constants: `INTERNATIONAL_RESULTS_URL`
  (`https://raw.githubusercontent.com/martj42/international_results/master/results.csv`),
  `FOOTBALL_DATA_BASE` (`https://www.football-data.co.uk/mmz4281`),
  `FIFA_RANKING_URL` (best-effort), `CLUBELO_URL` (`http://api.clubelo.com`).
- `classify_competition_type(name: str) -> str` → one of schemas.COMPETITION_TYPES.
  ("world cup"/"euro"/"copa"/"cup of nations"... + qualifier detection; friendly;
  default "tournament" for international comps, "club_league" otherwise).
- `cached_get(url, cache_dir="data/cache", ttl_days=7) -> str` — download to a
  hashed cache file, return local path; reuse if fresh. Caches all remote data
  (re-runs must not re-fetch). Network failure → raise informative error.
- `fetch_international_results(cache=True) -> pd.DataFrame` — via cached_get +
  load_international_results.
- `fetch_football_data(league="E0", season="2324", cache=True) -> pd.DataFrame`.
- `fetch_clubelo(team_or_date, cache=True) -> pd.DataFrame` — best-effort.
  All fetchers degrade gracefully.

## data/quality.py
- `quality_report(df) -> dict` — n_matches, date_min, date_max, n_teams,
  pct_with_odds, pct_with_xg, n_duplicate_fixtures, n_missing_scores,
  competition_breakdown.
- `find_duplicate_fixtures(df) -> pd.DataFrame`, `odds_coverage(df) -> float`,
  `xg_coverage(df) -> float`.

## data/synthetic.py
- `generate_league(n_teams=20, n_seasons=6, home_advantage=0.30, vig=0.05, seed=7)
  -> pd.DataFrame` — canonical club frame WITH `home_odds/draw_odds/away_odds`,
  `home_xg/away_xg`, `competition_type="club_league"`, `neutral_site=False`,
  realistic vig'd odds from true Poisson probs (port the uploaded synthetic.py
  but to canonical column names). Must pass `schemas.validate_matches`.
- `generate_international_history(n_teams=48, n_matches=1500, seed=7) ->
  pd.DataFrame` — canonical international frame (neutral_site mostly True, a mix
  of competitions incl. "FIFA World Cup qualification" and friendlies) for
  exercising the tournament simulator end to end.

## features/elo.py
- `class EloModel` — `__init__(k=20.0, home_advantage=65.0, base_rating=1500.0,
  scale=400.0)`; `expected_home(home, away, neutral=False) -> float`;
  `update(home, away, home_score, away_score, neutral=False) -> None` (margin-of-
  victory weighted, World-Football-Elo style); `rating(team) -> float`.
  (Port uploaded elo.py; add the `neutral` arg so home advantage is dropped on
  neutral ground.)
- `goal_diff_multiplier(goal_diff: int) -> float`.

## features/rolling.py
- `rolling_form(history: deque) -> tuple[float,float,float]` → (ppg, gf, ga) means.
- `rolling_mean(history: deque, key: str) -> float`.

## features/form.py
- `points_for(home_score, away_score) -> tuple[int,int]` → (home_pts, away_pts).

## features/rest_travel.py
- `rest_days(last_played: pd.Timestamp|None, date: pd.Timestamp) -> float`.
- `haversine_km(lat1, lon1, lat2, lon2) -> float`.
- `CITY_COORDS: dict[str, tuple[float,float]]` (a small starter set; degrade to
  NaN travel when a city is unknown).

## features/market.py
- `implied_probabilities(odds_h, odds_d, odds_a) -> np.ndarray` (de-vigged, len 3).
- `add_market_features(df) -> pd.DataFrame` — adds `mkt_imp_home/draw/away` columns
  where complete odds exist, else NaN.
- `MARKET_FEATURE_COLUMNS = ["mkt_imp_home","mkt_imp_draw","mkt_imp_away"]`.

## features/xg.py
- `XG_ROLL_COLUMNS = ["home_xg_for","home_xg_against","away_xg_for","away_xg_against"]`
- rolling xG helpers consumed by feature_store when `home_xg`/`away_xg` present.

## features/squad.py
- `SQUAD_FEATURE_COLUMNS = ["squad_value_diff","injury_score_diff",
  "fifa_rank_diff"]` (compute only when source cols present; else absent).
- `add_squad_features(df) -> pd.DataFrame`.

## features/international.py
- `add_international_features(df) -> pd.DataFrame` — host flag / neutral flag
  helpers; `INTL_FEATURE_COLUMNS`.

## features/feature_store.py  ← central, no-leakage builder
- `FEATURE_COLUMNS: list[str]` — base set:
  `["elo_home","elo_away","elo_diff","home_form_pts","away_form_pts",
    "home_gf","home_ga","away_gf","away_ga","home_rest_days","away_rest_days",
    "home_played","away_played","neutral_site"]`.
- `XG_FEATURE_COLUMNS`, `MARKET_FEATURE_COLUMNS` re-exported.
- `build_features(df, form_window=6, elo_kwargs=None) -> pd.DataFrame` — walks in
  date order, attaches pre-kickoff features only; updates Elo with `neutral_site`;
  adds optional xG-rolling / market-implied / rank / squad columns when their
  source columns are present and non-null. Adds `result` if absent. NO LEAKAGE.
- `available_feature_columns(df) -> list[str]` — base + whichever optional groups
  are present in `df`.

## models/market_baseline.py
- `class MarketBaseline(BaseModel)` — `fit` no-op; `predict_proba` de-vigs
  `home_odds/draw_odds/away_odds`. Rows lacking complete odds → row of `np.nan`
  (callers filter with `schemas.rows_with_complete_odds`). Never silently invent
  odds.

## models/dixon_coles.py
- `class DixonColes(ScorelineModel)` — `__init__(xi=0.0018, max_goals=10)`; fit by
  MLE (scipy SLSQP) with exponential time-decay weights and the DC low-score rho;
  store `attack`, `defence`, `home_adv`, `rho`, team index. `_score_matrix` uses
  `score_matrix_from_rates(lam, mu, max_goals, rho)`; honour `context["neutral_site"]`
  (drop home_adv). Unseen teams → league-average (attack=defence=0). Uses
  `home_score/away_score`. Inherit all market methods.

## models/poisson_xg.py
- `class PoissonXG(ScorelineModel)` (alias `XGPoisson = PoissonXG`) —
  `__init__(max_goals=10, shrink=6.0)`; fit team attack/defence from rolling
  `home_xg/away_xg`, shrink toward league mean; `_score_matrix` via
  `score_matrix_from_rates`; honour neutral_site. `fit` raises informative error
  if xG columns absent/empty. Unseen teams → strength 1.0.

## models/outcome_classifier.py
- `class OutcomeClassifier(BaseModel)` — `__init__(kind="gbm", build_feats=True,
  form_window=6)`. Uses `feature_store.build_features` +
  `available_feature_columns`. `kind="logreg"` → sklearn LogisticRegression with
  median-impute + standardize; `kind="gbm"` → LightGBM if importable else
  `HistGradientBoostingClassifier`. `y` from `result` mapped to {H:0,D:1,A:2}.
  `predict_proba` returns columns aligned to [H,D,A]. Expose `self.backend`.

## models/ensemble.py
- `class Ensemble(BaseModel)` — `__init__(components: list[tuple[BaseModel,float]])`;
  `fit` fits all; `predict_proba` = weighted average, renormalising weights over
  the components that produced a non-NaN row (so missing market odds redistribute
  that weight). 
- `default_ensemble(form_window=6) -> Ensemble` with weights
  DixonColes 0.35, OutcomeClassifier 0.25, MarketBaseline 0.25, EloGoalsModel 0.15
  (see below). If you need an Elo-based 1X2/scoreline model, ALSO add it here:
- `class EloGoalsModel(ScorelineModel)` — fit an internal `EloModel` over the
  training frame in date order; `_score_matrix` converts the Elo win expectation
  into a 1X2 target and a goal-sum estimate, builds a Poisson matrix, and tilts
  it to the target via `base.tilt_matrix_to_outcome`. Honour neutral_site. This
  is the robust minimum-viable scoreline model and the DEFAULT model the
  tournament simulator uses when nothing richer is supplied. Put it here or in
  its own `models/elo_goals.py` and export it.

## models/calibration.py
- `class CalibratedModel(BaseModel)` — `__init__(base: BaseModel, method="auto",
  holdout_frac=0.2)`. `fit` fits base on the earlier portion, fits a calibrator
  on base predictions over the later (time-ordered) holdout. `method`: "isotonic"
  (per-class one-vs-rest IsotonicRegression, renormalised) when holdout is large
  enough, else "temperature" (single scalar T minimising log loss). "auto" picks.
  `predict_proba` applies calibration then renormalises.
- `temperature_scale(probs, T) -> np.ndarray`, `fit_temperature(probs, outcomes)
  -> float` helpers.

## evaluation/metrics.py
- `log_loss(probs, outcomes)`, `rps(probs, outcomes)`, `brier(probs, outcomes)`,
  `accuracy(probs, outcomes)`, `expected_calibration_error(probs, outcomes,
  n_bins=10)`, `all_metrics(probs, outcomes) -> dict` (keys: log_loss, rps, brier,
  accuracy, ece, n). `outcomes` is an array of "H"/"D"/"A". Order H<D<A for RPS.

## evaluation/walk_forward.py
- `walk_forward(model_factory, df, min_train_frac=0.5, n_folds=6) ->
  (probs, outcomes, test_df)` — expanding window, sorted by date, uses `result`.
  model_factory is a zero-arg callable → fresh model.
- `walk_forward_goals_markets(model_factory, df, min_train_frac=0.5, n_folds=6,
  ou_line=2.5) -> dict` — over/under + BTTS log_loss/brier (uses
  home_score/away_score). Model must expose over_under / both_teams_to_score.
- `evaluate_model(model_factory, df, **kw) -> dict` — walk_forward + all_metrics.
- `compare_to_market(model_factory, df, **kw) -> dict` — model vs MarketBaseline
  on the SAME rows that have complete odds; returns {model, market,
  model_minus_market_log_loss}.

## evaluation/betting.py
- `betting_backtest(probs, test_df, edge_threshold=0.05, stake=1.0,
  kelly_fraction=0.0, best_edge_only=True) -> dict` — value betting vs the book.
  **best_edge_only=True places at most ONE bet per match (highest-EV qualifying
  outcome).** Returns n_bets, turnover, profit, roi, hit_rate, max_drawdown.
  Uses home_odds/draw_odds/away_odds + result.
- `closing_line_value(probs, test_df, edge_threshold=0.05) -> dict` — CLV vs
  `closing_*` odds when present (mean log CLV / % beat the close).
- `max_drawdown(pnl_series) -> float`.

## evaluation/reports.py
- `leaderboard(results: dict[str,dict]) -> pd.DataFrame` sorted by log_loss.
- `format_metrics(m: dict) -> str`.
- `calibration_table(probs, outcomes, n_bins=10) -> pd.DataFrame`.

## simulation/match.py
- `@dataclass SimulatedMatch` — fields: `match_id:str="", home_team, away_team,
  home_score:int, away_score:int, winner:str, stage:str="", extra_time:bool=False,
  penalties:bool=False`.
- `simulate_match(model, home, away, rng, context=None, knockout=False,
  tie=None) -> SimulatedMatch` — 90-min via `model.sample_score`; if knockout and
  drawn and `tie.extra_time`: add ET goals ~ Poisson(expected_goals * tie.et_fraction)
  per side; if still level and `tie.penalties`: winner via
  `model.win_probability_no_draw`. `tie` defaults to `rules.KnockoutTie()`.

## simulation/league.py
- `simulate_league(model, teams, rng, double_round_robin=True, fixtures=None,
  tiebreakers=rules.LEAGUE_TIEBREAKERS, context=None) -> list[dict]` — ranked
  standings rows (rules.new_standings_row schema + `rank`). Use provided fixtures'
  actual scores when present, else sample.

## simulation/group_stage.py
- `simulate_group_stage(model, groups, rng, fixtures=None,
  tiebreakers=rules.FIFA_GROUP_TIEBREAKERS, advance_per_group=2, n_best_third=0,
  neutral=True) -> tuple[standings, slot_map, third_assignment, matches]` where
  `standings: dict[group, list[rankedrow]]`, `slot_map: dict[str,str]` with keys
  like "1A","2A","3A", `third_assignment: dict[match_id, group_letter]` (best 8
  thirds via `rules.wc2026_third_place_assignment` when n_best_third>0 and the
  WC2026 third rules apply, else {}), `matches: list[SimulatedMatch]`. Build the
  within-group match list and pass it to `rules.rank_group`. Round-robin fixtures
  default via `itertools.combinations`.

## simulation/knockout.py
- `simulate_knockout(model, fmt, slot_map, third_assignment, rng) ->
  tuple[reached, matches]` where `fmt: rules.TournamentFormat`, `reached:
  dict[stage, list[team]]` including a final `"champion"` key, `matches:
  list[SimulatedMatch]`. Resolve `BracketMatch.home_src/away_src`: group slots
  ("1A"/"2A") via slot_map; best-third ("3A/B/..") via third_assignment[mid] →
  slot_map[f"3{g}"]; prior match ids ("M..") via stored winners; seed codes
  ("S#") via slot_map. Use `simulate_match(..., knockout=True, tie=fmt.knockout)`.
- `simulate_single_elimination(model, seeded_teams, rng, tie=None) ->
  tuple[reached, matches]` — convenience for a pure knockout cup.

## simulation/tournament.py
- `simulate_tournament_once(model, fmt, groups, rng, fixtures=None) -> dict`
  with keys `standings, slot_map, third_assignment, reached, matches`.
- `simulate_tournament(model, groups, fixtures=None, n_simulations=5000, seed=7,
  fmt=None) -> pd.DataFrame` — defaults `fmt = rules.world_cup_2026_format()`;
  delegates to monte_carlo. Returns df with `team, group` and
  `<stage>_probability` for every stage in fmt.stage_order PLUS
  `champion_probability`, plus `win_group_probability`. Sorted by
  champion_probability desc. (Keep this signature — cli/app/tests depend on it.)

## simulation/monte_carlo.py
- `monte_carlo_tournament(model, fmt, groups, fixtures=None, n_simulations=5000,
  seed=7, progress=None) -> pd.DataFrame` — runs simulate_tournament_once N times,
  aggregates per-team reach probabilities for every stage + champion + group win.
- `monte_carlo_league(model, teams, n_simulations=2000, seed=7, **kw) ->
  pd.DataFrame` — title / top-N / position probabilities per team.

## cli.py
`main()` with argparse subcommands (exact names):
`train, predict-match, backtest, simulate-league, simulate-tournament,
update-data, evaluate-market`. Mirror the examples in the framework spec. Default
results URL = sources.INTERNATIONAL_RESULTS_URL. `--format` defaults to
`world_cup_2026`.

## apps/streamlit_app.py
Four tabs: **Match Predictor**, **Tournament Simulator**, **Backtest Lab**,
**Betting Edge Scanner** (see framework spec). Use plotly. Cache data load / model
fit. Warn loudly when sample/synthetic data is in use. A model-selection control
(Elo / Dixon-Coles / xG / Ensemble). The Betting tab shows model fair odds vs
market odds vs edge + a calibration/confidence warning.

## tests/ (pytest, fast — use synthetic data, tiny sim counts)
- `test_data.py` — schema validate; loaders on synthetic + sample CSVs; dedupe;
  result column correctness.
- `test_features_no_leakage.py` — **the critical one**: for some match i,
  `build_features` row equals features computed from rows `< i` only; appending
  FUTURE rows then rebuilding must not change row i's features. Also assert
  `home_played`/elo monotonic sanity.
- `test_models.py` — predict_proba rows sum to 1; `score_matrix` sums to 1 and is
  square; DixonColes recovers strong>weak ordering on synthetic; MarketBaseline
  de-vig sums to 1; **over_under respects the line** (P(over 0.5) > P(over 4.5));
  CalibratedModel preserves shape & sums.
- `test_simulation.py` — WC2026 sim returns 48 teams, champion probs sum≈1, every
  probability in [0,1]; group standings have right team counts; single-elim works;
  knockout always yields a champion (no draws survive).
- `test_evaluation.py` — perfect probabilities → ~0 log loss; rps≤? sanity;
  walk_forward shapes align; betting best_edge_only places ≤1 bet/match.

## docs/
- `framework.md` — architecture overview (this design).
- `data_sources.md` — free/paid sources, club vs international, the 2026 FBref xG
  removal note, caching.
- `model_notes.md` — model stack, ensemble weights, calibration, tournament
  approximations (Annex C third place), betting caveats.
