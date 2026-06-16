# Model Notes

The modelling rationale behind `soccer_predictor`: the model stack, how the
ensemble is weighted, calibration, the walk-forward evaluation protocol, why we
score with RPS and ECE, the World Cup 2026 third-place approximation, and the
betting caveats. For the architecture see [`framework.md`](framework.md); for the
binding interfaces see [`CONTRACTS.md`](CONTRACTS.md). Foundation maths lives in
`models/base.py`; evaluation in `evaluation/`.

A guiding principle throughout, carried from hard lessons on adjacent projects: a
metric that looks great on a single in-sample split is usually overfit. Trust
**out-of-sample, time-ordered** numbers, prefer a few real edges combined over
many tuned sleeves, and treat a positive betting ROI as a hypothesis to be
falsified, not a result.

---

## 1. The model stack

Every model satisfies one of two contracts from `models/base.py`:

- `BaseModel` — `fit(train_df) -> self`, `predict_proba(test_df) -> (n, 3)` over
  columns `[H, D, A]`.
- `ScorelineModel(BaseModel)` — additionally a full score matrix
  `M[i, j] = P(home scores i, away scores j)`. **A subclass implements only
  `_score_matrix`; 1X2, over/under, BTTS, expected goals, sampling, and penalty
  resolution are all derived in the base.** This is the central anti-bug design:
  every market is collapsed from the one matrix (`collapse_1x2`,
  `markets_from_matrix`), so a model cannot, for example, return an over/under
  number that ignores the line.

Concrete models, roughly in order of richness:

1. **`MarketBaseline`** (`models/market_baseline.py`) — de-vigs
   `home_odds/draw_odds/away_odds`. It is the **benchmark first, a feature
   second**: the bookmaker's closing line is a very strong probability estimate
   and any model worth deploying must beat it out of sample. Rows without complete
   odds return NaN (callers filter with `schemas.rows_with_complete_odds`); it
   never invents odds.

2. **`EloGoalsModel`** (`models/ensemble.py`) — fits an internal World-Football
   Elo over the training frame in date order, converts the win expectation into a
   1X2 target and a goal-sum estimate, builds a Poisson score matrix, and tilts it
   to the target via `base.tilt_matrix_to_outcome`. This is the **robust
   minimum-viable scoreline model and the default the tournament simulator uses**
   when nothing richer is supplied — dependency-light and stable on sparse
   international data.

3. **`DixonColes`** (`models/dixon_coles.py`) — the classic 1997 model: per-team
   attack/defence + home advantage fit by maximum likelihood (scipy SLSQP) with
   exponential time-decay weights (`xi`) and the low-score ρ correction that fixes
   independent Poisson's under-rating of 0-0/1-0/0-1/1-1. Unseen teams fall back
   to league average. A strong scoreline model when there is enough match history.

4. **`PoissonXG`** (`models/poisson_xg.py`, alias `XGPoisson`) — team
   attack/defence estimated from rolling `home_xg`/`away_xg`, shrunk toward the
   league mean. xG is a less noisy signal of underlying quality than goals, so
   this is often the sharpest model **where xG exists** — mostly club football
   (see the FBref note in [`data_sources.md`](data_sources.md)). `fit` raises an
   informative error when xG columns are absent.

5. **`OutcomeClassifier`** (`models/outcome_classifier.py`) — a discriminative
   1X2 classifier over the engineered features from `feature_store`. Uses
   **LightGBM if importable, else sklearn `HistGradientBoostingClassifier`** (or
   logistic regression with median-impute + standardize for `kind="logreg"`). It
   captures interactions (form × rest × rank) the generative models cannot, but
   needs the most data and is the most prone to overfitting — hence the
   walk-forward discipline below.

6. **`Ensemble`** (`models/ensemble.py`) — a weighted average of the above,
   renormalising weights over the components that produced a non-NaN row (so when
   market odds are missing, that weight is redistributed rather than dragging the
   blend to NaN).

All scoreline models honour `context["neutral_site"]`: on neutral ground the
home-advantage term is dropped. Most World Cup matches are neutral, so this
matters for the tournament target.

---

## 2. Ensemble weights

`default_ensemble(form_window=6)` blends:

| Component | Weight | Why |
|---|---|---|
| `DixonColes` | 0.35 | Sharpest scoreline model with enough history; biggest single share. |
| `OutcomeClassifier` | 0.25 | Adds feature interactions the generative models miss. |
| `MarketBaseline` | 0.25 | Anchors to the book — the prior that is hard to beat. |
| `EloGoalsModel` | 0.15 | Robust fallback; stabilises the blend on sparse data. |

Rationale: no single model dominates across regimes (rich club data with odds vs
sparse neutral international data). The blend leans on Dixon-Coles where history
is thick, defers to the market where odds exist, and the Elo model keeps it sane
when both are thin. **Weights redistribute** when a component returns NaN for a
row, so a fixture with no odds still gets a coherent prediction from the
remaining three. These weights are sensible defaults, not the product of an
exhaustive search — re-fitting them is a known lever, but a small blend of real,
independent edges beats over-tuning the mix.

---

## 3. Calibration

Good *ranking* is not the same as good *probabilities*, and tournament
simulation and betting both need probabilities that mean what they say.
`CalibratedModel` (`models/calibration.py`) wraps any model:

- It splits the training frame **in time order**, fits the base model on the
  earlier portion, and fits a calibrator on the base model's predictions over the
  later holdout (default 20%). Calibrating on a time-ordered holdout — never the
  training rows — is what keeps the calibration honest.
- `method="isotonic"` — per-class one-vs-rest isotonic regression (renormalised),
  used when the holdout is large enough to support it.
- `method="temperature"` — a single scalar temperature `T` minimising log loss,
  used for small holdouts (helpers `temperature_scale`, `fit_temperature`).
  Temperature scaling is low-variance and a safe default when data is scarce.
- `method="auto"` picks between them by holdout size.

`predict_proba` applies the calibrator and renormalises so rows still sum to 1.
ECE (below) is how we check it worked.

---

## 4. Walk-forward evaluation

In-sample fit means nothing for forecasting. Everything is judged out of sample,
in time order, by `evaluation/walk_forward.py`:

- `walk_forward(model_factory, df, min_train_frac=0.5, n_folds=6)` — an
  **expanding window**: sort by date, train on the past, predict the next fold,
  expand, repeat. `model_factory` is a zero-arg callable returning a *fresh*
  model each fold, so no information leaks across folds.
- `walk_forward_goals_markets(...)` — the same protocol for over/under and BTTS.
- `evaluate_model(...)` — walk-forward + `all_metrics`.
- `compare_to_market(...)` — the decisive test: model vs `MarketBaseline` on
  **the same rows that have complete odds**. Beating the market out of sample is
  the bar; matching it is already respectable.

This pairs with the no-leakage feature builder (`feature_store.build_features`
walks in date order and reveals a match outcome only after recording its
features) and is guarded by `tests/test_features_no_leakage.py`. The combination
is what makes the metrics trustworthy rather than the usual phantom-Sharpe trap.

---

## 5. Why RPS and ECE

Accuracy is too crude for 1X2 forecasting — it ignores how confident a prediction
was and treats a near-miss like a blowout. The metrics that matter
(`evaluation/metrics.py`, `all_metrics` returns log_loss, rps, brier, accuracy,
ece, n):

- **Log loss** — proper scoring rule; punishes confident wrong calls hard. The
  headline calibration-plus-sharpness number and the leaderboard sort key.
- **RPS (Ranked Probability Score)** — the right metric for *ordered* outcomes.
  Home < Draw < Away is an ordinal scale: predicting Away when the result was Home
  should cost more than predicting Draw. RPS rewards getting the *distribution*
  right, not just the mode, which is exactly what a tournament simulator consumes.
  (This is why `RESULT_CLASSES` is fixed in H, D, A order.)
- **Brier** — a second proper scoring rule, easy to decompose into
  reliability/resolution; a cross-check on log loss.
- **ECE (Expected Calibration Error)** — bins predictions by confidence and
  measures the gap between predicted probability and observed frequency. It
  answers "when the model says 60%, does it happen 60% of the time?" — the
  question calibration (§3) targets and simulation/betting depend on.

We do **not** optimise for accuracy. A model can be more accurate yet worse
calibrated, and miscalibrated probabilities ruin both the tournament odds and any
betting edge.

---

## 6. Tournament simulation and the WC2026 Annex C third-place approximation

The simulator (`simulation/`) Monte-Carlos the declared format
(`rules.world_cup_2026_format()`): simulate each group's round-robin, rank with
`FIFA_GROUP_TIEBREAKERS`, take the top two per group plus the **eight best
third-placed teams**, then play the Round-of-32 → final bracket many times to
estimate each team's probability of reaching every stage and winning.

Two assumptions are approximations and are made **explicit** on
`TournamentFormat.notes` / `approximate=True` (a project non-negotiable):

- **Group tie-breakers.** `rules.FIFA_GROUP_TIEBREAKERS` applies points → goal
  difference → goals for → head-to-head points → head-to-head goal difference →
  fair play → random draw. The full FIFA drawing-of-lots procedure is approximated
  by the random-draw tail.

- **Third-place allocation (FIFA Annex C).** Which Round-of-32 slot each of the
  eight best thirds is assigned to depends on *which* group letters those thirds
  come from. FIFA publishes a fixed lookup table (Annex C) covering all
  C(12,8) = 495 combinations. `rules.wc2026_third_place_assignment` does **not**
  reproduce that table exactly. Instead it solves the constraint-satisfaction
  problem: each Round-of-32 best-third slot lists its eligible groups
  (`wc2026_third_slot_rules`, e.g. slot M74 accepts a third from A/B/C/D/F), and a
  backtracking search finds a valid assignment where every assigned group is
  eligible for its slot and each qualifying group is used once. **It is correct in
  that the bracket is always legal and feasible; it is approximate in that for a
  given set of qualifying groups the specific group→slot mapping may differ from
  FIFA's published one.** Over a Monte Carlo run this shifts *who plays whom* in
  some scenarios but leaves the qualifying set and overall reach probabilities
  close. Supplying the exact 495-row Annex C matrix is the clean upgrade and would
  remove this caveat.

The simulator consumes models only through the `ScorelineModel` interface
(`sample_score`, `expected_goals`, `win_probability_no_draw`), so any model — Elo
default through full ensemble — can drive it.

---

## 7. Betting caveats — positive ROI is a hypothesis

`evaluation/betting.py` provides a value-betting backtest (`betting_backtest`)
and closing-line-value analysis (`closing_line_value`). Read them with
discipline:

- **Positive ROI is a hypothesis, not proof.** A green backtest is the start of an
  investigation, not a conclusion. On small samples ROI is dominated by variance,
  and the usual failure mode is a number that looks good on every metric — which
  is itself an overfit signature.
- **Beat the *closing* line, not just the opening one.** Closing-line value is the
  most reliable forward-looking edge indicator: if your bets do not consistently
  beat the closing price, a long-run profit is unlikely regardless of historical
  ROI. The closing line already embeds the market's best information.
- **One bet per match.** `betting_backtest(..., best_edge_only=True)` places at
  most one bet per fixture (the highest-EV qualifying outcome), avoiding the
  illusion of edge from stacking correlated bets on the same game.
- **Vig and execution are real.** The book's margin, line movement between your
  estimate and the bet, limits, and stake sizing all erode a paper edge. The
  de-vigged market probability is a *very* hard baseline to beat.
- **Calibration first.** Edge is computed against the model's probabilities, so
  miscalibrated probabilities manufacture phantom edges. Confirm low ECE (§5) and
  out-of-sample log loss at or below the market (§4) **before** trusting any
  betting number. The Streamlit Betting Edge Scanner therefore shows fair odds vs
  market odds vs edge **alongside a calibration/confidence warning**.

Net: this stack is a calibrated forecasting and simulation tool. It can surface
candidate value bets, but any claim of a real, deployable betting edge requires
out-of-sample evidence — beating the market on log loss and beating the closing
line — that this repository deliberately does not assume.

---

## Adversarial review outcomes (2026-06-15) and the ML decision

A 24-agent adversarial review (find → reproduce → verify → synthesize) ran over
the whole package. Headline: **no data leakage, no critical defects, and the
non-ML core is trustworthy.** 16 confirmed findings were fixed; the ones that
matter for model trust:

- **Walk-forward feature reset (was invalidating every classifier/ensemble
  backtest).** `walk_forward` now builds leakage-safe features ONCE over the full
  ordered frame before slicing, so the classifier sees correctly-accumulated
  Elo/form/rest at predict time (still backward-only — no leakage). The skew had
  been *conservative* (made ML look worse, not better), so no inflated metric was
  hiding.
- **EloGoalsModel miscalibration → fake betting edges.** The old mapping carved a
  draw mass out of the Elo expectation, so on mismatches the draw and the
  underdog win collapsed toward zero and the model emitted enormous spurious
  "edges". It now maps the Elo expectation to a goal *supremacy* and reads 1X2 off
  a Dixon-Coles score matrix, so draws and longshots stay realistically floored.
- **Betting-edge safety.** The edge scanner (CLI `edges` + the app) now: matches
  team names through the fixed `&`→`and` alias (a silent mismatch used to degrade
  a known team to a generic 1500 Elo), flags any pick on a team the model never
  saw, and flags implausible model-vs-market divergence. `value=True` requires the
  edge clear the threshold AND not be flagged.

### What is worth keeping (the trustworthy core, before any ML)

1. **Canonical data schema + loaders** — consistent, normalizer applied to every
   feed; World Cup groups are double-verified (fixture reconstruction ∩ Wikipedia).
2. **Dixon-Coles scoreline model + the score-matrix math** — correct and the
   recommended backbone; the EloGoals fix routes through this same machinery.
3. **The simulation engine** — group ranking, third-place allocation, knockout
   (now incl. two-leg ties) are correct on the shipped paths.
4. **Market baseline + de-vig + EV arithmetic** — correct; treat de-vigged
   closing odds as the benchmark every model must beat.
5. **Walk-forward harness + proper scoring (log loss / RPS / Brier / ECE)** — the
   methodology is sound; it is the honest way to judge any addition.

### Recommendation on ML (the OutcomeClassifier)

Treat ML as **additive and unproven, not a starting point**. With the walk-forward
feature bug fixed you can now evaluate it honestly — but only deploy it if, after
calibration, it **beats both the bare Dixon-Coles model and the market baseline
out of sample** on log loss. If it does not clear that bar, leave it out: the
DC + market core is already the trustworthy, deployable part. Before any model
(ML or Elo) is allowed to flag a live `value` bet, wrap it in `CalibratedModel`
and keep the known-team / divergence guards on. Positive ROI remains a hypothesis
until it beats the closing line out of sample.

---

## National-team variables: what's in, and what the backtest says (2026-06)

The national-team model (`EloGoalsModel`) is a margin-weighted World-Football-Elo
turned into a Dixon-Coles score matrix. Elo already blends **long-run pedigree
with recent results**, so "is form / pedigree taken into account?" is partly
yes-by-construction. On top of it, several variables are available as **optional,
default-OFF, backtest-tunable** components (effective rating, in Elo points):

| variable | knob | what it adds |
|---|---|---|
| recent form / "run-in" | `form_weight` | trailing-window, importance- & time-decayed over/under-performance vs Elo expectation |
| historical pedigree | `pedigree_weight` | slow-Elo minus fast-Elo (mean-reversion to long-run level) |
| match importance | `use_importance` | K-factor weighted by competition (WC > qualifier > friendly) |
| player/squad strength | `squad_strength`,`squad_weight` | external per-team rating (FotMob / Transfermarkt / FIFA pts), **current-only** |
| market line | `MarketAnchoredModel` | blend toward de-vigged live odds on priced fixtures |

### What the out-of-sample backtest found (332 matches, World Cups 2006–2022)

Match-level proper scoring, each edition predicted from data strictly before it
(`soccer-predictor backtest-world-cups`):

| variant | log loss | RPS |
|---|---|---|
| **plain margin-weighted Elo** | **0.967** | **0.396** |
| + recent form | 0.978–0.981 | 0.40 |
| + importance weighting | 0.983 | 0.407 |
| + pedigree | 0.994 | 0.416 |

**None of the add-ons improved out-of-sample World Cup prediction.** Recent form
slightly *hurt*; pedigree hurt most — confirming the prior that historical
pedigree is weak (Elo already captures team level, and explicit form/pedigree just
add noise). Differences are small but consistent across every metric, so the
**parsimonious default is plain Elo** with all components off. They remain
available for experimentation and for other competitions, but the WC evidence does
not support switching them on.

### What is NOT backtestable here
- **Player/squad strength** — no free historical squad data, so it cannot be
  scored on past World Cups; it is a live-only overlay (supply a CSV; see
  `data/players.py`).
- **The market** — historical closing odds are not free, so the market anchor /
  baseline cannot be backtested on past World Cups. Live, the de-vigged market is
  still the strongest single signal and the bar any model must beat; use
  `MarketAnchoredModel` (or `world-cup --anchor`) for the current tournament and
  judge edges only after calibration + closing-line value.

---

## Squad strength: market VALUE vs player ABILITY (2026-06)

Two different problems, often conflated:

- **Market value** (Transfermarkt) is *transfer economics* — heavily **age-discounted**
  and position-skewed (an elite 34-year-old is cheap; an elite goalkeeper is valued
  far below an elite winger). It answers "what is the squad worth", not "how good
  are the players".
- **Ability** is what we actually want. There is no clean, free, position-aware
  ability number, so we approximate.

`fetch-ratings` therefore offers two metrics:

| `--metric` | what it is | use |
|---|---|---|
| `value` | raw Transfermarkt squad market value | quick, but age/position-biased |
| `ability` (default) | value **de-aged** via an age→value-retention curve | closer to current player quality |

The `ability` metric divides each player's value by an age-retention factor
(peak ≈ 24–28; veterans worth a fraction of their ability; youngsters carry a
potential premium), recovering an ability-equivalent value per player and summing
the squad. This directly fixes the veteran under-valuation.

**League of play (tiered).** `fetch-ratings --metric league` computes a squad's
value-weighted mean **league tier** — NOT binary top-5. `LEAGUE_TIERS` rates the
top-5 at 1.0, the strong non-top-5 leagues you flagged clearly above the tail
(Eredivisie / Primeira / Brazil Série A ≈ 0.80; Belgian / Championship / Saudi ≈
0.74; Argentine / Süper Lig ≈ 0.72; MLS / Scottish / Austrian / Swiss ≈ 0.62–0.66),
and the long tail at 0.45. Each player's CLUB is parsed from the squad page and
mapped via a curated, dated `CLUB_LEAGUE` map; the output carries a
`club_match_rate` so you see coverage (≈0.8–0.95 for the big teams, lower for
minnows whose clubs aren't in the map). Note this **overlaps market value** (a
top-5 player is already valued higher), so use it as a lens, not an independent
add-on.

**What it still does NOT capture (be honest):**
- **Position-specific quality.** Value (even de-aged) structurally under-weights
  goalkeepers and defenders. Genuinely hard without per-position ability ratings.
- **Long-tail club coverage.** The curated `CLUB_LEAGUE` map covers the well-known
  clubs; obscure clubs fall back to the default tier (watch `club_match_rate`).
- It is **CURRENT-only** (no free historical squads → not backtestable) and it
  **overlaps the market anchor**, which already prices player ability. So treat the
  squad overlay as a minor tilt, not a core signal — the market anchor is the
  better "how good are the players" lever for the live tournament.

Any better ability source (FotMob / SofaScore season ratings, a hand-built top-5
count, your own scouting numbers) drops straight in: export a `team,strength` CSV
and pass `world-cup --squad-csv`. The model consumes it the same way.

---

## The ML suite — built, and judged honestly (2026-06)

The ML layer is a **StackedEnsemble** (`models/stacking.py`): a super-learner that
trains a regularised meta-model on **time-series out-of-fold** predictions from the
base models (Dixon-Coles, Elo, market, GBM). OOF means each training row's base
features come from models fit only on *earlier* rows — leakage-safe by
construction. It is evaluated with `evaluation/ml_report.py` (CLI `ml-report`),
which adds a no-skill floor, an **overfit gap** (OOS − in-sample log loss), and a
**beat-the-market** verdict (bootstrap 95% CI on the per-match log-loss delta).

### Result on 2280 real EPL matches (with closing odds), 6 walk-forward folds

| model | log loss | ECE | vs market | beats market? | overfit gap |
|---|---|---|---|---|---|
| **market** (benchmark) | **0.937** | 0.018 | — | — | — |
| **stack** | **0.943** | **0.013** | +0.006 | No (CI [.001, .012]) | **−0.036** |
| logreg | 0.954 | 0.015 | +0.017 | No | +0.009 |
| ensemble | 0.955 | 0.017 | +0.018 | No | +0.30 |
| elo | 0.964 | 0.021 | +0.027 | No | −0.007 |
| dixon-coles | 0.967 | 0.008 | +0.030 | No | +0.008 |
| gbm | 1.239 | 0.135 | +0.30 | No | **+1.10** |

### What this says (honestly)

1. **Nothing beats the closing line out of sample.** Every model's `vs_market` is
   positive (worse than the book) and `beats_market` is False for all. The
   de-vigged market is the ceiling; there is **no free edge over the book** on the
   EPL 1X2 market. Treat any future "edge" as a bug until proven on closing prices.
2. **The raw GBM is overfit garbage** — log loss 1.24 (worse than no-skill 1.06!),
   an overfit gap of +1.10, and terrible calibration (ECE 0.135). Do **not** use
   the standalone gradient-booster. This is the failure mode to fear.
3. **The disciplined stack is the best free model** and is honest: it ≈ matches the
   market (within ~0.006 log loss), is the **best-calibrated** model (ECE 0.013),
   and has a **negative** overfit gap (no memorising). It achieves this by learning
   to lean on the market (~0.5 weight) and down-weight the GBM (~0.1) — i.e. it
   neutralises the overfitter. Its near-market score comes *from* including the
   market, so it does not find edge *over* the market; its value is being the best
   calibrated estimator **when the market is absent** (unpriced fixtures, the
   tournament simulator) or as a blend.
4. The **default fixed-weight ensemble overfits** (gap +0.30) because it carries
   the GBM at 0.25. Prefer the stack (or drop the GBM from the ensemble).

### Recommendation

Use the **stack as the research/estimation model** (best calibrated free
probabilities), keep the **market anchor** as the live signal, and **never deploy
the standalone GBM**. For internationals/World Cup there are no free historical
odds, so ML can't be validated against the market there and ties plain Elo — Elo
remains the WC engine. Run `soccer-predictor ml-report --league E0 --seasons ...`
to reproduce this table on any league.

---

## Ablation: does every variable earn its place? (2026-06)

The rule: a variable/hyperparameter stays ON only if it improves OUT-OF-SAMPLE
predictions **consistently across regimes** — not just on one backtest slice
(tuning to one slice is overfitting). Tested via `evaluation/ablation.py` (CLI
`ablation`, runner `tools/run_club_ablation.py`) on the World Cup match-level
backtest (1998–2022, 460 matches) AND clubs (EPL / La Liga / Bundesliga, ~2k
matches each, with closing odds).

### National-team model (World Cup backtest)

| variable | WC Δlog-loss | clubs | verdict |
|---|---|---|---|
| Elo **K = 30** (vs 20) | **−0.0056** (monotonic, better ECE) | neutral EPL, **+0.005 La Liga** | **keep K=20** — WC-only, doesn't generalise |
| supremacy **slope = 0.9** (vs 0.75) | −0.0049 | −0.001 EPL, +0.002 La Liga | **keep 0.75** — marginal/mixed |
| recent form `form_weight=150` | −0.0018 | (intl-only knob) | keep **off** — marginal, WC-only, peaked (250+ hurts) |
| match **importance** weighting | +0.016 | — | keep **off** — hurts |
| historical **pedigree** | +0.010 | — | keep **off** — hurts (your hunch, confirmed) |
| home advantage = 0 | +0.003 | +0.015–0.021 | keep **65** — dropping it hurts everywhere |
| ρ = 0 (no DC low-score) | ≈0 | ≈0 | keep small negative ρ |

**The K=30 trap is the headline lesson:** it was the single biggest WC-backtest
win (−0.0056, monotonic, better calibration) — and it **hurts La Liga**. Enshrining
it would have been textbook backtest-overfitting. The cross-regime gate caught it.
**No defaults changed.** A WC-tuned config (`EloGoalsModel(elo_kwargs={"k":30},
supremacy_slope=0.9, form_weight=150)`) is available, but it is tuned to past World
Cups and should not be trusted to transfer to 2026 — the robust defaults are safer.

### Club classifier feature groups (EPL, clean single-group drops)

| dropped group | Δlog-loss | reading |
|---|---|---|
| market-implied | **+0.019** | the dominant feature — keep |
| form (rolling pts/GF/GA) | **−0.005** | **adds nothing** (the market/Elo already price form) |
| rest days | **−0.005** | **adds nothing** |
| xG | 0.000 | absent in football-data EPL (needs an Understat merge) |
| everything but Elo | +0.002 | Elo + market is the core |

So even among the classifier's features, **Elo + market do the work**; rolling
form and rest are redundant (mildly harmful) once those are present. They stay in
the default feature set (they're the only signal for internationals, where there
is no market) but are prunable via `OutcomeClassifier(feature_cols=...)` — i.e.
**optional, as the "make it optional if it doesn't help" rule requires.**

### Cross-cutting (every league + the World Cup)

- **Nothing beats the de-vigged market out of sample** — in WC or in any of EPL,
  La Liga, Bundesliga, Serie A, Ligue 1. The market is the ceiling.
- **The raw GBM overfits everywhere** (overfit gap ≈ +1.1 in every league). Never
  deploy it standalone; the stack neutralises it.
- The robust defaults (K=20, slope=0.75, home-adv=65, all national-team components
  off) **survive the ablation** — every "improvement" was regime-specific noise.

Bottom line: the variables that genuinely earn their place are **Elo (the
backbone) and the market** (where it exists). Everything else is either optional
(and off by default for good reason) or can't be backtested (squad/anchor). Re-run
this on any competition with `soccer-predictor ablation` before trusting a tweak.
