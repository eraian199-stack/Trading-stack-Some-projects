# FIFA World Cup 2026 — live predictor

The tournament (11 June – 19 July 2026, USA / Canada / Mexico) is underway, so the
predictor runs on **real data** and conditions on games already played.

## One command

```bash
PYTHONPATH=src python -m soccer_predictor.cli world-cup --model elo --sims 20000 \
  --out outputs/world_cup_2026_probabilities.csv
```

or in Python:

```python
import soccer_predictor as sp
sims = sp.world_cup.simulate(model_name="elo", n_simulations=20000)
print(sims.sort_values("champion_probability", ascending=False).head(12))
```

or the app: `streamlit run app_unified.py` → **🏆 World Cup 2026 (live)** tab.

## How the real data is assembled (no fabrication)

Group **composition** and **labels** are established by two independent methods
that are cross-checked to agree exactly before anything is written:

1. **Composition** is reconstructed from the actual 72 group-stage fixtures in
   `martj42/international_results`. During a group stage each team plays exactly
   the other three in its group, so the "played-against" graph is 12 disjoint
   complete K4 sub-graphs — the 12 connected components *are* the groups.
2. **Labels (A–L)** are read from the official Wikipedia group pages.

`world_cup.refresh_groups()` writes `data/world_cup_2026_groups.csv` only if the
Wikipedia labels partition the teams identically to the fixture reconstruction
(it refuses to write a draw the two sources disagree on). Results come from the
same martj42 feed and can be topped up live from The Odds API `/scores` endpoint.

## The real groups (verified)

| Group | Teams |
|---|---|
| A | Mexico, South Africa, South Korea, Czechia |
| B | Canada, Bosnia-Herzegovina, Qatar, Switzerland |
| C | Brazil, Morocco, Haiti, Scotland |
| D | United States, Paraguay, Australia, Turkey |
| E | Germany, Curaçao, Côte d'Ivoire, Ecuador |
| F | Netherlands, Japan, Sweden, Tunisia |
| G | Belgium, Egypt, Iran, New Zealand |
| H | Spain, Cabo Verde, Saudi Arabia, Uruguay |
| I | France, Senegal, Iraq, Norway |
| J | Argentina, Algeria, Austria, Jordan |
| K | Portugal, Congo DR, Uzbekistan, Colombia |
| L | England, Croatia, Ghana, Panama |

(Committed to `data/world_cup_2026_groups.csv`; regenerate with
`world_cup.refresh_groups()` or `world-cup --refresh-groups`.)

## What the model is, and what it is NOT

- The default is `EloGoalsModel` — margin-weighted international Elo turned into a
  Poisson scoreline, fit on all real international history through today (games
  already played update the ratings; that is past info relative to the matches
  still being simulated, so it is **not** leakage).
- It is a **team-strength baseline**, deliberately chosen over Dixon-Coles for the
  national-team setting (DC's MLE over ~300 national teams is impractical and
  thin). It produces sensible champion odds (top contenders: Argentina, Spain,
  France, Brazil) but it is **not** as sharp as the betting market and it
  **overrates heavy underdogs** — do not read its raw "edges" as value (see
  `edges` below).
- The third-place→Round-of-32 allocation is a verified-valid **approximation** of
  FIFA's Annex C matrix (every assigned group is eligible; each group used once),
  not the exact published mapping.

## Live betting edge

```bash
PYTHONPATH=src python -m soccer_predictor.cli edges --model elo --sport soccer_fifa_world_cup
```

This compares model probabilities to the **best** live decimal odds across books.
It flags `likely_miscalibration=True` for longshot "edges" the market prices
below ~8% — those are model error, not value. **Calibrate the model and check
closing-line value before trusting any edge.** The Odds API free tier is ~500
requests/month; every call is cached (`data/cache/odds_api/`) and the remaining
quota is printed.

## Optional model knobs (and what the evidence says)

```bash
# Market-anchored live sim: blend near-term fixtures toward live odds; unpriced
# future knockout matchups fall back to the pure model.
python -m soccer_predictor.cli world-cup --model elo --anchor --anchor-weight 0.5 --sims 20000

# Add a CURRENT-only squad-strength overlay (supply ratings from FotMob /
# Transfermarkt / FIFA points as a team,strength CSV). Not backtestable.
python -m soccer_predictor.cli world-cup --model elo --squad-csv data/squad_strength.csv

# Recent-form weight (defaults to 0 — the backtest shows it does not help).
python -m soccer_predictor.cli world-cup --model elo --form-weight 250
```

### Backtest any of these on past World Cups

```bash
python -m soccer_predictor.cli backtest-world-cups --min-year 2006 --champions
```

This scores each model variant out-of-sample, match by match, on every World Cup
since `--min-year` (default 2006, the 32-team regime). **Finding:** plain
margin-weighted Elo beats Elo+form, Elo+importance, and Elo+pedigree on
out-of-sample log loss — none of the add-ons help, so they default OFF. See
[model_notes.md](model_notes.md) for the table. Historical market odds are not
free, so the market anchor is a *live-only* edge, not a backtested one.

## Definitive backtest — does this translate to good past-World-Cup predictions?

Two out-of-sample tests on the seven editions 1998–2022 (each predicted using only
data from BEFORE it started). Reproduce with `soccer-predictor backtest-world-cups`
(match level) and `evaluation.backtest_champions` (tournament level).

### Match level (default Elo model, 460 real WC matches)

| edition | log loss | RPS | accuracy | ECE |
|---|---|---|---|---|
| 1998 | 0.963 | 0.365 | 51.6% | 0.039 |
| 2002 | 1.032 | 0.431 | 50.0% | 0.044 |
| 2006 | 0.907 | 0.353 | 62.5% | 0.105 |
| 2010 | 0.966 | 0.387 | 53.1% | 0.049 |
| 2014 | 0.955 | 0.403 | 59.4% | 0.065 |
| 2018 | 0.965 | 0.408 | 56.3% | 0.062 |
| 2022 | 1.033 | 0.427 | 56.3% | 0.055 |
| **pooled** | **0.976** | **0.396** | **55.4%** | **0.047** |
| no-skill floor | 1.068 | — | — | — |

The model beats the no-skill baseline by ~0.09 log loss with good calibration
(ECE 0.047) — i.e. its probabilities are honest, not just its rankings. ~55%
straight 1X2 accuracy is strong for international football (where ~25% of games
are draws). This is the answer to "do our probabilities translate to good WC
predictions": **yes, out of sample, consistently across seven tournaments.**

### Tournament level (simulate each past WC pre-start, 4000 sims/edition)

| year | actual champion | model prob | pre-tournament rank | model favourite |
|---|---|---|---|---|
| 1998 | France | 10.1% | 3 | Brazil |
| 2002 | Brazil | 10.3% | 3 | France |
| 2006 | Italy | 5.9% | 8 | Brazil |
| 2010 | Spain | 26.3% | **1** | Spain |
| 2014 | Germany | 10.2% | 4 | Brazil |
| 2018 | France | 6.2% | 5 | Brazil |
| 2022 | Argentina | 14.6% | 2 | Brazil |

**The actual champion landed in the model's pre-tournament top 6 in 6 of 7 editions
(mean rank 3.7, median 3)** — the lone miss being Italy 2006, a famous upset (rank
8). Note the model's favourite (usually Brazil) rarely wins: the World Cup is
genuinely high-variance, so even a good model gives the eventual winner only
~6–26%. Treat tournament-level numbers as a sanity check (seven editions is a small
sample), but the signal is clear: our probabilities rank past champions well.

### Current live call (Elo + market anchor, 12/72 group games locked in)

Argentina ~19%, Spain ~16%, France ~9%, Brazil ~9%, England ~6%, Colombia/Portugal
~6%. Full table: `outputs/world_cup_2026_probabilities.csv` (regenerate any time
with `soccer-predictor world-cup`).
