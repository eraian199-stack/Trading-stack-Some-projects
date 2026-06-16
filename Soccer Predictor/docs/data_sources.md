# Data Sources

Where the numbers come from, what they cost, and how the package caches them.
Loaders normalise every source into the canonical match schema (see
[`framework.md`](framework.md) and `data/schemas.py`), so the rest of the system
never sees a source-specific format. The fetchers and constants referenced here
live in `data/sources.py`; the file/directory loaders in `data/loaders.py`.

A standing project rule (and a hard-won lesson from adjacent work): **every paid
or rate-limited adapter must cache to disk so re-runs do not re-bill or
re-hammer the source.** The caching policy is the last section below.

---

## 1. At a glance

| Source | Scope | Cost | Used for | Loader / fetcher |
|---|---|---|---|---|
| martj42 `international_results` | International (men's national teams, 1872→) | Free | Base historical results | `load_international_results`, `fetch_international_results` |
| football-data.co.uk | Club leagues (top European divisions) | Free | Club results **with closing odds** | `load_football_data`, `fetch_football_data` |
| ClubElo | Club Elo ratings (daily) | Free | Club strength prior | `fetch_clubelo` |
| FIFA men's world ranking | International ranking points | Free (best-effort) | `*_fifa_rank` feature | `FIFA_RANKING_URL` |
| Understat | Club shot/xG data | Free (scraping, fragile) | `home_xg`/`away_xg` | optional, guarded import |
| FBref (via soccerdata) | Club + some international xG | Free (scraping, fragile) | `home_xg`/`away_xg` | optional, guarded import |

"Best-effort" / "fragile" sources must fail gracefully — an informative error or
a skip, never a crash and never silently invented data.

---

## 2. Club vs international

The package serves two worlds through one table, distinguished by
`competition_type` (`club_league`, `club_cup`, `international_friendly`,
`qualifier`, `tournament`). They differ in what data exists:

- **Club football** is where the rich enrichment lives: closing betting odds,
  xG, and dense, regular fixtures. football-data.co.uk and ClubElo are the
  backbone. Most of the betting-evaluation and odds-feature machinery is
  exercised on club data.
- **International football** (the World Cup 2026 target) is sparser: fewer
  matches per team per year, mostly **neutral-site** tournament games, rarely any
  xG, and patchy odds. The martj42 results set plus FIFA ranking and (optionally)
  squad value carry most of the signal. Because matches are neutral, scoreline
  models drop the home-advantage term when `neutral_site` is true.

`data/sources.classify_competition_type(name)` maps a competition name to one of
the canonical types (World Cup / Euro / Copa / Cup of Nations → `tournament`;
"…qualification/qualifier" → `qualifier`; "friendly" → `international_friendly`;
default `tournament` for international comps, `club_league` otherwise).

---

## 3. Free sources

### martj42 `international_results` (recommended base for the World Cup)

A community-maintained CSV of essentially every men's international match since
1872, updated regularly.

```
https://raw.githubusercontent.com/martj42/international_results/master/results.csv
```

This is `sources.INTERNATIONAL_RESULTS_URL` and the CLI default. Schema:
`date, home_team, away_team, home_score, away_score, tournament, city, country,
neutral`. `load_international_results` maps `tournament → competition`,
`neutral → neutral_site`, and sets `competition_type` via
`classify_competition_type`. It is the single most important free source for this
project — but if it is stale, the model is stale, so refresh before a serious run.

### football-data.co.uk (club leagues, with closing odds)

Free historical CSVs per league per season, the standard free source for club
results **and bookmaker odds**, including closing-style columns that power
closing-line-value analysis.

```
https://www.football-data.co.uk/mmz4281/<season>/<league>.csv   # e.g. .../2324/E0.csv
```

Base URL is `sources.FOOTBALL_DATA_BASE`; `fetch_football_data(league="E0",
season="2324")` builds the path. Loader notes (`load_football_data`):

- Column map: `Date → date, HomeTeam → home_team, AwayTeam → away_team,
  FTHG → home_score, FTAG → away_score`.
- Odds: take the **first available** triple from
  `(AvgH,AvgD,AvgA) → (B365H,B365D,B365A) → (PSH,PSD,PSA) → (BWH,BWD,BWA) →
  (MaxH,MaxD,MaxA)` into `home_odds/draw_odds/away_odds`; closing-style columns,
  when present, map to `closing_*`.
- Dates are `dd/mm/yy` (`dayfirst=True`); encoding is `latin-1`.
- `competition_type = "club_league"`. A directory of CSVs is concatenated and
  sorted.

### ClubElo (club strength prior)

Daily Elo ratings for clubs, queryable by team or by date.

```
http://api.clubelo.com        # sources.CLUBELO_URL ; e.g. /Arsenal or /2024-08-01
```

`fetch_clubelo(team_or_date)` is best-effort and degrades gracefully. Useful as a
club strength prior or sanity check against the package's own Elo.

### FIFA men's world ranking (international ranking points)

Point-in-time ranking points are a strong international feature (they feed
`home_fifa_rank` / `away_fifa_rank`, consumed by `features/squad.py` as
`fifa_rank_diff`). `sources.FIFA_RANKING_URL` is a best-effort fetch; FIFA does
not publish a clean stable API, so treat this as a convenience source and verify
against the authoritative page before a serious run:

```
https://inside.fifa.com/fifa-world-ranking/men
```

---

## 4. xG sources and the 2026-01-20 FBref removal note

xG (expected goals) is optional enrichment. When `home_xg` / `away_xg` are
present and non-null, the feature store adds rolling-xG features and `PoissonXG`
becomes usable; when absent, everything else still works. Two common free
sources, both **scraping-based and therefore fragile** — guard their imports
(`understatapi`, `soccerdata`) with `try/except ImportError` and skip when
unavailable:

- **Understat** — shot-level and match xG for the big-five European club leagues.
- **FBref** (accessed via the `soccerdata` library) — broader club coverage and
  some international xG.

> **2026-01-20 — FBref removed public xG / advanced stats.** As of 20 January
> 2026 FBref (and its data partner Opta/StatsBomb) discontinued free public
> access to xG and advanced metrics on the site. Scrapers built on
> `soccerdata`'s FBref backend that worked before that date will return empty
> or error for xG columns afterward. **Do not assume FBref xG is available**;
> treat the xG feature group as optional and let it degrade. For going-forward xG
> you now need Understat (where still available) or a paid provider (e.g.
> StatsBomb / Opta licensing), and any paid adapter must be disk-cached per the
> policy below. Historical xG already cached on disk before the cutoff remains
> usable.

---

## 5. Paid sources (not required)

The package runs end-to-end on free data; paid sources are upgrades, not
dependencies.

- **Squad / player market value** (e.g. Transfermarkt-derived datasets) → feeds
  `home_squad_value` / `away_squad_value` and the `squad_value_diff` feature.
- **Injury / availability feeds** → `*_injury_score` and `injury_score_diff`.
- **Commercial odds APIs** (live and closing prices across many books) → richer
  `home_odds`/`closing_*` coverage for value betting and CLV.
- **Licensed xG / event data** (StatsBomb, Opta) → the durable replacement for
  the now-removed free FBref xG.

If any of these is wired in, it MUST cache to disk (next section).

---

## 6. Templates and sample data

Starter CSVs live in `data/templates/` and the canonical-schema samples in
`data/`:

- `data/templates/world_cup_2026_groups_template.csv` — `group,team`; 12 groups
  A–L × 4 **placeholder** teams.
- `data/templates/world_cup_2026_fixtures_template.csv` — the full 72-match group
  schedule (`match_id,date,group,home_team,away_team,neutral,home_score,
  away_score`) with placeholder teams, blank scores, `neutral=true`.
- `data/templates/club_league_template.csv` — the canonical club columns with two
  clearly-labelled **sample (not real)** rows.
- `data/sample_groups.csv`, `data/sample_results.csv` — synthetic smoke-test data.

**All template and sample data is placeholder / synthetic and labelled as not
real. Do not use it for real probabilities.** Fill the templates with verified
2026 groups, fixtures, and (after matches are played) scores. The authoritative
2026 references to verify against:

- Scores & fixtures: `https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures`
- FIFA men's ranking: `https://inside.fifa.com/fifa-world-ranking/men`
- FIFA WC2026 regulations and bracket rules (especially third-place allocation).

---

## 7. Caching policy

Caching is mandatory for every remote source, free or paid — re-runs must never
re-fetch and a paid adapter must never re-bill.

- **Single entry point.** `sources.cached_get(url, cache_dir="data/cache",
  ttl_days=7)` downloads a URL to a hashed file under `data/cache/` and returns
  the local path; on a fresh cache hit it returns the cached path without any
  network call. All remote fetchers (`fetch_international_results`,
  `fetch_football_data`, `fetch_clubelo`) go through it.
- **TTL.** Default freshness window is 7 days. Slow-moving reference data (old
  seasons, historical results) can use a longer TTL; live odds need a short one.
  An expired entry triggers a re-fetch and refresh.
- **Hashed keys.** The cache filename is a hash of the URL, so different
  leagues/seasons/dates never collide and the cache is safe to share across runs.
- **Graceful failure.** A network error raises an *informative* error (or the
  caller skips that source) — it never crashes the pipeline or substitutes
  fabricated data.
- **Paid adapters.** Any paid source (odds API, licensed xG, squad value) wraps
  the same disk-cache discipline: cache the raw response keyed by the exact
  request so identical re-runs are free. Treat the cache as the source of truth
  for already-billed calls.
- **Hygiene.** `data/cache/` is regenerable and should be git-ignored. Delete it
  to force a clean re-fetch; never commit cached payloads.
