"""
Historical national-squad market value, point-in-time, for BACKTESTING.

The live squad overlay (Transfermarkt/FotMob scrapes in :mod:`players`) is
current-only, so it cannot be validated on past World Cups. This module supplies
a *historical, as-of-date* squad-strength signal that CAN be backtested, sourced
legally and for free from the dcaribou ``transfermarkt-datasets`` release (CC0).

Provenance + legality (deliberately the clean route -- see the squad-strength
research note): Transfermarkt's own value-history endpoint (``/ceapi``) is
``Disallow``-ed in robots.txt for our user-agent, so we do NOT scrape it. Instead
we read the already-published CC0 dataset from its public Cloudflare-R2 DVC
remote. Each yearly ``players.json.gz`` carries, per player, ``citizenship`` and a
dated ``market_value_history`` -- so a player's value AS OF a past World Cup is a
lookup, with no leakage (we take the last valuation on or before kickoff).

HONEST LIMITATIONS (carried so the backtest stays disciplined):
  * Depth: the dataset's value snapshots begin ~2012, so only WCs 2014/2018/2022
    are covered. Earlier editions get NO squad signal (the model falls back to
    pure Elo for them) -- not a fabricated number.
  * Coverage bias: the player pool is European-league-weighted, so nations whose
    players are home-based (Japan, Mexico, Saudi Arabia, South Korea, ...) are
    under-covered. We therefore apply a COVERAGE GATE: a nation contributes a
    squad value only if at least ``min_covered`` of its players have an as-of
    value; otherwise it is omitted and the model uses pure Elo for it. This is
    the user's rule -- "when it's not available, it should not factor in" -- and
    avoids handing under-measured teams a spurious downgrade.
  * Value != ability: market value embeds age / hype / TV-money inflation. We
    normalise WITHIN each edition (z-score) downstream so cross-era EUR inflation
    cancels; the residual age/hype distortion is a known caveat.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .aliases import normalize_team_name
from .sources import _ssl_context

# Public, read-only DVC remote (Cloudflare R2) for the CC0 dataset.
_R2_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/dvc/files/md5"
# The DVC pointer for the raw scraper dataset lives in the project repo; reading
# it fresh keeps us current with the weekly refresh instead of pinning a hash.
_SCRAPER_DVC = (
    "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/"
    "data/raw/transfermarkt-scraper.dvc"
)
_CACHE_DIR = Path("data/cache/squad_value")

# World Cup kickoff dates -- the as-of cutoff for valuations (verified facts, not
# fabricated data). Only editions the dataset actually covers (2012+) are listed.
WC_ASOF: dict[int, str] = {
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}


def _md5_url(md5: str) -> str:
    return f"{_R2_BASE}/{md5[:2]}/{md5[2:]}"


def _http_get(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "soccer-predictor"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return r.read()


def _cached_bytes(url: str, name: str, ttl_days: float = 14.0) -> bytes:
    """GET ``url`` to a named cache file and reuse it while fresh (paid-data rule)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / name
    if path.exists() and (time.time() - path.stat().st_mtime) / 86400.0 < ttl_days:
        if path.stat().st_size > 0:
            return path.read_bytes()
    data = _http_get(url)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def _scraper_dir_listing() -> dict[str, str]:
    """relpath -> md5 for every file in the published scraper dataset."""
    dvc = _cached_bytes(_SCRAPER_DVC, "scraper.dvc", ttl_days=7.0).decode("utf-8")
    dir_md5 = None
    for line in dvc.splitlines():
        line = line.strip().lstrip("- ").strip()
        if line.startswith("md5:") and line.rstrip().endswith(".dir"):
            dir_md5 = line.split("md5:", 1)[1].strip()
            break
    if not dir_md5:
        raise RuntimeError("could not read scraper .dir md5 from the DVC pointer")
    listing = json.loads(_cached_bytes(_md5_url(dir_md5), "scraper_dir.json"))
    return {item["relpath"]: item["md5"] for item in listing}


def fetch_players(year: int) -> list[dict]:
    """Parse one year's ``players.json.gz`` (player metadata + value history)."""
    listing = _scraper_dir_listing()
    rel = f"{year}/players.json.gz"
    if rel not in listing:
        raise RuntimeError(f"{rel} not in the published dataset (have {len(listing)} files)")
    raw = _cached_bytes(_md5_url(listing[rel]), f"players_{year}.json.gz")
    out = []
    for line in gzip.decompress(raw).splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _citizenship(player: dict) -> str | None:
    c = player.get("citizenship")
    if isinstance(c, list):
        c = c[0] if c else None
    return c if isinstance(c, str) and c.strip() else None


def _value_as_of(player: dict, as_of_ms: int) -> float | None:
    """The player's market value (EUR) at the LAST valuation on/before as_of_ms."""
    hist = player.get("market_value_history")
    if not isinstance(hist, list) or not hist:
        return None
    best_x, best_y = None, None
    for pt in hist:
        x, y = pt.get("x"), pt.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if x <= as_of_ms and (best_x is None or x > best_x):
            best_x, best_y = x, float(y)
    return best_y


def build_squad_value(
    year: int,
    *,
    top_k: int = 20,
    min_covered: int = 15,
) -> pd.DataFrame:
    """Per-nation squad value as of ``year``'s World Cup (covered nations only).

    For each citizenship, take every player's as-of-kickoff value, keep the
    ``top_k`` highest (the strength core), and report their MEAN. A nation is
    included only if at least ``min_covered`` of its players have an as-of value
    -- under-covered nations are omitted so the model falls back to pure Elo
    rather than receiving a biased, partial-roster downgrade.
    """
    if year not in WC_ASOF:
        raise ValueError(f"no as-of date for {year}; covered editions: {sorted(WC_ASOF)}")
    as_of_ms = int(pd.Timestamp(WC_ASOF[year]).timestamp() * 1000)
    players = fetch_players(year)
    by_nat: dict[str, list[float]] = {}
    for p in players:
        nat = _citizenship(p)
        if not nat:
            continue
        v = _value_as_of(p, as_of_ms)
        if v is None or v <= 0:
            continue
        by_nat.setdefault(nat, []).append(v)
    rows = []
    for nat, vals in by_nat.items():
        if len(vals) < min_covered:
            continue
        vals.sort(reverse=True)
        core = vals[:top_k]
        rows.append({
            "team": normalize_team_name(nat),
            "year": year,
            "squad_value_eur": float(sum(core) / len(core)),
            "n_covered": len(vals),
        })
    df = pd.DataFrame(rows).sort_values("squad_value_eur", ascending=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# EA FC / FIFA player ABILITY ratings (real ability, not value), historical.
# Source: the public jsulz/FIFA23 mirror on Hugging Face (no auth) of the
# stefanoleone992 SoFIFA export -- FIFA 15..23 with overall + nationality + a
# dated update snapshot, so a squad's ability AS OF a past World Cup is a lookup.
# The live rating-site APIs (SofaScore/FotMob/WhoScored) are Cloudflare/signed-
# header gated to scripts, so this open mirror is how we "get the data".
# --------------------------------------------------------------------------- #
import io
import urllib.parse

_FIFA_URL = (
    "https://huggingface.co/datasets/jsulz/FIFA23/resolve/main/"
    + urllib.parse.quote("male_players (legacy).csv")
)
# WC edition -> the FIFA edition whose ratings are current at that tournament.
# 2014 has no pre-WC FIFA edition in the mirror (FIFA 15 launched Sept 2014, after
# the June WC), so it is flagged as leakage-affected and excluded from the clean
# backtest; 2018/2022 use the pre-tournament FIFA 18 / FIFA 23 launch rosters.
WC_FIFA_VERSION: dict[int, int] = {2014: 15, 2018: 18, 2022: 23}
FIFA_LEAKAGE_EDITIONS = {2014}  # rating snapshot post-dates the tournament


def fetch_fifa_players() -> pd.DataFrame:
    """FIFA 15..23 player ratings (overall, nationality, dated update) from HF."""
    raw = _cached_bytes(_FIFA_URL, "fifa_male_players_legacy.csv", ttl_days=90.0)
    return pd.read_csv(
        io.BytesIO(raw),
        usecols=["fifa_version", "fifa_update_date", "short_name", "overall",
                 "age", "nationality_name"],
        low_memory=False,
    )


def _fifa_snapshot(df: pd.DataFrame, version: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """The single rating update for ``version`` closest to (and at/ before) as_of."""
    sub = df[df["fifa_version"] == version].copy()
    if sub.empty:
        return sub
    sub["_d"] = pd.to_datetime(sub["fifa_update_date"], errors="coerce")
    pre = sub[sub["_d"] <= as_of]
    chosen = pre["_d"].max() if len(pre) else sub["_d"].min()  # min() only for leaky 2014
    return sub[sub["_d"] == chosen]


def _aggregate_ability(snap: pd.DataFrame, top_k: int, min_covered: int) -> pd.DataFrame:
    rows = []
    snap = snap.dropna(subset=["overall", "nationality_name"])
    for nat, grp in snap.groupby("nationality_name"):
        vals = sorted((float(v) for v in grp["overall"]), reverse=True)
        if len(vals) < min_covered:
            continue
        core = vals[:top_k]
        rows.append({"team": normalize_team_name(str(nat)),
                     "ability": float(sum(core) / len(core)), "n_covered": len(vals)})
    return pd.DataFrame(rows).sort_values("ability", ascending=False).reset_index(drop=True)


def build_fifa_ability(
    year: int, *, top_k: int = 20, min_covered: int = 15
) -> pd.DataFrame:
    """Per-nation FIFA squad ABILITY (top-K overall) as of ``year``'s World Cup."""
    if year not in WC_FIFA_VERSION:
        raise ValueError(f"no FIFA edition mapped for {year}")
    as_of = pd.Timestamp(WC_ASOF.get(year, f"{year}-06-15"))
    snap = _fifa_snapshot(fetch_fifa_players(), WC_FIFA_VERSION[year], as_of)
    return _aggregate_ability(snap, top_k, min_covered)


def build_fifa_ability_by_year(
    years: list[int] | None = None, *, top_k: int = 20, min_covered: int = 15,
) -> dict[int, dict[str, float]]:
    """{year: {team: ability}} for the FIFA-backtestable editions (clean: 2018/2022)."""
    years = years or [y for y in sorted(WC_FIFA_VERSION) if y not in FIFA_LEAKAGE_EDITIONS]
    out: dict[int, dict[str, float]] = {}
    for y in years:
        df = build_fifa_ability(y, top_k=top_k, min_covered=min_covered)
        out[y] = dict(zip(df["team"], df["ability"]))
    return out


def build_fifa_current_ability(*, top_k: int = 20, min_covered: int = 15) -> pd.DataFrame:
    """Latest available FIFA edition's squad ability (for the LIVE 2026 overlay)."""
    df = fetch_fifa_players()
    version = int(pd.to_numeric(df["fifa_version"], errors="coerce").max())
    sub = df[df["fifa_version"] == version].copy()
    sub["_d"] = pd.to_datetime(sub["fifa_update_date"], errors="coerce")
    snap = sub[sub["_d"] == sub["_d"].max()]
    out = _aggregate_ability(snap, top_k, min_covered)
    out["source_year"] = version
    return out


def _latest_year_with_values(listing: dict[str, str]) -> int:
    """Most recent edition whose ``players.json.gz`` still carries value histories.

    The newest scrapes (2023+) dropped the embedded ``market_value_history``, so we
    walk back to the freshest year that actually has values (currently 2022).
    """
    years = sorted(
        {int(k.split("/")[0]) for k in listing if k.endswith("players.json.gz")},
        reverse=True,
    )
    for year in years:
        players = fetch_players(year)
        if any(isinstance(p.get("market_value_history"), list) and p["market_value_history"]
               for p in players[:200]):
            return year
    raise RuntimeError("no players.json.gz with market_value_history in the dataset")


def build_current_ability(
    *,
    top_k: int = 20,
    min_covered: int = 15,
) -> pd.DataFrame:
    """Per-nation CURRENT squad ABILITY (age-adjusted), for the live overlay.

    Uses the most recent ``players.json.gz``: each player's latest valuation is
    DE-AGED via :func:`players._age_value_retention` so a 33-year-old great is
    rated on ability, not his age-discounted price (the user's "we want how good
    they are, not their value" point). Same top-K-mean + coverage gate as the
    historical builder, so under-covered nations fall back to pure Elo.

    NOTE: this is a Transfermarkt-derived ability proxy. Cleaner pure-ability
    ratings (FotMob/WhoScored) sit behind signed-header/anti-scrape walls, so they
    are supported only via a user-supplied CSV (``world-cup --squad-csv``).
    """
    from .players import _age_value_retention

    listing = _scraper_dir_listing()
    year = _latest_year_with_values(listing)
    players = fetch_players(year)
    by_nat: dict[str, list[float]] = {}
    for p in players:
        nat = _citizenship(p)
        if not nat:
            continue
        hist = p.get("market_value_history")
        if not isinstance(hist, list) or not hist:
            continue
        last = max(hist, key=lambda pt: pt.get("x", 0) if isinstance(pt.get("x"), (int, float)) else 0)
        y = last.get("y")
        if not isinstance(y, (int, float)) or y <= 0:
            continue
        try:
            age = float(last.get("age")) if last.get("age") not in (None, "") else None
        except (TypeError, ValueError):
            age = None
        ability = float(y) / _age_value_retention(age) if age is not None else float(y)
        by_nat.setdefault(nat, []).append(ability)
    rows = []
    for nat, vals in by_nat.items():
        if len(vals) < min_covered:
            continue
        vals.sort(reverse=True)
        core = vals[:top_k]
        rows.append({
            "team": normalize_team_name(nat),
            "ability": float(sum(core) / len(core)),
            "n_covered": len(vals),
            "source_year": year,
        })
    return pd.DataFrame(rows).sort_values("ability", ascending=False).reset_index(drop=True)


SQUAD_VALUE_CSV = Path("data/squad_value_by_year.csv")


def build_squad_value_by_year(
    years: list[int] | None = None,
    *,
    out: Path | None = SQUAD_VALUE_CSV,
    top_k: int = 20,
    min_covered: int = 15,
) -> pd.DataFrame:
    """Build (and optionally persist) the per-edition squad-value table."""
    years = years or sorted(WC_ASOF)
    frames = [build_squad_value(y, top_k=top_k, min_covered=min_covered) for y in years]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out is not None and len(table):
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
    return table


def load_ability_csv(
    path: str | Path, *, top_k: int = 20, min_covered: int = 1
) -> dict[str, float]:
    """Load ANY reputable rating site's export into a {team: strength} overlay.

    Source-agnostic by design (the rating-site APIs -- SofaScore, FotMob detail,
    WhoScored -- are Cloudflare/signed-header gated to scripts, so the reliable
    bridge is a CSV you export from the browser). Accepts either shape:
      * PER-TEAM: one row per national team (a ``team`` column + a numeric rating).
      * PER-PLAYER: many rows (a ``team`` column + a per-player ``rating`` column);
        rows are aggregated to the team's top-``top_k`` mean, with a coverage gate.
    Column names are detected loosely (team/nation/country; rating/ability/
    strength/value/score/overall, else the first numeric column). Scale doesn't
    matter -- it is z-scored downstream by :func:`players.to_elo_adjustment`.
    """
    df = pd.read_csv(path)
    if df.empty:
        return {}
    lc = {str(c).strip().lower(): c for c in df.columns}
    team_col = next((lc[k] for k in ("team", "nation", "country", "national_team",
                                     "squad") if k in lc), None)
    rating_col = next((lc[k] for k in ("rating", "ability", "strength", "value",
                                       "score", "overall", "mean_rating") if k in lc),
                      None)
    if team_col is None:
        return {}
    if rating_col is None:  # fall back to the first numeric column that isn't the team
        for c in df.columns:
            if c == team_col:
                continue
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                rating_col = c
                break
    if rating_col is None:
        return {}
    teams = df[team_col].map(normalize_team_name)
    ratings = pd.to_numeric(df[rating_col], errors="coerce")
    ok = teams.ne("") & ratings.notna()
    out: dict[str, float] = {}
    for team, grp in ratings[ok].groupby(teams[ok]):
        vals = sorted((float(v) for v in grp), reverse=True)
        if len(vals) < min_covered:
            continue
        core = vals[:top_k]
        out[str(team)] = float(sum(core) / len(core))
    return out


def combine_ability(sources: list[dict[str, float]]) -> dict[str, float]:
    """Blend several ability sources into one z-scored {team: strength}.

    Each source is z-scored over its OWN teams first, so different scales
    (FIFA overall 0-100, SofaScore 0-10, market value in €) become comparable;
    then per team we average the z-scores of whatever sources cover it. A team in
    only one source keeps that source's z; a team in none is absent (so the model
    falls back to pure Elo for it). This is the "supplement, optional, only where
    data exists" rule: pass ``[fifa]`` for FIFA-only, or ``[fifa, sofascore, ...]``
    to enrich it with any other reputable ratings you have.
    """
    import numpy as np

    zsources: list[dict[str, float]] = []
    for src in sources:
        if not src:
            continue
        vals = np.array(list(src.values()), dtype=float)
        mean = float(vals.mean())
        std = float(vals.std()) or 1.0
        zsources.append({t: (float(v) - mean) / std for t, v in src.items()})
    if not zsources:
        return {}
    teams: set[str] = set().union(*[set(z) for z in zsources])
    out: dict[str, float] = {}
    for t in teams:
        zs = [z[t] for z in zsources if t in z]
        if zs:
            out[t] = float(sum(zs) / len(zs))
    return out


def load_squad_value_by_year(path: Path = SQUAD_VALUE_CSV) -> dict[int, dict[str, float]]:
    """Read the persisted table into ``{year: {team: squad_value_eur}}``."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    out: dict[int, dict[str, float]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(int(r.year), {})[normalize_team_name(r.team)] = float(r.squad_value_eur)
    return out
