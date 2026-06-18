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


def load_squad_value_by_year(path: Path = SQUAD_VALUE_CSV) -> dict[int, dict[str, float]]:
    """Read the persisted table into ``{year: {team: squad_value_eur}}``."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    out: dict[int, dict[str, float]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(int(r.year), {})[normalize_team_name(r.team)] = float(r.squad_value_eur)
    return out
