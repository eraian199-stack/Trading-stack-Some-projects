"""
The Odds API adapter (https://the-odds-api.com).

This is the live-odds + live-scores bridge for the predictor. It powers three
things:
  1. a market PRIOR / benchmark (de-vigged bookmaker odds -- the line to beat),
  2. the Betting Edge Scanner (model fair odds vs the best available price),
  3. live RESULTS ingestion during a tournament (the /scores endpoint), so a
     World Cup simulation reflects games that have already been played.

The Odds API is QUOTA-LIMITED (the free tier is ~500 requests/month), so every
call is cached to disk with a short TTL and the remaining-quota header is logged.
Re-running must not silently burn the quota -- a project non-negotiable, same as
any paid source ([[feedback_paid_data_caching]]).

Key resolution mirrors the existing arb_scanner.py: ODDS_API_KEY env var, then
~/.odds_api_key, then a .env file in cwd / package dir / home. Everything
degrades gracefully: no key or no network raises an informative error rather than
a bare traceback, and a stale cached copy is served when the network is down.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .aliases import normalize_team_name
from .sources import _ssl_context  # reuse the certifi-aware SSL context

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Common soccer sport keys. Use list_sports() for the authoritative live list.
WORLD_CUP_SPORT = "soccer_fifa_world_cup"
WORLD_CUP_WINNER_SPORT = "soccer_fifa_world_cup_winner"  # outright/futures market
SOCCER_SPORT_KEYS: dict[str, str] = {
    "world_cup": "soccer_fifa_world_cup",
    "euros": "soccer_uefa_european_championship",
    "champions_league": "soccer_uefa_champs_league",
    "epl": "soccer_epl",
    "la_liga": "soccer_spain_la_liga",
    "bundesliga": "soccer_germany_bundesliga",
    "serie_a": "soccer_italy_serie_a",
    "ligue_one": "soccer_france_ligue_one",
    "copa_america": "soccer_conmebol_copa_america",
}

_CACHE_DIR = Path("data/cache/odds_api")
_DRAW = "Draw"


# --------------------------------------------------------------------------- #
# Key resolution (mirrors arb_scanner.load_api_key)
# --------------------------------------------------------------------------- #
def _load_env_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                os.environ.setdefault(key, value)
        except OSError:
            continue


def load_odds_api_key() -> str | None:
    """Resolve The Odds API key from env, ~/.odds_api_key, or a .env file."""
    _load_env_files(
        [Path.cwd() / ".env", Path(__file__).with_name(".env"), Path.home() / ".env"]
    )
    key = os.getenv("ODDS_API_KEY")
    if not key:
        key_file = Path.home() / ".odds_api_key"
        try:
            if key_file.is_file():
                key = key_file.read_text().strip()
                if key:
                    os.environ["ODDS_API_KEY"] = key
        except OSError:
            key = None
    return key or None


class OddsApiError(RuntimeError):
    """Raised when an Odds API request cannot be satisfied (and no cache)."""


# --------------------------------------------------------------------------- #
# Cached HTTP GET
# --------------------------------------------------------------------------- #
def _cache_path(url: str) -> Path:
    import hashlib

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return _CACHE_DIR / f"{digest}.json"


def _api_get(
    path: str,
    params: dict[str, str],
    *,
    cache: bool = True,
    ttl_hours: float = 3.0,
) -> object:
    """GET a JSON endpoint with disk caching and quota logging.

    Returns the parsed JSON. Raises OddsApiError on failure when no cached copy
    exists. A fresh cached copy (younger than ttl_hours) skips the network call.
    """
    key = load_odds_api_key()
    if not key:
        raise OddsApiError(
            "No Odds API key found. Set ODDS_API_KEY, create ~/.odds_api_key, "
            "or add ODDS_API_KEY=... to a .env file. Get a free key at "
            "https://the-odds-api.com (free tier ~500 requests/month)."
        )
    query = dict(params)
    query["apiKey"] = key
    url = f"{ODDS_API_BASE}{path}?{urllib.parse.urlencode(query)}"

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath = _cache_path(url)
    if cache and cpath.exists():
        age_h = (time.time() - cpath.stat().st_mtime) / 3600.0
        if age_h < ttl_hours and cpath.stat().st_size > 0:
            return json.loads(cpath.read_text())

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "soccer-predictor"})
        with urllib.request.urlopen(request, timeout=20, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8")
            remaining = resp.headers.get("x-requests-remaining")
            used = resp.headers.get("x-requests-used")
        if remaining is not None:
            # Visible so the user can watch their free-tier quota.
            print(f"[odds_api] quota remaining={remaining} used={used}")
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(raw)
        tmp.replace(cpath)
        return json.loads(raw)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        if cpath.exists() and cpath.stat().st_size > 0:
            print(f"[odds_api] network failed ({exc}); serving cached copy.")
            return json.loads(cpath.read_text())
        # Surface the API's own error body when present (e.g. bad key / quota).
        body = ""
        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = ": " + exc.read().decode("utf-8")[:300]
            except Exception:
                body = ""
        raise OddsApiError(
            f"Odds API request failed for {path} ({type(exc).__name__}: {exc}{body})"
        ) from exc


# --------------------------------------------------------------------------- #
# Public fetchers
# --------------------------------------------------------------------------- #
def list_sports(all_sports: bool = False, cache: bool = True) -> pd.DataFrame:
    """List sports/competitions the key can access. Free (does not cost quota)."""
    data = _api_get(
        "/sports",
        {"all": "true"} if all_sports else {},
        cache=cache,
        ttl_hours=24.0,
    )
    rows = [
        {
            "key": s.get("key"),
            "group": s.get("group"),
            "title": s.get("title"),
            "active": s.get("active"),
            "has_outrights": s.get("has_outrights"),
        }
        for s in (data or [])
    ]
    return pd.DataFrame(rows)


def fetch_match_odds(
    sport: str = WORLD_CUP_SPORT,
    *,
    regions: str = "eu",
    aggregate: str = "avg",
    cache: bool = True,
    ttl_hours: float = 1.0,
    neutral_site: bool = True,
    competition: str = "FIFA World Cup",
) -> pd.DataFrame:
    """Upcoming/live fixtures with 1X2 (h2h) decimal odds, in canonical columns.

    aggregate: "avg" averages each outcome's decimal odds across books (a robust
    market line / prior); "best" takes the max price per outcome (what you would
    actually bet into -- use this for the edge scanner).

    QUOTA: The Odds API bills 1 request unit PER REGION per call, so ``regions``
    multiplies cost against the ~500/month free tier. Default is a single region
    (``"eu"``, the widest book coverage); pass ``regions="us,uk,eu"`` only when
    you specifically want best-price-across-markets and accept the 3x cost. The
    short ``ttl_hours`` keeps live prices fresh; widen it to spare quota when
    prices are not moving.

    Returns columns: date, home_team, away_team, home_odds, draw_odds, away_odds,
    competition, competition_type, neutral_site, is_completed(False), plus n_books.
    Empty frame if the sport has no current games.
    """
    if aggregate not in ("avg", "best"):
        raise ValueError("aggregate must be 'avg' or 'best'")
    n_regions = len([r for r in regions.split(",") if r.strip()])
    if n_regions > 1:
        print(f"[odds_api] regions={regions!r} costs {n_regions} quota units this call.")
    games = _api_get(
        f"/sports/{sport}/odds",
        {"regions": regions, "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"},
        cache=cache,
        ttl_hours=ttl_hours,
    )
    rows = []
    for game in games or []:
        home = game.get("home_team")
        away = game.get("away_team")
        if not home or not away:
            continue
        # Collect each outcome's prices across all books.
        prices: dict[str, list[float]] = {home: [], away: [], _DRAW: []}
        for bk in game.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for oc in market.get("outcomes", []):
                    name, price = oc.get("name"), oc.get("price")
                    if name in prices and isinstance(price, (int, float)) and price > 1:
                        prices[name].append(float(price))
        if not all(prices[k] for k in (home, away, _DRAW)):
            continue  # need a complete 1X2 line
        agg = (lambda v: max(v)) if aggregate == "best" else (lambda v: sum(v) / len(v))
        rows.append(
            {
                "date": pd.to_datetime(game.get("commence_time"), errors="coerce", utc=True),
                "home_team": normalize_team_name(home),
                "away_team": normalize_team_name(away),
                "home_odds": round(agg(prices[home]), 4),
                "draw_odds": round(agg(prices[_DRAW]), 4),
                "away_odds": round(agg(prices[away]), 4),
                "competition": competition,
                "competition_type": "tournament",
                "neutral_site": neutral_site,
                "is_completed": False,
                "n_books": len(prices[home]),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_outrights(
    sport: str = WORLD_CUP_WINNER_SPORT,
    *,
    regions: str = "eu",
    aggregate: str = "avg",
    cache: bool = True,
    ttl_hours: float = 6.0,
) -> dict[str, float]:
    """De-vigged tournament-WINNER (futures) probabilities, ``{team: prob}``.

    The outright market is the bookmakers' direct champion-probability estimate --
    the single best title forecast available, and the market is the ceiling no
    model has beaten OOS here. Odds are aggregated across books then de-vigged by
    proportional normalisation (outright overrounds are large), so the returned
    probabilities sum to 1. Empty dict if the market has no prices / no key.
    Futures move slowly, so the default TTL is long to spare the free-tier quota.
    """
    games = _api_get(
        f"/sports/{sport}/odds",
        {"regions": regions, "markets": "outrights", "oddsFormat": "decimal",
         "dateFormat": "iso"},
        cache=cache,
        ttl_hours=ttl_hours,
    )
    prices: dict[str, list[float]] = {}
    for game in games or []:
        for bk in game.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for oc in market.get("outcomes", []):
                    name, price = oc.get("name"), oc.get("price")
                    if name and isinstance(price, (int, float)) and price > 1:
                        prices.setdefault(normalize_team_name(name), []).append(float(price))
    if not prices:
        return {}
    agg = (lambda v: max(v)) if aggregate == "best" else (lambda v: sum(v) / len(v))
    implied = {t: 1.0 / agg(v) for t, v in prices.items()}  # raw (overround > 1)
    total = sum(implied.values())
    return {t: p / total for t, p in implied.items()} if total > 0 else {}


def fetch_scores(
    sport: str = WORLD_CUP_SPORT,
    *,
    days_from: int = 3,
    cache: bool = True,
    ttl_hours: float = 1.0,
    competition: str = "FIFA World Cup",
    neutral_site: bool = True,
) -> pd.DataFrame:
    """Completed-match scores (canonical columns) for live results ingestion.

    days_from in 1..3 (free tier). Returns only completed games with both scores,
    as: date, home_team, away_team, home_score, away_score, competition,
    competition_type, neutral_site, is_completed(True). Empty if none completed.
    """
    data = _api_get(
        f"/sports/{sport}/scores",
        {"daysFrom": str(int(max(1, min(3, days_from)))), "dateFormat": "iso"},
        cache=cache,
        ttl_hours=ttl_hours,
    )
    rows = []
    for game in data or []:
        if not game.get("completed"):
            continue
        home, away = game.get("home_team"), game.get("away_team")
        scores = {s.get("name"): s.get("score") for s in (game.get("scores") or [])}
        try:
            hs, as_ = int(scores.get(home)), int(scores.get(away))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "date": pd.to_datetime(
                    game.get("commence_time"), errors="coerce", utc=True
                ),
                "home_team": normalize_team_name(home),
                "away_team": normalize_team_name(away),
                "home_score": hs,
                "away_score": as_,
                "competition": competition,
                "competition_type": "tournament",
                "neutral_site": neutral_site,
                "is_completed": True,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
    return df
