"""
Remote data sources + caching.

Free historical football data lives in a handful of well-known places:

- **martj42/international_results** (GitHub) -- every international men's result
  since 1872 in a clean, stable schema. The backbone of the national-team work.
- **football-data.co.uk** -- one CSV per club league per season (results +
  bookmaker odds) going back to the 1990s.
- **ClubElo** (api.clubelo.com) -- daily club Elo ratings (best-effort).
- **FIFA ranking** -- no stable free CSV endpoint; left best-effort.

Every remote fetch goes through `cached_get`, which downloads once to a hashed
file under `data/cache` and reuses it while fresh. Re-runs must NOT re-fetch (a
project non-negotiable -- the same rule applies to any paid source). All fetchers
degrade gracefully: a network failure raises an informative error rather than a
bare socket traceback, and `cached_get` falls back to a stale cached copy if one
exists.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import time
import urllib.error
import urllib.request

import pandas as pd


def _ssl_context() -> ssl.SSLContext | None:
    """Prefer certifi's CA bundle.

    Python on macOS frequently ships without OS trust roots wired in, so the
    default context raises CERTIFICATE_VERIFY_FAILED on perfectly valid HTTPS
    URLs. Using certifi's bundle (a dependency of most of our stack) fixes that
    for real runs. Returns None if certifi is unavailable, so urlopen falls back
    to the default context.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - certifi virtually always present
        return None

from . import schemas
from .loaders import load_international_results

# --------------------------------------------------------------------------- #
# Source URLs / constants
# --------------------------------------------------------------------------- #
INTERNATIONAL_RESULTS_URL: str = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
FOOTBALL_DATA_BASE: str = "https://www.football-data.co.uk/mmz4281"
# FIFA publishes rankings on its own site behind a JS app; there is no stable
# free CSV. Kept as a best-effort placeholder so callers can override it.
FIFA_RANKING_URL: str = "https://www.fifa.com/fifa-world-ranking/"
CLUBELO_URL: str = "http://api.clubelo.com"

# Default cache location (absolute-friendly: resolved relative to CWD by callers
# that pass a path, but the contract default is the project's data/cache).
DEFAULT_CACHE_DIR: str = "data/cache"

# Polite default request headers (some hosts 403 a bare urllib user-agent).
_HTTP_HEADERS = {
    "User-Agent": (
        "soccer_predictor/0.1 (+https://github.com/; research use) "
        "python-urllib"
    )
}


# --------------------------------------------------------------------------- #
# Competition-type classification
# --------------------------------------------------------------------------- #
# Substrings (lowercased) that flag a major-tournament finals competition.
_TOURNAMENT_KEYWORDS: tuple[str, ...] = (
    "world cup",
    "euro",
    "european championship",
    "championship",  # continental championships (e.g. "Continental Championship")
    "copa am",  # Copa America / Copa América
    "copa america",
    "cup of nations",  # Africa Cup of Nations, etc.
    "afc asian cup",
    "asian cup",
    "gold cup",
    "confederations cup",
    "nations league",
    "olympic",
    "finals",
)

# Substrings that flag a qualifier (checked AFTER tournament finals, because a
# qualifier name usually also contains the tournament name).
_QUALIFIER_KEYWORDS: tuple[str, ...] = (
    "qualif",  # qualification / qualifier / qualifying
    "preliminary",
    "play-off",
    "playoff",
)


def classify_competition_type(name: str) -> str:
    """Map a competition name onto one of ``schemas.COMPETITION_TYPES``.

    Order matters: a string like "FIFA World Cup qualification" must classify as
    a *qualifier*, not a tournament, so qualifier keywords are checked first.
    Friendlies are explicit; an unrecognised international-sounding competition
    defaults to "tournament"; anything else defaults to "club_league".
    """
    text = ("" if name is None else str(name)).strip().lower()
    if not text:
        return "club_league"

    if "friendly" in text or text == "friendlies":
        return "international_friendly"

    if any(k in text for k in _QUALIFIER_KEYWORDS):
        return "qualifier"

    if any(k in text for k in _TOURNAMENT_KEYWORDS):
        return "tournament"

    # Domestic club cups.
    if "cup" in text or "trophy" in text or "shield" in text:
        return "club_cup"

    # Named leagues / divisions.
    if any(k in text for k in ("league", "liga", "serie", "bundesliga", "division", "premier")):
        return "club_league"

    # Unknown but international-sounding (has a slash/qualifier vibe handled above);
    # fall back to a neutral default. We choose "tournament" only when the name
    # clearly is not a club competition -- otherwise club_league is the safer base.
    return "club_league"


# --------------------------------------------------------------------------- #
# Caching downloader
# --------------------------------------------------------------------------- #
def _cache_path(url: str, cache_dir: str) -> str:
    """Deterministic cache filename for a URL inside `cache_dir`."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    # Keep a hint of the original name for human-readable cache files.
    tail = url.rstrip("/").split("/")[-1] or "download"
    tail = "".join(c for c in tail if c.isalnum() or c in (".", "-", "_"))[:40]
    return os.path.join(cache_dir, f"{digest}_{tail}")


def cached_get(url: str, cache_dir: str = DEFAULT_CACHE_DIR, ttl_days: float = 7.0) -> str:
    """Download `url` to a hashed cache file and return the local path.

    If a fresh cached copy exists (younger than `ttl_days`) it is returned
    without any network access. On a network failure, a stale cached copy is
    returned if present; otherwise an informative ``RuntimeError`` is raised.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(url, cache_dir)

    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days < ttl_days and os.path.getsize(path) > 0:
            return path

    try:
        request = urllib.request.Request(url, headers=_HTTP_HEADERS)
        with urllib.request.urlopen(
            request, timeout=30, context=_ssl_context()
        ) as response:
            data = response.read()
        # Write atomically so a partial download never poisons the cache.
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        return path
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Graceful degradation: serve a stale copy rather than failing the run.
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        raise RuntimeError(
            f"Failed to download {url!r} and no cached copy is available "
            f"({type(exc).__name__}: {exc}). Check your network connection or "
            f"supply the file manually under {cache_dir!r}."
        ) from exc


# --------------------------------------------------------------------------- #
# Fetchers (all build on cached_get + a loader; all degrade gracefully)
# --------------------------------------------------------------------------- #
def fetch_international_results(cache: bool = True) -> pd.DataFrame:
    """Download + parse martj42 international results into a canonical frame."""
    ttl = 7.0 if cache else 0.0
    path = cached_get(INTERNATIONAL_RESULTS_URL, ttl_days=ttl)
    return load_international_results(path)


def fetch_football_data(
    league: str = "E0", season: str = "2324", cache: bool = True
) -> pd.DataFrame:
    """Download + parse one football-data.co.uk club CSV into a canonical frame.

    `league` is a football-data code (E0 = Premier League, D1 = Bundesliga, ...);
    `season` is the 4-digit short form (e.g. "2324" for 2023/24).
    """
    from .loaders import load_football_data  # local import avoids any cycle

    url = f"{FOOTBALL_DATA_BASE}/{season}/{league}.csv"
    ttl = 7.0 if cache else 0.0
    path = cached_get(url, ttl_days=ttl)
    return load_football_data(path)


def fetch_clubelo(team_or_date: str, cache: bool = True) -> pd.DataFrame:
    """Best-effort ClubElo fetch.

    The ClubElo API serves two CSV shapes:
      - ``/<TeamName>``  -> that club's full rating history.
      - ``/YYYY-MM-DD``  -> every club's rating on that date.
    Returns the raw CSV as a DataFrame (ClubElo's own columns; not coerced to the
    match schema). Degrades gracefully on network failure.
    """
    url = f"{CLUBELO_URL}/{team_or_date}"
    ttl = 1.0 if cache else 0.0  # ratings change daily
    try:
        path = cached_get(url, ttl_days=ttl)
        return pd.read_csv(path)
    except (RuntimeError, OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"ClubElo fetch for {team_or_date!r} failed ({exc})."
        ) from exc


__all__ = [
    "INTERNATIONAL_RESULTS_URL",
    "FOOTBALL_DATA_BASE",
    "FIFA_RANKING_URL",
    "CLUBELO_URL",
    "classify_competition_type",
    "cached_get",
    "fetch_international_results",
    "fetch_football_data",
    "fetch_clubelo",
]
