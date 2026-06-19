"""
Player / squad strength (optional, CURRENT-tournament only).

Squad quality is a real signal the results-based Elo can miss (a golden
generation peaking, or a depleted squad), but it has two hard limits you must
keep in mind:

  1. It is NOT freely available historically. Past-tournament squad ratings /
     market values are not in the free results feed, so squad strength CANNOT be
     backtested on past World Cups -- treat it as a live-only overlay.
  2. There is no clean official "national-team strength" number. The robust path
     is to SUPPLY one via CSV from whatever source you trust (FotMob average
     player rating, Transfermarkt squad market value, FIFA ranking points, your
     own), and let this module convert it to a zero-mean Elo-point adjustment that
     ``EloGoalsModel(squad_strength=..., squad_weight=...)`` consumes.

``fetch_fotmob_team_strength`` is a BEST-EFFORT convenience against FotMob's
unofficial API; its endpoints drift, so it degrades loudly rather than inventing
numbers. Prefer the CSV path for anything you rely on.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .aliases import normalize_team_name
from .sources import _ssl_context


def load_squad_strength_csv(path: str | Path, value_col: str | None = None) -> dict[str, float]:
    """Load a team -> strength map from a CSV.

    Expects a ``team`` column and a numeric value column (auto-detected from
    {strength, rating, value, squad_value, points} unless ``value_col`` is given).
    Team names are normalised through the alias map so they line up with the model.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    team_col = cols.get("team") or cols.get("country") or list(df.columns)[0]
    if value_col is None:
        for cand in ("strength", "rating", "value", "squad_value", "points", "score"):
            if cand in cols:
                value_col = cols[cand]
                break
    if value_col is None:
        # Fall back to the first numeric column that is not the team column.
        numeric = [c for c in df.columns if c != team_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError(
                f"No numeric strength column found in {path!r}; pass value_col=."
            )
        value_col = numeric[0]
    out: dict[str, float] = {}
    for r in df[[team_col, value_col]].itertuples(index=False):
        name = normalize_team_name(r[0])
        try:
            out[name] = float(r[1])
        except (TypeError, ValueError):
            continue
    return out


def to_elo_adjustment(
    strength: dict[str, float], spread: float = 60.0, log: bool = False
) -> dict[str, float]:
    """Convert arbitrary strength scores to zero-mean, z-scored Elo points.

    ``spread`` is the Elo points one standard deviation of strength is worth, so a
    typical strong/weak squad shifts the rating by roughly +/- ``spread``. Set
    ``log=True`` for heavy-tailed inputs like squad MARKET VALUE (a few elite
    squads dwarf the rest); a log transform keeps one giant value from swamping
    the z-score. The result is meant to be passed straight to
    ``EloGoalsModel(squad_strength=...)`` with a ``squad_weight`` you tune.
    """
    if not strength:
        return {}
    items = list(strength.items())
    raw = np.array([v for _, v in items], dtype=float)
    if log:
        raw = np.log1p(np.clip(raw, 0.0, None))
    mean = float(raw.mean())
    std = float(raw.std()) or 1.0
    return {team: (val - mean) / std * spread for (team, _), val in zip(items, raw)}


# --------------------------------------------------------------------------- #
# Transfermarkt national-team squad market value (best-effort, current only)
# --------------------------------------------------------------------------- #
_TM_URL = "https://www.transfermarkt.com/weltrangliste-der-nationalmannschaften/weltrangliste/statistik"
_TM_CACHE = Path("data/cache/transfermarkt_values.csv")


def _parse_tm_value(text: object) -> float | None:
    """Parse a Transfermarkt value string ('€807.50m', '€1.23bn') to € millions."""
    try:
        if pd.isna(text):
            return None
    except (TypeError, ValueError):
        pass
    s = str(text).strip().lower().replace("€", "").replace(",", "")
    if not s or s in ("-", "–", "n/a", "nan"):
        return None
    mult = 1.0
    if s.endswith("bn"):
        mult, s = 1000.0, s[:-2]
    elif s.endswith("m"):
        mult, s = 1.0, s[:-1]
    elif s.endswith("k") or s.endswith("th."):
        mult, s = 0.001, s.rstrip("th.").rstrip("k")
    try:
        return float(s) * mult
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# League tiers (level of competition a player is tested at)
# --------------------------------------------------------------------------- #
# Tier weight in [0, 1]: 1.0 = elite. A judgement-based scheme (UEFA-coefficient
# /strength intuition), NOT binary top-5 -- the strong non-top-5 leagues the user
# flagged (Eredivisie, Primeira, Brazilian, Argentine, Belgian, Turkish, Saudi,
# Championship...) sit well above the long tail. Approximate; revisit each season.
LEAGUE_TIERS: dict[str, float] = {
    "premier league": 1.00, "laliga": 1.00, "bundesliga": 1.00,
    "serie a": 1.00, "ligue 1": 1.00,
    "primeira liga": 0.80, "eredivisie": 0.80, "brazil serie a": 0.80,
    "championship": 0.74, "belgian pro league": 0.74, "saudi pro league": 0.74,
    "super lig": 0.72, "argentine primera": 0.72, "liga mx": 0.70,
    "mls": 0.66, "scottish premiership": 0.64, "swiss super league": 0.64,
    "austrian bundesliga": 0.64, "greek super league": 0.62, "russian premier": 0.62,
    "ligue 2": 0.60, "2. bundesliga": 0.62, "serie b": 0.60, "laliga 2": 0.60,
    "j1 league": 0.62, "k league": 0.60, "croatian": 0.58, "ukrainian": 0.58,
}
DEFAULT_LEAGUE_TIER = 0.45  # unknown / lower-tier league

# Curated club -> league for clubs that realistically appear in World Cup squads
# (normalised keys; see _norm_club). Dated mid-2026; approximate. Unmatched clubs
# fall back to DEFAULT_LEAGUE_TIER, and the match rate is reported so you can see
# how much of a squad was actually classified.
CLUB_LEAGUE: dict[str, str] = {
    # Premier League
    "manchester city": "premier league", "arsenal": "premier league",
    "liverpool": "premier league", "chelsea": "premier league",
    "manchester united": "premier league", "tottenham": "premier league",
    "newcastle": "premier league", "aston villa": "premier league",
    "west ham": "premier league", "brighton": "premier league",
    "crystal palace": "premier league", "everton": "premier league",
    "nottingham forest": "premier league", "brentford": "premier league",
    "fulham": "premier league", "bournemouth": "premier league",
    "wolverhampton": "premier league", "wolves": "premier league",
    "leeds": "premier league", "sunderland": "premier league",
    "burnley": "premier league",
    # LaLiga
    "real madrid": "laliga", "barcelona": "laliga", "atletico madrid": "laliga",
    "athletic bilbao": "laliga", "real sociedad": "laliga", "villarreal": "laliga",
    "real betis": "laliga", "sevilla": "laliga", "valencia": "laliga",
    "girona": "laliga", "celta vigo": "laliga", "osasuna": "laliga",
    # Bundesliga
    "bayern munich": "bundesliga", "borussia dortmund": "bundesliga",
    "rb leipzig": "bundesliga", "bayer leverkusen": "bundesliga",
    "vfb stuttgart": "bundesliga", "eintracht frankfurt": "bundesliga",
    "borussia monchengladbach": "bundesliga", "wolfsburg": "bundesliga",
    "werder bremen": "bundesliga", "hoffenheim": "bundesliga",
    # Serie A
    "inter": "serie a", "ac milan": "serie a", "milan": "serie a",
    "juventus": "serie a", "napoli": "serie a", "roma": "serie a",
    "lazio": "serie a", "atalanta": "serie a", "fiorentina": "serie a",
    "bologna": "serie a", "torino": "serie a",
    # Ligue 1
    "paris saint-germain": "ligue 1", "paris saint germain": "ligue 1",
    "marseille": "ligue 1", "monaco": "ligue 1", "lyon": "ligue 1",
    "lille": "ligue 1", "nice": "ligue 1", "rennes": "ligue 1", "lens": "ligue 1",
    # Primeira (Portugal)
    "benfica": "primeira liga", "porto": "primeira liga",
    "sporting cp": "primeira liga", "sporting": "primeira liga",
    "braga": "primeira liga",
    # Eredivisie
    "ajax": "eredivisie", "ajax amsterdam": "eredivisie", "psv": "eredivisie",
    "psv eindhoven": "eredivisie", "feyenoord": "eredivisie", "az alkmaar": "eredivisie",
    # Brazil
    "flamengo": "brazil serie a", "palmeiras": "brazil serie a",
    "botafogo": "brazil serie a", "sao paulo": "brazil serie a",
    "corinthians": "brazil serie a", "fluminense": "brazil serie a",
    "gremio": "brazil serie a", "internacional": "brazil serie a",
    "atletico mineiro": "brazil serie a", "cruzeiro": "brazil serie a",
    # Argentina
    "river plate": "argentine primera", "boca juniors": "argentine primera",
    "racing club": "argentine primera", "independiente": "argentine primera",
    # Belgium / Turkey / Saudi / others
    "club brugge": "belgian pro league", "anderlecht": "belgian pro league",
    "royal antwerp": "belgian pro league", "genk": "belgian pro league",
    "galatasaray": "super lig", "fenerbahce": "super lig",
    "besiktas": "super lig", "trabzonspor": "super lig",
    "al-nassr": "saudi pro league", "al nassr": "saudi pro league",
    "al-hilal": "saudi pro league", "al hilal": "saudi pro league",
    "al-ittihad": "saudi pro league", "al-ahli": "saudi pro league",
    "celtic": "scottish premiership", "rangers": "scottish premiership",
    "inter miami": "mls", "los angeles fc": "mls", "lafc": "mls",
}


def _norm_club(name: object) -> str:
    """Normalise a club name for CLUB_LEAGUE lookup (lower, strip FC/CF/accents)."""
    import unicodedata

    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[.\-']", " ", s)
    drop = {"fc", "cf", "afc", "sc", "ac", "ssc", "as", "ss", "rc", "cd", "ud",
            "club", "de", "calcio", "1899", "1909", "04", "05", "1846", "1860"}
    toks = [t for t in s.split() if t and t not in drop]
    return " ".join(toks)


def _club_tier(club: object) -> tuple[float, bool]:
    """(tier weight, matched?) for a club via CLUB_LEAGUE -> LEAGUE_TIERS."""
    key = _norm_club(club)
    league = CLUB_LEAGUE.get(key)
    if league is None:
        # try a couple of common alias collapses
        for k, v in CLUB_LEAGUE.items():
            if key and (key in k or k in key):
                league = v
                break
    if league is None:
        return DEFAULT_LEAGUE_TIER, False
    return LEAGUE_TIERS.get(league, DEFAULT_LEAGUE_TIER), True


def _age_value_retention(age: float) -> float:
    """Approx ratio of market VALUE to peak-ability value at a given age.

    Market value distorts ABILITY in BOTH directions, symmetrically here (~0.06
    per year off the peak window): a 34-year-old elite is cheap but still elite
    (boost), while a 19-year-old carries a big POTENTIAL premium -- his price
    bakes in future ability + resale upside, not what he is now (trim). Dividing
    value by this ratio recovers an *ability-equivalent* value. Heuristic (it
    cannot be validated on free data; FIFA `overall` already separates current
    ability from potential, which is why FIFA is the primary, un-aged source):

      <24  : 1.0 + 0.06/yr below 24, capped 1.6 (e.g. 18 -> 1.36, 21 -> 1.18)
      24-28: 1.0 (peak window: value ~ ability)
      >28  : 1.0 - 0.06/yr, floored 0.25 (e.g. 32 -> 0.76, 38 -> 0.40)
    """
    a = float(age)
    if a < 24.0:
        return min(1.6, 1.0 + 0.06 * (24.0 - a))
    if a <= 28.0:
        return 1.0
    return max(0.25, 1.0 - 0.06 * (a - 28.0))


def _parse_age(text: object) -> float | None:
    m = re.search(r"\((\d{2})\)", str(text))
    return float(m.group(1)) if m else None


def _transfermarkt_team_ids(pages: int) -> dict[str, str]:
    """{canonical_team_name: verein_id} from the ranking pages.

    The ranking table's Nation column (English names) is aligned, in row order,
    with the ordered ``/startseite/verein/<id>`` links (each team links twice --
    flag + name -- so consecutive duplicates are collapsed). The squad page itself
    ignores the URL slug, so only the id is needed downstream.
    """
    import io
    import urllib.request

    ids: dict[str, str] = {}
    for page in range(1, int(pages) + 1):
        url = _TM_URL if page == 1 else f"{_TM_URL}/plus/?page={page}"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) soccer-predictor"}
            )
            with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as resp:
                html = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError):
            continue
        seq = re.findall(r"/startseite/verein/(\d+)", html)
        ordered: list[str] = []
        for vid in seq:
            if not ordered or ordered[-1] != vid:
                ordered.append(vid)
        try:
            tables = [
                t for t in pd.read_html(io.StringIO(html), flavor="lxml")
                if any("nation" in str(c).lower() for c in t.columns)
            ]
        except ValueError:
            continue
        if not tables:
            continue
        t = tables[0]
        nat_col = next(c for c in t.columns if "nation" in str(c).lower())
        nations = [normalize_team_name(n) for n in t[nat_col]]
        for nat, vid in zip(nations, ordered):
            if nat:
                ids.setdefault(nat, vid)
    return ids


_SQUAD_CLUB_RE = re.compile(
    r'<td class="zentriert"><a title="([^"]+)" href="/[^"]+/startseite/verein/\d+"><img'
)
_SQUAD_METRICS_CACHE = Path("data/cache/transfermarkt_squad_metrics.csv")


def fetch_transfermarkt_squad_metrics(
    pages: int = 3, cache: bool = True, ttl_days: float = 7.0
) -> pd.DataFrame:
    """Per-team squad metrics from Transfermarkt squad pages, in ONE pass.

    Returns a DataFrame: team, value (raw squad €m), ability (age-adjusted €m),
    league_strength (value-weighted mean league tier in [0,1]), n_players,
    club_match_rate (share of the squad whose club was classified to a league --
    a data-quality flag, since unmatched clubs fall back to the default tier).

    age + market value parse from the table; the player's CLUB is pulled from the
    raw HTML club cell (it is an image in the parsed table) and mapped to a league
    tier via CLUB_LEAGUE/LEAGUE_TIERS. Best-effort, current-only, cached. Degrades
    loudly if nothing parses.
    """
    import io
    import time
    import urllib.request

    if cache and _SQUAD_METRICS_CACHE.exists():
        age_days = (time.time() - _SQUAD_METRICS_CACHE.stat().st_mtime) / 86400.0
        if age_days < ttl_days and _SQUAD_METRICS_CACHE.stat().st_size > 0:
            df = pd.read_csv(_SQUAD_METRICS_CACHE)
            df["team"] = df["team"].map(normalize_team_name)
            return df

    team_ids = _transfermarkt_team_ids(pages)
    rows = []
    for team, vid in team_ids.items():
        url = f"https://www.transfermarkt.com/x/kader/verein/{vid}/plus/1"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) soccer-predictor"}
            )
            with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as resp:
                html = resp.read().decode("utf-8", "replace")
            tables = [
                t for t in pd.read_html(io.StringIO(html), flavor="lxml")
                if "Market value" in [str(c) for c in t.columns]
            ]
            if not tables:
                continue
            t = tables[0].reset_index(drop=True)
            # The /plus/1 table has several sub-rows per player; only value-bearing
            # rows are players, and they line up (in order) with the club cells
            # parsed from the raw HTML. Filter to player rows FIRST, then zip --
            # indexing the club list by raw table position misaligns them.
            clubs = _SQUAD_CLUB_RE.findall(html)  # one per player, in order
            players_rc = [
                (_parse_age(row.get("Date of birth/Age")),
                 _parse_tm_value(row.get("Market value")))
                for _, row in t.iterrows()
            ]
            players_rc = [(a, v) for a, v in players_rc if v is not None]
            value = ability = tier_num = tier_den = 0.0
            matched = total = 0
            for idx, (age, val) in enumerate(players_rc):
                total += 1
                value += val
                ability += val / _age_value_retention(age) if age is not None else val
                club = clubs[idx] if idx < len(clubs) else None
                tier, ok = _club_tier(club) if club is not None else (DEFAULT_LEAGUE_TIER, False)
                matched += int(ok)
                tier_num += val * tier
                tier_den += val
            if total == 0:
                continue
            rows.append({
                "team": team,
                "value": round(value, 1),
                "ability": round(ability, 1),
                "league_strength": round(tier_num / tier_den, 4) if tier_den else DEFAULT_LEAGUE_TIER,
                "n_players": total,
                "club_match_rate": round(matched / total, 3),
            })
        except (urllib.error.URLError, OSError, ValueError):
            continue
    if not rows:
        raise RuntimeError(
            "Transfermarkt returned no usable squads (layout may have changed). "
            "Use --metric value, or supply a CSV via load_squad_strength_csv."
        )
    df = pd.DataFrame(rows)
    _SQUAD_METRICS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_SQUAD_METRICS_CACHE, index=False)
    return df


def fetch_transfermarkt_ability(
    pages: int = 3, cache: bool = True, ttl_days: float = 7.0
) -> dict[str, float]:
    """Age-adjusted squad ability (€m-equivalent); wrapper over the unified fetch.
    Closer to player quality than raw market value (which is age-discounted)."""
    df = fetch_transfermarkt_squad_metrics(pages, cache, ttl_days)
    return dict(zip(df["team"].map(normalize_team_name), df["ability"]))


def fetch_transfermarkt_league_strength(
    pages: int = 3, cache: bool = True, ttl_days: float = 7.0
) -> dict[str, float]:
    """Value-weighted mean league tier per squad ([0,1]); level-of-competition."""
    df = fetch_transfermarkt_squad_metrics(pages, cache, ttl_days)
    return dict(zip(df["team"].map(normalize_team_name), df["league_strength"]))


def fetch_transfermarkt_values(
    pages: int = 3, cache: bool = True, ttl_days: float = 7.0
) -> dict[str, float]:
    """National-team squad MARKET VALUE (€ millions) from Transfermarkt.

    This is the genuinely orthogonal "player strength" signal (it is not results-
    based, unlike Elo / FIFA points). Best-effort scrape of the national-team
    ranking table; ``pages`` x 25 teams (3 pages covers every World Cup side).
    Cached to ``data/cache/transfermarkt_values.csv``. CURRENT-only -- there is no
    free historical squad value, so it cannot be backtested on past World Cups.
    Degrades loudly (raises) rather than inventing numbers.
    """
    import io
    import time
    import urllib.request

    if cache and _TM_CACHE.exists():
        age_days = (time.time() - _TM_CACHE.stat().st_mtime) / 86400.0
        if age_days < ttl_days and _TM_CACHE.stat().st_size > 0:
            df = pd.read_csv(_TM_CACHE)
            return dict(zip(df["team"].map(normalize_team_name), df["strength"]))

    out: dict[str, float] = {}
    for page in range(1, int(pages) + 1):
        url = _TM_URL if page == 1 else f"{_TM_URL}/plus/?page={page}"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) soccer-predictor"}
            )
            with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as resp:
                html = resp.read().decode("utf-8", "replace")
            tables = [t for t in pd.read_html(io.StringIO(html), flavor="lxml") if len(t) >= 20]
            if not tables:
                continue
            t = tables[0]
            nat_col = next(c for c in t.columns if "nation" in str(c).lower())
            val_col = next(c for c in t.columns if "value" in str(c).lower())
            for nat, val in zip(t[nat_col], t[val_col]):
                v = _parse_tm_value(val)
                if v is not None:
                    out[normalize_team_name(nat)] = v
        except (urllib.error.URLError, OSError, ValueError, StopIteration):
            continue
    if not out:
        raise RuntimeError(
            "Transfermarkt returned no usable squad values (page layout may have "
            "changed, or the request was blocked). Supply a CSV via "
            "load_squad_strength_csv instead."
        )
    _TM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"team": list(out), "strength": list(out.values())}).to_csv(
        _TM_CACHE, index=False
    )
    return out


def fetch_fotmob_team_strength(
    team_ids: dict[str, int] | None = None,
    *,
    cache_dir: str = "data/cache/fotmob",
) -> dict[str, float]:
    """BEST-EFFORT squad strength from FotMob (unofficial API, current only).

    FotMob exposes team data at ``https://www.fotmob.com/api/teams?id=<id>``; this
    pulls each team's headline rating where present. You must supply ``team_ids``
    ({team_name: fotmob_team_id}) because there is no stable free name->id lookup.
    Endpoints drift -- on failure this raises a clear error rather than guessing,
    and partial results are returned for the ids that did resolve.

    This is a convenience only. For anything you rely on, export a CSV and use
    ``load_squad_strength_csv``.
    """
    if not team_ids:
        raise ValueError(
            "fetch_fotmob_team_strength needs team_ids={name: fotmob_id}. "
            "FotMob has no stable free name->id lookup; or supply a CSV via "
            "load_squad_strength_csv instead."
        )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    out: dict[str, float] = {}
    failures: list[str] = []
    for name, tid in team_ids.items():
        url = f"https://www.fotmob.com/api/teams?id={int(tid)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "soccer-predictor"})
            with urllib.request.urlopen(req, timeout=15, context=_ssl_context()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            rating = _extract_fotmob_rating(data)
            if rating is not None:
                out[normalize_team_name(name)] = rating
            else:
                failures.append(name)
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            failures.append(name)
    if not out:
        raise RuntimeError(
            "FotMob returned no usable ratings (endpoint may have changed). "
            f"Failed: {failures}. Use load_squad_strength_csv with a CSV instead."
        )
    if failures:
        print(f"[fotmob] no rating for {len(failures)} team(s): {failures}")
    return out


def _extract_fotmob_rating(data: dict) -> float | None:
    """Defensively pull a headline team rating from a FotMob payload."""
    # Field names drift; probe a few plausible locations without assuming any.
    for path in (("table", 0, "data", "rating"), ("details", "rating"), ("rating",)):
        node = data
        ok = True
        for key in path:
            try:
                node = node[key]
            except (KeyError, IndexError, TypeError):
                ok = False
                break
        if ok and isinstance(node, (int, float)):
            return float(node)
    return None
