"""
FIFA World Cup 2026 live-pipeline helpers.

These tie the real data sources to the simulator so a single call produces
current champion / advancement probabilities that condition on the group games
already played.

Data provenance (deliberately NOT fabricated -- two independent methods cross-
checked, see docs/model_notes.md):
  * Group COMPOSITION is reconstructed from the actual 72 group-stage fixtures in
    martj42/international_results (each group is a complete K4 sub-graph of
    who-plays-whom -- the 12 connected components ARE the groups).
  * Group LETTERS (A-L) come from the official Wikipedia group pages and are
    verified to partition identically to the fixture reconstruction; the verified
    result is persisted to data/world_cup_2026_groups.csv.
  * Results so far come straight from the same martj42 feed (and can be topped up
    live from the Odds API /scores endpoint).

Everything degrades gracefully: if the committed groups CSV is present it is used
as the authoritative draw; otherwise we rebuild it from live data.
"""

from __future__ import annotations

import io
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import loaders, schemas
from .aliases import normalize_team_name
from .sources import _ssl_context, fetch_international_results

WC2026_COMPETITION = "FIFA World Cup"
GROUPS_CSV = Path("data/world_cup_2026_groups.csv")
FIXTURES_CSV = Path("data/world_cup_2026_fixtures.csv")
GROUP_LETTERS = list("ABCDEFGHIJKL")
_WIKI_GROUP_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{}"


# --------------------------------------------------------------------------- #
# Group composition
# --------------------------------------------------------------------------- #
def wc2026_matches(history: pd.DataFrame) -> pd.DataFrame:
    """The 72 World Cup 2026 group-stage matches from a results frame."""
    mask = (history["date"] >= "2026-06-01") & history["competition"].str.fullmatch(
        WC2026_COMPETITION, case=False, na=False
    )
    wc = history[mask].sort_values("date", kind="mergesort")
    return wc.head(72).reset_index(drop=True)


def reconstruct_groups(history: pd.DataFrame) -> list[frozenset[str]]:
    """Reconstruct the 12 groups as connected components of the fixture graph.

    During a group stage each team plays exactly the other three in its group, so
    the "played-against" graph over the 72 group matches is 12 disjoint K4s. The
    connected components are the groups -- derived purely from real fixtures.
    """
    gs = wc2026_matches(history)
    opp: dict[str, set[str]] = defaultdict(set)
    for r in gs.itertuples(index=False):
        opp[r.home_team].add(r.away_team)
        opp[r.away_team].add(r.home_team)
    seen: set[str] = set()
    comps: list[frozenset[str]] = []
    for team in sorted(opp):
        if team in seen:
            continue
        comp = {team}
        stack = [team]
        while stack:
            x = stack.pop()
            for y in opp[x] - comp:
                comp.add(y)
                stack.append(y)
        seen |= comp
        comps.append(frozenset(comp))
    # The 48-team format must reconstruct to exactly 12 groups of 4. A wrong
    # count/size means a knockout or cross-group fixture leaked into the first 72
    # (e.g. a postponement) and merged groups -- fail loud rather than return a
    # silently-wrong draw.
    bad = [sorted(c) for c in comps if len(c) != 4]
    if len(comps) != 12 or bad:
        raise RuntimeError(
            f"WC2026 group reconstruction produced {len(comps)} groups with sizes "
            f"{sorted(len(c) for c in comps)} (expected 12 groups of 4). "
            f"Non-4 components: {bad}. The group-stage fixture set may be "
            "incomplete or contaminated by non-group matches."
        )
    return comps


def fetch_labeled_groups_from_wikipedia() -> dict[str, list[str]]:
    """Official A-L group membership scraped from Wikipedia (certifi + lxml).

    Uses our own certifi-aware opener (pandas' read_html network path trips the
    macOS SSL-verify bug). Raises on failure; callers fall back to the CSV.
    """

    def _fetch(url: str) -> str:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 soccer-predictor"}
        )
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            return resp.read().decode("utf-8", "replace")

    def _clean(text: object) -> str:
        s = re.sub(r"\[[^\]]*\]", "", str(text))
        s = re.sub(r"\(.*?\)", "", s)
        return normalize_team_name(re.sub(r"\s+", " ", s).strip())

    labeled: dict[str, list[str]] = {}
    for g in GROUP_LETTERS:
        tables = pd.read_html(io.StringIO(_fetch(_WIKI_GROUP_URL.format(g))), flavor="lxml")
        for tb in tables:
            tcol = next((c for c in tb.columns if "team" in str(c).lower()), None)
            if tcol is None:
                continue
            teams: list[str] = []
            for v in tb[tcol].tolist():
                c = _clean(v)
                if (
                    c
                    and c.lower() not in ("team", "teams")
                    and "advance" not in c.lower()
                    and c not in teams
                ):
                    teams.append(c)
            if len(teams) >= 4:
                labeled[g] = teams[:4]
                break
    if len(labeled) != 12:
        raise RuntimeError(f"Only parsed {len(labeled)}/12 Wikipedia groups.")
    return labeled


def refresh_groups(out: Path = GROUPS_CSV, verify: bool = True) -> dict[str, list[str]]:
    """Fetch official labelled groups, optionally verify against the fixture
    reconstruction, and persist to ``out``. Returns the {letter: [teams]} map."""
    labeled = fetch_labeled_groups_from_wikipedia()
    if verify:
        history = fetch_international_results()
        recon = set(reconstruct_groups(history))
        wiki = {frozenset(v) for v in labeled.values()}
        if recon and wiki != recon:
            raise RuntimeError(
                "Wikipedia group labels do not match the fixture-reconstructed "
                f"groups ({len(wiki & recon)}/12 agree); refusing to write a "
                "possibly-wrong draw."
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(g, t) for g in GROUP_LETTERS for t in labeled[g]], columns=["group", "team"]
    ).to_csv(out, index=False)
    return labeled


def load_groups(refresh: bool = False) -> dict[str, list[str]]:
    """Authoritative WC2026 groups: the committed CSV, or rebuilt from live data."""
    if not refresh and GROUPS_CSV.exists():
        return loaders.load_groups(str(GROUPS_CSV))
    refresh_groups()
    return loaders.load_groups(str(GROUPS_CSV))


# --------------------------------------------------------------------------- #
# Fixtures (with live results)
# --------------------------------------------------------------------------- #
def build_fixtures(
    history: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
    extra_results: pd.DataFrame | None = None,
    out: Path | None = FIXTURES_CSV,
) -> pd.DataFrame:
    """Canonical fixtures frame for the 72 group games, scores filled where played.

    ``extra_results`` (e.g. from odds_api.fetch_scores) tops up results that the
    history feed has not yet absorbed, matched on (home, away).
    """
    groups = groups or load_groups()
    team_to_group = {t: g for g, teams in groups.items() for t in teams}
    gs = wc2026_matches(history)

    extra_map: dict[tuple[str, str], tuple[int, int]] = {}
    if extra_results is not None and len(extra_results):
        for r in extra_results.itertuples(index=False):
            if pd.notna(getattr(r, "home_score", None)) and pd.notna(
                getattr(r, "away_score", None)
            ):
                extra_map[(r.home_team, r.away_team)] = (
                    int(r.home_score),
                    int(r.away_score),
                )

    fixture_pairs = {(r.home_team, r.away_team) for r in gs.itertuples(index=False)}
    rows = []
    used_extra: set[tuple[str, str]] = set()
    for i, r in enumerate(gs.itertuples(index=False), start=1):
        group = team_to_group.get(r.home_team) or team_to_group.get(r.away_team) or ""
        hs, as_ = r.home_score, r.away_score
        if (pd.isna(hs) or pd.isna(as_)) and (r.home_team, r.away_team) in extra_map:
            hs, as_ = extra_map[(r.home_team, r.away_team)]
            used_extra.add((r.home_team, r.away_team))
        rows.append(
            {
                "match_id": i,
                "date": pd.Timestamp(r.date).date() if pd.notna(r.date) else "",
                "group": group,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "neutral": True,
                "home_score": "" if pd.isna(hs) else int(hs),
                "away_score": "" if pd.isna(as_) else int(as_),
            }
        )
    # Surface live results that matched NO fixture (usually a team-name mismatch),
    # so a silently-dropped score never goes unnoticed.
    dropped = sorted(set(extra_map) - used_extra)
    if dropped:
        import sys

        print(
            f"[world_cup] {len(dropped)} live result(s) matched no group fixture "
            f"(likely a team-name mismatch -- add aliases): {dropped}",
            file=sys.stderr,
        )

    fixtures = pd.DataFrame(rows)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fixtures.to_csv(out, index=False)
    return fixtures


# --------------------------------------------------------------------------- #
# One-call live simulation
# --------------------------------------------------------------------------- #
def simulate(
    model=None,
    *,
    model_name: str = "elo",
    n_simulations: int = 10000,
    seed: int = 7,
    refresh_groups_data: bool = False,
    use_odds_api_scores: bool = False,
) -> pd.DataFrame:
    """Fetch real data, condition on games played, and Monte-Carlo the rest.

    Returns the standard per-team stage/champion probability table. The model is
    fit on ALL completed matches (history incl. games played so far) -- that is
    past information relative to the matches still to be simulated, so it is not
    leakage.
    """
    from ..models import DixonColes, EloGoalsModel, PoissonXG, default_ensemble
    from ..simulation.tournament import simulate_tournament

    history = fetch_international_results()
    groups = load_groups(refresh=refresh_groups_data)

    extra = None
    if use_odds_api_scores:
        try:
            from . import odds_api

            extra = odds_api.fetch_scores()
        except Exception as exc:  # graceful: live scores are a bonus, not required
            print(f"[world_cup] odds-api scores unavailable ({exc}); using history only.")

    fixtures_df = build_fixtures(history, groups, extra_results=extra)
    fixtures = loaders.load_fixtures(str(FIXTURES_CSV))

    if model is None:
        factory = {
            "elo": EloGoalsModel,
            "dixon-coles": DixonColes,
            "xg": PoissonXG,
            "ensemble": default_ensemble,
        }.get(model_name, EloGoalsModel)
        model = factory()
        model.fit(schemas.completed_matches(history).copy())

    return simulate_tournament(
        model, groups, fixtures=fixtures, n_simulations=n_simulations, seed=seed
    )
