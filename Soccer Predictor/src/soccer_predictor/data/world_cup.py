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
from itertools import combinations
from pathlib import Path

import pandas as pd

from . import loaders, schemas
from .aliases import normalize_team_name
from .sources import _ssl_context, fetch_international_results

WC2026_COMPETITION = "FIFA World Cup"
GROUPS_CSV = Path("data/world_cup_2026_groups.csv")
FIXTURES_CSV = Path("data/world_cup_2026_fixtures.csv")
# Durable ledger of every completed WC2026 group game ever observed, from ANY
# feed. The point is monotonicity: martj42 lags (and is cached up to 7 days) and
# the Odds-API /scores window only spans the last ~3 days, so on any single run a
# played game can be absent from BOTH live feeds. Once a result lands here it is
# never lost, so the live tab keeps every game it has ever seen locked in.
RESULTS_CSV = Path("data/world_cup_2026_results.csv")
# During a live tournament, refetch the martj42 feed if the cached copy is older
# than this so newly-played games appear promptly (the 7-day default is too slow
# while games are happening daily). The Odds-API feed + ledger cover anything
# still missing in between.
_WC_HISTORY_TTL_DAYS = 0.25  # ~6 hours
GROUP_LETTERS = list("ABCDEFGHIJKL")
_WIKI_GROUP_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{}"


def _pair_key(home: object, away: object) -> tuple[str, str]:
    """Order-insensitive key for a group pairing (each pair meets exactly once).

    World Cup group games are at neutral venues, so different feeds disagree on
    which side is "home". Keying on the sorted pair lets a result match its
    fixture regardless of orientation (the score still travels with its team).
    """
    return tuple(sorted((str(home), str(away))))


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


# --------------------------------------------------------------------------- #
# Durable results ledger (so every played game stays locked in)
# --------------------------------------------------------------------------- #
def _empty_results() -> pd.DataFrame:
    return schemas.validate_matches(
        pd.DataFrame(
            columns=[
                "date", "home_team", "away_team", "home_score", "away_score",
                "competition", "competition_type", "neutral_site",
            ]
        ),
        require_scores=True,
    )


def group_pair_keys(groups: dict[str, list[str]] | None = None) -> set[tuple[str, str]]:
    """The order-insensitive keys of the 72 group-stage pairings.

    Each group is a round-robin of 4 teams (6 pairings), so this is the exact set
    of pairs that may appear as group games. Used to SCOPE the results ledger to
    the group stage: a knockout-round rematch of a group pair would otherwise
    collide on the pair key and overwrite the group result, and a stray non-group
    game would persist forever. ``groups`` defaults to the authoritative draw.
    """
    groups = groups or load_groups()
    keys: set[tuple[str, str]] = set()
    for teams in groups.values():
        for a, b in combinations(teams, 2):
            keys.add(_pair_key(a, b))
    return keys


def _wc_scored_pair_keys(frame: pd.DataFrame | None) -> set[tuple[str, str]]:
    """Order-insensitive keys of the completed WC2026 games already in `frame`."""
    if frame is None or not len(frame) or "competition" not in frame.columns:
        return set()
    date = pd.to_datetime(frame["date"], errors="coerce")
    mask = (
        (date >= "2026-06-01")
        & frame["competition"].astype("string").str.fullmatch(
            WC2026_COMPETITION, case=False, na=False
        )
        & frame["home_score"].notna()
        & frame["away_score"].notna()
    )
    sub = frame[mask]
    return {_pair_key(h, a) for h, a in zip(sub["home_team"], sub["away_team"])}


def stored_results(store: Path = RESULTS_CSV) -> pd.DataFrame:
    """Read the durable results ledger (canonical completed-match schema).

    Degrades gracefully for READ paths: a missing or unreadable file yields an
    empty frame rather than raising. (The WRITE path, update_results_store,
    separately refuses to clobber a file it cannot parse -- see _ledger_corrupt.)
    """
    if not Path(store).exists():
        return _empty_results()
    try:
        raw = pd.read_csv(store)
    except (OSError, ValueError, pd.errors.ParserError):
        return _empty_results()
    if raw.empty:
        return _empty_results()
    try:
        return schemas.validate_matches(raw, require_scores=True)
    except schemas.SchemaError:
        return _empty_results()


def _ledger_corrupt(store: Path) -> bool:
    """True if the ledger file exists with content we cannot parse.

    A genuinely empty (header-only) ledger is NOT corrupt; only an unreadable or
    schema-invalid file is. Used to avoid overwriting recoverable bytes.
    """
    store = Path(store)
    if not store.exists() or store.stat().st_size == 0:
        return False
    try:
        raw = pd.read_csv(store)
    except (OSError, ValueError, pd.errors.ParserError):
        return True
    if raw.empty:
        return False
    try:
        schemas.validate_matches(raw, require_scores=True)
    except schemas.SchemaError:
        return True
    return False


def completed_wc_results(
    history: pd.DataFrame | None = None,
    extra_results: pd.DataFrame | None = None,
    *,
    use_store: bool = True,
    store: Path = RESULTS_CSV,
    valid_pairs: set[tuple[str, str]] | None = None,
    group_end: object = None,
) -> pd.DataFrame:
    """Every completed WC2026 group game seen so far, unioned across all feeds.

    Combines (a) the durable on-disk ledger at ``store``, (b) the scored WC games
    in the martj42 ``history`` (if given), and (c) live ``extra_results`` (e.g.
    the Odds-API ``/scores`` frame), de-duplicated on the order-insensitive
    pairing so the same game in two feeds (possibly with home/away swapped)
    collapses to one row.

    Precedence on a tie is martj42 history > live extra > ledger, i.e. the
    official feed wins over a possibly-provisional live score, which wins over a
    stale stored one.

    Two guards keep this to the GROUP stage so a knockout-round result cannot
    corrupt a group result that shares the same team pair:
      * ``valid_pairs`` (e.g. ``group_pair_keys()``) drops any pairing that is not
        one of the 72 group fixtures (knockouts between different-group teams);
      * ``group_end`` (a date) drops anything played after the group stage, which
        is the only thing that separates a final/SF rematch of two SAME-group
        teams from their group game. When ``history`` is given it is derived
        automatically from the 72 fixtures' last date.
    """
    if group_end is None and history is not None and len(history):
        gs_all = wc2026_matches(history)
        if len(gs_all):
            group_end = pd.to_datetime(gs_all["date"], errors="coerce").max()
    # Order so the most authoritative source is LAST (drop_duplicates keep="last").
    frames: list[pd.DataFrame] = []
    if use_store:
        frames.append(stored_results(store))
    if extra_results is not None and len(extra_results):
        frames.append(schemas.validate_matches(extra_results, require_scores=True))
    if history is not None and len(history):
        wc = wc2026_matches(history)
        scored = wc[wc["home_score"].notna() & wc["away_score"].notna()]
        if len(scored):
            frames.append(schemas.validate_matches(scored, require_scores=True))
    frames = [f for f in frames if len(f)]
    if not frames:
        return _empty_results()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        combined["home_score"].notna() & combined["away_score"].notna()
    ]
    if combined.empty:
        return _empty_results()
    if group_end is not None:
        end = pd.to_datetime(group_end, errors="coerce")
        if pd.notna(end):
            d = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
            combined = combined[d <= end.normalize()]
    combined["_pk"] = [
        _pair_key(h, a) for h, a in zip(combined["home_team"], combined["away_team"])
    ]
    if valid_pairs is not None:
        combined = combined[combined["_pk"].isin(valid_pairs)]
    if combined.empty:
        return _empty_results()
    combined = combined.drop_duplicates("_pk", keep="last").drop(columns="_pk")
    return combined.sort_values("date", kind="mergesort").reset_index(drop=True)


def update_results_store(
    history: pd.DataFrame | None = None,
    extra_results: pd.DataFrame | None = None,
    *,
    store: Path = RESULTS_CSV,
    valid_pairs: set[tuple[str, str]] | None = None,
    group_end: object = None,
) -> pd.DataFrame:
    """Fold newly-observed completed games into the durable ledger and persist it.

    Idempotent and monotonic: the ledger only ever grows (or refreshes a score),
    never loses a game, because the existing ledger is one of the unioned sources.
    Two safeguards protect that invariant against I/O faults:
      * if the existing file is present but UNPARSEABLE, it is quarantined to
        ``<store>.corrupt`` (not silently overwritten with the parseable subset),
        so its bytes can be recovered by hand;
      * the new ledger is written atomically (temp file + rename) so a crash or
        full disk mid-write can never truncate the live ledger.
    Returns the full up-to-date ledger.
    """
    store = Path(store)
    merged = completed_wc_results(
        history, extra_results, use_store=True, store=store,
        valid_pairs=valid_pairs, group_end=group_end,
    )
    if _ledger_corrupt(store):
        bak = store.with_suffix(store.suffix + ".corrupt")
        try:
            store.replace(bak)
            print(f"[world_cup] results ledger {store} was unreadable; quarantined "
                  f"to {bak} (its parseable games, if any, are preserved below).")
        except OSError:
            print(f"[world_cup] results ledger {store} unreadable and could not be "
                  f"quarantined; leaving it untouched rather than overwriting.")
            return merged  # never clobber data we could not back up
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        tmp = store.with_suffix(store.suffix + ".tmp")
        merged.to_csv(tmp, index=False)
        tmp.replace(store)  # atomic on POSIX -> readers never see a partial file
    except OSError as exc:  # persistence is best-effort; never crash the sim
        print(f"[world_cup] could not write results ledger {store} ({exc}).")
    return merged


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
    *,
    use_store: bool = True,
    store: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """Canonical fixtures frame for the 72 group games, scores filled where played.

    Scores come from three sources, in this priority per fixture: the history
    feed's own score, then live ``extra_results`` (e.g. odds_api.fetch_scores),
    then the durable on-disk ledger at ``store`` (``use_store``). Matching is
    ORDER-INSENSITIVE: World Cup games are at neutral venues, so a feed may list
    the sides swapped relative to the fixture -- we match on the sorted pair and
    orient each score back to the fixture's own home/away, so a result is never
    silently dropped over a home/away disagreement.
    """
    groups = groups or load_groups()
    team_to_group = {t: g for g, teams in groups.items() for t in teams}
    gs = wc2026_matches(history)

    # Order-insensitive result map: pair-key -> {team: score}. Build it from the
    # weakest source first so stronger ones overwrite (ledger < live extra).
    extra_map: dict[tuple[str, str], dict[str, int]] = {}
    sources_in_order = []
    if use_store:
        sources_in_order.append(stored_results(store))
    if extra_results is not None and len(extra_results):
        sources_in_order.append(extra_results)
    for frame in sources_in_order:
        for r in frame.itertuples(index=False):
            hs, as_ = getattr(r, "home_score", None), getattr(r, "away_score", None)
            if pd.notna(hs) and pd.notna(as_):
                extra_map[_pair_key(r.home_team, r.away_team)] = {
                    r.home_team: int(hs),
                    r.away_team: int(as_),
                }

    fixture_keys = {_pair_key(r.home_team, r.away_team) for r in gs.itertuples(index=False)}
    rows = []
    for i, r in enumerate(gs.itertuples(index=False), start=1):
        group = team_to_group.get(r.home_team) or team_to_group.get(r.away_team) or ""
        hs, as_ = r.home_score, r.away_score
        pk = _pair_key(r.home_team, r.away_team)
        if (pd.isna(hs) or pd.isna(as_)) and pk in extra_map:
            scored = extra_map[pk]
            if r.home_team in scored and r.away_team in scored:
                hs, as_ = scored[r.home_team], scored[r.away_team]
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
    # Surface live results whose pairing matches NO group fixture at all (a genuine
    # team-name mismatch needing an alias). A result that matched a fixture which
    # already had a history score is NOT flagged -- it was redundant, not dropped.
    dropped = sorted(k for k in extra_map if k not in fixture_keys)
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


def _has_score(hs: object, as_: object) -> bool:
    return (hs is not None and as_ is not None and not pd.isna(hs) and not pd.isna(as_)
            and str(hs).strip() != "" and str(as_).strip() != "")


def complete_groups(
    fixtures: pd.DataFrame, groups: dict[str, list[str]] | None = None
) -> set[str]:
    """Group letters whose every group game already has a score in ``fixtures``."""
    groups = groups or load_groups()
    need = {str(g).strip().upper(): len(list(combinations(teams, 2)))
            for g, teams in groups.items()}
    played: dict[str, int] = defaultdict(int)
    for r in fixtures.itertuples(index=False):
        g = str(getattr(r, "group", "")).strip().upper()
        if g and _has_score(getattr(r, "home_score", None), getattr(r, "away_score", None)):
            played[g] += 1
    return {g for g, n in need.items() if played.get(g, 0) >= n}


def live_third_assignment(
    slot_map: dict[str, str],
    complete: set[str],
    ko_fixtures: list[tuple[str, str]],
) -> dict[str, str]:
    """Pin best-third R32 slots from the ACTUAL published knockout fixtures.

    ``slot_map`` is ``{"1A": team, "2A": team, "3A": team, ...}`` (e.g. from
    ``simulate_group_stage`` on the locked fixtures); ``complete`` is the set of
    groups whose standings are final (only those are trusted); ``ko_fixtures`` is
    the list of (home, away) Round-of-32 pairings from the live odds feed.

    Returns ``{r32_match_id: group_letter}`` for every best-third slot we can read
    off a real fixture -- e.g. a game between a group winner (``1I``) and a
    third-placed team (``3F``) pins that match's best-third group to F. Slots not
    yet decided are simply omitted (the solver fills them). No fabricated table:
    the assignment is observed from the actual bracket as it is published.
    """
    from ..simulation import rules

    team_slot: dict[str, str] = {}
    for code, team in slot_map.items():
        if code[1:] in complete and team:
            team_slot[normalize_team_name(team)] = code
    elig = rules.wc2026_third_slot_rules()                 # match_id -> [groups]
    home_code = {m: h for m, h, _a in rules.WC2026_R32}    # match_id -> home slot code
    winner_to_match = {home_code[m]: m for m in elig}      # '1E' -> 'M74', ...

    out: dict[str, str] = {}
    for home, away in ko_fixtures:
        a = team_slot.get(normalize_team_name(home))
        b = team_slot.get(normalize_team_name(away))
        for win_code, third_code in ((a, b), (b, a)):
            if (win_code in winner_to_match and third_code
                    and third_code.startswith("3")):
                m = winner_to_match[win_code]
                grp = third_code[1:]
                if grp in elig[m]:
                    out[m] = grp
                break
    return out


def completed_ko_results(
    history: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    groups: dict[str, list[str]] | None = None,
) -> dict[frozenset, tuple]:
    """Completed WC KNOCKOUT games as ``{frozenset({home, away}): (winner, wg, lg)}``.

    A WC match that is not one of the 72 group pairings is a knockout game; a
    decisive score identifies the winner (draws are skipped -- a shootout winner
    isn't derivable from the reported score alone). Used to LOCK games already
    played so eliminated teams drop to 0 and survivors condition on what happened.
    """
    groups = groups or load_groups()
    grp = group_pair_keys(groups)
    out: dict[frozenset, tuple] = {}
    for frame in (wc2026_matches(history), scores):
        if frame is None or len(frame) == 0:
            continue
        for r in frame.itertuples(index=False):
            h = normalize_team_name(getattr(r, "home_team", ""))
            a = normalize_team_name(getattr(r, "away_team", ""))
            hs, as_ = getattr(r, "home_score", None), getattr(r, "away_score", None)
            if not h or not a or not _has_score(hs, as_):
                continue
            if _pair_key(h, a) in grp:
                continue  # group game
            try:
                hs, as_ = int(float(hs)), int(float(as_))
            except (TypeError, ValueError):
                continue
            if hs == as_:
                continue  # shootout -- winner not derivable from the score
            winner = h if hs > as_ else a
            out[frozenset((h, a))] = (winner, max(hs, as_), min(hs, as_))
    return out


# --------------------------------------------------------------------------- #
# One-call live simulation
# --------------------------------------------------------------------------- #
def live_training_frame(
    use_odds_api_scores: bool = True, *, use_store: bool = True, store: Path = RESULTS_CSV
) -> pd.DataFrame:
    """Completed international history, topped up with every WC game played so far.

    The martj42 feed lags (and is cached up to 7 days), so a game played today is
    often not in it yet; the Odds-API ``/scores`` window and the durable results
    ledger cover that gap. We add ONLY the WC games martj42 does not already have
    (matched order-insensitively on the team pairing) so nothing is double-counted
    and the historical rows are untouched. The model therefore trains on every
    result, including ones that have rolled out of the live API window.
    """
    raw = fetch_international_results(ttl_days=_WC_HISTORY_TTL_DAYS)
    history = schemas.completed_matches(raw)
    extra = None
    if use_odds_api_scores:
        try:
            from . import odds_api

            extra = odds_api.fetch_scores()
        except Exception as exc:  # graceful: live scores are a bonus, not required
            print(f"[world_cup] odds-api scores unavailable ({exc}); using history.")
    # All WC GROUP games seen anywhere (ledger + this run's live scores). Scoped
    # to group pairings AND the group-stage date window so a live knockout result
    # is not folded in via a stale pair-key (martj42 history below still carries
    # knockouts once it absorbs them). group_end MUST come from the full fixture
    # list (incl. not-yet-played games) -- deriving it from the completed-only
    # frame would collapse it to the last SCORED date and drop today's results.
    gs_all = wc2026_matches(raw)
    group_end = (
        pd.to_datetime(gs_all["date"], errors="coerce").max() if len(gs_all) else None
    )
    played = completed_wc_results(
        extra_results=extra, use_store=use_store, store=store,
        valid_pairs=group_pair_keys(), group_end=group_end,
    )
    if played.empty:
        return history.sort_values("date", kind="mergesort").reset_index(drop=True)
    have = _wc_scored_pair_keys(history)
    new_mask = [
        _pair_key(h, a) not in have
        for h, a in zip(played["home_team"], played["away_team"])
    ]
    new = played[pd.Series(new_mask, index=played.index)]
    if new.empty:
        return history.sort_values("date", kind="mergesort").reset_index(drop=True)
    combined = pd.concat([history, new], ignore_index=True)
    return combined.sort_values("date", kind="mergesort").reset_index(drop=True)


def simulate(
    model=None,
    *,
    model_name: str = "elo",
    n_simulations: int = 10000,
    seed: int = 7,
    refresh_groups_data: bool = False,
    use_odds_api_scores: bool = True,
) -> pd.DataFrame:
    """Fetch real data, condition on games PLAYED (incl. live results), and
    Monte-Carlo the rest.

    By default this tops up the (lagging, cached) martj42 history with live scores
    from the Odds API, so a result like "France beat Senegal today" both LOCKS IN
    that group game and updates the fitted model -- the probabilities move as
    games are played. The model is fit on completed matches only (past info
    relative to the games still being simulated, so no leakage).
    """
    from ..models import DixonColes, EloGoalsModel, PoissonXG, default_ensemble
    from ..simulation.tournament import simulate_tournament

    history = fetch_international_results(ttl_days=_WC_HISTORY_TTL_DAYS)
    groups = load_groups(refresh=refresh_groups_data)

    extra = None
    if use_odds_api_scores:
        try:
            from . import odds_api

            extra = odds_api.fetch_scores()
        except Exception as exc:  # graceful: live scores are a bonus, not required
            print(f"[world_cup] odds-api scores unavailable ({exc}); using history only.")

    # Persist every group game seen so far so it stays locked in on future runs
    # even after it rolls out of the Odds-API window or the martj42 cache goes
    # stale. Scope to the 72 group pairings so a knockout rematch can't overwrite
    # a group result on the shared pair key.
    update_results_store(history, extra, valid_pairs=group_pair_keys(groups))
    build_fixtures(history, groups, extra_results=extra)
    fixtures = loaders.load_fixtures(str(FIXTURES_CSV))

    if model is None:
        factory = {
            "elo": EloGoalsModel,
            "dixon-coles": DixonColes,
            "xg": PoissonXG,
            "ensemble": default_ensemble,
        }.get(model_name, EloGoalsModel)
        model = factory()
        model.fit(live_training_frame(use_odds_api_scores).copy())

    return simulate_tournament(
        model, groups, fixtures=fixtures, n_simulations=n_simulations, seed=seed
    )
