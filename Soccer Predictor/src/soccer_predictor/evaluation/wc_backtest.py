"""
World Cup backtesting.

The honest way to decide whether the model predicts well -- and which national-
team variables actually help -- is to score it out-of-sample on PAST World Cups
(default back to 1998, the start of the 32-team era), not to assert. Backtests:

1. ``backtest_world_cups`` (match-level) -- proper scoring of every actual finals
   match. For each edition, fit a model on all completed international matches
   BEFORE the tournament started, then predict every match pre-kickoff and score
   it (log loss / RPS / Brier / accuracy / ECE). Optionally update the model
   within the tournament as results come in. No leakage (training strictly
   precedes each edition; venue neutrality is taken from the data so host
   advantage is honoured).

2. ``backtest_tournament`` -- scores the SAME tournament-simulation pipeline the
   live World Cup 2026 tab uses (reconstruct groups, fit pre-tournament,
   Monte-Carlo the bracket) on every edition: per-edition champion rank/top-N,
   and a STAGE-REACH CALIBRATION table (pooled predicted P(reach R16/QF/SF/final/
   title) vs reality, as a Brier skill score over the base rate). This answers
   "do the simulated probabilities translate into good predictions?".

3. ``backtest_champions`` (thin/legacy) -- the champion-probability slice of (2),
   kept for back-compat.

NOTE ON THE MARKET: historical closing odds are not freely available, so the
market baseline (the real bar) cannot be backtested on past World Cups here --
these backtests compare MODEL variants to each other, which is what tells you
whether form / pedigree / importance-weighting earn their keep.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from ..data import schemas
from . import metrics

WC_COMPETITION = "FIFA World Cup"


def identify_world_cups(history: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Map edition year -> that edition's finals matches (date-sorted)."""
    wc = history[
        history["competition"].str.fullmatch(WC_COMPETITION, case=False, na=False)
    ].copy()
    wc = schemas.completed_matches(wc)
    if "result" not in wc.columns:
        wc = schemas.add_result_column(wc)
    wc["year"] = pd.to_datetime(wc["date"]).dt.year
    return {
        int(y): g.sort_values("date", kind="mergesort").reset_index(drop=True)
        for y, g in wc.groupby("year")
    }


def _editions(history: pd.DataFrame, years, min_year: int) -> list[int]:
    eds = identify_world_cups(history)
    chosen = sorted(y for y in eds if y >= min_year)
    if years is not None:
        chosen = [y for y in chosen if y in set(years)]
    return chosen


def _row_probs(model, home: str, away: str, neutral: bool) -> np.ndarray:
    """1X2 from any model: scoreline models via outcome_probs, else predict_proba."""
    ctx = {"neutral_site": bool(neutral)}
    if hasattr(model, "outcome_probs"):
        return np.asarray(model.outcome_probs(home, away, ctx), dtype=float)
    frame = pd.DataFrame(
        [{"home_team": home, "away_team": away, "neutral_site": bool(neutral),
          "competition": WC_COMPETITION, "competition_type": "tournament"}]
    )
    return np.asarray(model.predict_proba(frame), dtype=float)[0]


def backtest_world_cups(
    history: pd.DataFrame,
    model_factory: Callable[[], object],
    years: list[int] | None = None,
    min_year: int = 1998,
    update_within: bool = True,
) -> dict:
    """Match-level out-of-sample backtest over past World Cups.

    For each edition: fit a FRESH model on matches strictly before the edition's
    first game, predict every finals match pre-kickoff (venue neutrality from the
    data), and -- if ``update_within`` and the model exposes ``observe`` -- feed
    each result back so later rounds see the group stage. Returns per-edition and
    pooled metrics plus the pooled probabilities/outcomes.
    """
    history = schemas.completed_matches(history)
    if "result" not in history.columns:
        history = schemas.add_result_column(history)
    history = history.sort_values("date", kind="mergesort").reset_index(drop=True)

    editions = _editions(history, years, min_year)
    per_edition: dict[int, dict] = {}
    pooled_probs: list[np.ndarray] = []
    pooled_outcomes: list[str] = []

    for year in editions:
        matches = identify_world_cups(history)[year]
        start = matches["date"].min()
        train = history[history["date"] < start]
        if len(train) < 200 or len(matches) == 0:
            continue
        model = model_factory()
        model.fit(train.copy())

        probs, outcomes = [], []
        can_observe = update_within and hasattr(model, "observe")
        for r in matches.itertuples(index=False):
            neutral = bool(getattr(r, "neutral_site", True))
            p = _row_probs(model, r.home_team, r.away_team, neutral)
            probs.append(p)
            outcomes.append(r.result)
            if can_observe:
                model.observe(
                    r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                    neutral=neutral, competition=WC_COMPETITION,
                )
        probs_arr = np.vstack(probs)
        out_arr = np.array(outcomes, dtype=object)
        per_edition[year] = metrics.all_metrics(probs_arr, out_arr)
        pooled_probs.append(probs_arr)
        pooled_outcomes.extend(outcomes)

    if not pooled_probs:
        return {"editions": {}, "pooled": {}, "probs": np.zeros((0, 3)), "outcomes": []}

    probs = np.vstack(pooled_probs)
    outcomes = np.array(pooled_outcomes, dtype=object)
    return {
        "editions": per_edition,
        "pooled": metrics.all_metrics(probs, outcomes),
        "probs": probs,
        "outcomes": outcomes,
    }


def compare_models_on_world_cups(
    history: pd.DataFrame,
    factories: dict[str, Callable[[], object]],
    years: list[int] | None = None,
    min_year: int = 1998,
    update_within: bool = True,
) -> pd.DataFrame:
    """Run the match-level backtest for several model variants -> leaderboard.

    Returns a DataFrame indexed by model name with pooled log_loss / rps / brier /
    accuracy / ece / n, sorted by log loss (lower is better). This is the table
    that answers "does form / pedigree / squad / importance-weighting help?".
    """
    rows = {}
    for name, factory in factories.items():
        res = backtest_world_cups(
            history, factory, years=years, min_year=min_year, update_within=update_within
        )
        rows[name] = res["pooled"] or {
            "log_loss": float("nan"), "rps": float("nan"), "brier": float("nan"),
            "accuracy": float("nan"), "ece": float("nan"), "n": 0,
        }
    df = pd.DataFrame(rows).T
    cols = [c for c in ["log_loss", "rps", "brier", "accuracy", "ece", "n"] if c in df.columns]
    return df[cols].sort_values("log_loss")


# --------------------------------------------------------------------------- #
# Secondary: tournament-simulation champion backtest (noisy)
# --------------------------------------------------------------------------- #
def _reconstruct_year_groups(matches: pd.DataFrame, n_groups: int = 8) -> dict[str, list[str]] | None:
    """Reconstruct an edition's groups from its first ``n_groups*6`` fixtures.

    Connected components of the group-stage "played-against" graph are the groups.
    Returns ``{A: [...], B: [...]}`` labelled by descending component strength is
    not attempted (labels are arbitrary); returns None if it is not a clean
    ``n_groups`` x 4 partition.
    """
    gs = matches.sort_values("date", kind="mergesort").head(n_groups * 6)
    opp: dict[str, set[str]] = defaultdict(set)
    for r in gs.itertuples(index=False):
        opp[r.home_team].add(r.away_team)
        opp[r.away_team].add(r.home_team)
    seen: set[str] = set()
    comps: list[list[str]] = []
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
        comps.append(sorted(comp))
    if len(comps) != n_groups or any(len(c) != 4 for c in comps):
        return None
    letters = [chr(ord("A") + i) for i in range(n_groups)]
    return {letters[i]: comps[i] for i in range(n_groups)}


# Verified men's World Cup champions (historical fact). Used as the evaluation
# target because penalty-decided finals (2006, 2022) are draws in the regulation
# score, so the winner cannot be read from the score alone.
WC_CHAMPIONS: dict[int, str] = {
    1998: "France", 2002: "Brazil", 2006: "Italy", 2010: "Spain",
    2014: "Germany", 2018: "France", 2022: "Argentina",
}


def actual_stage_reach(
    matches: pd.DataFrame, fmt, champion: str | None = None
) -> dict[str, set[str]] | None:
    """Which teams actually PLAYED IN each knockout stage of an edition.

    Derived from participation (who took the pitch), NOT from who won, so
    penalty-shootout results -- which the score alone cannot resolve -- never
    matter. Knockout rounds are temporally separated, so date order assigns them
    cleanly; the third-place playoff is skipped by taking the FINAL to be the
    chronologically last match. Returns ``{stage: {teams}}`` for every stage in
    ``fmt.stage_order`` plus ``"champion"`` (from the verified champions map), or
    None if the bracket does not have the expected shape.
    """
    m = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
    n_group_games = fmt.n_groups * 6  # round-robin of four => six games per group
    ko = m.iloc[n_group_games:].reset_index(drop=True)
    n_ko_teams = fmt.n_groups * fmt.advance_per_group + fmt.n_best_third
    sizes, t = [], n_ko_teams
    while t >= 2:
        sizes.append(t // 2)
        t //= 2
    stages = list(fmt.stage_order)
    if len(sizes) != len(stages) or len(ko) < sum(sizes[:-1]) + 1:
        return None

    def teams_of(rows: pd.DataFrame) -> set[str]:
        return set(rows["home_team"]) | set(rows["away_team"])

    reach: dict[str, set[str]] = {}
    idx = 0
    for i, stage in enumerate(stages):
        if i < len(stages) - 1:
            rows = ko.iloc[idx:idx + sizes[i]]
            idx += sizes[i]
        else:
            rows = ko.iloc[-1:]  # the final is the last match (3rd-place playoff skipped)
        reach[stage] = teams_of(rows)
    if champion is not None:
        from ..data.aliases import normalize_team_name

        reach["champion"] = {normalize_team_name(champion)}
    return reach


def simulate_past_world_cup(
    history: pd.DataFrame,
    model_factory: Callable[[], object],
    year: int,
    n_simulations: int = 4000,
    seed: int = 7,
    champions: dict[int, str] | None = None,
) -> dict | None:
    """Pre-tournament simulation of ONE past edition -- the live pipeline on history.

    Reconstructs the edition's groups, fits the model on data strictly before the
    tournament, and Monte-Carlos the 32-team legacy bracket -- exactly what the
    live World Cup 2026 tab does, but for a finished edition so the prediction can
    be put next to what actually happened. Returns
    ``{year, groups, sims, champion, champion_rank, reached}`` (``sims`` is the
    full per-team probability table) or None if the edition can't be reconstructed.
    """
    from ..simulation import rules
    from ..simulation.tournament import simulate_tournament
    from ..data.aliases import normalize_team_name

    champions = champions or WC_CHAMPIONS
    history = schemas.completed_matches(history)
    history = history.sort_values("date", kind="mergesort").reset_index(drop=True)
    fmt = rules.world_cup_legacy_format()
    eds = identify_world_cups(history)
    if year not in eds:
        return None
    matches = eds[year]
    groups = _reconstruct_year_groups(matches, n_groups=fmt.n_groups)
    if groups is None:
        return None
    start = matches["date"].min()
    train = history[history["date"] < start]
    if len(train) < 200:
        return None
    model = model_factory()
    model.fit(train.copy())
    sims = simulate_tournament(
        model, groups, n_simulations=n_simulations, seed=seed, fmt=fmt
    ).reset_index(drop=True)

    champ = champions.get(year)
    champ = normalize_team_name(champ) if champ else None
    reached = actual_stage_reach(matches, fmt, champion=champ) if champ else None
    rank = -1
    if champ:
        ranked = sims.sort_values("champion_probability", ascending=False).reset_index(drop=True)
        loc = ranked.index[ranked["team"] == champ]
        rank = int(loc[0]) + 1 if len(loc) else -1
    return {
        "year": year, "groups": groups, "sims": sims,
        "champion": champ, "champion_rank": rank, "reached": reached,
    }


def backtest_tournament(
    history: pd.DataFrame,
    model_factory: Callable[[], object],
    years: list[int] | None = None,
    min_year: int = 1998,
    n_simulations: int = 4000,
    seed: int = 7,
    champions: dict[int, str] | None = None,
) -> dict:
    """Score the FULL tournament-simulation methodology on past World Cups.

    This is the same pipeline the live World Cup 2026 tab uses -- reconstruct the
    groups, fit the model on pre-tournament data, Monte-Carlo the bracket -- run
    on every completed edition (default back to 1998) and scored properly:

      * per-edition champion probability / pre-tournament rank / top-N hit;
      * STAGE-REACH CALIBRATION -- pool every team's predicted P(reach R16 / QF /
        SF / final / title) against whether it actually did, across all editions,
        and report a Brier skill score vs the base rate (this is the real test of
        "do the simulated probabilities translate into good predictions?");
      * champion log-loss vs a uniform prior.

    Returns ``{editions, calibration, champion_skill}``. Historical odds are not
    free, so there is no market anchor here -- this scores the pure model.
    """
    from ..simulation import rules
    from ..simulation.tournament import simulate_tournament
    from ..data.aliases import normalize_team_name

    champions = champions or WC_CHAMPIONS
    history = schemas.completed_matches(history)
    history = history.sort_values("date", kind="mergesort").reset_index(drop=True)
    fmt = rules.world_cup_legacy_format()
    editions = _editions(history, years, min_year)
    eds_matches = identify_world_cups(history)
    stages = list(fmt.stage_order) + ["champion"]

    per_edition: list[dict] = []
    # Pooled (prediction, outcome) per stage across all editions, for calibration.
    pool: dict[str, list[tuple[float, float]]] = {s: [] for s in stages}
    champ_neglogp: list[float] = []
    n_teams_seen: list[int] = []

    for year in editions:
        champ = champions.get(year)
        if champ is None:
            continue
        champ = normalize_team_name(champ)
        matches = eds_matches[year]
        groups = _reconstruct_year_groups(matches, n_groups=fmt.n_groups)
        if groups is None:
            per_edition.append({"year": year, "note": "group reconstruction failed"})
            continue
        reach = actual_stage_reach(matches, fmt, champion=champ)
        if reach is None:
            per_edition.append({"year": year, "note": "bracket reconstruction failed"})
            continue
        start = matches["date"].min()
        train = history[history["date"] < start]
        if len(train) < 200:
            continue
        model = model_factory()
        model.fit(train.copy())
        sims = simulate_tournament(
            model, groups, n_simulations=n_simulations, seed=seed, fmt=fmt
        ).reset_index(drop=True)

        ranked = sims.sort_values("champion_probability", ascending=False).reset_index(drop=True)
        crow = ranked[ranked["team"] == champ]
        c_prob = float(crow["champion_probability"].iloc[0]) if len(crow) else 0.0
        c_rank = int(crow.index[0]) + 1 if len(crow) else -1
        n_teams = len(sims)
        n_teams_seen.append(n_teams)
        champ_neglogp.append(-np.log(max(c_prob, 1e-9)))

        # Accumulate calibration pairs for every team and stage.
        for s in stages:
            col = "champion_probability" if s == "champion" else f"{s}_probability"
            if col not in sims.columns:
                continue
            reached = reach.get(s, set())
            for _, srow in sims.iterrows():
                pred = float(srow[col])
                actual = 1.0 if srow["team"] in reached else 0.0
                pool[s].append((pred, actual))

        per_edition.append({
            "year": year,
            "actual_champion": champ,
            "model_champion_prob": round(c_prob, 4),
            "champion_pre_rank": c_rank,
            "champ_in_top4": c_rank != -1 and c_rank <= 4,
            "champ_in_top8": c_rank != -1 and c_rank <= 8,
            "favourite": ranked.iloc[0]["team"] if len(ranked) else "",
        })

    # Stage-reach calibration table (Brier skill vs the base rate).
    cal_rows = []
    for s in stages:
        pairs = pool[s]
        if not pairs:
            continue
        preds = np.array([p for p, _ in pairs])
        outs = np.array([o for _, o in pairs])
        base = float(outs.mean())
        brier_model = float(np.mean((preds - outs) ** 2))
        brier_base = base * (1.0 - base)  # Brier of the best constant predictor
        skill = 1.0 - brier_model / brier_base if brier_base > 0 else float("nan")
        cal_rows.append({
            "stage": s, "n": len(pairs), "base_rate": round(base, 3),
            "mean_pred": round(float(preds.mean()), 3),
            "brier_model": round(brier_model, 4), "brier_base": round(brier_base, 4),
            "brier_skill": round(skill, 3),
        })
    calibration = pd.DataFrame(cal_rows)

    n_teams = int(np.median(n_teams_seen)) if n_teams_seen else 0
    champion_skill = {
        "editions": len(champ_neglogp),
        "mean_neglogp_model": round(float(np.mean(champ_neglogp)), 3) if champ_neglogp else float("nan"),
        "mean_neglogp_uniform": round(float(np.log(n_teams)), 3) if n_teams else float("nan"),
    }
    return {
        "editions": pd.DataFrame(per_edition),
        "calibration": calibration,
        "champion_skill": champion_skill,
    }


def backtest_champions(
    history: pd.DataFrame,
    model_factory: Callable[[], object],
    years: list[int] | None = None,
    min_year: int = 1998,
    n_simulations: int = 2000,
    seed: int = 7,
    champions: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Per-edition champion probability the model gave the ACTUAL winner.

    Reconstructs each edition's groups, fits the model on pre-tournament data, and
    Monte-Carlos the 32-team legacy format. Reports the eventual winner, the
    probability assigned to them, and their pre-tournament rank. Noisy (few
    editions) -- read it as a sanity check, not a metric.
    """
    from ..simulation import rules
    from ..simulation.tournament import simulate_tournament

    champions = champions or WC_CHAMPIONS
    history = schemas.completed_matches(history)
    history = history.sort_values("date", kind="mergesort").reset_index(drop=True)
    fmt = rules.world_cup_legacy_format()
    editions = _editions(history, years, min_year)
    eds_matches = identify_world_cups(history)

    rows = []
    for year in editions:
        # Only backtest completed 32-team editions with a known champion (skips the
        # ongoing 2026 and any 48-team / unknown edition).
        champ = champions.get(year)
        if champ is None:
            continue
        from ..data.aliases import normalize_team_name

        champ = normalize_team_name(champ)
        matches = eds_matches[year]
        groups = _reconstruct_year_groups(matches, n_groups=fmt.n_groups)
        if groups is None:
            rows.append({"year": year, "note": "group reconstruction failed"})
            continue
        start = matches["date"].min()
        train = history[history["date"] < start]
        if len(train) < 200:
            continue
        model = model_factory()
        model.fit(train.copy())
        sims = simulate_tournament(
            model, groups, n_simulations=n_simulations, seed=seed, fmt=fmt
        )
        sims = sims.reset_index(drop=True)
        champ_row = sims[sims["team"] == champ]
        prob = float(champ_row["champion_probability"].iloc[0]) if len(champ_row) else float("nan")
        rank = (
            int(sims.sort_values("champion_probability", ascending=False)
                .reset_index(drop=True).index[sims.sort_values(
                    "champion_probability", ascending=False).reset_index(drop=True)["team"] == champ][0]) + 1
            if len(champ_row) else -1
        )
        rows.append({
            "year": year, "actual_champion": champ,
            "model_champion_prob": round(prob, 4), "champion_pre_rank": rank,
            "favourite": sims.sort_values("champion_probability", ascending=False)
                .iloc[0]["team"],
        })
    return pd.DataFrame(rows)
