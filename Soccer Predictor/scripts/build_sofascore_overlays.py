"""
Build SofaScore club-season ability overlays from the raw league-season pulls.

Reads the per-(league, edition, page) CSVs written under data/cache/sofascore_raw/
(SofaScore top-rated players by club-league season, name + rating only), joins each
player to a NATIONALITY via the local FIFA dataset (name -> nationality, the only
free way to map club ratings to national squads -- SofaScore's API omits country),
then writes per-edition player-level files data/sofascore_ratings_<wc>.csv.

Club-season ratings are the season that PRECEDED each World Cup (leakage-free):
2017/18 -> WC2018, 2021/22 -> WC2022, 2025/26 -> WC2026 (2026 nationality falls back
to FIFA 23, so players who debuted after 2022 are dropped -- a known coverage gap).

LEAGUE ADJUSTMENT: raw SofaScore ratings aren't comparable across leagues (a 7.5 in
the Saudi league != a 7.5 in the Premier League). Each player's ability is therefore
LEAGUE_STRENGTH[league] + b * rating, where LEAGUE_STRENGTH is the true mean FIFA
overall of that league (unbiased by which players we sampled) and b is the empirical
within-league overall-per-rating slope. This raised agreement with the FIFA/TM
ability consensus (corr vs FIFA 0.27 -> 0.40, vs TM 0.41 -> 0.48).

VERDICT (backtested OOS on WC 2018 & 2022, match-level pooled log loss): the league
adjustment shrank the dilution (Elo+FIFA+TM @0.5 = 0.9902; +SofaScore @0.5 was 0.9926
raw, now 0.9921 league-adjusted; @1.0 ~ plain Elo) but FIFA+TM is STILL best -- even
league-adjusted, SofaScore *performance* is noisier than FIFA scouted *ability* on
the two clean editions. So these files are written to data/sofascore_ratings_<wc>.csv,
which the default blend does NOT auto-consume; the app exposes them behind an explicit
'blend SofaScore' toggle for experimentation.
"""
from __future__ import annotations

import glob
import io
import re
import unicodedata

import pandas as pd

from soccer_predictor.data import squad_value as sv
from soccer_predictor.data.aliases import normalize_team_name

RAW = "data/cache/sofascore_raw"
# WC edition -> (FIFA version for nationality, snapshot date)
WC_FIFA = {2018: (18, "2018-06-01"), 2022: (23, "2022-06-01"), 2026: (23, "2025-09-01")}


def norm(s: object) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _load_fifa() -> pd.DataFrame:
    raw = sv._cached_bytes(sv._FIFA_URL, "fifa_male_players_legacy.csv", ttl_days=90.0)
    return pd.read_csv(
        io.BytesIO(raw),
        usecols=["fifa_version", "fifa_update_date", "short_name", "long_name",
                 "overall", "nationality_name", "player_positions"],
        low_memory=False,
    )


OVERALL_GATE = 70.0   # drop obscure low-minute backups (their FIFA overall is low)
AMBIG_MARGIN = 5.0    # if top-2 same-name candidates differ in nationality & are this
                      # close in overall, the name is ambiguous -> drop (no wrong guess)


def _name_keys(short_name: str, long_name: str) -> list[str]:
    """Match keys SofaScore (full names) can hit. FIFA short_names are often
    abbreviated ('L. Díaz'), so also key the long name as first-token + EACH later
    token ('luis diaz' from 'luis fernando diaz marulanda')."""
    keys = []
    sn = norm(short_name)
    if sn and "." not in str(short_name):  # skip abbreviated short names
        keys.append(sn)
    ln = norm(long_name)
    if ln:
        keys.append(ln)
        toks = ln.split()
        if len(toks) >= 2:
            keys += [f"{toks[0]} {t}" for t in toks[1:]]
    return list(dict.fromkeys(keys))


def nationality_map(fifa: pd.DataFrame, version: int, as_of: str) -> dict[str, list]:
    """key -> list of (nationality, overall) candidates for one FIFA snapshot, so the
    resolver can detect cross-nationality ambiguity instead of silently guessing."""
    snap = sv._fifa_snapshot(fifa, version, pd.Timestamp(as_of)).dropna(subset=["nationality_name"])
    m: dict[str, list] = {}
    for r in snap.itertuples(index=False):
        nat = str(r.nationality_name)
        ov = float(r.overall) if pd.notna(r.overall) else 0.0
        is_gk = "GK" in str(r.player_positions)  # keeper season ratings are noisy
        for k in _name_keys(r.short_name, r.long_name):
            m.setdefault(k, []).append((nat, ov, is_gk))
    return m


def resolve_player(name: str, nmap: dict[str, list]) -> tuple[str, float] | None:
    """(nationality, FIFA overall) for a SofaScore name, or None if unknown /
    ambiguous / sub-gate / goalkeeper."""
    nk = norm(name)
    cands = nmap.get(nk)
    if not cands:
        toks = nk.split()
        # try first-token + each other token as a fallback key
        for i in range(1, len(toks)):
            cands = nmap.get(f"{toks[0]} {toks[i]}")
            if cands:
                break
    if not cands:
        return None
    cands = sorted(cands, key=lambda c: -c[1])
    top_nat, top_ov, top_gk = cands[0]
    if top_ov < OVERALL_GATE:           # obscure / low-minute player -> drop
        return None
    if top_gk:                          # exclude goalkeepers (noisy small-sample ratings)
        return None
    for nat, ov, _gk in cands[1:]:      # cross-nationality collision near the top -> drop
        if nat != top_nat and (top_ov - ov) < AMBIG_MARGIN:
            return None
    return top_nat, top_ov


# SofaScore unique-tournament id -> league label (for the league-strength report).
TID_LEAGUE = {
    17: "Premier League", 8: "LaLiga", 23: "Serie A", 35: "Bundesliga",
    34: "Ligue 1", 37: "Eredivisie", 238: "Primeira Liga", 955: "Saudi Pro League",
    242: "MLS",
}
# League STRENGTH anchor = the TRUE mean FIFA overall of every player in that league
# (FIFA 23 full rosters; UNBIASED by which players we sampled -- using the sampled
# stars' mean would wrongly rank star-heavy weak leagues like Saudi as strong).
# Saudi is absent from FIFA 23, so it's set manually near its true level.
LEAGUE_STRENGTH = {
    8: 72.9, 17: 72.7, 23: 72.5, 34: 70.9, 35: 69.2,
    238: 68.6, 37: 66.9, 242: 64.8, 955: 64.0,
}


def main() -> None:
    fifa = _load_fifa()
    # ---- PASS 1: collect every joined (player, league, raw rating, FIFA overall) ----
    # raw_rows[wc] = list of (tid, player, team, raw_rating, overall)
    raw_rows: dict[int, list] = {}
    for wc, (ver, asof) in WC_FIFA.items():
        files = sorted(glob.glob(f"{RAW}/*_{wc}_p*.csv"))
        if not files:
            continue
        nmap = nationality_map(fifa, ver, asof)
        rows, raw_n = [], 0
        for fp in files:
            tid = int(fp.split("/")[-1].split("_")[0])
            try:
                d = pd.read_csv(fp)
            except Exception:
                continue
            if "name" not in d.columns or "rating" not in d.columns:
                continue
            for r in d.itertuples(index=False):
                try:
                    rating = float(r.rating)
                except Exception:
                    continue
                if not (4.0 <= rating <= 10.0):
                    continue
                raw_n += 1
                res = resolve_player(str(r.name), nmap)
                if res is None:
                    continue
                nat, ov = res
                rows.append((tid, str(r.name), normalize_team_name(nat), rating, ov))
        raw_rows[wc] = rows
        print(f"WC{wc}: {len(files)} files, {raw_n} rated rows, {len(rows)} joined "
              f"({len(rows) / max(raw_n, 1):.0%})")

    # ---- LEAGUE ADJUSTMENT ----
    # ability = LEAGUE_STRENGTH[league] + b * rating, where b is the global
    # WITHIN-league slope of FIFA overall on SofaScore rating (overall pts per
    # rating pt). Because SofaScore normalises every league's average player to
    # ~the same rating, the cross-league level lives entirely in LEAGUE_STRENGTH:
    # the same 7.5 anchors lower in a weak league than a strong one, while within a
    # league higher ratings still separate. (Any global constant cancels under the
    # z-score the blend applies, so only the per-league anchor + b*rating matter.)
    grp_rt: dict[tuple, list] = {}
    grp_ov: dict[tuple, list] = {}
    for wc, rows in raw_rows.items():
        for tid, _name, _team, rt, ov in rows:
            grp_rt.setdefault((wc, tid), []).append(rt)
            grp_ov.setdefault((wc, tid), []).append(ov)
    mean_rt = {k: sum(v) / len(v) for k, v in grp_rt.items()}
    mean_ov = {k: sum(v) / len(v) for k, v in grp_ov.items()}
    num = den = 0.0
    for wc, rows in raw_rows.items():
        for tid, _name, _team, rt, ov in rows:
            dr = rt - mean_rt[(wc, tid)]
            num += dr * (ov - mean_ov[(wc, tid)])
            den += dr * dr
    b = num / den if den else 0.0
    ref = sum(LEAGUE_STRENGTH.values()) / len(LEAGUE_STRENGTH)
    print(f"\nLeague-adjustment: within-league slope b = {b:.1f} overall-pts per rating-pt")
    print("League strength anchor (true FIFA mean overall, vs reference "
          f"{ref:.1f}):")
    for tid, strg in sorted(LEAGUE_STRENGTH.items(), key=lambda kv: -kv[1]):
        print(f"   {TID_LEAGUE.get(tid, tid):18} {strg:5.1f}  (offset {strg - ref:+.1f})")

    # ---- PASS 2: write league-adjusted per-edition overlays ----
    for wc, rows in raw_rows.items():
        out_rows = [{"player": name, "team": team,
                     "rating": round(LEAGUE_STRENGTH.get(tid, ref) + b * rt, 2)}
                    for tid, name, team, rt, ov in rows]
        df = (pd.DataFrame(out_rows)
              .sort_values("rating", ascending=False)
              .drop_duplicates(subset=["player", "team"]))
        out = f"data/sofascore_ratings_{wc}.csv"
        df.to_csv(out, index=False)
        print(f"WC{wc}: wrote {len(df)} players, {df['team'].nunique()} nations -> {out}")
        if wc == 2026:  # spot-check: top players for a few nations (now overall-scale)
            for nat in ["Germany", "Colombia", "Argentina", "Brazil", "Spain", "Portugal"]:
                t = normalize_team_name(nat)
                top = df[df["team"] == t].head(4)
                shown = ", ".join(f"{r.player} {r.rating}" for r in top.itertuples(index=False))
                print(f"    {nat:10}: {shown}")


if __name__ == "__main__":
    main()
