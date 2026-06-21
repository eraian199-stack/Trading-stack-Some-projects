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

VERDICT (backtested OOS on WC 2018 & 2022, match-level pooled log loss): adding this
SofaScore club-season source to the ability blend does NOT help -- it DILUTES it
(Elo+FIFA+TM @0.5 = 0.9902; +SofaScore @0.5 = 0.9926; @1.0 worse than plain Elo).
FIFA `overall` is a cleaner ability signal than SofaScore *performance* ratings, so
these files are written to data/sofascore_ratings_<wc>.csv, which the blend does NOT
auto-consume. To experiment with them anyway, copy one to
data/ability_overlay_<wc>.csv (the path the blend globs).
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


def resolve_nat(name: str, nmap: dict[str, list]) -> str | None:
    """Best nationality for a SofaScore name, or None if unknown/ambiguous/sub-gate."""
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
    return top_nat


def main() -> None:
    fifa = _load_fifa()
    for wc, (ver, asof) in WC_FIFA.items():
        files = sorted(glob.glob(f"{RAW}/*_{wc}_p*.csv"))
        if not files:
            print(f"WC{wc}: no raw files, skipping")
            continue
        nmap = nationality_map(fifa, ver, asof)
        rows, raw_n, matched = [], 0, 0
        for fp in files:
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
                if not (4.0 <= rating <= 10.0):  # drop blanks / parse junk
                    continue
                raw_n += 1
                nat = resolve_nat(str(r.name), nmap)
                if nat is None:
                    continue
                matched += 1
                rows.append({"player": str(r.name), "team": normalize_team_name(nat),
                             "rating": rating})
        if not rows:
            print(f"WC{wc}: 0 players joined from {len(files)} files")
            continue
        df = (pd.DataFrame(rows)
              .sort_values("rating", ascending=False)
              .drop_duplicates(subset=["player", "team"]))  # dedup overlapping pages
        out = f"data/sofascore_ratings_{wc}.csv"
        df.to_csv(out, index=False)
        print(f"WC{wc}: {len(files)} files, {raw_n} rated rows, {matched} joined "
              f"({matched / max(raw_n, 1):.0%}) -> {len(df)} players, "
              f"{df['team'].nunique()} nations -> {out}")
        if wc == 2026:  # quality spot-check: top players for a few nations
            for nat in ["Germany", "Colombia", "Argentina", "Brazil", "Spain", "Portugal"]:
                t = normalize_team_name(nat)
                top = df[df["team"] == t].head(4)
                shown = ", ".join(f"{r.player} {r.rating}" for r in top.itertuples(index=False))
                print(f"    {nat:10}: {shown}")


if __name__ == "__main__":
    main()
