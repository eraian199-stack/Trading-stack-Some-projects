from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


GROUPS = list("ABCDEFGHIJKL")
BASE_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{}"


def _clean_team(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\[[^\]]+\]", "", text).strip()
    return text


def fetch_group(group: str) -> pd.DataFrame:
    url = BASE_URL.format(group)
    tables = pd.read_html(url)
    candidates = []
    for table in tables:
        columns = [str(col).lower() for col in table.columns]
        if any("team" == col or "team" in col for col in columns):
            candidates.append(table)
    for table in candidates:
        team_col = next((col for col in table.columns if "team" in str(col).lower()), None)
        if team_col is None:
            continue
        teams = [_clean_team(value) for value in table[team_col].tolist()]
        teams = [team for team in teams if team and team.lower() != "team"]
        teams = [team for team in teams if "winner" not in team.lower()]
        if len(teams) >= 4:
            return pd.DataFrame({"group": group, "team": teams[:4]})
    raise RuntimeError(f"Could not find four teams for Group {group} from {url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/world_cup_2026_groups.csv",
        help="Output CSV path relative to this project or absolute.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = project_root / out
    rows = [fetch_group(group) for group in GROUPS]
    frame = pd.concat(rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
