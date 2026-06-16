"""
Team-name alias maps.

Different sources spell the same team differently ("Man United" vs "Manchester
United", "USA" vs "United States", "Korea Republic" vs "South Korea"). Every
loader funnels names through `normalize_team_name`, so the rest of the system
only ever sees one canonical spelling. When a join (e.g. xG merge) reports
unmatched rows, the fix is almost always a new entry here.

Two alias families:
- INTERNATIONAL_ALIASES: national teams (used for World Cup / continental work).
- CLUB_ALIASES: clubs (mostly Europe's top leagues, where odds + xG live).

`normalize_team_name` is intentionally cheap and pure so it can be mapped over
large columns and called inside the simulator's hot loop.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- #
# International (national teams) -> canonical
# --------------------------------------------------------------------------- #
INTERNATIONAL_ALIASES: dict[str, str] = {
    "usa": "United States",
    "usmnt": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "czech republic": "Czechia",
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "ir iran": "Iran",
    "iran (islamic republic of)": "Iran",
    "dr congo": "Congo DR",
    "democratic republic of the congo": "Congo DR",
    "congo dr": "Congo DR",
    "republic of ireland": "Ireland",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "cape verde": "Cabo Verde",
    "the gambia": "Gambia",
    "bosnia and herzegovina": "Bosnia-Herzegovina",
    "north macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
    "uae": "United Arab Emirates",
    "ksa": "Saudi Arabia",
    "china pr": "China",
    "chinese taipei": "Chinese Taipei",
    "curacao": "Curaçao",
    "st kitts and nevis": "Saint Kitts and Nevis",
    "st lucia": "Saint Lucia",
    "st vincent and the grenadines": "Saint Vincent and the Grenadines",
}

# --------------------------------------------------------------------------- #
# Clubs -> canonical (lowercased keys)
# --------------------------------------------------------------------------- #
CLUB_ALIASES: dict[str, str] = {
    "manchester united": "Man United",
    "manchester utd": "Man United",
    "man utd": "Man United",
    "manchester city": "Man City",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "nottingham forest": "Nott'm Forest",
    "nott'ham forest": "Nott'm Forest",
    "sheffield united": "Sheffield United",
    "leeds united": "Leeds",
    "afc bournemouth": "Bournemouth",
    "paris saint germain": "Paris SG",
    "paris saint-germain": "Paris SG",
    "psg": "Paris SG",
    "bayern munich": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "borussia m.gladbach": "M'gladbach",
    "borussia monchengladbach": "M'gladbach",
    "atletico madrid": "Atletico Madrid",
    "atlético madrid": "Atletico Madrid",
    "athletic club": "Athletic Bilbao",
    "athletic bilbao": "Athletic Bilbao",
    "inter milan": "Inter",
    "internazionale": "Inter",
    "ac milan": "Milan",
}

# Suffixes stripped before alias lookup (club naming noise).
_CLUB_SUFFIXES: tuple[str, ...] = (" fc", " afc", " cf", " sc", " cd", " ac")


def normalize_team_name(value: object) -> str:
    """Return the canonical spelling for a team name (club or national).

    Pure and forgiving: NaN -> "", strips whitespace, applies international then
    club alias maps (case-insensitive). Unknown names pass through with their
    original casing preserved.
    """
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return ""
    # Canonical lookup key: lower-case, '&' -> 'and', collapsed whitespace. This
    # makes feeds that spell "Bosnia & Herzegovina" resolve to the same alias as
    # "Bosnia and Herzegovina" (silent name mismatches otherwise degrade a known
    # team to a generic base rating with no warning).
    key = " ".join(text.lower().replace("&", " and ").split())
    if key in INTERNATIONAL_ALIASES:
        return INTERNATIONAL_ALIASES[key]
    if key in CLUB_ALIASES:
        return CLUB_ALIASES[key]
    stripped = key.replace("_", " ")
    for suffix in _CLUB_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
            break
    if stripped in INTERNATIONAL_ALIASES:
        return INTERNATIONAL_ALIASES[stripped]
    if stripped in CLUB_ALIASES:
        return CLUB_ALIASES[stripped]
    return text
