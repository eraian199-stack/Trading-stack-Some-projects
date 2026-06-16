"""
Rest and travel features.

Two pre-kickoff facts that matter, especially in congested fixture lists and
international tournaments:

- Rest days: days since a team last played. A team coming off three days' rest is
  measurably worse than one off a week. The feature store tracks each team's last
  match date and calls :func:`rest_days`.
- Travel distance: great-circle (haversine) kilometres between the two teams'
  reference cities. Only computable when both cities are in ``CITY_COORDS``;
  unknown cities degrade to NaN rather than guessing.

The coordinate table is a small, explicitly-incomplete starter set (host-nation
and major-confederation cities). Missing entries are a feature gap, not a bug:
downstream code treats NaN travel as "unknown".
"""

from __future__ import annotations

import math

import pandas as pd

# A small starter set of city -> (latitude, longitude). Deliberately partial:
# unknown cities yield NaN travel rather than a fabricated distance. Extend as
# needed (the same pattern as the alias maps in data/aliases.py).
CITY_COORDS: dict[str, tuple[float, float]] = {
    # WC2026 hosts (USA / Canada / Mexico) and major hubs.
    "new york": (40.7128, -74.0060),
    "new jersey": (40.8136, -74.0744),
    "los angeles": (34.0522, -118.2437),
    "dallas": (32.7767, -96.7970),
    "houston": (29.7604, -95.3698),
    "atlanta": (33.7490, -84.3880),
    "miami": (25.7617, -80.1918),
    "seattle": (47.6062, -122.3321),
    "san francisco": (37.7749, -122.4194),
    "kansas city": (39.0997, -94.5786),
    "boston": (42.3601, -71.0589),
    "philadelphia": (39.9526, -75.1652),
    "toronto": (43.6532, -79.3832),
    "vancouver": (49.2827, -123.1207),
    "mexico city": (19.4326, -99.1332),
    "guadalajara": (20.6597, -103.3496),
    "monterrey": (25.6866, -100.3161),
    # European hubs.
    "london": (51.5074, -0.1278),
    "manchester": (53.4808, -2.2426),
    "liverpool": (53.4084, -2.9916),
    "madrid": (40.4168, -3.7038),
    "barcelona": (41.3851, 2.1734),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "munich": (48.1351, 11.5820),
    "rome": (41.9028, 12.4964),
    "milan": (45.4642, 9.1900),
    "lisbon": (38.7223, -9.1393),
    "amsterdam": (52.3676, 4.9041),
    "brussels": (50.8503, 4.3517),
    # Other confederations.
    "buenos aires": (-34.6037, -58.3816),
    "rio de janeiro": (-22.9068, -43.1729),
    "sao paulo": (-23.5505, -46.6333),
    "tokyo": (35.6762, 139.6503),
    "seoul": (37.5665, 126.9780),
    "doha": (25.2854, 51.5310),
    "riyadh": (24.7136, 46.6753),
    "cairo": (30.0444, 31.2357),
    "lagos": (6.5244, 3.3792),
    "johannesburg": (-26.2041, 28.0473),
    "sydney": (-33.8688, 151.2093),
}


def rest_days(last_played: pd.Timestamp | None, date: pd.Timestamp) -> float:
    """Days between a team's previous match and ``date``.

    ``last_played`` is None (no prior match) -> NaN. NaT inputs also yield NaN.
    """
    if last_played is None:
        return float("nan")
    try:
        if pd.isna(last_played) or pd.isna(date):
            return float("nan")
    except (TypeError, ValueError):
        pass
    return float((pd.Timestamp(date) - pd.Timestamp(last_played)).days)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    radius = 6371.0088  # mean Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return float(2.0 * radius * math.asin(min(1.0, math.sqrt(a))))


def travel_km_for_city(home_city: str, away_city: str) -> float:
    """Distance the AWAY team travels, from its city to the host city.

    Either city unknown -> NaN. City names are matched case-insensitively against
    :data:`CITY_COORDS`.
    """
    h = CITY_COORDS.get(str(home_city).strip().lower())
    a = CITY_COORDS.get(str(away_city).strip().lower())
    if h is None or a is None:
        return float("nan")
    return haversine_km(a[0], a[1], h[0], h[1])
