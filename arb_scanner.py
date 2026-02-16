import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# --- CONFIGURATION ---
SPORT = "basketball_nba"  # 'basketball_nba', 'americanfootball_nfl', 'baseball_mlb', etc.
REGIONS = "us"  # 'us' for US books (NY relevant)
MARKETS = "h2h"  # Moneyline winner
ODDS_FORMAT = "decimal"  # Decimal is easiest for math
DATE_FORMAT = "iso"

# NY-legal books (adjust as needed)
NY_SAFE_BOOKS = {
    "DraftKings",
    "FanDuel",
    "Caesars",
    "BetMGM",
    "BetRivers",
    "Fanatics",
    "Fanatics Sportsbook",
    "Bally Bet",
    "BallyBet",
    "Resorts World",
    "Resorts World Bet",
    "PointsBet (US)",
    "ESPN BET",
    "ESPN Bet",
}
NY_BOOKS = NY_SAFE_BOOKS
NY_ONLY = True

# Optional: print games that are close to an arb (e.g., 1.02 means within 2%)
NEAR_ARB_THRESHOLD = None  # e.g., 1.02

# Stake sizing (optional)
ENABLE_STAKE_SIZING = True
TOTAL_INVESTMENT = 100.0
ROUND_TO = 5  # Set to 10 for $10s, or None to disable rounding


def _load_env_file(path: Path) -> bool:
    if not path.exists():
        return False
    loaded = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded = True
    return loaded


def _load_env_files(paths: List[Path]) -> bool:
    loaded_any = False
    for path in paths:
        loaded_any = _load_env_file(path) or loaded_any
    return loaded_any


def _read_key_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def load_api_key() -> Optional[str]:
    _load_env_files(
        [
            Path.cwd() / ".env",
            Path(__file__).with_name(".env"),
            Path.home() / ".env",
        ]
    )
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        api_key = _read_key_file(Path.home() / ".odds_api_key")
        if api_key:
            os.environ["ODDS_API_KEY"] = api_key
    return api_key


def fetch_odds(
    api_key: str,
    sport: str = SPORT,
    regions: str = REGIONS,
    markets: str = MARKETS,
    odds_format: str = ODDS_FORMAT,
    date_format: str = DATE_FORMAT,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
    response = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
        params={
            "api_key": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        },
        timeout=15,
    )

    if response.status_code != 200:
        return None, None, f"{response.status_code}, {response.text}"

    return response.json(), response.headers.get("x-requests-remaining"), None


def _best_prices_for_game(
    game: Dict, ny_only: bool
) -> Tuple[Dict[str, Tuple[float, str]], List[str]]:
    teams = [game.get("home_team"), game.get("away_team")]
    best: Dict[str, Tuple[float, str]] = {team: (0.0, "") for team in teams if team}
    considered_books: List[str] = []

    for bookmaker in game.get("bookmakers", []):
        bookie_name = bookmaker.get("title", "")
        if ny_only and bookie_name not in NY_SAFE_BOOKS:
            continue
        considered_books.append(bookie_name)

        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                team = outcome.get("name")
                price = outcome.get("price", 0)
                if team in best and price and price > best[team][0]:
                    best[team] = (price, bookie_name)

    return best, considered_books


def _round_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return value
    return math.floor((value / increment) + 0.5) * increment


def calculate_arb_stakes(
    odds_a: float,
    odds_b: float,
    total_investment: float = 100.0,
    round_to: Optional[float] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    ip_a = 1 / odds_a
    ip_b = 1 / odds_b
    total_ip = ip_a + ip_b

    if total_ip >= 1:
        return None

    stake_a = (total_investment * ip_a) / total_ip
    stake_b = (total_investment * ip_b) / total_ip
    payout = stake_a * odds_a
    profit = payout - total_investment
    roi = (profit / total_investment) * 100

    result: Dict[str, Dict[str, float]] = {
        "exact": {
            "stake_a": stake_a,
            "stake_b": stake_b,
            "payout": payout,
            "profit": profit,
            "roi": roi,
            "total": total_investment,
            "total_ip": total_ip,
        }
    }

    if round_to:
        rounded_a = _round_to_increment(stake_a, round_to)
        rounded_b = _round_to_increment(stake_b, round_to)
        rounded_total = rounded_a + rounded_b
        payout_a = rounded_a * odds_a
        payout_b = rounded_b * odds_b
        profit_a = payout_a - rounded_total
        profit_b = payout_b - rounded_total
        roi_a = (profit_a / rounded_total) * 100 if rounded_total else 0.0
        roi_b = (profit_b / rounded_total) * 100 if rounded_total else 0.0
        result["rounded"] = {
            "stake_a": rounded_a,
            "stake_b": rounded_b,
            "total": rounded_total,
            "payout_a": payout_a,
            "payout_b": payout_b,
            "profit_a": profit_a,
            "profit_b": profit_b,
            "roi_a": roi_a,
            "roi_b": roi_b,
        }

    return result


def scan_arbs(
    odds_json: List[Dict[str, Any]],
    ny_only: bool = NY_ONLY,
    near_threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    arbs: List[Dict[str, Any]] = []
    near_arbs: List[Dict[str, Any]] = []

    for game in odds_json:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        if not home_team or not away_team:
            continue

        best, _ = _best_prices_for_game(game, ny_only)
        home_price, home_bookie = best.get(home_team, (0.0, ""))
        away_price, away_bookie = best.get(away_team, (0.0, ""))

        if home_price <= 0 or away_price <= 0:
            continue

        implied_prob_home = 1 / home_price
        implied_prob_away = 1 / away_price
        total_implied_prob = implied_prob_home + implied_prob_away

        item = {
            "home_team": home_team,
            "away_team": away_team,
            "home_price": home_price,
            "away_price": away_price,
            "home_bookie": home_bookie,
            "away_bookie": away_bookie,
            "total_implied_prob": total_implied_prob,
            "profit_margin": (1 - total_implied_prob) * 100,
        }

        if total_implied_prob < 1.0:
            arbs.append(item)
        elif near_threshold and total_implied_prob < near_threshold:
            near_arbs.append(item)

    return arbs, near_arbs


def find_arbitrage() -> None:
    api_key = load_api_key()
    if not api_key:
        print("Error: Please set your API key in ODDS_API_KEY or a .env file.")
        sys.exit(1)

    odds_json, remaining, error = fetch_odds(api_key)
    if error:
        print(f"Failed to get odds: {error}")
        return

    remaining = remaining if remaining is not None else None
    if remaining is not None:
        print(f"Requests remaining: {remaining}")
    print(f"Scanning {len(odds_json)} upcoming games for arbitrage...\n")

    arbs, near_arbs = scan_arbs(odds_json, ny_only=NY_ONLY, near_threshold=NEAR_ARB_THRESHOLD)
    if not arbs:
        print("No arbitrage opportunities found.\n")

    for arb in arbs:
        away_team = arb["away_team"]
        home_team = arb["home_team"]
        away_price = arb["away_price"]
        home_price = arb["home_price"]
        away_bookie = arb["away_bookie"]
        home_bookie = arb["home_bookie"]
        profit_margin = arb["profit_margin"]

        print("!!! ARBITRAGE FOUND !!!")
        print(f"Game: {away_team} vs {home_team}")
        print(f"Profit Margin: {profit_margin:.2f}%")
        print(f"  Bet 1 ({away_team}): {away_price} @ {away_bookie}")
        print(f"  Bet 2 ({home_team}): {home_price} @ {home_bookie}")
        if ENABLE_STAKE_SIZING:
            stakes = calculate_arb_stakes(
                away_price,
                home_price,
                total_investment=TOTAL_INVESTMENT,
                round_to=ROUND_TO,
            )
            if stakes:
                exact = stakes["exact"]
                print(f"Stake sizing (total ${exact['total']:.2f}):")
                print(
                    f"  Exact: {away_team} ${exact['stake_a']:.2f} | "
                    f"{home_team} ${exact['stake_b']:.2f} | "
                    f"Payout ${exact['payout']:.2f} | "
                    f"Profit ${exact['profit']:.2f} ({exact['roi']:.2f}%)"
                )
                rounded = stakes.get("rounded")
                if rounded:
                    print(f"  Rounded to nearest ${ROUND_TO}:")
                    print(
                        f"    {away_team}: ${rounded['stake_a']:.2f} | "
                        f"{home_team}: ${rounded['stake_b']:.2f}"
                    )
                    print(f"    Total wagered: ${rounded['total']:.2f}")
                    print(
                        f"    Profit if {away_team} wins: "
                        f"${rounded['profit_a']:.2f} ({rounded['roi_a']:.2f}%)"
                    )
                    print(
                        f"    Profit if {home_team} wins: "
                        f"${rounded['profit_b']:.2f} ({rounded['roi_b']:.2f}%)"
                    )
        print("-" * 30)

    if near_arbs:
        for near in near_arbs:
            away_team = near["away_team"]
            home_team = near["home_team"]
            print("Near-arb:")
            print(f"Game: {away_team} vs {home_team}")
            print(f"Total implied prob: {near['total_implied_prob']:.4f}")
            print(f"  {away_team}: {near['away_price']} @ {near['away_bookie']}")
            print(f"  {home_team}: {near['home_price']} @ {near['home_bookie']}")
            print("-" * 30)


if __name__ == "__main__":
    find_arbitrage()
