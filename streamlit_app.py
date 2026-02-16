import os
import subprocess
import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

import arb_scanner as scanner

if get_script_run_ctx() is None and os.environ.get("STREAMLIT_RUN_ACTIVE") != "1":
    os.environ["STREAMLIT_RUN_ACTIVE"] = "1"
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
    sys.exit(0)

st.set_page_config(page_title="NY Live Sports Arbitrage Scanner", layout="wide")
st.title("NY Live Sports Arbitrage Scanner")
st.caption("Find moneyline arbitrage across NY books and size bets intelligently.")

loaded_key = scanner.load_api_key()
if loaded_key:
    st.success("API key loaded from .env or ~/.odds_api_key")

api_key_input = st.text_input("API Key (optional)", type="password", key="api_key_input")
api_key = api_key_input.strip() if api_key_input.strip() else loaded_key

with st.sidebar:
    st.header("Scanner Settings")
    sport = st.selectbox(
        "Sport",
        options=[
            "basketball_nba",
            "americanfootball_nfl",
            "baseball_mlb",
            "icehockey_nhl",
        ],
        index=0,
        key="sport_select",
    )
    ny_only = st.checkbox("NY books only", value=True, key="ny_only")
    total_investment = st.number_input(
        "Total investment ($)",
        min_value=10.0,
        value=100.0,
        step=10.0,
        key="total_investment_main",
    )
    round_choice = st.selectbox(
        "Round stakes to", options=["None", "$5", "$10"], index=1, key="round_choice_main"
    )
    round_to = None if round_choice == "None" else float(round_choice.strip("$"))
    near_threshold = st.number_input(
        "Near-arb threshold (optional)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Set to 1.02 to see near arbs within 2%.",
        key="near_threshold",
    )
    near_threshold = None if near_threshold == 0.0 else float(near_threshold)
    st.caption("NY safe list:")
    ny_books = getattr(scanner, "NY_SAFE_BOOKS", getattr(scanner, "NY_BOOKS", []))
    st.caption(", ".join(sorted(ny_books)))

scan = st.button("Scan for arbs", key="scan_button")
check_books = st.button("Check available books", key="check_books_button")

if check_books:
    if not api_key:
        st.error("Set ODDS_API_KEY in .env or enter it above.")
    else:
        with st.spinner("Fetching books..."):
            odds_json, _, error = scanner.fetch_odds(api_key, sport=sport)
        if error:
            st.error(f"Failed to get odds: {error}")
        else:
            books = scanner.extract_bookmakers(odds_json)
            safe, other = scanner.split_bookmakers(books)
            st.subheader("Books in US region")
            if safe:
                st.write("NY safe books detected:")
                st.write(", ".join(safe))
            if other:
                st.write("Other books detected (filtered out):")
                st.write(", ".join(other))

if scan:
    if not api_key:
        st.error("Set ODDS_API_KEY in .env or enter it above.")
    else:
        with st.spinner("Fetching odds..."):
            odds_json, remaining, error = scanner.fetch_odds(api_key, sport=sport)
        if error:
            st.error(f"Failed to get odds: {error}")
        else:
            if remaining is not None:
                st.caption(f"Requests remaining: {remaining}")
            st.write(f"Scanning {len(odds_json)} upcoming games...")

            arbs, near_arbs = scanner.scan_arbs(
                odds_json, ny_only=ny_only, near_threshold=near_threshold
            )

            if not arbs:
                st.info("No arbitrage opportunities found.")
            else:
                st.subheader(f"Arbitrage opportunities ({len(arbs)})")
                for arb in arbs:
                    away_team = arb["away_team"]
                    home_team = arb["home_team"]
                    away_price = arb["away_price"]
                    home_price = arb["home_price"]
                    away_bookie = arb["away_bookie"]
                    home_bookie = arb["home_bookie"]
                    profit_margin = arb["profit_margin"]

                    st.markdown(f"### {away_team} vs {home_team}")
                    st.write(f"Profit margin: {profit_margin:.2f}%")
                    st.write(f"{away_team}: {away_price} @ {away_bookie}")
                    st.write(f"{home_team}: {home_price} @ {home_bookie}")

                    stakes = scanner.calculate_arb_stakes(
                        away_price,
                        home_price,
                        total_investment=total_investment,
                        round_to=round_to,
                    )
                    if stakes:
                        exact = stakes["exact"]
                        st.write(
                            "Exact stakes: "
                            f"{away_team} ${exact['stake_a']:.2f} | "
                            f"{home_team} ${exact['stake_b']:.2f} | "
                            f"Payout ${exact['payout']:.2f} | "
                            f"Profit ${exact['profit']:.2f} ({exact['roi']:.2f}%)"
                        )
                        rounded = stakes.get("rounded")
                        if rounded:
                            st.write(f"Rounded to nearest ${round_to:.0f}:")
                            st.write(
                                f"{away_team} ${rounded['stake_a']:.2f} | "
                                f"{home_team} ${rounded['stake_b']:.2f} | "
                                f"Total ${rounded['total']:.2f}"
                            )
                            st.write(
                                f"Profit if {away_team} wins: "
                                f"${rounded['profit_a']:.2f} ({rounded['roi_a']:.2f}%)"
                            )
                            st.write(
                                f"Profit if {home_team} wins: "
                                f"${rounded['profit_b']:.2f} ({rounded['roi_b']:.2f}%)"
                            )
                            if rounded["profit_a"] < 0 or rounded["profit_b"] < 0:
                                st.warning(
                                    "Rounded stakes remove the arb on one side. "
                                    "Try smaller rounding or a larger total stake."
                                )

            if near_arbs:
                st.subheader("Near-arbs")
                for near in near_arbs:
                    st.write(
                        f"{near['away_team']} vs {near['home_team']} | "
                        f"Total implied prob: {near['total_implied_prob']:.4f}"
                    )

st.divider()
st.subheader("Quick Stake Calculator")
calc_cols = st.columns(3)
with calc_cols[0]:
    odds_a = st.number_input(
        "Team A odds (decimal)", min_value=1.01, value=2.05, step=0.01, key="odds_a"
    )
with calc_cols[1]:
    odds_b = st.number_input(
        "Team B odds (decimal)", min_value=1.01, value=2.02, step=0.01, key="odds_b"
    )
with calc_cols[2]:
    total_bet = st.number_input(
        "Total investment ($)",
        min_value=10.0,
        value=500.0,
        step=10.0,
        key="total_bet_calc",
    )

calc_round = st.selectbox(
    "Round stakes to", options=["None", "$5", "$10"], index=1, key="round_choice_calc"
)
calc_round_to = None if calc_round == "None" else float(calc_round.strip("$"))

stakes = scanner.calculate_arb_stakes(odds_a, odds_b, total_investment=total_bet, round_to=calc_round_to)
if stakes:
    exact = stakes["exact"]
    st.write(
        f"Exact: Team A ${exact['stake_a']:.2f} | Team B ${exact['stake_b']:.2f} | "
        f"Payout ${exact['payout']:.2f} | Profit ${exact['profit']:.2f} ({exact['roi']:.2f}%)"
    )
    rounded = stakes.get("rounded")
    if rounded:
        st.write(
            f"Rounded: Team A ${rounded['stake_a']:.2f} | Team B ${rounded['stake_b']:.2f} | "
            f"Total ${rounded['total']:.2f}"
        )
        st.write(
            f"Profit if Team A wins: ${rounded['profit_a']:.2f} ({rounded['roi_a']:.2f}%)"
        )
        st.write(
            f"Profit if Team B wins: ${rounded['profit_b']:.2f} ({rounded['roi_b']:.2f}%)"
        )
else:
    st.info("No arbitrage at those odds (total implied probability ≥ 1.0).")

st.divider()
st.subheader("How to Bet (Anti-Ban Checklist)")
st.markdown(
    """
1) The “Weird Number” Rule  
   - Avoid odd stake sizes (e.g., $137.42).  
   - Round to $5 or $10 increments so you look like a normal bettor.  

2) Stick to Main Markets  
   - Moneyline, spreads, totals on major leagues (NBA/NFL/MLB/NHL).  
   - Avoid obscure props/low-liquidity markets.  

3) Avoid Palpable Errors  
   - If ROI looks too good (>5% on a major market), double-check.  
   - Palpable errors are often voided and can flag accounts.  

4) Don’t Withdraw Immediately  
   - Leave winnings in the account for a few days.  
   - Move money in larger, less frequent chunks.  

5) “Mug Bet” Camouflage  
   - Occasionally place a small, recreational bet (e.g., a parlay).  
   - Think of it as a small subscription to keep accounts healthy.  

6) Don’t Double-Dip Promotions  
   - If you use a boost on one book, hedge on a different book.  
   - Never show one book both sides of your trade.  

**Workflow**  
Run the scanner → Verify odds manually → Calculate stakes (round) →  
Place bets quickly → Track bankroll per book in a spreadsheet.
"""
)
