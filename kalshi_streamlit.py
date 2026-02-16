import asyncio
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_UP
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websockets = None

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    st_autorefresh = None

try:
    from kalshi_bot import (
        build_rest_headers,
        build_ws_headers,
        load_settings,
        parse_message,
        pick_default_market_ticker,
        subscribe,
        ws_connect_kwargs,
    )
except Exception:  # pragma: no cover - keep UI alive if imports fail
    build_rest_headers = None
    build_ws_headers = None
    load_settings = None
    parse_message = None
    pick_default_market_ticker = None
    subscribe = None
    ws_connect_kwargs = None


if get_script_run_ctx() is None and os.environ.get("STREAMLIT_RUN_ACTIVE") != "1":
    os.environ["STREAMLIT_RUN_ACTIVE"] = "1"
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
    sys.exit(0)


def kalshi_fee(price_dollars: Decimal, contracts: int, rate: Decimal) -> Decimal:
    raw = rate * Decimal(contracts) * price_dollars * (Decimal("1") - price_dollars)
    return (raw * 100).quantize(Decimal("1"), rounding=ROUND_UP) / 100


st.set_page_config(page_title="Kalshi Arb Suite", layout="wide")
st.title("Kalshi Arb Suite")
st.caption("Monitor live state and backtest fee-aware negative spread strategies.")


def normalize_price_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.dropna().empty:
        return clean
    max_val = clean.max()
    if max_val > 1:
        clean = clean / 100
    return clean


def fee_series(price: pd.Series, contracts: int, rate: float) -> pd.Series:
    rate_dec = Decimal(str(rate))
    output: list[float] = []
    for value in price:
        if pd.isna(value):
            output.append(np.nan)
            continue
        fee = kalshi_fee(Decimal(str(float(value))), int(contracts), rate_dec)
        output.append(float(fee))
    return pd.Series(output, index=price.index, dtype=float)


def fee_series_variable_rate(
    price: pd.Series, contracts: int, rate: pd.Series
) -> pd.Series:
    aligned_rate = rate.reindex(price.index)
    output: list[float] = []
    for idx, value in price.items():
        row_rate = aligned_rate.loc[idx] if idx in aligned_rate.index else np.nan
        if pd.isna(value) or pd.isna(row_rate):
            output.append(np.nan)
            continue
        fee = kalshi_fee(
            Decimal(str(float(value))),
            int(contracts),
            Decimal(str(float(row_rate))),
        )
        output.append(float(fee))
    return pd.Series(output, index=price.index, dtype=float)


def _market_text_blob(market_ticker: Optional[str], market: Optional[dict] = None) -> str:
    fields = []
    if market_ticker:
        fields.append(str(market_ticker))
    if isinstance(market, dict):
        for key in (
            "ticker",
            "market_ticker",
            "event_ticker",
            "series_ticker",
            "rulebook_ticker",
            "rulebook_name",
            "title",
            "subtitle",
        ):
            value = market.get(key)
            if value:
                fields.append(str(value))
    return " ".join(fields).upper()


def infer_taker_rate_for_market(
    base_rate: float,
    market_ticker: Optional[str],
    market: Optional[dict] = None,
    enable_special_schedule: bool = True,
) -> float:
    if not enable_special_schedule:
        return float(base_rate)
    text = _market_text_blob(market_ticker, market)
    if not text:
        return float(base_rate)
    if "NASDAQ100" in text:
        return min(float(base_rate), 0.035)
    tokens = re.split(r"[^A-Z0-9]+", text)
    if any(token.startswith("INX") for token in tokens if token):
        return min(float(base_rate), 0.035)
    return float(base_rate)


def taker_rate_series_for_tickers(
    ticker_series: pd.Series,
    base_rate: float,
    enable_special_schedule: bool,
) -> pd.Series:
    if not enable_special_schedule:
        return pd.Series(float(base_rate), index=ticker_series.index, dtype=float)
    return ticker_series.apply(
        lambda value: infer_taker_rate_for_market(
            float(base_rate),
            str(value) if value is not None else None,
            market=None,
            enable_special_schedule=True,
        )
    ).astype(float)


def build_ask_series(
    df: pd.DataFrame,
    ask_col: Optional[str],
    bid_col: Optional[str],
) -> pd.Series:
    if ask_col:
        return normalize_price_series(df[ask_col])
    if bid_col:
        bid = normalize_price_series(df[bid_col])
        return 1 - bid
    return pd.Series(dtype=float)


def _price_to_dollars(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    is_cents = isinstance(value, int)
    if isinstance(value, str):
        raw = value.strip()
        is_cents = raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit())
    try:
        dec = Decimal(str(value))
    except Exception:
        return None
    if is_cents and dec >= 1:
        dec = dec / Decimal("100")
    elif dec > 1:
        dec = dec / Decimal("100")
    if dec < 0 or dec > 1:
        return None
    return dec


def _normalize_levels(
    levels: object, depth_levels: int = 50, allow_zero: bool = False
) -> list[tuple[Decimal, float]]:
    if levels is None:
        return []
    bids: list[tuple[Decimal, float]] = []
    if isinstance(levels, dict):
        for key, value in levels.items():
            price = _price_to_dollars(key)
            if price is None:
                continue
            try:
                size = float(value)
            except Exception:
                size = 0.0
            if size < 0 or (size == 0 and not allow_zero):
                continue
            bids.append((price, size))
    elif isinstance(levels, list):
        for level in levels:
            price = None
            size = None
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                price = _price_to_dollars(level[0])
                try:
                    size = float(level[1])
                except Exception:
                    size = None
            elif isinstance(level, dict):
                price = _price_to_dollars(
                    level.get("price_dollars")
                    or level.get("price")
                    or level.get("p")
                    or level.get("price_cents")
                )
                for key in ("quantity", "qty", "size", "count", "contracts"):
                    if key in level:
                        try:
                            size = float(level.get(key))
                        except Exception:
                            size = None
                        break
            if price is None or size is None or size < 0 or (size == 0 and not allow_zero):
                continue
            bids.append((price, size))
    bids.sort(key=lambda x: x[0], reverse=True)
    if depth_levels > 0:
        bids = bids[:depth_levels]
    return bids


def _best_bid_from_levels(levels: object) -> Optional[Decimal]:
    bids = _normalize_levels(levels, depth_levels=1)
    if not bids:
        return None
    return bids[0][0]


def _ws_payload(message: dict) -> dict:
    if isinstance(message.get("msg"), dict):
        return message["msg"]
    if isinstance(message.get("data"), dict):
        return message["data"]
    if isinstance(message.get("payload"), dict):
        return message["payload"]
    return message


def _ws_market_ticker(message: dict, payload: dict) -> Optional[str]:
    for src in (payload, message):
        ticker = src.get("market_ticker") or src.get("ticker")
        if ticker:
            return str(ticker)
    return None


def _extract_side_levels(
    payload: dict, side: str, allow_zero: bool
) -> tuple[bool, list[tuple[Decimal, float]]]:
    candidates = (
        f"{side}_dollars",
        side,
        f"{side}_fp",
        f"{side}_dollars_fp",
    )
    containers = [payload]
    orderbook = payload.get("orderbook")
    orderbook_fp = payload.get("orderbook_fp")
    if isinstance(orderbook, dict):
        containers.append(orderbook)
    if isinstance(orderbook_fp, dict):
        containers.append(orderbook_fp)
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in candidates:
            if key in container:
                raw = container.get(key)
                return True, _normalize_levels(raw, depth_levels=0, allow_zero=allow_zero)
    return False, []


def _replace_book(book: dict[Decimal, float], levels: list[tuple[Decimal, float]]) -> None:
    book.clear()
    for price, size in levels:
        if size > 0:
            book[price] = size


def _apply_book_delta(book: dict[Decimal, float], levels: list[tuple[Decimal, float]]) -> None:
    for price, size in levels:
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size


def _book_levels(book: dict[Decimal, float], depth_levels: int) -> list[tuple[Decimal, float]]:
    levels = [(price, size) for price, size in book.items() if size > 0]
    levels.sort(key=lambda x: x[0], reverse=True)
    if depth_levels > 0:
        levels = levels[:depth_levels]
    return levels


def fetch_orderbook_snapshot(
    ticker: str,
    settings,
    depth_levels: int = 50,
) -> tuple[Optional[Decimal], Optional[Decimal], list[tuple[Decimal, float]], list[tuple[Decimal, float]], dict]:
    encoded = quote(ticker, safe="")
    path = f"/trade-api/v2/markets/{encoded}/orderbook"
    url = f"https://api.elections.kalshi.com{path}"
    headers = {}
    try:
        if build_rest_headers is not None and settings is not None:
            headers = build_rest_headers(settings, "GET", path)
    except Exception as exc:
        return (
            None,
            None,
            [],
            [],
            {"error": f"auth headers failed: {exc}", "status": None, "latency_ms": None},
        )
    request = Request(url, headers=headers, method="GET")
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
            status = response.getcode()
    except Exception as exc:
        return None, None, [], [], {"error": str(exc), "status": None, "latency_ms": None}
    latency_ms = int((time.perf_counter() - start) * 1000)
    try:
        data = json.loads(payload)
    except Exception as exc:
        return (
            None,
            None,
            [],
            [],
            {"error": f"invalid json: {exc}", "status": status, "latency_ms": latency_ms},
        )

    if isinstance(data, dict) and data.get("error"):
        return (
            None,
            None,
            [],
            [],
            {"error": str(data.get("error")), "status": status, "latency_ms": latency_ms},
        )
    if isinstance(data, dict) and data.get("message"):
        # Some endpoints return message on auth errors
        if str(data.get("message")).lower().startswith("unauthorized"):
            return (
                None,
                None,
                [],
                [],
                {"error": str(data.get("message")), "status": status, "latency_ms": latency_ms},
            )

    orderbook = data.get("orderbook") if isinstance(data, dict) else None
    orderbook_fp = data.get("orderbook_fp") if isinstance(data, dict) else None
    if not isinstance(orderbook, dict) and not isinstance(orderbook_fp, dict):
        return (
            None,
            None,
            [],
            [],
            {"error": "missing orderbook", "status": status, "latency_ms": latency_ms},
        )

    def pick_side(book, side: str):
        if not isinstance(book, dict):
            return None
        return (
            book.get(side)
            or book.get(f"{side}_dollars")
            or book.get(f"{side}_fp")
            or book.get(f"{side}_dollars_fp")
        )

    yes_bids_raw = pick_side(orderbook, "yes") or pick_side(orderbook_fp, "yes")
    no_bids_raw = pick_side(orderbook, "no") or pick_side(orderbook_fp, "no")
    yes_bids = _normalize_levels(yes_bids_raw, depth_levels=depth_levels)
    no_bids = _normalize_levels(no_bids_raw, depth_levels=depth_levels)
    best_yes_bid = yes_bids[0][0] if yes_bids else None
    best_no_bid = no_bids[0][0] if no_bids else None
    yes_ask = (Decimal("1") - best_no_bid) if best_no_bid is not None else None
    no_ask = (Decimal("1") - best_yes_bid) if best_yes_bid is not None else None
    if best_yes_bid is None and best_no_bid is None:
        return (
            None,
            None,
            [],
            [],
            {"error": "empty orderbook", "status": status, "latency_ms": latency_ms},
        )
    return (
        yes_ask,
        no_ask,
        yes_bids,
        no_bids,
        {"error": None, "status": status, "latency_ms": latency_ms},
    )


def _parse_money(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    is_cents = isinstance(value, int)
    if isinstance(value, str):
        raw = value.strip()
        is_cents = raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit())
    try:
        dec = Decimal(str(value))
    except Exception:
        return None
    if is_cents and dec >= 1:
        dec = dec / Decimal("100")
    elif dec > 1:
        dec = dec / Decimal("100")
    if dec < 0 or dec > 1:
        return None
    return dec


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def fetch_market(
    ticker: str,
    settings,
) -> tuple[Optional[dict], Optional[str]]:
    if build_rest_headers is None:
        return None, "missing auth helpers"
    encoded = quote(ticker, safe="")
    path = f"/trade-api/v2/markets/{encoded}"
    url = f"https://api.elections.kalshi.com{path}"
    headers = build_rest_headers(settings, "GET", path)
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
    except Exception as exc:
        return None, str(exc)
    try:
        data = json.loads(payload)
    except Exception as exc:
        return None, f"invalid json: {exc}"
    market = data.get("market") if isinstance(data, dict) else None
    if not isinstance(market, dict):
        return None, "missing market"
    return market, None


def market_has_bids(market: dict) -> tuple[Optional[Decimal], Optional[Decimal]]:
    yes_bid = _parse_money(market.get("yes_bid_dollars") or market.get("yes_bid"))
    no_bid = _parse_money(market.get("no_bid_dollars") or market.get("no_bid"))
    return yes_bid, no_bid


def market_has_valid_asks(
    market: dict,
    min_price: Decimal = Decimal("0.01"),
    max_price: Decimal = Decimal("0.99"),
) -> tuple[bool, Optional[Decimal], Optional[Decimal]]:
    yes_ask = _parse_money(market.get("yes_ask_dollars") or market.get("yes_ask"))
    no_ask = _parse_money(market.get("no_ask_dollars") or market.get("no_ask"))
    valid = (
        yes_ask is not None
        and no_ask is not None
        and yes_ask >= min_price
        and yes_ask <= max_price
        and no_ask >= min_price
        and no_ask <= max_price
    )
    return valid, yes_ask, no_ask


def classify_quote_issue(
    yes_ask: Optional[Decimal],
    no_ask: Optional[Decimal],
    min_price: Decimal = Decimal("0.01"),
    max_price: Decimal = Decimal("0.99"),
) -> str:
    if yes_ask is None or no_ask is None:
        return "missing_side"
    if yes_ask <= 0 or no_ask <= 0:
        return "zero_ask"
    if yes_ask >= 1 or no_ask >= 1:
        return "at_one"
    if yes_ask < min_price or no_ask < min_price:
        return "below_min_filter"
    if yes_ask > max_price or no_ask > max_price:
        return "above_max_filter"
    return "ok"


def market_score(market: dict) -> float:
    for key in ("liquidity_dollars", "volume_24h_fp", "open_interest_fp", "volume_fp"):
        value = market.get(key)
        try:
            if value is not None:
                return float(value)
        except Exception:
            continue
    for key in ("volume_24h", "open_interest", "volume"):
        try:
            if market.get(key) is not None:
                return float(market.get(key))
        except Exception:
            continue
    return 0.0


def fetch_markets_page(
    settings,
    limit: int = 50,
    status: str = "open",
    cursor: Optional[str] = None,
    extra_params: Optional[dict[str, str]] = None,
) -> tuple[list[dict], Optional[str], Optional[str]]:
    if build_rest_headers is None:
        return [], None, "missing auth helpers"
    path = "/trade-api/v2/markets"
    params = {"limit": str(limit), "status": status}
    if cursor:
        params["cursor"] = cursor
    if extra_params:
        for key, value in extra_params.items():
            if value is None:
                continue
            params[str(key)] = str(value)
    url = f"https://api.elections.kalshi.com{path}?{urlencode(params)}"
    headers = build_rest_headers(settings, "GET", path)
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
    except Exception as exc:
        return [], None, str(exc)
    try:
        data = json.loads(payload)
    except Exception as exc:
        return [], None, f"invalid json: {exc}"
    markets = data.get("markets") if isinstance(data, dict) else None
    next_cursor = data.get("cursor") or data.get("next_cursor") if isinstance(data, dict) else None
    if not isinstance(markets, list):
        return [], next_cursor, "missing markets"
    return markets, next_cursor, None


def find_liquid_market(
    settings,
    max_checks: int = 200,
    status: str = "open",
    ticker_filters: Optional[list[str]] = None,
    exclude_ticker_filters: Optional[list[str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    checked = 0
    cursor = None
    best_ticker = None
    best_score = -1.0
    fallback_candidates: list[str] = []
    filters = [f.strip().upper() for f in (ticker_filters or []) if f.strip()]
    excludes = [f.strip().upper() for f in (exclude_ticker_filters or []) if f.strip()]
    while checked < max_checks:
        markets, cursor, err = fetch_markets_page(settings, limit=50, status=status, cursor=cursor)
        if err:
            return None, err
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("market_type") not in (None, "binary"):
                continue
            ticker = market.get("ticker")
            if not ticker:
                continue
            ticker_str = str(ticker)
            if filters and not any(f in ticker_str.upper() for f in filters):
                continue
            if excludes and any(f in ticker_str.upper() for f in excludes):
                continue
            valid_asks, _yes_ask, _no_ask = market_has_valid_asks(market)
            yes_bid, no_bid = market_has_bids(market)
            if valid_asks or (
                yes_bid is not None
                and no_bid is not None
                and yes_bid > Decimal("0")
                and yes_bid < Decimal("1")
                and no_bid > Decimal("0")
                and no_bid < Decimal("1")
            ):
                checked += 1
                score = market_score(market)
                if score > best_score:
                    best_score = score
                    best_ticker = ticker_str
            else:
                fallback_candidates.append(ticker_str)
            if checked >= max_checks:
                break
        if best_ticker:
            return best_ticker, None
        if not cursor:
            break
    # Fallback: check orderbooks for a small sample if market list lacks bid fields.
    for ticker in fallback_candidates[:50]:
        yes_ask, no_ask, _yes_bids, _no_bids, meta = fetch_orderbook_snapshot(
            ticker, settings
        )
        if (
            yes_ask is not None
            and no_ask is not None
            and yes_ask > 0
            and yes_ask < 1
            and no_ask > 0
            and no_ask < 1
        ):
            return ticker, None
    return None, "no liquid markets found"


def find_two_sided_market(
    settings,
    max_checks: int = 500,
    status: str = "open",
    depth_levels: int = 5,
    min_bid: Decimal = Decimal("0.01"),
    max_bid: Decimal = Decimal("0.99"),
    ticker_filters: Optional[list[str]] = None,
    exclude_ticker_filters: Optional[list[str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    checked = 0
    cursor = None
    fallback_candidates: list[str] = []
    filters = [f.strip().upper() for f in (ticker_filters or []) if f.strip()]
    excludes = [f.strip().upper() for f in (exclude_ticker_filters or []) if f.strip()]
    while checked < max_checks:
        markets, cursor, err = fetch_markets_page(settings, limit=50, status=status, cursor=cursor)
        if err:
            return None, err
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("market_type") not in (None, "binary"):
                continue
            ticker = market.get("ticker")
            if not ticker:
                continue
            ticker_str = str(ticker)
            if filters and not any(f in ticker_str.upper() for f in filters):
                continue
            if excludes and any(f in ticker_str.upper() for f in excludes):
                continue
            valid_asks, yes_ask, no_ask = market_has_valid_asks(
                market, min_price=min_bid, max_price=max_bid
            )
            yes_bid, no_bid = market_has_bids(market)
            if valid_asks:
                return ticker_str, None
            if (
                yes_bid is not None
                and no_bid is not None
                and yes_bid > min_bid
                and yes_bid < max_bid
                and no_bid > min_bid
                and no_bid < max_bid
            ):
                return ticker_str, None
            if yes_ask is not None or no_ask is not None or yes_bid is not None or no_bid is not None:
                checked += 1
                fallback_candidates.append(ticker_str)
            if checked >= max_checks:
                break
        if not cursor:
            break

    for ticker in fallback_candidates[:100]:
        yes_ask, no_ask, yes_bids, no_bids, meta = fetch_orderbook_snapshot(
            ticker, settings, depth_levels=depth_levels
        )
        if meta and meta.get("error"):
            continue
        if yes_ask is not None and no_ask is not None and yes_bids and no_bids:
            return ticker, None
    return None, "no two-sided markets found"


def _event_group_key(market: dict) -> Optional[str]:
    for key in ("event_ticker", "event", "event_code"):
        value = market.get(key)
        if value:
            return str(value)
    ticker = str(market.get("ticker") or "").strip()
    if not ticker:
        return None
    parts = ticker.split("-")
    if len(parts) >= 4 and parts[-2].isdigit():
        return "-".join(parts[:-2])
    if len(parts) >= 3:
        return "-".join(parts[:-1])
    return ticker


def fetch_event_markets(
    settings,
    event_key: str,
    max_pages: int = 8,
) -> tuple[list[dict], dict]:
    stats = {
        "event_lookup_mode": "event_ticker",
        "event_fetch_pages": 0,
        "event_fetch_error": None,
    }
    seen: set[str] = set()
    matched: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        markets, cursor, err = fetch_markets_page(
            settings,
            limit=200,
            status="open",
            cursor=cursor,
            extra_params={"event_ticker": event_key},
        )
        stats["event_fetch_pages"] += 1
        if err:
            stats["event_fetch_error"] = err
            break
        for market in markets:
            if not isinstance(market, dict):
                continue
            ticker = str(market.get("ticker") or "")
            if not ticker or ticker in seen:
                continue
            group_key = _event_group_key(market)
            if group_key == event_key or str(market.get("event_ticker") or "") == event_key:
                seen.add(ticker)
                matched.append(market)
        if not cursor:
            break

    if matched:
        stats["event_markets_fetched"] = len(matched)
        stats["event_complete_known"] = True
        return matched, stats

    # Fallback when event_ticker filter is unsupported/empty on this environment.
    stats["event_lookup_mode"] = "full_scan_fallback"
    cursor = None
    for _ in range(max_pages):
        markets, cursor, err = fetch_markets_page(
            settings,
            limit=200,
            status="open",
            cursor=cursor,
        )
        stats["event_fetch_pages"] += 1
        if err:
            if stats["event_fetch_error"] is None:
                stats["event_fetch_error"] = err
            break
        for market in markets:
            if not isinstance(market, dict):
                continue
            ticker = str(market.get("ticker") or "")
            if not ticker or ticker in seen:
                continue
            group_key = _event_group_key(market)
            if group_key == event_key:
                seen.add(ticker)
                matched.append(market)
        if not cursor:
            break

    stats["event_markets_fetched"] = len(matched)
    stats["event_complete_known"] = len(matched) > 0
    return matched, stats


def strict_confirm_event_basket(
    settings,
    event_key: str,
    scan_legs: list[dict],
    contracts: int,
    base_fee_rate: float,
    safety_margin: float,
    depth_levels: int,
    auto_special_taker_fees: bool,
    require_complete_event: bool,
) -> dict:
    event_markets, fetch_stats = fetch_event_markets(settings, event_key)
    complete_known = bool(fetch_stats.get("event_complete_known", False))
    if event_markets:
        tickers = [str(m.get("ticker")) for m in event_markets if m.get("ticker")]
    else:
        tickers = [str(leg.get("market_ticker")) for leg in scan_legs if leg.get("market_ticker")]

    expected_legs = len(tickers)
    if expected_legs == 0:
        return {
            "strict_status": "missing_event_legs",
            "strict_reason": "no event legs available",
            "strict_legs_expected": 0,
            "strict_legs_fillable": 0,
            "strict_cost_yes": np.nan,
            "strict_distance_to_arb": np.nan,
            "strict_crossed": False,
            "strict_complete_known": complete_known,
            "strict_complete_event": False,
            "strict_lookup_mode": fetch_stats.get("event_lookup_mode"),
            "strict_top_fillable_contracts": 0,
            "strict_top_cost_per_contract": np.nan,
            "strict_top_ev_per_contract": np.nan,
            "strict_top_ev_total": np.nan,
            "strict_leg_rows": [],
        }

    gross_yes = 0.0
    fee_yes = 0.0
    fillable = 0
    failures = 0
    leg_rows: list[dict] = []
    for ticker in tickers:
        row_rate = infer_taker_rate_for_market(
            float(base_fee_rate),
            ticker,
            market=None,
            enable_special_schedule=bool(auto_special_taker_fees),
        )
        yes_ask, _no_ask, _yes_bids, no_bids, meta = fetch_orderbook_snapshot(
            ticker,
            settings,
            depth_levels=max(int(depth_levels), int(contracts)),
        )
        top_no_bid = no_bids[0][0] if no_bids else None
        top_no_bid_size = float(no_bids[0][1]) if no_bids else 0.0
        top_yes_ask = (Decimal("1") - top_no_bid) if top_no_bid is not None else None
        top_contracts = int(np.floor(top_no_bid_size)) if top_no_bid_size > 0 else 0
        if meta and meta.get("error"):
            failures += 1
            leg_rows.append(
                {
                    "market_ticker": ticker,
                    "fee_rate_applied": float(row_rate),
                    "top_yes_ask": float(top_yes_ask) if top_yes_ask is not None else np.nan,
                    "top_no_bid_size": top_no_bid_size,
                    "top_fillable_contracts": top_contracts,
                    "req_exec_yes": np.nan,
                    "req_fillable_contracts": 0.0,
                    "req_fillable": False,
                    "error": str(meta.get("error")),
                }
            )
            continue
        yes_exec, filled = effective_ask_from_bids(no_bids, int(contracts))
        if yes_exec is None or filled < float(contracts):
            failures += 1
            leg_rows.append(
                {
                    "market_ticker": ticker,
                    "fee_rate_applied": float(row_rate),
                    "top_yes_ask": float(top_yes_ask) if top_yes_ask is not None else np.nan,
                    "top_no_bid_size": top_no_bid_size,
                    "top_fillable_contracts": top_contracts,
                    "req_exec_yes": np.nan,
                    "req_fillable_contracts": float(filled),
                    "req_fillable": False,
                    "error": "insufficient depth",
                }
            )
            continue
        row_fee = float(kalshi_fee(yes_exec, int(contracts), Decimal(str(row_rate))))
        gross_yes += float(yes_exec)
        fee_yes += row_fee
        fillable += 1
        leg_rows.append(
            {
                "market_ticker": ticker,
                "fee_rate_applied": float(row_rate),
                "top_yes_ask": float(top_yes_ask) if top_yes_ask is not None else np.nan,
                "top_no_bid_size": top_no_bid_size,
                "top_fillable_contracts": top_contracts,
                "req_exec_yes": float(yes_exec),
                "req_fillable_contracts": float(filled),
                "req_fillable": True,
                "error": None,
            }
        )

    complete_event = complete_known and (fillable == expected_legs)
    all_fillable = fillable == expected_legs

    if require_complete_event and not complete_known:
        status = "incomplete_event_unknown"
        reason = "could not confirm full event leg set"
    elif not all_fillable:
        status = "unfillable_legs"
        reason = f"{fillable}/{expected_legs} legs fillable at depth"
    else:
        status = "ready"
        reason = "all legs fillable"

    cost_yes = gross_yes + fee_yes if all_fillable else np.nan
    crossed = bool(np.isfinite(cost_yes) and cost_yes < float(safety_margin))
    if status == "ready":
        status = "crossed" if crossed else "no_cross"

    top_fillable_contracts = 0
    top_cost_per_contract = float("nan")
    top_ev_per_contract = float("nan")
    top_ev_total = float("nan")
    top_ready = (
        len(leg_rows) == expected_legs
        and all(np.isfinite(float(row.get("top_yes_ask", np.nan))) for row in leg_rows)
        and all(float(row.get("top_no_bid_size", 0.0)) > 0 for row in leg_rows)
    )
    if top_ready:
        top_fillable_contracts = min(
            int(row.get("top_fillable_contracts", 0)) for row in leg_rows
        )
        if top_fillable_contracts > 0:
            gross_top = sum(float(row["top_yes_ask"]) for row in leg_rows)
            fee_top = 0.0
            for row in leg_rows:
                row_ask = Decimal(str(float(row["top_yes_ask"])))
                row_rate = Decimal(str(float(row.get("fee_rate_applied", base_fee_rate))))
                fee_top += float(kalshi_fee(row_ask, 1, row_rate))
            top_cost_per_contract = gross_top + fee_top
            top_ev_per_contract = 1.0 - top_cost_per_contract
            top_ev_total = top_ev_per_contract * float(top_fillable_contracts)

    return {
        "strict_status": status,
        "strict_reason": reason,
        "strict_legs_expected": expected_legs,
        "strict_legs_fillable": fillable,
        "strict_failures": failures,
        "strict_cost_yes": cost_yes,
        "strict_distance_to_arb": (cost_yes - float(safety_margin)) if np.isfinite(cost_yes) else np.nan,
        "strict_crossed": crossed,
        "strict_complete_known": complete_known,
        "strict_complete_event": complete_event,
        "strict_lookup_mode": fetch_stats.get("event_lookup_mode"),
        "strict_top_fillable_contracts": int(top_fillable_contracts),
        "strict_top_cost_per_contract": top_cost_per_contract,
        "strict_top_ev_per_contract": top_ev_per_contract,
        "strict_top_ev_total": top_ev_total,
        "strict_leg_rows": leg_rows,
    }


def replay_strict_event_basket(
    settings,
    event_key: str,
    contracts: int,
    base_fee_rate: float,
    safety_margin: float,
    depth_levels: int,
    auto_special_taker_fees: bool,
    require_complete_event: bool,
    replay_delays_ms: tuple[int, ...] = (0, 100, 250, 500),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_rows: list[dict] = []
    leg_rows: list[dict] = []
    base_ts = time.time()
    for delay_ms in replay_delays_ms:
        if delay_ms > 0:
            time.sleep(float(delay_ms) / 1000.0)
        strict_info = strict_confirm_event_basket(
            settings=settings,
            event_key=event_key,
            scan_legs=[],
            contracts=int(contracts),
            base_fee_rate=float(base_fee_rate),
            safety_margin=float(safety_margin),
            depth_levels=int(depth_levels),
            auto_special_taker_fees=bool(auto_special_taker_fees),
            require_complete_event=bool(require_complete_event),
        )
        replay_rows.append(
            {
                "event_key": event_key,
                "delay_ms": int(delay_ms),
                "elapsed_ms": int((time.time() - base_ts) * 1000),
                "strict_status": strict_info.get("strict_status"),
                "strict_crossed": bool(strict_info.get("strict_crossed", False)),
                "strict_cost_yes": strict_info.get("strict_cost_yes"),
                "strict_distance_to_arb": strict_info.get("strict_distance_to_arb"),
                "strict_top_fillable_contracts": strict_info.get(
                    "strict_top_fillable_contracts"
                ),
                "strict_top_ev_total": strict_info.get("strict_top_ev_total"),
            }
        )
        for leg in strict_info.get("strict_leg_rows", []) or []:
            leg_rows.append(
                {
                    "event_key": event_key,
                    "delay_ms": int(delay_ms),
                    **leg,
                }
            )
    return pd.DataFrame(replay_rows), pd.DataFrame(leg_rows)


def scan_event_basket_candidates(
    settings,
    max_checks: int,
    fee_rate: float,
    contracts: int,
    safety_margin: float,
    top_n: int = 10,
    require_valid_quotes: bool = True,
    require_explicit_asks: bool = True,
    auto_special_taker_fees: bool = True,
    strict_executable: bool = False,
    strict_require_complete_event: bool = True,
    strict_filter_only: bool = False,
    strict_confirm_top_n: int = 10,
    strict_depth_levels: int = 20,
    ticker_filters: Optional[list[str]] = None,
    exclude_ticker_filters: Optional[list[str]] = None,
    min_liquidity_dollars: float = 0.0,
    min_volume_24h: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    cursor = None
    scanned = 0
    filters = [f.strip().upper() for f in (ticker_filters or []) if f.strip()]
    excludes = [f.strip().upper() for f in (exclude_ticker_filters or []) if f.strip()]
    stats = {
        "markets_scanned": 0,
        "legs_used": 0,
        "legs_skipped_invalid": 0,
        "legs_skipped_implied": 0,
        "events_with_legs": 0,
        "events_with_2plus_legs": 0,
        "events_crossed": 0,
        "filtered_by_liquidity": 0,
        "filtered_by_volume": 0,
        "strict_checked": 0,
        "strict_crossed": 0,
        "strict_ready": 0,
        "strict_failed": 0,
    }
    event_legs: dict[str, list[dict]] = {}
    while scanned < max_checks:
        markets, cursor, err = fetch_markets_page(settings, limit=50, status="open", cursor=cursor)
        if err:
            stats["error"] = err
            break
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("market_type") not in (None, "binary"):
                continue
            ticker = market.get("ticker")
            if not ticker:
                continue
            ticker_str = str(ticker)
            if filters and not any(f in ticker_str.upper() for f in filters):
                continue
            if excludes and any(f in ticker_str.upper() for f in excludes):
                continue
            liquidity = _parse_float(
                market.get("liquidity_dollars") or market.get("liquidity")
            )
            vol_24h = _parse_float(
                market.get("volume_24h_fp") or market.get("volume_24h")
            )
            if min_liquidity_dollars and (
                liquidity is None or liquidity < min_liquidity_dollars
            ):
                stats["filtered_by_liquidity"] += 1
                continue
            if min_volume_24h and (vol_24h is None or vol_24h < min_volume_24h):
                stats["filtered_by_volume"] += 1
                continue
            scanned += 1
            stats["markets_scanned"] += 1
            event_key = _event_group_key(market)
            if not event_key:
                continue
            yes_bid, no_bid = market_has_bids(market)
            yes_ask = _parse_money(market.get("yes_ask_dollars") or market.get("yes_ask"))
            no_ask = _parse_money(market.get("no_ask_dollars") or market.get("no_ask"))
            source = "market_ask"
            if yes_ask is None or no_ask is None:
                if require_explicit_asks:
                    stats["legs_skipped_implied"] += 1
                    continue
                if yes_bid is None or no_bid is None:
                    continue
                yes_ask = Decimal("1") - no_bid
                no_ask = Decimal("1") - yes_bid
                source = "implied_from_bids"
            quote_issue = classify_quote_issue(yes_ask, no_ask)
            valid_quote = quote_issue == "ok"
            if require_valid_quotes and not valid_quote:
                stats["legs_skipped_invalid"] += 1
                continue
            row_fee_rate = infer_taker_rate_for_market(
                float(fee_rate),
                ticker_str,
                market=market,
                enable_special_schedule=bool(auto_special_taker_fees),
            )
            fee_yes = float(kalshi_fee(yes_ask, contracts, Decimal(str(row_fee_rate))))
            event_legs.setdefault(event_key, []).append(
                {
                    "market_ticker": ticker_str,
                    "yes_ask": float(yes_ask),
                    "no_ask": float(no_ask),
                    "yes_bid": float(yes_bid) if yes_bid is not None else None,
                    "no_bid": float(no_bid) if no_bid is not None else None,
                    "fee_yes": fee_yes,
                    "fee_rate_applied": row_fee_rate,
                    "source": source,
                    "quote_issue": quote_issue,
                    "liquidity_dollars": liquidity,
                    "volume_24h": vol_24h,
                }
            )
            stats["legs_used"] += 1
            if scanned >= max_checks:
                break
        if not cursor:
            break

    stats["events_with_legs"] = len(event_legs)
    rows: list[dict] = []
    for event_key, legs in event_legs.items():
        if len(legs) < 2:
            continue
        stats["events_with_2plus_legs"] += 1
        gross_yes = float(sum(leg["yes_ask"] for leg in legs))
        fee_yes = float(sum(leg["fee_yes"] for leg in legs))
        cost_yes = gross_yes + fee_yes
        distance = cost_yes - float(safety_margin)
        crossed = cost_yes < float(safety_margin)
        if crossed:
            stats["events_crossed"] += 1
        top_legs = sorted(legs, key=lambda leg: leg["yes_ask"])[: min(8, len(legs))]
        rows.append(
            {
                "event_key": event_key,
                "legs": len(legs),
                "gross_yes": gross_yes,
                "fee_yes": fee_yes,
                "cost_yes": cost_yes,
                "distance_to_arb": distance,
                "crossed": crossed,
                "min_leg_yes": float(min(leg["yes_ask"] for leg in legs)),
                "max_leg_yes": float(max(leg["yes_ask"] for leg in legs)),
                "sample_legs": ", ".join(str(leg["market_ticker"]) for leg in top_legs),
                "avg_fee_rate": float(np.mean([leg["fee_rate_applied"] for leg in legs])),
            }
        )

    if not rows:
        return pd.DataFrame(), stats
    df = pd.DataFrame(rows).sort_values("cost_yes")

    if strict_executable:
        check_n = max(1, min(int(strict_confirm_top_n), len(df)))
        for idx in df.head(check_n).index:
            event_key = str(df.loc[idx, "event_key"])
            strict_info = strict_confirm_event_basket(
                settings=settings,
                event_key=event_key,
                scan_legs=event_legs.get(event_key, []),
                contracts=int(contracts),
                base_fee_rate=float(fee_rate),
                safety_margin=float(safety_margin),
                depth_levels=int(strict_depth_levels),
                auto_special_taker_fees=bool(auto_special_taker_fees),
                require_complete_event=bool(strict_require_complete_event),
            )
            stats["strict_checked"] += 1
            if strict_info.get("strict_status") in {"crossed", "no_cross"}:
                stats["strict_ready"] += 1
            else:
                stats["strict_failed"] += 1
            if strict_info.get("strict_crossed"):
                stats["strict_crossed"] += 1
            for key, value in strict_info.items():
                if isinstance(value, (list, dict)):
                    continue
                df.loc[idx, key] = value
        df["strict_status"] = df.get("strict_status", pd.Series(index=df.index)).fillna("not_checked")
        if strict_filter_only:
            df = df[df["strict_status"].isin(["crossed", "no_cross"])]

    return df.head(top_n), stats


def scan_negative_spread_candidates(
    settings,
    max_checks: int,
    depth_levels: int,
    fee_rate: float,
    contracts: int,
    safety_margin: float,
    top_n: int = 10,
    confirm_top_n: int = 10,
    confirm_all: bool = False,
    confirm_limit: int = 200,
    ticker_filters: Optional[list[str]] = None,
    exclude_ticker_filters: Optional[list[str]] = None,
    require_valid_quotes: bool = False,
    require_explicit_asks: bool = False,
    auto_special_taker_fees: bool = True,
    min_liquidity_dollars: float = 0.0,
    min_volume_24h: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    scored: list[
        tuple[
            float,
            str,
            Optional[Decimal],
            Optional[Decimal],
            Optional[Decimal],
            Optional[Decimal],
        ]
    ] = []
    cursor = None
    stats = {
        "markets_scanned": 0,
        "markets_with_bids": 0,
        "orderbook_confirmed": 0,
        "orderbook_errors": 0,
        "orderbook_fallback_used": False,
        "invalid_asks": 0,
        "ask_from_market": 0,
        "ask_implied": 0,
        "only_invalid_quotes": False,
        "skipped_implied": 0,
        "filtered_by_liquidity": 0,
        "filtered_by_volume": 0,
    }
    market_meta: dict[str, dict[str, Optional[float]]] = {}
    filters = [f.strip().upper() for f in (ticker_filters or []) if f.strip()]
    excludes = [f.strip().upper() for f in (exclude_ticker_filters or []) if f.strip()]
    while len(scored) < max_checks:
        markets, cursor, err = fetch_markets_page(settings, limit=50, status="open", cursor=cursor)
        if err:
            stats["error"] = err
            break
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("market_type") not in (None, "binary"):
                continue
            ticker = market.get("ticker")
            if not ticker:
                continue
            ticker_str = str(ticker)
            if filters and not any(f in ticker_str.upper() for f in filters):
                continue
            if excludes and any(f in ticker_str.upper() for f in excludes):
                continue
            liquidity = _parse_float(
                market.get("liquidity_dollars") or market.get("liquidity")
            )
            vol_24h = _parse_float(
                market.get("volume_24h_fp") or market.get("volume_24h")
            )
            if min_liquidity_dollars and (
                liquidity is None or liquidity < min_liquidity_dollars
            ):
                stats["filtered_by_liquidity"] += 1
                continue
            if min_volume_24h and (vol_24h is None or vol_24h < min_volume_24h):
                stats["filtered_by_volume"] += 1
                continue
            stats["markets_scanned"] += 1
            yes_bid, no_bid = market_has_bids(market)
            yes_ask = _parse_money(market.get("yes_ask_dollars") or market.get("yes_ask"))
            no_ask = _parse_money(market.get("no_ask_dollars") or market.get("no_ask"))
            if yes_bid is not None and no_bid is not None:
                stats["markets_with_bids"] += 1
            market_meta[ticker_str] = {
                "liquidity_dollars": liquidity,
                "volume_24h": vol_24h,
                "fee_rate": infer_taker_rate_for_market(
                    float(fee_rate),
                    ticker_str,
                    market=market,
                    enable_special_schedule=bool(auto_special_taker_fees),
                ),
            }
            scored.append((market_score(market), ticker_str, yes_bid, no_bid, yes_ask, no_ask))
            if len(scored) >= max_checks:
                break
        if not cursor:
            break

    if not scored:
        return pd.DataFrame(), stats

    scored.sort(reverse=True, key=lambda x: x[0])
    results = []
    invalid_results = []
    for score, ticker, yes_bid, no_bid, yes_ask, no_ask in scored:
        ask_source = None
        if yes_ask is not None and no_ask is not None:
            ask_source = "market_ask"
            stats["ask_from_market"] += 1
        elif require_explicit_asks:
            stats["skipped_implied"] += 1
            continue
        elif yes_bid is not None and no_bid is not None:
            yes_ask = Decimal("1") - no_bid
            no_ask = Decimal("1") - yes_bid
            ask_source = "implied_from_bids"
            stats["ask_implied"] += 1
        else:
            continue
        quote_issue = classify_quote_issue(yes_ask, no_ask)
        valid_quote = quote_issue == "ok"
        if not valid_quote:
            stats["invalid_asks"] += 1
        meta = market_meta.get(ticker, {})
        row_fee_rate = float(meta.get("fee_rate") or fee_rate)
        fee_yes = float(kalshi_fee(yes_ask, contracts, Decimal(str(row_fee_rate))))
        fee_no = float(kalshi_fee(no_ask, contracts, Decimal(str(row_fee_rate))))
        cost_now = float(yes_ask + no_ask) + fee_yes + fee_no
        gross_now = float(yes_ask + no_ask)
        distance_to_arb = cost_now - float(safety_margin)
        target = results if valid_quote else invalid_results
        target.append(
            {
                "market_ticker": ticker,
                "yes_ask": float(yes_ask),
                "no_ask": float(no_ask),
                "yes_bid": float(yes_bid) if yes_bid is not None else None,
                "no_bid": float(no_bid) if no_bid is not None else None,
                "fee_yes": fee_yes,
                "fee_no": fee_no,
                "gross_now": gross_now,
                "cost_now": cost_now,
                "distance_to_arb": distance_to_arb,
                "crossed": cost_now < safety_margin,
                "score": score,
                "source": ask_source or "market_snapshot",
                "fee_rate_applied": row_fee_rate,
                "valid_quote": valid_quote,
                "quote_issue": quote_issue,
                "liquidity_dollars": meta.get("liquidity_dollars"),
                "volume_24h": meta.get("volume_24h"),
            }
        )
    if not results:
        stats["orderbook_fallback_used"] = True
        for score, ticker, _yes_bid, _no_bid, _yes_ask, _no_ask in scored[: min(200, len(scored))]:
            yes_ask, no_ask, _yes_bids, _no_bids, meta = fetch_orderbook_snapshot(
                ticker, settings, depth_levels=depth_levels
            )
            if meta and meta.get("error"):
                stats["orderbook_errors"] += 1
                continue
            if yes_ask is None or no_ask is None:
                continue
            quote_issue = classify_quote_issue(yes_ask, no_ask)
            valid_quote = quote_issue == "ok"
            if not valid_quote:
                stats["invalid_asks"] += 1
            meta = market_meta.get(ticker, {})
            row_fee_rate = float(meta.get("fee_rate") or fee_rate)
            fee_yes = float(kalshi_fee(yes_ask, contracts, Decimal(str(row_fee_rate))))
            fee_no = float(kalshi_fee(no_ask, contracts, Decimal(str(row_fee_rate))))
            cost_now = float(yes_ask + no_ask) + fee_yes + fee_no
            gross_now = float(yes_ask + no_ask)
            distance_to_arb = cost_now - float(safety_margin)
            target = results if valid_quote else invalid_results
            target.append(
                {
                    "market_ticker": ticker,
                    "yes_ask": float(yes_ask),
                    "no_ask": float(no_ask),
                    "yes_bid": float(Decimal("1") - no_ask),
                    "no_bid": float(Decimal("1") - yes_ask),
                    "fee_yes": fee_yes,
                    "fee_no": fee_no,
                    "gross_now": gross_now,
                    "cost_now": cost_now,
                    "distance_to_arb": distance_to_arb,
                    "crossed": cost_now < safety_margin,
                    "score": score,
                    "source": "orderbook_fallback",
                    "fee_rate_applied": row_fee_rate,
                    "valid_quote": valid_quote,
                    "quote_issue": quote_issue,
                    "liquidity_dollars": meta.get("liquidity_dollars"),
                    "volume_24h": meta.get("volume_24h"),
                }
            )
        if not results and not invalid_results:
            return pd.DataFrame(), stats

    if not results:
        if invalid_results:
            stats["only_invalid_quotes"] = True
            df_invalid = pd.DataFrame(invalid_results)
            if "cost_now" in df_invalid.columns:
                df_invalid = df_invalid.sort_values("cost_now").head(top_n)
            return df_invalid, stats
        return pd.DataFrame(), stats

    df = pd.DataFrame(results)
    if confirm_all:
        stats["orderbook_fallback_used"] = True
        results = []
        invalid_results = []
        limit = max(1, min(confirm_limit, len(scored)))
        for score, ticker, _yes_bid, _no_bid, _yes_ask, _no_ask in scored[:limit]:
            yes_ask, no_ask, _yes_bids, _no_bids, meta = fetch_orderbook_snapshot(
                ticker, settings, depth_levels=depth_levels
            )
            if meta and meta.get("error"):
                stats["orderbook_errors"] += 1
                continue
            if yes_ask is None or no_ask is None:
                continue
            quote_issue = classify_quote_issue(yes_ask, no_ask)
            valid_quote = quote_issue == "ok"
            if not valid_quote:
                stats["invalid_asks"] += 1
            meta = market_meta.get(ticker, {})
            row_fee_rate = float(meta.get("fee_rate") or fee_rate)
            fee_yes = float(kalshi_fee(yes_ask, contracts, Decimal(str(row_fee_rate))))
            fee_no = float(kalshi_fee(no_ask, contracts, Decimal(str(row_fee_rate))))
            cost_now = float(yes_ask + no_ask) + fee_yes + fee_no
            gross_now = float(yes_ask + no_ask)
            distance_to_arb = cost_now - float(safety_margin)
            target = results if valid_quote else invalid_results
            target.append(
                {
                    "market_ticker": ticker,
                    "yes_ask": float(yes_ask),
                    "no_ask": float(no_ask),
                    "yes_bid": float(Decimal("1") - no_ask),
                    "no_bid": float(Decimal("1") - yes_ask),
                    "fee_yes": fee_yes,
                    "fee_no": fee_no,
                    "gross_now": gross_now,
                    "cost_now": cost_now,
                    "distance_to_arb": distance_to_arb,
                    "crossed": cost_now < safety_margin,
                    "score": score,
                    "source": "orderbook",
                    "fee_rate_applied": row_fee_rate,
                    "valid_quote": valid_quote,
                    "quote_issue": quote_issue,
                    "liquidity_dollars": meta.get("liquidity_dollars"),
                    "volume_24h": meta.get("volume_24h"),
                }
            )
        if not results:
            if invalid_results:
                stats["only_invalid_quotes"] = True
                df_invalid = pd.DataFrame(invalid_results)
                if "cost_now" in df_invalid.columns:
                    df_invalid = df_invalid.sort_values("cost_now").head(top_n)
                return df_invalid, stats
            return pd.DataFrame(), stats
        df = pd.DataFrame(results)
    if "cost_now" in df.columns:
        df = df.sort_values("cost_now")
    if require_valid_quotes and "valid_quote" in df.columns:
        df = df[df["valid_quote"] == True]
    confirm_n = max(0, min(confirm_top_n, len(df)))
    if confirm_n > 0:
        for idx in df.head(confirm_n).index:
            ticker = df.loc[idx, "market_ticker"]
            yes_ask, no_ask, yes_bids, no_bids, meta = fetch_orderbook_snapshot(
                ticker, settings, depth_levels=depth_levels
            )
            if meta and meta.get("error"):
                stats["orderbook_errors"] += 1
                continue
            if yes_ask is None or no_ask is None:
                continue
            quote_issue = classify_quote_issue(yes_ask, no_ask)
            valid_quote = quote_issue == "ok"
            if not valid_quote:
                stats["invalid_asks"] += 1
            row_fee_rate = float(df.loc[idx, "fee_rate_applied"]) if "fee_rate_applied" in df.columns else float(fee_rate)
            fee_yes = float(kalshi_fee(yes_ask, contracts, Decimal(str(row_fee_rate))))
            fee_no = float(kalshi_fee(no_ask, contracts, Decimal(str(row_fee_rate))))
            cost_now = float(yes_ask + no_ask) + fee_yes + fee_no
            gross_now = float(yes_ask + no_ask)
            distance_to_arb = cost_now - float(safety_margin)
            df.loc[idx, "yes_ask"] = float(yes_ask)
            df.loc[idx, "no_ask"] = float(no_ask)
            df.loc[idx, "yes_bid"] = float(Decimal("1") - no_ask)
            df.loc[idx, "no_bid"] = float(Decimal("1") - yes_ask)
            df.loc[idx, "fee_yes"] = fee_yes
            df.loc[idx, "fee_no"] = fee_no
            df.loc[idx, "gross_now"] = gross_now
            df.loc[idx, "cost_now"] = cost_now
            df.loc[idx, "distance_to_arb"] = distance_to_arb
            df.loc[idx, "crossed"] = cost_now < safety_margin
            df.loc[idx, "source"] = "orderbook"
            df.loc[idx, "fee_rate_applied"] = row_fee_rate
            df.loc[idx, "valid_quote"] = valid_quote
            df.loc[idx, "quote_issue"] = quote_issue
            stats["orderbook_confirmed"] += 1

    if "cost_now" in df.columns:
        df = df.sort_values("cost_now").head(top_n)
    if require_valid_quotes and "valid_quote" in df.columns:
        df = df[df["valid_quote"] == True]
    return df, stats


def scan_market_visibility(
    settings,
    max_checks: int,
    depth_levels: int,
    fee_rate: float,
    contracts: int,
    safety_margin: float,
    top_n: int = 20,
    require_explicit_asks: bool = False,
    auto_special_taker_fees: bool = True,
    ticker_filters: Optional[list[str]] = None,
    exclude_ticker_filters: Optional[list[str]] = None,
    min_liquidity_dollars: float = 0.0,
    min_volume_24h: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    scored: list[tuple[float, str]] = []
    cursor = None
    stats = {
        "markets_scanned": 0,
        "markets_with_bids": 0,
        "markets_with_asks": 0,
        "valid_quotes": 0,
        "invalid_quotes": 0,
        "skipped_implied": 0,
        "filtered_by_liquidity": 0,
        "filtered_by_volume": 0,
    }
    filters = [f.strip().upper() for f in (ticker_filters or []) if f.strip()]
    excludes = [f.strip().upper() for f in (exclude_ticker_filters or []) if f.strip()]
    while len(scored) < max_checks:
        markets, cursor, err = fetch_markets_page(settings, limit=50, status="open", cursor=cursor)
        if err:
            stats["error"] = err
            break
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("market_type") not in (None, "binary"):
                continue
            ticker = market.get("ticker")
            if not ticker:
                continue
            ticker_str = str(ticker)
            if filters and not any(f in ticker_str.upper() for f in filters):
                continue
            if excludes and any(f in ticker_str.upper() for f in excludes):
                continue
            liquidity = _parse_float(
                market.get("liquidity_dollars") or market.get("liquidity")
            )
            vol_24h = _parse_float(
                market.get("volume_24h_fp") or market.get("volume_24h")
            )
            if min_liquidity_dollars and (
                liquidity is None or liquidity < min_liquidity_dollars
            ):
                stats["filtered_by_liquidity"] += 1
                continue
            if min_volume_24h and (vol_24h is None or vol_24h < min_volume_24h):
                stats["filtered_by_volume"] += 1
                continue
            stats["markets_scanned"] += 1
            scored.append((market_score(market), ticker_str))
            if len(scored) >= max_checks:
                break
        if not cursor:
            break

    rows = []
    for score, ticker in sorted(scored, reverse=True, key=lambda x: x[0]):
        market, err = fetch_market(ticker, settings)
        if err or not market:
            continue
        yes_bid, no_bid = market_has_bids(market)
        yes_ask = _parse_money(market.get("yes_ask_dollars") or market.get("yes_ask"))
        no_ask = _parse_money(market.get("no_ask_dollars") or market.get("no_ask"))
        if yes_bid is not None and no_bid is not None:
            stats["markets_with_bids"] += 1
        if yes_ask is not None and no_ask is not None:
            stats["markets_with_asks"] += 1
        source = "market_ask" if (yes_ask is not None and no_ask is not None) else "implied"
        if yes_ask is None or no_ask is None:
            if require_explicit_asks:
                stats["skipped_implied"] += 1
                continue
            if yes_bid is None or no_bid is None:
                continue
            yes_ask = Decimal("1") - no_bid
            no_ask = Decimal("1") - yes_bid
        quote_issue = classify_quote_issue(yes_ask, no_ask)
        valid_quote = quote_issue == "ok"
        if valid_quote:
            stats["valid_quotes"] += 1
        else:
            stats["invalid_quotes"] += 1
        liquidity = _parse_float(
            market.get("liquidity_dollars") or market.get("liquidity")
        )
        vol_24h = _parse_float(
            market.get("volume_24h_fp") or market.get("volume_24h")
        )
        row_fee_rate = infer_taker_rate_for_market(
            float(fee_rate),
            ticker,
            market=market,
            enable_special_schedule=bool(auto_special_taker_fees),
        )
        fee_yes = float(kalshi_fee(yes_ask, contracts, Decimal(str(row_fee_rate))))
        fee_no = float(kalshi_fee(no_ask, contracts, Decimal(str(row_fee_rate))))
        cost_now = float(yes_ask + no_ask) + fee_yes + fee_no
        gross_now = float(yes_ask + no_ask)
        distance_to_arb = cost_now - float(safety_margin)
        rows.append(
            {
                "market_ticker": ticker,
                "yes_ask": float(yes_ask),
                "no_ask": float(no_ask),
                "yes_bid": float(yes_bid) if yes_bid is not None else None,
                "no_bid": float(no_bid) if no_bid is not None else None,
                "gross_now": gross_now,
                "fee_yes": fee_yes,
                "fee_no": fee_no,
                "cost_now": cost_now,
                "distance_to_arb": distance_to_arb,
                "crossed": cost_now < safety_margin,
                "valid_quote": valid_quote,
                "quote_issue": quote_issue,
                "source": source,
                "fee_rate_applied": row_fee_rate,
                "score": score,
                "liquidity_dollars": liquidity,
                "volume_24h": vol_24h,
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows), stats


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()


def safe_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    exp_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(exp_rerun_fn):
        exp_rerun_fn()


async def capture_ws_data(
    market_ticker: Optional[str],
    channels: tuple[str, ...],
    duration_s: int,
    min_interval_ms: int,
    max_rows: int,
    fallback_to_ticker: bool,
    rest_fallback: bool,
    rest_poll_ms: int,
    depth_levels: int,
    rest_always: bool,
    allow_partial: bool,
) -> tuple[pd.DataFrame, dict]:
    if websockets is None or build_ws_headers is None or load_settings is None:
        raise RuntimeError("WebSocket dependencies are missing.")
    if parse_message is None or subscribe is None or ws_connect_kwargs is None:
        raise RuntimeError("Kalshi WS helpers not available.")

    settings = load_settings()
    headers = build_ws_headers(settings)
    rows = []
    stats = {
        "raw_messages": 0,
        "parsed_messages": 0,
        "last_type": None,
        "last_market": None,
        "last_error": None,
        "last_message": None,
        "fallback": None,
        "channels": list(channels),
        "market_ticker": market_ticker,
        "rest_polls": 0,
        "rest_rows": 0,
        "partial_rows": 0,
        "rest_depth_missing": 0,
        "rest_partial_depth": 0,
        "rest_last_latency_ms": None,
        "rest_avg_latency_ms": None,
        "ws_last_gap_ms": None,
        "ws_avg_gap_ms": None,
        "ws_book_updates": 0,
        "ws_book_rows": 0,
        "ws_book_missing_fields": 0,
    }
    last_emit = 0.0
    last_poll = 0.0
    last_ws_emit = None
    ws_gap_total = 0.0
    ws_gap_count = 0
    rest_latency_total = 0.0
    rest_latency_count = 0
    current_yes: Optional[Decimal] = None
    current_no: Optional[Decimal] = None
    local_yes_book: dict[Decimal, float] = {}
    local_no_book: dict[Decimal, float] = {}
    start = time.monotonic()

    tickers = (market_ticker,) if market_ticker else ()

    def _poll_rest(reason: str) -> None:
        nonlocal last_poll, rest_latency_total, rest_latency_count, current_yes, current_no, last_ws_emit
        if not market_ticker:
            return
        now = time.monotonic()
        if (now - last_poll) < (rest_poll_ms / 1000):
            return
        last_poll = now
        try:
            yes_ask, no_ask, yes_bids, no_bids, meta = fetch_orderbook_snapshot(
                market_ticker, settings, depth_levels=depth_levels
            )
        except Exception as exc:
            stats["last_error"] = f"REST {reason} failed: {exc}"
            return
        if meta:
            stats["rest_last_latency_ms"] = meta.get("latency_ms")
            if meta.get("latency_ms") is not None:
                rest_latency_total += float(meta["latency_ms"])
                rest_latency_count += 1
                stats["rest_avg_latency_ms"] = round(rest_latency_total / rest_latency_count, 2)
            if meta.get("error"):
                stats["last_error"] = str(meta.get("error"))
            if meta.get("status") and meta.get("status") >= 400:
                stats["last_error"] = f"HTTP {meta.get('status')}"
        stats["rest_polls"] += 1
        if not yes_bids or not no_bids or yes_ask is None or no_ask is None:
            stats["rest_depth_missing"] += 1
            if yes_bids or no_bids:
                stats["rest_partial_depth"] += 1
            return
        current_yes = yes_ask
        current_no = no_ask
        if last_ws_emit is not None:
            gap_ms = (now - last_ws_emit) * 1000
            stats["ws_last_gap_ms"] = round(gap_ms, 2)
        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_ticker": market_ticker,
                "yes_ask": float(yes_ask),
                "no_ask": float(no_ask),
                "source": "rest_orderbook",
                "latency_ms": stats["rest_last_latency_ms"],
                "yes_bids": serialize_bids(yes_bids),
                "no_bids": serialize_bids(no_bids),
                "partial": False,
            }
        )
        last_ws_emit = now
        stats["rest_rows"] += 1
        stats["fallback"] = stats["fallback"] or f"rest {reason}"

    async with websockets.connect(settings.ws_url, **ws_connect_kwargs(headers)) as ws:
        if "orderbook_delta" in channels and "ticker" in channels and tickers:
            await subscribe(ws, ("orderbook_delta",), tickers)
            await subscribe(ws, ("ticker",), ())
        else:
            await subscribe(ws, channels, tickers)
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= duration_s:
                break
            timeout = min(1.0, duration_s - elapsed)
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                if rest_fallback and market_ticker:
                    _poll_rest("fallback")
                continue
            data = json.loads(raw)
            stats["raw_messages"] += 1
            if isinstance(data, dict):
                stats["last_type"] = data.get("type")
                if data.get("type") == "error":
                    stats["last_error"] = str(data.get("msg", data))
            stats["last_message"] = data if stats["raw_messages"] <= 3 else stats["last_message"]

            market = None
            yes_ask = None
            no_ask = None
            source = None
            yes_levels_for_row: list[tuple[Decimal, float]] = []
            no_levels_for_row: list[tuple[Decimal, float]] = []

            payload = _ws_payload(data) if isinstance(data, dict) else {}
            msg_type = str(data.get("type")) if isinstance(data, dict) else ""
            if isinstance(payload, dict) and msg_type in {
                "orderbook_snapshot",
                "orderbook_delta",
                "orderbook_update",
            }:
                market = _ws_market_ticker(data, payload)
                if market:
                    stats["last_market"] = market
                if not market_ticker or not market or market == market_ticker:
                    allow_zero = msg_type != "orderbook_snapshot"
                    yes_found, yes_levels = _extract_side_levels(
                        payload, "yes", allow_zero=allow_zero
                    )
                    no_found, no_levels = _extract_side_levels(
                        payload, "no", allow_zero=allow_zero
                    )
                    if yes_found or no_found:
                        stats["ws_book_updates"] += 1
                    else:
                        stats["ws_book_missing_fields"] += 1
                    if msg_type == "orderbook_snapshot":
                        if yes_found:
                            _replace_book(local_yes_book, yes_levels)
                        else:
                            local_yes_book.clear()
                        if no_found:
                            _replace_book(local_no_book, no_levels)
                        else:
                            local_no_book.clear()
                    else:
                        if yes_found:
                            _apply_book_delta(local_yes_book, yes_levels)
                        if no_found:
                            _apply_book_delta(local_no_book, no_levels)

                    yes_levels_for_row = _book_levels(local_yes_book, int(depth_levels))
                    no_levels_for_row = _book_levels(local_no_book, int(depth_levels))
                    if yes_levels_for_row and no_levels_for_row:
                        best_yes_bid = yes_levels_for_row[0][0]
                        best_no_bid = no_levels_for_row[0][0]
                        yes_ask = Decimal("1") - best_no_bid
                        no_ask = Decimal("1") - best_yes_bid
                        source = "orderbook_local"
                        stats["ws_book_rows"] += 1

            if source is None:
                parsed = parse_message(data)
                market, yes_ask, no_ask, source = parsed
            if market:
                stats["last_market"] = market
            if yes_ask is None and no_ask is None:
                if (
                    fallback_to_ticker
                    and "ticker" not in channels
                    and stats["raw_messages"] >= 2
                    and stats["parsed_messages"] == 0
                    and stats["fallback"] is None
                ):
                    await subscribe(ws, ("ticker",), tickers)
                    stats["fallback"] = "subscribed to ticker"
                continue
            stats["parsed_messages"] += 1
            if yes_ask is not None:
                current_yes = yes_ask
            if no_ask is not None:
                current_no = no_ask
            if rest_always:
                _poll_rest("polling")
            if market_ticker and market and market != market_ticker:
                continue
            if (current_yes is None or current_no is None) and not allow_partial:
                continue
            if allow_partial and current_yes is None and current_no is None:
                continue
            now = time.monotonic()
            if min_interval_ms > 0 and (now - last_emit) < (min_interval_ms / 1000):
                continue
            last_emit = now
            if last_ws_emit is not None:
                gap_ms = (now - last_ws_emit) * 1000
                ws_gap_total += gap_ms
                ws_gap_count += 1
                stats["ws_last_gap_ms"] = round(gap_ms, 2)
                stats["ws_avg_gap_ms"] = round(ws_gap_total / ws_gap_count, 2)
            rows.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "market_ticker": market or market_ticker,
                    "yes_ask": float(current_yes) if current_yes is not None else None,
                    "no_ask": float(current_no) if current_no is not None else None,
                    "source": source,
                    "latency_ms": None,
                    "yes_bids": serialize_bids(yes_levels_for_row),
                    "no_bids": serialize_bids(no_levels_for_row),
                    "partial": bool(current_yes is None or current_no is None),
                }
            )
            if current_yes is None or current_no is None:
                stats["partial_rows"] += 1
            last_ws_emit = now
            stats["last_market"] = market or market_ticker
            if max_rows and len(rows) >= max_rows:
                break

    return pd.DataFrame(rows), stats

def apply_time_delay(
    df: pd.DataFrame,
    ts_col: str,
    yes_ask: pd.Series,
    no_ask: pd.Series,
    delay_ms: int,
) -> tuple[pd.Series, pd.Series]:
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    base = pd.DataFrame({"ts": ts, "yes_ask": yes_ask, "no_ask": no_ask})
    base = base.dropna(subset=["ts"]).sort_values("ts")
    if base.empty or delay_ms <= 0:
        return yes_ask, no_ask
    exec_times = base["ts"] + pd.to_timedelta(delay_ms, unit="ms")
    left = pd.DataFrame({"exec_time": exec_times, "idx": base.index}).sort_values("exec_time")
    right = base[["ts", "yes_ask", "no_ask"]].sort_values("ts")
    merged = pd.merge_asof(
        left,
        right,
        left_on="exec_time",
        right_on="ts",
        direction="forward",
        allow_exact_matches=True,
    )
    exec_yes = pd.Series(index=base.index, dtype=float)
    exec_no = pd.Series(index=base.index, dtype=float)
    exec_yes.loc[merged["idx"]] = merged["yes_ask"].to_numpy()
    exec_no.loc[merged["idx"]] = merged["no_ask"].to_numpy()
    return exec_yes.reindex(df.index), exec_no.reindex(df.index)


def apply_time_delay_series(
    df: pd.DataFrame,
    ts_col: str,
    yes_ask: pd.Series,
    no_ask: pd.Series,
    delay_ms: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    base = pd.DataFrame(
        {"ts": ts, "yes_ask": yes_ask, "no_ask": no_ask, "delay_ms": delay_ms}
    )
    base = base.dropna(subset=["ts"]).sort_values("ts")
    if base.empty:
        return yes_ask, no_ask
    exec_times = base["ts"] + pd.to_timedelta(base["delay_ms"], unit="ms")
    left = pd.DataFrame({"exec_time": exec_times, "idx": base.index}).sort_values("exec_time")
    right = base[["ts", "yes_ask", "no_ask"]].sort_values("ts")
    merged = pd.merge_asof(
        left,
        right,
        left_on="exec_time",
        right_on="ts",
        direction="forward",
        allow_exact_matches=True,
    )
    exec_yes = pd.Series(index=base.index, dtype=float)
    exec_no = pd.Series(index=base.index, dtype=float)
    exec_yes.loc[merged["idx"]] = merged["yes_ask"].to_numpy()
    exec_no.loc[merged["idx"]] = merged["no_ask"].to_numpy()
    return exec_yes.reindex(df.index), exec_no.reindex(df.index)


def apply_signal_cooldown(
    signal: pd.Series, ts_col: str, df: pd.DataFrame, cooldown_ms: int
) -> pd.Series:
    if cooldown_ms <= 0 or ts_col == "(none)":
        return signal
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    last_time = None
    filtered = []
    for idx, is_signal in signal.items():
        if not is_signal:
            filtered.append(False)
            continue
        current_time = ts.loc[idx]
        if pd.isna(current_time):
            filtered.append(False)
            continue
        if last_time is None or (current_time - last_time).total_seconds() * 1000 >= cooldown_ms:
            filtered.append(True)
            last_time = current_time
        else:
            filtered.append(False)
    return pd.Series(filtered, index=signal.index)


def _parse_bids_value(value: object) -> list[tuple[Decimal, float]]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except Exception:
            return []
    if not isinstance(data, list):
        return []
    bids: list[tuple[Decimal, float]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            price = _price_to_dollars(item[0])
            try:
                size = float(item[1])
            except Exception:
                size = None
        elif isinstance(item, dict):
            price = _price_to_dollars(
                item.get("price_dollars")
                or item.get("price")
                or item.get("p")
                or item.get("price_cents")
            )
            size = None
            for key in ("quantity", "qty", "size", "count", "contracts"):
                if key in item:
                    try:
                        size = float(item.get(key))
                    except Exception:
                        size = None
                    break
        else:
            continue
        if price is None or size is None or size <= 0:
            continue
        bids.append((price, size))
    bids.sort(key=lambda x: x[0], reverse=True)
    return bids


def effective_ask_from_bids(
    bids: list[tuple[Decimal, float]], contracts: int
) -> tuple[Optional[Decimal], float]:
    if not bids or contracts <= 0:
        return None, 0.0
    remaining = float(contracts)
    filled = 0.0
    cost = Decimal("0")
    for price, size in bids:
        if remaining <= 0:
            break
        take = min(size, remaining)
        cost += (Decimal("1") - price) * Decimal(str(take))
        filled += take
        remaining -= take
    if filled < contracts:
        return None, filled
    avg_price = cost / Decimal(str(contracts))
    return avg_price, filled


def serialize_bids(bids: list[tuple[Decimal, float]] | None) -> list[list[float]]:
    if not bids:
        return []
    output = []
    for price, size in bids:
        try:
            output.append([float(price), float(size)])
        except Exception:
            continue
    return output

def apply_row_delay(
    yes_ask: pd.Series,
    no_ask: pd.Series,
    delay_rows: int,
) -> tuple[pd.Series, pd.Series]:
    if delay_rows <= 0:
        return yes_ask, no_ask
    return yes_ask.shift(-delay_rows), no_ask.shift(-delay_rows)


def apply_best_candidate() -> None:
    ticker = st.session_state.get("scan_best_ticker")
    if ticker:
        st.session_state["market_ticker_input"] = ticker


def _read_jsonl_rows(path: Path, max_rows: int = 10000) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
    except Exception:
        return []
    if max_rows > 0 and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def _append_jsonl_row(path: Path, row: dict) -> Optional[str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        return None
    except Exception as exc:
        return str(exc)


def ensure_log_loaded(
    session_key: str,
    loaded_key: str,
    path: Path,
    enabled: bool,
    max_rows: int = 10000,
) -> None:
    if not enabled:
        return
    path_str = str(path)
    if st.session_state.get(loaded_key) == path_str:
        return
    loaded = _read_jsonl_rows(path, max_rows=max_rows)
    st.session_state[session_key] = loaded
    st.session_state[loaded_key] = path_str


def _best_numeric(df: Optional[pd.DataFrame], col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return float("nan")
    series = pd.to_numeric(df[col], errors="coerce")
    if not series.notna().any():
        return float("nan")
    return float(series.min())


def _latest_numeric(df: Optional[pd.DataFrame], col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return float("nan")
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def _max_numeric(df: Optional[pd.DataFrame], col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return float("nan")
    series = pd.to_numeric(df[col], errors="coerce")
    if not series.notna().any():
        return float("nan")
    return float(series.max())


def _count_truthy(df: Optional[pd.DataFrame], col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    series = df[col].fillna(False)
    return int(series.astype(bool).sum())


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p_hat = float(successes) / float(trials)
    denom = 1.0 + (z * z / float(trials))
    center = (p_hat + (z * z / (2.0 * float(trials)))) / denom
    half = (
        z
        * math.sqrt(
            (p_hat * (1.0 - p_hat) + (z * z / (4.0 * float(trials))))
            / float(trials)
        )
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def append_availability_log(
    source: str,
    safety_margin: float,
    single_df: Optional[pd.DataFrame] = None,
    basket_df: Optional[pd.DataFrame] = None,
    scan_stats: Optional[dict] = None,
    basket_stats: Optional[dict] = None,
    extra_stats: Optional[dict] = None,
    persist_to_file: bool = False,
    persist_path: Optional[Path] = None,
) -> None:
    log = st.session_state.setdefault("arb_availability_log", [])
    best_single_cost = _best_numeric(single_df, "cost_now")
    best_single_distance = _best_numeric(single_df, "distance_to_arb")
    single_cross_count = _count_truthy(single_df, "crossed")
    best_basket_cost = _best_numeric(basket_df, "cost_yes")
    best_basket_distance = _best_numeric(basket_df, "distance_to_arb")
    basket_cross_count = _count_truthy(basket_df, "crossed")
    best_strict_basket_cost = _best_numeric(basket_df, "strict_cost_yes")
    best_strict_basket_distance = _best_numeric(basket_df, "strict_distance_to_arb")
    best_strict_top_fillable = _max_numeric(basket_df, "strict_top_fillable_contracts")
    best_strict_top_ev_total = _max_numeric(basket_df, "strict_top_ev_total")
    strict_cross_count = 0
    strict_ready_count = 0
    if basket_df is not None and not basket_df.empty and "strict_status" in basket_df.columns:
        strict_status = basket_df["strict_status"].fillna("").astype(str)
        strict_cross_count = int((strict_status == "crossed").sum())
        strict_ready_count = int(strict_status.isin(["crossed", "no_cross"]).sum())

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "safety_margin": float(safety_margin),
        "single_rows": int(single_df.shape[0]) if single_df is not None else 0,
        "single_cross_count": single_cross_count,
        "best_single_cost": best_single_cost,
        "best_single_distance": best_single_distance,
        "basket_rows": int(basket_df.shape[0]) if basket_df is not None else 0,
        "basket_cross_count": basket_cross_count,
        "best_basket_cost": best_basket_cost,
        "best_basket_distance": best_basket_distance,
        "strict_ready_count": strict_ready_count,
        "strict_cross_count": strict_cross_count,
        "best_strict_basket_cost": best_strict_basket_cost,
        "best_strict_basket_distance": best_strict_basket_distance,
        "best_strict_top_fillable_contracts": best_strict_top_fillable,
        "best_strict_top_ev_total": best_strict_top_ev_total,
        "scan_markets_scanned": int((scan_stats or {}).get("markets_scanned", 0)),
        "basket_events_scanned": int((basket_stats or {}).get("events_with_legs", 0)),
    }
    if isinstance(extra_stats, dict):
        for key, value in extra_stats.items():
            row[str(key)] = value
    log.append(row)
    if persist_to_file and persist_path is not None:
        _append_jsonl_row(persist_path, row)
    if len(log) > 5000:
        st.session_state["arb_availability_log"] = log[-5000:]


def append_opportunity_journal(
    source: str,
    safety_margin: float,
    near_arb_cents: float,
    single_df: Optional[pd.DataFrame] = None,
    basket_df: Optional[pd.DataFrame] = None,
    replay_summary: Optional[dict] = None,
    persist_to_file: bool = False,
    persist_path: Optional[Path] = None,
) -> dict:
    near_threshold = float(near_arb_cents) / 100.0
    ts = datetime.now(timezone.utc).isoformat()
    entries: list[dict] = []

    if single_df is not None and not single_df.empty:
        single_copy = single_df.copy()
        single_copy["distance_to_arb"] = pd.to_numeric(
            single_copy.get("distance_to_arb"), errors="coerce"
        )
        single_copy["cost_now"] = pd.to_numeric(single_copy.get("cost_now"), errors="coerce")
        if "crossed" in single_copy.columns:
            cross_rows = single_copy[single_copy["crossed"].fillna(False).astype(bool)]
            for _, row in cross_rows.iterrows():
                entries.append(
                    {
                        "timestamp": ts,
                        "source": source,
                        "kind": "single_cross",
                        "target": str(row.get("market_ticker") or ""),
                        "cost": float(row.get("cost_now")),
                        "distance_to_arb": float(row.get("distance_to_arb")),
                        "safety_margin": float(safety_margin),
                    }
                )
        near_rows = single_copy[single_copy["distance_to_arb"] <= near_threshold]
        for _, row in near_rows.iterrows():
            entries.append(
                {
                    "timestamp": ts,
                    "source": source,
                    "kind": "single_near",
                    "target": str(row.get("market_ticker") or ""),
                    "cost": float(row.get("cost_now")),
                    "distance_to_arb": float(row.get("distance_to_arb")),
                    "safety_margin": float(safety_margin),
                }
            )

    if basket_df is not None and not basket_df.empty:
        basket_copy = basket_df.copy()
        basket_copy["distance_to_arb"] = pd.to_numeric(
            basket_copy.get("distance_to_arb"), errors="coerce"
        )
        basket_copy["strict_distance_to_arb"] = pd.to_numeric(
            basket_copy.get("strict_distance_to_arb"), errors="coerce"
        )
        basket_copy["strict_top_ev_total"] = pd.to_numeric(
            basket_copy.get("strict_top_ev_total"), errors="coerce"
        )
        basket_copy["strict_top_fillable_contracts"] = pd.to_numeric(
            basket_copy.get("strict_top_fillable_contracts"), errors="coerce"
        )
        if "strict_status" in basket_copy.columns:
            cross_rows = basket_copy[
                basket_copy["strict_status"].fillna("").astype(str) == "crossed"
            ]
            for _, row in cross_rows.iterrows():
                entries.append(
                    {
                        "timestamp": ts,
                        "source": source,
                        "kind": "basket_strict_cross",
                        "target": str(row.get("event_key") or ""),
                        "cost": float(row.get("strict_cost_yes")),
                        "distance_to_arb": float(row.get("strict_distance_to_arb")),
                        "top_fillable_contracts": float(
                            row.get("strict_top_fillable_contracts")
                        )
                        if pd.notna(row.get("strict_top_fillable_contracts"))
                        else np.nan,
                        "top_ev_total": float(row.get("strict_top_ev_total"))
                        if pd.notna(row.get("strict_top_ev_total"))
                        else np.nan,
                        "safety_margin": float(safety_margin),
                    }
                )
        near_rows = basket_copy[basket_copy["distance_to_arb"] <= near_threshold]
        for _, row in near_rows.iterrows():
            entries.append(
                {
                    "timestamp": ts,
                    "source": source,
                    "kind": "basket_near",
                    "target": str(row.get("event_key") or ""),
                    "cost": float(row.get("cost_yes")),
                    "distance_to_arb": float(row.get("distance_to_arb")),
                    "safety_margin": float(safety_margin),
                }
            )

    if replay_summary:
        entries.append(
            {
                "timestamp": ts,
                "source": source,
                "kind": "replay_summary",
                "events_replayed": int(replay_summary.get("events_replayed", 0)),
                "events_persisted": int(replay_summary.get("events_persisted", 0)),
                "persistence_rate": replay_summary.get("persistence_rate", np.nan),
            }
        )

    journal = st.session_state.setdefault("arb_opportunity_journal", [])
    journal.extend(entries)
    if len(journal) > 20000:
        st.session_state["arb_opportunity_journal"] = journal[-20000:]
        journal = st.session_state["arb_opportunity_journal"]
    if persist_to_file and persist_path is not None:
        for row in entries:
            _append_jsonl_row(persist_path, row)

    counts = {
        "entries_added": int(len(entries)),
        "single_cross": int(sum(1 for e in entries if e.get("kind") == "single_cross")),
        "single_near": int(sum(1 for e in entries if e.get("kind") == "single_near")),
        "basket_strict_cross": int(
            sum(1 for e in entries if e.get("kind") == "basket_strict_cross")
        ),
        "basket_near": int(sum(1 for e in entries if e.get("kind") == "basket_near")),
    }
    return counts


def render_availability_panel(
    safety_margin: float,
    near_arb_cents: float,
    default_window: int = 200,
) -> None:
    log = st.session_state.get("arb_availability_log", [])
    st.subheader("Arb Availability Monitor")
    if not log:
        st.info("No availability history yet. Run scans/watch to build an objective track record.")
        return
    log_df = pd.DataFrame(log)
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce", utc=True)
    log_df = log_df.sort_values("timestamp")
    window_n = st.number_input(
        "Availability window (latest scans)",
        min_value=20,
        value=max(20, min(default_window, len(log_df))),
        step=10,
        key="availability_window_n",
    )
    view = log_df.tail(int(window_n)).copy()
    has_single = view["best_single_cost"].notna()
    has_strict = view["best_strict_basket_cost"].notna()
    single_cross_rate = (
        float((view.loc[has_single, "single_cross_count"] > 0).mean()) if has_single.any() else 0.0
    )
    strict_cross_rate = (
        float((view.loc[has_strict, "strict_cross_count"] > 0).mean()) if has_strict.any() else 0.0
    )
    near_threshold = float(near_arb_cents) / 100.0
    near_single_rate = (
        float(
            (
                pd.to_numeric(view.loc[has_single, "best_single_distance"], errors="coerce")
                <= near_threshold
            ).mean()
        )
        if has_single.any()
        else 0.0
    )
    best_single = _best_numeric(view, "best_single_cost")
    best_strict = _best_numeric(view, "best_strict_basket_cost")
    best_single_dist = _best_numeric(view, "best_single_distance")
    best_strict_dist = _best_numeric(view, "best_strict_basket_distance")
    best_strict_fillable = _max_numeric(view, "best_strict_top_fillable_contracts")
    best_strict_ev_total = _max_numeric(view, "best_strict_top_ev_total")
    watch_view = (
        view[view["source"] == "watch"].copy()
        if "source" in view.columns
        else pd.DataFrame()
    )
    watch_count = int(watch_view.shape[0])
    expected_interval = _latest_numeric(watch_view, "watch_interval_s")
    gap_s = (
        pd.to_numeric(watch_view.get("watch_gap_s"), errors="coerce")
        if not watch_view.empty
        else pd.Series(dtype=float)
    )
    median_gap_s = (
        float(gap_s.dropna().median())
        if not gap_s.dropna().empty
        else float("nan")
    )
    missed_intervals = int(
        pd.to_numeric(watch_view.get("watch_missed_intervals"), errors="coerce")
        .fillna(0)
        .sum()
    ) if not watch_view.empty else 0
    scan_seconds = (
        pd.to_numeric(watch_view.get("watch_scan_seconds"), errors="coerce")
        if not watch_view.empty
        else pd.Series(dtype=float)
    )
    avg_scan_s = (
        float(scan_seconds.dropna().mean())
        if not scan_seconds.dropna().empty
        else float("nan")
    )
    watch_scans_per_min = float("nan")
    if watch_count >= 2 and "timestamp" in watch_view.columns:
        t0 = watch_view["timestamp"].iloc[0]
        t1 = watch_view["timestamp"].iloc[-1]
        if pd.notna(t0) and pd.notna(t1):
            elapsed_min = max((t1 - t0).total_seconds() / 60.0, 1e-9)
            watch_scans_per_min = float((watch_count - 1) / elapsed_min)
    strict_cross_scans = int(
        (
            pd.to_numeric(watch_view.get("strict_cross_count"), errors="coerce")
            .fillna(0)
            > 0
        ).sum()
    ) if not watch_view.empty else 0
    strict_cross_ci_lo, strict_cross_ci_hi = wilson_interval(
        strict_cross_scans,
        watch_count,
    )
    strict_crosses_per_hour = float("nan")
    if np.isfinite(expected_interval) and float(expected_interval) > 0:
        strict_crosses_per_hour = (
            float(strict_cross_scans) / float(max(watch_count, 1))
        ) * (3600.0 / float(expected_interval))
    strict_ev_series = (
        pd.to_numeric(watch_view.get("best_strict_top_ev_total"), errors="coerce")
        if not watch_view.empty
        else pd.Series(dtype=float)
    )
    persist_series = (
        pd.to_numeric(watch_view.get("watch_replay_persistence_rate"), errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
        if not watch_view.empty
        else pd.Series(dtype=float)
    )
    realized_ev_proxy = (
        strict_ev_series.clip(lower=0.0) * persist_series
        if not strict_ev_series.empty
        else pd.Series(dtype=float)
    )
    avg_ev_per_scan = (
        float(realized_ev_proxy.mean())
        if not realized_ev_proxy.empty
        and realized_ev_proxy.notna().any()
        else float("nan")
    )
    ev_per_hour_proxy = float("nan")
    if np.isfinite(avg_ev_per_scan) and np.isfinite(expected_interval) and float(expected_interval) > 0:
        ev_per_hour_proxy = float(avg_ev_per_scan) * (3600.0 / float(expected_interval))

    m1 = st.columns(4)
    m1[0].metric("Scans Logged", f"{len(log_df)}")
    m1[1].metric("Watch Scans", f"{watch_count}")
    m1[2].metric("Single Cross Rate", f"{single_cross_rate:.2%}")
    m1[3].metric("Near-Arb Rate", f"{near_single_rate:.2%}")

    m2 = st.columns(4)
    m2[0].metric("Strict Basket Cross Rate", f"{strict_cross_rate:.2%}")
    m2[1].metric("Best Single Cost", f"{best_single:.4f}" if np.isfinite(best_single) else "n/a")
    m2[2].metric("Best Strict Basket Cost", f"{best_strict:.4f}" if np.isfinite(best_strict) else "n/a")
    m2[3].metric(
        "Best Strict Basket EV",
        f"{best_strict_ev_total:.2f}" if np.isfinite(best_strict_ev_total) else "n/a",
    )

    m3 = st.columns(4)
    m3[0].metric(
        "Watch Interval Target (s)",
        f"{expected_interval:.2f}" if np.isfinite(expected_interval) else "n/a",
    )
    m3[1].metric(
        "Watch Interval Observed (s)",
        f"{median_gap_s:.2f}" if np.isfinite(median_gap_s) else "n/a",
    )
    m3[2].metric(
        "Watch Scans / Min",
        f"{watch_scans_per_min:.2f}" if np.isfinite(watch_scans_per_min) else "n/a",
    )
    m3[3].metric("Missed Intervals", f"{missed_intervals}")

    m4 = st.columns(4)
    m4[0].metric(
        "Best Single Distance",
        f"{best_single_dist:.4f}" if np.isfinite(best_single_dist) else "n/a",
    )
    m4[1].metric(
        "Best Strict Distance",
        f"{best_strict_dist:.4f}" if np.isfinite(best_strict_dist) else "n/a",
    )
    m4[2].metric(
        "Best Strict Fillable Contracts",
        f"{best_strict_fillable:.0f}" if np.isfinite(best_strict_fillable) else "n/a",
    )
    m4[3].metric(
        "Avg Watch Scan Time (s)",
        f"{avg_scan_s:.2f}" if np.isfinite(avg_scan_s) else "n/a",
    )

    m5 = st.columns(4)
    m5[0].metric(
        "Strict Cross Scans",
        f"{strict_cross_scans}/{watch_count}" if watch_count > 0 else "0/0",
    )
    m5[1].metric(
        "Strict Cross 95% CI",
        (
            f"{strict_cross_ci_lo:.2%} to {strict_cross_ci_hi:.2%}"
            if np.isfinite(strict_cross_ci_lo)
            and np.isfinite(strict_cross_ci_hi)
            else "n/a"
        ),
    )
    m5[2].metric(
        "Strict Crosses / Hour",
        f"{strict_crosses_per_hour:.2f}" if np.isfinite(strict_crosses_per_hour) else "n/a",
    )
    m5[3].metric(
        "EV / Hour (Proxy)",
        f"{ev_per_hour_proxy:.2f}" if np.isfinite(ev_per_hour_proxy) else "n/a",
    )

    verdict = "Insufficient evidence yet"
    if len(view) >= 20:
        if single_cross_rate == 0 and strict_cross_rate == 0:
            if np.isfinite(best_single) and best_single > float(safety_margin) + 0.05:
                verdict = "Current regime: very low taker-arb availability"
            else:
                verdict = "Current regime: no confirmed arb yet, but near-arb windows appear"
        elif strict_cross_rate > 0:
            verdict = "Strict executable basket arb appears intermittently"
        else:
            verdict = "Single-market crosses appear intermittently"
    if watch_count >= 50 and np.isfinite(strict_cross_ci_hi) and strict_cross_ci_hi < 0.01:
        verdict = "Watch evidence suggests strict crosses are currently very rare (<1%)."
    if np.isfinite(ev_per_hour_proxy) and ev_per_hour_proxy < 0:
        verdict = "Observed strict opportunities are currently negative after replay persistence."
    st.caption(f"Availability verdict: {verdict}")

    chart = view[["timestamp", "best_single_cost", "best_basket_cost", "best_strict_basket_cost"]].copy()
    chart = chart.set_index("timestamp")
    chart["safety_margin"] = float(safety_margin)
    st.line_chart(chart)
    with st.expander("Availability log (latest)"):
        st.dataframe(view.tail(50), use_container_width=True)


def render_opportunity_journal_panel() -> None:
    journal = st.session_state.get("arb_opportunity_journal", [])
    st.subheader("Opportunity Journal")
    if not journal:
        st.info("No opportunities logged yet.")
        return
    df = pd.DataFrame(journal)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp")
    latest = df.tail(500).copy()
    kind_series = (
        latest["kind"]
        if "kind" in latest.columns
        else pd.Series(index=latest.index, dtype=str)
    )
    kind_series = kind_series.fillna("").astype(str)
    strict_cross = latest[kind_series == "basket_strict_cross"]
    single_cross = latest[kind_series == "single_cross"]
    strict_near = latest[kind_series == "basket_near"]
    single_near = latest[kind_series == "single_near"]
    top_ev_series = (
        latest["top_ev_total"]
        if "top_ev_total" in latest.columns
        else pd.Series(index=latest.index, dtype=float)
    )
    top_ev = pd.to_numeric(top_ev_series, errors="coerce")
    top_ev_non_na = top_ev.dropna()
    top_ev_sum = float(top_ev_non_na.sum()) if not top_ev_non_na.empty else float("nan")
    top_ev_mean = float(top_ev_non_na.mean()) if not top_ev_non_na.empty else float("nan")

    c1 = st.columns(5)
    c1[0].metric("Journal Entries", f"{int(df.shape[0])}")
    c1[1].metric("Strict Basket Crosses", f"{int(strict_cross.shape[0])}")
    c1[2].metric("Single Crosses", f"{int(single_cross.shape[0])}")
    c1[3].metric("Strict Near", f"{int(strict_near.shape[0])}")
    c1[4].metric("Single Near", f"{int(single_near.shape[0])}")

    c2 = st.columns(2)
    c2[0].metric(
        "Strict EV Sum (Top Fill)",
        f"{top_ev_sum:.2f}" if np.isfinite(top_ev_sum) else "n/a",
    )
    c2[1].metric(
        "Strict EV Mean (Top Fill)",
        f"{top_ev_mean:.2f}" if np.isfinite(top_ev_mean) else "n/a",
    )

    with st.expander("Journal rows (latest 200)"):
        st.dataframe(latest.tail(200), use_container_width=True)


tabs = st.tabs(["Monitor", "Backtest"])

with tabs[0]:
    state_path = Path(
        st.sidebar.text_input("Bot state file", value="kalshi_bot_state.json")
    )
    contracts = st.sidebar.number_input("Contracts", min_value=1, value=1, step=1)
    fee_rate = Decimal(st.sidebar.text_input("Taker fee rate", value="0.07"))
    safety_margin = Decimal(st.sidebar.text_input("Safety margin", value="0.99"))

    st.subheader("Manual Negative-Spread Check")
    cols = st.columns(2)
    with cols[0]:
        yes_ask = Decimal(
            str(st.number_input("YES ask ($)", min_value=0.0, max_value=1.0, value=0.48, step=0.01))
        )
    with cols[1]:
        no_ask = Decimal(
            str(st.number_input("NO ask ($)", min_value=0.0, max_value=1.0, value=0.50, step=0.01))
        )

    fee_yes = kalshi_fee(yes_ask, contracts, fee_rate)
    fee_no = kalshi_fee(no_ask, contracts, fee_rate)
    total = yes_ask + no_ask + fee_yes + fee_no

    st.write(f"Fee YES: ${fee_yes:.2f}")
    st.write(f"Fee NO: ${fee_no:.2f}")
    st.write(f"Total cost: ${total:.2f}")

    if total < safety_margin:
        st.success("Trigger: negative spread (after fees).")
    else:
        st.info("No trigger.")

    st.divider()
    st.subheader("Bot State Snapshot")
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            st.json(state)
        except Exception as exc:  # pragma: no cover - UI-only
            st.error(f"Failed to read state file: {exc}")
    else:
        st.caption("State file not found yet. Start `kalshi_bot.py` to create it.")

with tabs[1]:
    st.subheader("Negative-Spread Backtest")
    st.caption("Upload a CSV to backtest fee-aware signals with decision/execution delay.")
    st.info(
        "Backtest results (EV, Summary, Signal Trades) appear below once a data source is loaded. "
        "If you're on Live capture, you must capture data first."
    )

    with st.sidebar:
        st.header("Backtest Parameters")
        bt_contracts = st.number_input("Contracts per trade", min_value=1, value=1, step=1)
        fee_mode = st.selectbox(
            "Fee schedule",
            ["Taker", "Maker", "SPX/NDX Taker", "Custom"],
            index=0,
        )
        if fee_mode == "Maker":
            bt_fee_rate = 0.0175
        elif fee_mode == "SPX/NDX Taker":
            bt_fee_rate = 0.035
        elif fee_mode == "Taker":
            bt_fee_rate = 0.07
        else:
            bt_fee_rate = st.number_input(
                "Custom fee rate", min_value=0.0, value=0.07, step=0.001
            )
        auto_special_taker_fees = st.checkbox(
            "Auto market-specific taker fees (INX/NASDAQ100 at 0.035)",
            value=True,
            disabled=fee_mode != "Taker",
        )
        st.caption(
            "Fee model: ceil(rate * contracts * price * (1-price), to nearest cent), "
            "applied per side."
        )
        if fee_mode == "Taker" and bool(auto_special_taker_fees):
            st.caption(
                "Using auto fee schedule: 0.07 is the base taker rate, and special "
                "markets (e.g., INX/NASDAQ100) are auto-priced at 0.035."
            )
        bt_safety_margin = st.number_input(
            "Safety margin (BT)", min_value=0.5, max_value=1.0, value=0.99, step=0.01
        )
        bt_decision_delay_ms = st.number_input(
            "Decision delay (ms)", min_value=0, value=50, step=10
        )
        bt_execution_delay_ms = st.number_input(
            "Execution delay (ms)", min_value=0, value=250, step=10
        )
        bt_extra_latency_ms = st.number_input(
            "Extra latency (ms)", min_value=0, value=0, step=10
        )
        bt_min_price = st.number_input(
            "Min price filter", min_value=0.0, max_value=0.49, value=0.01, step=0.01
        )
        bt_max_price = st.number_input(
            "Max price filter", min_value=0.51, max_value=1.0, value=0.99, step=0.01
        )
        depth_mode = st.selectbox(
            "Depth mode",
            ["Fallback to top-of-book", "Require depth (strict)", "Off"],
            index=0,
        )
        use_depth = depth_mode != "Off"
        signal_cooldown_ms = st.number_input(
            "Signal cooldown (ms)", min_value=0, value=0, step=50
        )

    data_source = st.radio(
        "Data source",
        ["Live capture (WS)", "Replay recording", "Upload CSV", "Sample data"],
        horizontal=True,
    )

    df = None
    if data_source == "Live capture (WS)":
        st.info("Live WS capture is the most accurate source for arb backtests.")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            default_ticker = ""
            state_path = Path("kalshi_bot_state.json")
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    default_ticker = (
                        state.get("book", {}).get("market_ticker") or ""
                    )
                except Exception:
                    default_ticker = ""
            market_ticker = st.text_input(
                "Market ticker (optional)",
                value=default_ticker,
                key="market_ticker_input",
            )
        with col_b:
            duration_s = st.number_input("Capture seconds", min_value=5, value=60, step=5)
        with col_c:
            min_interval_ms = st.number_input("Min update interval (ms)", min_value=0, value=50, step=10)
        with col_d:
            max_rows = st.number_input("Max rows (0 = no limit)", min_value=0, value=0, step=100)

        channel = st.selectbox("Channel", ["orderbook_delta", "ticker"], index=0)
        auto_select = st.checkbox("Auto-select an open market if ticker is blank", value=True)
        fallback_to_ticker = st.checkbox("Auto-fallback to ticker if orderbook is empty", value=True)
        subscribe_both = st.checkbox("Subscribe to both orderbook_delta + ticker", value=True)
        rest_fallback = st.checkbox("REST fallback if WS is idle", value=True)
        rest_always = st.checkbox("Always poll REST for depth", value=True)
        allow_partial = st.checkbox("Allow partial rows (if one side missing)", value=True)
        rest_poll_ms = st.number_input("REST poll interval (ms)", min_value=250, value=1000, step=250)
        depth_levels = st.number_input("Depth levels (REST)", min_value=1, value=20, step=1)
        seed_from_rest = st.checkbox("Seed from REST snapshot before WS", value=True)
        require_two_sided = st.checkbox("Require two-sided market", value=True)
        fallback_one_sided = st.checkbox(
            "Fallback to one-sided if none found", value=True
        )
        scan_limit = st.number_input("Market scan limit", min_value=50, value=300, step=50)
        auto_best = st.checkbox("Auto-select best arb candidate", value=True)
        save_capture = st.checkbox("Save capture to file", value=True)
        capture_label = st.text_input("Capture label (optional)", value="")
        category_filters = st.multiselect(
            "Category filter (ticker contains)",
            [
                "FED",
                "FOMC",
                "CPI",
                "INFLATION",
                "JOBS",
                "NFP",
                "GDP",
                "RATE",
                "ELECTION",
                "UNEMP",
            ],
            default=[],
        )
        custom_filters = st.text_input(
            "Custom ticker keywords (comma-separated)", value=""
        )
        exclude_filters = st.text_input(
            "Exclude ticker keywords (comma-separated)",
            value="KXMVESPORTSMULTIGAMEEXTENDED",
        )
        liquidity_only = st.checkbox("Liquidity-only scan (valid quotes)", value=True)
        explicit_asks_only = st.checkbox(
            "Require explicit asks (no implied bids->asks)",
            value=False,
        )
        auto_refresh_watch = st.checkbox("Auto-refresh arb watch", value=True)
        persist_arb_logs = st.checkbox("Persist arb logs to disk", value=True)
        load_persisted_logs = st.checkbox(
            "Load persisted logs on startup", value=True
        )
        availability_log_file = st.text_input(
            "Availability log path",
            value="kalshi_recordings/arb_availability_log.jsonl",
        )
        opportunity_log_file = st.text_input(
            "Opportunity journal path",
            value="kalshi_recordings/arb_opportunity_journal.jsonl",
        )
        persisted_rows_limit = st.number_input(
            "Load persisted rows (max)", min_value=100, value=10000, step=100
        )
        min_liquidity = st.number_input(
            "Min liquidity ($)", min_value=0.0, value=0.0, step=10.0
        )
        min_volume = st.number_input(
            "Min 24h volume", min_value=0.0, value=0.0, step=10.0
        )
        near_arb_cents = st.number_input(
            "Near-arb threshold (cents)", min_value=0.0, value=3.0, step=1.0
        )
        near_arb_limit = st.number_input(
            "Near-arb rows", min_value=1, value=10, step=1
        )
        strict_basket_mode = st.checkbox(
            "Strict basket mode (depth + completeness)", value=True
        )
        strict_basket_complete = st.checkbox(
            "Strict basket: require complete event", value=True
        )
        strict_basket_filter_only = st.checkbox(
            "Strict basket: show executable only", value=True
        )
        strict_basket_confirm_n = st.number_input(
            "Strict basket confirm top N", min_value=1, value=3, step=1
        )
        strict_basket_depth_levels = st.number_input(
            "Strict basket depth levels", min_value=1, value=20, step=1
        )
        find_active = st.button("Find active market")
        scan_candidates = st.button("Scan for arb candidates")
        scan_baskets = st.button("Scan event basket arb")
        scan_availability_probe = st.button("Run availability probe")
        show_visibility = st.button("Show market visibility")
        clear_availability_log = st.button("Clear availability log")
        clear_opportunity_journal = st.button("Clear opportunity journal")
        show_market_debug = st.checkbox("Show market debug panel", value=False)
        confirm_mode = st.selectbox(
            "Orderbook confirm mode",
            ["Top N only", "All scanned (slow)"],
            index=0,
        )
        confirm_top_n = st.number_input("Confirm top N via orderbook", min_value=0, value=10, step=1)
        confirm_limit = st.number_input(
            "Orderbook scan cap (slow mode)", min_value=50, value=200, step=50
        )
        filter_list = [f.strip() for f in category_filters if f.strip()]
        if custom_filters.strip():
            filter_list += [f.strip() for f in custom_filters.split(",") if f.strip()]
        exclude_filter_list = [f.strip() for f in exclude_filters.split(",") if f.strip()]
        confirm_all = confirm_mode.startswith("All")
        availability_log_path = Path(availability_log_file).expanduser()
        opportunity_log_path = Path(opportunity_log_file).expanduser()
        ensure_log_loaded(
            session_key="arb_availability_log",
            loaded_key="arb_availability_loaded_path",
            path=availability_log_path,
            enabled=bool(load_persisted_logs),
            max_rows=int(persisted_rows_limit),
        )
        ensure_log_loaded(
            session_key="arb_opportunity_journal",
            loaded_key="arb_opportunity_loaded_path",
            path=opportunity_log_path,
            enabled=bool(load_persisted_logs),
            max_rows=int(persisted_rows_limit),
        )
        if clear_availability_log:
            st.session_state["arb_availability_log"] = []
            st.session_state["arb_availability_loaded_path"] = None
            if bool(persist_arb_logs) and availability_log_path.exists():
                availability_log_path.unlink(missing_ok=True)
            st.success("Availability log cleared.")
        if clear_opportunity_journal:
            st.session_state["arb_opportunity_journal"] = []
            st.session_state["arb_opportunity_loaded_path"] = None
            if bool(persist_arb_logs) and opportunity_log_path.exists():
                opportunity_log_path.unlink(missing_ok=True)
            st.success("Opportunity journal cleared.")

        if show_market_debug and load_settings is not None:
            st.subheader("Market Inspector")
            inspect_ticker = st.text_input("Inspect ticker", value=market_ticker.strip())
            if st.button("Inspect market"):
                settings = load_settings()
                market, err = fetch_market(inspect_ticker, settings)
                if err:
                    st.error(f"Market fetch failed: {err}")
                else:
                    yes_bid, no_bid = market_has_bids(market)
                    valid_asks, yes_ask, no_ask = market_has_valid_asks(market)
                    st.json(
                        {
                            "ticker": market.get("ticker"),
                            "market_type": market.get("market_type"),
                            "status": market.get("status"),
                            "yes_bid_dollars": market.get("yes_bid_dollars"),
                            "no_bid_dollars": market.get("no_bid_dollars"),
                            "yes_ask_dollars": market.get("yes_ask_dollars"),
                            "no_ask_dollars": market.get("no_ask_dollars"),
                            "liquidity_dollars": market.get("liquidity_dollars"),
                            "volume_24h_fp": market.get("volume_24h_fp"),
                            "yes_bid": str(yes_bid) if yes_bid is not None else None,
                            "no_bid": str(no_bid) if no_bid is not None else None,
                            "yes_ask": str(yes_ask) if yes_ask is not None else None,
                            "no_ask": str(no_ask) if no_ask is not None else None,
                            "valid_asks": valid_asks,
                            "quote_issue": classify_quote_issue(yes_ask, no_ask),
                        }
                    )
                    yes_ask, no_ask, yes_bids, no_bids, meta = fetch_orderbook_snapshot(
                        inspect_ticker, settings
                    )
                    st.json(
                        {
                            "orderbook_yes_ask": str(yes_ask) if yes_ask is not None else None,
                            "orderbook_no_ask": str(no_ask) if no_ask is not None else None,
                            "yes_bids_levels": len(yes_bids) if yes_bids else 0,
                            "no_bids_levels": len(no_bids) if no_bids else 0,
                            "best_yes_bid": str(yes_bids[0][0]) if yes_bids else None,
                            "best_no_bid": str(no_bids[0][0]) if no_bids else None,
                            "orderbook_meta": meta,
                        }
                    )

        if find_active and load_settings is not None:
            with st.spinner("Searching for active markets..."):
                settings = load_settings()
                ticker_input = market_ticker.strip()
                if ticker_input:
                    try:
                        yes_ask, no_ask, _yes_bids, _no_bids, meta = fetch_orderbook_snapshot(
                            ticker_input,
                            settings,
                        )
                    except Exception as exc:
                        st.error(f"Orderbook check failed: {exc}")
                    else:
                        if yes_ask is not None and no_ask is not None:
                            st.success("Current ticker has two-sided quotes.")
                        elif meta and meta.get("error"):
                            st.warning(f"No orderbook: {meta.get('error')}")
                else:
                    if require_two_sided:
                        ticker, err = find_two_sided_market(
                            settings,
                            max_checks=int(scan_limit),
                            depth_levels=int(depth_levels),
                            ticker_filters=filter_list,
                            exclude_ticker_filters=exclude_filter_list,
                        )
                    else:
                        ticker, err = find_liquid_market(
                            settings,
                            max_checks=int(scan_limit),
                            ticker_filters=filter_list,
                            exclude_ticker_filters=exclude_filter_list,
                        )
                    if ticker:
                        st.info(f"Try this market: {ticker}")
                    elif err:
                        st.warning(f"Could not find liquid market: {err}")

        if scan_candidates and load_settings is not None:
            with st.spinner("Scanning for arb candidates..."):
                settings = load_settings()
                results, scan_stats = scan_negative_spread_candidates(
                    settings=settings,
                    max_checks=int(scan_limit),
                    depth_levels=int(depth_levels),
                    fee_rate=float(bt_fee_rate),
                    contracts=int(bt_contracts),
                    safety_margin=float(bt_safety_margin),
                    top_n=10,
                    confirm_top_n=int(confirm_top_n),
                    confirm_all=bool(confirm_all),
                    confirm_limit=int(confirm_limit),
                    ticker_filters=filter_list,
                    exclude_ticker_filters=exclude_filter_list,
                    require_valid_quotes=bool(liquidity_only),
                    require_explicit_asks=bool(explicit_asks_only),
                    auto_special_taker_fees=bool(auto_special_taker_fees),
                    min_liquidity_dollars=float(min_liquidity),
                    min_volume_24h=float(min_volume),
                )
            append_availability_log(
                source="scan_single",
                safety_margin=float(bt_safety_margin),
                single_df=results,
                basket_df=None,
                scan_stats=scan_stats,
                basket_stats=None,
                persist_to_file=bool(persist_arb_logs),
                persist_path=availability_log_path,
            )
            append_opportunity_journal(
                source="scan_single",
                safety_margin=float(bt_safety_margin),
                near_arb_cents=float(near_arb_cents),
                single_df=results,
                basket_df=None,
                persist_to_file=bool(persist_arb_logs),
                persist_path=opportunity_log_path,
            )
            if results.empty:
                st.warning("No candidates found in this scan window.")
                st.caption("Scan diagnostics")
                st.json(scan_stats)
            else:
                st.subheader("Top candidates (lowest net cost)")
                st.dataframe(results, use_container_width=True)
                best = results.iloc[0]
                if not bool(best.get("valid_quote", True)) or scan_stats.get("only_invalid_quotes"):
                    st.warning(
                        "Only invalid 0/1 quotes found in this scan. "
                        "No tradable candidates right now."
                    )
                else:
                    st.info(
                        f"Best candidate: {best['market_ticker']} (cost {best['cost_now']:.4f})"
                    )
                    if float(best["cost_now"]) >= float(bt_safety_margin):
                        st.warning(
                            "No taker negative-spread in this scan: best net cost is "
                            f"{float(best['cost_now']):.4f}, threshold is {float(bt_safety_margin):.4f}."
                        )
                    st.session_state["scan_best_ticker"] = best["market_ticker"]
                    st.button("Use best candidate", on_click=apply_best_candidate)
                if "distance_to_arb" in results.columns:
                    st.subheader("Nearest to arb (distance to safety margin)")
                    near_df = results.copy()
                    near_df["distance_to_arb"] = pd.to_numeric(
                        near_df["distance_to_arb"], errors="coerce"
                    )
                    near_df = near_df.sort_values("distance_to_arb")
                    st.dataframe(near_df.head(int(near_arb_limit)), use_container_width=True)
                    near_threshold = float(near_arb_cents) / 100.0
                    within = int((near_df["distance_to_arb"] <= near_threshold).sum())
                    st.caption(
                        f"{within} rows within {near_threshold:.2f} of the safety margin."
                    )
                    if within > 0:
                        st.success(
                            f"Near-arb detected: {within} rows within {near_threshold:.2f}."
                        )
                st.caption("Scan diagnostics")
                st.json(scan_stats)

        if scan_baskets and load_settings is not None:
            with st.spinner("Scanning event baskets..."):
                settings = load_settings()
                basket_df, basket_stats = scan_event_basket_candidates(
                    settings=settings,
                    max_checks=int(scan_limit),
                    fee_rate=float(bt_fee_rate),
                    contracts=int(bt_contracts),
                    safety_margin=float(bt_safety_margin),
                    top_n=20,
                    require_valid_quotes=bool(liquidity_only),
                    require_explicit_asks=bool(explicit_asks_only),
                    auto_special_taker_fees=bool(auto_special_taker_fees),
                    strict_executable=bool(strict_basket_mode),
                    strict_require_complete_event=bool(strict_basket_complete),
                    strict_filter_only=bool(strict_basket_filter_only),
                    strict_confirm_top_n=int(strict_basket_confirm_n),
                    strict_depth_levels=int(strict_basket_depth_levels),
                    ticker_filters=filter_list,
                    exclude_ticker_filters=exclude_filter_list,
                    min_liquidity_dollars=float(min_liquidity),
                    min_volume_24h=float(min_volume),
                )
            append_availability_log(
                source="scan_basket",
                safety_margin=float(bt_safety_margin),
                single_df=None,
                basket_df=basket_df,
                scan_stats=None,
                basket_stats=basket_stats,
                persist_to_file=bool(persist_arb_logs),
                persist_path=availability_log_path,
            )
            append_opportunity_journal(
                source="scan_basket",
                safety_margin=float(bt_safety_margin),
                near_arb_cents=float(near_arb_cents),
                single_df=None,
                basket_df=basket_df,
                persist_to_file=bool(persist_arb_logs),
                persist_path=opportunity_log_path,
            )
            st.subheader("Event Basket Candidates (sum YES across event)")
            if basket_df.empty:
                st.warning("No event-basket candidates found in this scan window.")
                st.caption("Basket diagnostics")
                st.json(basket_stats)
            else:
                st.dataframe(basket_df, use_container_width=True)
                best_basket = basket_df.iloc[0]
                if float(best_basket["cost_yes"]) < float(bt_safety_margin):
                    st.success(
                        f"Basket crossed: {best_basket['event_key']} cost {float(best_basket['cost_yes']):.4f}"
                    )
                else:
                    st.info(
                        "No basket cross in this scan: best basket cost is "
                        f"{float(best_basket['cost_yes']):.4f}."
                    )
                near_threshold = float(near_arb_cents) / 100.0
                near_baskets = basket_df[
                    pd.to_numeric(basket_df["distance_to_arb"], errors="coerce") <= near_threshold
                ]
                st.caption(
                    f"{int(near_baskets.shape[0])} baskets within {near_threshold:.2f} of safety margin."
                )
                if bool(strict_basket_mode):
                    strict_cols = [
                        "strict_status",
                        "strict_crossed",
                        "strict_cost_yes",
                        "strict_distance_to_arb",
                    ]
                    available = [col for col in strict_cols if col in basket_df.columns]
                    if available:
                        strict_df = basket_df.copy()
                        if "strict_status" in strict_df.columns:
                            strict_ready = int(
                                strict_df["strict_status"].isin(["crossed", "no_cross"]).sum()
                            )
                            strict_crossed = int(
                                strict_df["strict_status"].isin(["crossed"]).sum()
                            )
                        else:
                            strict_ready = 0
                            strict_crossed = 0
                        st.subheader("Event Basket Execution Summary")
                        s_cols = st.columns(6)
                        s_cols[0].metric(
                            "Strict Checked",
                            f"{int(basket_stats.get('strict_checked', 0))}",
                        )
                        s_cols[1].metric("Strict Ready", f"{strict_ready}")
                        s_cols[2].metric("Strict Crossed", f"{strict_crossed}")
                        best_strict = pd.to_numeric(
                            strict_df.get("strict_cost_yes"), errors="coerce"
                        )
                        best_strict_val = (
                            float(best_strict.min()) if best_strict.notna().any() else np.nan
                        )
                        s_cols[3].metric(
                            "Best Strict Cost",
                            f"{best_strict_val:.4f}"
                            if np.isfinite(best_strict_val)
                            else "n/a",
                        )
                        best_fillable = pd.to_numeric(
                            strict_df.get("strict_top_fillable_contracts"),
                            errors="coerce",
                        )
                        best_fillable_val = (
                            float(best_fillable.max())
                            if best_fillable.notna().any()
                            else np.nan
                        )
                        s_cols[4].metric(
                            "Best Fillable (Top)",
                            f"{best_fillable_val:.0f}"
                            if np.isfinite(best_fillable_val)
                            else "n/a",
                        )
                        best_top_ev = pd.to_numeric(
                            strict_df.get("strict_top_ev_total"), errors="coerce"
                        )
                        best_top_ev_val = (
                            float(best_top_ev.max())
                            if best_top_ev.notna().any()
                            else np.nan
                        )
                        s_cols[5].metric(
                            "Best Strict EV (Top)",
                            f"{best_top_ev_val:.2f}"
                            if np.isfinite(best_top_ev_val)
                            else "n/a",
                        )
                        if strict_crossed == 0:
                            st.warning(
                                "No strict executable basket arb in this scan window."
                            )
                st.caption("Basket diagnostics")
                st.json(basket_stats)

        if show_visibility and load_settings is not None:
            with st.spinner("Scanning market visibility..."):
                settings = load_settings()
                visibility_df, visibility_stats = scan_market_visibility(
                    settings=settings,
                    max_checks=int(scan_limit),
                    depth_levels=int(depth_levels),
                    fee_rate=float(bt_fee_rate),
                    contracts=int(bt_contracts),
                    safety_margin=float(bt_safety_margin),
                    top_n=20,
                    require_explicit_asks=bool(explicit_asks_only),
                    auto_special_taker_fees=bool(auto_special_taker_fees),
                    ticker_filters=filter_list,
                    exclude_ticker_filters=exclude_filter_list,
                    min_liquidity_dollars=float(min_liquidity),
                    min_volume_24h=float(min_volume),
                )
            append_availability_log(
                source="visibility",
                safety_margin=float(bt_safety_margin),
                single_df=visibility_df,
                basket_df=None,
                scan_stats=visibility_stats,
                basket_stats=None,
                persist_to_file=bool(persist_arb_logs),
                persist_path=availability_log_path,
            )
            append_opportunity_journal(
                source="visibility",
                safety_margin=float(bt_safety_margin),
                near_arb_cents=float(near_arb_cents),
                single_df=visibility_df,
                basket_df=None,
                persist_to_file=bool(persist_arb_logs),
                persist_path=opportunity_log_path,
            )
            st.subheader("Market visibility (top 20 by activity)")
            st.dataframe(visibility_df, use_container_width=True)
            if not visibility_df.empty and "valid_quote" in visibility_df.columns:
                valid_count = int(visibility_df["valid_quote"].sum())
                if valid_count == 0:
                    st.warning(
                        "All visible markets show invalid 0/1 quotes right now. "
                        "That means no true two-sided markets were found in this scan window."
                    )
            if not visibility_df.empty and "distance_to_arb" in visibility_df.columns:
                near_vis = visibility_df.copy()
                near_vis["distance_to_arb"] = pd.to_numeric(
                    near_vis["distance_to_arb"], errors="coerce"
                )
                near_vis = near_vis.sort_values("distance_to_arb")
                st.caption("Closest to arb in visibility scan")
                st.dataframe(
                    near_vis.head(int(near_arb_limit)), use_container_width=True
                )
                near_threshold = float(near_arb_cents) / 100.0
                within = int((near_vis["distance_to_arb"] <= near_threshold).sum())
                st.caption(
                    f"{within} rows within {near_threshold:.2f} of the safety margin."
                )
            st.caption("Visibility diagnostics")
            st.json(visibility_stats)

        if scan_availability_probe and load_settings is not None:
            with st.spinner("Running availability probe..."):
                settings = load_settings()
                probe_single, probe_single_stats = scan_negative_spread_candidates(
                    settings=settings,
                    max_checks=int(scan_limit),
                    depth_levels=int(depth_levels),
                    fee_rate=float(bt_fee_rate),
                    contracts=int(bt_contracts),
                    safety_margin=float(bt_safety_margin),
                    top_n=10,
                    confirm_top_n=max(1, int(confirm_top_n)),
                    confirm_all=False,
                    confirm_limit=int(confirm_limit),
                    ticker_filters=filter_list,
                    exclude_ticker_filters=exclude_filter_list,
                    require_valid_quotes=bool(liquidity_only),
                    require_explicit_asks=bool(explicit_asks_only),
                    auto_special_taker_fees=bool(auto_special_taker_fees),
                    min_liquidity_dollars=float(min_liquidity),
                    min_volume_24h=float(min_volume),
                )
                probe_basket, probe_basket_stats = scan_event_basket_candidates(
                    settings=settings,
                    max_checks=int(scan_limit),
                    fee_rate=float(bt_fee_rate),
                    contracts=int(bt_contracts),
                    safety_margin=float(bt_safety_margin),
                    top_n=10,
                    require_valid_quotes=bool(liquidity_only),
                    require_explicit_asks=bool(explicit_asks_only),
                    auto_special_taker_fees=bool(auto_special_taker_fees),
                    strict_executable=bool(strict_basket_mode),
                    strict_require_complete_event=bool(strict_basket_complete),
                    strict_filter_only=bool(strict_basket_filter_only),
                    strict_confirm_top_n=max(1, int(strict_basket_confirm_n)),
                    strict_depth_levels=int(strict_basket_depth_levels),
                    ticker_filters=filter_list,
                    exclude_ticker_filters=exclude_filter_list,
                    min_liquidity_dollars=float(min_liquidity),
                    min_volume_24h=float(min_volume),
                )
            append_availability_log(
                source="probe",
                safety_margin=float(bt_safety_margin),
                single_df=probe_single,
                basket_df=probe_basket,
                scan_stats=probe_single_stats,
                basket_stats=probe_basket_stats,
                persist_to_file=bool(persist_arb_logs),
                persist_path=availability_log_path,
            )
            append_opportunity_journal(
                source="probe",
                safety_margin=float(bt_safety_margin),
                near_arb_cents=float(near_arb_cents),
                single_df=probe_single,
                basket_df=probe_basket,
                persist_to_file=bool(persist_arb_logs),
                persist_path=opportunity_log_path,
            )
            st.success("Availability probe completed and logged.")
            pcols = st.columns(3)
            pcols[0].metric(
                "Probe Best Single Cost",
                f"{_best_numeric(probe_single, 'cost_now'):.4f}"
                if np.isfinite(_best_numeric(probe_single, "cost_now"))
                else "n/a",
            )
            pcols[1].metric(
                "Probe Best Basket Cost",
                f"{_best_numeric(probe_basket, 'cost_yes'):.4f}"
                if np.isfinite(_best_numeric(probe_basket, "cost_yes"))
                else "n/a",
            )
            pcols[2].metric(
                "Probe Best Strict Basket Cost",
                f"{_best_numeric(probe_basket, 'strict_cost_yes'):.4f}"
                if np.isfinite(_best_numeric(probe_basket, "strict_cost_yes"))
                else "n/a",
            )
            with st.expander("Probe diagnostics"):
                st.json({"single": probe_single_stats, "basket": probe_basket_stats})

        capture = st.button("Capture live data")
        if capture:
            if websockets is None:
                st.error("websockets is not installed in this environment.")
                st.stop()
            if load_settings is None or build_ws_headers is None:
                st.error("Kalshi WS helpers unavailable. Check kalshi_bot.py.")
                st.stop()

            ticker = market_ticker.strip() or None
            settings = load_settings()
            attempts = 0
            max_attempts = 3
            selected_tickers = []
            captured = pd.DataFrame()
            stats = {}
            while attempts < max_attempts:
                if ticker and require_two_sided:
                    yes_ask, no_ask, yes_bids, no_bids, meta = fetch_orderbook_snapshot(
                        ticker, settings, depth_levels=int(depth_levels)
                    )
                    if yes_ask is None or no_ask is None or not yes_bids or not no_bids:
                        if auto_select:
                            ticker = None
                        else:
                            st.error(
                                "Selected market does not have two-sided quotes. "
                                "Enable auto-select or choose a different market."
                            )
                            st.stop()

                if not ticker and auto_select:
                    err = None
                    try:
                        if auto_best:
                            results, scan_stats = scan_negative_spread_candidates(
                                settings=settings,
                                max_checks=int(scan_limit),
                                depth_levels=int(depth_levels),
                                fee_rate=float(bt_fee_rate),
                                contracts=int(bt_contracts),
                                safety_margin=float(bt_safety_margin),
                                top_n=1,
                                confirm_top_n=max(1, int(confirm_top_n))
                                if require_two_sided
                                else int(confirm_top_n),
                                ticker_filters=filter_list,
                                exclude_ticker_filters=exclude_filter_list,
                                require_valid_quotes=bool(liquidity_only),
                                require_explicit_asks=bool(explicit_asks_only),
                                auto_special_taker_fees=bool(auto_special_taker_fees),
                                min_liquidity_dollars=float(min_liquidity),
                                min_volume_24h=float(min_volume),
                            )
                            if not results.empty:
                                if "valid_quote" in results.columns:
                                    valid_rows = results[results["valid_quote"] == True]
                                else:
                                    valid_rows = results
                                if not valid_rows.empty:
                                    ticker = str(valid_rows.iloc[0]["market_ticker"])
                                else:
                                    ticker = None
                                    err = "no valid candidates (only 0/1 quotes)"
                            else:
                                ticker = None
                                err = "no candidates found"
                            if ticker is None:
                                # Fallback to any two-sided/liquid market so capture can proceed
                                if require_two_sided:
                                    ticker, err = find_two_sided_market(
                                        settings,
                                        max_checks=int(scan_limit),
                                        depth_levels=int(depth_levels),
                                        ticker_filters=filter_list,
                                        exclude_ticker_filters=exclude_filter_list,
                                    )
                                if not ticker and fallback_one_sided:
                                    st.warning(
                                        "No two-sided markets found; falling back to one-sided capture."
                                    )
                                    require_two_sided = False
                                    allow_partial = True
                                    ticker, err = find_liquid_market(
                                        settings,
                                        max_checks=int(scan_limit),
                                        ticker_filters=filter_list,
                                        exclude_ticker_filters=exclude_filter_list,
                                    )
                        else:
                            if require_two_sided:
                                ticker, err = find_two_sided_market(
                                    settings,
                                    max_checks=int(scan_limit),
                                    depth_levels=int(depth_levels),
                                    ticker_filters=filter_list,
                                    exclude_ticker_filters=exclude_filter_list,
                                )
                                if not ticker and fallback_one_sided:
                                    st.warning(
                                        "No two-sided markets found; falling back to one-sided capture."
                                    )
                                    require_two_sided = False
                                    allow_partial = True
                                    ticker, err = find_liquid_market(
                                        settings,
                                        max_checks=int(scan_limit),
                                        ticker_filters=filter_list,
                                        exclude_ticker_filters=exclude_filter_list,
                                    )
                            else:
                                ticker, err = find_liquid_market(
                                    settings,
                                    max_checks=int(scan_limit),
                                    ticker_filters=filter_list,
                                    exclude_ticker_filters=exclude_filter_list,
                                )
                        if err and not ticker:
                            st.error(f"Failed to auto-select market: {err}")
                            st.stop()
                    except Exception as exc:
                        st.error(f"Failed to auto-select market: {exc}")
                        st.stop()

                if not ticker:
                    st.error("Provide a market ticker or enable auto-select.")
                    st.stop()

                selected_tickers.append(ticker)
                channels = (channel,)
                if channel == "orderbook_delta" and subscribe_both:
                    channels = ("orderbook_delta", "ticker")

                if ("orderbook_delta" in channels or "ticker" in channels) and not ticker:
                    st.error("Provide a market ticker or enable auto-select.")
                    st.stop()

                with st.spinner(f"Capturing live data (attempt {attempts + 1}/{max_attempts})..."):
                    try:
                        captured, stats = _run_async(
                            capture_ws_data(
                                ticker,
                                channels,
                                int(duration_s),
                                int(min_interval_ms),
                                int(max_rows),
                                bool(fallback_to_ticker),
                                bool(rest_fallback),
                                int(rest_poll_ms),
                                int(depth_levels),
                                bool(rest_always),
                                bool(allow_partial),
                            )
                        )
                        if seed_from_rest and ticker:
                            try:
                                yes_ask, no_ask, yes_bids, no_bids, meta = fetch_orderbook_snapshot(
                                    ticker, settings, depth_levels=int(depth_levels)
                                )
                                if yes_ask is not None and no_ask is not None:
                                    seed_row = {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "market_ticker": ticker,
                                        "yes_ask": float(yes_ask),
                                        "no_ask": float(no_ask),
                                        "source": "rest_seed",
                                        "latency_ms": meta.get("latency_ms") if meta else None,
                                        "yes_bids": serialize_bids(yes_bids),
                                        "no_bids": serialize_bids(no_bids),
                                        "partial": False,
                                    }
                                    captured = pd.concat([pd.DataFrame([seed_row]), captured], ignore_index=True)
                                    stats["rest_rows"] = stats.get("rest_rows", 0) + 1
                                    stats["fallback"] = stats.get("fallback") or "rest seed"
                                elif meta and meta.get("error"):
                                    stats["last_error"] = meta.get("error")
                            except Exception as exc:
                                stats["last_error"] = f"REST seed failed: {exc}"
                    except Exception as exc:
                        st.error(f"Capture failed: {exc}")
                        st.stop()

                complete_rows = int(captured[["yes_ask", "no_ask"]].notna().all(axis=1).sum())
                if require_two_sided and complete_rows == 0 and auto_select:
                    ticker = None
                    attempts += 1
                    continue
                break

            if captured.empty:
                if stats and stats.get("parsed_messages", 0) > 0:
                    st.error(
                        "No complete two-sided quotes captured. Enable partial rows, "
                        "increase duration, or choose a more liquid market."
                    )
                else:
                    st.error("No data captured. Try a longer duration or different market.")
                if stats:
                    st.caption("Capture diagnostics")
                    st.json(stats)
            else:
                complete_rows = int(captured[["yes_ask", "no_ask"]].notna().all(axis=1).sum())
                if require_two_sided and complete_rows == 0:
                    st.error(
                        "Captured data has no two-sided quotes. "
                        "Try a different market or increase scan limit."
                    )
                    st.stop()
                if selected_tickers:
                    st.caption(f"Capture attempts: {', '.join(selected_tickers)}")
                st.session_state["captured_df"] = captured
                st.session_state["captured_at"] = datetime.now(timezone.utc).isoformat()
                st.session_state["capture_stats"] = stats
                if save_capture:
                    recordings_dir = Path("kalshi_recordings")
                    recordings_dir.mkdir(exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    label = capture_label.strip().replace(" ", "_")
                    filename = f"{stamp}_{ticker or 'capture'}"
                    if label:
                        filename = f"{filename}_{label}"
                    out_path = recordings_dir / f"{filename}.jsonl"
                    with out_path.open("w", encoding="utf-8") as handle:
                        for row in captured.to_dict(orient="records"):
                            handle.write(json.dumps(row) + "\n")
                    st.success(f"Saved capture: {out_path}")

        st.subheader("Live Arb Watch")
        if "arb_watch_on" not in st.session_state:
            st.session_state["arb_watch_on"] = False
        if "arb_watch_last" not in st.session_state:
            st.session_state["arb_watch_last"] = 0.0
        if "arb_watch_results" not in st.session_state:
            st.session_state["arb_watch_results"] = None
        if "arb_watch_last_scan_at" not in st.session_state:
            st.session_state["arb_watch_last_scan_at"] = None
        if "arb_watch_prev_scan_ts" not in st.session_state:
            st.session_state["arb_watch_prev_scan_ts"] = None
        if "arb_watch_last_scan_seconds" not in st.session_state:
            st.session_state["arb_watch_last_scan_seconds"] = float("nan")

        reset_logs_on_start = st.checkbox(
            "Reset availability/journal logs on Start",
            value=True,
        )
        watch_cols = st.columns(3)
        with watch_cols[0]:
            if st.button("Start arb watch"):
                st.session_state["arb_watch_on"] = True
                st.session_state["arb_watch_last"] = 0.0
                st.session_state["arb_watch_prev_scan_ts"] = None
                if bool(reset_logs_on_start):
                    st.session_state["arb_availability_log"] = []
                    st.session_state["arb_opportunity_journal"] = []
                    st.session_state["arb_availability_loaded_path"] = None
                    st.session_state["arb_opportunity_loaded_path"] = None
                    if bool(persist_arb_logs) and availability_log_path.exists():
                        availability_log_path.unlink(missing_ok=True)
                    if bool(persist_arb_logs) and opportunity_log_path.exists():
                        opportunity_log_path.unlink(missing_ok=True)
                    st.info("Started watch with fresh availability and journal logs.")
        with watch_cols[1]:
            if st.button("Stop arb watch"):
                st.session_state["arb_watch_on"] = False
        with watch_cols[2]:
            watch_interval = st.number_input(
                "Watch interval (s)", min_value=2, value=300, step=1
            )
        watch_cfg_cols = st.columns(4)
        with watch_cfg_cols[0]:
            watch_scan_limit = st.number_input(
                "Watch scan limit",
                min_value=50,
                value=50,
                step=50,
            )
        with watch_cfg_cols[1]:
            watch_confirm_top_n = st.number_input(
                "Watch confirm top N",
                min_value=0,
                value=1,
                step=1,
            )
        with watch_cfg_cols[2]:
            watch_confirm_all = st.checkbox("Watch confirm all (slow)", value=False)
        with watch_cfg_cols[3]:
            watch_adaptive_budget = st.checkbox(
                "Adaptive watch budget", value=True
            )
        watch_min_scan_limit = st.number_input(
            "Adaptive min scan limit",
            min_value=50,
            value=50,
            step=50,
        )
        watch_stage_cols = st.columns(2)
        with watch_stage_cols[0]:
            watch_two_stage_mode = st.checkbox(
                "Two-stage watch (fast scan -> strict confirm on trigger)",
                value=True,
            )
        with watch_stage_cols[1]:
            watch_two_stage_trigger_cents = st.number_input(
                "Two-stage trigger (cents to arb)",
                min_value=0.0,
                value=5.0,
                step=1.0,
            )
        watch_log = st.session_state.get("arb_availability_log", [])
        watch_entries = [
            row
            for row in watch_log
            if isinstance(row, dict) and str(row.get("source")) == "watch"
        ]
        watch_df = pd.DataFrame(watch_entries) if watch_entries else pd.DataFrame()
        last_scan_at = st.session_state.get("arb_watch_last_scan_at")
        if st.session_state["arb_watch_on"]:
            st.caption(
                f"Watch status: running | interval target: {int(watch_interval)}s | "
                f"logged watch scans: {len(watch_entries)} | "
                f"last scan: {last_scan_at or 'n/a'}"
            )
        else:
            st.caption(
                f"Watch status: stopped | logged watch scans: {len(watch_entries)} | "
                f"last scan: {last_scan_at or 'n/a'}"
            )
        if not watch_df.empty:
            observed_gap = pd.to_numeric(
                watch_df.get("watch_gap_s"), errors="coerce"
            ).dropna()
            observed_s = (
                float(observed_gap.median())
                if not observed_gap.empty
                else float("nan")
            )
            missed_total = int(
                pd.to_numeric(
                    watch_df.get("watch_missed_intervals"), errors="coerce"
                )
                .fillna(0)
                .sum()
            )
            st.caption(
                "Cadence audit: "
                + (
                    f"observed median interval {observed_s:.2f}s"
                    if np.isfinite(observed_s)
                    else "observed interval n/a"
                )
                + f", missed intervals {missed_total}."
            )
            last_scan_s = pd.to_numeric(
                watch_df.get("watch_scan_seconds"), errors="coerce"
            ).dropna()
            if not last_scan_s.empty:
                latest_scan_s = float(last_scan_s.iloc[-1])
                if latest_scan_s > float(watch_interval):
                    st.warning(
                        "Latest watch scan exceeded your interval "
                        f"({latest_scan_s:.1f}s > {float(watch_interval):.1f}s). "
                        "Lower watch scan limit or keep adaptive budget ON."
                    )
        watch_include_baskets = st.checkbox("Include event-basket watch", value=False)
        watch_replay_strict = st.checkbox(
            "Replay strict basket hits (+100/+250/+500ms)", value=False
        )
        watch_replay_top_n = st.number_input(
            "Replay strict hits (top N)", min_value=1, value=1, step=1
        )
        if st.session_state["arb_watch_on"] and not auto_refresh_watch:
            st.warning(
                "Watch is running but auto-refresh is OFF; scans only run on manual rerun."
            )
        if st.button("Run watch scan now"):
            st.session_state["arb_watch_last"] = 0.0

        if st.session_state["arb_watch_on"]:
            now = time.time()
            if now - st.session_state["arb_watch_last"] >= float(watch_interval):
                scan_started = time.time()
                prev_scan_ts = st.session_state.get("arb_watch_prev_scan_ts")
                watch_gap_s = (
                    float(scan_started - float(prev_scan_ts))
                    if prev_scan_ts is not None
                    else np.nan
                )
                missed_intervals = 0
                if np.isfinite(watch_gap_s) and float(watch_interval) > 0:
                    missed_intervals = max(
                        0, int(np.floor(watch_gap_s / float(watch_interval))) - 1
                    )
                settings = load_settings() if load_settings is not None else None
                if settings is not None:
                    effective_scan_limit = int(watch_scan_limit)
                    effective_confirm_n = int(min(watch_confirm_top_n, effective_scan_limit))
                    if watch_adaptive_budget:
                        last_scan_s = st.session_state.get("arb_watch_last_scan_seconds", np.nan)
                        try:
                            last_scan_s = float(last_scan_s)
                        except Exception:
                            last_scan_s = float("nan")
                        target_budget_s = max(1.0, float(watch_interval) * 0.8)
                        if np.isfinite(last_scan_s) and last_scan_s > target_budget_s:
                            scale = max(0.1, target_budget_s / max(last_scan_s, 1e-9))
                            effective_scan_limit = max(
                                int(watch_min_scan_limit),
                                int(max(50, np.floor(float(watch_scan_limit) * scale))),
                            )
                            effective_confirm_n = min(
                                effective_confirm_n,
                                max(1, int(np.ceil(effective_scan_limit / 40))),
                            )
                    stage2_threshold = max(0.0, float(watch_two_stage_trigger_cents)) / 100.0
                    single_stage1_confirm_n = (
                        0 if bool(watch_two_stage_mode) else max(0, int(effective_confirm_n))
                    )
                    single_stage1_confirm_all = (
                        False if bool(watch_two_stage_mode) else bool(watch_confirm_all)
                    )
                    results, scan_stats = scan_negative_spread_candidates(
                        settings=settings,
                        max_checks=int(effective_scan_limit),
                        depth_levels=int(depth_levels),
                        fee_rate=float(bt_fee_rate),
                        contracts=int(bt_contracts),
                        safety_margin=float(bt_safety_margin),
                        top_n=5,
                        confirm_top_n=int(single_stage1_confirm_n),
                        confirm_all=bool(single_stage1_confirm_all),
                        confirm_limit=int(confirm_limit),
                        ticker_filters=filter_list,
                        exclude_ticker_filters=exclude_filter_list,
                        require_valid_quotes=bool(liquidity_only),
                        require_explicit_asks=bool(explicit_asks_only),
                        auto_special_taker_fees=bool(auto_special_taker_fees),
                        min_liquidity_dollars=float(min_liquidity),
                        min_volume_24h=float(min_volume),
                    )
                    basket_results = pd.DataFrame()
                    basket_stats = {}
                    basket_scan_limit = int(
                        max(int(watch_min_scan_limit), int(effective_scan_limit))
                    )
                    if watch_include_baskets:
                        basket_results, basket_stats = scan_event_basket_candidates(
                            settings=settings,
                            max_checks=int(basket_scan_limit),
                            fee_rate=float(bt_fee_rate),
                            contracts=int(bt_contracts),
                            safety_margin=float(bt_safety_margin),
                            top_n=5,
                            require_valid_quotes=bool(liquidity_only),
                            require_explicit_asks=bool(explicit_asks_only),
                            auto_special_taker_fees=bool(auto_special_taker_fees),
                            strict_executable=bool(
                                strict_basket_mode and not watch_two_stage_mode
                            ),
                            strict_require_complete_event=bool(strict_basket_complete),
                            strict_filter_only=bool(strict_basket_filter_only),
                            strict_confirm_top_n=max(1, int(strict_basket_confirm_n)),
                            strict_depth_levels=int(strict_basket_depth_levels),
                            ticker_filters=filter_list,
                            exclude_ticker_filters=exclude_filter_list,
                            min_liquidity_dollars=float(min_liquidity),
                            min_volume_24h=float(min_volume),
                        )
                    watch_stage2_triggered = False
                    watch_stage2_reason = "disabled"
                    if bool(watch_two_stage_mode):
                        near_single = False
                        near_basket = False
                        if results is not None and not results.empty and "distance_to_arb" in results.columns:
                            single_dist = pd.to_numeric(
                                results["distance_to_arb"], errors="coerce"
                            ).dropna()
                            near_single = bool((single_dist <= stage2_threshold).any())
                        if (
                            watch_include_baskets
                            and basket_results is not None
                            and not basket_results.empty
                            and "distance_to_arb" in basket_results.columns
                        ):
                            basket_dist = pd.to_numeric(
                                basket_results["distance_to_arb"], errors="coerce"
                            ).dropna()
                            near_basket = bool((basket_dist <= stage2_threshold).any())
                        watch_stage2_triggered = bool(near_single or near_basket)
                        if watch_stage2_triggered:
                            if near_single and near_basket:
                                watch_stage2_reason = "single_and_basket"
                            elif near_single:
                                watch_stage2_reason = "single"
                            else:
                                watch_stage2_reason = "basket"
                            if int(effective_confirm_n) > 0 or bool(watch_confirm_all):
                                results, scan_stats = scan_negative_spread_candidates(
                                    settings=settings,
                                    max_checks=int(effective_scan_limit),
                                    depth_levels=int(depth_levels),
                                    fee_rate=float(bt_fee_rate),
                                    contracts=int(bt_contracts),
                                    safety_margin=float(bt_safety_margin),
                                    top_n=5,
                                    confirm_top_n=max(0, int(effective_confirm_n)),
                                    confirm_all=bool(watch_confirm_all),
                                    confirm_limit=int(confirm_limit),
                                    ticker_filters=filter_list,
                                    exclude_ticker_filters=exclude_filter_list,
                                    require_valid_quotes=bool(liquidity_only),
                                    require_explicit_asks=bool(explicit_asks_only),
                                    auto_special_taker_fees=bool(auto_special_taker_fees),
                                    min_liquidity_dollars=float(min_liquidity),
                                    min_volume_24h=float(min_volume),
                                )
                            if watch_include_baskets and bool(strict_basket_mode):
                                basket_results, basket_stats = scan_event_basket_candidates(
                                    settings=settings,
                                    max_checks=int(basket_scan_limit),
                                    fee_rate=float(bt_fee_rate),
                                    contracts=int(bt_contracts),
                                    safety_margin=float(bt_safety_margin),
                                    top_n=5,
                                    require_valid_quotes=bool(liquidity_only),
                                    require_explicit_asks=bool(explicit_asks_only),
                                    auto_special_taker_fees=bool(auto_special_taker_fees),
                                    strict_executable=True,
                                    strict_require_complete_event=bool(strict_basket_complete),
                                    strict_filter_only=bool(strict_basket_filter_only),
                                    strict_confirm_top_n=max(1, int(strict_basket_confirm_n)),
                                    strict_depth_levels=int(strict_basket_depth_levels),
                                    ticker_filters=filter_list,
                                    exclude_ticker_filters=exclude_filter_list,
                                    min_liquidity_dollars=float(min_liquidity),
                                    min_volume_24h=float(min_volume),
                                )
                        else:
                            watch_stage2_reason = "not_triggered"
                    replay_results = pd.DataFrame()
                    replay_legs = pd.DataFrame()
                    replay_summary: dict = {}
                    if (
                        watch_include_baskets
                        and watch_replay_strict
                        and basket_results is not None
                        and not basket_results.empty
                        and "strict_status" in basket_results.columns
                    ):
                        strict_hits = basket_results[
                            basket_results["strict_status"].astype(str) == "crossed"
                        ].head(int(watch_replay_top_n))
                        replay_frames: list[pd.DataFrame] = []
                        leg_frames: list[pd.DataFrame] = []
                        for _, hit in strict_hits.iterrows():
                            event_key = str(hit.get("event_key") or "")
                            if not event_key:
                                continue
                            rep_df, rep_legs = replay_strict_event_basket(
                                settings=settings,
                                event_key=event_key,
                                contracts=int(bt_contracts),
                                base_fee_rate=float(bt_fee_rate),
                                safety_margin=float(bt_safety_margin),
                                depth_levels=int(strict_basket_depth_levels),
                                auto_special_taker_fees=bool(auto_special_taker_fees),
                                require_complete_event=bool(strict_basket_complete),
                                replay_delays_ms=(0, 100, 250, 500),
                            )
                            if not rep_df.empty:
                                replay_frames.append(rep_df)
                            if not rep_legs.empty:
                                leg_frames.append(rep_legs)
                        if replay_frames:
                            replay_results = pd.concat(
                                replay_frames, ignore_index=True
                            )
                        if leg_frames:
                            replay_legs = pd.concat(leg_frames, ignore_index=True)
                        if not replay_results.empty:
                            delayed = replay_results[
                                replay_results["delay_ms"] > 0
                            ].copy()
                            persisted_events = 0
                            persistence_rate = float("nan")
                            if not delayed.empty:
                                per_event = (
                                    delayed.groupby("event_key")["strict_crossed"]
                                    .apply(lambda x: bool(pd.Series(x).all()))
                                )
                                persisted_events = int(per_event.sum())
                                persistence_rate = float(per_event.mean())
                            replay_summary = {
                                "events_replayed": int(
                                    replay_results["event_key"].nunique()
                                ),
                                "events_persisted": persisted_events,
                                "persistence_rate": persistence_rate,
                            }
                    scan_elapsed = float(time.time() - scan_started)
                    append_availability_log(
                        source="watch",
                        safety_margin=float(bt_safety_margin),
                        single_df=results,
                        basket_df=basket_results,
                        scan_stats=scan_stats,
                        basket_stats=basket_stats,
                        extra_stats={
                            "watch_interval_s": float(watch_interval),
                            "watch_gap_s": watch_gap_s,
                            "watch_missed_intervals": int(missed_intervals),
                            "watch_scan_seconds": scan_elapsed,
                            "watch_replay_events": int(
                                replay_summary.get("events_replayed", 0)
                            ),
                            "watch_replay_persistence_rate": replay_summary.get(
                                "persistence_rate", np.nan
                            ),
                            "watch_effective_scan_limit": int(effective_scan_limit),
                            "watch_effective_confirm_n": int(effective_confirm_n),
                            "watch_budget_adaptive": bool(watch_adaptive_budget),
                            "watch_two_stage_mode": bool(watch_two_stage_mode),
                            "watch_stage2_triggered": bool(watch_stage2_triggered),
                            "watch_stage2_reason": str(watch_stage2_reason),
                            "watch_stage2_trigger_cents": float(
                                watch_two_stage_trigger_cents
                            ),
                        },
                        persist_to_file=bool(persist_arb_logs),
                        persist_path=availability_log_path,
                    )
                    append_opportunity_journal(
                        source="watch",
                        safety_margin=float(bt_safety_margin),
                        near_arb_cents=float(near_arb_cents),
                        single_df=results,
                        basket_df=basket_results,
                        replay_summary=replay_summary,
                        persist_to_file=bool(persist_arb_logs),
                        persist_path=opportunity_log_path,
                    )
                    st.session_state["arb_watch_results"] = {
                        "single_results": results,
                        "single_stats": scan_stats,
                        "basket_results": basket_results,
                        "basket_stats": basket_stats,
                        "replay_results": replay_results,
                        "replay_legs": replay_legs,
                        "replay_summary": replay_summary,
                        "scan_elapsed_s": float(scan_elapsed),
                        "effective_scan_limit": int(effective_scan_limit),
                        "effective_confirm_n": int(effective_confirm_n),
                        "watch_two_stage_mode": bool(watch_two_stage_mode),
                        "watch_stage2_triggered": bool(watch_stage2_triggered),
                        "watch_stage2_reason": str(watch_stage2_reason),
                        "watch_stage2_trigger_cents": float(watch_two_stage_trigger_cents),
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    }
                    st.session_state["arb_watch_last_scan_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    st.session_state["arb_watch_last"] = scan_started
                    st.session_state["arb_watch_prev_scan_ts"] = scan_started
                    st.session_state["arb_watch_last_scan_seconds"] = scan_elapsed
            if st.session_state["arb_watch_results"]:
                payload = st.session_state["arb_watch_results"]
                if isinstance(payload, tuple):
                    results, scan_stats = payload
                    basket_results = pd.DataFrame()
                    basket_stats = {}
                    replay_results = pd.DataFrame()
                    replay_legs = pd.DataFrame()
                    replay_summary = {}
                    watch_two_stage_mode_used = bool(watch_two_stage_mode)
                    watch_stage2_triggered = False
                    watch_stage2_reason = "legacy_payload"
                    watch_stage2_trigger_cents_used = float(watch_two_stage_trigger_cents)
                else:
                    results = payload.get("single_results", pd.DataFrame())
                    scan_stats = payload.get("single_stats", {})
                    basket_results = payload.get("basket_results", pd.DataFrame())
                    basket_stats = payload.get("basket_stats", {})
                    replay_results = payload.get("replay_results", pd.DataFrame())
                    replay_legs = payload.get("replay_legs", pd.DataFrame())
                    replay_summary = payload.get("replay_summary", {})
                    scan_elapsed_s = payload.get("scan_elapsed_s", np.nan)
                    effective_scan_limit = payload.get("effective_scan_limit", np.nan)
                    effective_confirm_n = payload.get("effective_confirm_n", np.nan)
                    watch_two_stage_mode_used = bool(
                        payload.get("watch_two_stage_mode", False)
                    )
                    watch_stage2_triggered = bool(
                        payload.get("watch_stage2_triggered", False)
                    )
                    watch_stage2_reason = str(
                        payload.get("watch_stage2_reason", "n/a")
                    )
                    _raw_stage2_trigger_cents = payload.get(
                        "watch_stage2_trigger_cents",
                        watch_two_stage_trigger_cents,
                    )
                    try:
                        watch_stage2_trigger_cents_used = float(
                            _raw_stage2_trigger_cents
                        )
                    except Exception:
                        watch_stage2_trigger_cents_used = float("nan")

                if "scan_elapsed_s" not in locals():
                    scan_elapsed_s = np.nan
                if "effective_scan_limit" not in locals():
                    effective_scan_limit = np.nan
                if "effective_confirm_n" not in locals():
                    effective_confirm_n = np.nan
                if "watch_two_stage_mode_used" not in locals():
                    watch_two_stage_mode_used = False
                if "watch_stage2_triggered" not in locals():
                    watch_stage2_triggered = False
                if "watch_stage2_reason" not in locals():
                    watch_stage2_reason = "n/a"
                if "watch_stage2_trigger_cents_used" not in locals():
                    watch_stage2_trigger_cents_used = float("nan")
                st.caption(
                    "Watch runtime: "
                    + (
                        f"{float(scan_elapsed_s):.2f}s, "
                        if pd.notna(scan_elapsed_s)
                        else "n/a, "
                    )
                    + (
                        f"effective scan limit {int(effective_scan_limit)}, "
                        if pd.notna(effective_scan_limit)
                        else "effective scan limit n/a, "
                    )
                    + (
                        f"confirm top N {int(effective_confirm_n)}."
                        if pd.notna(effective_confirm_n)
                        else "confirm top N n/a."
                    )
                )
                st.caption(
                    "Watch mode: "
                    + ("two-stage ON, " if bool(watch_two_stage_mode_used) else "two-stage OFF, ")
                    + (
                        f"trigger {float(watch_stage2_trigger_cents_used):.1f}c, "
                        if np.isfinite(float(watch_stage2_trigger_cents_used))
                        else "trigger n/a, "
                    )
                    + (
                        f"stage2 executed ({watch_stage2_reason})."
                        if bool(watch_stage2_triggered)
                        else f"stage2 skipped ({watch_stage2_reason})."
                    )
                )

                st.caption("Watch summary: single-market")
                single_cols = st.columns(4)
                single_cols[0].metric(
                    "Single Candidates", f"{0 if results is None else int(results.shape[0])}"
                )
                single_cols[1].metric(
                    "Single Crossed",
                    f"{_count_truthy(results, 'crossed')}",
                )
                best_watch_single_cost = _best_numeric(results, "cost_now")
                single_cols[2].metric(
                    "Best Single Cost",
                    f"{best_watch_single_cost:.4f}"
                    if np.isfinite(best_watch_single_cost)
                    else "n/a",
                )
                best_watch_single_dist = _best_numeric(results, "distance_to_arb")
                single_cols[3].metric(
                    "Best Single Distance",
                    f"{best_watch_single_dist:.4f}"
                    if np.isfinite(best_watch_single_dist)
                    else "n/a",
                )

                if watch_include_baskets:
                    st.caption("Watch summary: event basket")
                    basket_cols = st.columns(4)
                    basket_cols[0].metric(
                        "Basket Candidates",
                        f"{0 if basket_results is None else int(basket_results.shape[0])}",
                    )
                    if (
                        basket_results is not None
                        and not basket_results.empty
                        and "strict_status" in basket_results.columns
                    ):
                        strict_status = basket_results["strict_status"].fillna("").astype(str)
                        strict_ready = int(strict_status.isin(["crossed", "no_cross"]).sum())
                        strict_crossed = int((strict_status == "crossed").sum())
                    else:
                        strict_ready = 0
                        strict_crossed = 0
                    basket_cols[1].metric("Strict Ready", f"{strict_ready}")
                    basket_cols[2].metric("Strict Crossed", f"{strict_crossed}")
                    best_watch_strict_ev = _max_numeric(
                        basket_results, "strict_top_ev_total"
                    )
                    basket_cols[3].metric(
                        "Best Strict EV (top)",
                        f"{best_watch_strict_ev:.2f}"
                        if np.isfinite(best_watch_strict_ev)
                        else "n/a",
                    )

                if results is None or results.empty:
                    st.warning("No arb candidates found in latest watch scan.")
                    st.caption("Watch diagnostics")
                    st.json(scan_stats)
                else:
                    if scan_stats.get("only_invalid_quotes"):
                        st.warning("Watch scan only saw invalid 0/1 quotes.")
                    st.success("Latest watch scan results")
                    st.dataframe(results, use_container_width=True)
                    if results is not None and "distance_to_arb" in results.columns:
                        near_df = results.copy()
                        near_df["distance_to_arb"] = pd.to_numeric(
                            near_df["distance_to_arb"], errors="coerce"
                        )
                        near_df = near_df.sort_values("distance_to_arb")
                        st.caption("Nearest to arb (watch)")
                        st.dataframe(
                            near_df.head(int(near_arb_limit)), use_container_width=True
                        )
                        near_threshold = float(near_arb_cents) / 100.0
                        within = int((near_df["distance_to_arb"] <= near_threshold).sum())
                        if within > 0:
                            st.success(
                                f"Near-arb detected in watch: {within} rows within {near_threshold:.2f}."
                            )
                if watch_include_baskets:
                    st.caption("Event basket watch")
                    if basket_results is None or basket_results.empty:
                        st.info("No event-basket arb candidates in latest watch scan.")
                        if basket_stats:
                            st.caption("Basket watch diagnostics")
                            st.json(basket_stats)
                    else:
                        st.dataframe(basket_results, use_container_width=True)
                        if "distance_to_arb" in basket_results.columns:
                            near_basket_df = basket_results.copy()
                            near_basket_df["distance_to_arb"] = pd.to_numeric(
                                near_basket_df["distance_to_arb"], errors="coerce"
                            )
                            near_basket_df = near_basket_df.sort_values("distance_to_arb")
                            near_threshold = float(near_arb_cents) / 100.0
                            within = int(
                                (
                                    near_basket_df["distance_to_arb"] <= near_threshold
                                ).sum()
                            )
                            if within > 0:
                                st.success(
                                    f"Near-arb basket detected in watch: {within} rows within {near_threshold:.2f}."
                                )
                        if bool(strict_basket_mode) and "strict_status" in basket_results.columns:
                            strict_ready = int(
                                basket_results["strict_status"].isin(["crossed", "no_cross"]).sum()
                            )
                            strict_crossed = int(
                                basket_results["strict_status"].isin(["crossed"]).sum()
                            )
                            st.caption(
                                f"Strict basket watch: ready={strict_ready}, crossed={strict_crossed}."
                            )
                    if replay_results is not None and not replay_results.empty:
                        st.caption("Strict basket replay survivability")
                        st.dataframe(replay_results, use_container_width=True)
                        replay_cols = st.columns(3)
                        replay_cols[0].metric(
                            "Events Replayed",
                            f"{int(replay_summary.get('events_replayed', 0))}",
                        )
                        replay_cols[1].metric(
                            "Events Persisted",
                            f"{int(replay_summary.get('events_persisted', 0))}",
                        )
                        replay_rate = replay_summary.get("persistence_rate", np.nan)
                        replay_cols[2].metric(
                            "Persistence Rate",
                            f"{float(replay_rate):.2%}"
                            if pd.notna(replay_rate)
                            else "n/a",
                        )
                        with st.expander("Replay leg snapshots"):
                            st.dataframe(replay_legs, use_container_width=True)
            if auto_refresh_watch:
                if callable(st_autorefresh):
                    st_autorefresh(
                        interval=max(500, int(float(watch_interval) * 1000)),
                        key="arb_watch_autorefresh",
                    )
                else:
                    elapsed = time.time() - float(st.session_state.get("arb_watch_last", 0.0))
                    sleep_s = max(0.25, float(watch_interval) - elapsed)
                    time.sleep(sleep_s)
                    safe_rerun()

        render_availability_panel(
            safety_margin=float(bt_safety_margin),
            near_arb_cents=float(near_arb_cents),
            default_window=200,
        )
        render_opportunity_journal_panel()

        if "captured_df" in st.session_state:
            df = st.session_state["captured_df"]
            captured_at = st.session_state.get("captured_at")
            st.caption(f"Using captured data{f' from {captured_at}' if captured_at else ''}.")
            if "capture_stats" in st.session_state:
                capture_stats = st.session_state["capture_stats"]
                complete_mask = df[["yes_ask", "no_ask"]].notna().all(axis=1)
                complete_rows = int(complete_mask.sum())
                total_rows = int(len(df))
                complete_ratio = (complete_rows / total_rows) if total_rows else 0.0
                depth_available_rows = 0
                if "yes_bids" in df.columns and "no_bids" in df.columns:
                    yes_depth = df["yes_bids"].apply(_parse_bids_value).apply(len) > 0
                    no_depth = df["no_bids"].apply(_parse_bids_value).apply(len) > 0
                    depth_available_rows = int((yes_depth & no_depth).sum())
                depth_ratio = (depth_available_rows / total_rows) if total_rows else 0.0
                st.caption("Capture quality")
                quality_cols = st.columns(4)
                quality_cols[0].metric("Two-sided rows", f"{complete_rows}/{total_rows}")
                quality_cols[1].metric("Two-sided ratio", f"{complete_ratio:.1%}")
                quality_cols[2].metric("Depth rows", f"{depth_available_rows}/{total_rows}")
                quality_cols[3].metric("Depth ratio", f"{depth_ratio:.1%}")
                if complete_ratio < 0.6:
                    st.warning(
                        "Capture quality is weak: less than 60% of rows are two-sided. "
                        "Increase duration, scan limit, or switch market."
                    )
                if depth_ratio < 0.4:
                    st.warning(
                        "Depth coverage is low: depth-aware backtest will heavily fallback to top-of-book."
                    )
                ws_updates = int(capture_stats.get("ws_book_updates", 0) or 0)
                ws_rows = int(capture_stats.get("ws_book_rows", 0) or 0)
                ws_missing = int(capture_stats.get("ws_book_missing_fields", 0) or 0)
                if ws_updates > 0:
                    st.caption(
                        "Local WS book: "
                        f"{ws_rows} two-sided rows from {ws_updates} updates "
                        f"(missing fields: {ws_missing})."
                    )
                with st.expander("Capture diagnostics"):
                    st.json(capture_stats)
        else:
            st.info(
                "Backtest results are hidden until data is captured. "
                "Click 'Capture live data', or switch the data source above to "
                "'Replay recording', 'Upload CSV', or 'Sample data' to see EV immediately."
            )
            st.stop()

    elif data_source == "Replay recording":
        recordings_dir = Path("kalshi_recordings")
        recordings_dir.mkdir(exist_ok=True)
        files = sorted(recordings_dir.glob("*.jsonl"))
        if not files:
            st.info("No recordings found. Capture live data to create one.")
            st.stop()
        selected = st.selectbox("Recording file", [f.name for f in files])
        selected_path = recordings_dir / selected
        rows = []
        with selected_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if not rows:
            st.error("Recording is empty or invalid.")
            st.stop()
        df = pd.DataFrame(rows)
        st.caption(f"Loaded recording: {selected_path}")

    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("Upload a CSV to begin.")
            st.stop()
    else:
        sample_csv = io.StringIO(
            "timestamp,yes_bid,yes_ask,no_bid,no_ask\n"
            "2024-01-01T00:00:00Z,0.48,0.49,0.50,0.51\n"
            "2024-01-01T00:01:00Z,0.49,0.50,0.49,0.50\n"
            "2024-01-01T00:02:00Z,0.47,0.48,0.51,0.52\n"
        )
        df = pd.read_csv(sample_csv)

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    def _col_index(options, name: str) -> int:
        return options.index(name) if name in options else 0

    columns = ["(none)"] + list(df.columns)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        ts_col = st.selectbox("Timestamp column", columns, index=_col_index(columns, "timestamp"))
    with col2:
        yes_ask_col = st.selectbox("YES ask column", columns, index=_col_index(columns, "yes_ask"))
    with col3:
        no_ask_col = st.selectbox("NO ask column", columns, index=_col_index(columns, "no_ask"))
    with col4:
        yes_bid_col = st.selectbox("YES bid column", columns, index=_col_index(columns, "yes_bid"))
    with col5:
        no_bid_col = st.selectbox("NO bid column", columns, index=_col_index(columns, "no_bid"))

    depth_cols = st.columns(2)
    with depth_cols[0]:
        yes_bids_col = st.selectbox(
            "YES bids depth column",
            columns,
            index=_col_index(columns, "yes_bids"),
        )
    with depth_cols[1]:
        no_bids_col = st.selectbox(
            "NO bids depth column",
            columns,
            index=_col_index(columns, "no_bids"),
        )

    depth_filtered = 0
    depth_rows = 0
    depth_fallback_rows = 0
    depth_fallback_mask = pd.Series(False, index=df.index)
    price_filtered = 0

    yes_ask_top = build_ask_series(
        df,
        None if yes_ask_col == "(none)" else yes_ask_col,
        None if no_bid_col == "(none)" else no_bid_col,
    )
    no_ask_top = build_ask_series(
        df,
        None if no_ask_col == "(none)" else no_ask_col,
        None if yes_bid_col == "(none)" else yes_bid_col,
    )
    if yes_ask_top.empty:
        yes_ask_top = pd.Series(index=df.index, dtype=float)
    if no_ask_top.empty:
        no_ask_top = pd.Series(index=df.index, dtype=float)

    if (
        use_depth
        and yes_bids_col != "(none)"
        and no_bids_col != "(none)"
        and yes_bids_col in df.columns
        and no_bids_col in df.columns
    ):
        yes_bids_series = df[yes_bids_col].apply(_parse_bids_value)
        no_bids_series = df[no_bids_col].apply(_parse_bids_value)
        depth_rows = int(
            ((yes_bids_series.apply(len) > 0) & (no_bids_series.apply(len) > 0)).sum()
        )
        if yes_bids_series.apply(len).sum() == 0 or no_bids_series.apply(len).sum() == 0:
            st.warning(
                "Depth-aware pricing is enabled, but no depth data was found. "
                "Re-capture with REST fallback (depth) or switch Depth mode to Off."
            )

        def _ask_and_fill_from_bids(bids):
            price, filled = effective_ask_from_bids(bids, int(bt_contracts))
            return (float(price) if price is not None else np.nan, filled)

        yes_depth = no_bids_series.apply(_ask_and_fill_from_bids)
        no_depth = yes_bids_series.apply(_ask_and_fill_from_bids)
        yes_ask = yes_depth.apply(lambda x: x[0])
        no_ask = no_depth.apply(lambda x: x[0])
        yes_fill = yes_depth.apply(lambda x: x[1])
        no_fill = no_depth.apply(lambda x: x[1])
        depth_ok = (yes_fill >= float(bt_contracts)) & (no_fill >= float(bt_contracts))
        depth_filtered = int((~depth_ok).sum())
        yes_ask = yes_ask.where(depth_ok)
        no_ask = no_ask.where(depth_ok)

        if depth_mode == "Fallback to top-of-book":
            fallback_mask = yes_ask.isna() | no_ask.isna()
            if fallback_mask.any():
                yes_ask = yes_ask.where(~fallback_mask, yes_ask_top)
                no_ask = no_ask.where(~fallback_mask, no_ask_top)
                depth_fallback_rows = int(fallback_mask.sum())
                depth_fallback_mask = fallback_mask
    else:
        yes_ask = yes_ask_top
        no_ask = no_ask_top

    if yes_ask.empty or no_ask.empty:
        st.error("Provide YES/NO asks, or provide the opposite-side bids to derive asks.")
        st.stop()

    missing_mask = yes_ask.isna() | no_ask.isna()
    valid_mask = (
        (yes_ask >= float(bt_min_price))
        & (yes_ask <= float(bt_max_price))
        & (no_ask >= float(bt_min_price))
        & (no_ask <= float(bt_max_price))
    )
    price_filtered = int(((~valid_mask) & ~missing_mask).sum())
    missing_rows = int(missing_mask.sum())
    yes_ask = yes_ask.where(valid_mask)
    no_ask = no_ask.where(valid_mask)

    total_delay_ms = int(bt_decision_delay_ms + bt_execution_delay_ms + bt_extra_latency_ms)
    use_time_delay = False
    delay_rows = 0
    if ts_col != "(none)":
        use_time_delay = st.sidebar.checkbox("Use time-based delay", value=True)
    if not use_time_delay:
        delay_rows = st.sidebar.number_input(
            "Row delay (snapshots)", min_value=0, value=1 if total_delay_ms > 0 else 0, step=1
        )

    fee_rate_series = pd.Series(float(bt_fee_rate), index=df.index, dtype=float)
    use_auto_fee_series = (
        bool(auto_special_taker_fees)
        and fee_mode == "Taker"
        and "market_ticker" in df.columns
    )
    if use_auto_fee_series:
        fee_rate_series = taker_rate_series_for_tickers(
            df["market_ticker"].astype(str),
            float(bt_fee_rate),
            enable_special_schedule=True,
        )

    fee_yes_now = fee_series_variable_rate(yes_ask, int(bt_contracts), fee_rate_series)
    fee_no_now = fee_series_variable_rate(no_ask, int(bt_contracts), fee_rate_series)
    cost_now = yes_ask + no_ask + fee_yes_now + fee_no_now
    signal = cost_now < float(bt_safety_margin)
    if signal_cooldown_ms > 0:
        signal = apply_signal_cooldown(signal, ts_col, df, int(signal_cooldown_ms))

    latency_series = (
        pd.to_numeric(df.get("latency_ms"), errors="coerce") if "latency_ms" in df.columns else None
    )
    if use_time_delay and ts_col != "(none)":
        if latency_series is not None:
            delay_series = latency_series.fillna(0).astype(float) + float(total_delay_ms)
            exec_yes, exec_no = apply_time_delay_series(
                df, ts_col, yes_ask, no_ask, delay_series
            )
        else:
            exec_yes, exec_no = apply_time_delay(df, ts_col, yes_ask, no_ask, total_delay_ms)
    else:
        exec_yes, exec_no = apply_row_delay(yes_ask, no_ask, int(delay_rows))

    fee_yes_exec = fee_series_variable_rate(exec_yes, int(bt_contracts), fee_rate_series)
    fee_no_exec = fee_series_variable_rate(exec_no, int(bt_contracts), fee_rate_series)
    cost_exec = exec_yes + exec_no + fee_yes_exec + fee_no_exec
    profit_exec = (1 - cost_exec) * int(bt_contracts)
    executed = signal & cost_exec.notna()
    profit_exec = profit_exec.where(executed, 0)
    slippage = cost_exec - cost_now

    result = pd.DataFrame(
        {
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "fee_yes_now": fee_yes_now,
            "fee_no_now": fee_no_now,
            "cost_now": cost_now,
            "signal": signal,
            "yes_ask_exec": exec_yes,
            "no_ask_exec": exec_no,
            "fee_yes_exec": fee_yes_exec,
            "fee_no_exec": fee_no_exec,
            "cost_exec": cost_exec,
            "slippage": slippage,
            "executed": executed,
            "profit_exec": profit_exec,
        }
    )

    if ts_col != "(none)":
        result.insert(0, "timestamp", df[ts_col])

    trades = result[result["signal"]].copy()

    st.subheader("Summary")
    total_signals = int(trades.shape[0])
    total_executed = int(trades["executed"].sum()) if total_signals else 0
    total_profit = float(trades["profit_exec"].sum()) if total_signals else 0.0
    total_cost = float(trades["cost_exec"].where(trades["executed"], 0).sum())
    roi = (total_profit / total_cost * 100) if total_cost else 0.0
    ev_per_signal = float(trades["profit_exec"].mean()) if total_signals else 0.0
    ev_per_snapshot = float(result["profit_exec"].mean())
    edge_avg = float((1 - trades["cost_exec"]).where(trades["executed"]).mean()) if total_executed else 0.0
    avg_slip = float(trades["slippage"].where(trades["executed"]).mean()) if total_executed else 0.0
    win_rate = (
        float((trades["cost_exec"] < float(bt_safety_margin)).where(trades["executed"]).mean())
        if total_executed
        else 0.0
    )
    fill_rate = (total_executed / total_signals) if total_signals else 0.0

    row1 = st.columns(4)
    row1[0].metric("Signals", f"{total_signals}")
    row1[1].metric("Executed", f"{total_executed}")
    row1[2].metric("Fill Rate", f"{fill_rate:.2%}")
    row1[3].metric("Win Rate", f"{win_rate:.2%}")

    row2 = st.columns(4)
    row2[0].metric("Total EV ($)", f"{total_profit:.2f}")
    row2[1].metric("EV/Signal ($)", f"{ev_per_signal:.4f}")
    row2[2].metric("EV/Snapshot ($)", f"{ev_per_snapshot:.4f}")
    row2[3].metric("ROI (%)", f"{roi:.2f}")
    if use_auto_fee_series:
        unique_rates = sorted(set(fee_rate_series.dropna().round(4).tolist()))
        if len(unique_rates) > 1:
            st.caption(
                "Auto fee schedule active. Applied taker rates in this backtest: "
                + ", ".join(f"{rate:.4f}" for rate in unique_rates)
            )

    row3 = st.columns(5)
    row3[0].metric("Avg Edge", f"{edge_avg:.4f}")
    row3[1].metric("Avg Slippage", f"{avg_slip:.4f}")
    row3[2].metric("Delay (ms)", f"{total_delay_ms}")
    row3[3].metric("Price Filtered", f"{price_filtered}")
    row3[4].metric("Missing Rows", f"{missing_rows}")

    row4 = st.columns(3)
    row4[0].metric("Depth Rows", f"{depth_rows}")
    row4[1].metric("Depth Filtered", f"{depth_filtered}")
    row4[2].metric("Depth Fallback", f"{depth_fallback_rows}")

    row5 = st.columns(3)
    best_cost_now = float(cost_now.min()) if cost_now.notna().any() else float("nan")
    best_cost_exec = float(cost_exec.min()) if cost_exec.notna().any() else float("nan")
    best_gross_now = (
        float((yes_ask + no_ask).min()) if (yes_ask + no_ask).notna().any() else float("nan")
    )
    row5[0].metric("Best Cost Now", f"{best_cost_now:.4f}" if np.isfinite(best_cost_now) else "n/a")
    row5[1].metric("Best Cost Exec", f"{best_cost_exec:.4f}" if np.isfinite(best_cost_exec) else "n/a")
    row5[2].metric("Best Gross Now", f"{best_gross_now:.4f}" if np.isfinite(best_gross_now) else "n/a")

    cost_gross = yes_ask + no_ask
    gross_signal = (cost_gross < 1.0) & cost_gross.notna()
    gross_signals = int(gross_signal.sum())
    fees_blocked = max(gross_signals - total_signals, 0)
    best_distance = (
        float((cost_now - float(bt_safety_margin)).min())
        if cost_now.notna().any()
        else float("nan")
    )
    best_gross_distance = (
        float((cost_gross - float(bt_safety_margin)).min())
        if cost_gross.notna().any()
        else float("nan")
    )

    if total_signals == 0 and np.isfinite(best_cost_now):
        st.warning(
            f"No negative spread found. Best cost now is {best_cost_now:.4f} "
            f"vs safety margin {float(bt_safety_margin):.2f}."
        )
        if np.isfinite(best_gross_now):
            if best_gross_now >= float(bt_safety_margin):
                st.caption(
                    "No pre-fee cross either: even gross YES+NO is above your safety margin."
                )
            elif gross_signals > 0 and fees_blocked > 0:
                st.caption(
                    "Pre-fee crosses exist, but fees remove the edge at current prices."
                )
    elif total_signals == 0 and not np.isfinite(best_cost_now):
        if price_filtered > 0 and missing_rows == 0:
            st.warning(
                "All captured rows were removed by your price filter. "
                "Widen Min/Max price filter to inspect these rows."
            )
        elif missing_rows > 0:
            st.warning(
                "No executable rows after filtering because one or both sides were missing."
            )

    row6 = st.columns(3)
    row6[0].metric("Gross Crosses (<$1)", f"{gross_signals}")
    row6[1].metric("Blocked by Fees", f"{fees_blocked}")
    row6[2].metric(
        "Best Distance to Arb",
        f"{best_distance:.4f}" if np.isfinite(best_distance) else "n/a",
    )
    gross_distance_text = f"{best_gross_distance:.4f}" if np.isfinite(best_gross_distance) else "n/a"
    st.caption(f"Best gross distance to safety margin: {gross_distance_text}")

    with st.expander("Threshold sensitivity"):
        margins = sorted(
            set(
                [
                    round(float(bt_safety_margin), 2),
                    0.99,
                    1.00,
                    1.01,
                    1.02,
                    1.03,
                    1.05,
                ]
            )
        )
        sensitivity_rows = []
        for margin in margins:
            signal_count = int((cost_now < margin).sum()) if cost_now.notna().any() else 0
            best_edge = (
                float((margin - cost_now).max()) if cost_now.notna().any() else float("nan")
            )
            sensitivity_rows.append(
                {
                    "threshold": margin,
                    "signals": signal_count,
                    "best_edge_vs_threshold": best_edge,
                }
            )
        st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True)

    with st.expander("Latency robustness (execution stress)"):
        stress_delays = [0, 50, 100, 250, 500, 1000]
        stress_rows = []
        base_row_delay = int(delay_rows) if not use_time_delay else 0
        base_exec_step_ms = max(1, int(bt_execution_delay_ms) if int(bt_execution_delay_ms) > 0 else 250)
        for extra_delay_ms in stress_delays:
            effective_delay_ms = int(total_delay_ms + extra_delay_ms)
            if use_time_delay and ts_col != "(none)":
                if latency_series is not None:
                    eff_series = latency_series.fillna(0).astype(float) + float(effective_delay_ms)
                    exec_yes_s, exec_no_s = apply_time_delay_series(
                        df, ts_col, yes_ask, no_ask, eff_series
                    )
                else:
                    exec_yes_s, exec_no_s = apply_time_delay(
                        df, ts_col, yes_ask, no_ask, effective_delay_ms
                    )
            else:
                extra_rows = int(np.ceil(float(extra_delay_ms) / float(base_exec_step_ms)))
                exec_yes_s, exec_no_s = apply_row_delay(
                    yes_ask, no_ask, int(base_row_delay + extra_rows)
                )

            fee_yes_s = fee_series_variable_rate(
                exec_yes_s, int(bt_contracts), fee_rate_series
            )
            fee_no_s = fee_series_variable_rate(
                exec_no_s, int(bt_contracts), fee_rate_series
            )
            cost_exec_s = exec_yes_s + exec_no_s + fee_yes_s + fee_no_s
            executed_s = signal & cost_exec_s.notna()
            profit_s = ((1 - cost_exec_s) * int(bt_contracts)).where(executed_s, 0.0)
            total_signals_s = int(signal.sum())
            executed_count_s = int(executed_s.sum())
            fill_rate_s = (
                float(executed_count_s) / float(total_signals_s)
                if total_signals_s > 0
                else 0.0
            )
            total_ev_s = float(profit_s.sum()) if executed_count_s > 0 else 0.0
            ev_per_signal_s = (
                float(total_ev_s) / float(total_signals_s)
                if total_signals_s > 0
                else 0.0
            )
            best_cost_exec_s = (
                float(cost_exec_s.min()) if cost_exec_s.notna().any() else float("nan")
            )
            stress_rows.append(
                {
                    "extra_delay_ms": int(extra_delay_ms),
                    "total_delay_ms": int(effective_delay_ms),
                    "signals": int(total_signals_s),
                    "executed": int(executed_count_s),
                    "fill_rate": float(fill_rate_s),
                    "total_ev": float(total_ev_s),
                    "ev_per_signal": float(ev_per_signal_s),
                    "best_cost_exec": (
                        float(best_cost_exec_s) if np.isfinite(best_cost_exec_s) else np.nan
                    ),
                }
            )
        stress_df = pd.DataFrame(stress_rows)
        st.dataframe(stress_df, use_container_width=True)
        if not stress_df.empty:
            worst_ev = pd.to_numeric(stress_df["total_ev"], errors="coerce").min()
            best_ev = pd.to_numeric(stress_df["total_ev"], errors="coerce").max()
            st.caption(
                "Latency stress EV range: "
                f"{worst_ev:.2f} to {best_ev:.2f} across +0ms to +1000ms."
            )

    with st.expander("Filter Diagnostics"):
        diag = pd.DataFrame(
            {
                "timestamp": df[ts_col] if ts_col != "(none)" else pd.Series(dtype=object),
                "yes_bids_len": yes_bids_series.apply(len) if "yes_bids_series" in locals() else None,
                "no_bids_len": no_bids_series.apply(len) if "no_bids_series" in locals() else None,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "depth_ok": depth_ok if "depth_ok" in locals() else None,
                "depth_fallback": depth_fallback_mask if "depth_fallback_mask" in locals() else None,
                "price_ok": valid_mask,
                "missing": missing_mask,
            }
        )
        st.dataframe(diag.head(20), use_container_width=True)

    st.subheader("Signal Trades")
    st.dataframe(trades, use_container_width=True)

    st.subheader("Cost Over Time (Now vs. Execution)")
    st.line_chart(result[["cost_now", "cost_exec"]])
