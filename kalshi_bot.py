import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_UP
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.parse import urlencode

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency for skeleton
    websockets = None


@dataclass(frozen=True)
class Settings:
    api_key: str
    private_key_path: Optional[Path]
    private_key: Optional[str]
    ws_url: str
    rest_url: str
    market_tickers: Tuple[str, ...]
    ws_channels: Tuple[str, ...]
    fee_rate_taker: Decimal
    fee_rate_maker: Decimal
    safety_margin: Decimal
    contracts: int
    state_path: Path


def _env_list(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(parts)


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


def _load_env_files(paths: Iterable[Path]) -> bool:
    loaded_any = False
    for path in paths:
        loaded_any = _load_env_file(path) or loaded_any
    return loaded_any


def _resolve_path_with_fallbacks(path: Optional[Path]) -> Optional[Path]:
    if not path:
        return None
    expanded = path.expanduser()
    candidates: list[Path] = []
    if expanded.is_absolute():
        candidates.append(expanded)
    else:
        candidates.extend(
            [
                Path.cwd() / expanded,
                Path(__file__).resolve().parent / expanded,
                Path.home() / expanded,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0] if candidates else expanded


def _read_private_key(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Private key file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _read_text_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _load_api_key() -> Optional[str]:
    api_key = os.getenv("KALSHI_API_KEY", "").strip()
    if api_key:
        return api_key
    for path in (
        Path.home() / ".kalshi_api_key",
        Path.home() / ".kalshi_api_key_id",
        Path.home() / ".kalshi" / "api_key",
    ):
        found = _read_text_file(path)
        if found:
            return found
    return None


def load_settings() -> Settings:
    _load_env_files(
        [
            Path.cwd() / ".env",
            Path(__file__).with_name(".env"),
            Path.home() / ".env",
        ]
    )
    private_key_path = (
        _resolve_path_with_fallbacks(Path(os.getenv("KALSHI_PRIVATE_KEY_PATH", "")))
        if os.getenv("KALSHI_PRIVATE_KEY_PATH")
        else None
    )
    if not private_key_path:
        for fallback in (
            Path(__file__).with_name("kalshi_private_key.pem"),
            Path.cwd() / "kalshi_private_key.pem",
            Path.home() / ".kalshi_private_key.pem",
            Path.home() / ".kalshi_private_key.key",
            Path.home() / ".kalshi" / "private_key.pem",
        ):
            resolved = _resolve_path_with_fallbacks(fallback)
            if resolved and resolved.exists():
                private_key_path = resolved
                break
    private_key = os.getenv("KALSHI_PRIVATE_KEY")
    if not private_key:
        private_key = _read_private_key(private_key_path)
    market_tickers = _env_list("KALSHI_MARKET_TICKERS")
    ws_channels_env = _env_list("KALSHI_WS_CHANNELS")
    if ws_channels_env:
        ws_channels = ws_channels_env
    else:
        ws_channels = ("orderbook_delta",) if market_tickers else ("ticker",)

    return Settings(
        api_key=_load_api_key() or "",
        private_key_path=private_key_path,
        private_key=private_key,
        ws_url=os.getenv("KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2"),
        rest_url=os.getenv("KALSHI_REST_URL", "https://api.elections.kalshi.com"),
        market_tickers=market_tickers,
        ws_channels=ws_channels,
        fee_rate_taker=Decimal(os.getenv("KALSHI_FEE_TAKER", "0.07")),
        fee_rate_maker=Decimal(os.getenv("KALSHI_FEE_MAKER", "0.0175")),
        safety_margin=Decimal(os.getenv("KALSHI_SAFETY_MARGIN", "0.99")),
        contracts=int(os.getenv("KALSHI_CONTRACTS", "1")),
        state_path=Path(os.getenv("KALSHI_STATE_PATH", "kalshi_bot_state.json")),
    )


def kalshi_fee(price_dollars: Decimal, contracts: int, rate: Decimal) -> Decimal:
    raw = rate * Decimal(contracts) * price_dollars * (Decimal("1") - price_dollars)
    return (raw * 100).quantize(Decimal("1"), rounding=ROUND_UP) / 100


def total_cost(
    yes_ask: Decimal,
    no_ask: Decimal,
    contracts: int,
    rate: Decimal,
) -> Decimal:
    return (
        yes_ask
        + no_ask
        + kalshi_fee(yes_ask, contracts, rate)
        + kalshi_fee(no_ask, contracts, rate)
    )


class OrderBook:
    def __init__(self) -> None:
        self.market_ticker: Optional[str] = None
        self.best_yes_ask: Optional[Decimal] = None
        self.best_no_ask: Optional[Decimal] = None
        self.source: Optional[str] = None
        self.last_update: Optional[datetime] = None

    def update(
        self,
        market_ticker: Optional[str],
        yes_ask: Optional[Decimal],
        no_ask: Optional[Decimal],
        source: Optional[str],
    ) -> None:
        if market_ticker:
            self.market_ticker = market_ticker
        if yes_ask is not None:
            self.best_yes_ask = yes_ask
        if no_ask is not None:
            self.best_no_ask = no_ask
        if source:
            self.source = source
        self.last_update = datetime.now(timezone.utc)

    def snapshot(self) -> Dict[str, Optional[str]]:
        return {
            "market_ticker": self.market_ticker,
            "yes_ask": str(self.best_yes_ask) if self.best_yes_ask is not None else None,
            "no_ask": str(self.best_no_ask) if self.best_no_ask is not None else None,
            "source": self.source,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


def _coerce_decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if isinstance(value, str):
        return Decimal(value)
    return None


def _coerce_price_cents(value: object) -> Optional[Decimal]:
    dec = _coerce_decimal(value)
    if dec is None:
        return None
    if dec >= 1:
        dec = dec / Decimal("100")
    if dec < 0 or dec > 1:
        return None
    return dec


def _coerce_price_dollars(value: object) -> Optional[Decimal]:
    dec = _coerce_decimal(value)
    if dec is None:
        return None
    if dec > 1:
        dec = dec / Decimal("100")
    if dec < 0 or dec > 1:
        return None
    return dec


def _extract_price(data: Dict[str, object], keys: Iterable[str]) -> Optional[Decimal]:
    for key in keys:
        if key in data:
            if "dollars" in key:
                price = _coerce_price_dollars(data.get(key))
            else:
                price = _coerce_price_cents(data.get(key))
            if price is not None:
                return price
    return None


def _best_bid_from_levels(levels: object) -> Optional[Decimal]:
    if not isinstance(levels, list):
        return None
    best: Optional[Decimal] = None
    for level in levels:
        price: Optional[Decimal] = None
        if isinstance(level, (list, tuple)) and level:
            price = _coerce_price_cents(level[0])
        elif isinstance(level, dict):
            price = _extract_price(level, ("price_dollars", "price", "p"))
        if price is None:
            continue
        if best is None or price > best:
            best = price
    return best


def _derive_no_ask_from_yes_bid(yes_bid: Optional[Decimal]) -> Optional[Decimal]:
    if yes_bid is None:
        return None
    no_ask = Decimal("1") - yes_bid
    if no_ask < 0 or no_ask > 1:
        return None
    return no_ask


def parse_message(
    message: Dict[str, object]
) -> Tuple[Optional[str], Optional[Decimal], Optional[Decimal], Optional[str]]:
    msg_type = message.get("type")
    payload = None
    if isinstance(message.get("msg"), dict):
        payload = message.get("msg")
    elif isinstance(message.get("data"), dict):
        payload = message.get("data")
    elif isinstance(message.get("payload"), dict):
        payload = message.get("payload")
    elif isinstance(message, dict):
        payload = message

    if msg_type in {"ticker", "ticker_v2"} and isinstance(payload, dict):
        market_ticker = payload.get("market_ticker")
        yes_bid = _extract_price(
            payload,
            (
                "yes_bid_dollars",
                "yes_bid",
                "bid_dollars",
                "bid",
                "best_bid_dollars",
                "best_bid",
            ),
        )
        yes_ask = _extract_price(
            payload,
            (
                "yes_ask_dollars",
                "yes_ask",
                "ask_dollars",
                "ask",
                "best_ask_dollars",
                "best_ask",
            ),
        )
        no_bid = _extract_price(payload, ("no_bid_dollars", "no_bid"))
        no_ask = _extract_price(payload, ("no_ask_dollars", "no_ask"))
        if no_ask is None:
            no_ask = _derive_no_ask_from_yes_bid(yes_bid)
        if yes_ask is None and no_bid is not None:
            yes_ask = _derive_no_ask_from_yes_bid(no_bid)
        return (
            str(market_ticker) if market_ticker is not None else None,
            yes_ask,
            no_ask,
            str(msg_type),
        )

    if msg_type in {"orderbook_snapshot", "orderbook_delta", "orderbook_update"} and isinstance(
        payload, dict
    ):
        market_ticker = payload.get("market_ticker")
        orderbook = (
            payload.get("orderbook") if isinstance(payload.get("orderbook"), dict) else payload
        )
        yes_bids = None
        no_bids = None
        if isinstance(orderbook, dict):
            yes_bids = orderbook.get("yes_dollars") or orderbook.get("yes")
            no_bids = orderbook.get("no_dollars") or orderbook.get("no")
        best_yes_bid = _best_bid_from_levels(yes_bids)
        best_no_bid = _best_bid_from_levels(no_bids)
        yes_ask = _derive_no_ask_from_yes_bid(best_no_bid)
        no_ask = _derive_no_ask_from_yes_bid(best_yes_bid)
        return (
            str(market_ticker) if market_ticker is not None else None,
            yes_ask,
            no_ask,
            str(msg_type),
        )

    return None, None, None, None


def _requires_auth(settings: Settings) -> bool:
    return bool(settings.api_key and settings.private_key)


def _ws_signing_payload(ws_url: str, timestamp_ms: str) -> str:
    path = urlparse(ws_url).path or "/trade-api/ws/v2"
    return f"{timestamp_ms}GET{path}"


def _sign_message(private_key_pem: str, message: str) -> str:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "cryptography is required for Kalshi auth. Install it in your env."
        ) from exc

    key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"), password=None
    )
    signature = key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def build_ws_headers(settings: Settings) -> Dict[str, str]:
    if not _requires_auth(settings):
        return {}
    timestamp_ms = str(int(time.time() * 1000))
    payload = _ws_signing_payload(settings.ws_url, timestamp_ms)
    signature = _sign_message(settings.private_key or "", payload)
    return {
        "KALSHI-ACCESS-KEY": settings.api_key,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


def ws_connect_kwargs(headers: Dict[str, str]) -> Dict[str, object]:
    if not headers:
        return {}
    try:
        import inspect

        params = inspect.signature(websockets.connect).parameters
        if "additional_headers" in params:
            return {"additional_headers": headers}
        if "extra_headers" in params:
            return {"extra_headers": headers}
    except Exception:
        pass
    return {"extra_headers": headers}


def _rest_signing_payload(timestamp_ms: str, method: str, path: str) -> str:
    clean_path = path.split("?", 1)[0]
    return f"{timestamp_ms}{method}{clean_path}"


def build_rest_headers(settings: Settings, method: str, path: str) -> Dict[str, str]:
    timestamp_ms = str(int(time.time() * 1000))
    payload = _rest_signing_payload(timestamp_ms, method.upper(), path)
    signature = _sign_message(settings.private_key or "", payload)
    return {
        "KALSHI-ACCESS-KEY": settings.api_key,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


def _rest_get_json(settings: Settings, path: str, params: Optional[Dict[str, str]] = None) -> Dict:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{settings.rest_url}{path}{query}"
    headers = build_rest_headers(settings, "GET", path)
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=15) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def pick_default_market_ticker(settings: Settings) -> Optional[str]:
    status = os.getenv("KALSHI_MARKET_STATUS", "open")
    try:
        response = _rest_get_json(
            settings,
            "/trade-api/v2/markets",
            {"limit": "1", "status": status},
        )
    except Exception:
        try:
            response = _rest_get_json(settings, "/trade-api/v2/markets", {"limit": "1"})
        except Exception:
            return None

    markets = response.get("markets")
    if isinstance(markets, list) and markets:
        market = markets[0]
        if isinstance(market, dict) and market.get("ticker"):
            return str(market["ticker"])
    return None


async def subscribe(
    ws,
    channels: Iterable[str],
    market_tickers: Iterable[str],
    message_id: int = 1,
) -> int:
    params: Dict[str, object] = {"channels": list(channels)}
    market_list = list(market_tickers)
    if market_list:
        if len(market_list) == 1:
            params["market_ticker"] = market_list[0]
        else:
            params["market_tickers"] = market_list
    payload = {"id": message_id, "cmd": "subscribe", "params": params}
    await ws.send(json.dumps(payload))
    return message_id + 1


def should_trade(
    yes_ask: Decimal,
    no_ask: Decimal,
    settings: Settings,
) -> bool:
    cost = total_cost(yes_ask, no_ask, settings.contracts, settings.fee_rate_taker)
    return cost < settings.safety_margin


def write_state(
    settings: Settings, book: OrderBook, note: str, details: Optional[Dict[str, str]] = None
) -> None:
    payload = {
        "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "book": book.snapshot(),
    }
    if details:
        payload["details"] = details
    settings.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def run_listener(settings: Settings) -> None:
    if websockets is None:
        raise RuntimeError("websockets is not installed. pip install websockets")
    if not settings.api_key:
        raise RuntimeError("Set KALSHI_API_KEY (or ~/.kalshi_api_key).")
    if not settings.private_key:
        raise RuntimeError(
            "Set KALSHI_PRIVATE_KEY, KALSHI_PRIVATE_KEY_PATH, or ~/.kalshi_private_key.pem."
        )

    book = OrderBook()
    write_state(settings, book, "starting")

    channels = settings.ws_channels
    market_tickers = settings.market_tickers
    if not market_tickers:
        default_ticker = pick_default_market_ticker(settings)
        if default_ticker:
            market_tickers = (default_ticker,)
            channels = ("orderbook_delta",)
            write_state(
                settings,
                book,
                "auto-selected market ticker",
                {"market_ticker": default_ticker},
            )
        else:
            channels = tuple(ch for ch in channels if ch != "orderbook_delta") or ("ticker",)
            write_state(settings, book, "no tickers set; using ticker channel only")

    headers = build_ws_headers(settings)
    async with websockets.connect(settings.ws_url, **ws_connect_kwargs(headers)) as ws:
        await subscribe(ws, channels, market_tickers)
        async for raw in ws:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("type") == "error":
                write_state(
                    settings,
                    book,
                    "ws error",
                    {"error": str(data.get("msg", data))},
                )
                continue
            market_ticker, yes_ask, no_ask, source = parse_message(data)
            if yes_ask is None and no_ask is None:
                continue
            book.update(market_ticker, yes_ask, no_ask, source)

            if book.best_yes_ask is None or book.best_no_ask is None:
                continue

            if should_trade(book.best_yes_ask, book.best_no_ask, settings):
                cost = total_cost(
                    book.best_yes_ask, book.best_no_ask, settings.contracts, settings.fee_rate_taker
                )
                write_state(
                    settings,
                    book,
                    "trigger: negative spread",
                    {"cost": f"{cost:.4f}", "margin": str(settings.safety_margin)},
                )
                # TODO: execute trade via SDK or REST.
            else:
                write_state(settings, book, "no trigger")


def main() -> None:
    settings = load_settings()
    asyncio.run(run_listener(settings))


if __name__ == "__main__":
    main()
