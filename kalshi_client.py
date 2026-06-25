"""Thin signed HTTP client for the Kalshi trade API (v2)."""
import base64
import os
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("KALSHI_API_BASE", "https://external-api.kalshi.com/trade-api/v2")
API_KEY_ID = os.getenv("API_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", ".key")

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class KalshiClient:
    def __init__(self):
        if not API_KEY_ID:
            raise RuntimeError("API_KEY_ID is not set in .env")
        with open(PRIVATE_KEY_PATH, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self._session.mount("https://", adapter)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        url = API_BASE + path
        max_attempts = 4
        for attempt in range(max_attempts):
            timestamp_ms = str(int(time.time() * 1000))
            headers = {
                "KALSHI-ACCESS-KEY": API_KEY_ID,
                "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
                "KALSHI-ACCESS-SIGNATURE": self._sign(timestamp_ms, method, "/trade-api/v2" + path),
            }
            resp = self._session.request(method, url, headers=headers, params=params, timeout=30)
            if resp.status_code not in _RETRYABLE_STATUSES or attempt == max_attempts - 1:
                resp.raise_for_status()
                return resp.json()
            time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def _paginate(self, path: str, params: dict, items_key: str, page_limit: int | None = None):
        params = dict(params)
        params.setdefault("limit", 1000)
        pages = 0
        while True:
            data = self._request("GET", path, params)
            yield from data.get(items_key, [])
            pages += 1
            cursor = data.get("cursor")
            if not cursor or (page_limit and pages >= page_limit):
                break
            params["cursor"] = cursor

    def get_series_list(self, category: str | None = None, include_volume: bool = True):
        params = {"include_volume": include_volume}
        if category:
            params["category"] = category
        return list(self._paginate("/series", params, "series"))

    def get_events(self, series_ticker: str, min_close_ts: int | None = None, page_limit: int | None = None):
        params = {"series_ticker": series_ticker, "with_nested_markets": True, "limit": 200}
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        return self._paginate("/events", params, "events", page_limit=page_limit)

    def get_trades(self, ticker: str, min_ts: int | None = None, max_ts: int | None = None):
        params = {"ticker": ticker}
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        return list(self._paginate("/markets/trades", params, "trades"))

    def get_historical_trades(self, ticker: str, min_ts: int | None = None, max_ts: int | None = None):
        params = {"ticker": ticker}
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        return list(self._paginate("/historical/trades", params, "trades"))

    def get_candlesticks(self, series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_interval: int):
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        data = self._request("GET", path, params)
        return data.get("candlesticks", [])
