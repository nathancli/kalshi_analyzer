"""Query orchestration: category + volume + timeframe -> a trades DataFrame."""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from kalshi_client import KalshiClient

MAX_SERIES_SCANNED = 200
MAX_EVENT_PAGES_PER_SERIES = 10
# Kalshi's basic tier sustains ~20 read req/s; stay comfortably under that.
MAX_WORKERS = 15


def list_categories(client: KalshiClient) -> list[str]:
    series = client.get_series_list(include_volume=False)
    return sorted({s["category"] for s in series if s.get("category")})


def find_candidate_markets(client: KalshiClient, category: str | None, min_volume: float, max_volume: float, max_markets: int, window_start_ts: int):
    """Resolve category + volume + timeframe filters down to a capped, volume-ranked list of markets.

    Returns (markets, truncated) where each market dict is enriched with series_ticker/category.
    """
    series_list = client.get_series_list(category=category)
    # A series' lifetime volume is the sum of its markets' volumes, so a series below
    # min_volume can't contain any single market that meets the bar. Safe, exact prune.
    series_list = [s for s in series_list if float(s.get("volume_fp") or 0) >= min_volume]
    series_list.sort(key=lambda s: float(s.get("volume_fp") or 0), reverse=True)

    truncated = len(series_list) > MAX_SERIES_SCANNED
    series_list = series_list[:MAX_SERIES_SCANNED]

    def _markets_for_series(series):
        series_ticker = series["ticker"]
        found = []
        # A market that already closed before the window can't have trades inside it,
        # so min_close_ts both narrows results correctly and avoids paginating through
        # a series' entire history for high-frequency series (e.g. 15-minute markets).
        events = client.get_events(series_ticker, min_close_ts=window_start_ts, page_limit=MAX_EVENT_PAGES_PER_SERIES)
        for event in events:
            for market in event.get("markets", []):
                volume = float(market.get("volume_fp") or 0)
                if volume < min_volume or volume > max_volume:
                    continue
                found.append({
                    **market,
                    "series_ticker": series_ticker,
                    "category": series.get("category"),
                })
        return found

    candidates = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for found in pool.map(_markets_for_series, series_list):
            candidates.extend(found)

    candidates.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
    if len(candidates) > max_markets:
        truncated = True
        candidates = candidates[:max_markets]
    return candidates, truncated


def fetch_trades_for_markets(client: KalshiClient, markets: list[dict], min_ts: int, max_ts: int) -> pd.DataFrame:
    """Fetch + merge live and historical trades for each market, enriched with market metadata."""
    market_by_ticker = {m["ticker"]: m for m in markets}

    def _trades_for_ticker(ticker):
        trades = {}
        for t in client.get_trades(ticker, min_ts, max_ts):
            trades[t["trade_id"]] = t
        for t in client.get_historical_trades(ticker, min_ts, max_ts):
            trades[t["trade_id"]] = t
        return ticker, trades.values()

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = pool.map(_trades_for_ticker, market_by_ticker.keys())
        for ticker, trades in results:
            market = market_by_ticker[ticker]
            for t in trades:
                rows.append({
                    "trade_id": t["trade_id"],
                    "ticker": ticker,
                    "series_ticker": market["series_ticker"],
                    "category": market["category"],
                    "market_title": market.get("title"),
                    "created_time": t.get("created_time"),
                    "yes_price_dollars": t.get("yes_price_dollars"),
                    "no_price_dollars": t.get("no_price_dollars"),
                    "count": t.get("count_fp"),
                    "taker_outcome_side": t.get("taker_outcome_side"),
                    "market_volume": market.get("volume_fp"),
                    "market_status": market.get("status"),
                    "market_result": market.get("result"),
                    "settlement_ts": market.get("settlement_ts"),
                    "close_time": market.get("close_time"),
                    "open_time": market.get("open_time"),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("created_time", ascending=False).reset_index(drop=True)
