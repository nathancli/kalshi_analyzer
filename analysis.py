"""Settlement-accuracy backtest: was the favored side right X days before settlement?"""
from datetime import datetime, timedelta

import pandas as pd

from kalshi_client import KalshiClient

CANDLE_PERIOD_INTERVAL = 1440  # daily


def _parse_ts(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def _yes_price_near(client: KalshiClient, series_ticker: str, ticker: str, target_ts: int) -> float | None:
    start_ts = target_ts - 2 * 86400
    end_ts = target_ts + 2 * 86400
    candles = client.get_candlesticks(series_ticker, ticker, start_ts, end_ts, CANDLE_PERIOD_INTERVAL)
    if not candles:
        return None
    closest = min(candles, key=lambda c: abs(c["end_period_ts"] - target_ts))
    price = closest.get("price") or {}
    for field in ("mean_dollars", "close_dollars", "open_dollars"):
        if price.get(field) is not None:
            return float(price[field])
    return None


def compute_accuracy(client: KalshiClient, trades_df: pd.DataFrame, days_before: int) -> tuple[dict, pd.DataFrame]:
    """For each settled market in trades_df, compare the favored side `days_before` settlement
    against the actual result. Returns (summary, detail_df).
    """
    markets = trades_df.drop_duplicates("ticker")
    settled = markets[markets["market_result"].isin(["yes", "no"])]

    rows = []
    skipped_no_settlement_time = 0
    skipped_insufficient_history = 0
    skipped_no_price_data = 0

    for _, m in settled.iterrows():
        settlement_ref = m["settlement_ts"] or m["close_time"]
        if not settlement_ref:
            skipped_no_settlement_time += 1
            continue
        settlement_dt = _parse_ts(settlement_ref)
        target_dt = settlement_dt - timedelta(days=days_before)
        if m["open_time"] and target_dt < _parse_ts(m["open_time"]):
            skipped_insufficient_history += 1
            continue
        target_ts = int(target_dt.timestamp())

        price = _yes_price_near(client, m["series_ticker"], m["ticker"], target_ts)
        if price is None:
            skipped_no_price_data += 1
            continue

        predicted = "yes" if price > 0.5 else "no"
        rows.append({
            "ticker": m["ticker"],
            "market_title": m["market_title"],
            "result": m["market_result"],
            "yes_price_x_days_before": price,
            "predicted_side": predicted,
            "correct": predicted == m["market_result"],
        })

    detail_df = pd.DataFrame(rows)
    n = len(detail_df)
    summary = {
        "evaluated": n,
        "correct": int(detail_df["correct"].sum()) if n else 0,
        "accuracy": float(detail_df["correct"].mean()) if n else None,
        "skipped_no_settlement_time": skipped_no_settlement_time,
        "skipped_insufficient_history": skipped_insufficient_history,
        "skipped_no_price_data": skipped_no_price_data,
    }
    return summary, detail_df
