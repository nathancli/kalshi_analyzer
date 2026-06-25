"""Kalshi historical trade analyzer."""
from datetime import datetime, time, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import compute_accuracy
from data import fetch_trades_for_markets, find_candidate_markets, list_categories
from kalshi_client import KalshiClient

DISPLAY_CAP = 5000

st.set_page_config(page_title="Kalshi Trade Analyzer", layout="wide")


@st.cache_resource
def get_client():
    return KalshiClient()


@st.cache_data(ttl=3600)
def cached_categories(_client):
    return list_categories(_client)


@st.cache_data(ttl=600)
def cached_query(_client, category, min_volume, max_volume, max_markets, start_ts, end_ts):
    markets, truncated = find_candidate_markets(_client, category, min_volume, max_volume, max_markets, start_ts)
    df = fetch_trades_for_markets(_client, markets, start_ts, end_ts)
    return df, truncated


def _candle_period_for_duration(open_iso: str, close_iso: str) -> int:
    open_dt = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    close_dt = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
    duration_days = (close_dt - open_dt).total_seconds() / 86400
    if duration_days <= 2:
        return 1
    if duration_days <= 30:
        return 60
    return 1440


def render_odds_chart(client, ticker, series_ticker, open_iso, close_iso, market_trades):
    period_interval = _candle_period_for_duration(open_iso, close_iso)
    start_ts = int(datetime.fromisoformat(open_iso.replace("Z", "+00:00")).timestamp())
    end_ts = int(datetime.fromisoformat(close_iso.replace("Z", "+00:00")).timestamp())
    candles = client.get_candlesticks(series_ticker, ticker, start_ts, end_ts, period_interval)

    times, prices = [], []
    for c in candles:
        price = (c.get("price") or {}).get("mean_dollars")
        if price is None:
            continue
        times.append(datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc))
        prices.append(float(price))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=prices, mode="lines", name="Yes price"))
    trade_times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in market_trades["created_time"]]
    trade_prices = market_trades["yes_price_dollars"].astype(float).tolist()
    fig.add_trace(go.Scatter(x=trade_times, y=trade_prices, mode="markers", name="Selected trade(s)",
                              marker=dict(size=10, color="red", symbol="x")))
    fig.update_layout(title=ticker, yaxis_title="Implied yes probability ($)", yaxis_range=[0, 1])
    return fig


def main():
    st.title("Kalshi Historical Trade Analyzer")
    client = get_client()

    with st.sidebar:
        st.header("Query filters")
        categories = ["All"] + cached_categories(client)
        category = st.selectbox("Category", categories)

        col1, col2 = st.columns(2)
        min_volume = col1.number_input("Min market volume", min_value=0, value=0, step=100)
        max_volume = col2.number_input("Max market volume", min_value=0, value=1_000_000_000, step=100)

        today = datetime.now(timezone.utc).date()
        start_date, end_date = st.date_input("Time frame", value=(today - timedelta(days=30), today))

        max_markets = st.slider("Max markets to scan", min_value=5, max_value=200, value=50, step=5)

        run_query = st.button("Run query", type="primary")

    if run_query:
        start_ts = int(datetime.combine(start_date, time.min, tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc).timestamp())
        category_param = None if category == "All" else category
        with st.spinner("Querying Kalshi..."):
            df, truncated = cached_query(client, category_param, float(min_volume), float(max_volume), max_markets, start_ts, end_ts)
        st.session_state["trades_df"] = df
        st.session_state["truncated"] = truncated

    trades_df = st.session_state.get("trades_df")
    truncated = st.session_state.get("truncated", False)

    tab_query, tab_odds, tab_download, tab_analysis = st.tabs(["Query", "Trade & Odds", "Download", "Analysis"])

    with tab_query:
        if trades_df is None:
            st.info("Set filters in the sidebar and click 'Run query'.")
        elif trades_df.empty:
            st.warning("No trades matched these filters.")
        else:
            if truncated:
                st.warning("Results may be truncated by the market/series scan caps — narrow your filters for a complete picture.")
            st.caption(f"{len(trades_df)} trades across {trades_df['ticker'].nunique()} markets.")
            display_df = trades_df.head(DISPLAY_CAP)
            if len(trades_df) > DISPLAY_CAP:
                st.caption(f"Showing the {DISPLAY_CAP} most recent trades. Download tab includes all {len(trades_df)}.")
            event = st.dataframe(
                display_df,
                on_select="rerun",
                selection_mode="multi-row",
                hide_index=True,
                key="trades_table",
            )
            selected_rows = event["selection"]["rows"]
            st.session_state["selected_trade_ids"] = display_df.iloc[selected_rows]["trade_id"].tolist() if selected_rows else []

    with tab_odds:
        selected_ids = st.session_state.get("selected_trade_ids", [])
        if trades_df is None or not selected_ids:
            st.info("Select one or more trades in the Query tab to view their market's odds history.")
        else:
            selected_df = trades_df[trades_df["trade_id"].isin(selected_ids)]
            for ticker, group in selected_df.groupby("ticker"):
                market = group.iloc[0]
                fig = render_odds_chart(client, ticker, market["series_ticker"], market["open_time"], market["close_time"], group)
                st.plotly_chart(fig, use_container_width=True)

    with tab_download:
        if trades_df is None or trades_df.empty:
            st.info("Run a query first.")
        else:
            csv = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download all queried trades as CSV", csv, file_name="kalshi_trades.csv", mime="text/csv")

    with tab_analysis:
        if trades_df is None or trades_df.empty:
            st.info("Run a query first.")
        else:
            days_before = st.number_input("Days before settlement", min_value=0, value=1, step=1)
            if st.button("Run analysis"):
                with st.spinner("Computing accuracy..."):
                    summary, detail_df = compute_accuracy(client, trades_df, int(days_before))
                if summary["evaluated"] == 0:
                    st.warning("No settled markets with enough history to evaluate at this X.")
                else:
                    st.metric("Accuracy", f"{summary['accuracy']*100:.1f}%", help=f"{summary['correct']} of {summary['evaluated']} markets")
                    st.caption(
                        f"Skipped: {summary['skipped_no_settlement_time']} (no settlement time), "
                        f"{summary['skipped_insufficient_history']} (market didn't exist yet), "
                        f"{summary['skipped_no_price_data']} (no trades that day)"
                    )
                    st.dataframe(detail_df, hide_index=True)


if __name__ == "__main__":
    main()
