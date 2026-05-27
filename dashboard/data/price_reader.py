"""
Fetch recent OHLC price bars for the dashboard crypto chart.

Uses yfinance directly (same underlying source as signal_testing).
Returns an empty DataFrame on any error so the chart degrades gracefully.
"""
from __future__ import annotations

import pandas as pd


def fetch_crypto_ohlc(
    ticker: str = "BTC",
    interval: str = "5m",
    period: str = "2d",
) -> pd.DataFrame:
    """
    Fetch recent intraday bars for a crypto ticker.

    Parameters
    ----------
    ticker   : IBKR-style symbol, e.g. "BTC" or "ETH"
    interval : Bar size — "1m", "5m", "15m", "1h"
    period   : Lookback — "1d", "2d", "5d" (yfinance limits 5m to 60d)

    Returns
    -------
    DataFrame with tz-aware DatetimeIndex and columns:
        open, high, low, close, volume
    Empty DataFrame on failure.
    """
    try:
        import yfinance as yf

        symbol = f"{ticker}-USD"
        raw    = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if raw.empty:
            return pd.DataFrame()

        # yfinance ≥0.2 returns MultiIndex columns — flatten them
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower() for c in raw.columns]
        return raw[["open", "high", "low", "close", "volume"]].dropna()

    except Exception:
        return pd.DataFrame()
