"""
Fetch and cache SPY daily data for the benchmark comparison chart.
Uses yahooquery (already a dependency of the main codebase).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd

_CACHE: dict = {}   # {"data": pd.Series, "fetched_at": float}
_CACHE_TTL  = 3600  # 1 hour


def fetch_spy(period: str = "6mo") -> pd.Series:
    """
    Return SPY daily close prices as a pd.Series with DatetimeIndex.
    Cached for 1 hour. Returns empty Series on failure.
    """
    now = time.monotonic()
    if _CACHE.get("data") is not None:
        if now - _CACHE["fetched_at"] < _CACHE_TTL:
            return _CACHE["data"]

    try:
        from yahooquery import Ticker
        tk  = Ticker("SPY", asynchronous=False)
        raw = tk.history(period=period, interval="1d")

        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            return pd.Series(dtype=float)

        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs("SPY", level=0)

        raw.index = pd.to_datetime(raw.index)
        raw.columns = [c.lower() for c in raw.columns]
        series = raw["close"].sort_index().dropna()

        _CACHE["data"]       = series
        _CACHE["fetched_at"] = now
        return series

    except Exception:
        return pd.Series(dtype=float)


def normalize_to_100(series: pd.Series, align_date: datetime | None) -> pd.Series:
    """
    Normalize series so its value at (or just after) align_date equals 100.
    If align_date is None, anchors to the first data point.
    Returns empty Series if normalization fails.
    """
    if series.empty:
        return pd.Series(dtype=float)
    try:
        # Find value at or just after align_date; fall back to series start
        if align_date is None:
            after = series
        else:
            anchor_date = pd.Timestamp(align_date)
            after = series[series.index >= anchor_date]
        if after.empty:
            after = series
        base = after.iloc[0]
        if base == 0:
            return pd.Series(dtype=float)
        return (series / base * 100.0).round(4)
    except Exception:
        return pd.Series(dtype=float)


def get_benchmark_series(align_date: datetime | None, period: str = "6mo") -> pd.Series:
    """
    Public API: SPY normalized to 100.0 at align_date.
    Slices from align_date - 1 day forward so there's one point before.
    """
    spy = fetch_spy(period=period)
    if spy.empty:
        return pd.Series(dtype=float)

    normed = normalize_to_100(spy, align_date)
    if normed.empty:
        return normed

    if align_date is None:
        return normed

    cutoff = pd.Timestamp(align_date) - pd.Timedelta(days=1)
    return normed[normed.index >= cutoff]
