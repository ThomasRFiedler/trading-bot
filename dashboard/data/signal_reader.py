"""
Compute real-time per-bar indicator signals for the dashboard crypto chart.

Uses the same indicator logic and deployed params as the live trader so the
chart shows exactly what the trader is seeing on the current bar.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config  # trading-agent config


def _fetch_daily_yf(ticker: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars — needed for the OBV indicator."""
    try:
        import yfinance as yf
        raw = yf.download(
            f"{ticker}-USD",
            period="60d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def compute_bar_signals(
    ticker: str = "BTC",
    interval: str = "5m",
) -> tuple[pd.DataFrame, dict]:
    """
    Compute per-bar signal sum + per-indicator breakdown using the currently
    deployed crypto params.

    Data is fetched with a 5-day lookback so indicators have enough warm-up
    bars; the caller can trim to the visible chart window.

    Returns
    -------
    (signals_df, meta)

    signals_df : DataFrame indexed like price_df with columns:
        signal_sum   — weighted indicator sum at each bar
        ind_<name>   — raw (-1 / 0 / +1) for each indicator
    meta : dict
        params_loaded : bool
        n             : float — entry threshold (±n triggers a trade)
        weights       : list[float]
        ticker        : str
        ind_names     : list[str] — indicator short-names in weight order
    """
    from .price_reader import fetch_crypto_ohlc

    # ── Load deployed crypto params ──────────────────────────────────────────
    params = config.load_deployed_params(crypto=True)
    if not params:
        return pd.DataFrame(), {"params_loaded": False, "ticker": ticker}

    weights = params["weights"]
    n       = float(params["n"])

    # ── Fetch intraday price (5d for indicator warm-up) ───────────────────────
    price_df = fetch_crypto_ohlc(ticker=ticker, interval=interval, period="5d")
    if price_df.empty:
        return pd.DataFrame(), {
            "params_loaded": True, "n": n, "ticker": ticker,
            "weights": weights,
        }

    # ── Fetch daily bars for OBV context ─────────────────────────────────────
    daily_df = _fetch_daily_yf(ticker)

    ctx = {
        "daily_price":  daily_df,
        # Equity-specific context — weights are 0 in crypto params,
        # so these can be empty without affecting the signal sum.
        "vix":          pd.DataFrame(),
        "spx_price":    pd.DataFrame(),
        "spx_daily":    pd.DataFrame(),
        "fundamentals": {},
    }

    # ── Run indicator matrix (trading-app copy for consistency with live trader) ──
    sys.path.insert(0, str(config.TRADING_APP_DIR))
    from app.indicators import INDICATORS, compute_signals_matrix  # noqa: E402

    signals_matrix = compute_signals_matrix(price_df, ctx)      # (n_bars, n_ind)
    weights_arr    = np.array(weights, dtype=np.float32)
    signal_sums    = (signals_matrix.astype(np.float32) * weights_arr).sum(axis=1)

    result = pd.DataFrame({"signal_sum": signal_sums}, index=price_df.index)

    ind_names = []
    for j, (ind_fn, _) in enumerate(INDICATORS):
        name = ind_fn.__name__.replace("indicator_", "")
        ind_names.append(name)
        result[f"ind_{name}"] = signals_matrix[:, j].astype(np.int8)

    return result, {
        "params_loaded": True,
        "n":             n,
        "weights":       weights,
        "ticker":        ticker,
        "ind_names":     ind_names,
    }
