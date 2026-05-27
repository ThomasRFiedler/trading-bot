"""
Tool: compute_divergence

Compares the live/paper trade distribution against what the currently
deployed params produced in backtest — the key signal for whether your
strategy is behaving as expected in real market conditions.

Metrics compared
----------------
exit_reasons     : TP / SL / EOD rate (pp difference)
win_rate         : % of trades with net_pnl > 0
avg_net_pnl      : mean P/L per trade ($)
avg_hold_bars    : mean bars held per trade
trade_freq_day   : mean trades per trading day
profit_factor    : gross_profit / gross_loss
sharpe_ratio     : annualised trade-return Sharpe

Flags
-----
Each metric is flagged when the absolute divergence exceeds a threshold
calibrated to what's practically significant (not just statistically):
  exit reason rates : > 15 percentage points
  win_rate          : > 12 percentage points
  avg_net_pnl       : > 40% relative
  avg_hold_bars     : > 50% relative
  trade_freq_day    : > 50% relative
  profit_factor     : > 40% relative
  sharpe_ratio      : > 0.8 absolute
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))

MIN_LIVE_TRADES = 5   # don't report divergence with fewer live trades

# Divergence thresholds — absolute for pp metrics, relative for ratio metrics
_THRESHOLDS = {
    "tp_rate":       ("pp",       15.0),
    "sl_rate":       ("pp",       15.0),
    "eod_rate":      ("pp",       15.0),
    "win_rate":      ("pp",       12.0),
    "avg_net_pnl":   ("relative", 40.0),
    "avg_hold_bars": ("relative", 50.0),
    "trade_freq_day":("relative", 50.0),
    "profit_factor": ("relative", 40.0),
    "sharpe_ratio":  ("absolute",  0.8),
}


# ---------------------------------------------------------------------------
# Stats extraction helpers
# ---------------------------------------------------------------------------

def _stats_from_backtest_df(df: pd.DataFrame, interval_min: int = 5) -> dict:
    """Compute comparable stats from a backtest trades_df."""
    if df.empty:
        return {}

    n = len(df)

    tp  = (df["exit_reason"] == "take_profit").sum()
    sl  = (df["exit_reason"] == "stop_loss").sum()
    eod = (df["exit_reason"] == "eod_exit").sum()

    wins    = (df["net_pnl"] > 0).sum()
    g_profit = df.loc[df["net_pnl"] > 0, "net_pnl"].sum()
    g_loss   = abs(df.loc[df["net_pnl"] < 0, "net_pnl"].sum())

    # Holding bars from entry/exit date+time strings
    hold_bars = []
    for _, row in df.iterrows():
        try:
            entry_dt = datetime.strptime(
                f"{row['entry_date']} {row['entry_time']}", "%Y-%m-%d %H:%M"
            )
            exit_dt = datetime.strptime(
                f"{row['exit_date']} {row['exit_time']}", "%Y-%m-%d %H:%M"
            )
            diff_min = (exit_dt - entry_dt).total_seconds() / 60
            hold_bars.append(max(1, diff_min / interval_min))
        except Exception:
            pass

    # Trade frequency: trades per trading day
    try:
        first = datetime.strptime(df["entry_date"].iloc[0],  "%Y-%m-%d")
        last  = datetime.strptime(df["exit_date"].iloc[-1],  "%Y-%m-%d")
        days  = max(1, (last - first).days + 1)
    except Exception:
        days = 1

    pnl_arr = df["net_pnl"].values
    pos_size = float(df["position_size"].iloc[0]) if "position_size" in df.columns else 1.0
    returns  = pnl_arr / pos_size
    sharpe   = (float(np.mean(returns) / np.std(returns)) * np.sqrt(252)
                if np.std(returns) > 0 else 0.0)

    return {
        "total_trades":   n,
        "tp_rate":        tp  / n * 100,
        "sl_rate":        sl  / n * 100,
        "eod_rate":       eod / n * 100,
        "win_rate":       wins / n * 100,
        "avg_net_pnl":    float(np.mean(pnl_arr)),
        "avg_hold_bars":  float(np.mean(hold_bars)) if hold_bars else 0.0,
        "trade_freq_day": n / days,
        "profit_factor":  g_profit / g_loss if g_loss > 0 else float("inf"),
        "sharpe_ratio":   sharpe,
        # Distribution arrays for histograms
        "pnl_dist":       pnl_arr.tolist(),
        "exit_counts":    {"take_profit": int(tp), "stop_loss": int(sl), "eod_exit": int(eod)},
    }


def _stats_from_ledger(trades: list[dict], interval_min: int = 5) -> dict:
    """Compute comparable stats from live/paper ledger records."""
    if not trades:
        return {}

    n = len(trades)

    tp  = sum(1 for t in trades if t["exit_reason"] == "take_profit")
    sl  = sum(1 for t in trades if t["exit_reason"] == "stop_loss")
    eod = sum(1 for t in trades if t["exit_reason"] == "eod_exit")

    pnl_arr = np.array([t["net_pnl"] for t in trades])
    wins    = (pnl_arr > 0).sum()
    g_profit = pnl_arr[pnl_arr > 0].sum() if (pnl_arr > 0).any() else 0.0
    g_loss   = abs(pnl_arr[pnl_arr < 0].sum()) if (pnl_arr < 0).any() else 0.0

    # Holding bars from entry_time / exit_time strings in ledger
    hold_bars = []
    for t in trades:
        try:
            ets = pd.Timestamp(t.get("entry_time", ""))
            xts = pd.Timestamp(t.get("exit_time",  "") or t["timestamp"].isoformat())
            diff_min = (xts - ets).total_seconds() / 60
            hold_bars.append(max(1, diff_min / interval_min))
        except Exception:
            pass

    # Trade frequency
    try:
        timestamps = sorted(t["timestamp"] for t in trades)
        span_days  = max(1, (timestamps[-1] - timestamps[0]).days + 1)
    except Exception:
        span_days = 1

    pos_size = float(trades[0].get("position_size", 1.0)) if trades else 1.0
    returns  = pnl_arr / pos_size
    sharpe   = (float(np.mean(returns) / np.std(returns)) * np.sqrt(252)
                if np.std(returns) > 0 else 0.0)

    return {
        "total_trades":   n,
        "tp_rate":        tp  / n * 100,
        "sl_rate":        sl  / n * 100,
        "eod_rate":       eod / n * 100,
        "win_rate":       wins / n * 100,
        "avg_net_pnl":    float(np.mean(pnl_arr)),
        "avg_hold_bars":  float(np.mean(hold_bars)) if hold_bars else 0.0,
        "trade_freq_day": n / span_days,
        "profit_factor":  g_profit / g_loss if g_loss > 0 else float("inf"),
        "sharpe_ratio":   sharpe,
        "pnl_dist":       pnl_arr.tolist(),
        "exit_counts":    {"take_profit": tp, "stop_loss": sl, "eod_exit": eod},
    }


def _flag(metric: str, bt_val: float, live_val: float) -> tuple[float, bool]:
    """Return (divergence_value, is_flagged) for a metric."""
    kind, threshold = _THRESHOLDS.get(metric, ("relative", 30.0))

    if kind == "pp":
        diff    = live_val - bt_val
        flagged = abs(diff) > threshold
        return diff, flagged

    if kind == "absolute":
        diff    = live_val - bt_val
        flagged = abs(diff) > threshold
        return diff, flagged

    # relative
    if abs(bt_val) < 1e-9:
        return 0.0, False
    diff    = (live_val - bt_val) / abs(bt_val) * 100
    flagged = abs(diff) > threshold
    return diff, flagged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_divergence(crypto: bool = False) -> dict:
    """
    Compare live/paper trade distribution against backtest expectations.

    Returns
    -------
    dict with keys:
        status          : "ok" | "warning" | "alert" | "insufficient_data" | "no_params"
        live_trades     : int
        backtest_trades : int
        metrics         : list of per-metric comparison dicts
        bt_exit_counts  : {"take_profit": n, "stop_loss": n, "eod_exit": n}
        live_exit_counts: same
        bt_pnl_dist     : list of floats (backtest per-trade P/L)
        live_pnl_dist   : list of floats (live per-trade P/L)
        flagged_count   : int
    """
    from signal_testing.backtest import backtest
    from dashboard.data.state_reader import read_trade_history

    # ── Load deployed params ────────────────────────────────────────────
    params = config.load_deployed_params(crypto=crypto)
    if not params:
        return {"status": "no_params", "live_trades": 0, "backtest_trades": 0,
                "metrics": [], "flagged_count": 0}

    ticker = config.CRYPTO_TICKER if crypto else config.TICKER

    # ── Run backtest to get full trades_df ──────────────────────────────
    try:
        bt_result = backtest(
            ticker=ticker,
            n=params["n"],
            time_frame=config.TIME_FRAME,
            interval=config.INTERVAL,
            take_profit=params["take_profit"],
            stop_loss=params["stop_loss"],
            weights=params.get("weights"),
            position_size=config.OPT_POSITION_SIZE,
            verbose=False,
            no_eod_exit=crypto,
            commission_rate=0.0018 if crypto else 0.0,
            commission_min=1.75   if crypto else 0.0,
        )
        bt_df = bt_result["trades_df"]
    except Exception as exc:
        return {"status": "backtest_error", "error": str(exc), "live_trades": 0,
                "backtest_trades": 0, "metrics": [], "flagged_count": 0}

    # ── Load live/paper trades, filter by mode ──────────────────────────
    mode = "crypto" if crypto else "stock"
    all_trades  = read_trade_history()
    live_trades = [
        t for t in all_trades
        if t.get("trading_mode") == mode or
           (t.get("trading_mode") == "" and t.get("account_type") != "unknown")
    ]

    if len(live_trades) < MIN_LIVE_TRADES:
        return {
            "status":          "insufficient_data",
            "live_trades":     len(live_trades),
            "backtest_trades": len(bt_df),
            "min_required":    MIN_LIVE_TRADES,
            "metrics":         [],
            "flagged_count":   0,
            "bt_exit_counts":  _stats_from_backtest_df(bt_df).get("exit_counts", {}),
            "live_exit_counts":{},
            "bt_pnl_dist":     bt_df["net_pnl"].tolist() if not bt_df.empty else [],
            "live_pnl_dist":   [],
        }

    # ── Compute stats ───────────────────────────────────────────────────
    interval_min = int(config.INTERVAL.replace("m", "").replace("h", "")) * (
        60 if "h" in config.INTERVAL else 1
    )
    bt_stats   = _stats_from_backtest_df(bt_df, interval_min)
    live_stats = _stats_from_ledger(live_trades, interval_min)

    # ── Build per-metric comparison ──────────────────────────────────────
    metric_keys = [
        "tp_rate", "sl_rate", "win_rate",
        "avg_net_pnl", "avg_hold_bars", "trade_freq_day",
        "profit_factor", "sharpe_ratio",
    ]
    if not crypto:
        metric_keys.insert(2, "eod_rate")   # EOD exits only meaningful for stocks

    metrics = []
    flagged_count = 0
    for key in metric_keys:
        bt_val   = bt_stats.get(key, 0.0)
        live_val = live_stats.get(key, 0.0)
        diff, flagged = _flag(key, bt_val, live_val)
        if flagged:
            flagged_count += 1
        metrics.append({
            "metric":   key,
            "backtest": bt_val,
            "live":     live_val,
            "diff":     diff,
            "flagged":  flagged,
        })

    if flagged_count == 0:
        status = "ok"
    elif flagged_count <= 2:
        status = "warning"
    else:
        status = "alert"

    return {
        "status":           status,
        "live_trades":      len(live_trades),
        "backtest_trades":  len(bt_df),
        "metrics":          metrics,
        "flagged_count":    flagged_count,
        "bt_exit_counts":   bt_stats.get("exit_counts", {}),
        "live_exit_counts": live_stats.get("exit_counts", {}),
        "bt_pnl_dist":      bt_stats.get("pnl_dist", []),
        "live_pnl_dist":    live_stats.get("pnl_dist", []),
    }
