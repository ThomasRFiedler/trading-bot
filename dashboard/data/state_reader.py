"""
Read trading-app state.json and trades.jsonl (with trading.log as fallback).

All functions return safe defaults — never raise.

Primary data source: trades.jsonl (JSON Lines ledger written by trader.py).
Fallback:           trading.log  (regex parsing, used for sessions before the
                                  ledger was introduced).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd

_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

# Log line pattern — fallback only (for pre-ledger history)
_RE_EXIT = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+INFO\s+app\.trader\s+—\s+"
    r"EXIT \[(\w+)\] (\w+) \w+ "
    r"entry=([\d.]+) exit=([\d.]+) "
    r"net_pnl=\$([+-]?[\d.]+) cum_pnl=\$([+-]?[\d.]+)"
)
_RE_EQUITY = re.compile(
    r"Account equity: \$([\d,]+\.\d{2})"
)

STALE_THRESHOLD_S = 600   # state.json older than 10 min = offline


def read_state() -> dict:
    """
    Read trading-app state.json.

    Returns dict with keys:
        connected, in_position, position_type, entry_price, entry_time,
        cumulative_pnl, total_trades, last_updated (ISO str | None),
        stale (bool), account_equity (float | None)
    """
    result = {
        "connected":      False,
        "in_position":    False,
        "position_type":  None,
        "entry_price":    None,
        "entry_time":     None,
        "quantity":       None,
        "cumulative_pnl": 0.0,
        "total_trades":   0,
        "last_updated":   None,
        "stale":          True,
        "account_type":   "unknown",   # "paper" | "live" | "unknown"
        "trading_mode":   "",          # "stock" | "crypto"
        "account_equity": _read_account_equity_from_log(),
    }

    try:
        if not config.STATE_FILE.exists():
            return result

        raw = json.loads(Path(config.STATE_FILE).read_text())
        mtime = Path(config.STATE_FILE).stat().st_mtime
        age_s = datetime.now(timezone.utc).timestamp() - mtime

        result.update({
            "connected":      True,
            "in_position":    raw.get("in_position", False),
            "position_type":  raw.get("position_type"),
            "entry_price":    raw.get("entry_price"),
            "entry_time":     raw.get("entry_time"),
            "quantity":       raw.get("quantity"),
            "cumulative_pnl": float(raw.get("cumulative_pnl", 0.0)),
            "total_trades":   int(raw.get("total_trades", 0)),
            "last_updated":   datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
            "stale":          age_s > STALE_THRESHOLD_S,
            "account_type":   raw.get("account_type", "unknown"),
            "trading_mode":   raw.get("trading_mode", ""),
        })
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    return result


def _read_account_equity_from_log() -> float | None:
    """Parse the most recent 'Account equity: $X' line from trading.log."""
    try:
        if not config.LOG_FILE.exists():
            return None
        # Read last 500 lines for efficiency
        lines = Path(config.LOG_FILE).read_text().splitlines()[-500:]
        for line in reversed(lines):
            m = _RE_EQUITY.search(line)
            if m:
                return float(m.group(1).replace(",", ""))
    except OSError:
        pass
    return None


def read_trade_history() -> list[dict]:
    """
    Return all closed trades, oldest first.

    Reads from trades.jsonl (structured ledger) when available.
    Falls back to regex-parsing trading.log for pre-ledger history,
    and merges both so no trades are lost during the transition.

    Every dict has at minimum:
        timestamp    (datetime)
        direction    (str, upper-cased)
        exit_reason  (str)
        entry_price  (float)
        exit_price   (float)
        net_pnl      (float)
        cum_pnl      (float)

    Ledger records also carry:
        account_type, trading_mode, ticker, quantity,
        position_size, gross_pnl, commission, closed_at
    """
    trades = []

    # ── Primary: structured ledger ────────────────────────────────────────────
    if config.LEDGER_FILE.exists():
        try:
            with open(config.LEDGER_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    trades.append({
                        "timestamp":    datetime.fromisoformat(r["closed_at"]),
                        "direction":    r["direction"].upper(),
                        "exit_reason":  r["exit_reason"],
                        "entry_price":  float(r["entry_price"]),
                        "exit_price":   float(r["exit_price"]),
                        "net_pnl":      float(r["net_pnl"]),
                        "cum_pnl":      float(r["cumulative_pnl"]),
                        # enriched fields from ledger
                        "account_type": r.get("account_type", "unknown"),
                        "trading_mode": r.get("trading_mode", ""),
                        "ticker":       r.get("ticker", ""),
                        "quantity":     float(r.get("quantity", 0)),
                        "position_size":float(r.get("position_size", 0)),
                        "gross_pnl":    float(r.get("gross_pnl", r["net_pnl"])),
                        "commission":   float(r.get("commission", 0)),
                        "entry_time":   r.get("entry_time", ""),
                        "exit_time":    r.get("exit_time", ""),
                    })
        except Exception:
            pass

    # ── Fallback: parse log for any history predating the ledger ─────────────
    ledger_start = trades[0]["timestamp"] if trades else None
    try:
        if config.LOG_FILE.exists():
            lines = Path(config.LOG_FILE).read_text().splitlines()[-10_000:]
            for line in lines:
                m = _RE_EXIT.search(line)
                if not m:
                    continue
                ts_str, reason, direction, entry, exit_, net_pnl, cum_pnl = m.groups()
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                # Skip entries already covered by the ledger
                if ledger_start and ts >= ledger_start:
                    continue
                trades.append({
                    "timestamp":    ts,
                    "direction":    direction.upper(),
                    "exit_reason":  reason,
                    "entry_price":  float(entry),
                    "exit_price":   float(exit_),
                    "net_pnl":      float(net_pnl),
                    "cum_pnl":      float(cum_pnl),
                    "account_type": "unknown",
                    "trading_mode": "",
                    "ticker":       "",
                    "quantity":     0.0,
                    "position_size":0.0,
                    "gross_pnl":    float(net_pnl),
                    "commission":   0.0,
                    "entry_time":   "",
                    "exit_time":    "",
                })
    except OSError:
        pass

    trades.sort(key=lambda t: t["timestamp"])
    return trades


def build_equity_series(trades: list[dict], account_equity: float | None = None) -> pd.DataFrame:
    """
    Build a DataFrame of equity over time from trade history.

    Returns DataFrame with columns: timestamp, equity, cum_pnl
    Uses account_equity as base (falls back to 10,000 if unavailable).
    """
    base = account_equity if account_equity else 10_000.0

    if not trades:
        return pd.DataFrame(columns=["timestamp", "equity", "cum_pnl"])

    rows = []
    for t in trades:
        rows.append({
            "timestamp": t["timestamp"],
            "equity":    base + t["cum_pnl"],
            "cum_pnl":   t["cum_pnl"],
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    # Prepend the starting point one second before first trade
    start = pd.DataFrame([{
        "timestamp": df["timestamp"].iloc[0] - pd.Timedelta(seconds=1),
        "equity":    base,
        "cum_pnl":   0.0,
    }])
    return pd.concat([start, df], ignore_index=True)


def pnl_today(trades: list[dict]) -> float:
    """Sum of net_pnl for trades closed today."""
    today = date.today()
    return sum(t["net_pnl"] for t in trades
               if t["timestamp"].date() == today)


def win_rate(trades: list[dict]) -> float | None:
    """Fraction of winning trades. None if no trades."""
    if not trades:
        return None
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    return wins / len(trades)


def live_sharpe(trades: list[dict]) -> float:
    """Rough annualised Sharpe from trade P&L list (needs >= 5 trades)."""
    if len(trades) < 5:
        return 0.0
    import statistics
    pnls = [t["net_pnl"] for t in trades]
    mean = statistics.mean(pnls)
    std  = statistics.stdev(pnls) or 1e-9
    # Assume ~2 trades/day × 252 days
    return round(mean / std * (504 ** 0.5), 3)
