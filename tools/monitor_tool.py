"""
Tool: get_live_status
Reads trading-app state.json (live position) and trades.jsonl (closed-trade
history) to produce a live performance summary.

state.json  — authoritative for current open exposure only
trades.jsonl — append-only ledger; authoritative for all closed-trade metrics
"""
import json
import logging
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

_RECENT_TRADES_N = 10


def _read_ledger() -> list[dict]:
    """
    Read trades.jsonl and return closed-trade records oldest-first.

    Fault-tolerant: missing file, partial last line, and individually
    malformed records are all skipped rather than raised.
    """
    path = config.LEDGER_FILE
    if not path.exists():
        return []
    records = []
    skipped = 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
    except OSError:
        pass

    if skipped:
        total_lines = len(records) + skipped
        pct = skipped / total_lines * 100
        logger.warning(
            "[ledger] Skipped %d/%d unparseable lines (%.0f%%) in %s%s",
            skipped, total_lines, pct, path,
            " — possible corruption" if pct > 25 else "",
        )

    return records


def _ledger_summary(records: list[dict]) -> dict:
    """
    Derive summary metrics from a list of closed-trade records.
    Pure computation — no I/O, no trading logic reinterpretation.
    """
    empty = {
        "total_trades":         0,
        "win_rate":             0.0,
        "gross_pnl":            0.0,
        "net_pnl":              0.0,
        "cumulative_pnl":       0.0,
        "recent_trades":        [],
        "exit_reasons":         {},
        "by_account_type":      {},
        "by_trading_mode":      {},
        "live_sharpe_estimate": 0.0,
    }
    if not records:
        return empty

    total = len(records)
    wins  = sum(1 for r in records if r.get("net_pnl", 0.0) > 0)
    gross = sum(r.get("gross_pnl", 0.0) for r in records)
    net   = sum(r.get("net_pnl",   0.0) for r in records)
    cum   = records[-1].get("cumulative_pnl", net)

    exit_reasons: dict[str, int] = {}
    for r in records:
        reason = r.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    def _split(key: str) -> dict:
        groups: dict[str, list] = {}
        for r in records:
            groups.setdefault(r.get(key, "unknown"), []).append(r)
        return {
            k: {
                "trades":   len(g),
                "net_pnl":  round(sum(r.get("net_pnl", 0.0) for r in g), 4),
                "win_rate": round(sum(1 for r in g if r.get("net_pnl", 0.0) > 0) / len(g), 4),
            }
            for k, g in groups.items()
        }

    sharpe = 0.0
    if total >= 5:
        pnls = [r.get("net_pnl", 0.0) for r in records]
        std  = statistics.stdev(pnls)
        if std > 0:
            raw = (statistics.mean(pnls) / std) * math.sqrt(504)
            sharpe = round(raw, 3) if math.isfinite(raw) else 0.0

    return {
        "total_trades":         total,
        "win_rate":             round(wins / total, 4),
        "gross_pnl":            round(gross, 4),
        "net_pnl":              round(net,   4),
        "cumulative_pnl":       round(cum,   4),
        "recent_trades":        records[-_RECENT_TRADES_N:],
        "exit_reasons":         exit_reasons,
        "by_account_type":      _split("account_type"),
        "by_trading_mode":      _split("trading_mode"),
        "live_sharpe_estimate": sharpe,
    }


def get_live_status() -> dict:
    """
    Read the trading-app's live state and closed-trade ledger.

    Returns
    -------
    From state.json (live exposure):
        connected        : bool — trader has written state recently
        in_position      : bool
        position_type    : "long" | "short" | None
        entry_price      : float | None
        last_updated     : ISO UTC timestamp of state file modification

    From trades.jsonl (closed-trade history):
        total_trades         : int
        win_rate             : float  0.0–1.0
        gross_pnl            : float
        net_pnl              : float  (after commission)
        cumulative_pnl       : float  (running total from last ledger record)
        recent_trades        : list[dict]  last 10 closed trades (schema as-is)
        exit_reasons         : dict  {"take_profit": N, "stop_loss": N, ...}
        by_account_type      : dict  {"paper": {trades, net_pnl, win_rate}, ...}
        by_trading_mode      : dict  {"stock": {trades, net_pnl, win_rate}, ...}
        live_sharpe_estimate : float  (net P&L Sharpe, annualised, ≥5 trades)

    Metadata:
        deployed_params    : dict | None
        params_deployed_at : ISO timestamp of last deployment
    """
    deployed_params = config.load_deployed_params()
    result = {
        "connected":            False,
        "in_position":          False,
        "position_type":        None,
        "entry_price":          None,
        "last_updated":         None,
        "deployed_params":      deployed_params,
        "params_deployed_at":   deployed_params.get("deployed_at") if deployed_params else None,
    }

    if config.STATE_FILE.exists():
        try:
            state = json.loads(config.STATE_FILE.read_text())
            result["connected"]     = True
            result["in_position"]   = state.get("in_position", False)
            result["position_type"] = state.get("position_type")
            result["entry_price"]   = state.get("entry_price")
            mtime = config.STATE_FILE.stat().st_mtime
            result["last_updated"]  = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except (json.JSONDecodeError, OSError):
            pass

    result.update(_ledger_summary(_read_ledger()))
    return result


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "get_live_status",
    "description": (
        "Read the live trading-app state and closed-trade ledger. "
        "Returns current open-position exposure (from state.json) and "
        "closed-trade metrics — total trades, win rate, gross/net P&L, "
        "cumulative P&L, exit-reason breakdown, paper/live and stock/crypto "
        "splits, and an annualised Sharpe estimate (from trades.jsonl). "
        "Use this to decide whether re-optimisation is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
