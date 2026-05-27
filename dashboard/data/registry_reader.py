"""Read model registry and deployment history."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config


def read_models() -> list[dict]:
    """Load registry/models.json. Returns [] on missing/corrupt."""
    try:
        if config.MODELS_FILE.exists():
            data = json.loads(config.MODELS_FILE.read_text())
            return data.get("models", [])
    except (OSError, json.JSONDecodeError):
        pass
    return []


def latest_model() -> dict | None:
    models = read_models()
    return models[-1] if models else None


def model_history_df() -> pd.DataFrame:
    """
    DataFrame of all deployments, newest first.

    Columns: deployed_at, model_type, ticker, sharpe, max_drawdown,
             profit_factor, total_trades, pnl, overfit_gap, notes
    """
    models = read_models()
    if not models:
        return pd.DataFrame()

    rows = []
    for m in reversed(models):   # newest first
        metrics = m.get("metrics", {})
        rows.append({
            "deployed_at":    m.get("deployed_at", "")[:19].replace("T", " "),
            "model_type":     m.get("model_type", ""),
            "ticker":         m.get("ticker", ""),
            "sharpe":         round(metrics.get("sharpe_ratio", 0), 4),
            "max_drawdown":   round(metrics.get("max_drawdown", 0), 4),
            "profit_factor":  round(metrics.get("profit_factor", 0), 2),
            "total_trades":   int(metrics.get("total_trades", 0)),
            "pnl":            round(metrics.get("pnl", 0), 2),
            "overfit_gap":    round(metrics.get("overfit_gap", 0), 4),
            "notes":          m.get("notes", ""),
        })

    return pd.DataFrame(rows)


def read_watchlist() -> list[str]:
    """Return the current screened watchlist ticker list."""
    try:
        if config.WATCHLIST_FILE.exists():
            data = json.loads(config.WATCHLIST_FILE.read_text())
            return data.get("tickers", [])
    except (OSError, json.JSONDecodeError):
        pass
    return []
