"""Tool: run_sentiment_backtest — daily-bar backtest for the sentiment model."""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from sentiment_testing.backtest import backtest  # noqa: E402
from sentiment_testing.indicators import INDICATORS  # noqa: E402


def run_sentiment_backtest(
    ticker: str,
    n: float = None,
    take_profit: float = None,
    stop_loss: float = None,
    weights: list = None,
    lookback_years: float = 3.0,
    max_hold_days: int = 21,
    params_file: str = None,
) -> dict:
    """
    Run a sentiment model backtest on a single ticker.

    Parameters can be passed directly or loaded from a JSON params file.
    If params_file is given, it overrides n/take_profit/stop_loss/weights.
    """
    if params_file:
        import json
        with open(params_file) as f:
            p = json.load(f)
        n           = p.get("n", n)
        take_profit = p.get("take_profit", take_profit)
        stop_loss   = p.get("stop_loss", stop_loss)
        weights     = p.get("weights", weights)
    elif config.SENTIMENT_PARAMS_FILE.exists():
        import json
        with open(config.SENTIMENT_PARAMS_FILE) as f:
            p = json.load(f)
        n           = n           or p.get("n", 2.0)
        take_profit = take_profit or p.get("take_profit", 0.15)
        stop_loss   = stop_loss   or p.get("stop_loss", 0.07)
        weights     = weights     or p.get("weights")

    # Defaults if still None
    n           = n           or 2.0
    take_profit = take_profit or 0.15
    stop_loss   = stop_loss   or 0.07
    weights     = weights     or [w for _, w in INDICATORS]

    result = backtest(
        ticker=ticker,
        n=n,
        take_profit=take_profit,
        stop_loss=stop_loss,
        weights=weights,
        lookback_years=lookback_years,
        max_hold_days=max_hold_days,
        verbose=True,
    )

    return {
        "ticker":        ticker,
        "pnl":           result["pnl"],
        "sharpe_ratio":  result["sharpe_ratio"],
        "max_drawdown":  result["max_drawdown"],
        "total_trades":  result["total_trades"],
        "profit_factor": result["profit_factor"],
        "n":             n,
        "stop_loss":     stop_loss,
        "take_profit":   take_profit,
    }


TOOL_SPEC = {
    "name": "run_sentiment_backtest",
    "description": (
        "Run a daily-bar backtest for the sentiment model on a single ticker. "
        "Uses max_hold_days=21 (sentiment signals decay faster than fundamentals). "
        "Loads params from SENTIMENT_PARAMS_FILE by default. Use this to validate "
        "sentiment model performance on individual tickers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker":        {"type": "string",  "description": "Stock symbol"},
            "lookback_years":{"type": "number",  "description": "History window (default 3.0)"},
            "max_hold_days": {"type": "integer", "description": "Max days to hold (default 21)"},
            "n":             {"type": "number",  "description": "Signal threshold override"},
            "take_profit":   {"type": "number",  "description": "TP override (0–1)"},
            "stop_loss":     {"type": "number",  "description": "SL override (0–1)"},
            "params_file":   {"type": "string",  "description": "Path to JSON params file"},
        },
        "required": ["ticker"],
    },
}
