"""
Tool: run_fundamental_backtest
Backtests the fundamental model on daily bars.
"""
import sys
from pathlib import Path

# Ensure trading-agent root is importable regardless of CWD
_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from fundamental_testing.backtest import backtest  # noqa: E402


def run_fundamental_backtest(
    ticker: str,
    params: dict,
    lookback_years: float = 3.0,
) -> dict:
    """
    Run a daily-bar backtest with the given fundamental params.

    params keys: weights (list[float] len 15), n, stop_loss, take_profit
    Returns: sharpe_ratio, max_drawdown, profit_factor, total_trades, pnl
    """
    result = backtest(
        ticker=ticker,
        n=params["n"],
        take_profit=params["take_profit"],
        stop_loss=params["stop_loss"],
        lookback_years=lookback_years,
        weights=params.get("weights"),
        verbose=False,
    )
    return {
        "sharpe_ratio":  result["sharpe_ratio"],
        "max_drawdown":  result["max_drawdown"],
        "profit_factor": result["profit_factor"],
        "total_trades":  result["total_trades"],
        "pnl":           result["pnl"],
    }


TOOL_SPEC = {
    "name": "run_fundamental_backtest",
    "description": (
        "Backtest the fundamental model on daily bars for a given ticker and "
        "parameter set. Returns Sharpe ratio, max drawdown, profit factor, "
        "total trades, and P&L."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker":       {"type": "string"},
            "params":       {"type": "object",
                             "description": "weights (list[float]*15), n, stop_loss, take_profit"},
            "lookback_years": {"type": "number", "description": "Years of history (default 3.0)"},
        },
        "required": ["ticker", "params"],
    },
}
