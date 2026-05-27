"""
Tool: run_fundamental_optimization
Runs the SNES optimizer for the fundamental model.
"""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from fundamental_testing.optimizer import run_optimization  # noqa: E402


def optimize_fundamental(
    ticker: str,
    extra_train_tickers: list[str] | None = None,
    lookback_years: float = 3.0,
    generations: int | None = None,
    popsize: int | None = None,
    min_trades: int | None = None,
) -> dict:
    """
    Run SNES optimization for the fundamental model.

    Returns dict: weights (15 floats), n, stop_loss, take_profit, sharpe,
                  generations_run, stopped_early, train_tickers
    """
    train_tickers = [ticker]
    if extra_train_tickers:
        train_tickers.extend(extra_train_tickers)

    result = run_optimization(
        train_tickers=train_tickers,
        lookback_years=lookback_years,
        n_generations=generations or config.OPT_GENERATIONS,
        popsize=popsize or config.OPT_POPSIZE,
        min_trades=min_trades or config.FUND_GATE_MIN_TRADES,
    )
    return {
        "weights":         result["weights"],
        "n":               result["n"],
        "stop_loss":       result["stop_loss"],
        "take_profit":     result["take_profit"],
        "sharpe":          result["sharpe"],
        "generations_run": result["generations_run"],
        "stopped_early":   result["stopped_early"],
        "train_tickers":   result["train_tickers"],
        "n_indicators":    result.get("n_indicators", 15),
    }


TOOL_SPEC = {
    "name": "run_fundamental_optimization",
    "description": (
        "Run the SNES evolutionary optimizer to find optimal indicator weights, "
        "signal threshold, stop-loss, and take-profit for the 15-indicator "
        "fundamental model. Operates on daily bars using quarterly financial data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Primary training ticker"},
            "extra_train_tickers": {
                "type": "array", "items": {"type": "string"},
                "description": "Additional tickers for multi-ticker generalization",
            },
            "lookback_years": {"type": "number", "description": "Years of history (default 3.0)"},
            "generations":    {"type": "integer", "description": "Max SNES generations (default 150)"},
            "popsize":        {"type": "integer", "description": "Population size (default 50)"},
            "min_trades":     {"type": "integer", "description": "Min trades for non-penalty fitness"},
        },
        "required": ["ticker"],
    },
}
