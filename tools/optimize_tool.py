"""
Tool: run_optimization
Runs the SNES evolutionary optimizer from stock-signal-testing.
"""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from signal_testing.optimizer import run_optimization  # noqa: E402


_CRYPTO_COMMISSION_RATE = 0.0018
_CRYPTO_COMMISSION_MIN  = 1.75
# Stock commission pulled from config (override via STOCK_COMMISSION_RATE in .env)
_STOCK_COMMISSION_RATE  = config.STOCK_COMMISSION_RATE
_STOCK_COMMISSION_MIN   = config.STOCK_COMMISSION_MIN


def optimize(
    ticker: str,
    extra_train_tickers: list[str] | None = None,
    time_frame: str | None = None,
    interval: str | None = None,
    generations: int | None = None,
    popsize: int | None = None,
    min_trades: int | None = None,
    crypto: bool = False,
    preloaded_data: dict | None = None,
    preloaded_data_map: dict | None = None,
    val_fraction: float = 0.2,
    k_folds: int = 3,
) -> dict:
    """
    Run evolutionary optimization for the technical model.

    Parameters
    ----------
    ticker               : Primary training ticker, e.g. "AAPL" or "BTC"
    extra_train_tickers  : Additional tickers for multi-ticker training (improves generalization)
    time_frame           : "1w", "60d", or "1y"
    interval             : "1m", "5m", "15m", "30m", "1h"
    generations          : Max optimizer generations (default from config)
    popsize              : Population size (default from config)
    min_trades           : Minimum trades required to accept params (default from config)
    crypto               : If True, enable crypto mode (wider bounds, pin equity indicators to 0,
                           apply IBKR crypto commission, disable EOD exit)
    preloaded_data       : Pre-fetched data dict for the primary ticker only.
    preloaded_data_map   : Pre-fetched data for multiple tickers {ticker: data_dict}.
                           Takes precedence over preloaded_data when provided.
                           Use this when training on multiple tickers with pre-split slices so
                           each ticker's training data stays within the correct time window.

    Returns
    -------
    dict with keys: weights, n, stop_loss, take_profit, sharpe,
                    generations_run, stopped_early
    """
    train_tickers = [ticker]
    if extra_train_tickers:
        train_tickers.extend(extra_train_tickers)

    if preloaded_data_map is not None:
        # Caller supplied a fully-keyed map (e.g. multi-ticker with pre-split slices).
        # Ensure the primary ticker is included.
        preloaded_map = preloaded_data_map
        # Add any extra tickers not already in the map to train_tickers for fresh fetch.
        train_tickers = list(preloaded_map.keys())
    elif preloaded_data is not None:
        preloaded_map = {ticker: preloaded_data}
    else:
        preloaded_map = None

    result = run_optimization(
        train_tickers=train_tickers,
        interval=interval or config.INTERVAL,
        time_frame=time_frame or config.TIME_FRAME,
        preloaded_data_map=preloaded_map,
        n_generations=generations or config.OPT_GENERATIONS,
        popsize=popsize or config.OPT_POPSIZE,
        min_trades=min_trades or config.OPT_MIN_TRADES,
        patience=config.OPT_PATIENCE,
        position_size=config.OPT_POSITION_SIZE,
        crypto=crypto,
        commission_rate=_CRYPTO_COMMISSION_RATE if crypto else _STOCK_COMMISSION_RATE,
        commission_min=_CRYPTO_COMMISSION_MIN  if crypto else _STOCK_COMMISSION_MIN,
        val_fraction=val_fraction,
        k_folds=k_folds,
        fixed_stop_loss=config.FIXED_STOP_LOSS,
        fixed_take_profit=config.FIXED_TAKE_PROFIT,
    )

    return {
        "weights":        result["weights"],
        "n":              result["n"],
        "stop_loss":      result["stop_loss"],
        "take_profit":    result["take_profit"],
        "sharpe":         result["sharpe"],
        "generations_run": result["generations_run"],
        "stopped_early":  result["stopped_early"],
        "train_tickers":  result["train_tickers"],
    }


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "run_optimization",
    "description": (
        "Run the SNES evolutionary optimizer to find optimal indicator weights, "
        "signal threshold (n), stop-loss, and take-profit for the technical model. "
        "Returns the best params found along with the training Sharpe ratio."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Primary training ticker, e.g. 'AAPL'",
            },
            "extra_train_tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional tickers for multi-ticker training to improve generalization",
            },
            "time_frame": {
                "type": "string",
                "description": "Training data lookback: '1w', '60d', '1y'",
            },
            "interval": {
                "type": "string",
                "description": "Bar size: '1m', '5m', '15m'",
            },
            "generations": {
                "type": "integer",
                "description": "Max optimizer generations (default 200)",
            },
            "popsize": {
                "type": "integer",
                "description": "Population size per generation (default 50)",
            },
            "min_trades": {
                "type": "integer",
                "description": "Minimum trades required to accept params (default 10)",
            },
        },
        "required": ["ticker"],
    },
}
