"""
Tool: run_backtest
Runs a backtest against stock-signal-testing using a given params dict.
"""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

# Make signal_testing importable without installing it
sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from signal_testing.backtest import backtest  # noqa: E402


_CRYPTO_COMMISSION_RATE = 0.0018
_CRYPTO_COMMISSION_MIN  = 1.75
_STOCK_COMMISSION_RATE  = config.STOCK_COMMISSION_RATE
_STOCK_COMMISSION_MIN   = config.STOCK_COMMISSION_MIN


_STARTING_EQUITY = 10_000.0


def run_backtest(
    ticker: str,
    params: dict,
    time_frame: str = None,
    interval: str = None,
    crypto: bool = False,
    preloaded_data: dict = None,
) -> dict:
    """
    Run a vectorized backtest using the given params.

    Parameters
    ----------
    ticker         : Ticker symbol, e.g. "AAPL" or "BTC"
    params         : Dict with keys: weights, n, stop_loss, take_profit
    time_frame     : Override config.TIME_FRAME (e.g. "60d")
    interval       : Override config.INTERVAL (e.g. "5m")
    crypto         : If True, disable EOD exit and apply IBKR crypto commission
    preloaded_data : Pre-fetched data dict (output of fetch_data). If provided,
                     skips the API call. Use to test on a held-out data split.

    Returns
    -------
    dict with keys:
        sharpe_ratio, max_drawdown (fraction of equity, e.g. -0.15 = -15%),
        profit_factor, total_trades, pnl
    """
    tf = time_frame or config.TIME_FRAME
    iv = interval or config.INTERVAL

    result = backtest(
        ticker=ticker,
        n=params["n"],
        time_frame=tf,
        interval=iv,
        take_profit=params["take_profit"],
        stop_loss=params["stop_loss"],
        weights=params.get("weights"),
        position_size=config.OPT_POSITION_SIZE,
        starting_equity=_STARTING_EQUITY,
        preloaded_data=preloaded_data,
        verbose=False,
        no_eod_exit=crypto,
        commission_rate=_CRYPTO_COMMISSION_RATE if crypto else _STOCK_COMMISSION_RATE,
        commission_min=_CRYPTO_COMMISSION_MIN  if crypto else _STOCK_COMMISSION_MIN,
    )

    # backtest() returns max_drawdown as a raw dollar amount (e.g. -$172).
    # Convert to a fraction of starting equity so deploy gates and display
    # use consistent units (e.g. -0.172 = -17.2%).
    max_dd_fraction = result["max_drawdown"] / _STARTING_EQUITY

    return {
        "sharpe_ratio":  result["sharpe_ratio"],
        "max_drawdown":  max_dd_fraction,
        "profit_factor": result["profit_factor"],
        "total_trades":  result["total_trades"],
        "pnl":           result["pnl"],
    }


# ---------------------------------------------------------------------------
# Tool schema for Claude API tool_use
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "run_backtest",
    "description": (
        "Run a vectorized backtest of the technical model for a given ticker "
        "and parameter set. Returns Sharpe ratio, max drawdown, profit factor, "
        "total trades, and P&L."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'AAPL'",
            },
            "params": {
                "type": "object",
                "description": "Strategy params with keys: weights (list of 14 floats), n (float), stop_loss (float), take_profit (float)",
            },
            "time_frame": {
                "type": "string",
                "description": "Lookback window: '1w', '60d', or '1y'. Defaults to config value.",
            },
            "interval": {
                "type": "string",
                "description": "Bar size: '1m', '5m', '15m'. Defaults to config value.",
            },
        },
        "required": ["ticker", "params"],
    },
}
