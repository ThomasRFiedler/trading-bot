"""
Tool: get_universe
Fetches a market-cap-filtered stock universe as candidates for fundamental screening.
"""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from fundamental_testing.universe import get_universe as _get_universe  # noqa: E402


def get_universe(
    tier: list[str] | None = None,
    sectors: list[str] | None = None,
    max_tickers: int = 80,
) -> dict:
    """
    Fetch small/mid/micro cap candidates from the stock universe.

    Parameters
    ----------
    tier        : Market cap tier(s). Options: "nano","micro","small","mid","large".
                  Default: ["small"] ($2B–$10B).
    sectors     : Optional GICS sector filter, e.g. ["Technology", "Financials"]
    max_tickers : Max tickers to return (default 80)

    Returns
    -------
    dict with:
        tickers    : list of ticker symbols
        universe   : full list with market_cap, sector, tier per ticker
        n_found    : count of tickers found
        tier       : tier(s) used
    """
    if tier is None:
        tier = ["small"]

    results = _get_universe(
        tier=tier,
        sectors=sectors,
        max_tickers=max_tickers,
        verbose=False,
    )

    return {
        "tickers":  [r["ticker"] for r in results],
        "universe": results,
        "n_found":  len(results),
        "tier":     tier,
    }


TOOL_SPEC = {
    "name": "get_universe",
    "description": (
        "Fetch a list of stock tickers filtered by market cap tier. "
        "Returns small-cap ($2B–$10B), mid-cap ($10B–$50B), or micro-cap ($300M–$2B) "
        "candidates across all sectors. Use this before screen_fundamentals to build "
        "a dynamic watchlist of low-cap stocks instead of relying on a fixed list. "
        "Typical workflow: get_universe → screen_fundamentals → deploy top candidates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tier": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["nano", "micro", "small", "mid", "large"],
                },
                "description": "Market cap tier(s). Default: ['small'] ($2B–$10B).",
            },
            "sectors": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional GICS sector filter. Valid values: Technology, "
                    "Health Care, Industrials, Consumer Discretionary, "
                    "Consumer Staples, Energy, Financials, Materials, "
                    "Real Estate, Utilities, Communication Services"
                ),
            },
            "max_tickers": {
                "type": "integer",
                "description": "Maximum number of tickers to return (default 80)",
            },
        },
        "required": [],
    },
}
