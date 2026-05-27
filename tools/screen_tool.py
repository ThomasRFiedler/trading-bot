"""
Tool: screen_fundamentals
Runs the fundamental screener over a watchlist of tickers.
Returns ranked candidates for the technical model to trade.
"""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from fundamental_testing.screener import screen_tickers  # noqa: E402


def screen_fundamentals(
    tickers: list[str],
    weights: list[float] | None = None,
    n: float | None = None,
    lookback_years: float = 1.0,
) -> dict:
    """
    Score a watchlist of tickers on the fundamental model.

    If weights/n are not provided, loads the currently deployed
    fundamental params from config.FUNDAMENTAL_PARAMS_FILE.

    Returns dict with:
        passed   : list of tickers that passed the screen (score >= n)
        results  : full ranked list with scores and DCF values
        n_passed : count of tickers that passed
        n_total  : count of tickers screened
    """
    # Load deployed fundamental params if not provided
    if weights is None or n is None:
        params = _load_fundamental_params()
        if params is None:
            return {
                "passed": [],
                "results": [],
                "n_passed": 0,
                "n_total": len(tickers),
                "error": "No fundamental params deployed. Run run_fundamental_optimization first.",
            }
        weights = params.get("weights", weights)
        n       = params.get("n", n)

    results = screen_tickers(
        tickers=tickers,
        weights=weights,
        n=n,
        lookback_years=lookback_years,
        verbose=False,
    )

    passed  = [r["ticker"] for r in results if r["passed_screen"]]
    summary = []
    for r in results:
        summary.append({
            "ticker":            r["ticker"],
            "fundamental_score": r["fundamental_score"],
            "passed_screen":     r["passed_screen"],
            "sector":            r.get("sector", ""),
            "dcf_mid_upside":    r.get("dcf", {}).get("mid_upside", "N/A"),
            "market_cap":        r.get("market_cap"),
            "fetch_error":       r.get("fetch_error"),
        })

    return {
        "passed":   passed,
        "results":  summary,
        "n_passed": len(passed),
        "n_total":  len(tickers),
    }


def _load_fundamental_params() -> dict | None:
    import json
    if config.FUNDAMENTAL_PARAMS_FILE.exists():
        with open(config.FUNDAMENTAL_PARAMS_FILE) as f:
            return json.load(f)
    return None


TOOL_SPEC = {
    "name": "screen_fundamentals",
    "description": (
        "Screen a watchlist of tickers using the fundamental model. "
        "Returns a ranked list of tickers with fundamental scores and DCF valuations. "
        "Tickers that pass the screen (score >= n) are candidates for the technical "
        "model to trade. Use this before placing trades to filter for fundamentally "
        "sound stocks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols to screen, e.g. ['AAPL','MSFT','NVDA']",
            },
            "weights": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Fundamental model weights (15 floats). Uses deployed params if omitted.",
            },
            "n": {
                "type": "number",
                "description": "Signal threshold. Uses deployed params if omitted.",
            },
            "lookback_years": {
                "type": "number",
                "description": "Years of price history to fetch (default 1.0 for speed)",
            },
        },
        "required": ["tickers"],
    },
}
