"""Tool: screen_sentiment — score a watchlist on the sentiment model."""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from sentiment_testing.screener import screen_sentiment as _screen  # noqa: E402
from sentiment_testing.indicators import INDICATORS                   # noqa: E402


def screen_sentiment_tickers(
    tickers: list[str],
    weights: list[float] = None,
    n: float = None,
    lookback_years: float = 1.0,
    max_workers: int = 4,
) -> dict:
    """
    Score tickers on the sentiment model and return ranked results.

    Loads params from SENTIMENT_PARAMS_FILE by default.
    """
    if config.SENTIMENT_PARAMS_FILE.exists():
        import json
        with open(config.SENTIMENT_PARAMS_FILE) as f:
            p = json.load(f)
        weights = weights or p.get("weights", [w for _, w in INDICATORS])
        n       = n       or p.get("n", 2.0)
    else:
        weights = weights or [w for _, w in INDICATORS]
        n       = n       or 2.0

    results = _screen(
        tickers=tickers,
        weights=weights,
        n=n,
        lookback_years=lookback_years,
        max_workers=max_workers,
        verbose=True,
    )

    passed = [r["ticker"] for r in results if r["passed_screen"]]
    return {
        "tickers_passed":  passed,
        "n_passed":        len(passed),
        "n_screened":      len(results),
        "threshold_n":     n,
        "full_ranking":    results,
    }


TOOL_SPEC = {
    "name": "screen_sentiment",
    "description": (
        "Score a list of tickers on the 9 sentiment indicators and return "
        "a ranked list. Tickers with score >= n pass the sentiment gate. "
        "Use this weekly to identify stocks with favorable sentiment setup "
        "(VIX environment, short squeeze potential, dark pool accumulation, "
        "insider and institutional buying). Typically run after get_universe "
        "and screen_fundamentals to build the final watchlist."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tickers":       {"type": "array", "items": {"type": "string"},
                              "description": "Tickers to screen (from fundamental screen output)"},
            "lookback_years":{"type": "number",
                              "description": "Data lookback in years (default 1.0 — faster)"},
            "max_workers":   {"type": "integer",
                              "description": "Parallel fetch threads (default 4)"},
            "n":             {"type": "number",
                              "description": "Signal threshold override (default from params file)"},
        },
        "required": ["tickers"],
    },
}
