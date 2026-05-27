"""Tool: run_sentiment_optimization — SNES optimizer for the sentiment model."""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from sentiment_testing.optimizer import run_optimization  # noqa: E402


def optimize_sentiment(
    tickers: list[str],
    lookback_years: float = 3.0,
    generations: int = 150,
    popsize: int = 40,
    save: bool = True,
) -> dict:
    """
    Run SNES optimization for the sentiment model.

    Returns weights, n, stop_loss, take_profit, sharpe, and run metadata.
    Saves params to config.SENTIMENT_PARAMS_FILE if save=True.
    """
    result = run_optimization(
        train_tickers=tickers,
        lookback_years=lookback_years,
        n_generations=generations,
        popsize=popsize,
    )

    if save:
        import json
        save_data = {k: v for k, v in result.items() if k != "progress_df"}
        config.SENTIMENT_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.SENTIMENT_PARAMS_FILE, "w") as f:
            json.dump(save_data, f, indent=2)
        result["saved_to"] = str(config.SENTIMENT_PARAMS_FILE)

    return {
        "weights":         result["weights"],
        "stop_loss":       result["stop_loss"],
        "take_profit":     result["take_profit"],
        "n":               result["n"],
        "sharpe":          result["sharpe"],
        "train_tickers":   result["train_tickers"],
        "generations_run": result["generations_run"],
        "stopped_early":   result["stopped_early"],
        "saved_to":        result.get("saved_to"),
    }


TOOL_SPEC = {
    "name": "run_sentiment_optimization",
    "description": (
        "Run SNES evolutionary optimization for the sentiment model. "
        "Trains on the provided tickers, fitting 9 indicator weights plus "
        "n, stop_loss, and take_profit. Includes L2 regularization and early "
        "stopping to reduce overfitting. Run this monthly or when sentiment "
        "backtest Sharpe degrades significantly."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tickers":       {"type": "array", "items": {"type": "string"},
                              "description": "Training tickers (use diverse sectors)"},
            "lookback_years":{"type": "number",
                              "description": "Years of history (default 3.0)"},
            "generations":   {"type": "integer",
                              "description": "Max SNES generations (default 150)"},
            "popsize":       {"type": "integer",
                              "description": "Population size per generation (default 40)"},
            "save":          {"type": "boolean",
                              "description": "Save params to SENTIMENT_PARAMS_FILE (default true)"},
        },
        "required": ["tickers"],
    },
}
