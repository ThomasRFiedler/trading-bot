"""
Tool: run_fundamental_walk_forward
Runs rolling walk-forward optimization for the fundamental model.
"""
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from fundamental_testing.walk_forward import walk_forward_optimize  # noqa: E402


def run_fundamental_walk_forward(
    tickers: list[str],
    lookback_years: float = 3.0,
    train_months: int = 12,
    test_months: int = 6,
    generations: int = 100,
    popsize: int = 40,
) -> dict:
    """
    Run walk-forward optimization for the fundamental model.

    Returns per-window train/test Sharpe, overfit gaps, and an overall verdict.
    """
    save_dir = str(config.HISTORY_DIR / "fundamental_walk_forward")

    df = walk_forward_optimize(
        tickers=tickers,
        lookback_years=lookback_years,
        train_months=train_months,
        test_months=test_months,
        n_generations=generations,
        popsize=popsize,
        save_dir=save_dir,
    )

    if df.empty:
        return {"error": "No walk-forward windows produced results."}

    windows = df.to_dict(orient="records")
    mean_train = float(df["train_sharpe"].mean())
    mean_test  = float(df["test_sharpe"].mean())
    mean_gap   = float(df["overfit_gap"].mean())
    consistency= float((df["test_sharpe"] > 0).mean())

    verdict = (
        "LIKELY_OVERFIT" if mean_gap > 1.5
        else "SOME_DEGRADATION" if mean_gap > 0.75
        else "GOOD_GENERALIZATION"
    )

    return {
        "windows":          windows,
        "mean_train_sharpe":round(mean_train, 4),
        "mean_test_sharpe": round(mean_test, 4),
        "mean_overfit_gap": round(mean_gap, 4),
        "consistency":      round(consistency, 2),
        "verdict":          verdict,
        "n_windows":        len(windows),
    }


TOOL_SPEC = {
    "name": "run_fundamental_walk_forward",
    "description": (
        "Run rolling walk-forward optimization for the fundamental model. "
        "Splits the historical data into rolling 12-month train / 6-month test "
        "windows. Reports per-window train/test Sharpe ratios and overfit gap. "
        "A mean gap > 1.5 is flagged as likely overfit. Use this after "
        "run_fundamental_optimization to validate generalization before deploying."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tickers":       {"type": "array", "items": {"type": "string"},
                              "description": "Training tickers to use"},
            "lookback_years":{"type": "number",
                              "description": "Years of history (default 3.0)"},
            "train_months":  {"type": "integer",
                              "description": "Training window in months (default 12)"},
            "test_months":   {"type": "integer",
                              "description": "Test window in months (default 6)"},
            "generations":   {"type": "integer",
                              "description": "SNES generations per window (default 100)"},
            "popsize":       {"type": "integer",
                              "description": "Population size per generation (default 40)"},
        },
        "required": ["tickers"],
    },
}
