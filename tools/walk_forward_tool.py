"""
Tool: run_walk_forward
Runs rolling walk-forward optimization to test robustness across time windows.

Window sizes are automatically derived from the actual number of available bars,
so the tool works correctly whether data comes from Yahoo Finance (60d cap on 5m)
or IBKR (up to 6 months of 5m bars).

With IBKR data (9210 bars @ 5m):
  Default split: train=1560 (4 weeks)  test=780 (2 weeks)  → ~9 windows.

With Yahoo data (4558 bars @ 5m):
  Default split: train=780  (2 weeks)  test=390 (1 week)   → ~9 windows.

Walk-forward here is used for VALIDATION, not for selecting deployment params.
Generations are therefore capped at a lighter value (WF_GENERATIONS) to keep
runtime manageable. Windows run in parallel (n_jobs=-1) across available CPUs.
"""
import logging
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
from signal_testing.walk_forward import walk_forward_optimize  # noqa: E402

logger = logging.getLogger("app.walk_forward_tool")

# Lighter generation count for validation runs — keeps WF fast.
_WF_GENERATIONS = 50
_WF_POPSIZE     = 30

# Bars-per-trading-day by interval (approximate, accounts for 6.5h NYSE session)
_BARS_PER_DAY = {"1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7}


def _derive_window_sizes(interval: str, total_bars: int) -> tuple[int, int]:
    """
    Return (train_bars, test_bars) sized to fit within the available data.

    Target: 4-week train + 2-week test (gives ~15–20 trades per window at 5m).
    Falls back to 2-week/1-week if total_bars is too small.

    Uses actual bar count rather than TIME_FRAME string so IBKR and Yahoo
    data are handled consistently regardless of config.TIME_FRAME.
    """
    bars_day = _BARS_PER_DAY.get(interval, 78)

    # Target: 4-week train / 2-week test
    train = bars_day * 20   # 4 weeks
    test  = bars_day * 10   # 2 weeks

    # Halve until at least one window fits
    while train + test > total_bars and train > bars_day:
        train //= 2
        test  //= 2

    return max(train, bars_day), max(test, bars_day // 2)


def walk_forward(
    ticker: str,
    interval: str | None = None,
    train_bars: int | None = None,
    test_bars: int | None = None,
    generations: int | None = None,
    popsize: int | None = None,
    preloaded_data: dict | None = None,
    val_fraction: float = 0.2,
    k_folds: int = 3,
) -> dict:
    """
    Run walk-forward validation and return summary statistics.

    Window sizes are derived from the actual bar count (preloaded_data if
    provided, otherwise from config.TIME_FRAME). With IBKR data this gives
    4-week/2-week windows; with Yahoo 60d data it falls back to 2-week/1-week.

    Parameters
    ----------
    preloaded_data : dict, optional
        Pre-fetched data dict (output of fetch_data / fetch_data_ibkr).
        When provided, skips the internal data fetch and uses actual bar count
        to derive optimal window sizes. Pass this to avoid a second API call
        when optimize_and_deploy has already fetched data.

    Returns
    -------
    dict with keys:
        windows          : list of per-window dicts (train_sharpe, test_sharpe, ...)
        mean_test_sharpe : average out-of-sample Sharpe across windows
        mean_overfit_gap : average (train_sharpe - test_sharpe)
        best_window      : window index with highest test_sharpe
        best_params      : params from the best window
        bars_available   : total bars in dataset
        error            : str | None — set if walk-forward could not run
    """
    iv       = interval or config.INTERVAL
    save_dir = str(config.HISTORY_DIR / f"wf_{ticker}")

    # Derive window sizes from actual bar count when data is available,
    # otherwise fall back to TIME_FRAME estimate.
    if preloaded_data is not None:
        total_bars = len(preloaded_data["price"])
    else:
        from signal_testing.data import fetch_data as _fd
        _FRAME_DAYS = {"1w": 5, "2w": 10, "30d": 21, "60d": 42, "90d": 63, "1y": 252}
        bars_day   = _BARS_PER_DAY.get(iv, 78)
        total_bars = _FRAME_DAYS.get(config.TIME_FRAME, 42) * bars_day

    auto_train, auto_test = _derive_window_sizes(iv, total_bars)
    tb = train_bars or auto_train
    ob = test_bars  or auto_test

    logger.info(
        "Walk-forward: ticker=%s interval=%s bars=%d train=%d test=%d gens=%d",
        ticker, iv, total_bars, tb, ob, generations or _WF_GENERATIONS,
    )

    try:
        df = walk_forward_optimize(
            ticker=ticker,
            interval=iv,
            train_bars=tb,
            test_bars=ob,
            step_bars=ob,
            n_generations=generations or _WF_GENERATIONS,
            popsize=popsize or _WF_POPSIZE,
            min_trades=config.OPT_MIN_TRADES,
            save_dir=save_dir,
            time_frame=config.TIME_FRAME,
            preloaded_data=preloaded_data,
            commission_rate=config.STOCK_COMMISSION_RATE,
            commission_min=config.STOCK_COMMISSION_MIN,
            n_jobs=-1,
            val_fraction=val_fraction,
            k_folds=k_folds,
        )
    except ValueError as exc:
        msg = str(exc)
        logger.warning("Walk-forward failed: %s", msg)
        return {
            "windows":          [],
            "mean_test_sharpe": None,
            "mean_overfit_gap": None,
            "best_window":      None,
            "best_params":      None,
            "bars_available":   total_bars,
            "error":            msg,
        }

    windows = []
    for _, row in df.iterrows():
        windows.append({
            "window":       int(row["window"]),
            "train_sharpe": float(row["train_sharpe"]),
            "test_sharpe":  float(row["test_sharpe"]),
            "test_pnl":     float(row["test_pnl"]),
            "test_trades":  int(row["test_trades"]),
            "params": {
                "weights":     row["weights"],
                "n":           float(row["n"]),
                "stop_loss":   float(row["stop_loss"]),
                "take_profit": float(row["take_profit"]),
            },
        })

    if not windows:
        return {
            "windows": [], "mean_test_sharpe": 0.0, "mean_overfit_gap": 0.0,
            "best_window": None, "best_params": None, "bars_available": total_bars,
            "error": "No windows completed.",
        }

    test_sharpes = [w["test_sharpe"] for w in windows]
    overfit_gaps = [w["train_sharpe"] - w["test_sharpe"] for w in windows]
    best_idx     = max(range(len(windows)), key=lambda i: windows[i]["test_sharpe"])

    return {
        "windows":          windows,
        "mean_test_sharpe": round(sum(test_sharpes) / len(test_sharpes), 4),
        "mean_overfit_gap": round(sum(overfit_gaps) / len(overfit_gaps), 4),
        "best_window":      windows[best_idx]["window"],
        "best_params":      windows[best_idx]["params"],
        "bars_available":   total_bars,
        "error":            None,
    }


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "run_walk_forward",
    "description": (
        "Run rolling walk-forward validation across time windows to test "
        "parameter robustness out-of-sample. Window sizes are automatically "
        "derived from the actual bar count (IBKR: 4-week/2-week windows; "
        "Yahoo 60d fallback: 2-week/1-week). Returns per-window train/test "
        "Sharpe ratios, mean_test_sharpe, mean_overfit_gap, and best OOS params. "
        "Returns an 'error' key (not an exception) if data is unavailable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'NVDA'",
            },
            "interval": {
                "type": "string",
                "description": "Bar size: '1m', '5m', '15m'. Defaults to config value.",
            },
            "train_bars": {
                "type": "integer",
                "description": "Bars per training window. Auto-derived from bar count if omitted.",
            },
            "test_bars": {
                "type": "integer",
                "description": "Bars per test window. Auto-derived from bar count if omitted.",
            },
            "generations": {
                "type": "integer",
                "description": f"Optimizer generations per window (default {_WF_GENERATIONS} for fast validation).",
            },
        },
        "required": ["ticker"],
    },
}
