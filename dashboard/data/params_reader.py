"""Read deployed model params from trading-app/params/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

# Exact indicator names in the order they appear in the weights array
TECHNICAL_INDICATOR_NAMES = [
    "SMA Cross",
    "RSI",
    "MACD",
    "Bollinger Bands",
    "VWAP",
    "OBV",
    "Volume Surge",
    "VIX",
    "P/E Ratio",
    "Debt Signaling",
    "Short Interest",
    "Advance / Decline",
    "McClellan Osc.",
    "Relative Strength",
]

FUNDAMENTAL_INDICATOR_NAMES = [
    "DCF Low",
    "DCF Mid",
    "DCF High",
    "Op. Margin Trend",
    "FCF Yield",
    "Current Ratio",
    "Shares Dilution",
    "Insider Buy",
    "EPS Trend",
    "P/E vs Sector",
    "Revenue Accel.",
    "Sector RS",
    "Buyback Yield",
    "Dividend Safety",
    "Debt Trend",
]

SENTIMENT_INDICATOR_NAMES = [
    "VIX Regime",
    "VIX Trend",
    "VIX Term Structure",
    "Short Squeeze",
    "Short Level",
    "Dark Pool Accum.",
    "Dark Pool Trend",
    "Insider Buy",
    "Institutional 13F",
]


def _load(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return None


def read_technical_params() -> dict | None:
    return _load(config.PARAMS_FILE)


def read_fundamental_params() -> dict | None:
    return _load(config.FUNDAMENTAL_PARAMS_FILE)


def read_sentiment_params() -> dict | None:
    return _load(config.SENTIMENT_PARAMS_FILE)


def weights_with_names(params: dict, model_type: str = "technical") -> list[tuple[str, float]]:
    """
    Pair indicator names with weight values.

    Returns list of (name, weight) sorted by weight descending.
    """
    weights = params.get("weights", [])
    if model_type == "fundamental":
        names = FUNDAMENTAL_INDICATOR_NAMES
    elif model_type == "sentiment":
        names = SENTIMENT_INDICATOR_NAMES
    else:
        names = TECHNICAL_INDICATOR_NAMES

    pairs = list(zip(names[:len(weights)], weights))
    return sorted(pairs, key=lambda x: x[1], reverse=True)
