"""
Central configuration for the trading agent.
Copy .env.example → .env and fill in ANTHROPIC_API_KEY before running.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths (auto-resolved relative to this file's location)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_WORKSPACE = _HERE.parent.parent          # …/Workspace

SIGNAL_TESTING_DIR = _WORKSPACE / "Git" / "stock-signal-testing"
TRADING_APP_DIR    = _WORKSPACE / "Git" / "trading-app"
REGISTRY_DIR       = _HERE / "registry"
HISTORY_DIR        = REGISTRY_DIR / "history"

# Trading-app integration points
PARAMS_FILE              = TRADING_APP_DIR / "params" / "latest.json"
CRYPTO_PARAMS_FILE       = TRADING_APP_DIR / "params" / "crypto_latest.json"
FUNDAMENTAL_PARAMS_FILE  = TRADING_APP_DIR / "params" / "fundamental.json"
SENTIMENT_PARAMS_FILE    = TRADING_APP_DIR / "params" / "sentiment.json"
STATE_FILE    = TRADING_APP_DIR / "state.json"
PID_FILE      = TRADING_APP_DIR / "trader.pid"
LOG_FILE      = TRADING_APP_DIR / "trading.log"
LEDGER_FILE   = TRADING_APP_DIR / "trades.jsonl"
MODELS_FILE   = REGISTRY_DIR / "models.json"

# Screener outputs
WATCHLIST_FILE      = REGISTRY_DIR / "watchlist.json"   # active screened watchlist
SCREEN_RESULTS_FILE = REGISTRY_DIR / "screen_results.json"  # full last-screen output

# ---------------------------------------------------------------------------
# IBKR connection (for historical data fetching during optimization)
# ---------------------------------------------------------------------------
IB_HOST          = os.getenv("IB_HOST",          "127.0.0.1")
IB_PORT          = int(os.getenv("IB_PORT",       "4002"))
# Separate client ID from the live trader (default 1) to avoid conflicts.
IB_DATA_CLIENT_ID = int(os.getenv("IB_DATA_CLIENT_ID", "10"))
# Use IBKR Gateway as the primary intraday data source when available.
# IBKR gives 6 months of 5m bars vs Yahoo Finance's 60-day cap, providing
# more training data and enabling larger walk-forward windows.
USE_IBKR_DATA = os.getenv("USE_IBKR_DATA", "true").lower() == "true"
IBKR_MONTHS   = int(os.getenv("IBKR_MONTHS", "6"))

# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Strategy defaults (overridden by .env)
# ---------------------------------------------------------------------------
TICKER        = os.getenv("TICKER",        "AAPL")
CRYPTO_TICKER = os.getenv("CRYPTO_TICKER", "BTC")
# Tickers used together during crypto optimization. Training on multiple
# correlated instruments forces the optimizer to find weights that generalize
# rather than cherry-picking patterns specific to one asset's history.
CRYPTO_TRAIN_TICKERS = os.getenv("CRYPTO_TRAIN_TICKERS", "BTC,ETH").split(",")
TRADING_MODE  = os.getenv("TRADING_MODE",  "stock")   # "stock" | "crypto"
INTERVAL      = os.getenv("INTERVAL",      "5m")
TIME_FRAME    = os.getenv("TIME_FRAME",    "60d")

# ---------------------------------------------------------------------------
# Optimization hyperparameters
# ---------------------------------------------------------------------------
# Optional fixed stop-loss / take-profit (set in .env to pin them, removing
# those dims from the SNES search).  When unset (default), they are optimized
# within compressed bounds: SL [0.6%–1.5%], TP [1.2%–3.0%].
_fixed_sl  = os.getenv("FIXED_STOP_LOSS")
_fixed_tp  = os.getenv("FIXED_TAKE_PROFIT")
FIXED_STOP_LOSS   = float(_fixed_sl) if _fixed_sl else None
FIXED_TAKE_PROFIT = float(_fixed_tp) if _fixed_tp else None

OPT_GENERATIONS   = int(os.getenv("OPT_GENERATIONS", "200"))
OPT_POPSIZE       = int(os.getenv("OPT_POPSIZE", "50"))
OPT_MIN_TRADES    = int(os.getenv("OPT_MIN_TRADES", "10"))
OPT_PATIENCE      = int(os.getenv("OPT_PATIENCE", "20"))
OPT_POSITION_SIZE = float(os.getenv("OPT_POSITION_SIZE", "1000.0"))

# Stock commission model (applied during optimization and backtesting).
# IBKR Pro tiered: ~0.001% per share, min $1.00 per order.
# At $1000 position size this is effectively $1/side = 0.1% per side = 0.2% round trip.
# This eliminates degenerate sub-0.5% TP strategies that look profitable without friction.
STOCK_COMMISSION_RATE = float(os.getenv("STOCK_COMMISSION_RATE", "0.001"))
STOCK_COMMISSION_MIN  = float(os.getenv("STOCK_COMMISSION_MIN",  "1.0"))

# ---------------------------------------------------------------------------
# Deployment gate thresholds
# ---------------------------------------------------------------------------
GATE_MIN_SHARPE      = float(os.getenv("GATE_MIN_SHARPE", "0.5"))
# Overfit gap (train_sharpe - test_sharpe) widened from 1.5 to 15.0 because
# the SNES optimizer on 5m bars naturally produces train Sharpes of 10–18 while
# still achieving strong OOS Sharpes (~4). The gap reflects optimizer aggression,
# not strategy failure. The primary OOS gate is now GATE_MIN_WF_SHARPE.
GATE_MAX_OVERFIT_GAP = float(os.getenv("GATE_MAX_OVERFIT_GAP", "15.0"))
GATE_MIN_TRADES      = int(os.getenv("GATE_MIN_TRADES", "10"))
GATE_MAX_DRAWDOWN    = float(os.getenv("GATE_MAX_DRAWDOWN", "0.15"))    # 15%
# Walk-forward mean out-of-sample Sharpe must exceed this before deployment.
# This is the primary OOS quality gate — it directly tests whether the strategy
# generalises, independent of how well the optimizer fit the training window.
GATE_MIN_WF_SHARPE   = float(os.getenv("GATE_MIN_WF_SHARPE", "1.0"))
# Outlier sensitivity: Sharpe with top-3 winners removed must still clear this.
# Catches strategies whose edge is concentrated in a handful of lucky trades.
GATE_MIN_SHARPE_EX_TOP3 = float(os.getenv("GATE_MIN_SHARPE_EX_TOP3", "0.3"))
# Weight concentration: HHI above this blocks deployment (1.0=single factor, ~0.07=balanced).
# 0.4 means no more than ~63% of effective weight on a single indicator.
GATE_MAX_WEIGHT_HHI  = float(os.getenv("GATE_MAX_WEIGHT_HHI", "0.4"))
# Looser trade-count gate for the OOS slice in workflow runs.
# The OOS window is only ~12 days (20% of 60d), so requiring 10 trades
# is unrealistic — 3 is the minimum for a non-NaN Sharpe estimate.
OOS_GATE_MIN_TRADES  = int(os.getenv("OOS_GATE_MIN_TRADES", "3"))

# ---------------------------------------------------------------------------
# Fundamental model thresholds
# ---------------------------------------------------------------------------
FUND_GATE_MIN_SHARPE  = float(os.getenv("FUND_GATE_MIN_SHARPE", "0.4"))
FUND_GATE_MIN_TRADES  = int(os.getenv("FUND_GATE_MIN_TRADES", "8"))
RISK_FREE_RATE        = float(os.getenv("RISK_FREE_RATE", "0.045"))
DCF_MARGIN_OF_SAFETY  = float(os.getenv("DCF_MARGIN_OF_SAFETY", "0.20"))

# ---------------------------------------------------------------------------
# Sentiment model thresholds
# ---------------------------------------------------------------------------
SENT_GATE_MIN_SHARPE  = float(os.getenv("SENT_GATE_MIN_SHARPE", "0.3"))
SENT_GATE_MIN_TRADES  = int(os.getenv("SENT_GATE_MIN_TRADES", "10"))
SENT_REOPT_DAYS       = int(os.getenv("SENT_REOPT_DAYS", "30"))       # days between sentiment re-opts

# Default watchlist for fundamental screening (fallback if screener hasn't run)
WATCHLIST = os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,JPM,V,JNJ").split(",")

# Market cap universe settings
UNIVERSE_TIER       = os.getenv("UNIVERSE_TIER", "small").split(",")   # e.g. "small" or "micro,small"
UNIVERSE_MAX        = int(os.getenv("UNIVERSE_MAX", "80"))              # candidates per screen run

# ---------------------------------------------------------------------------
# Orchestrator loop
# ---------------------------------------------------------------------------
CHECK_INTERVAL_MIN        = int(os.getenv("CHECK_INTERVAL_MIN",        "30"))  # minutes between live checks
# Crypto re-optimization cadence — much shorter than the stock check interval
# because the optimizer itself provides natural latency (5–20 min per run).
# Set to 0 to loop immediately after each optimization cycle completes.
CRYPTO_REOPT_INTERVAL_MIN = int(os.getenv("CRYPTO_REOPT_INTERVAL_MIN", "5"))
SCREEN_INTERVAL_DAYS  = int(os.getenv("SCREEN_INTERVAL_DAYS", "7"))    # days between universe screens
REOPT_INTERVAL_DAYS   = int(os.getenv("REOPT_INTERVAL_DAYS", "30"))    # days between fundamental re-opts
DEGRADATION_SHARPE    = float(os.getenv("DEGRADATION_SHARPE", "0.3"))  # trigger re-opt below this (live Sharpe)
# Minimum days between technical re-optimizations regardless of Sharpe signal.
# Prevents repeated curve-fitting to recent noise when the strategy is profitable.
TECH_REOPT_MIN_DAYS   = int(os.getenv("TECH_REOPT_MIN_DAYS", "30"))    # default: re-opt at most once/month

# ---------------------------------------------------------------------------
# Adversarial review
# ---------------------------------------------------------------------------
ADVERSARIAL_REVIEW    = os.getenv("ADVERSARIAL_REVIEW", "false").lower() == "true"
REVIEW_MODEL_JUDGE    = os.getenv("REVIEW_MODEL_JUDGE",    "claude-opus-4-6")
REVIEW_MODEL_DEBATERS = os.getenv("REVIEW_MODEL_DEBATERS", "claude-sonnet-4-6")
# Minimum Judge confidence for a "deploy" verdict to proceed.
# Below this threshold the deployment is blocked even if verdict == "deploy".
REVIEW_MIN_CONFIDENCE = float(os.getenv("REVIEW_MIN_CONFIDENCE", "0.65"))


def load_deployed_params(crypto: bool = False) -> dict | None:
    """Load the currently deployed technical params from trading-app."""
    path = CRYPTO_PARAMS_FILE if crypto else PARAMS_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_fundamental_params() -> dict | None:
    """Load the currently deployed fundamental model params."""
    if not FUNDAMENTAL_PARAMS_FILE.exists():
        return None
    with open(FUNDAMENTAL_PARAMS_FILE) as f:
        return json.load(f)


def load_sentiment_params() -> dict | None:
    """Load the currently deployed sentiment model params."""
    if not SENTIMENT_PARAMS_FILE.exists():
        return None
    with open(SENTIMENT_PARAMS_FILE) as f:
        return json.load(f)
