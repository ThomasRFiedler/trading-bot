"""
Shared fixtures for all test modules.
All tests run with filesystem I/O redirected to a tmp_path so they never
touch the real trading-app or registry.
"""
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make the trading-agent package importable from any working directory
# ---------------------------------------------------------------------------
AGENT_ROOT = Path(__file__).parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


# ---------------------------------------------------------------------------
# Minimal valid params / metrics fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def good_params():
    return {
        "weights":     [1.0] * 14,
        "n":           2.0,
        "stop_loss":   0.02,
        "take_profit": 0.04,
        "sharpe":      1.5,
    }


@pytest.fixture
def good_metrics():
    return {
        "sharpe_ratio":  1.2,
        "max_drawdown":  0.05,
        "total_trades":  20,
        "pnl":           50.0,
        "overfit_gap":   0.3,
    }


# ---------------------------------------------------------------------------
# Isolated filesystem — redirect config paths into tmp_path
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """
    Patch config module so all file paths point inside tmp_path.
    Returns the patched config module for convenience.
    """
    import config

    params_dir = tmp_path / "params"
    params_dir.mkdir()
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    history_dir  = registry_dir / "history"
    history_dir.mkdir()

    monkeypatch.setattr(config, "PARAMS_FILE",        params_dir / "latest.json")
    monkeypatch.setattr(config, "STATE_FILE",         tmp_path / "state.json")
    monkeypatch.setattr(config, "LOG_FILE",           tmp_path / "trading.log")
    monkeypatch.setattr(config, "LEDGER_FILE",        tmp_path / "trades.jsonl")
    monkeypatch.setattr(config, "PID_FILE",           tmp_path / "trader.pid")
    monkeypatch.setattr(config, "REGISTRY_DIR",       registry_dir)
    monkeypatch.setattr(config, "HISTORY_DIR",        history_dir)
    monkeypatch.setattr(config, "MODELS_FILE",        registry_dir / "models.json")
    monkeypatch.setattr(config, "ADVERSARIAL_REVIEW", False)
    monkeypatch.setattr(config, "ADVERSARIAL_STRICT",    True)
    monkeypatch.setattr(config, "ADVERSARIAL_FAIL_OPEN", True)

    # Seed an empty registry
    (registry_dir / "models.json").write_text(json.dumps({"models": []}))

    return config
