"""
Tests for config.py — path resolution, defaults, and load_deployed_params.
"""
import json
import sys
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGENT_ROOT))

import config


class TestPaths:
    def test_signal_testing_dir_resolves(self):
        """SIGNAL_TESTING_DIR must point at the Git project root."""
        assert config.SIGNAL_TESTING_DIR.name == "stock-signal-testing"

    def test_trading_app_dir_resolves(self):
        assert config.TRADING_APP_DIR.name == "trading-app"

    def test_registry_dir_resolves(self):
        assert config.REGISTRY_DIR.name == "registry"
        assert config.REGISTRY_DIR.parent == AGENT_ROOT

    def test_params_file_inside_trading_app(self):
        assert config.PARAMS_FILE.parent.parent == config.TRADING_APP_DIR

    def test_history_dir_inside_registry(self):
        assert config.HISTORY_DIR.parent == config.REGISTRY_DIR


class TestDefaults:
    def test_gate_thresholds_are_positive(self):
        assert config.GATE_MIN_SHARPE > 0
        assert config.GATE_MIN_TRADES > 0
        assert 0 < config.GATE_MAX_DRAWDOWN < 1
        assert config.GATE_MAX_OVERFIT_GAP > 0

    def test_check_interval_positive(self):
        assert config.CHECK_INTERVAL_MIN > 0

    def test_degradation_sharpe_below_gate(self):
        # Degradation threshold should be below deployment minimum
        assert config.DEGRADATION_SHARPE < config.GATE_MIN_SHARPE

    def test_default_ticker_set(self):
        assert isinstance(config.TICKER, str)
        assert len(config.TICKER) > 0


class TestLoadDeployedParams:
    def test_returns_none_when_no_file(self, isolated_config):
        # params file doesn't exist yet
        assert isolated_config.load_deployed_params() is None

    def test_returns_dict_when_file_exists(self, isolated_config, good_params):
        isolated_config.PARAMS_FILE.write_text(json.dumps(good_params))
        result = isolated_config.load_deployed_params()
        assert isinstance(result, dict)
        assert result["n"] == good_params["n"]

    def test_weights_preserved(self, isolated_config, good_params):
        isolated_config.PARAMS_FILE.write_text(json.dumps(good_params))
        result = isolated_config.load_deployed_params()
        assert result["weights"] == good_params["weights"]


class TestAdversarialPolicyConfig:
    def test_adversarial_strict_defaults_true(self, monkeypatch):
        monkeypatch.delenv("ADVERSARIAL_STRICT", raising=False)
        import importlib, config as cfg
        importlib.reload(cfg)
        assert cfg.ADVERSARIAL_STRICT is True

    def test_adversarial_strict_false_via_env(self, monkeypatch):
        monkeypatch.setenv("ADVERSARIAL_STRICT", "false")
        import importlib, config as cfg
        importlib.reload(cfg)
        assert cfg.ADVERSARIAL_STRICT is False

    def test_adversarial_fail_open_defaults_true(self, monkeypatch):
        monkeypatch.delenv("ADVERSARIAL_FAIL_OPEN", raising=False)
        import importlib, config as cfg
        importlib.reload(cfg)
        assert cfg.ADVERSARIAL_FAIL_OPEN is True

    def test_adversarial_fail_open_false_via_env(self, monkeypatch):
        monkeypatch.setenv("ADVERSARIAL_FAIL_OPEN", "false")
        import importlib, config as cfg
        importlib.reload(cfg)
        assert cfg.ADVERSARIAL_FAIL_OPEN is False
