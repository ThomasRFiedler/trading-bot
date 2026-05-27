"""
Tests for tools/deploy_tool.py — validation gates and atomic deployment.
"""
import json
import os
import signal
import sys
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGENT_ROOT))

from tools.deploy_tool import _validate, deploy_params, _signal_trader_reload


# ---------------------------------------------------------------------------
# _validate — gate logic
# ---------------------------------------------------------------------------
class TestValidate:
    def test_all_gates_pass(self, good_params, good_metrics):
        passed, failures = _validate(good_params, good_metrics)
        assert passed
        assert failures == []

    def test_sharpe_too_low(self, good_params, good_metrics):
        good_metrics["sharpe_ratio"] = 0.1
        passed, failures = _validate(good_params, good_metrics)
        assert not passed
        assert any("Sharpe" in f for f in failures)

    def test_sharpe_fallback_to_params(self, good_params, good_metrics):
        """If metrics has no sharpe_ratio, should fall back to params['sharpe']."""
        del good_metrics["sharpe_ratio"]
        good_params["sharpe"] = 0.1
        passed, failures = _validate(good_params, good_metrics)
        assert not passed

    def test_too_few_trades(self, good_params, good_metrics):
        good_metrics["total_trades"] = 2
        passed, failures = _validate(good_params, good_metrics)
        assert not passed
        assert any("trades" in f for f in failures)

    def test_drawdown_too_high(self, good_params, good_metrics):
        good_metrics["max_drawdown"] = 0.5
        passed, failures = _validate(good_params, good_metrics)
        assert not passed
        assert any("drawdown" in f.lower() for f in failures)

    def test_overfit_gap_too_large(self, good_params, good_metrics):
        good_metrics["overfit_gap"] = 20.0   # GATE_MAX_OVERFIT_GAP is 15.0
        passed, failures = _validate(good_params, good_metrics)
        assert not passed
        assert any("overfit" in f.lower() for f in failures)

    def test_multiple_failures_reported(self, good_params, good_metrics):
        good_metrics["sharpe_ratio"] = 0.1
        good_metrics["total_trades"] = 1
        passed, failures = _validate(good_params, good_metrics)
        assert not passed
        assert len(failures) >= 2

    def test_overfit_gap_missing_defaults_to_zero(self, good_params, good_metrics):
        """Missing overfit_gap should not cause a gate failure."""
        del good_metrics["overfit_gap"]
        passed, failures = _validate(good_params, good_metrics)
        assert passed


# ---------------------------------------------------------------------------
# deploy_params — integration with filesystem
# ---------------------------------------------------------------------------
class TestDeployParams:
    def test_successful_deploy_writes_params_file(
        self, isolated_config, good_params, good_metrics
    ):
        result = deploy_params(good_params, good_metrics)

        assert result["deployed"] is True
        assert result["failures"] == []
        assert isolated_config.PARAMS_FILE.exists()

    def test_deployed_file_has_correct_keys(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics)
        written = json.loads(isolated_config.PARAMS_FILE.read_text())
        for key in ("weights", "n", "stop_loss", "take_profit", "sharpe"):
            assert key in written

    def test_deployed_file_sharpe_from_metrics(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics)
        written = json.loads(isolated_config.PARAMS_FILE.read_text())
        assert written["sharpe"] == good_metrics["sharpe_ratio"]

    def test_registry_updated_on_deploy(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics)
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert len(registry["models"]) == 1
        entry = registry["models"][0]
        assert entry["model_type"] == "technical"

    def test_history_file_created(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics)
        history_files = list(isolated_config.HISTORY_DIR.iterdir())
        assert len(history_files) == 1

    def test_multiple_deploys_accumulate_in_registry(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics)
        deploy_params(good_params, good_metrics)
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert len(registry["models"]) == 2

    def test_rejected_deploy_does_not_write_file(
        self, isolated_config, good_params, good_metrics
    ):
        good_metrics["sharpe_ratio"] = 0.0   # fails gate
        result = deploy_params(good_params, good_metrics)
        assert result["deployed"] is False
        assert not isolated_config.PARAMS_FILE.exists()

    def test_rejected_deploy_returns_failures(
        self, isolated_config, good_params, good_metrics
    ):
        good_metrics["sharpe_ratio"] = 0.0
        result = deploy_params(good_params, good_metrics)
        assert len(result["failures"]) >= 1

    def test_custom_model_type_stored(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics, model_type="fundamental")
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert registry["models"][0]["model_type"] == "fundamental"

    def test_notes_stored_in_registry(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics, notes="test run")
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert registry["models"][0]["notes"] == "test run"

    def test_atomic_write_no_partial_file(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        """If os.replace fails, no corrupt file should be left behind."""
        import os
        original_replace = os.replace

        call_count = {"n": 0}

        def failing_replace(src, dst):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated disk full")
            original_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError):
            deploy_params(good_params, good_metrics)

        assert not isolated_config.PARAMS_FILE.exists()


# ---------------------------------------------------------------------------
# _signal_trader_reload — SIGHUP behavior
# ---------------------------------------------------------------------------
class TestSignalTraderReload:
    def test_sends_sighup_when_pid_file_exists(self, isolated_config, monkeypatch):
        """Valid PID file → SIGHUP sent to that PID."""
        signals_sent = []

        def fake_kill(pid, sig):
            signals_sent.append((pid, sig))

        monkeypatch.setattr(os, "kill", fake_kill)
        isolated_config.PID_FILE.write_text("12345\n")

        _signal_trader_reload()

        assert signals_sent == [(12345, signal.SIGHUP)]

    def test_no_error_when_pid_file_missing(self, isolated_config):
        """Missing PID file → logs warning, does not raise."""
        assert not isolated_config.PID_FILE.exists()
        _signal_trader_reload()   # must not raise

    def test_stale_pid_removes_pid_file(self, isolated_config, monkeypatch):
        """ProcessLookupError → stale PID file is cleaned up."""
        def fake_kill(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(os, "kill", fake_kill)
        isolated_config.PID_FILE.write_text("99999\n")

        _signal_trader_reload()   # must not raise
        assert not isolated_config.PID_FILE.exists()

    def test_deploy_sends_sighup_on_success(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        """Full deploy_params path: successful deploy triggers SIGHUP."""
        signals_sent = []

        def fake_kill(pid, sig):
            signals_sent.append((pid, sig))

        monkeypatch.setattr(os, "kill", fake_kill)
        isolated_config.PID_FILE.write_text("42\n")

        result = deploy_params(good_params, good_metrics)

        assert result["deployed"] is True
        assert signals_sent == [(42, signal.SIGHUP)]

    def test_deploy_skips_sighup_on_gate_failure(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        """Failed gate → no SIGHUP sent (live params untouched)."""
        signals_sent = []

        def fake_kill(pid, sig):
            signals_sent.append((pid, sig))

        monkeypatch.setattr(os, "kill", fake_kill)
        isolated_config.PID_FILE.write_text("42\n")

        good_metrics["sharpe_ratio"] = 0.1   # fails gate
        result = deploy_params(good_params, good_metrics)

        assert result["deployed"] is False
        assert signals_sent == []
