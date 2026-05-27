"""
Tests for agent.py — _check(), _is_market_hours(), CLI argument handling.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGENT_ROOT))

# Import functions directly to avoid triggering main()
from agent import _check, _is_market_hours


# ---------------------------------------------------------------------------
# _is_market_hours
# ---------------------------------------------------------------------------
class TestIsMarketHours:
    def _mock_utc(self, weekday: int, hour_utc: int, minute: int = 0):
        """Return a UTC datetime with a specific weekday and time."""
        # Find a date with the desired weekday (0=Mon, 6=Sun)
        # Use fixed base: 2026-04-06 (Monday)
        base_monday = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta
        dt = base_monday + timedelta(days=weekday, hours=hour_utc, minutes=minute)
        return dt

    def test_monday_930_et_is_open(self):
        # 9:30 ET = 13:30 UTC (EDT = UTC-4)
        dt = self._mock_utc(0, 13, 30)
        with patch("agent.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is True

    def test_monday_before_open_is_closed(self):
        # 9:00 ET = 13:00 UTC
        dt = self._mock_utc(0, 13, 0)
        with patch("agent.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_monday_after_close_is_closed(self):
        # 16:30 ET = 20:30 UTC
        dt = self._mock_utc(0, 20, 30)
        with patch("agent.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_saturday_is_closed(self):
        # Saturday 12:00 ET = 16:00 UTC
        dt = self._mock_utc(5, 16, 0)
        with patch("agent.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_sunday_is_closed(self):
        dt = self._mock_utc(6, 16, 0)
        with patch("agent.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is False

    def test_friday_330pm_et_is_open(self):
        # 15:30 ET = 19:30 UTC
        dt = self._mock_utc(4, 19, 30)
        with patch("agent.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert _is_market_hours() is True


# ---------------------------------------------------------------------------
# _check
# ---------------------------------------------------------------------------
class TestCheck:
    def test_check_fails_without_api_key(self, isolated_config, monkeypatch):
        monkeypatch.setattr(isolated_config, "ANTHROPIC_API_KEY", "")
        # Patch the config imported inside agent.py
        with patch("agent.config", isolated_config):
            result = _check()
        assert result is False

    def test_check_passes_with_key_and_existing_dirs(self, isolated_config, monkeypatch, tmp_path):
        # Point SIGNAL_TESTING_DIR and TRADING_APP_DIR to real tmp dirs
        sst = tmp_path / "stock-signal-testing"
        sst.mkdir()
        app = tmp_path / "trading-app"
        app.mkdir()
        monkeypatch.setattr(isolated_config, "SIGNAL_TESTING_DIR", sst)
        monkeypatch.setattr(isolated_config, "TRADING_APP_DIR", app)
        monkeypatch.setattr(isolated_config, "ANTHROPIC_API_KEY", "sk-test-key")

        with patch("agent.config", isolated_config):
            result = _check()
        assert result is True

    def test_check_fails_when_signal_testing_missing(self, isolated_config, monkeypatch, tmp_path):
        monkeypatch.setattr(isolated_config, "SIGNAL_TESTING_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr(isolated_config, "ANTHROPIC_API_KEY", "sk-test-key")
        with patch("agent.config", isolated_config):
            result = _check()
        assert result is False


# ---------------------------------------------------------------------------
# tools/__init__.py — registry completeness
# ---------------------------------------------------------------------------
class TestToolRegistry:
    def test_all_tools_have_spec(self):
        from tools import ALL_TOOL_SPECS, TOOL_DISPATCH
        spec_names    = {s["name"] for s in ALL_TOOL_SPECS}
        dispatch_names = set(TOOL_DISPATCH.keys())
        assert spec_names == dispatch_names

    def test_each_spec_has_required_fields(self):
        from tools import ALL_TOOL_SPECS
        for spec in ALL_TOOL_SPECS:
            assert "name" in spec
            assert "description" in spec
            assert "input_schema" in spec
            schema = spec["input_schema"]
            assert schema.get("type") == "object"
            assert "properties" in schema

    def test_dispatch_values_are_callable(self):
        from tools import TOOL_DISPATCH
        for name, fn in TOOL_DISPATCH.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# agents/__init__.py — agent registry completeness
# ---------------------------------------------------------------------------
class TestAgentRegistry:
    def test_all_three_agents_registered(self):
        from agents import AGENTS
        assert "technical"   in AGENTS
        assert "fundamental" in AGENTS
        assert "sentiment"   in AGENTS

    def test_each_agent_has_required_attributes(self):
        from agents import AGENTS
        for name, agent in AGENTS.items():
            assert hasattr(agent, "SYSTEM_PROMPT"), f"{name} missing SYSTEM_PROMPT"
            assert hasattr(agent, "ALLOWED_TOOLS"),  f"{name} missing ALLOWED_TOOLS"
            assert hasattr(agent, "filter_tools"),   f"{name} missing filter_tools"

    def test_technical_agent_has_tools(self):
        from agents.technical_agent import ALLOWED_TOOLS
        assert len(ALLOWED_TOOLS) > 0

    def test_filter_tools_returns_subset(self):
        from agents.technical_agent import filter_tools
        from tools import ALL_TOOL_SPECS
        filtered = filter_tools(ALL_TOOL_SPECS)
        assert len(filtered) <= len(ALL_TOOL_SPECS)
        for spec in filtered:
            assert spec in ALL_TOOL_SPECS
