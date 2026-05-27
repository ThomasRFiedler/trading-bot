"""
Tests for tools/monitor_tool.py — state.json (live position) and ledger parsing.
"""
import json
import sys
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGENT_ROOT))

from tools.monitor_tool import get_live_status, _read_ledger, _ledger_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _trade(net_pnl=1.0, gross_pnl=None, cumulative_pnl=None,
           exit_reason="take_profit", account_type="paper",
           trading_mode="stock", direction="long"):
    return {
        "closed_at":      "2026-05-01T10:00:00+00:00",
        "account_type":   account_type,
        "trading_mode":   trading_mode,
        "ticker":         "AAPL",
        "direction":      direction,
        "entry_price":    100.0,
        "exit_price":     101.0,
        "entry_time":     "2026-05-01T09:30:00",
        "exit_time":      "2026-05-01T10:00:00",
        "quantity":       10.0,
        "position_size":  1000.0,
        "gross_pnl":      gross_pnl if gross_pnl is not None else net_pnl + 0.5,
        "commission":     0.5,
        "net_pnl":        net_pnl,
        "cumulative_pnl": cumulative_pnl if cumulative_pnl is not None else net_pnl,
        "exit_reason":    exit_reason,
    }


def _write_ledger(isolated_config, trades: list[dict]) -> None:
    lines = [json.dumps(t) for t in trades]
    isolated_config.LEDGER_FILE.write_text("\n".join(lines) + "\n")


def _write_state(isolated_config, **kwargs) -> None:
    defaults = {"in_position": False, "position_type": None, "entry_price": None}
    defaults.update(kwargs)
    isolated_config.STATE_FILE.write_text(json.dumps(defaults))


# ---------------------------------------------------------------------------
# No files present
# ---------------------------------------------------------------------------
class TestNoFiles:
    def test_returns_dict(self, isolated_config):
        assert isinstance(get_live_status(), dict)

    def test_connected_false(self, isolated_config):
        assert get_live_status()["connected"] is False

    def test_ledger_defaults(self, isolated_config):
        r = get_live_status()
        assert r["total_trades"]         == 0
        assert r["win_rate"]             == 0.0
        assert r["gross_pnl"]            == 0.0
        assert r["net_pnl"]              == 0.0
        assert r["cumulative_pnl"]       == 0.0
        assert r["recent_trades"]        == []
        assert r["exit_reasons"]         == {}
        assert r["by_account_type"]      == {}
        assert r["by_trading_mode"]      == {}
        assert r["live_sharpe_estimate"] == 0.0


# ---------------------------------------------------------------------------
# state.json — live position only
# ---------------------------------------------------------------------------
class TestStateFile:
    def test_connected_true_when_state_exists(self, isolated_config):
        _write_state(isolated_config)
        assert get_live_status()["connected"] is True

    def test_reads_in_position(self, isolated_config):
        _write_state(isolated_config, in_position=True, position_type="long", entry_price=175.0)
        r = get_live_status()
        assert r["in_position"]   is True
        assert r["position_type"] == "long"
        assert r["entry_price"]   == 175.0

    def test_last_updated_is_iso_string(self, isolated_config):
        _write_state(isolated_config)
        r = get_live_status()
        assert isinstance(r["last_updated"], str)
        assert "T" in r["last_updated"]

    def test_corrupt_state_does_not_raise(self, isolated_config):
        isolated_config.STATE_FILE.write_text("not valid json {{{{")
        assert get_live_status()["connected"] is False


# ---------------------------------------------------------------------------
# _read_ledger — fault tolerance
# ---------------------------------------------------------------------------
class TestReadLedger:
    def test_missing_file_returns_empty(self, isolated_config):
        assert _read_ledger() == []

    def test_reads_valid_records(self, isolated_config):
        _write_ledger(isolated_config, [_trade(1.0), _trade(2.0)])
        assert len(_read_ledger()) == 2

    def test_skips_malformed_lines(self, isolated_config):
        isolated_config.LEDGER_FILE.write_text(
            json.dumps(_trade(1.0)) + "\n"
            "not json at all\n"
            + json.dumps(_trade(2.0)) + "\n"
        )
        records = _read_ledger()
        assert len(records) == 2

    def test_tolerates_partial_last_line(self, isolated_config):
        isolated_config.LEDGER_FILE.write_text(
            json.dumps(_trade(1.0)) + "\n"
            '{"net_pnl": 2.0, "incomplete":'   # truncated write
        )
        records = _read_ledger()
        assert len(records) == 1
        assert records[0]["net_pnl"] == pytest.approx(1.0)

    def test_empty_file_returns_empty(self, isolated_config):
        isolated_config.LEDGER_FILE.write_text("")
        assert _read_ledger() == []

    def test_logs_warning_on_skipped_lines(self, isolated_config, caplog):
        import logging
        isolated_config.LEDGER_FILE.write_text(
            json.dumps(_trade(1.0)) + "\n"
            "bad line\n"
            "another bad line\n"
        )
        with caplog.at_level(logging.WARNING):
            _read_ledger()
        assert any("Skipped" in r.message for r in caplog.records)

    def test_corruption_tag_when_majority_bad(self, isolated_config, caplog):
        import logging
        # 3 bad lines out of 4 total = 75% — should mention "corruption"
        isolated_config.LEDGER_FILE.write_text(
            json.dumps(_trade(1.0)) + "\n"
            "bad\nbad\nbad\n"
        )
        with caplog.at_level(logging.WARNING):
            _read_ledger()
        assert any("corruption" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _ledger_summary — metric derivation
# ---------------------------------------------------------------------------
class TestLedgerSummary:
    def test_empty_records(self):
        s = _ledger_summary([])
        assert s["total_trades"] == 0
        assert s["win_rate"]     == 0.0

    def test_total_trades(self):
        assert _ledger_summary([_trade(1.0), _trade(-1.0)])["total_trades"] == 2

    def test_win_rate(self):
        records = [_trade(1.0), _trade(1.0), _trade(-1.0), _trade(-1.0)]
        assert _ledger_summary(records)["win_rate"] == pytest.approx(0.5)

    def test_all_wins(self):
        assert _ledger_summary([_trade(1.0), _trade(2.0)])["win_rate"] == pytest.approx(1.0)

    def test_all_losses(self):
        assert _ledger_summary([_trade(-1.0), _trade(-2.0)])["win_rate"] == pytest.approx(0.0)

    def test_net_pnl_sums(self):
        records = [_trade(1.0), _trade(-0.5), _trade(2.0)]
        assert _ledger_summary(records)["net_pnl"] == pytest.approx(2.5)

    def test_gross_pnl_sums(self):
        records = [_trade(net_pnl=1.0, gross_pnl=1.5), _trade(net_pnl=2.0, gross_pnl=2.5)]
        assert _ledger_summary(records)["gross_pnl"] == pytest.approx(4.0)

    def test_cumulative_pnl_from_last_record(self):
        records = [
            _trade(1.0, cumulative_pnl=1.0),
            _trade(2.0, cumulative_pnl=3.0),
            _trade(-0.5, cumulative_pnl=2.5),
        ]
        assert _ledger_summary(records)["cumulative_pnl"] == pytest.approx(2.5)

    def test_exit_reasons_counted(self):
        records = [
            _trade(exit_reason="take_profit"),
            _trade(exit_reason="take_profit"),
            _trade(exit_reason="stop_loss"),
        ]
        reasons = _ledger_summary(records)["exit_reasons"]
        assert reasons["take_profit"] == 2
        assert reasons["stop_loss"]   == 1

    def test_recent_trades_capped_at_10(self):
        records = [_trade(float(i)) for i in range(15)]
        recent = _ledger_summary(records)["recent_trades"]
        assert len(recent) == 10
        assert recent[-1]["net_pnl"] == pytest.approx(14.0)

    def test_recent_trades_fewer_than_10(self):
        records = [_trade(1.0), _trade(2.0)]
        assert len(_ledger_summary(records)["recent_trades"]) == 2

    def test_by_account_type_split(self):
        records = [
            _trade(1.0, account_type="paper"),
            _trade(2.0, account_type="paper"),
            _trade(-1.0, account_type="live"),
        ]
        split = _ledger_summary(records)["by_account_type"]
        assert split["paper"]["trades"]   == 2
        assert split["paper"]["net_pnl"]  == pytest.approx(3.0)
        assert split["paper"]["win_rate"] == pytest.approx(1.0)
        assert split["live"]["trades"]    == 1
        assert split["live"]["net_pnl"]   == pytest.approx(-1.0)
        assert split["live"]["win_rate"]  == pytest.approx(0.0)

    def test_by_trading_mode_split(self):
        records = [
            _trade(1.0, trading_mode="stock"),
            _trade(2.0, trading_mode="crypto"),
        ]
        split = _ledger_summary(records)["by_trading_mode"]
        assert set(split.keys()) == {"stock", "crypto"}
        assert split["stock"]["trades"]  == 1
        assert split["crypto"]["trades"] == 1

    def test_sharpe_not_computed_below_five_trades(self):
        records = [_trade(float(i)) for i in range(4)]
        assert _ledger_summary(records)["live_sharpe_estimate"] == 0.0

    def test_sharpe_computed_at_five_trades(self):
        records = [_trade(float(i + 1)) for i in range(5)]
        assert _ledger_summary(records)["live_sharpe_estimate"] != 0.0

    def test_sharpe_zero_when_all_pnls_identical(self):
        # std=0 → no meaningful Sharpe; must not return a spurious huge value
        records = [_trade(1.0) for _ in range(5)]
        assert _ledger_summary(records)["live_sharpe_estimate"] == 0.0


# ---------------------------------------------------------------------------
# get_live_status — integration
# ---------------------------------------------------------------------------
class TestGetLiveStatus:
    def test_ledger_metrics_in_result(self, isolated_config):
        _write_ledger(isolated_config, [_trade(1.0), _trade(-0.5)])
        r = get_live_status()
        assert r["total_trades"] == 2
        assert r["net_pnl"]      == pytest.approx(0.5)

    def test_state_and_ledger_combined(self, isolated_config):
        _write_state(isolated_config, in_position=True, position_type="long", entry_price=50.0)
        _write_ledger(isolated_config, [_trade(3.0)])
        r = get_live_status()
        assert r["connected"]    is True
        assert r["in_position"]  is True
        assert r["total_trades"] == 1
        assert r["net_pnl"]      == pytest.approx(3.0)

    def test_deployed_params_included(self, isolated_config, good_params):
        isolated_config.PARAMS_FILE.write_text(json.dumps(good_params))
        assert get_live_status()["deployed_params"]["n"] == good_params["n"]
