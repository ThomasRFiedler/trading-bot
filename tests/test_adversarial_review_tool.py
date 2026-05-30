"""Tests for adversarial_review_tool — _parse_verdict and ReviewVerdict."""
import sys
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGENT_ROOT))

from tools.adversarial_review_tool import _parse_verdict, ReviewVerdict


class TestParseVerdict:
    def test_approve_from_fenced_json(self):
        raw = '```json\n{"verdict": "APPROVE", "confidence": 0.9, "key_risks": [], "reasoning": "Solid metrics."}\n```'
        v = _parse_verdict(raw)
        assert v.outcome == "APPROVE"
        assert v.confidence == 0.9
        assert v.reason == "Solid metrics."

    def test_reject_from_bare_json(self):
        raw = '{"verdict": "REJECT", "confidence": 0.8, "key_risks": ["overfitted"], "reasoning": "Too risky."}'
        v = _parse_verdict(raw)
        assert v.outcome == "REJECT"
        assert v.reason == "Too risky."

    def test_malformed_json_returns_error_not_reject(self):
        raw = "this is not json at all"
        v = _parse_verdict(raw)
        assert v.outcome == "ERROR"

    def test_unknown_verdict_returns_error(self):
        raw = '{"verdict": "needs_more", "confidence": 0.4, "reasoning": "uncertain"}'
        v = _parse_verdict(raw)
        assert v.outcome == "ERROR"

    def test_missing_verdict_field_returns_error(self):
        raw = '{"confidence": 0.9, "reasoning": "no verdict field here"}'
        v = _parse_verdict(raw)
        assert v.outcome == "ERROR"

    def test_raw_artifacts_present_on_parse_error(self):
        raw = "not json"
        v = _parse_verdict(raw)
        assert v.outcome == "ERROR"
        assert "raw_judge" in v.raw_artifacts
        assert v.raw_artifacts["raw_judge"] == raw

    def test_raw_artifacts_present_on_unknown_verdict(self):
        raw = '{"verdict": "deploy", "confidence": 0.9, "reasoning": "old schema"}'
        v = _parse_verdict(raw)
        assert v.outcome == "ERROR"
        assert "raw_judge" in v.raw_artifacts

    def test_approve_lowercase_normalized(self):
        raw = '{"verdict": "approve", "confidence": 0.85, "reasoning": "ok"}'
        v = _parse_verdict(raw)
        assert v.outcome == "APPROVE"

    def test_reject_lowercase_normalized(self):
        raw = '{"verdict": "reject", "confidence": 0.75, "reasoning": "bad"}'
        v = _parse_verdict(raw)
        assert v.outcome == "REJECT"

    def test_confidence_is_none_when_absent(self):
        raw = '{"verdict": "APPROVE", "reasoning": "no confidence field"}'
        v = _parse_verdict(raw)
        assert v.outcome == "APPROVE"
        assert v.confidence is None

    def test_review_verdict_is_dataclass(self):
        v = ReviewVerdict(outcome="APPROVE", reason="test")
        assert v.outcome == "APPROVE"
        assert v.raw_artifacts == {}


from tools.adversarial_review_tool import run_adversarial_review

GOOD_CONTEXT = {"ticker": "AAPL", "interval": "5m", "time_frame": "60d"}
GOOD_PARAMS  = {"weights": [1.0]*14, "n": 2.0, "stop_loss": 0.02, "take_profit": 0.04}
GOOD_METRICS = {"sharpe_ratio": 1.2, "max_drawdown": 0.05, "total_trades": 20, "overfit_gap": 0.3}

APPROVE_RAW = '{"verdict": "APPROVE", "confidence": 0.9, "key_risks": [], "reasoning": "All good."}'
REJECT_RAW  = '{"verdict": "REJECT",  "confidence": 0.8, "key_risks": ["risk"], "reasoning": "Too risky."}'


class TestRunAdversarialReview:
    def test_returns_review_verdict_on_approve(self, monkeypatch):
        monkeypatch.setattr(
            "tools.adversarial_review_tool._call",
            lambda system, user, model, max_tokens=1024: APPROVE_RAW if "Judge" in system else "arg",
        )
        v = run_adversarial_review(GOOD_PARAMS, GOOD_METRICS, GOOD_CONTEXT)
        assert isinstance(v, ReviewVerdict)
        assert v.outcome == "APPROVE"

    def test_returns_error_on_api_exception(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("connection refused")
        monkeypatch.setattr("tools.adversarial_review_tool._call", boom)
        v = run_adversarial_review(GOOD_PARAMS, GOOD_METRICS, GOOD_CONTEXT)
        assert v.outcome == "ERROR"
        assert "connection refused" in v.reason

    def test_missing_deployment_context_raises(self):
        with pytest.raises(TypeError):
            run_adversarial_review(GOOD_PARAMS, GOOD_METRICS)  # missing deployment_context

    def test_none_walk_forward_handled(self, monkeypatch):
        monkeypatch.setattr(
            "tools.adversarial_review_tool._call",
            lambda system, user, model, max_tokens=1024: APPROVE_RAW if "Judge" in system else "arg",
        )
        v = run_adversarial_review(GOOD_PARAMS, GOOD_METRICS, GOOD_CONTEXT, walk_forward_results=None)
        assert v.outcome == "APPROVE"

    def test_none_current_params_handled(self, monkeypatch):
        monkeypatch.setattr(
            "tools.adversarial_review_tool._call",
            lambda system, user, model, max_tokens=1024: APPROVE_RAW if "Judge" in system else "arg",
        )
        v = run_adversarial_review(GOOD_PARAMS, GOOD_METRICS, GOOD_CONTEXT, current_params=None)
        assert v.outcome == "APPROVE"
