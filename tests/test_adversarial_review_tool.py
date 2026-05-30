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
