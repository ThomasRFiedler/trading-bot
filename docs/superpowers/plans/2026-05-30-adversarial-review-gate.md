# Adversarial Review Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing proposer/skeptic/judge review into a spec-compliant deployment gate with `APPROVE`/`REJECT`/`ERROR` verdict schema, strict/advisory and fail-open/fail-closed policy, and `review_status` audit trail in every registry entry.

**Architecture:** `ReviewVerdict` dataclass replaces the raw verdict dict throughout. `_parse_verdict` normalizes parse failures to `ERROR` (never `REJECT`). `deploy_params` owns all policy (strict/advisory, fail-open/fail-closed) and stamps `review_status` on every registry write.

**Tech Stack:** Python 3.10+, `anthropic` SDK, `dataclasses`, `pytest`, `monkeypatch`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tools/adversarial_review_tool.py` | Modify | `ReviewVerdict` dataclass; refactored `_parse_verdict`; updated `run_adversarial_review` signature + error handling |
| `agents/review_agent.py` | Modify | Judge prompt schema (APPROVE/REJECT only); deployment_context in all three templates |
| `config.py` | Modify | Add `ADVERSARIAL_STRICT`, `ADVERSARIAL_FAIL_OPEN` |
| `tools/deploy_tool.py` | Modify | Policy logic (strict/advisory, fail-open/fail-closed); `review_status` in registry; loud fail-open |
| `tests/conftest.py` | Modify | Patch new config vars in `isolated_config` |
| `tests/test_adversarial_review_tool.py` | Create | `_parse_verdict` unit tests; `run_adversarial_review` API-error tests |
| `tests/test_deploy_tool.py` | Modify | Add `TestAdversarialReviewGate` class — full behavioral matrix |

---

## Task 1: `ReviewVerdict` dataclass + `_parse_verdict` refactor

**Files:**
- Modify: `tools/adversarial_review_tool.py`
- Create: `tests/test_adversarial_review_tool.py`

- [ ] **Step 1: Write failing tests for `_parse_verdict`**

Create `tests/test_adversarial_review_tool.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent
pytest tests/test_adversarial_review_tool.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` — `ReviewVerdict` does not exist yet.

- [ ] **Step 3: Add `ReviewVerdict` dataclass and refactor `_parse_verdict`**

In `tools/adversarial_review_tool.py`, add after the existing imports (after `import anthropic` and before the `logger =` line):

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ReviewVerdict:
    outcome: Literal["APPROVE", "REJECT", "ERROR"]
    reason: str
    confidence: float | None = None
    raw_artifacts: dict = field(default_factory=dict)
```

Replace the entire `_parse_verdict` function:

```python
def _parse_verdict(raw: str) -> ReviewVerdict:
    """
    Extract and parse the JSON block from the Judge's response.
    Parse failure or unrecognized verdict → ERROR (never REJECT).
    Raw judge output is always logged and stored in raw_artifacts on failure.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    json_str = match.group(1) if match else raw.strip()

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        logger.error("Judge returned unparseable response (stored in raw_artifacts):\n%s", raw)
        return ReviewVerdict(
            outcome="ERROR",
            reason=f"Judge response could not be parsed. Raw: {raw[:300]}",
            raw_artifacts={"raw_judge": raw},
        )

    verdict_str = str(data.get("verdict", "")).upper()
    if verdict_str not in ("APPROVE", "REJECT"):
        logger.error(
            "Judge returned unrecognized verdict %r (stored in raw_artifacts):\n%s",
            verdict_str, raw,
        )
        return ReviewVerdict(
            outcome="ERROR",
            reason=f"Unrecognized verdict field: {verdict_str!r}",
            raw_artifacts={"raw_judge": raw, "parsed": data},
        )

    return ReviewVerdict(
        outcome=verdict_str,
        reason=str(data.get("reasoning", "")),
        confidence=data.get("confidence"),
        raw_artifacts={"verdict_data": data},
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_adversarial_review_tool.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent add \
    tools/adversarial_review_tool.py \
    tests/test_adversarial_review_tool.py
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent commit -m "feat(review): ReviewVerdict dataclass + ERROR-normalized _parse_verdict"
```

---

## Task 2: Config additions + conftest update

**Files:**
- Modify: `config.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write failing config tests**

Add to `tests/test_config.py` (append after existing tests):

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_config.py::TestAdversarialPolicyConfig -v
```

Expected: `AttributeError: module 'config' has no attribute 'ADVERSARIAL_STRICT'`

- [ ] **Step 3: Add config vars to `config.py`**

In `config.py`, in the `# Adversarial review` section (after `REVIEW_MIN_CONFIDENCE`), add:

```python
ADVERSARIAL_STRICT    = os.getenv("ADVERSARIAL_STRICT",    "true").lower() == "true"
ADVERSARIAL_FAIL_OPEN = os.getenv("ADVERSARIAL_FAIL_OPEN", "true").lower() == "true"
```

- [ ] **Step 4: Update `isolated_config` fixture in `tests/conftest.py`**

In the `isolated_config` fixture, add two lines after `monkeypatch.setattr(config, "ADVERSARIAL_REVIEW", False)`:

```python
    monkeypatch.setattr(config, "ADVERSARIAL_STRICT",    True)
    monkeypatch.setattr(config, "ADVERSARIAL_FAIL_OPEN", True)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_config.py::TestAdversarialPolicyConfig tests/test_deploy_tool.py -v
```

Expected: new config tests PASS, existing deploy_tool tests still PASS.

- [ ] **Step 6: Commit**

```bash
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent add \
    config.py tests/conftest.py tests/test_config.py
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent commit -m "feat(config): ADVERSARIAL_STRICT + ADVERSARIAL_FAIL_OPEN env vars"
```

---

## Task 3: Update judge prompt — APPROVE/REJECT only, deployment_context in templates

**Files:**
- Modify: `agents/review_agent.py`

- [ ] **Step 1: Replace `JUDGE_SYSTEM` in `agents/review_agent.py`**

Replace the entire `JUDGE_SYSTEM` string:

```python
JUDGE_SYSTEM = """You are the Judge in an adversarial review of a trading strategy deployment.
You have read both the Proposer's case for deployment and the Skeptic's rebuttal.

Your job is to render a final verdict. You must output a JSON block (and nothing else)
in this exact format:

```json
{
  "verdict": "APPROVE" | "REJECT",
  "confidence": <float 0.0–1.0>,
  "key_risks": ["<risk 1>", "<risk 2>", ...],
  "reasoning": "<2–4 sentence explanation of your decision>"
}
```

Verdict definitions:
- "APPROVE" — the evidence clearly supports deployment; risks are acceptable
- "REJECT"  — risks identified by the Skeptic are material enough to block deployment,
               OR you are substantively uncertain. Express uncertainty as REJECT with
               low confidence and a clear reason — do not abstain.

Confidence guidance:
- 0.9+  : overwhelming evidence one way
- 0.7–0.9: clear preponderance
- 0.5–0.7: marginal; lean one way but risks are real
- <0.5  : always REJECT — do not approve at this confidence level

Rules:
- You cannot override numeric gate failures — those are handled upstream.
- An "APPROVE" with confidence < 0.65 must list at least 2 key_risks.
- Output ONLY the JSON block. No preamble, no trailing text.
- Valid verdicts are APPROVE and REJECT only. Do not output any other value."""
```

- [ ] **Step 2: Add `deployment_context` section to all three user templates**

Replace `PROPOSER_USER_TEMPLATE`:

```python
PROPOSER_USER_TEMPLATE = """Review this candidate deployment and argue FOR it.

## Deployment context
Ticker: {ticker}
Interval: {interval}
Time frame: {time_frame}

## Candidate params
{params_json}

## Backtest metrics
{metrics_json}

## Walk-forward results
{wf_json}

## Currently deployed params (for comparison)
{current_params_json}

## Deployment gates (all passed — review is qualitative)
- Sharpe > {gate_sharpe}
- Total trades >= {gate_trades}
- Max drawdown < {gate_dd:.0%}
- Overfit gap (train - test Sharpe) < {gate_overfit}

Make the strongest honest case for deployment."""
```

Replace `SKEPTIC_USER_TEMPLATE`:

```python
SKEPTIC_USER_TEMPLATE = """Review this candidate deployment and challenge it.

## Deployment context
Ticker: {ticker}
Interval: {interval}
Time frame: {time_frame}

## Candidate params
{params_json}

## Backtest metrics
{metrics_json}

## Walk-forward results
{wf_json}

## Currently deployed params (for comparison)
{current_params_json}

## Proposer's argument
{proposer_argument}

Make the strongest honest case AGAINST deployment. Flag real risks, not imagined ones."""
```

Replace `JUDGE_USER_TEMPLATE`:

```python
JUDGE_USER_TEMPLATE = """Render your verdict on this deployment.

## Deployment context
Ticker: {ticker}
Interval: {interval}
Time frame: {time_frame}

## Candidate params
{params_json}

## Backtest metrics
{metrics_json}

## Walk-forward results
{wf_json}

## Proposer's argument
{proposer_argument}

## Skeptic's rebuttal
{skeptic_rebuttal}

Output only the JSON verdict block."""
```

- [ ] **Step 3: Verify no existing tests break**

```bash
pytest tests/ -v --ignore=tests/test_adversarial_review_tool.py -q
```

Expected: all existing tests PASS (review_agent.py has no direct tests — changes are templates only).

- [ ] **Step 4: Commit**

```bash
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent add agents/review_agent.py
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent commit -m "feat(review): judge prompt APPROVE/REJECT only; deployment_context in all templates"
```

---

## Task 4: Update `run_adversarial_review` — new signature, sanitization, API error handling

**Files:**
- Modify: `tools/adversarial_review_tool.py`

- [ ] **Step 1: Write failing tests for the updated `run_adversarial_review`**

Add to `tests/test_adversarial_review_tool.py`:

```python
import unittest.mock as mock
from tools.adversarial_review_tool import run_adversarial_review, ReviewVerdict

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
        import pytest
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_adversarial_review_tool.py::TestRunAdversarialReview -v
```

Expected: `TypeError` on the signature test, others likely `AttributeError` (signature mismatch).

- [ ] **Step 3: Add `_sanitize_str` helper and update `run_adversarial_review`**

In `tools/adversarial_review_tool.py`, add `_sanitize_str` after the `_call` function:

```python
def _sanitize_str(value: object, max_len: int = 64) -> str:
    """Strip prompt-injectable characters from a string context field."""
    return re.sub(r"[^\w\s\-\./]", "", str(value))[:max_len]
```

Replace the entire `run_adversarial_review` function:

```python
def run_adversarial_review(
    params: dict,
    metrics: dict,
    deployment_context: dict,
    walk_forward_results: dict | None = None,
    current_params: dict | None = None,
) -> ReviewVerdict:
    """
    Run the full Proposer → Skeptic → Judge debate.

    Parameters
    ----------
    params               : Candidate strategy params (weights, n, stop_loss, take_profit)
    metrics              : Backtest metrics (sharpe_ratio, max_drawdown, total_trades, overfit_gap)
    deployment_context   : Required dict with keys: ticker, interval, time_frame
    walk_forward_results : Optional walk-forward output dict
    current_params       : Currently deployed params for comparison; None on cold-start

    Returns
    -------
    ReviewVerdict with outcome APPROVE, REJECT, or ERROR
    """
    logger.info("Starting adversarial review...")

    # Sanitize string context fields before prompt injection
    ticker     = _sanitize_str(deployment_context.get("ticker",     ""))
    interval   = _sanitize_str(deployment_context.get("interval",   ""))
    time_frame = _sanitize_str(deployment_context.get("time_frame", ""))

    params_json         = json.dumps(params,               indent=2, default=str)
    metrics_json        = json.dumps(metrics,              indent=2, default=str)
    wf_json             = json.dumps(walk_forward_results, indent=2, default=str) if walk_forward_results else "Not run."
    current_params_json = json.dumps(current_params,       indent=2, default=str) if current_params else "No current deployment."

    shared_ctx = dict(
        ticker=ticker,
        interval=interval,
        time_frame=time_frame,
        params_json=params_json,
        metrics_json=metrics_json,
        wf_json=wf_json,
        current_params_json=current_params_json,
        gate_sharpe=config.GATE_MIN_SHARPE,
        gate_trades=config.GATE_MIN_TRADES,
        gate_dd=config.GATE_MAX_DRAWDOWN,
        gate_overfit=config.GATE_MAX_OVERFIT_GAP,
    )

    try:
        logger.info("  [1/3] Proposer arguing for deployment...")
        proposer_argument = _call(
            system=PROPOSER_SYSTEM,
            user=PROPOSER_USER_TEMPLATE.format(**shared_ctx),
            model=DEBATER_MODEL,
            max_tokens=1024,
        )
        logger.info("  Proposer done (%d chars)", len(proposer_argument))

        logger.info("  [2/3] Skeptic challenging deployment...")
        skeptic_rebuttal = _call(
            system=SKEPTIC_SYSTEM,
            user=SKEPTIC_USER_TEMPLATE.format(**shared_ctx, proposer_argument=proposer_argument),
            model=DEBATER_MODEL,
            max_tokens=1024,
        )
        logger.info("  Skeptic done (%d chars)", len(skeptic_rebuttal))

        logger.info("  [3/3] Judge rendering verdict...")
        raw_verdict = _call(
            system=JUDGE_SYSTEM,
            user=JUDGE_USER_TEMPLATE.format(
                **shared_ctx,
                proposer_argument=proposer_argument,
                skeptic_rebuttal=skeptic_rebuttal,
            ),
            model=JUDGE_MODEL,
            max_tokens=512,
        )
        logger.info("  Judge done")

    except Exception as exc:
        safe_err = str(exc)[:200]
        logger.error("Adversarial review API call failed: %s", safe_err)
        return ReviewVerdict(
            outcome="ERROR",
            reason=f"API call failed: {safe_err}",
            raw_artifacts={"exception": safe_err},
        )

    verdict = _parse_verdict(raw_verdict)
    verdict.raw_artifacts["proposer_argument"] = proposer_argument
    verdict.raw_artifacts["skeptic_rebuttal"]  = skeptic_rebuttal

    logger.info(
        "Adversarial review verdict: %s (confidence=%s)",
        verdict.outcome,
        f"{verdict.confidence:.2f}" if verdict.confidence is not None else "n/a",
    )
    return verdict
```

- [ ] **Step 4: Update `review_deployed_params` to pass `deployment_context`**

Replace `review_deployed_params` function:

```python
def review_deployed_params(crypto: bool = False) -> ReviewVerdict:
    """
    Dry-run review of the currently deployed params (no deployment).
    Used by --review-only in run.sh / agent.py.
    """
    from tools.backtest_tool import run_backtest

    current = config.load_deployed_params(crypto=crypto)
    if not current:
        logger.error("No deployed params found — nothing to review.")
        return ReviewVerdict(
            outcome="REJECT",
            reason="No deployed params found.",
            raw_artifacts={},
        )

    ticker = config.CRYPTO_TICKER if crypto else config.TICKER
    logger.info("Running backtest on deployed params for review...")
    metrics = run_backtest(ticker=ticker, params=current, crypto=crypto)
    logger.info("Backtest metrics: %s", json.dumps(metrics, default=str))

    return run_adversarial_review(
        params=current,
        metrics=metrics,
        deployment_context={
            "ticker":     ticker,
            "interval":   config.INTERVAL,
            "time_frame": config.TIME_FRAME,
        },
        current_params=current,
    )
```

- [ ] **Step 5: Run all review tool tests**

```bash
pytest tests/test_adversarial_review_tool.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Verify full suite still passes**

```bash
pytest tests/ -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent add \
    tools/adversarial_review_tool.py \
    tests/test_adversarial_review_tool.py
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent commit -m "feat(review): deployment_context, _sanitize_str, API error → ERROR verdict, return ReviewVerdict"
```

---

## Task 5: Update `deploy_params` — policy logic, `review_status`, loud fail-open

**Files:**
- Modify: `tools/deploy_tool.py`

- [ ] **Step 1: Write failing behavioral matrix tests**

Add `TestAdversarialReviewGate` class to `tests/test_deploy_tool.py`:

```python
from tools.adversarial_review_tool import ReviewVerdict


REVIEW_MODULE = "tools.adversarial_review_tool"


def _mock_review(outcome: str, reason: str = "test reason", confidence: float = 0.8):
    """Return a monkeypatch-compatible callable that yields a ReviewVerdict."""
    def _inner(*args, **kwargs):
        return ReviewVerdict(outcome=outcome, reason=reason, confidence=confidence)
    return _inner


class TestAdversarialReviewGate:
    """Behavioral matrix: outcome × strict/advisory × fail-open/fail-closed."""

    # ---- APPROVE --------------------------------------------------------

    def test_approve_deploys_and_status_approved(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("APPROVE"))
        result = deploy_params(good_params, good_metrics)
        assert result["deployed"] is True
        assert result["review_status"] == "approved"

    def test_approve_registry_entry_has_review_status_approved(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("APPROVE"))
        deploy_params(good_params, good_metrics)
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert registry["models"][0]["review_status"] == "approved"

    # ---- REJECT strict --------------------------------------------------

    def test_reject_strict_blocks_deployment(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_STRICT", True)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("REJECT"))
        result = deploy_params(good_params, good_metrics)
        assert result["deployed"] is False
        assert result["review_status"] == "rejected"
        assert any("REJECT" in f for f in result["failures"])

    def test_reject_strict_does_not_write_params_file(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_STRICT", True)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("REJECT"))
        deploy_params(good_params, good_metrics)
        assert not isolated_config.PARAMS_FILE.exists()

    # ---- REJECT advisory ------------------------------------------------

    def test_reject_advisory_deploys_with_warning(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_STRICT", False)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("REJECT"))
        result = deploy_params(good_params, good_metrics)
        assert result["deployed"] is True
        assert result["review_status"] == "rejected"

    def test_reject_advisory_registry_entry_has_review_status_rejected(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_STRICT", False)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("REJECT"))
        deploy_params(good_params, good_metrics)
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert registry["models"][0]["review_status"] == "rejected"

    # ---- ERROR fail-open ------------------------------------------------

    def test_error_fail_open_deploys(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_FAIL_OPEN", True)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("ERROR"))
        result = deploy_params(good_params, good_metrics)
        assert result["deployed"] is True
        assert result["review_status"] == "error_fail_open"

    def test_error_fail_open_emits_critical_log(
        self, isolated_config, good_params, good_metrics, monkeypatch, caplog
    ):
        import logging
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_FAIL_OPEN", True)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("ERROR"))
        with caplog.at_level(logging.CRITICAL):
            deploy_params(good_params, good_metrics)
        assert any("error_fail_open" in r.message or "ADVERSARIAL_FAIL_OPEN" in r.message
                   for r in caplog.records if r.levelno == logging.CRITICAL)

    def test_error_fail_open_registry_entry_has_error_fail_open_status(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_FAIL_OPEN", True)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("ERROR"))
        deploy_params(good_params, good_metrics)
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert registry["models"][0]["review_status"] == "error_fail_open"

    # ---- ERROR fail-closed ----------------------------------------------

    def test_error_fail_closed_blocks_deployment(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_FAIL_OPEN", False)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("ERROR"))
        result = deploy_params(good_params, good_metrics)
        assert result["deployed"] is False
        assert any("ERROR" in f or "fail-closed" in f for f in result["failures"])

    def test_error_fail_closed_does_not_write_params_file(
        self, isolated_config, good_params, good_metrics, monkeypatch
    ):
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_REVIEW", True)
        monkeypatch.setattr(isolated_config, "ADVERSARIAL_FAIL_OPEN", False)
        monkeypatch.setattr(f"{REVIEW_MODULE}.run_adversarial_review", _mock_review("ERROR"))
        deploy_params(good_params, good_metrics)
        assert not isolated_config.PARAMS_FILE.exists()

    # ---- review disabled ------------------------------------------------

    def test_review_disabled_review_status_is_none(
        self, isolated_config, good_params, good_metrics
    ):
        # isolated_config already sets ADVERSARIAL_REVIEW=False
        result = deploy_params(good_params, good_metrics)
        assert result["deployed"] is True
        assert result["review_status"] is None

    def test_review_disabled_registry_entry_review_status_none(
        self, isolated_config, good_params, good_metrics
    ):
        deploy_params(good_params, good_metrics)
        registry = json.loads(isolated_config.MODELS_FILE.read_text())
        assert registry["models"][0]["review_status"] is None
```

- [ ] **Step 2: Run to confirm all new tests fail**

```bash
pytest tests/test_deploy_tool.py::TestAdversarialReviewGate -v 2>&1 | head -30
```

Expected: failures due to missing `review_status` key, missing `run_adversarial_review` import path, missing policy logic.

- [ ] **Step 3: Update `deploy_params` in `tools/deploy_tool.py`**

Replace the adversarial review block (lines 134–177) and the registry entry build (lines 209–218) with the following. The full updated `deploy_params` after `passed, failures = _validate(...)`:

```python
    # --- Adversarial review ---
    review_verdict = None
    review_status: str | None = None

    if config.ADVERSARIAL_REVIEW:
        from tools.adversarial_review_tool import run_adversarial_review
        current_params = config.load_deployed_params(crypto=crypto)
        deployment_context = {
            "ticker":     ticker or config.TICKER,
            "interval":   config.INTERVAL,
            "time_frame": config.TIME_FRAME,
        }
        review_verdict = run_adversarial_review(
            params=params,
            metrics=metrics,
            deployment_context=deployment_context,
            walk_forward_results=walk_forward_results,
            current_params=current_params,
        )

        if review_verdict.outcome == "APPROVE":
            review_status = "approved"
            logger.info(
                "Adversarial review APPROVED (confidence=%s).",
                f"{review_verdict.confidence:.2f}" if review_verdict.confidence is not None else "n/a",
            )

        elif review_verdict.outcome == "REJECT":
            review_status = "rejected"
            if config.ADVERSARIAL_STRICT:
                _archive_rejected(params, metrics, review_verdict, model_type, ticker or config.TICKER)
                return {
                    "deployed":       False,
                    "failures":       [f"Adversarial review REJECTED: {review_verdict.reason}"],
                    "registry_entry": None,
                    "review":         review_verdict,
                    "review_status":  "rejected",
                }
            else:
                logger.warning(
                    "Adversarial review REJECTED (advisory mode — proceeding): %s",
                    review_verdict.reason,
                )

        elif review_verdict.outcome == "ERROR":
            if config.ADVERSARIAL_FAIL_OPEN:
                review_status = "error_fail_open"
                logger.critical(
                    "[adversarial-review] Review returned ERROR — proceeding under "
                    "ADVERSARIAL_FAIL_OPEN=true. Deployment has NOT been reviewed. "
                    "Reason: %s. Set ADVERSARIAL_FAIL_OPEN=false to block on review errors.",
                    review_verdict.reason,
                )
            else:
                return {
                    "deployed":       False,
                    "failures":       [f"Adversarial review ERROR (fail-closed): {review_verdict.reason}"],
                    "registry_entry": None,
                    "review":         review_verdict,
                    "review_status":  None,
                }
```

Add `import dataclasses` to the imports at the top of `tools/deploy_tool.py` (after the existing standard library imports).

Replace the registry entry dict construction (find the block starting with `entry = {`):

```python
    entry = {
        "deployed_at":   now,
        "model_type":    model_type,
        "ticker":        ticker or config.TICKER,
        "params":        deploy_payload,
        "metrics":       metrics,
        "notes":         notes,
        "review":        dataclasses.asdict(review_verdict) if review_verdict is not None else None,
        "review_status": review_status,
    }
```

Add `"review_status": review_status` to the final return dict:

```python
    return {
        "deployed":       True,
        "failures":       [],
        "registry_entry": entry,
        "review":         review_verdict,
        "review_status":  review_status,
    }
```

- [ ] **Step 4: Update `_archive_rejected` to accept `ReviewVerdict`**

Replace the `_archive_rejected` function signature and body:

```python
def _archive_rejected(
    params: dict, metrics: dict, review: "ReviewVerdict", model_type: str, ticker: str
) -> None:
    """Write a rejected candidate to registry/rejected/ for audit."""
    rejected_dir = config.REGISTRY_DIR / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "rejected_at": now,
        "model_type":  model_type,
        "ticker":      ticker,
        "params":      params,
        "metrics":     metrics,
        "review":      vars(review),
    }
    name = f"{now[:19].replace(':', '-')}_{model_type}_{ticker}_rejected.json"
    (rejected_dir / name).write_text(json.dumps(entry, indent=2))
    logger.info("Archived rejected candidate to registry/rejected/%s", name)
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: ALL tests PASS including the new `TestAdversarialReviewGate` matrix.

- [ ] **Step 6: Commit**

```bash
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent add \
    tools/deploy_tool.py \
    tests/test_deploy_tool.py
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent commit -m "feat(deploy): strict/advisory + fail-open/fail-closed policy; review_status audit trail; loud fail-open"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run the complete test suite**

```bash
cd /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent
pytest tests/ -v --tb=short
```

Expected output: all tests PASS. Confirm counts include:
- `TestParseVerdict` — 11 tests
- `TestRunAdversarialReview` — 5 tests
- `TestAdversarialPolicyConfig` — 4 tests
- `TestAdversarialReviewGate` — 13 tests
- All pre-existing tests unchanged

- [ ] **Step 2: Verify spec coverage**

Check each spec requirement has a test:

| Spec requirement | Test |
|---|---|
| APPROVE → deployment proceeds | `test_approve_deploys_and_status_approved` |
| REJECT + strict → blocks | `test_reject_strict_blocks_deployment` |
| REJECT + advisory → proceeds with warning | `test_reject_advisory_deploys_with_warning` |
| ERROR + fail-open → proceeds | `test_error_fail_open_deploys` |
| ERROR + fail-open → critical log | `test_error_fail_open_emits_critical_log` |
| ERROR + fail-closed → blocks | `test_error_fail_closed_blocks_deployment` |
| Parse failure → ERROR not REJECT | `test_malformed_json_returns_error_not_reject` |
| Unknown verdict → ERROR | `test_unknown_verdict_returns_error` |
| Raw artifacts present on error | `test_raw_artifacts_present_on_parse_error` |
| review_status in registry | `test_approve_registry_entry_*`, `test_reject_advisory_registry_*`, `test_error_fail_open_registry_*` |
| review disabled → review_status None | `test_review_disabled_review_status_is_none` |
| API error → ERROR verdict | `test_returns_error_on_api_exception` |
| deployment_context required | `test_missing_deployment_context_raises` |

- [ ] **Step 3: Commit final state**

```bash
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent add -A
git -C /mnt/c/Users/tommy/Documents/Workspace/Claude/trading-agent commit -m "chore: final verification pass — all spec requirements covered" --allow-empty
```
