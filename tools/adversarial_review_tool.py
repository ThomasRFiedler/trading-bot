"""
Adversarial review — three-model debate before any deployment.

Flow:
  1. Proposer (Sonnet) argues FOR deployment
  2. Skeptic  (Sonnet) challenges every claim
  3. Judge    (Opus)   reads the debate and emits a structured JSON verdict

Called directly by deploy_tool.deploy_params() after numeric gates pass,
before the atomic file write. Not exposed as a Claude tool-use spec.
"""
import json
import logging
import re
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from dataclasses import dataclass, field
from typing import Literal

import anthropic
import config
from agents.review_agent import (
    DEBATER_MODEL, JUDGE_MODEL,
    PROPOSER_SYSTEM, PROPOSER_USER_TEMPLATE,
    SKEPTIC_SYSTEM,  SKEPTIC_USER_TEMPLATE,
    JUDGE_SYSTEM,    JUDGE_USER_TEMPLATE,
)

logger = logging.getLogger("app.adversarial_review")


@dataclass
class ReviewVerdict:
    outcome: Literal["APPROVE", "REJECT", "ERROR"]
    reason: str
    confidence: float | None = None
    raw_artifacts: dict = field(default_factory=dict)


def _call(system: str, user: str, model: str, max_tokens: int = 1024) -> str:
    """Single non-streaming Claude API call. Returns assistant text."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _sanitize_str(value: object, max_len: int = 64) -> str:
    """Strip prompt-injectable characters from a string context field."""
    return re.sub(r"[^\w\s\-\./]", "", str(value))[:max_len]


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
