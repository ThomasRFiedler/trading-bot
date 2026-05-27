"""
Tool: deploy_params
Validates a candidate params dict against deployment gates then atomically
writes it to trading-app/params/latest.json and logs to the model registry.

If config.ADVERSARIAL_REVIEW is True, an additional three-model debate
(Proposer → Skeptic → Judge) runs after numeric gates pass. The Judge's
verdict must be "deploy" for the write to proceed.
"""
import json
import logging
import os
import signal
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import config

logger = logging.getLogger("app.deploy_tool")


def _validate(params: dict, metrics: dict,
              min_trades_override: int | None = None) -> tuple[bool, list[str]]:
    """
    Check all deployment gates.

    Parameters
    ----------
    min_trades_override : int, optional
        Override config.GATE_MIN_TRADES for this validation only.
        Use config.OOS_GATE_MIN_TRADES when validating an OOS test slice
        (which covers only ~12 days and cannot accumulate 10 trades).

    Returns (passed: bool, failures: list[str])
    """
    failures = []

    sharpe = metrics.get("sharpe_ratio", params.get("sharpe", 0.0))
    if sharpe < config.GATE_MIN_SHARPE:
        failures.append(
            f"Sharpe {sharpe:.3f} < threshold {config.GATE_MIN_SHARPE}"
        )

    min_trades   = min_trades_override if min_trades_override is not None else config.GATE_MIN_TRADES
    total_trades = metrics.get("total_trades", 0)
    if total_trades < min_trades:
        failures.append(
            f"Total trades {total_trades} < minimum {min_trades}"
        )

    max_dd = abs(metrics.get("max_drawdown", 0.0))
    if max_dd > config.GATE_MAX_DRAWDOWN:
        failures.append(
            f"Max drawdown {max_dd:.1%} > limit {config.GATE_MAX_DRAWDOWN:.1%}"
        )

    # Walk-forward OOS Sharpe gate — primary generalisation check.
    # Only applied when the agent includes mean_wf_test_sharpe in metrics.
    # Explicitly reject NaN — nan < threshold evaluates False in Python,
    # which would silently pass a completely invalid walk-forward result.
    mean_wf_sharpe = metrics.get("mean_wf_test_sharpe")
    if mean_wf_sharpe is not None:
        import math
        if math.isnan(mean_wf_sharpe) or mean_wf_sharpe < config.GATE_MIN_WF_SHARPE:
            failures.append(
                f"Walk-forward mean test Sharpe {mean_wf_sharpe} "
                f"< minimum {config.GATE_MIN_WF_SHARPE}"
            )

    # Overfit gap — backstop for extreme overfitting only (widened from 1.5 to 15.0
    # to accommodate SNES optimizer's naturally high training Sharpes on 5m bars).
    overfit_gap = metrics.get("overfit_gap", 0.0)
    if overfit_gap > config.GATE_MAX_OVERFIT_GAP:
        failures.append(
            f"Overfit gap {overfit_gap:.3f} > limit {config.GATE_MAX_OVERFIT_GAP}"
        )

    sharpe_ex_top3 = metrics.get("sharpe_ex_top3")
    if sharpe_ex_top3 is not None and sharpe_ex_top3 < config.GATE_MIN_SHARPE_EX_TOP3:
        failures.append(
            f"Sharpe ex-top3 {sharpe_ex_top3:.3f} < threshold {config.GATE_MIN_SHARPE_EX_TOP3} "
            f"(edge too concentrated in outlier trades)"
        )

    weight_hhi = metrics.get("weight_hhi")
    if weight_hhi is not None and weight_hhi > config.GATE_MAX_WEIGHT_HHI:
        failures.append(
            f"Weight HHI {weight_hhi:.3f} > limit {config.GATE_MAX_WEIGHT_HHI} "
            f"(indicator concentration too high)"
        )

    return len(failures) == 0, failures


def deploy_params(
    params: dict,
    metrics: dict,
    model_type: str = "technical",
    ticker: str | None = None,
    notes: str = "",
    crypto: bool = False,
    min_trades_override: int | None = None,
    walk_forward_results: dict | None = None,
) -> dict:
    """
    Validate and deploy params to the trading-app.

    Parameters
    ----------
    params               : Dict with weights, n, stop_loss, take_profit
    metrics              : Dict with sharpe_ratio, max_drawdown, total_trades, pnl
                           Optionally overfit_gap (train_sharpe - test_sharpe)
    model_type           : "technical", "fundamental", or "sentiment"
    ticker               : Ticker these params were trained on
    notes                : Free-text notes for the registry entry
    walk_forward_results : Optional walk-forward output to pass to adversarial review

    Returns
    -------
    dict with keys: deployed (bool), failures (list), registry_entry (dict),
                    review (dict | None)
    """
    passed, failures = _validate(params, metrics, min_trades_override=min_trades_override)

    if not passed:
        return {"deployed": False, "failures": failures, "registry_entry": None, "review": None}

    # --- Adversarial review (optional, enabled via ADVERSARIAL_REVIEW=true in .env) ---
    review = None
    if config.ADVERSARIAL_REVIEW:
        from tools.adversarial_review_tool import run_adversarial_review
        current_params = config.load_deployed_params(crypto=crypto)
        review = run_adversarial_review(
            params=params,
            metrics=metrics,
            walk_forward_results=walk_forward_results,
            current_params=current_params,
        )

        if review["verdict"] != "deploy":
            logger.warning(
                "Adversarial review blocked deployment: %s (confidence=%.2f)",
                review["verdict"].upper(), review["confidence"],
            )
            # Archive the rejected candidate so it can be audited later
            _archive_rejected(params, metrics, review, model_type, ticker or config.TICKER)
            failure_msg = f"Adversarial review [{review['verdict'].upper()}]: {review['reasoning']}"
            if review.get("needs_more_detail"):
                failure_msg += f" — {review['needs_more_detail']}"
            return {
                "deployed":      False,
                "failures":      [failure_msg],
                "registry_entry": None,
                "review":        review,
            }

        if review["confidence"] < config.REVIEW_MIN_CONFIDENCE:
            logger.warning(
                "Adversarial review confidence %.2f below threshold %.2f — blocking.",
                review["confidence"], config.REVIEW_MIN_CONFIDENCE,
            )
            _archive_rejected(params, metrics, review, model_type, ticker or config.TICKER)
            return {
                "deployed": False,
                "failures": [
                    f"Adversarial review confidence {review['confidence']:.2f} "
                    f"below minimum {config.REVIEW_MIN_CONFIDENCE}"
                ],
                "registry_entry": None,
                "review": review,
            }

        logger.info(
            "Adversarial review approved deployment (confidence=%.2f).",
            review["confidence"],
        )

    # --- Determine target params file ---
    params_file = config.CRYPTO_PARAMS_FILE if crypto else config.PARAMS_FILE
    params_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Atomic write via temp file ---
    now = datetime.now(timezone.utc).isoformat()
    deploy_payload = {
        "deployed_at": now,
        "weights":     params["weights"],
        "n":           params["n"],
        "stop_loss":   params["stop_loss"],
        "take_profit": params["take_profit"],
        "sharpe":      metrics.get("sharpe_ratio", params.get("sharpe", 0.0)),
    }

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=params_file.parent,
        delete=False,
        suffix=".tmp",
    )
    json.dump(deploy_payload, tmp, indent=2)
    tmp.close()
    os.replace(tmp.name, params_file)

    # --- Build registry entry ---
    entry = {
        "deployed_at": now,
        "model_type":  model_type,
        "ticker":      ticker or config.TICKER,
        "params":      deploy_payload,
        "metrics":     metrics,
        "notes":       notes,
        "review":      review,   # None when ADVERSARIAL_REVIEW is off
    }

    # --- Append to registry ---
    config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if config.MODELS_FILE.exists():
        registry = json.loads(config.MODELS_FILE.read_text())
    else:
        registry = {"models": []}

    registry["models"].append(entry)
    config.MODELS_FILE.write_text(json.dumps(registry, indent=2))

    # --- Archive a timestamped copy ---
    archive_name = f"{now[:19].replace(':', '-')}_{model_type}_{ticker or config.TICKER}.json"
    archive_path = config.HISTORY_DIR / archive_name
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(entry, indent=2))

    _signal_trader_reload()

    return {"deployed": True, "failures": [], "registry_entry": entry, "review": review}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _signal_trader_reload() -> None:
    """Send SIGHUP to the running trader so it hot-reloads params from disk.

    Non-fatal: logs a warning if the PID file is missing or stale rather than
    raising, so a deploy to a stopped trader still succeeds.
    """
    pid_file = config.PID_FILE
    if not pid_file.exists():
        logger.warning("[deploy] trader.pid not found — params written but trader not signalled. "
                       "Start the trader or run: kill -HUP <pid>")
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGHUP)
        logger.info("[deploy] Sent SIGHUP to trader (PID %d) — params will reload at next bar.", pid)
    except ProcessLookupError:
        logger.warning("[deploy] PID %d in trader.pid is not running — params written but not reloaded.", pid)
        pid_file.unlink(missing_ok=True)
    except (ValueError, OSError) as exc:
        logger.warning("[deploy] Could not signal trader: %s", exc)


def _archive_rejected(
    params: dict, metrics: dict, review: dict, model_type: str, ticker: str
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
        "review":      review,
    }
    name = f"{now[:19].replace(':', '-')}_{model_type}_{ticker}_rejected.json"
    (rejected_dir / name).write_text(json.dumps(entry, indent=2))
    logger.info("Archived rejected candidate to registry/rejected/%s", name)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "deploy_params",
    "description": (
        "Validate a candidate parameter set against deployment gates "
        "(Sharpe, drawdown, trade count, overfitting) and, if all gates pass, "
        "atomically deploy them to the live trading app and log the deployment "
        "to the model registry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "description": "Strategy params: weights (list[float]*14), n, stop_loss, take_profit",
            },
            "metrics": {
                "type": "object",
                "description": (
                    "Backtest metrics: sharpe_ratio, max_drawdown (fraction), total_trades, pnl. "
                    "Optional: overfit_gap (train_sharpe - test_sharpe), "
                    "mean_wf_test_sharpe (from run_walk_forward — required for WF gate), "
                    "sharpe_ex_top3, weight_hhi."
                ),
            },
            "model_type": {
                "type": "string",
                "enum": ["technical", "fundamental", "sentiment"],
                "description": "Which model type these params belong to",
            },
            "ticker": {
                "type": "string",
                "description": "Ticker these params were trained/validated on",
            },
            "notes": {
                "type": "string",
                "description": "Optional notes describing why these params are being deployed",
            },
        },
        "required": ["params", "metrics"],
    },
}
