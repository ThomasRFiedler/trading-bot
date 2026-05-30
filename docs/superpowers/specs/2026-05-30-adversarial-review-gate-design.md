# Adversarial Review Gate — Design Spec
**Date:** 2026-05-30
**Status:** Approved, pending implementation

---

## 1. Integration Point

The adversarial review sits inside `tools/deploy_tool.py` after the four existing numeric gates pass and before any atomic write to the live params file or registry.

Ordering principle: cheap deterministic checks first, expensive model-based judgment second, side effects last.

```
deploy_params()
  ├─ Gate 1: out-of-sample Sharpe ≥ GATE_MIN_SHARPE
  ├─ Gate 2: total trades ≥ GATE_MIN_TRADES
  ├─ Gate 3: max drawdown ≤ GATE_MAX_DRAWDOWN
  ├─ Gate 4: overfit gap ≤ GATE_MAX_OVERFIT_GAP
  ├─ [NEW] adversarial_review(context) → verdict
  │     APPROVE  → continue
  │     REJECT   → block (strict) or warn (advisory)
  │     ERROR    → respect ADVERSARIAL_FAIL_OPEN
  └─ atomic write to params file + registry
```

The review function is decoupled from deployment mechanics. `deploy_params()` assembles context, calls the review function, receives a normalized verdict object, and owns the policy decision. The review function never reads `ADVERSARIAL_STRICT` or `ADVERSARIAL_FAIL_OPEN` — those live in the caller.

This boundary keeps the review function reusable for a future review-only audit mode without redesigning the interface.

---

## 2. Review Inputs

The review function receives a single structured context object:

| Field | Type | Notes |
|---|---|---|
| `candidate_params` | dict | weights array, `n`, `stop_loss`, `take_profit`, `sharpe` |
| `backtest_metrics` | dict | OOS Sharpe, trade count, max drawdown, overfit gap |
| `gate_summary` | dict | pass margin for each of the four numeric gates (review only runs after all pass) |
| `walk_forward_results` | list or None | rolling OOS windows with per-period Sharpe; None if not run |
| `current_deployed_params` | dict or None | live `latest.json` contents for comparison; None on cold-start |
| `deployment_context` | dict | `ticker`, `interval`, `time_frame` — required for regime-risk arguments |

Raw price data, trade logs, and IBKR state are excluded. The reviewer works only from distilled optimizer artifacts, keeping the prompt compact and deterministic.

`walk_forward_results` and `current_deployed_params` are optional named arguments. The reviewer degrades gracefully when either is absent.

---

## 3. Verdict Schema

The review function returns a normalized verdict object with transport/runtime status separated from the model's substantive judgment:

```python
@dataclass
class ReviewVerdict:
    outcome: Literal["APPROVE", "REJECT", "ERROR"]
    reason: str                   # short human-readable explanation
    confidence: float | None      # 0–1, optional, for logging only
    raw_artifacts: dict           # proposer/skeptic/judge outputs for debugging
```

**Outcome semantics:**

- `APPROVE` — judge found the candidate acceptable; deployment proceeds
- `REJECT` — judge found a substantive objection; deployment blocked (strict) or warned (advisory). Substantive uncertainty also normalizes to REJECT — the judge cannot abstain
- `ERROR` — API exception, timeout, empty model response, or parse failure; deployment follows `ADVERSARIAL_FAIL_OPEN`

**Parser contract:**

- The judge prompt solicits only `APPROVE` or `REJECT`. `ERROR` is not a valid model output — it is applied structurally by the calling code
- If judge output is missing required fields or cannot be parsed, the parser normalizes to `ERROR`, not `REJECT`, so malformed model output is handled by the operational failure policy rather than misclassified as a trading objection
- Raw judge output is logged before normalization on any parse failure, to distinguish "model returned garbage" from "parser mishandled a coherent REJECT"

---

## 4. Three-Role Execution

Maps onto the existing `adversarial_review_tool.py` proposer/skeptic/judge pattern:

**Proposer** — argues for deployment.
System prompt frames it as the optimizer: "these params passed all quantitative gates, here is the evidence, make the case for deploying them." Receives the full context object.

**Skeptic** — argues against.
System prompt frames it as a risk auditor: looking for overfit signals, regime fragility, stop-loss/take-profit ratios that don't survive friction, or suspiciously high Sharpe on thin trade counts. Receives the same context plus the proposer's argument.

**Judge** — renders verdict.
Receives proposer and skeptic arguments. System prompt instructs it to output a parseable verdict block with `judgment` (`APPROVE` or `REJECT` only), `reason`, and optionally `confidence`. The parser targets this block specifically rather than free-text. Substantive uncertainty must be expressed as `REJECT` with low confidence and a reason — the judge cannot output `ERROR` or abstain.

---

## 5. Config and Error Handling

**Environment variables (caller-owned):**

| Variable | Default | Effect |
|---|---|---|
| `ADVERSARIAL_STRICT` | `true` | `true`: REJECT blocks deployment. `false`: REJECT logs warning only (advisory mode) |
| `ADVERSARIAL_FAIL_OPEN` | `true` | `true`: ERROR allows deployment to proceed. `false`: ERROR blocks deployment |
| `ADVERSARIAL_PROVIDER` | `anthropic` | Reserved for future cross-provider support; inert in v1 |

Rationale for `ADVERSARIAL_FAIL_OPEN=true` default: the adversarial reviewer is a second-layer heuristic on top of four deterministic gates, not the primary safety system. Blocking all deployments on a transient model-service outage turns an auxiliary control into a single point of operational failure. The four numeric gates remain the hard floor; the reviewer adds judgment on top, not instead.

Provider and model settings remain in the existing `config.py`.

**Error handling — explicit at each boundary:**

| Failure | Normalized outcome | Logged artifact |
|---|---|---|
| API exception | `ERROR` | exception message |
| Timeout | `ERROR` | elapsed time |
| Empty model response | `ERROR` | raw response |
| JSON parse failure | `ERROR` | raw judge output |
| Missing required verdict fields | `ERROR` | raw judge output |
| Coherent REJECT with uncertainty | `REJECT` | full debate artifacts |

**Post-decision logging:**

`deploy_params()` records the review artifact and policy decision before blocking or continuing. Every deployment record must carry a `review_status` field with one of three values:

| `review_status` value | Meaning |
|---|---|
| `approved` | Judge returned APPROVE; deployment proceeded with review confirmation |
| `rejected` | Judge returned REJECT; deployment blocked (strict) or warned (advisory) |
| `error_fail_open` | Review returned ERROR and ADVERSARIAL_FAIL_OPEN=true; deployment proceeded without review |

`approved` and `error_fail_open` are never conflated. A fail-open deployment emits a high-severity log event, persists the raw error outcome in registry metadata, and records `review_status=error_fail_open` — it is never silently recorded as equivalent to an approved deployment.

---

## 6. Testing Matrix

All combinations of outcome × policy must be covered by tests:

| Outcome | `ADVERSARIAL_STRICT=true` | `ADVERSARIAL_STRICT=false` |
|---|---|---|
| `APPROVE` | deployment proceeds | deployment proceeds |
| `REJECT` | deployment blocked | warning logged, deployment proceeds |

| Outcome | `ADVERSARIAL_FAIL_OPEN=true` | `ADVERSARIAL_FAIL_OPEN=false` |
|---|---|---|
| `ERROR` (transport) | deployment proceeds | deployment blocked |
| `ERROR` (parse failure) | deployment proceeds | deployment blocked |
| `ERROR` (malformed verdict) | deployment proceeds | deployment blocked |

Additional behavioral assertions:
- Malformed judge output normalizes to `ERROR`, not `REJECT`
- Judge substantive uncertainty normalizes to `REJECT`, not `ERROR`
- Raw judge output is present in `raw_artifacts` on any parse failure
- `deployment_context` absent from context → raises before calling model (not silently omitted)
- `walk_forward_results=None` and `current_deployed_params=None` both handled without error
- Review function never reads `ADVERSARIAL_STRICT` or `ADVERSARIAL_FAIL_OPEN`

---

## Out of Scope (v1)

- Review-only audit mode for live deployed params (no deployment attempt)
- Cross-provider review (OpenAI/Codex as skeptic) — `ADVERSARIAL_PROVIDER` key reserved but inert
- Weight updates or scaffold self-modification (SIA-style) — future track
