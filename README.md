# Trading Agent

An AI-powered orchestration layer that continuously tests, optimizes, and deploys trading strategy parameters for the IBKR-connected trading app.

The agent uses Claude (`claude-sonnet-4-6`) as the reasoning engine. It calls five tools that wrap the existing `stock-signal-testing` optimizer/backtester and writes validated params to the live `trading-app`.

---

## Architecture

```
Workspace/
├── Git/
│   ├── stock-signal-testing/   ← Optimizer + backtester (14-indicator SNES)
│   └── trading-app/            ← IBKR live execution engine
└── Claude/
    └── trading-agent/          ← This project (orchestration layer)
```

### How it works

```
agent.py
  └─ orchestrator.py  (Claude API tool-use loop)
       ├─ get_live_status    → reads state.json + trading.log
       ├─ run_backtest       → runs vectorized backtest
       ├─ run_optimization   → SNES evolutionary optimizer
       ├─ run_walk_forward   → rolling out-of-sample validation
       └─ deploy_params      → validates gates → writes params/latest.json
```

The agent checks live performance on a configurable interval. If the deployed model's Sharpe degrades below a threshold, it triggers a re-optimization cycle, validates the candidate params against deployment gates, and atomically deploys them.

### Deployment gates

All four must pass before params are written to the trading app:

| Gate | Default | Setting |
|------|---------|---------|
| Out-of-sample Sharpe | ≥ 0.5 | `GATE_MIN_SHARPE` |
| Total trades | ≥ 10 | `GATE_MIN_TRADES` |
| Max drawdown | ≤ 15% | `GATE_MAX_DRAWDOWN` |
| Overfit gap (train − test Sharpe) | ≤ 1.5 | `GATE_MAX_OVERFIT_GAP` |

### Three-model roadmap

| Model | Status | Description |
|-------|--------|-------------|
| Technical | **Active** | 14 weighted technical/fundamental indicators |
| Fundamental | Stub | P/E, debt-to-equity, earnings surprises |
| Sentiment | Stub | VIX regime, breadth, news/social sentiment |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- `stock-signal-testing` and `trading-app` cloned at `../Git/`
- IBKR Gateway running on `127.0.0.1:4002` (paper) for live runs
- An Anthropic API key

### 2. Install dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
```

Key settings in `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Required
TICKER=AAPL                    # Stock to trade
INTERVAL=5m                    # Bar size
TIME_FRAME=60d                 # Optimization lookback
CHECK_INTERVAL_MIN=30          # Minutes between market-hours checks
DEGRADATION_SHARPE=0.3         # Re-optimize below this live Sharpe
```

---

## Usage

All commands go through `run.sh`, which handles Python detection, `.env` loading, and `PYTHONPATH` setup automatically.

```bash
# Validate setup — check paths and API key
./run.sh --check

# Run the technical agent once (analyze → optionally optimize → optionally deploy)
./run.sh

# Continuous loop — checks every CHECK_INTERVAL_MIN during market hours
./run.sh --loop

# Backtest currently deployed params (no deployment)
./run.sh --backtest-only

# Full cycle: optimize → validate → deploy (skips Claude reasoning layer)
./run.sh --optimize-and-deploy

# Run a specific model agent
./run.sh --agent technical

# Run inside OpenShell sandbox (requires OpenShell installed)
./run.sh --sandbox --loop

# Run the test suite
./run.sh --test
```

You can also call `agent.py` directly if you've activated the venv and set `PYTHONPATH`:

```bash
export PYTHONPATH=../../Git/stock-signal-testing:$PYTHONPATH
python agent.py --check
python agent.py --backtest-only
```

---

## Security (OpenShell)

The `.openshell/` directory contains sandbox policies for [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell).

**Filesystem restrictions:**
- Read: `stock-signal-testing/`, `trading-app/state.json`, `trading-app/trading.log`
- Write: `trading-agent/registry/`, `trading-app/params/latest.json` only

**Network restrictions:**
- Allowed: `127.0.0.1:4002` (IBKR), `api.anthropic.com` (Claude API), Yahoo Finance
- Blocked: all other outbound connections

Install OpenShell, then run sandboxed:

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | bash

# Run sandboxed
./run.sh --sandbox --check
./run.sh --sandbox --loop
```

---

## File structure

```
trading-agent/
├── .openshell/
│   ├── policy.yaml          # Filesystem + network ACLs
│   └── sandbox.yaml         # Container definition
├── agents/
│   ├── orchestrator.py      # Claude API tool-use loop
│   ├── technical_agent.py   # Active model agent
│   ├── fundamental_agent.py # Stub
│   └── sentiment_agent.py   # Stub
├── tools/
│   ├── backtest_tool.py     # Wraps signal_testing.backtest.backtest()
│   ├── optimize_tool.py     # Wraps signal_testing.optimizer.run_optimization()
│   ├── walk_forward_tool.py # Wraps signal_testing.walk_forward.walk_forward_optimize()
│   ├── deploy_tool.py       # Validates gates → writes params/latest.json + registry
│   └── monitor_tool.py      # Reads state.json + trading.log
├── registry/
│   ├── models.json          # Deployment history (all versions)
│   └── history/             # Timestamped JSON archives per deployment
├── tests/
│   ├── conftest.py          # Shared fixtures (isolated filesystem)
│   ├── test_config.py       # Path resolution and defaults
│   ├── test_deploy_tool.py  # Validation gates and atomic write
│   ├── test_monitor_tool.py # State + log parsing
│   └── test_agent.py        # Market hours, _check(), tool/agent registry
├── agent.py                 # Entry point
├── config.py                # All settings (paths, thresholds, env vars)
├── run.sh                   # Launcher script
├── requirements.txt
└── .env.example             # Config template
```

---

## Development

### Running tests

```bash
./run.sh --test
# or directly:
python -m pytest tests/ -v
```

Tests are isolated — they redirect all file I/O to `tmp_path` and never touch the real trading-app or registry.

### Adding a new model agent

1. Implement `agents/fundamental_agent.py` — set `ALLOWED_TOOLS`, `SYSTEM_PROMPT`, and `filter_tools()`
2. Add any new tools to `tools/` following the same `TOOL_SPEC` + callable pattern
3. Register new tools in `tools/__init__.py`
4. Add tests in `tests/`

### Registry format

`registry/models.json` tracks every deployment:

```json
{
  "models": [
    {
      "deployed_at": "2026-04-03T20:00:00+00:00",
      "model_type":  "technical",
      "ticker":      "AAPL",
      "params":      { "weights": [...], "n": 2.1, "stop_loss": 0.02, "take_profit": 0.04, "sharpe": 1.1 },
      "metrics":     { "sharpe_ratio": 1.1, "max_drawdown": 0.04, "total_trades": 18, "pnl": 43.2 },
      "notes":       ""
    }
  ]
}
```
