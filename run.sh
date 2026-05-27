#!/usr/bin/env bash
# =============================================================================
# run.sh — Trading Agent launcher
#
# Usage:
#   ./run.sh                          # Run technical agent once
#   ./run.sh --loop                   # Continuous loop (market hours)
#   ./run.sh --check                  # Validate config & paths
#   ./run.sh --backtest-only          # Backtest deployed params
#   ./run.sh --optimize-and-deploy    # Full optimize → validate → deploy
#   ./run.sh --review-only            # Adversarial review of deployed params (dry run)
#   ./run.sh --agent fundamental      # Run a specific model agent
#   ./run.sh --workflow               # Full scheduler + dashboard on :8051
#   ./run.sh --workflow crypto        # Crypto scheduler + dashboard on :8051
#   ./run.sh --test                   # Run the test suite
#   ./run.sh --sandbox [args...]      # Run inside OpenShell sandbox
#
# Adversarial review gate:
#   Enable by setting ADVERSARIAL_REVIEW=true in .env.
#   When enabled, every deployment attempt runs a Proposer→Skeptic→Judge
#   debate. The Judge (Opus) must return verdict="deploy" with confidence
#   >= REVIEW_MIN_CONFIDENCE (default 0.65) for the write to proceed.
#   Rejected candidates are archived to registry/rejected/.
#
# Environment:
#   Copy .env.example → .env and set ANTHROPIC_API_KEY before first run.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[run.sh]${NC} $*"; }
warn()    { echo -e "${YELLOW}[run.sh]${NC} $*"; }
error()   { echo -e "${RED}[run.sh]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Locate Python
# ---------------------------------------------------------------------------
find_python() {
    # Prefer venv if present
    if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
        echo "$SCRIPT_DIR/venv/bin/python"
    elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
        echo "$SCRIPT_DIR/.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        error "No Python interpreter found."
        exit 1
    fi
}

PYTHON="$(find_python)"
info "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    # Export variables without requiring 'export' in the file
    set -o allexport
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +o allexport
    info "Loaded .env"
else
    warn ".env not found — using environment variables as-is."
    warn "Copy .env.example → .env and set ANTHROPIC_API_KEY."
fi

# ---------------------------------------------------------------------------
# PYTHONPATH — ensure signal_testing is importable
# ---------------------------------------------------------------------------
SIGNAL_TESTING_DIR="$(realpath "$SCRIPT_DIR/../../Git/stock-signal-testing" 2>/dev/null || echo "")"
if [[ -d "$SIGNAL_TESTING_DIR" ]]; then
    export PYTHONPATH="$SIGNAL_TESTING_DIR:${PYTHONPATH:-}"
    info "PYTHONPATH includes: $SIGNAL_TESTING_DIR"
else
    warn "stock-signal-testing not found at expected path — optimization tools may fail."
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="once"
AGENT_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --test)    MODE="test" ;;
        --sandbox) MODE="sandbox"; shift; SANDBOX_ARGS=("$@"); break ;;
        *)         AGENT_ARGS+=("$arg") ;;
    esac
done

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in

    test)
        info "Running test suite..."
        # Install pytest if needed
        if ! "$PYTHON" -m pytest --version &>/dev/null 2>&1; then
            warn "pytest not found — installing..."
            "$PYTHON" -m pip install pytest --quiet
        fi
        "$PYTHON" -m pytest tests/ -v --tb=short
        ;;

    sandbox)
        if ! command -v openshell &>/dev/null; then
            error "OpenShell not installed. See: https://github.com/NVIDIA/OpenShell"
            exit 1
        fi
        info "Launching in OpenShell sandbox..."
        openshell sandbox create -- "$PYTHON" agent.py "${SANDBOX_ARGS[@]:-}"
        ;;

    once|*)
        # If this is a --workflow run, start the dashboard in the background first.
        _is_workflow=false
        for _a in "${AGENT_ARGS[@]:-}"; do
            [[ "$_a" == "--workflow" ]] && _is_workflow=true && break
        done

        if [[ "$_is_workflow" == true ]]; then
            DASHBOARD_PID_FILE="$SCRIPT_DIR/dashboard.pid"

            # Kill any stale dashboard from a previous run
            if [[ -f "$DASHBOARD_PID_FILE" ]]; then
                _old_pid=$(cat "$DASHBOARD_PID_FILE")
                kill -0 "$_old_pid" 2>/dev/null && kill "$_old_pid" 2>/dev/null || true
                rm -f "$DASHBOARD_PID_FILE"
            fi

            info "Starting dashboard on http://127.0.0.1:8051 ..."
            nohup "$PYTHON" -m dashboard.app --no-browser \
                >> "$SCRIPT_DIR/dashboard.log" 2>&1 &
            echo $! > "$DASHBOARD_PID_FILE"
            info "Dashboard PID $(cat "$DASHBOARD_PID_FILE") — logs: dashboard.log"
        fi

        info "Launching agent.py ${AGENT_ARGS[*]:-}"
        if [[ ${#AGENT_ARGS[@]} -gt 0 ]]; then
            "$PYTHON" agent.py "${AGENT_ARGS[@]}"
        else
            "$PYTHON" agent.py
        fi
        ;;

esac
