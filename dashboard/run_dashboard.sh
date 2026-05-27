#!/usr/bin/env bash
# Launch the trading dashboard.
# Run from anywhere — this script finds the project root automatically.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
exec python -m dashboard.app "$@"
