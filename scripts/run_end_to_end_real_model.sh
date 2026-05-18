#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  exec uv run python scripts/run_end_to_end_real_model.py "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/run_end_to_end_real_model.py "$@"
fi

exec python scripts/run_end_to_end_real_model.py "$@"
