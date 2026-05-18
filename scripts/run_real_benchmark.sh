#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec bash scripts/run_end_to_end_real_model.sh "$@"
