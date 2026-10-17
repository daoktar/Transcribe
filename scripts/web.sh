#!/usr/bin/env bash
# Start the web UI at http://127.0.0.1:7860 (run scripts/setup.sh once first).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m transcribe.web "$@"
