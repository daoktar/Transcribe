#!/usr/bin/env bash
# CLI wrapper: scripts/transcribe.sh <media...> [--engine qwen|whisper] [--language ru] [--speakers]
# Paths are resolved against the directory you call it from.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$ROOT/.venv/bin/python" -m transcribe.cli "$@"
