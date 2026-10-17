#!/usr/bin/env bash
# Create .venv in the project root and install requirements (needs Python 3.10+ and ffmpeg).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "ffmpeg not found: brew install ffmpeg (macOS) / apt install ffmpeg"; exit 1; }
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Done. Web UI: scripts/web.sh   CLI: scripts/transcribe.sh <file> [--language ru] [--speakers]"
