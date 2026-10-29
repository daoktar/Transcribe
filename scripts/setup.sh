#!/usr/bin/env bash
# Create .venv in the project root and install requirements (needs Python 3.10+ and ffmpeg).
# On Linux with an NVIDIA GPU it also sets up the GPU path: CUDA torch, whisper.cpp built
# with CUDA and the Qwen3-ASR CUDA backend (needs nvcc; TRANSCRIBE_CPU_ONLY=1 skips it).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "ffmpeg not found: brew install ffmpeg (macOS) / apt install ffmpeg"; exit 1; }
PY="${PYTHON:-python3}"
CUDA=0
if [ "$(uname -s)" = Linux ] && [ "${TRANSCRIBE_CPU_ONLY:-0}" != 1 ] \
        && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    command -v nvcc >/dev/null || {
        echo "NVIDIA GPU found but nvcc is missing: sudo apt install nvidia-cuda-toolkit"
        echo "(or TRANSCRIBE_CPU_ONLY=1 scripts/setup.sh for a CPU-only install)"
        exit 1
    }
    CUDA=1
fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$CUDA" = 1 ]; then
    # Before requirements.txt, which would pull the default wheel: it targets the newest CUDA
    # and needs a newer driver than e.g. Debian 13's 550 (CUDA 12.4). cu126 runs on >= 525;
    # 2.11 is the last release with a cu126 torchaudio.
    .venv/bin/python -m pip install "torch==2.11.0" "torchaudio==2.11.0" \
        --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
fi
.venv/bin/python -m pip install -r requirements.txt
if [ "$CUDA" = 1 ]; then
    .venv/bin/python -m pip install -r requirements-cuda.txt
    scripts/build_whisper_cuda.sh
fi
echo "Done. Web UI: scripts/web.sh   CLI: scripts/transcribe.sh <file> [--language ru] [--speakers]"
