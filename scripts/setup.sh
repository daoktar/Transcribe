#!/usr/bin/env bash
# Create .venv in the project root and install requirements (needs Python 3.10-3.14 and ffmpeg).
# On Linux with an NVIDIA GPU it also sets up the GPU path: CUDA torch, whisper.cpp built
# with CUDA and the Qwen3-ASR CUDA backend (needs nvcc; TRANSCRIBE_CPU_ONLY=1 skips it).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "ffmpeg not found: brew install ffmpeg (macOS) / apt install ffmpeg"; exit 1; }
# Homebrew's python3 is 3.15, which torchcodec (needed by pyannote) has no wheels for, so pip
# fails. Without $PYTHON take the first interpreter in 3.10-3.14.
# ponytail: raise the 14 once torchcodec ships cp315 wheels.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
        "$cand" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))' \
            2>/dev/null && { PY="$cand"; break; }
    done
    [ -n "$PY" ] || { echo "Python 3.10-3.14 not found (3.15 lacks torchcodec wheels): brew install python@3.14"; exit 1; }
fi
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
    # 2.11 is the last release with a cu126 torchaudio. TORCH_CUDA picks another build (cu128
    # for RTX 50xx, which cu126 lacks kernels for); TORCH_INDEX_URL a mirror serving it. The
    # +label makes pip replace a plain 2.11.0 already in the venv (a bare ==2.11.0 would count
    # it as satisfied).
    TORCH_CUDA="${TORCH_CUDA:-cu126}"
    .venv/bin/python -m pip install "torch==2.11.0+$TORCH_CUDA" "torchaudio==2.11.0+$TORCH_CUDA" \
        --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$TORCH_CUDA}"
fi
.venv/bin/python -m pip install -r requirements.txt
if [ "$CUDA" = 1 ]; then
    .venv/bin/python -m pip install -r requirements-cuda.txt
    scripts/build_whisper_cuda.sh
fi
echo "Done. Web UI: scripts/web.sh   CLI: scripts/transcribe.sh <file> [--language ru] [--speakers]"
