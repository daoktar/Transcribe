#!/usr/bin/env bash
# Rebuild pywhispercpp in .venv with whisper.cpp's CUDA backend (the PyPI wheel is CPU-only
# on Linux). Needs nvcc (Debian/Ubuntu: sudo apt install nvidia-cuda-toolkit), the CUDA
# torch from scripts/setup.sh and patchelf from requirements-cuda.txt. Takes ~15 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo "nvcc not found: sudo apt install nvidia-cuda-toolkit"; exit 1; }
PY=.venv/bin/python
VERSION="$("$PY" -c 'import importlib.metadata as m; print(m.version("pywhispercpp"))')"
# Compute capability of the first GPU (8.6 -> 86): building for one arch is much faster.
ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}"

GGML_CUDA=1 CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$ARCH" \
    "$PY" -m pip install --no-binary pywhispercpp --no-deps --force-reinstall --no-cache-dir \
    "pywhispercpp==$VERSION"

# The wheel installs its libraries as copies whose RUNPATH points into pip's deleted build
# dir, so `import pywhispercpp` fails with "libwhisper.so.1: cannot open shared object file".
# libggml-cuda must also load the CUDA runtime and cuBLAS that ship with torch rather than
# the toolkit's: whichever libcudart.so.12 loads first wins for the whole process, and torch
# fails to import next to the toolkit's older one (whisper then segfaults on allocation).
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for lib in "$SITE"/_pywhispercpp*.so "$SITE"/libwhisper.so* "$SITE"/libggml*.so*; do
    case "$(basename "$lib")" in
        libggml-cuda.so*) rpath='$ORIGIN:$ORIGIN/nvidia/cuda_runtime/lib:$ORIGIN/nvidia/cublas/lib' ;;
        *) rpath='$ORIGIN' ;;
    esac
    .venv/bin/patchelf --set-rpath "$rpath" "$lib"
done

"$PY" - <<'PY'
import pywhispercpp.model  # noqa: F401  (loads the patched libraries)
import _pywhispercpp
import torch

assert "CUDA" in _pywhispercpp.whisper_print_system_info(), "whisper.cpp built without CUDA"
assert torch.cuda.is_available(), "torch cannot use the GPU next to whisper.cpp"
print("whisper.cpp CUDA build OK:", torch.cuda.get_device_name(0))
PY
