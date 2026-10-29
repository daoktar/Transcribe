"""NVIDIA GPU helpers shared by the engines: total VRAM and one-stage-at-a-time access.

A 4 GB card holds one engine's models at a time, and whisper.cpp segfaults instead of raising
when it cannot allocate, so GPU work from concurrent jobs (the web server runs each job in its
own thread) is serialized with :func:`exclusive`: one stage at a time (a transcription, a
diarization), so two jobs may take turns between stages. Apple Silicon (Metal/MLX, unified memory) is
left as it was: there the lock is a no-op.
"""
from __future__ import annotations

import contextlib
import functools
import gc
import importlib
import os
import sys
import threading
import traceback

# Reentrant so a locked stage could call another locked stage without deadlocking.
_LOCK = threading.RLock()


@functools.cache
def cuda_total_gb() -> float | None:
    """Total memory of CUDA device 0 in GiB, or None without a CUDA-enabled torch.

    Cards report a little less than their nominal size (an "8 GB" card ~7.6-8.0 GiB, the
    4 GB RTX 3050 Laptop 3.6 GiB), so thresholds compared against this need a margin.
    """
    # Answer without importing torch (seconds, hundreds of MB) where CUDA cannot exist: macOS,
    # and Linux with neither the NVIDIA kernel driver nor WSL2's GPU bridge (/dev/dxg; WSL2
    # has no /proc/driver/nvidia, its CUDA comes from the Windows driver).
    if sys.platform == "darwin" or (
        sys.platform.startswith("linux")
        and not os.path.exists("/proc/driver/nvidia/version")
        and not os.path.exists("/dev/dxg")
    ):
        return None
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        return None


def exclusive():
    """Context manager that holds the process-wide GPU lock while CUDA is in use."""
    return _LOCK if cuda_total_gb() is not None else contextlib.nullcontext()


def exclusive_call(func):
    """Decorator: run ``func`` under :func:`exclusive`.

    When it raises, the traceback's frames still reference the stage's models (GBs of VRAM)
    until the caller drops the exception; clear them, still holding the lock, so the next
    stage does not load beside them.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with exclusive():
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                traceback.clear_frames(exc.__traceback__)
                gc.collect()
                raise
    return wrapper
