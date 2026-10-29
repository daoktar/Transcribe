from unittest.mock import patch

import pytest


@pytest.fixture
def sample_result():
    """A typical transcription result dict."""
    return {
        "text": "Hello world. This is a test.",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "Hello world."},
            {"start": 1.5, "end": 3.2, "text": "This is a test."},
        ],
        "language": "en",
    }


@pytest.fixture
def empty_result():
    """A transcription result with no segments."""
    return {
        "text": "",
        "segments": [],
        "language": "unknown",
    }


@pytest.fixture(autouse=True)
def _no_gpu_probing():
    """Keep tests off the real GPU and the same on every machine.

    No VRAM probe (so no GPU lock), whisper names as given, and neither Qwen backend probe —
    so ``backend()`` is None on every machine (a Mac with mlx-audio installed included), as
    in CI. Tests of those helpers restore the import-time originals. ``backend()`` is
    memoized, so its cache is reset around every test.
    """
    from transcribe import qwen_engine

    qwen_engine.backend.cache_clear()
    with patch("transcribe.core._whisper_model_for_device", side_effect=lambda size: size), \
         patch("transcribe.gpu.cuda_total_gb", return_value=None), \
         patch("transcribe.qwen_engine._cuda_available", return_value=False), \
         patch("transcribe.qwen_engine._mlx_available", return_value=False):
        yield
    qwen_engine.backend.cache_clear()
