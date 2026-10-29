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
def _no_gpu_model_swap():
    """Keep whisper model names as given: the real check imports torch and probes the GPU."""
    with patch("transcribe.core._whisper_model_for_device", side_effect=lambda size: size):
        yield
