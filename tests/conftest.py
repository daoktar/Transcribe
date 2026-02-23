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
