"""Tests for transcribe.diarize module."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock torch before importing diarize (torch may not be installed in test env)
# ---------------------------------------------------------------------------

_mock_torch = MagicMock()
# Make torch.from_numpy return a MagicMock that supports .float()
_mock_tensor = MagicMock()
_mock_tensor.float.return_value = _mock_tensor
_mock_torch.from_numpy.return_value = _mock_tensor
sys.modules.setdefault("torch", _mock_torch)

# Mock pyannote.audio so the module can be imported
_mock_pyannote = MagicMock()
sys.modules.setdefault("pyannote", _mock_pyannote)
sys.modules.setdefault("pyannote.audio", _mock_pyannote)


# ---------------------------------------------------------------------------
# Helpers for building mock pyannote Annotation objects
# ---------------------------------------------------------------------------


class FakeSegment:
    """Mimics pyannote.core.Segment with start/end attributes."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


def _make_annotation(turns: list[tuple[float, float, str]]):
    """Build a mock DiarizeOutput (pyannote v4) wrapping an Annotation."""
    annotation = MagicMock()
    annotation.itertracks.return_value = [
        (FakeSegment(s, e), None, spk) for s, e, spk in turns
    ]
    # pyannote.audio v4 returns DiarizeOutput with .speaker_diarization
    return SimpleNamespace(speaker_diarization=annotation)


# ---------------------------------------------------------------------------
# TestLoadPipeline
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    """Tests for _load_pipeline.

    Pipeline is imported lazily inside the function from pyannote.audio,
    which is mocked via sys.modules at the top of this file.  We access
    the mock's Pipeline attribute directly via _mock_pyannote.Pipeline.
    """

    def setup_method(self):
        import transcribe.diarize as mod
        mod._pipeline = None
        mod._pipeline_token_hash = None
        _mock_pyannote.Pipeline.from_pretrained.reset_mock()
        _mock_pyannote.Pipeline.from_pretrained.return_value = MagicMock()

    def teardown_method(self):
        import transcribe.diarize as mod
        mod._pipeline = None
        mod._pipeline_token_hash = None

    def test_caches_after_first_call(self):
        """Pipeline.from_pretrained should only be called once for same token."""
        import transcribe.diarize as mod

        mod._load_pipeline("tok-1")
        mod._load_pipeline("tok-1")

        assert _mock_pyannote.Pipeline.from_pretrained.call_count == 1

    def test_reloads_on_token_change(self):
        """Pipeline should reload when a different token is provided."""
        import transcribe.diarize as mod

        mod._load_pipeline("tok-1")
        mod._load_pipeline("tok-2")

        assert _mock_pyannote.Pipeline.from_pretrained.call_count == 2

    def test_raw_token_never_stored(self):
        """The raw token should never be stored — only a hash."""
        import transcribe.diarize as mod

        secret = "hf_supersecrettoken12345"
        mod._load_pipeline(secret)

        # The raw token must NOT appear in any module-level variable
        assert mod._pipeline_token_hash is not None
        assert mod._pipeline_token_hash != secret
        assert secret not in str(mod._pipeline_token_hash)

    def test_resets_on_failure(self):
        """Pipeline and token hash should be reset when from_pretrained fails."""
        import transcribe.diarize as mod

        _mock_pyannote.Pipeline.from_pretrained.side_effect = Exception("auth error")

        with pytest.raises(RuntimeError, match="Failed to load"):
            mod._load_pipeline("bad-token")

        assert mod._pipeline is None
        assert mod._pipeline_token_hash is None

        # Clean up: restore normal side_effect for subsequent tests
        _mock_pyannote.Pipeline.from_pretrained.side_effect = None
        _mock_pyannote.Pipeline.from_pretrained.return_value = MagicMock()

    def test_retries_after_failure_with_new_token(self):
        """After a failure, a new token should trigger a fresh load attempt."""
        import transcribe.diarize as mod

        # First call fails
        _mock_pyannote.Pipeline.from_pretrained.side_effect = Exception("bad")
        with pytest.raises(RuntimeError):
            mod._load_pipeline("bad-token")

        # Second call with new token succeeds
        _mock_pyannote.Pipeline.from_pretrained.side_effect = None
        _mock_pyannote.Pipeline.from_pretrained.return_value = MagicMock()

        result = mod._load_pipeline("good-token")
        assert result is not None
        assert mod._pipeline is not None
        assert mod._pipeline_token_hash is not None


# ---------------------------------------------------------------------------
# TestDiarize
# ---------------------------------------------------------------------------


class TestDiarize:
    def _make_audio(self, duration_s: float = 10.0) -> np.ndarray:
        return np.zeros(int(16000 * duration_s), dtype=np.float32)

    @patch("transcribe.diarize._load_pipeline")
    def test_empty_segments_returns_empty(self, mock_load):
        from transcribe.diarize import diarize

        result, count = diarize(self._make_audio(), [], "tok")
        assert result == []
        assert count == 0
        mock_load.assert_not_called()

    @patch("transcribe.diarize._load_pipeline")
    def test_two_speakers_assigned(self, mock_load):
        """Segments should be labeled by overlapping speaker turns."""
        from transcribe.diarize import diarize

        annotation = _make_annotation([
            (0.0, 2.0, "SPEAKER_00"),
            (2.0, 5.0, "SPEAKER_01"),
        ])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello."},
            {"start": 2.5, "end": 4.0, "text": "Hi there."},
        ]
        result, count = diarize(self._make_audio(), segments, "tok")

        assert count == 2
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"
        # Originals not mutated
        assert "speaker" not in segments[0]

    @patch("transcribe.diarize._load_pipeline")
    def test_single_speaker(self, mock_load):
        from transcribe.diarize import diarize

        annotation = _make_annotation([
            (0.0, 5.0, "SPEAKER_00"),
        ])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello."},
            {"start": 2.0, "end": 4.0, "text": "World."},
        ]
        result, count = diarize(self._make_audio(), segments, "tok")

        assert count == 1
        assert all(s["speaker"] == "Speaker 1" for s in result)

    @patch("transcribe.diarize._load_pipeline")
    def test_speaker_labels_ordered_by_first_appearance(self, mock_load):
        """Speaker 1 should always be the first speaker heard chronologically."""
        from transcribe.diarize import diarize

        # SPEAKER_01 appears first in time despite higher ID
        annotation = _make_annotation([
            (0.0, 2.0, "SPEAKER_01"),
            (2.0, 5.0, "SPEAKER_00"),
        ])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        segments = [
            {"start": 0.5, "end": 1.5, "text": "First."},
            {"start": 2.5, "end": 4.0, "text": "Second."},
        ]
        result, _ = diarize(self._make_audio(), segments, "tok")

        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"

    @patch("transcribe.diarize._load_pipeline")
    def test_majority_vote_overlap(self, mock_load):
        """When a segment overlaps two speakers, the one with more overlap wins."""
        from transcribe.diarize import diarize

        annotation = _make_annotation([
            (0.0, 1.0, "SPEAKER_00"),   # 0.5s overlap with segment
            (1.0, 5.0, "SPEAKER_01"),   # 1.5s overlap with segment
        ])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        segments = [
            {"start": 0.5, "end": 2.5, "text": "Overlapping."},
        ]
        result, _ = diarize(self._make_audio(), segments, "tok")

        # SPEAKER_01 has more overlap (1.5s vs 0.5s)
        assert result[0]["speaker"] == "Speaker 2"

    @patch("transcribe.diarize._load_pipeline")
    def test_no_overlap_nearest_window(self, mock_load):
        """Segment with no overlapping turns uses nearest turn."""
        from transcribe.diarize import diarize

        annotation = _make_annotation([
            (0.0, 1.0, "SPEAKER_00"),
            (5.0, 6.0, "SPEAKER_01"),
        ])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        # Segment at 2.0-3.0 has no overlap; nearest is SPEAKER_00 (center 0.5 vs 5.5)
        segments = [
            {"start": 2.0, "end": 3.0, "text": "Gap segment."},
        ]
        result, _ = diarize(self._make_audio(), segments, "tok")

        assert result[0]["speaker"] == "Speaker 1"

    @patch("transcribe.diarize._load_pipeline")
    def test_no_speaker_turns_detected(self, mock_load):
        """When pyannote returns no turns, all segments get Speaker 1."""
        from transcribe.diarize import diarize

        annotation = _make_annotation([])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello."},
        ]
        result, count = diarize(self._make_audio(), segments, "tok")

        assert count == 1
        assert result[0]["speaker"] == "Speaker 1"

    @patch("transcribe.diarize._load_pipeline")
    def test_num_speakers_passed_to_pipeline(self, mock_load):
        """num_speakers kwarg should be forwarded to the pipeline call."""
        from transcribe.diarize import diarize

        annotation = _make_annotation([(0.0, 5.0, "SPEAKER_00")])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello."}]
        diarize(self._make_audio(), segments, "tok", num_speakers=3)

        # Verify num_speakers was passed to the pipeline call
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["num_speakers"] == 3

    @patch("transcribe.diarize._load_pipeline")
    def test_progress_callback_called(self, mock_load):
        """Progress callback should receive increasing fractions."""
        from transcribe.diarize import diarize

        annotation = _make_annotation([(0.0, 5.0, "SPEAKER_00")])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation
        mock_load.return_value = mock_pipeline

        fractions: list[float] = []
        messages: list[str] = []

        def on_progress(frac, msg):
            fractions.append(frac)
            messages.append(msg)

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello."}]
        diarize(self._make_audio(), segments, "tok", progress_callback=on_progress)

        assert len(fractions) >= 2
        assert fractions[0] == 0.0
        assert fractions[-1] == 1.0
        # Fractions should be non-decreasing
        assert all(a <= b for a, b in zip(fractions, fractions[1:]))
