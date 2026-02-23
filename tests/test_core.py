import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from transcribe.core import (
    _detect_language,
    _format_eta,
    _get_media_duration,
    save_txt,
    transcribe_video,
)


# ---------------------------------------------------------------------------
# _format_eta
# ---------------------------------------------------------------------------

class TestFormatEta:
    def test_seconds(self):
        assert _format_eta(0) == "0s"
        assert _format_eta(5) == "5s"
        assert _format_eta(59) == "59s"

    def test_minutes(self):
        assert _format_eta(60) == "1m 00s"
        assert _format_eta(90) == "1m 30s"
        assert _format_eta(3599) == "59m 59s"

    def test_hours(self):
        assert _format_eta(3600) == "1h 00m"
        assert _format_eta(3661) == "1h 01m"
        assert _format_eta(7200) == "2h 00m"

    def test_fractional_seconds(self):
        assert _format_eta(0.7) == "0s"
        assert _format_eta(5.9) == "5s"


# ---------------------------------------------------------------------------
# _get_media_duration
# ---------------------------------------------------------------------------

class TestGetMediaDuration:
    def test_success(self):
        mock_result = MagicMock()
        mock_result.stdout = "123.456\n"
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            assert _get_media_duration("test.mp4") == 123.456

    def test_empty_output(self):
        mock_result = MagicMock()
        mock_result.stdout = "\n"
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            assert _get_media_duration("test.mp4") is None

    def test_negative_duration(self):
        mock_result = MagicMock()
        mock_result.stdout = "-1.0\n"
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            assert _get_media_duration("test.mp4") is None

    def test_zero_duration(self):
        mock_result = MagicMock()
        mock_result.stdout = "0.0\n"
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            assert _get_media_duration("test.mp4") is None

    def test_timeout(self):
        with patch(
            "transcribe.core.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
        ):
            assert _get_media_duration("test.mp4") is None

    def test_non_numeric(self):
        mock_result = MagicMock()
        mock_result.stdout = "N/A\n"
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            assert _get_media_duration("test.mp4") is None


# ---------------------------------------------------------------------------
# _detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_success(self):
        model = MagicMock()
        model.auto_detect_language.return_value = (("en", 0.95), {"en": 0.95, "fr": 0.03})
        lang, err = _detect_language(model, "test.wav")
        assert lang == "en"
        assert err is None

    def test_failure_returns_unknown(self):
        model = MagicMock()
        model.auto_detect_language.side_effect = RuntimeError("model error")
        lang, err = _detect_language(model, "test.wav")
        assert lang == "unknown"
        assert "RuntimeError" in err
        assert "model error" in err


# ---------------------------------------------------------------------------
# save_txt
# ---------------------------------------------------------------------------

class TestSaveTxt:
    def test_writes_text(self, tmp_path, sample_result):
        out = save_txt(sample_result, tmp_path / "out.txt")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert content == "Hello world. This is a test."

    def test_empty_result(self, tmp_path, empty_result):
        out = save_txt(empty_result, tmp_path / "out.txt")
        assert out.read_text(encoding="utf-8") == ""

    def test_returns_path_object(self, tmp_path, sample_result):
        out = save_txt(sample_result, str(tmp_path / "out.txt"))
        assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# transcribe_video — validation & edge cases (mocked Model)
# ---------------------------------------------------------------------------

def _make_segment(t0, t1, text):
    """Create a mock pywhispercpp segment (centisecond timestamps)."""
    return SimpleNamespace(t0=t0, t1=t1, text=text)


class TestTranscribeVideoValidation:
    def test_missing_ffmpeg(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        with patch("transcribe.core.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                transcribe_video(str(video))

    def test_missing_video_file(self):
        with patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"):
            with pytest.raises(FileNotFoundError, match="Video file not found"):
                transcribe_video("/nonexistent/video.mp4")

    def test_output_dir_created(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        out_dir = tmp_path / "nested" / "output"

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock()

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
        ):
            transcribe_video(str(video), output_dir=str(out_dir), language="en")

        assert out_dir.exists()

    def test_progress_callback_stages(self, tmp_path):
        """Progress callback should be called with loading, transcribing, and done stages."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock()

        progress_calls = []

        def track_progress(fraction, msg):
            progress_calls.append((fraction, msg))

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
        ):
            transcribe_video(
                str(video), language="en", progress_callback=track_progress
            )

        # Should have at least: loading, model loaded, language, done
        assert len(progress_calls) >= 3
        assert progress_calls[0][0] == 0.0  # loading starts at 0
        assert "Loading model" in progress_calls[0][1]
        assert progress_calls[-1][0] == 1.0  # done at 100%
        assert "Done" in progress_calls[-1][1]

    def test_return_dict_shape(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock()

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
        ):
            result = transcribe_video(str(video), language="en")

        assert "text" in result
        assert "segments" in result
        assert "language" in result
        assert isinstance(result["segments"], list)
        assert result["language"] == "en"


class TestTranscribeVideoSegmentProcessing:
    """Test the new_segment_callback logic by capturing and invoking it."""

    def _run_with_segments(self, tmp_path, fake_segments, duration=10.0):
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        captured_callback = {}

        def fake_transcribe(path, new_segment_callback=None, **kwargs):
            captured_callback["cb"] = new_segment_callback
            if new_segment_callback:
                for seg in fake_segments:
                    new_segment_callback(seg)

        mock_model.transcribe = fake_transcribe

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=duration),
        ):
            return transcribe_video(str(video), language="en")

    def test_normal_segments(self, tmp_path):
        segs = [
            _make_segment(0, 150, " Hello world. "),
            _make_segment(150, 320, " This is a test. "),
        ]
        result = self._run_with_segments(tmp_path, segs)
        assert result["text"] == "Hello world. This is a test."
        assert len(result["segments"]) == 2
        assert result["segments"][0] == {"start": 0.0, "end": 1.5, "text": "Hello world."}
        assert result["segments"][1] == {"start": 1.5, "end": 3.2, "text": "This is a test."}

    def test_empty_text_filtered(self, tmp_path):
        segs = [
            _make_segment(0, 150, " Hello. "),
            _make_segment(150, 200, "   "),  # empty after strip
            _make_segment(200, 350, " World. "),
        ]
        result = self._run_with_segments(tmp_path, segs)
        assert len(result["segments"]) == 2
        assert result["text"] == "Hello. World."

    def test_invalid_timestamps_filtered(self, tmp_path):
        segs = [
            _make_segment(0, 150, "Good."),
            _make_segment(200, 200, "Zero length."),  # t0 == t1
            _make_segment(300, 100, "Reversed."),  # t0 > t1
            _make_segment(150, 320, "Also good."),
        ]
        result = self._run_with_segments(tmp_path, segs)
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Good."
        assert result["segments"][1]["text"] == "Also good."

    def test_no_segments(self, tmp_path):
        result = self._run_with_segments(tmp_path, [])
        assert result["text"] == ""
        assert result["segments"] == []

    def test_duration_none_still_works(self, tmp_path):
        """When ffprobe fails, transcription should still work and report segment count."""
        segs = [_make_segment(0, 150, "Hello.")]
        progress_calls = []

        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()

        def fake_transcribe(path, new_segment_callback=None, **kwargs):
            if new_segment_callback:
                for seg in segs:
                    new_segment_callback(seg)

        mock_model.transcribe = fake_transcribe

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=None),
        ):
            result = transcribe_video(
                str(video),
                language="en",
                progress_callback=lambda f, m: progress_calls.append((f, m)),
            )

        assert result["text"] == "Hello."
        # Should have progress messages mentioning "segments processed"
        segment_msgs = [m for _, m in progress_calls if "segments processed" in m]
        assert len(segment_msgs) >= 1

    def test_language_autodetect(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        mock_model.auto_detect_language.return_value = (("fr", 0.9), {"fr": 0.9})
        mock_model.transcribe = MagicMock()

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
        ):
            result = transcribe_video(str(video), language=None)

        assert result["language"] == "fr"
        mock_model.auto_detect_language.assert_called_once()
