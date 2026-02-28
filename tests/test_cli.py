import sys
from unittest.mock import patch

import pytest

from transcribe.cli import main


class TestCliArgParsing:
    def test_missing_video_exits(self, capsys):
        """CLI should exit 1 and print error for nonexistent file."""
        with patch("sys.argv", ["cli", "/nonexistent/video.mp4"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "file not found" in captured.err.lower()

    def test_no_args_exits(self):
        """CLI should exit 2 (argparse error) when no arguments provided."""
        with patch("sys.argv", ["cli"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_invalid_model_exits(self):
        """CLI should reject invalid model names."""
        with patch("sys.argv", ["cli", "video.mp4", "--model", "invalid"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_successful_transcription(self, tmp_path, capsys):
        """CLI should call transcribe_video and save a .txt file."""
        video = tmp_path / "video.mp4"
        video.touch()

        fake_result = {
            "text": "Hello world.",
            "segments": [{"start": 0.0, "end": 1.5, "text": "Hello world."}],
            "language": "en",
        }

        with patch("sys.argv", [
            "cli", str(video), "--model", "tiny", "--language", "en",
        ]):
            with patch("transcribe.cli.transcribe_media", return_value=fake_result):
                main()

        captured = capsys.readouterr()
        assert "Detected language: en" in captured.out
        assert "Segments: 1" in captured.out
        saved_lines = [l for l in captured.out.splitlines() if l.startswith("Saved:")]
        assert len(saved_lines) == 1
        assert saved_lines[0].endswith(".txt")

    def test_custom_output_dir(self, tmp_path, capsys):
        """--output-dir should save files to the specified directory."""
        video = tmp_path / "video.mp4"
        video.touch()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        fake_result = {
            "text": "Test.",
            "segments": [],
            "language": "en",
        }

        with patch("sys.argv", [
            "cli", str(video), "--model", "tiny", "--language", "en",
            "--output-dir", str(out_dir),
        ]):
            with patch("transcribe.cli.transcribe_media", return_value=fake_result) as mock_tv:
                main()

        # Verify output_dir was passed to transcribe_video
        call_kwargs = mock_tv.call_args
        assert call_kwargs[1]["output_dir"] == str(out_dir) or call_kwargs[0][3] == str(out_dir)
