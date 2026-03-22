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

    def test_speakers_flag_with_env_token(self, tmp_path, capsys):
        """--speakers flag should pass diarize=True using HF_TOKEN env var."""
        video = tmp_path / "video.mp4"
        video.touch()

        fake_result = {
            "text": "Hello.",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello.", "speaker": "Speaker 1"}],
            "language": "en",
            "speakers": 1,
        }

        with patch("sys.argv", ["cli", str(video), "--speakers"]):
            with patch.dict("os.environ", {"HF_TOKEN": "env-tok-123"}):
                with patch("transcribe.cli.transcribe_media", return_value=fake_result) as mock_tm:
                    main()

        call_kwargs = mock_tm.call_args[1]
        assert call_kwargs["diarize"] is True
        assert call_kwargs["hf_token"] == "env-tok-123"

        captured = capsys.readouterr()
        assert "Speakers: 1" in captured.out

    def test_num_speakers_flag(self, tmp_path, capsys):
        """--num-speakers should be forwarded to transcribe_media."""
        video = tmp_path / "video.mp4"
        video.touch()

        fake_result = {
            "text": "Hello.",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello."}],
            "language": "en",
        }

        with patch("sys.argv", [
            "cli", str(video), "--speakers", "--num-speakers", "3",
        ]):
            with patch.dict("os.environ", {"HF_TOKEN": "env-tok-123"}):
                with patch("transcribe.cli.transcribe_media", return_value=fake_result) as mock_tm:
                    main()

        call_kwargs = mock_tm.call_args[1]
        assert call_kwargs["num_speakers"] == 3

    def test_speakers_without_token_exits(self, tmp_path, capsys):
        """--speakers without HF_TOKEN env var should exit with error."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("sys.argv", ["cli", str(video), "--speakers"]):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "huggingface token" in captured.err.lower()
        # Ensure error message does NOT mention --hf-token CLI flag
        assert "--hf-token" not in captured.err

    def test_multiple_files(self, tmp_path, capsys):
        """CLI should accept and process multiple files."""
        v1 = tmp_path / "video1.mp4"
        v2 = tmp_path / "video2.mp4"
        v1.touch()
        v2.touch()

        fake_result = {
            "text": "Hello.",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello."}],
            "language": "en",
        }

        with patch("sys.argv", ["cli", str(v1), str(v2), "--model", "tiny"]):
            with patch("transcribe.cli.transcribe_media", return_value=fake_result) as mock_tm:
                main()

        assert mock_tm.call_count == 2
        captured = capsys.readouterr()
        assert "[1/2]" in captured.out
        assert "[2/2]" in captured.out
        assert "Batch complete" in captured.out

    def test_multiple_files_one_missing(self, tmp_path, capsys):
        """CLI should exit 1 if any file is missing (validates all up front)."""
        v1 = tmp_path / "video1.mp4"
        v1.touch()

        with patch("sys.argv", ["cli", str(v1), "/nonexistent/video2.mp4"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "file not found" in captured.err.lower()
