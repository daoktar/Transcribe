import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcribe.core import (
    _deduplicate_segments,
    _detect_language,
    _detect_speech_regions,
    _extract_audio_pcm,
    _format_eta,
    _get_media_duration,
    _language_windows,
    _load_screen_model,
    _merge_speech_regions,
    retry_diarize,
    save_txt,
    save_txt_alongside,
    transcribe_media as transcribe_video,
)


@pytest.fixture(autouse=True)
def _no_screen_model():
    """No test loads (or downloads) the real whisper tiny screen; the main mock hears every region."""
    with patch("transcribe.core._load_screen_model", return_value=None):
        yield


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
        assert _detect_language(model, np.zeros(16000, dtype=np.float32)) == "en"

    def test_failure_falls_back_to_whisper_auto_detect(self):
        model = MagicMock()
        model.auto_detect_language.side_effect = RuntimeError("model error")
        assert _detect_language(model, np.zeros(16000, dtype=np.float32)) == ""


# ---------------------------------------------------------------------------
# _language_windows
# ---------------------------------------------------------------------------

def _lid(*langs):
    """auto_detect_language side effect returning *langs* in call order."""
    return [((lang, 0.99), {lang: 0.99}) for lang in langs]


class TestLanguageWindows:
    AUDIO = np.zeros(16000 * 30, dtype=np.float32)

    def test_cuts_chunk_in_the_pause_where_language_switches(self):
        model = MagicMock()
        model.auto_detect_language.side_effect = _lid("ru", "en", "ru")
        regions = [
            (0.0, 1.0),    # too short to identify: joins the first detected language
            (1.2, 9.4),    # ru
            (10.0, 19.4),  # en
            (19.8, 20.8),  # "Ок, го." — too short, joins the en before it
            (21.2, 29.0),  # ru
        ]
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, regions)
        assert windows == [(0.0, 9.7, "ru"), (9.7, 21.0, "en"), (21.0, 30.0, "ru")]
        # One detection per long region, each on that region's audio only
        assert [len(c.args[0]) for c in model.auto_detect_language.call_args_list] == [
            int(9.4 * 16000) - int(1.2 * 16000),
            int(19.4 * 16000) - int(10.0 * 16000),
            int(29.0 * 16000) - int(21.2 * 16000),
        ]

    def test_single_language_chunk_stays_whole(self):
        model = MagicMock()
        model.auto_detect_language.side_effect = _lid("ru", "ru")
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, [(0.5, 12.0), (12.6, 29.5)])
        assert windows == [(0.0, 30.0, "ru")]

    def test_fewer_than_two_identifiable_regions_detects_the_whole_chunk(self):
        model = MagicMock()
        model.auto_detect_language.side_effect = _lid("ru")
        windows = _language_windows(model, self.AUDIO, 0.0, 12.0, [(0.5, 9.0), (9.5, 10.5)])
        assert windows == [(0.0, 12.0, "ru")]
        assert len(model.auto_detect_language.call_args.args[0]) == 12 * 16000

    def test_regions_are_clipped_to_the_chunk(self):
        """A region straddling the chunk edge only counts with its part inside the chunk."""
        model = MagicMock()
        model.auto_detect_language.side_effect = _lid("ru")
        # (28.0, 31.0) has only 1 s inside (0, 29): not identifiable, so one whole-chunk pass
        windows = _language_windows(model, self.AUDIO, 0.0, 29.0, [(0.5, 20.0), (28.0, 31.0)])
        assert windows == [(0.0, 29.0, "ru")]
        model.auto_detect_language.assert_called_once()

    def test_detection_failure_leaves_it_to_whisper(self):
        model = MagicMock()
        model.auto_detect_language.side_effect = RuntimeError("model error")
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, [(0.5, 12.0), (12.6, 29.5)])
        assert windows == [(0.0, 30.0, "")]

    REGIONS = [(0.5, 9.0), (9.5, 19.0), (19.5, 29.5)]

    def test_screen_hearing_one_language_identifies_the_chunk_whole(self):
        model, screen = MagicMock(), MagicMock()
        screen.auto_detect_language.side_effect = _lid("ru", "ru", "ru")
        model.auto_detect_language.side_effect = _lid("ru")
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, self.REGIONS, screen)
        assert windows == [(0.0, 30.0, "ru")]
        # The main model's only pass is the whole chunk, the detection whisper.cpp runs itself
        assert len(model.auto_detect_language.call_args.args[0]) == 30 * 16000
        model.auto_detect_language.assert_called_once()

    def test_whole_chunk_id_stands_only_where_the_screen_agrees(self):
        """English chunk ending in Russian that VAD merged into the last region: the whole-chunk
        ID says ru, and alone it would translate the English. The regions decide instead."""
        model, screen = MagicMock(), MagicMock()
        screen.auto_detect_language.side_effect = _lid("en", "en", "en")
        model.auto_detect_language.side_effect = _lid("ru", "en", "en", "en")
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, self.REGIONS, screen)
        assert windows == [(0.0, 30.0, "en")]
        assert model.auto_detect_language.call_count == 4  # whole chunk, then each region

    def test_whole_chunk_id_failure_falls_back_to_the_regions(self):
        model, screen = MagicMock(), MagicMock()
        screen.auto_detect_language.side_effect = _lid("ru", "ru", "ru")
        model.auto_detect_language.side_effect = [RuntimeError("model error"), *_lid("ru", "ru", "ru")]
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, self.REGIONS, screen)
        assert windows == [(0.0, 30.0, "ru")]

    def test_main_model_decides_where_the_screen_hears_a_switch(self):
        """tiny mishears Russian as Bulgarian: that costs the per-region passes, never a window."""
        model, screen = MagicMock(), MagicMock()
        screen.auto_detect_language.side_effect = _lid("ru", "bg", "ru")
        model.auto_detect_language.side_effect = _lid("ru", "ru", "ru")
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, self.REGIONS, screen)
        assert windows == [(0.0, 30.0, "ru")]
        assert model.auto_detect_language.call_count == 3

    def test_screen_failure_counts_as_a_switch(self):
        model, screen = MagicMock(), MagicMock()
        screen.auto_detect_language.side_effect = RuntimeError("model error")
        model.auto_detect_language.side_effect = _lid("ru", "en", "ru")
        windows = _language_windows(model, self.AUDIO, 0.0, 30.0, self.REGIONS, screen)
        assert windows == [(0.0, 9.25, "ru"), (9.25, 19.25, "en"), (19.25, 30.0, "ru")]


# ---------------------------------------------------------------------------
# _load_screen_model
# ---------------------------------------------------------------------------

COMPLETE_MODEL = b"lmgg" + bytes(16)  # as long as the patched _SCREEN_MODEL_BYTES


def _download_writing(content, dirs=None):
    """download_model side effect: writes *content* into download_dir like pywhispercpp does."""
    def download(name, download_dir=None):
        if dirs is not None:
            dirs.append(Path(download_dir))
        path = Path(download_dir) / f"ggml-{name}.bin"
        path.write_bytes(content)
        return str(path)
    return download


@patch("transcribe.core._SCREEN_MODEL_BYTES", len(COMPLETE_MODEL))
class TestLoadScreenModel:
    @pytest.fixture(autouse=True)
    def models_dir(self, tmp_path):
        with patch("transcribe.core.MODELS_DIR", str(tmp_path)):
            yield tmp_path

    def test_uses_a_complete_cached_model(self, models_dir):
        (models_dir / "ggml-tiny.bin").write_bytes(COMPLETE_MODEL)
        with (
            patch("transcribe.core.download_model") as download,
            patch("transcribe.core.Model") as model_cls,
        ):
            assert _load_screen_model() is model_cls.return_value
        download.assert_not_called()
        model_cls.assert_called_once_with(str(models_dir / "ggml-tiny.bin"))

    def test_downloads_beside_the_model_path_and_moves_it_in_whole(self, models_dir):
        dirs, loads = [], []
        path = models_dir / "ggml-tiny.bin"
        with (
            patch("transcribe.core.download_model", side_effect=_download_writing(COMPLETE_MODEL, dirs)),
            patch("transcribe.core.Model", side_effect=lambda p: loads.append((p, path.exists())) or "model"),
        ):
            assert _load_screen_model() == "model"
        assert dirs[0].parent == models_dir  # same filesystem, so the move is atomic
        assert dirs[0].name.startswith("ggml-tiny-download-")  # so a killed one is found and removed
        assert list(models_dir.iterdir()) == [path]  # temp dir gone
        assert path.read_bytes() == COMPLETE_MODEL
        # Loaded from the private copy before it is moved: a --model tiny job may truncate the
        # shared path at any moment (its in-place download opens it with "wb").
        assert loads == [(str(dirs[0] / "ggml-tiny.bin"), False)]

    def test_removes_only_downloads_abandoned_for_an_hour(self, models_dir):
        (models_dir / "ggml-tiny.bin").write_bytes(COMPLETE_MODEL)
        dirs = {}
        for name in ("ggml-tiny-download-killed", "ggml-tiny-download-live", "tmp-not-ours"):
            (models_dir / name).mkdir()
            (models_dir / name / "ggml-tiny.bin").write_bytes(COMPLETE_MODEL[:12])
            dirs[name] = models_dir / name
        hours_ago = time.time() - 2 * 3600
        for d in (dirs["ggml-tiny-download-killed"], dirs["tmp-not-ours"]):
            os.utime(d / "ggml-tiny.bin", (hours_ago, hours_ago))
            os.utime(d, (hours_ago, hours_ago))
        os.utime(dirs["ggml-tiny-download-live"], (hours_ago, hours_ago))  # its file is still being written
        with patch("transcribe.core.Model"):
            _load_screen_model()
        assert sorted(p.name for p in models_dir.iterdir()) == [
            "ggml-tiny-download-live", "ggml-tiny.bin", "tmp-not-ours",
        ]

    def test_replaces_a_download_cut_short(self, models_dir):
        """A truncated file keeps the ggml magic; whisper.cpp "loads" it and segfaults on use."""
        (models_dir / "ggml-tiny.bin").write_bytes(COMPLETE_MODEL[:12])
        with (
            patch("transcribe.core.download_model", side_effect=_download_writing(COMPLETE_MODEL)),
            patch("transcribe.core.Model") as model_cls,
        ):
            assert _load_screen_model() is model_cls.return_value
        assert (models_dir / "ggml-tiny.bin").read_bytes() == COMPLETE_MODEL

    @pytest.mark.parametrize(
        "content", [COMPLETE_MODEL[:12], b"<html>429 Too Many Requests</html>"], ids=["cut-short", "error-page"],
    )
    def test_bad_download_never_reaches_the_model_path(self, models_dir, content):
        """download_model keeps a download cut short and saves an HTTP error page as the model."""
        with (
            patch("transcribe.core.download_model", side_effect=_download_writing(content)),
            patch("transcribe.core.Model") as model_cls,
        ):
            assert _load_screen_model() is None
        model_cls.assert_not_called()
        assert list(models_dir.iterdir()) == []

    def test_offline_leaves_every_region_to_the_main_model(self, models_dir):
        partial = models_dir / "ggml-tiny.bin"
        partial.write_bytes(COMPLETE_MODEL[:12])  # maybe a --model tiny job is still writing it
        with (
            patch("transcribe.core.download_model", side_effect=ConnectionError("offline")),
            patch("transcribe.core.Model") as model_cls,
        ):
            assert _load_screen_model() is None
        model_cls.assert_not_called()
        assert partial.read_bytes() == COMPLETE_MODEL[:12]  # never deleted under a writer


# ---------------------------------------------------------------------------
# _deduplicate_segments
# ---------------------------------------------------------------------------

class TestDeduplicateSegments:
    def test_empty_list(self):
        assert _deduplicate_segments([]) == []

    def test_no_duplicates(self):
        segs = [
            {"start": 0, "end": 1, "text": "Hello."},
            {"start": 1, "end": 2, "text": "World."},
            {"start": 2, "end": 3, "text": "Goodbye."},
        ]
        result = _deduplicate_segments(segs)
        assert result == segs

    def test_keeps_up_to_max_repeats(self):
        segs = [
            {"start": 0, "end": 1, "text": "Hello."},
            {"start": 1, "end": 2, "text": "Hello."},
            {"start": 2, "end": 3, "text": "Hello."},
        ]
        # default max_repeats=2, so keeps first 2
        result = _deduplicate_segments(segs)
        assert len(result) == 2
        assert result[0]["start"] == 0
        assert result[1]["start"] == 1

    def test_massive_repetition(self):
        """Simulates the hallucination scenario: same phrase repeated 100+ times."""
        real = [
            {"start": 0, "end": 5, "text": "This is a real sentence."},
            {"start": 5, "end": 10, "text": "Another real sentence."},
        ]
        hallucinated = [
            {"start": 10 + i, "end": 11 + i, "text": "Repeated phrase."}
            for i in range(100)
        ]
        segs = real + hallucinated
        result = _deduplicate_segments(segs)
        # 2 real + at most 2 repeated = 4
        assert len(result) == 4
        assert result[0]["text"] == "This is a real sentence."
        assert result[1]["text"] == "Another real sentence."
        assert result[2]["text"] == "Repeated phrase."
        assert result[3]["text"] == "Repeated phrase."

    def test_non_consecutive_duplicates_kept(self):
        """Same text appearing in non-consecutive positions should be kept."""
        segs = [
            {"start": 0, "end": 1, "text": "Hello."},
            {"start": 1, "end": 2, "text": "World."},
            {"start": 2, "end": 3, "text": "Hello."},  # not consecutive with first "Hello."
        ]
        result = _deduplicate_segments(segs)
        assert len(result) == 3

    def test_custom_max_repeats(self):
        segs = [{"start": i, "end": i + 1, "text": "Same."} for i in range(10)]
        result = _deduplicate_segments(segs, max_repeats=1)
        assert len(result) == 1
        result = _deduplicate_segments(segs, max_repeats=5)
        assert len(result) == 5

    def test_multiple_repeated_groups(self):
        """Different phrases repeated in separate groups."""
        segs = [
            {"start": 0, "end": 1, "text": "A"},
            {"start": 1, "end": 2, "text": "A"},
            {"start": 2, "end": 3, "text": "A"},
            {"start": 3, "end": 4, "text": "A"},
            {"start": 4, "end": 5, "text": "B"},
            {"start": 5, "end": 6, "text": "B"},
            {"start": 6, "end": 7, "text": "B"},
            {"start": 7, "end": 8, "text": "C"},
        ]
        result = _deduplicate_segments(segs)
        # 2 A's + 2 B's + 1 C = 5
        assert len(result) == 5
        texts = [s["text"] for s in result]
        assert texts == ["A", "A", "B", "B", "C"]


# ---------------------------------------------------------------------------
# _extract_audio_pcm
# ---------------------------------------------------------------------------

class TestExtractAudioPcm:
    def test_returns_float32_array(self):
        """Mock ffmpeg to return known PCM bytes, verify numpy conversion."""
        pcm_bytes = b'\x00\x00' * 100  # 100 samples of silence
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pcm_bytes
        mock_result.stderr = b""
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            audio = _extract_audio_pcm("test.mp4")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 100
        assert np.all(audio == 0.0)

    def test_nonzero_samples(self):
        """Verify correct int16-to-float32 normalization."""
        import struct
        pcm_bytes = struct.pack('<h', 32767) * 10  # max positive int16
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pcm_bytes
        mock_result.stderr = b""
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            audio = _extract_audio_pcm("test.mp4")
        assert np.allclose(audio, 32767 / 32768.0, atol=1e-4)

    def test_ffmpeg_failure_raises(self):
        """RuntimeError when ffmpeg exits non-zero."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"error decoding"
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                _extract_audio_pcm("test.mp4")

    def test_empty_audio(self):
        """Empty stdout produces empty array."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""
        mock_result.stderr = b""
        with patch("transcribe.core.subprocess.run", return_value=mock_result):
            audio = _extract_audio_pcm("test.mp4")
        assert len(audio) == 0
        assert audio.dtype == np.float32


# ---------------------------------------------------------------------------
# _detect_speech_regions
# ---------------------------------------------------------------------------

class TestDetectSpeechRegions:
    def test_silence_returns_empty(self):
        """Pure silence (VAD always returns False) → no speech regions."""
        audio = np.zeros(16000 * 5, dtype=np.float32)  # 5s silence
        with patch("transcribe.core.webrtcvad.Vad") as MockVad:
            mock_vad = MockVad.return_value
            mock_vad.is_speech.return_value = False
            regions = _detect_speech_regions(audio)
        assert regions == []

    def test_all_speech(self):
        """Continuous speech → one region spanning the audio."""
        audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s
        with patch("transcribe.core.webrtcvad.Vad") as MockVad:
            mock_vad = MockVad.return_value
            mock_vad.is_speech.return_value = True
            regions = _detect_speech_regions(audio)
        assert len(regions) == 1
        assert regions[0][0] == pytest.approx(0.0, abs=0.5)
        assert regions[0][1] == pytest.approx(3.0, abs=0.5)

    def test_speech_silence_speech(self):
        """Two speech regions separated by a long silence gap."""
        # 9 seconds of audio
        audio = np.zeros(16000 * 9, dtype=np.float32)

        # Frame-by-frame: first 1s speech, then 4s silence, then 4s speech
        # At 30ms/frame: 0-33 speech (~1s), 34-166 silence (~4s), 167-299 speech (~4s)
        def is_speech_side_effect(buf, sr):
            idx = is_speech_side_effect.count
            is_speech_side_effect.count += 1
            if idx < 33:
                return True
            elif idx < 167:
                return False
            else:
                return True
        is_speech_side_effect.count = 0

        with patch("transcribe.core.webrtcvad.Vad") as MockVad:
            mock_vad = MockVad.return_value
            mock_vad.is_speech.side_effect = is_speech_side_effect
            regions = _detect_speech_regions(audio)
        assert len(regions) == 2

    def test_empty_audio(self):
        """Empty audio returns no regions."""
        audio = np.array([], dtype=np.float32)
        regions = _detect_speech_regions(audio)
        assert regions == []


# ---------------------------------------------------------------------------
# _merge_speech_regions
# ---------------------------------------------------------------------------

class TestMergeSpeechRegions:
    def test_empty_input(self):
        assert _merge_speech_regions([]) == []

    def test_single_region_with_padding(self):
        result = _merge_speech_regions([(5.0, 10.0)], total_duration=60.0)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(4.5, abs=0.01)  # 5.0 - 0.5 padding
        assert result[0][1] == pytest.approx(10.5, abs=0.01)  # 10.0 + 0.5 padding

    def test_merges_close_regions(self):
        """Regions within max_gap should be merged."""
        regions = [(1.0, 3.0), (5.0, 8.0)]  # gap = 2s < default 5s
        result = _merge_speech_regions(regions, total_duration=60.0)
        assert len(result) == 1

    def test_keeps_distant_regions_separate(self):
        """Regions with gap > max_gap stay separate."""
        regions = [(1.0, 3.0), (20.0, 25.0)]  # gap = 17s > default 5s
        result = _merge_speech_regions(regions, total_duration=60.0)
        assert len(result) == 2

    def test_filters_short_regions(self):
        """Regions shorter than min_duration are removed."""
        regions = [(1.0, 1.3), (5.0, 10.0)]  # first is 0.3s < 1.0s default
        result = _merge_speech_regions(regions, total_duration=60.0)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(4.5, abs=0.01)

    def test_splits_long_regions(self):
        """A single region longer than max_segment is hard-split into touching pieces."""
        regions = [(0.0, 90.0)]  # 90s > 30s default max
        result = _merge_speech_regions(regions, max_segment=30.0, total_duration=90.0)
        assert len(result) == 3
        # Each sub-chunk should be at most ~30s + padding
        for start, end in result:
            assert (end - start) <= 31.5  # 30 + padding tolerance
        # No padding at the cuts: neighbours touch, nothing is transcribed twice
        assert [r[1] for r in result[:-1]] == [r[0] for r in result[1:]]

    def test_split_lands_in_a_pause_without_overlap(self):
        """Reproducer: the old blind cut + padding put "Sounds good." (30.69s) in both chunks."""
        regions = [(1.23, 10.35), (11.34, 21.36), (22.53, 29.7), (30.69, 34.53)]
        result = _merge_speech_regions(regions, total_duration=36.0)
        assert len(result) == 2
        assert result[0][0] == pytest.approx(0.73)
        assert result[-1][1] == pytest.approx(35.03)
        assert result[0][1] <= result[1][0]
        # Both sides of the cut are inside the same pause — never inside speech
        assert any(e <= result[0][1] and result[1][0] <= s for (_, e), (s, _) in zip(regions, regions[1:]))
        # Every utterance lands in exactly one chunk
        for s, e in regions:
            assert sum(a <= s and e <= b for a, b in result) == 1

    def test_split_prefers_longest_pause(self):
        """The cut goes in the longest pause within reach, not the last one."""
        regions = [(0.0, 16.0), (18.0, 27.0), (27.4, 40.0)]
        result = _merge_speech_regions(regions, total_duration=60.0)
        assert len(result) == 2
        assert 16.0 <= result[0][1] <= 18.0

    def test_split_equal_pauses_takes_latest(self):
        """Pauses equal at VAD resolution (float noise aside) -> the latest, for longer pieces."""
        # VAD timestamps from a TTS clip: every pause is 0.90 or 0.93s,
        # yet 30.599999999999998 - 29.67 < 21.27 - 20.34 in floats
        regions = [(1.02, 5.5200000000000005), (6.45, 10.469999999999999),
                   (11.37, 15.509999999999998), (16.439999999999998, 20.34), (21.27, 25.71),
                   (26.61, 29.67), (30.599999999999998, 31.77)]
        result = _merge_speech_regions(regions, total_duration=36.0)
        assert len(result) == 2
        assert 29.67 <= result[0][1] <= result[1][0] <= 30.6

    def test_split_never_cuts_through_short_region(self):
        """A sub-min_duration region inside a stretch (a "да") is speech too: no cut through it."""
        regions = [(0.0, 14.0), (15.5, 16.2), (16.3, 35.0)]
        result = _merge_speech_regions(regions, total_duration=60.0)
        assert len(result) == 2
        assert any(a <= 15.5 and 16.2 <= b for a, b in result)

    def test_long_conversation_never_overlaps(self):
        """Minutes of back-to-back speech: chunks never overlap and hold every region whole."""
        long = [(t, t + 4.0) for t in np.arange(0.0, 180.0, 5.4)]
        short = [(t + 4.4, t + 5.0) for t in np.arange(0.0, 175.0, 5.4)]  # "да" between, 0.4s pauses
        regions = sorted(long + short)
        regions.append((regions[-1][1] + 5.1, regions[-1][1] + 9.0))  # separate stretch
        result = _merge_speech_regions(regions, total_duration=200.0)
        assert len(result) > 6
        for (_, prev_end), (next_start, _) in zip(result, result[1:]):
            assert prev_end <= next_start
        for start, end in result:
            assert end - start <= 31.0  # max_segment + padding on both sides
        for s, e in regions:
            assert sum(a <= s and e <= b for a, b in result) == 1

    def test_padding_clamped_to_zero(self):
        """Padding should not produce negative start times."""
        regions = [(0.2, 5.0)]
        result = _merge_speech_regions(regions, padding=0.5, total_duration=60.0)
        assert result[0][0] == 0.0

    def test_padding_clamped_to_duration(self):
        """Padding should not exceed total_duration."""
        regions = [(55.0, 59.8)]
        result = _merge_speech_regions(regions, padding=0.5, total_duration=60.0)
        assert result[0][1] <= 60.0

    def test_all_regions_too_short(self):
        """All regions filtered out returns empty list."""
        regions = [(1.0, 1.1), (5.0, 5.2)]
        result = _merge_speech_regions(regions, min_duration=1.0)
        assert result == []


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

    def test_writes_speaker_format(self, tmp_path):
        result = {
            "text": "Hello. World.",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello.", "speaker": "Speaker 1"},
                {"start": 65.0, "end": 67.0, "text": "World.", "speaker": "Speaker 2"},
            ],
            "language": "en",
            "speakers": 2,
        }
        out = save_txt(result, tmp_path / "out.txt")
        content = out.read_text(encoding="utf-8")
        assert content == "Speaker 1: Hello.\nSpeaker 2: World."

    def test_no_speaker_field_uses_plain_text(self, tmp_path, sample_result):
        """When segments lack 'speaker' field, save plain text as before."""
        out = save_txt(sample_result, tmp_path / "out.txt")
        content = out.read_text(encoding="utf-8")
        assert content == "Hello world. This is a test."


# ---------------------------------------------------------------------------
# save_txt_alongside
# ---------------------------------------------------------------------------


class TestSaveTxtAlongside:
    """Tests for save_txt_alongside — saves .txt next to the media file."""

    def test_basic_save(self, tmp_path, sample_result):
        media = tmp_path / "video.mp4"
        media.touch()
        out = save_txt_alongside(sample_result, media)
        assert out == tmp_path / "video.txt"
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "Hello world. This is a test."

    def test_collision_increments_suffix(self, tmp_path, sample_result):
        media = tmp_path / "video.mp4"
        media.touch()
        (tmp_path / "video.txt").write_text("existing")
        out = save_txt_alongside(sample_result, media)
        assert out == tmp_path / "video_1.txt"
        assert out.exists()

    def test_multiple_collisions(self, tmp_path, sample_result):
        media = tmp_path / "video.mp4"
        media.touch()
        (tmp_path / "video.txt").write_text("existing")
        (tmp_path / "video_1.txt").write_text("existing")
        (tmp_path / "video_2.txt").write_text("existing")
        out = save_txt_alongside(sample_result, media)
        assert out == tmp_path / "video_3.txt"

    def test_audio_extension(self, tmp_path, sample_result):
        media = tmp_path / "recording.mp3"
        media.touch()
        out = save_txt_alongside(sample_result, media)
        assert out == tmp_path / "recording.txt"

    def test_permission_error(self, tmp_path, sample_result):
        read_only = tmp_path / "readonly"
        read_only.mkdir()
        read_only.chmod(0o444)
        media = read_only / "video.mp4"
        try:
            with pytest.raises((PermissionError, OSError)):
                save_txt_alongside(sample_result, media)
        finally:
            read_only.chmod(0o755)

    def test_returns_path_object(self, tmp_path, sample_result):
        media = tmp_path / "video.mp4"
        media.touch()
        out = save_txt_alongside(sample_result, str(media))
        assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# retry_diarize
# ---------------------------------------------------------------------------


class TestRetryDiarize:
    """Tests for retry_diarize — retries only diarization on an existing result."""

    def _base_result(self):
        return {
            "text": "Hello. World.",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello."},
                {"start": 2.0, "end": 3.5, "text": "World."},
            ],
            "language": "en",
            "speakers": 0,
            "diarize_error": "Bad token",
        }

    def test_success_removes_error(self):
        """On success, diarize_error should be removed from result."""
        labeled = [
            {"start": 0.0, "end": 1.5, "text": "Hello.", "speaker": "Speaker 1"},
            {"start": 2.0, "end": 3.5, "text": "World.", "speaker": "Speaker 2"},
        ]

        with patch("transcribe.core._extract_audio_pcm", return_value=np.zeros(16000 * 5, dtype=np.float32)):
            with patch("transcribe.diarize.diarize", return_value=(labeled, 2)):
                result = retry_diarize("fake.mp4", self._base_result(), "new-token")

        assert "diarize_error" not in result
        assert result["speakers"] == 2
        assert result["segments"][0]["speaker"] == "Speaker 1"

    def test_failure_returns_error(self):
        """On failure, result should have diarize_error and preserve segments."""
        with patch("transcribe.core._extract_audio_pcm", return_value=np.zeros(16000 * 5, dtype=np.float32)):
            with patch("transcribe.diarize.diarize", side_effect=RuntimeError("Invalid token")):
                result = retry_diarize("fake.mp4", self._base_result(), "bad-token")

        assert "diarize_error" in result
        assert "Invalid token" in result["diarize_error"]
        # Original segments preserved without speaker labels
        assert result["segments"][0]["text"] == "Hello."
        assert "speaker" not in result["segments"][0]

    def test_empty_segments_returns_original(self):
        """If result has no segments, return it unchanged."""
        empty = {"text": "", "segments": [], "language": "en"}
        result = retry_diarize("fake.mp4", empty, "token")
        assert result is empty

    def test_progress_callback_called(self):
        """Progress callback should receive increasing fractions."""
        labeled = [
            {"start": 0.0, "end": 1.5, "text": "Hello.", "speaker": "Speaker 1"},
        ]
        fractions = []

        def on_progress(frac, msg):
            fractions.append(frac)

        with patch("transcribe.core._extract_audio_pcm", return_value=np.zeros(16000 * 5, dtype=np.float32)):
            with patch("transcribe.diarize.diarize", return_value=(labeled, 1)):
                retry_diarize(
                    "fake.mp4",
                    self._base_result(),
                    "tok",
                    progress_callback=on_progress,
                )

        assert len(fractions) >= 2
        assert fractions[0] == 0.0
        assert fractions[-1] == 1.0


# ---------------------------------------------------------------------------
# transcribe_video — validation & edge cases (mocked Model + VAD)
# ---------------------------------------------------------------------------

def _make_segment(t0, t1, text):
    """Create a mock pywhispercpp segment (centisecond timestamps)."""
    return SimpleNamespace(t0=t0, t1=t1, text=text)


def _vad_patches(duration=10.0):
    """Return the standard set of VAD-related patches for transcribe_video tests.

    Mocks _extract_audio_pcm to return a fake audio array and
    _detect_speech_regions + _merge_speech_regions to return one region
    covering the full duration.
    """
    fake_audio = np.zeros(int((duration or 10.0) * 16000), dtype=np.float32)
    dur = duration or 10.0
    return (
        patch("transcribe.core._extract_audio_pcm", return_value=fake_audio),
        patch("transcribe.core._detect_speech_regions", return_value=[(0.0, dur)]),
        patch("transcribe.core._merge_speech_regions", return_value=[(0.0, dur)]),
    )


class TestTranscribeVideoValidation:
    def test_missing_ffmpeg(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        with patch("transcribe.core.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                transcribe_video(str(video))

    def test_missing_video_file(self):
        with patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"):
            with pytest.raises(FileNotFoundError, match="Media file not found"):
                transcribe_video("/nonexistent/video.mp4")

    def test_output_dir_created(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        out_dir = tmp_path / "nested" / "output"

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock()
        p1, p2, p3 = _vad_patches(10.0)

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
            p1, p2, p3,
        ):
            transcribe_video(str(video), output_dir=str(out_dir), language="en")

        assert out_dir.exists()

    def test_progress_callback_stages(self, tmp_path):
        """Progress callback should include loading, VAD, and done stages."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock()
        p1, p2, p3 = _vad_patches(10.0)

        progress_calls = []

        def track_progress(fraction, msg):
            progress_calls.append((fraction, msg))

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
            p1, p2, p3,
        ):
            transcribe_video(
                str(video), language="en", progress_callback=track_progress
            )

        # Should have at least: audio extraction, VAD, model loading, language, done
        assert len(progress_calls) >= 5
        assert progress_calls[0][0] < 0.12  # setup (audio extraction) reports first, low fraction
        assert progress_calls[-1][0] == 1.0  # done at 100%
        assert "Done" in progress_calls[-1][1]
        # Fractions are monotonically non-decreasing
        fractions = [f for f, _ in progress_calls]
        assert fractions == sorted(fractions)
        # Check the expected stages exist (model now loads inside the engine, after VAD)
        messages = [m for _, m in progress_calls]
        assert any("Extracting audio" in m for m in messages)
        assert any("speech region" in m for m in messages)
        assert any("Loading model" in m for m in messages)

    def test_return_dict_shape(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock()
        p1, p2, p3 = _vad_patches(10.0)

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
            p1, p2, p3,
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

        def fake_transcribe(media, new_segment_callback=None, **kwargs):
            if new_segment_callback:
                for seg in fake_segments:
                    new_segment_callback(seg)

        mock_model.transcribe = fake_transcribe
        p1, p2, p3 = _vad_patches(duration)

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=duration),
            p1, p2, p3,
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

    def test_repeated_segments_deduplicated(self, tmp_path):
        """Consecutive identical segments should be capped at 2 (anti-hallucination)."""
        segs = [
            _make_segment(0, 100, "Real speech."),
            _make_segment(100, 200, "Hallucinated."),
            _make_segment(200, 300, "Hallucinated."),
            _make_segment(300, 400, "Hallucinated."),
            _make_segment(400, 500, "Hallucinated."),
            _make_segment(500, 600, "Hallucinated."),
        ]
        result = self._run_with_segments(tmp_path, segs)
        # 1 real + 2 kept duplicates = 3
        assert len(result["segments"]) == 3
        assert result["segments"][0]["text"] == "Real speech."
        assert result["segments"][1]["text"] == "Hallucinated."
        assert result["segments"][2]["text"] == "Hallucinated."
        assert result["text"] == "Real speech. Hallucinated. Hallucinated."

    def test_language_autodetect(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        mock_model.auto_detect_language.return_value = (("fr", 0.9), {"fr": 0.9})
        mock_model.transcribe = MagicMock()
        p1, p2, p3 = _vad_patches(10.0)

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
            p1, p2, p3,
        ):
            result = transcribe_video(str(video), language=None)

        assert result["language"] == "fr"
        mock_model.auto_detect_language.assert_called_once()

    @pytest.mark.parametrize(
        ("language", "detect", "expected"),
        [
            (None, {"return_value": (("ru", 0.99), {"ru": 0.99})}, "ru"),  # detected per chunk
            (None, {"side_effect": RuntimeError("model error")}, ""),      # whisper's own auto
            ("ru", None, "ru"),
        ],
    )
    def test_language_always_passed_to_whisper(self, tmp_path, language, detect, expected):
        """An omitted language makes whisper.cpp fall back to "en" and translate Russian."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        if detect:
            mock_model.auto_detect_language.configure_mock(**detect)
        mock_model.transcribe = MagicMock()
        p1, p2, p3 = _vad_patches(10.0)

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
            p1, p2, p3,
        ):
            transcribe_video(str(video), language=language)

        assert mock_model.transcribe.call_args.kwargs["language"] == expected
        # Windows inside a chunk must not condition on each other (that dropped utterances).
        assert mock_model.transcribe.call_args.kwargs["n_max_text_ctx"] == 0

    def test_no_speech_detected_returns_empty(self, tmp_path):
        """When VAD finds no speech, return empty result without transcribing."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        fake_audio = np.zeros(16000 * 10, dtype=np.float32)

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=10.0),
            patch("transcribe.core._extract_audio_pcm", return_value=fake_audio),
            patch("transcribe.core._detect_speech_regions", return_value=[]),
            patch("transcribe.core._merge_speech_regions", return_value=[]),
        ):
            result = transcribe_video(str(video), language="en")

        assert result["text"] == ""
        assert result["segments"] == []
        assert result["language"] == "en"
        # model.transcribe should NOT have been called
        mock_model.transcribe.assert_not_called()

    def test_multi_chunk_timestamp_offsets(self, tmp_path):
        """Segments from different chunks should have correct absolute timestamps."""
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        fake_audio = np.zeros(16000 * 20, dtype=np.float32)

        # Two chunks: [0, 5] and [10, 15]
        # chunk 1 returns segment at relative 0.0-1.5
        # chunk 2 returns segment at relative 0.0-2.0
        chunk_call_count = [0]
        chunk_segments = [
            [SimpleNamespace(t0=0, t1=150, text=" Hello. ")],    # chunk 1
            [SimpleNamespace(t0=0, t1=200, text=" World. ")],    # chunk 2
        ]

        def fake_transcribe(media, new_segment_callback=None, **kwargs):
            idx = chunk_call_count[0]
            chunk_call_count[0] += 1
            if new_segment_callback and idx < len(chunk_segments):
                for seg in chunk_segments[idx]:
                    new_segment_callback(seg)

        mock_model.transcribe = fake_transcribe

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=20.0),
            patch("transcribe.core._extract_audio_pcm", return_value=fake_audio),
            patch("transcribe.core._detect_speech_regions", return_value=[(0.0, 5.0), (10.0, 15.0)]),
            patch("transcribe.core._merge_speech_regions", return_value=[(0.0, 5.0), (10.0, 15.0)]),
        ):
            result = transcribe_video(str(video), language="en")

        assert len(result["segments"]) == 2
        # Chunk 1 segment: 0.0 + 0.0 = 0.0, 0.0 + 1.5 = 1.5
        assert result["segments"][0] == {"start": 0.0, "end": 1.5, "text": "Hello."}
        # Chunk 2 segment: 10.0 + 0.0 = 10.0, 10.0 + 2.0 = 12.0
        assert result["segments"][1] == {"start": 10.0, "end": 12.0, "text": "World."}
        assert result["text"] == "Hello. World."

    def test_mixed_language_chunk_is_transcribed_per_language(self, tmp_path):
        """RU, EN, RU in one chunk: one call per language, or whisper drops the English."""
        video = tmp_path / "video.mp4"
        video.touch()

        calls = []

        def fake_transcribe(media, new_segment_callback=None, **kwargs):
            calls.append((len(media), kwargs["language"]))
            new_segment_callback(SimpleNamespace(t0=0, t1=100, text=kwargs["language"]))

        mock_model = MagicMock()
        mock_model.auto_detect_language.side_effect = _lid("ru", "en", "ru")
        mock_model.transcribe = fake_transcribe

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._get_media_duration", return_value=30.0),
            patch("transcribe.core._extract_audio_pcm",
                  return_value=np.zeros(16000 * 30, dtype=np.float32)),
            patch("transcribe.core._detect_speech_regions",
                  return_value=[(0.5, 9.5), (10.5, 15.5), (16.5, 29.5)]),
            patch("transcribe.core._merge_speech_regions", return_value=[(0.0, 30.0)]),
        ):
            result = transcribe_video(str(video), language=None)

        # Cut in the middle of each pause: (0, 10) ru, (10, 16) en, (16, 30) ru
        assert calls == [(10 * 16000, "ru"), (6 * 16000, "en"), (14 * 16000, "ru")]
        assert [(s["start"], s["text"]) for s in result["segments"]] == [
            (0.0, "ru"), (10.0, "en"), (16.0, "ru"),
        ]
        assert result["language"] == "ru"  # 24 s of Russian vs 6 s of English

    def test_screen_hearing_one_language_transcribes_the_chunk_whole(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()

        calls = []
        mock_model = MagicMock()
        mock_model.auto_detect_language.side_effect = _lid("ru")
        mock_model.transcribe = lambda media, **kwargs: calls.append((len(media), kwargs["language"]))
        screen = MagicMock()
        screen.auto_detect_language.side_effect = _lid("ru", "ru", "ru")

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._load_screen_model", return_value=screen),
            patch("transcribe.core._get_media_duration", return_value=30.0),
            patch("transcribe.core._extract_audio_pcm",
                  return_value=np.zeros(16000 * 30, dtype=np.float32)),
            patch("transcribe.core._detect_speech_regions",
                  return_value=[(0.5, 9.5), (10.5, 15.5), (16.5, 29.5)]),
            patch("transcribe.core._merge_speech_regions", return_value=[(0.0, 30.0)]),
        ):
            transcribe_video(str(video), language=None)

        assert calls == [(30 * 16000, "ru")]
        assert screen.auto_detect_language.call_count == 3

    @pytest.mark.parametrize(
        ("language", "model_size", "loads_screen"),
        [(None, "large-v3", True), ("ru", "large-v3", False), (None, "tiny", False)],
    )
    def test_screen_only_when_detecting_with_a_bigger_model(
        self, tmp_path, language, model_size, loads_screen,
    ):
        video = tmp_path / "video.mp4"
        video.touch()

        mock_model = MagicMock()
        mock_model.auto_detect_language.return_value = (("ru", 0.99), {"ru": 0.99})
        p1, p2, _ = _vad_patches(10.0)

        with (
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core.Model", return_value=mock_model),
            patch("transcribe.core._load_screen_model", return_value=None) as load_screen,
            patch("transcribe.core._get_media_duration", return_value=10.0),
            patch("transcribe.core._merge_speech_regions", return_value=[(0.0, 5.0), (5.0, 10.0)]),
            p1, p2,
        ):
            transcribe_video(str(video), model_size=model_size, language=language)

        assert load_screen.call_count == int(loads_screen)  # once per transcription, not per chunk


# ---------------------------------------------------------------------------
# Engine selection & Qwen3-ASR fallback
# ---------------------------------------------------------------------------


class TestEngineDispatch:
    """The engine= arg selects whisper/qwen; whisper failure falls back to qwen."""

    def _common_patches(self, whisper_side_effect=None):
        fake_audio = np.zeros(16000 * 10, dtype=np.float32)
        model_patch = (
            patch("transcribe.core.Model", side_effect=whisper_side_effect)
            if whisper_side_effect is not None
            else patch("transcribe.core.Model", return_value=MagicMock())
        )
        return [
            patch("transcribe.core.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("transcribe.core._get_media_duration", return_value=10.0),
            patch("transcribe.core._extract_audio_pcm", return_value=fake_audio),
            patch("transcribe.core._detect_speech_regions", return_value=[(0.0, 10.0)]),
            patch("transcribe.core._merge_speech_regions", return_value=[(0.0, 10.0)]),
            model_patch,
        ]

    def test_qwen_engine_selected_skips_whisper(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        qwen_ret = ([{"start": 0.0, "end": 10.0, "text": "привет"}], "ru")
        patches = self._common_patches()
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.transcribe_regions", return_value=qwen_ret) as mock_qwen,
        ):
            result = transcribe_video(str(video), engine="qwen")
        mock_qwen.assert_called_once()
        assert result["text"] == "привет"
        assert result["language"] == "ru"

    def test_qwen_engine_uses_default_context_prompt(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        from transcribe.qwen_prompt import DEFAULT_CONTEXT_PROMPT
        qwen_ret = ([{"start": 0.0, "end": 10.0, "text": "текст"}], "ru")
        patches = self._common_patches()
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.transcribe_regions", return_value=qwen_ret) as mock_qwen,
        ):
            transcribe_video(str(video), engine="qwen")
        _, kwargs = mock_qwen.call_args
        assert kwargs["system_prompt"] == DEFAULT_CONTEXT_PROMPT

    def test_custom_context_prompt_forwarded(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        qwen_ret = ([{"start": 0.0, "end": 10.0, "text": "x"}], "ru")
        patches = self._common_patches()
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.transcribe_regions", return_value=qwen_ret) as mock_qwen,
        ):
            transcribe_video(str(video), engine="qwen", qwen_context="MY CONTEXT")
        _, kwargs = mock_qwen.call_args
        assert kwargs["system_prompt"] == "MY CONTEXT"

    def test_fallback_to_qwen_on_whisper_failure(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        qwen_ret = ([{"start": 0.0, "end": 10.0, "text": "fallback text"}], "ru")
        patches = self._common_patches(whisper_side_effect=RuntimeError("whisper boom"))
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.is_available", return_value=True),
            patch("transcribe.core.qwen_engine.is_model_cached", return_value=True),
            patch("transcribe.core.qwen_engine.transcribe_regions", return_value=qwen_ret) as mock_qwen,
        ):
            result = transcribe_video(str(video), language="en", allow_qwen_fallback=True)
        mock_qwen.assert_called_once()
        assert result["text"] == "fallback text"
        # Provenance: the fallback engine and flag must be recorded.
        assert result["engine"] == "qwen"
        assert result["fallback"] is True

    def test_whisper_is_released_before_qwen_fallback(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        released = []

        class FailingWhisper:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, *_args, **_kwargs):
                raise RuntimeError("whisper boom")

            def __del__(self):
                released.append(True)

        def qwen_fallback(*_args, **_kwargs):
            assert released == [True]
            return ([{"start": 0.0, "end": 10.0, "text": "fallback text"}], "en")

        patches = self._common_patches()
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4],
            patch("transcribe.core.Model", FailingWhisper),
            patch("transcribe.core.qwen_engine.is_available", return_value=True),
            patch("transcribe.core.qwen_engine.is_model_cached", return_value=True),
            patch("transcribe.core.qwen_engine.transcribe_regions", side_effect=qwen_fallback),
        ):
            result = transcribe_video(str(video), language="en")

        assert result["fallback"] is True
        assert released == [True]

    def test_memory_error_is_not_retried_on_qwen(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        patches = self._common_patches(whisper_side_effect=MemoryError("oom"))
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.is_available", return_value=True),
            patch("transcribe.core.qwen_engine.is_model_cached", return_value=True),
            patch("transcribe.core.qwen_engine.transcribe_regions") as mock_qwen,
        ):
            with pytest.raises(MemoryError):
                transcribe_video(str(video), language="en")
        mock_qwen.assert_not_called()  # must NOT load a bigger model after an OOM

    def test_fallback_skipped_when_weights_not_cached(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        patches = self._common_patches(whisper_side_effect=RuntimeError("whisper boom"))
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.is_available", return_value=True),
            patch("transcribe.core.qwen_engine.is_model_cached", return_value=False),
            patch("transcribe.core.qwen_engine.transcribe_regions") as mock_qwen,
        ):
            with pytest.raises(RuntimeError, match="whisper boom"):
                transcribe_video(str(video), language="en")
        mock_qwen.assert_not_called()  # no multi-GB download for an unattended fallback

    def test_no_fallback_when_disabled_reraises(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        patches = self._common_patches(whisper_side_effect=RuntimeError("whisper boom"))
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.transcribe_regions") as mock_qwen,
        ):
            with pytest.raises(RuntimeError, match="whisper boom"):
                transcribe_video(str(video), language="en", allow_qwen_fallback=False)
        mock_qwen.assert_not_called()

    def test_fallback_unavailable_reraises_original(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        patches = self._common_patches(whisper_side_effect=RuntimeError("whisper boom"))
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.is_available", return_value=False),
            patch("transcribe.core.qwen_engine.transcribe_regions") as mock_qwen,
        ):
            with pytest.raises(RuntimeError, match="whisper boom"):
                transcribe_video(str(video), language="en")
        mock_qwen.assert_not_called()

    def _run_qwen(self, tmp_path, mock_qwen, **kwargs):
        video = tmp_path / "v.mp4"
        video.touch()
        qwen_ret = ([{"start": 0.0, "end": 10.0, "text": "привет."}], "ru")
        mock_qwen.return_value = qwen_ret
        patches = self._common_patches()
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patch("transcribe.core.qwen_engine.is_model_cached", return_value=True),
            patch("transcribe.core.qwen_engine.transcribe_regions", mock_qwen),
        ):
            return transcribe_video(str(video), engine="qwen", **kwargs)

    def test_qwen_align_defaults_off_without_diarize(self, tmp_path):
        mock_qwen = MagicMock()
        self._run_qwen(tmp_path, mock_qwen)
        assert mock_qwen.call_args.kwargs["align"] is False

    def test_engine_provenance_recorded_for_direct_qwen(self, tmp_path):
        mock_qwen = MagicMock()
        result = self._run_qwen(tmp_path, mock_qwen)
        assert result["engine"] == "qwen"
        assert "fallback" not in result  # only set when whisper fell back

    def test_qwen_align_explicit_true(self, tmp_path):
        mock_qwen = MagicMock()
        self._run_qwen(tmp_path, mock_qwen, qwen_align=True)
        assert mock_qwen.call_args.kwargs["align"] is True

    def test_diarize_auto_enables_alignment(self, tmp_path):
        mock_qwen = MagicMock()
        labeled = [{"start": 0.0, "end": 10.0, "text": "привет.", "speaker": "Speaker 1"}]
        with patch("transcribe.diarize.diarize", return_value=(labeled, 1)):
            self._run_qwen(tmp_path, mock_qwen, language="ru", diarize=True, hf_token="tok")
        assert mock_qwen.call_args.kwargs["align"] is True
