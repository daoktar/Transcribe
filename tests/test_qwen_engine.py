"""Tests for the Qwen3-ASR MLX fallback engine (transcribe.qwen_engine).

The heavy MLX model is always mocked — these tests exercise the region-slicing,
segment-mapping, language-vote and progress logic without loading any weights.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from transcribe import qwen_engine


def _out(text, language):
    """Minimal stand-in for mlx_audio's STTOutput."""
    return SimpleNamespace(text=text, language=language)


def _item(text, start, end):
    """Minimal stand-in for a ForcedAlignItem (start/end in seconds)."""
    return SimpleNamespace(text=text, start_time=start, end_time=end)


class TestTranscribeRegions:
    def test_maps_regions_to_segments(self):
        audio = np.zeros(16000 * 20, dtype=np.float32)
        regions = [(0.0, 5.0), (10.0, 15.0)]
        model = MagicMock()
        model.generate.side_effect = [
            _out(" Привет мир. ", ["ru"]),
            _out("Deploy the item.", ["ru"]),
        ]
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, lang = qwen_engine.transcribe_regions(
                audio, regions, language=None, system_prompt="ctx"
            )
        assert segs == [
            {"start": 0.0, "end": 5.0, "text": "Привет мир."},
            {"start": 10.0, "end": 15.0, "text": "Deploy the item."},
        ]
        assert lang == "ru"
        # context prompt + temperature forwarded to the model
        _, kwargs = model.generate.call_args
        assert kwargs["system_prompt"] == "ctx"
        assert kwargs["temperature"] == 0.0

    def test_forced_language_wins_over_detection(self):
        audio = np.zeros(16000 * 10, dtype=np.float32)
        model = MagicMock()
        model.generate.return_value = _out("текст", ["en"])
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            _segs, lang = qwen_engine.transcribe_regions(
                audio, [(0.0, 5.0)], language="ru"
            )
        assert lang == "ru"
        _, kwargs = model.generate.call_args
        assert kwargs["language"] == "ru"

    def test_majority_language_vote(self):
        audio = np.zeros(16000 * 30, dtype=np.float32)
        regions = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
        model = MagicMock()
        model.generate.side_effect = [
            _out("a", ["ru"]),
            _out("b", ["en"]),
            _out("c", ["ru"]),
        ]
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            _segs, lang = qwen_engine.transcribe_regions(audio, regions, language=None)
        assert lang == "ru"

    def test_empty_text_skipped(self):
        audio = np.zeros(16000 * 10, dtype=np.float32)
        model = MagicMock()
        model.generate.return_value = _out("   ", ["ru"])
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(audio, [(0.0, 5.0)])
        assert segs == []

    def test_empty_chunk_skips_generate(self):
        audio = np.zeros(16000 * 5, dtype=np.float32)  # only 5s of audio
        model = MagicMock()
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, lang = qwen_engine.transcribe_regions(audio, [(100.0, 105.0)])
        model.generate.assert_not_called()
        assert segs == []
        assert lang == "unknown"

    def test_progress_callback_reports_each_region(self):
        audio = np.zeros(16000 * 20, dtype=np.float32)
        regions = [(0.0, 5.0), (10.0, 15.0)]
        model = MagicMock()
        model.generate.return_value = _out("x", ["ru"])
        calls = []
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            qwen_engine.transcribe_regions(
                audio, regions, progress_callback=lambda d, t: calls.append((d, t))
            )
        assert calls == [(1, 2), (2, 2)]

    def test_region_past_end_is_clamped_not_dropped(self):
        audio = np.zeros(16000 * 5, dtype=np.float32)  # 5s of audio
        model = MagicMock()
        model.generate.return_value = _out("край", ["ru"])
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(audio, [(4.0, 10.0)], language="ru")
        # Region overruns the audio; the in-bounds tail must still be transcribed.
        model.generate.assert_called_once()
        assert len(segs) == 1

    def test_region_fully_past_end_is_dropped(self):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        model = MagicMock()
        calls = []
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(
                audio, [(100.0, 105.0)], language="ru",
                progress_callback=lambda d, t: calls.append((d, t)),
            )
        model.generate.assert_not_called()
        assert segs == []
        assert calls == [(1, 1)]  # progress still advances


def test_is_model_cached_returns_bool_without_network():
    # local_files_only lookup → no network; unknown repo → False.
    assert qwen_engine.is_model_cached("nonexistent-org/nonexistent-repo-xyz") is False


class TestModelCache:
    def test_load_qwen_model_caches(self):
        qwen_engine._MODEL_CACHE.clear()
        sentinel = object()
        with patch("mlx_audio.stt.utils.load_model", return_value=sentinel) as loader:
            first = qwen_engine.load_qwen_model("fake/repo")
            second = qwen_engine.load_qwen_model("fake/repo")
        assert first is sentinel and second is sentinel
        loader.assert_called_once()  # cached on second call
        qwen_engine._MODEL_CACHE.clear()


def test_is_available_returns_bool():
    assert isinstance(qwen_engine.is_available(), bool)


class TestRegroupWithAlignment:
    def test_zip_preserves_punctuation_and_splits_on_sentences(self):
        text = "Привет, Сереж. Как дела?"
        items = [_item("Привет", 0.0, 0.4), _item("Сереж", 0.4, 0.9),
                 _item("Как", 1.0, 1.2), _item("дела", 1.2, 1.6)]
        segs = qwen_engine._regroup_with_alignment(text, items, offset=10.0)
        assert segs == [
            {"start": 10.0, "end": 10.9, "text": "Привет, Сереж."},
            {"start": 11.0, "end": 11.6, "text": "Как дела?"},
        ]

    def test_splits_on_long_pause(self):
        text = "раз два три"
        items = [_item("раз", 0.0, 0.3), _item("два", 0.4, 0.7), _item("три", 2.0, 2.3)]
        segs = qwen_engine._regroup_with_alignment(text, items, offset=0.0, gap_thresh=0.7)
        assert [s["text"] for s in segs] == ["раз два", "три"]

    def test_splits_on_max_duration(self):
        text = "a b c"
        items = [_item("a", 0.0, 5.0), _item("b", 5.0, 12.0), _item("c", 12.0, 13.0)]
        segs = qwen_engine._regroup_with_alignment(text, items, offset=0.0, max_dur=10.0)
        assert [s["text"] for s in segs] == ["a b", "c"]

    def test_fallback_on_token_count_mismatch(self):
        text = "one two three"  # 3 original tokens
        items = [_item("one", 0.0, 0.3), _item("two", 0.3, 0.6)]  # 2 aligned → mismatch
        segs = qwen_engine._regroup_with_alignment(text, items, offset=0.0)
        assert [s["text"] for s in segs] == ["one two"]  # aligner tokens, no punctuation

    def test_empty_items(self):
        assert qwen_engine._regroup_with_alignment("x", [], 0.0) == []

    def test_ultrashort_word_never_zero_duration(self):
        # start and end that round to the same 3-decimal value must not collapse to
        # a zero-duration segment (would break diarization / subtitles).
        text = "Да."
        items = [_item("Да", 5.0001, 5.0004)]
        segs = qwen_engine._regroup_with_alignment(text, items, offset=10.0)
        assert len(segs) == 1
        assert segs[0]["end"] > segs[0]["start"]
        assert segs[0]["text"] == "Да."


class TestAlignRegion:
    def test_unsupported_language_returns_none(self):
        aligner = MagicMock()
        out = qwen_engine._align_region(
            aligner, np.zeros(1600, dtype=np.float32), "text", "xx", 0.0
        )
        assert out is None
        aligner.generate.assert_not_called()

    def test_exception_returns_none(self):
        aligner = MagicMock()
        aligner.generate.side_effect = RuntimeError("boom")
        out = qwen_engine._align_region(
            aligner, np.zeros(1600, dtype=np.float32), "текст", "ru", 0.0
        )
        assert out is None

    def test_success_offsets_and_maps_language(self):
        aligner = MagicMock()
        aligner.generate.return_value = SimpleNamespace(items=[_item("Да", 0.0, 0.5)])
        out = qwen_engine._align_region(
            aligner, np.zeros(1600, dtype=np.float32), "Да.", "ru", 5.0
        )
        assert out == [{"start": 5.0, "end": 5.5, "text": "Да."}]
        assert aligner.generate.call_args.kwargs["language"] == "Russian"


class TestTranscribeRegionsWithAlignment:
    def test_align_true_emits_fine_segments(self):
        audio = np.zeros(16000 * 20, dtype=np.float32)
        asr = MagicMock()
        asr.generate.return_value = _out("Привет. Как дела?", ["ru"])
        aligner = MagicMock()
        aligner.generate.return_value = SimpleNamespace(
            items=[_item("Привет", 0.0, 0.5), _item("Как", 1.0, 1.2), _item("дела", 1.2, 1.6)]
        )

        def fake_load(path):
            return aligner if "ForcedAligner" in path else asr

        with patch("transcribe.qwen_engine.load_qwen_model", side_effect=fake_load):
            segs, lang = qwen_engine.transcribe_regions(
                audio, [(0.0, 20.0)], language="ru", align=True
            )
        assert [s["text"] for s in segs] == ["Привет.", "Как дела?"]
        assert lang == "ru"

    def test_align_false_emits_one_coarse_segment(self):
        audio = np.zeros(16000 * 20, dtype=np.float32)
        asr = MagicMock()
        asr.generate.return_value = _out("Привет. Как дела?", ["ru"])
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=asr):
            segs, _lang = qwen_engine.transcribe_regions(audio, [(0.0, 20.0)], language="ru")
        assert segs == [{"start": 0.0, "end": 20.0, "text": "Привет. Как дела?"}]
