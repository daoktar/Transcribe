"""Tests for the Qwen3-ASR MLX fallback engine (transcribe.qwen_engine).

The heavy MLX model is always mocked — these tests exercise the region-slicing,
segment-mapping, language-vote and progress logic without loading any weights.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcribe import qwen_engine

# conftest stubs both backend probes to False; these tests exercise the real ones.
_real_cuda_available = qwen_engine._cuda_available
_real_mlx_available = qwen_engine._mlx_available


@pytest.fixture
def real_backend_probes():
    with patch("transcribe.qwen_engine._cuda_available", _real_cuda_available), \
         patch("transcribe.qwen_engine._mlx_available", _real_mlx_available):
        yield


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


def test_is_model_cached_rejects_config_only_cache():
    """A partial cache must not trigger an automatic multi-GB model download."""
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        side_effect=lambda _repo, filename: "/tmp/config.json" if filename == "config.json" else None,
    ):
        assert qwen_engine.is_model_cached() is False


def test_is_model_cached_accepts_complete_default_cache():
    with patch("huggingface_hub.try_to_load_from_cache", return_value="/tmp/cached-file"):
        assert qwen_engine.is_model_cached() is True


class TestModelCache:
    def test_load_qwen_model_caches(self):
        qwen_engine._MODEL_CACHE.clear()
        sentinel = object()
        with patch("transcribe.qwen_engine.backend", return_value="mlx"), \
             patch("transcribe.qwen_engine._load_mlx_model", return_value=sentinel) as loader:
            first = qwen_engine.load_qwen_model("fake/repo")
            second = qwen_engine.load_qwen_model("fake/repo")
        assert first is sentinel and second is sentinel
        loader.assert_called_once()  # cached on second call
        qwen_engine._MODEL_CACHE.clear()


@pytest.mark.usefixtures("real_backend_probes")
def test_is_available_returns_bool():
    with patch("transcribe.qwen_engine.importlib.import_module", return_value=object()) as loader:
        assert qwen_engine.is_available() is True
    assert loader.call_args_list[0].args == ("mlx.core",)
    assert loader.call_args_list[1].args == ("mlx_audio.stt.utils",)


@pytest.mark.usefixtures("real_backend_probes")
def test_is_available_handles_import_failure():
    with patch("transcribe.qwen_engine.importlib.import_module", side_effect=ImportError):
        assert qwen_engine.is_available() is False


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

        def fake_load(path, **_kind):
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


class TestPromptEchoFilter:
    """Qwen sometimes returns its own system_prompt instead of speech (see _strip_prompt_echo)."""

    PROMPT = (
        "Это рабочие созвоны команды по модерации рекламы. Частые имена: Сергей, Андрей. "
        "Английские технические термины сохраняй латиницей, не транслитерируй в кириллицу. "
        "Инструменты: Jira, Confluence, Redash."
    )

    def _run(self, texts, prompt):
        audio = np.zeros(16000 * 10 * len(texts), dtype=np.float32)
        regions = [(float(i * 10), float(i * 10 + 5)) for i in range(len(texts))]
        model = MagicMock()
        model.generate.side_effect = [_out(t, ["ru"]) for t in texts]
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(
                audio, regions, language="ru", system_prompt=prompt
            )
        return [s["text"] for s in segs]

    def test_echoed_prompt_is_dropped(self):
        # Contaminated file: most of the prompt comes back verbatim, smeared across regions
        # and interleaved with real speech.
        texts = [
            "Давайте обсудим медиатеку.",
            "Это рабочие созвоны команды по модерации рекламы.",
            "Частые имена: Сергей, Андрей.",
            "Английские технические термины",
            "сохраняй латиницей,",
            "не транслитерируй в кириллицу.",
            "Инструменты: Jira,",
            "Confluence,",
            "Redash.",
            "Файл привязан к рекламодателю.",
        ]
        assert self._run(texts, self.PROMPT) == [
            "Давайте обсудим медиатеку.",
            "Файл привязан к рекламодателю.",
        ]

    def test_incidental_matches_are_kept(self):
        # Clean file: a few segments happen to be substrings of the prompt (stop-words, a
        # tool name someone actually said). Coverage stays low, nothing may be dropped.
        texts = ["Я", "Да.", "Redash.", "Мы тянем блок ER из Redash, а не из DWH."]
        assert self._run(texts, self.PROMPT) == texts

    def test_no_prompt_means_no_filtering(self):
        assert self._run(["Инструменты: Jira,"], None) == ["Инструменты: Jira,"]


class TestPromptEchoRetry:
    """A region that decodes the prompt is re-transcribed, not just deleted."""

    PROMPT = (
        "Это рабочие созвоны команды по модерации рекламы и по борьбе с дублями объявлений. "
        "Английские технические термины сохраняй латиницей, не транслитерируй в кириллицу."
    )

    def test_echoing_region_is_retranscribed_with_nudged_window(self):
        audio = np.zeros(16000 * 40, dtype=np.float32)
        model = MagicMock()
        model.generate.side_effect = [_out(self.PROMPT, ["ru"]), _out("Водник медиатеки.", ["ru"])]
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(
                audio, [(5.0, 36.0)], language="ru", system_prompt=self.PROMPT
            )
        assert [s["text"] for s in segs] == ["Водник медиатеки."]
        # retry moved the window 100 ms earlier and kept the prompt
        assert model.generate.call_count == 2
        first, second = (c.args[0].size for c in model.generate.call_args_list)
        assert second == first + 1600

    def test_prompt_free_pass_is_the_last_resort(self):
        audio = np.zeros(16000 * 40, dtype=np.float32)
        model = MagicMock()
        model.generate.side_effect = [
            _out(self.PROMPT, ["ru"]),   # original
            _out(self.PROMPT, ["ru"]),   # nudge earlier
            _out(self.PROMPT, ["ru"]),   # nudge later
            _out("настоящая речь", ["ru"]),  # without the prompt
        ]
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(
                audio, [(5.0, 36.0)], language="ru", system_prompt=self.PROMPT
            )
        assert [s["text"] for s in segs] == ["настоящая речь"]
        assert model.generate.call_args_list[-1].kwargs["system_prompt"] is None

    def test_unrecoverable_region_falls_through_to_the_file_filter(self):
        audio = np.zeros(16000 * 40, dtype=np.float32)
        model = MagicMock()
        model.generate.return_value = _out(self.PROMPT, ["ru"])  # echoes every time
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(
                audio, [(5.0, 36.0)], language="ru", system_prompt=self.PROMPT
            )
        assert segs == []  # dropped by _strip_prompt_echo, never reaches the transcript

    def test_real_speech_is_not_retried(self):
        audio = np.zeros(16000 * 40, dtype=np.float32)
        model = MagicMock()
        model.generate.return_value = _out("Обсуждаем медиатеку и дубли объявлений.", ["ru"])
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, _lang = qwen_engine.transcribe_regions(
                audio, [(5.0, 36.0)], language="ru", system_prompt=self.PROMPT
            )
        assert model.generate.call_count == 1
        assert len(segs) == 1


def _fake_import(available: set[str]):
    """import_module stand-in: only the named modules import."""
    def importer(name):
        if name not in available:
            raise ImportError(name)
        return object()
    return importer


class TestBackend:
    """The GPU itself comes from gpu.cuda_total_gb (conftest: None), the packages from imports."""

    @pytest.fixture(autouse=True)
    def _real_probes(self, real_backend_probes):
        yield

    def _backend(self, modules, gpu_gb=3.6):
        # cuda_total_gb first: the import_module patch is global (importlib is shared), and
        # mock.patch resolves its targets through it.
        with patch("transcribe.gpu.cuda_total_gb", return_value=gpu_gb), \
             patch("transcribe.qwen_engine.importlib.import_module",
                   side_effect=_fake_import(modules)):
            return qwen_engine.backend(), qwen_engine.default_engine()

    def test_mlx_wins_when_both_import(self):
        assert self._backend({"mlx.core", "mlx_audio.stt.utils", "qwen_asr"})[0] == "mlx"

    def test_cuda_without_mlx(self):
        assert self._backend({"qwen_asr"}) == ("cuda", "qwen")

    def test_no_gpu_is_unavailable(self):
        assert self._backend({"qwen_asr"}, gpu_gb=None) == (None, "whisper")

    def test_gpu_without_qwen_asr_is_unavailable(self):
        assert self._backend(set())[0] is None


class TestResolveModelPath:
    def test_cuda_maps_default_mlx_ids(self):
        with patch("transcribe.qwen_engine.backend", return_value="cuda"):
            assert qwen_engine.resolve_model_path(qwen_engine.DEFAULT_QWEN_MODEL) == "Qwen/Qwen3-ASR-1.7B"
            assert (qwen_engine.resolve_model_path(qwen_engine.DEFAULT_ALIGNER_MODEL)
                    == "Qwen/Qwen3-ForcedAligner-0.6B")
            assert qwen_engine.resolve_model_path("me/custom") == "me/custom"

    def test_mlx_keeps_ids(self):
        with patch("transcribe.qwen_engine.backend", return_value="mlx"):
            assert qwen_engine.resolve_model_path(qwen_engine.DEFAULT_QWEN_MODEL) == qwen_engine.DEFAULT_QWEN_MODEL


class TestShardedCache:
    def _cache(self, tmp_path, files):
        index = tmp_path / "model.safetensors.index.json"
        index.write_text('{"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}')
        paths = {"model.safetensors.index.json": str(index)} | {f: f"/tmp/{f}" for f in files}
        return lambda _repo, filename: paths.get(filename)

    def test_all_shards_cached(self, tmp_path):
        lookup = self._cache(tmp_path, ["config.json", "model-1.safetensors", "model-2.safetensors"])
        with patch("transcribe.qwen_engine.backend", return_value="cuda"), \
             patch("huggingface_hub.try_to_load_from_cache", side_effect=lookup):
            assert qwen_engine.is_model_cached() is True

    def test_missing_shard_is_uncached(self, tmp_path):
        lookup = self._cache(tmp_path, ["config.json", "model-1.safetensors"])
        with patch("transcribe.qwen_engine.backend", return_value="cuda"), \
             patch("huggingface_hub.try_to_load_from_cache", side_effect=lookup):
            assert qwen_engine.is_model_cached() is False


class TestCudaAdapters:
    def test_asr_maps_language_code_and_prompt(self):
        inner = MagicMock()
        inner.transcribe.return_value = [SimpleNamespace(text="Привет", language="Russian,English")]
        chunk = np.zeros(1600, dtype=np.float32)
        out = qwen_engine._CudaASR(inner).generate(chunk, language="ru", system_prompt="ctx")
        kwargs = inner.transcribe.call_args.kwargs
        assert kwargs["language"] == "Russian" and kwargs["context"] == "ctx"
        assert kwargs["audio"][1] == qwen_engine.SAMPLE_RATE
        assert out.text == "Привет" and out.language == ["Russian", "English"]

    def test_asr_auto_language_and_no_prompt(self):
        inner = MagicMock()
        inner.transcribe.return_value = [SimpleNamespace(text="", language="")]
        out = qwen_engine._CudaASR(inner).generate(np.zeros(10, dtype=np.float32))
        assert inner.transcribe.call_args.kwargs["language"] is None
        assert inner.transcribe.call_args.kwargs["context"] == ""
        assert out.language == [""]

    def test_aligner_returns_first_result(self):
        inner = MagicMock()
        inner.align.return_value = ["first"]
        res = qwen_engine._CudaAligner(inner).generate(np.zeros(10, dtype=np.float32), "hi", language="English")
        assert res == "first"
        assert inner.align.call_args.kwargs["language"] == "English"

    def test_load_on_cuda_resolves_path_and_kind(self):
        qwen_engine._MODEL_CACHE.clear()
        with patch("transcribe.qwen_engine.backend", return_value="cuda"), \
             patch("transcribe.qwen_engine._load_cuda_model", return_value="aligner") as loader:
            qwen_engine.load_qwen_model(qwen_engine.DEFAULT_ALIGNER_MODEL, aligner=True)
        loader.assert_called_once_with("Qwen/Qwen3-ForcedAligner-0.6B", True)
        qwen_engine._MODEL_CACHE.clear()


class TestCudaQuantization:
    def _quant(self, total_gb):
        with patch("transcribe.gpu.cuda_total_gb", return_value=total_gb):
            return qwen_engine._cuda_quantization()

    def test_small_gpu_gets_nf4(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBE_QWEN_CUDA_QUANT", raising=False)
        assert self._quant(3.6) == "nf4"  # RTX 3050 Laptop "4 GB"

    def test_8gb_card_gets_nf4(self, monkeypatch):
        # bf16 ASR + aligner (~6.5 GB) leaves too little of the 7.6-8.0 GiB it reports.
        monkeypatch.delenv("TRANSCRIBE_QWEN_CUDA_QUANT", raising=False)
        assert self._quant(7.99) == "nf4"

    def test_10gb_card_gets_bf16(self, monkeypatch):
        monkeypatch.delenv("TRANSCRIBE_QWEN_CUDA_QUANT", raising=False)
        assert self._quant(9.8) == "bf16"  # RTX 3080 "10 GB"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIBE_QWEN_CUDA_QUANT", "BF16")
        assert self._quant(3.6) == "bf16"


class TestSupportedLanguage:
    def test_supported_code_kept(self):
        assert qwen_engine._supported_language("ru") == "ru"

    def test_name_becomes_code(self):
        assert qwen_engine._supported_language("Russian") == "ru"

    def test_alias(self):
        assert qwen_engine._supported_language("tl") == "fil"

    def test_unsupported_falls_back_to_auto(self, caplog):
        assert qwen_engine._supported_language("uk") is None
        assert "does not support language 'uk'" in caplog.text

    def test_none(self):
        assert qwen_engine._supported_language(None) is None

    def test_unsupported_language_is_not_forced_on_the_model(self):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        model = MagicMock()
        model.generate.return_value = _out("Привіт", ["Russian"])
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            segs, lang = qwen_engine.transcribe_regions(audio, [(0.0, 5.0)], language="uk")
        assert model.generate.call_args.kwargs["language"] is None
        assert lang == "ru"  # detected, not the unsupported forced code
        assert segs[0]["text"] == "Привіт"

    def test_aligner_map_derives_from_names(self):
        assert qwen_engine._CODE_TO_ALIGNER_LANG["ru"] == "Russian"
        assert set(qwen_engine._CODE_TO_ALIGNER_LANG) == set(qwen_engine._ALIGNER_CODES)


class TestCudaModelPath:
    def test_other_mlx_checkpoint_is_rejected_on_cuda(self):
        with patch("transcribe.qwen_engine.backend", return_value="cuda"), \
             pytest.raises(ValueError, match="MLX checkpoint"):
            qwen_engine.resolve_model_path("mlx-community/Qwen3-ASR-1.7B-8bit")

    def test_rejected_checkpoint_reports_uncached(self):
        with patch("transcribe.qwen_engine.backend", return_value="cuda"), \
             patch("huggingface_hub.try_to_load_from_cache", return_value="/tmp/x"):
            assert qwen_engine.is_model_cached("mlx-community/Qwen3-ASR-1.7B-8bit") is False

    def test_backend_is_probed_once(self):
        with patch("transcribe.qwen_engine._mlx_available", return_value=False) as mlx, \
             patch("transcribe.qwen_engine._cuda_available", return_value=True):
            assert qwen_engine.backend() == "cuda"
            assert qwen_engine.backend() == "cuda"
        mlx.assert_called_once()


class TestReleaseGpuModels:
    def test_drops_only_cuda_models_and_empties_cache(self):
        qwen_engine._MODEL_CACHE.clear()
        mlx_model = object()
        qwen_engine._MODEL_CACHE.update({
            "a": qwen_engine._CudaASR(MagicMock()),
            "b": qwen_engine._CudaAligner(MagicMock()),
            "m": mlx_model,
        })
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch}), \
             patch("transcribe.gpu.cuda_total_gb", return_value=3.6):
            qwen_engine.release_gpu_models()
        assert qwen_engine._MODEL_CACHE == {"m": mlx_model}
        torch.cuda.empty_cache.assert_called_once()
        qwen_engine._MODEL_CACHE.clear()

    def test_nothing_cached_never_imports_torch(self):
        qwen_engine._MODEL_CACHE.clear()
        with patch.dict("sys.modules", {"torch": None}):  # import would raise
            qwen_engine.release_gpu_models()


class TestRequestedLanguageReported:
    def test_alias_reports_the_code_the_caller_gave(self):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        model = MagicMock()
        model.generate.return_value = _out("Kumusta", ["Filipino"])
        with patch("transcribe.qwen_engine.load_qwen_model", return_value=model):
            _, lang = qwen_engine.transcribe_regions(audio, [(0.0, 5.0)], language="tl")
        assert model.generate.call_args.kwargs["language"] == "fil"
        assert lang == "tl"


class TestReleaseEmptiesTorchCache:
    def test_empties_cache_without_cached_models_when_torch_is_loaded(self):
        qwen_engine._MODEL_CACHE.clear()
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch}), \
             patch("transcribe.gpu.cuda_total_gb", return_value=3.6):
            qwen_engine.release_gpu_models()
        torch.cuda.empty_cache.assert_called_once()

    def test_skips_without_cuda(self):
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch}):
            qwen_engine.release_gpu_models()  # conftest: cuda_total_gb() is None
        torch.cuda.empty_cache.assert_not_called()
