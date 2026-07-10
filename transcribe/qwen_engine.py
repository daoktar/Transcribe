"""Qwen3-ASR (MLX) transcription engine — the fallback for the whisper.cpp path.

Runs Alibaba's open-weight **Qwen3-ASR-1.7B** on Apple Silicon via MLX (no PyTorch/CUDA).
It plugs into the same VAD-chunk pipeline as the default whisper engine: each detected
speech region is transcribed independently and emitted as a timestamped segment, with a
domain *context prompt* biasing vocabulary and — unlike Whisper — keeping English tech
terms in Latin script instead of transliterating them into Cyrillic.

MLX is an optional, Apple-Silicon-only dependency. Every mlx import is lazy so the rest of
the app (and non-macOS installs) keep working without it — call :func:`is_available` first.
"""
from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Bound HF Hub downloads so a stalled first-run fetch surfaces an error instead of
# wedging the worker thread forever (only sets a default; respects any user override).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

DEFAULT_QWEN_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"
SAMPLE_RATE = 16000

# Loaded models are cached per repo id — loading costs several seconds and ~1.8-3.4 GB.
_MODEL_CACHE: dict[str, object] = {}

# The forced aligner takes a language *name* and supports only these 11 languages.
_CODE_TO_ALIGNER_LANG = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "de": "German",
    "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "ko": "Korean", "ja": "Japanese",
}

# Qwen3-ASR reports languages as full names (e.g. "Russian"); map the documented 30
# languages to ISO codes so the result matches whisper's ("ru", "en", ...).
_LANG_NAME_TO_CODE = {
    "chinese": "zh", "english": "en", "cantonese": "yue", "arabic": "ar",
    "german": "de", "french": "fr", "spanish": "es", "portuguese": "pt",
    "indonesian": "id", "italian": "it", "korean": "ko", "russian": "ru",
    "thai": "th", "vietnamese": "vi", "japanese": "ja", "turkish": "tr",
    "hindi": "hi", "malay": "ms", "dutch": "nl", "swedish": "sv", "danish": "da",
    "finnish": "fi", "polish": "pl", "czech": "cs", "filipino": "fil",
    "persian": "fa", "greek": "el", "hungarian": "hu", "macedonian": "mk",
    "romanian": "ro",
}


def _normalize_language(value: str | None) -> str:
    """Map a Qwen language name/code to an ISO code (pass through unknown values)."""
    if not value:
        return "unknown"
    key = value.strip().lower()
    return _LANG_NAME_TO_CODE.get(key, key)


def is_available() -> bool:
    """Return True if the MLX stack needed for Qwen3-ASR is importable on this machine."""
    try:
        importlib.import_module("mlx.core")
        importlib.import_module("mlx_audio.stt.utils")

        return True
    except Exception as exc:
        # Not fatal — the app runs whisper-only — but log so a *broken* MLX install
        # (as opposed to simply absent / non-Apple-Silicon) is diagnosable.
        logger.debug("Qwen3-ASR unavailable: %s: %s", type(exc).__name__, exc)
        return False


def is_model_cached(model_path: str = DEFAULT_QWEN_MODEL) -> bool:
    """Return True if the model can load from the local HF cache.

    Used to gate the automatic whisper→Qwen fallback so an unattended job never kicks
    off a multi-GB download. Requires both config.json and the unsharded weight file —
    checking config.json alone is insufficient since it can remain after an interrupted
    download. Sharded custom models (no single model.safetensors) are treated
    conservatively and report uncached.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        return all(
            isinstance(try_to_load_from_cache(model_path, filename), str)
            for filename in ("config.json", "model.safetensors")
        )
    except Exception:
        return False


def _load_mlx_model(model_path: str):
    """Load one MLX model while keeping the optional import local and mockable."""
    from mlx_audio.stt.utils import load_model

    return load_model(model_path)


def load_qwen_model(model_path: str = DEFAULT_QWEN_MODEL):
    """Load (and cache) a Qwen3-ASR MLX model. Downloads weights from HF on first use."""
    cached = _MODEL_CACHE.get(model_path)
    if cached is None:
        cached = _load_mlx_model(model_path)
        _MODEL_CACHE[model_path] = cached
    return cached


def _regroup_with_alignment(
    region_text: str,
    items: list,
    offset: float,
    max_dur: float = 14.0,
    gap_thresh: float = 0.7,
) -> list[dict]:
    """Turn a region's aligned word items into phrase-level segments.

    The aligner strips punctuation from its tokens, so we zip its per-word *timestamps*
    back onto the original whitespace-split (punctuated) tokens when the counts line up,
    then split into phrases at sentence-ending punctuation, long pauses, or a max length.
    Timestamps are offset to absolute file time. Falls back to aligner tokens on mismatch.
    """
    aligned = [it for it in items if it.text.strip()]
    if not aligned:
        return []

    orig = region_text.split()
    if len(orig) == len(aligned):
        pairs = [(orig[i], aligned[i].start_time, aligned[i].end_time) for i in range(len(aligned))]
    else:
        pairs = [(it.text, it.start_time, it.end_time) for it in aligned]

    segments: list[dict] = []
    words: list[str] = []
    start: float | None = None
    prev_end: float | None = None

    def emit(end: float) -> None:
        seg_start = round(start + offset, 3)
        seg_end = round(end + offset, 3)
        if seg_end <= seg_start:
            # A single ultra-short word can round to zero duration; keep it strictly
            # positive so downstream (diarization, subtitles) never sees start == end.
            seg_end = round(seg_start + 0.01, 3)
        segments.append({"start": seg_start, "end": seg_end, "text": " ".join(words)})

    for tok, tok_start, tok_end in pairs:
        # A long pause since the previous word closes the current phrase first.
        if words and prev_end is not None and (tok_start - prev_end) > gap_thresh:
            emit(prev_end)
            words, start = [], None
        if start is None:
            start = tok_start
        words.append(tok)
        prev_end = tok_end
        if tok[-1:] in ".!?…" or (tok_end - start) >= max_dur:
            emit(tok_end)
            words, start, prev_end = [], None, None

    if words and start is not None:
        emit(prev_end if prev_end is not None else start)
    return segments


def _align_region(
    aligner,
    audio_chunk: np.ndarray,
    text: str,
    lang_code: str,
    offset: float,
) -> list[dict] | None:
    """Force-align one region's transcript, returning fine segments (or None on failure)."""
    aligner_lang = _CODE_TO_ALIGNER_LANG.get(lang_code)
    if aligner_lang is None:
        return None  # aligner doesn't support this language → caller keeps the coarse segment
    try:
        result = aligner.generate(audio_chunk, text, language=aligner_lang)
    except Exception:
        return None
    fine = _regroup_with_alignment(text, list(getattr(result, "items", [])), offset)
    return fine or None


def transcribe_regions(
    audio_pcm: np.ndarray,
    speech_regions: list[tuple[float, float]],
    *,
    language: str | None = None,
    system_prompt: str | None = None,
    model_path: str = DEFAULT_QWEN_MODEL,
    sample_rate: int = SAMPLE_RATE,
    align: bool = False,
    aligner_model: str = DEFAULT_ALIGNER_MODEL,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], str]:
    """Transcribe VAD speech regions with Qwen3-ASR.

    Slices ``audio_pcm`` at each ``(start, end)`` region and runs one Qwen3-ASR generation
    per region. By default each region becomes a single segment (coarse timestamps). When
    ``align`` is True, Qwen3-ForcedAligner is run on each region to split it into
    phrase-level segments with accurate word-derived timestamps (Whisper-like granularity,
    better for diarization and subtitles). The ``system_prompt`` is the domain context
    (see :mod:`transcribe.qwen_prompt`). ``language`` forces a transcription language
    (e.g. ``"ru"``); ``None`` lets the model auto-detect.

    Args:
        audio_pcm: Float32 mono waveform at ``sample_rate``.
        speech_regions: ``(start_s, end_s)`` tuples from the VAD stage.
        language: Language code to force, or None to auto-detect.
        system_prompt: Domain context prompt (hotwords / instructions).
        model_path: HF repo id of the MLX Qwen3-ASR model.
        sample_rate: Sample rate of ``audio_pcm`` (Hz).
        align: If True, force-align each region into phrase-level segments.
        aligner_model: HF repo id of the MLX Qwen3-ForcedAligner model.
        progress_callback: Optional ``(done_regions, total_regions) -> None`` reporter.

    Returns:
        ``(segments, detected_language)`` where each segment is
        ``{"start": float, "end": float, "text": str}``.
    """
    model = load_qwen_model(model_path)
    aligner = load_qwen_model(aligner_model) if align else None

    segments: list[dict] = []
    lang_counts: dict[str, int] = {}
    total = len(speech_regions)
    n_samples = len(audio_pcm)
    align_fallbacks = 0

    for idx, (start, end) in enumerate(speech_regions):
        start_sample = int(start * sample_rate)
        # Clamp to the decoded length: a padded/split region can extend past the audio
        # when the container's reported duration disagrees with the actual PCM. Without
        # this, an out-of-bounds slice is silently empty and that speech is lost.
        end_sample = min(int(end * sample_rate), n_samples)
        if start_sample >= n_samples:
            logger.warning(
                "Qwen: speech region %.2f-%.2fs starts past end of audio (%.2fs) — dropped",
                start, end, n_samples / sample_rate,
            )
            if progress_callback is not None:
                progress_callback(idx + 1, total)
            continue
        chunk = audio_pcm[start_sample:end_sample]

        if chunk.size > 0:
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            out = model.generate(
                chunk,
                language=language,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            # out.language is a list (one entry per internal sub-chunk); count this
            # region's first non-empty value toward a whole-file majority vote.
            langs = out.language if isinstance(out.language, list) else [out.language]
            region_lang = None
            for lang_value in langs:
                if lang_value:
                    lang_counts[lang_value] = lang_counts.get(lang_value, 0) + 1
                    region_lang = lang_value
                    break

            text = (out.text or "").strip()
            if text:
                fine = None
                if aligner is not None:
                    region_code = _normalize_language(language or region_lang)
                    fine = _align_region(aligner, chunk, text, region_code, float(start))
                    if fine is None:
                        align_fallbacks += 1
                if fine:
                    segments.extend(fine)
                else:
                    segments.append(
                        {
                            "start": round(float(start), 3),
                            "end": round(float(end), 3),
                            "text": text,
                        }
                    )

        if progress_callback is not None:
            progress_callback(idx + 1, total)

    if aligner is not None and align_fallbacks:
        logger.warning(
            "Qwen alignment fell back to coarse timestamps for %d/%d regions "
            "(unsupported language, aligner error, or token mismatch)",
            align_fallbacks, total,
        )

    if language:
        detected_language = _normalize_language(language)
    elif lang_counts:
        detected_language = _normalize_language(max(lang_counts, key=lang_counts.get))
    else:
        detected_language = "unknown"

    return segments, detected_language
