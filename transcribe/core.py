import collections
import gc
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import webrtcvad
from pywhispercpp.constants import MODELS_DIR
from pywhispercpp.model import Model
from pywhispercpp.utils import download_model

from transcribe import qwen_engine
from transcribe.paths import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _get_media_duration(file_path: str) -> float | None:
    """Get media duration in seconds using ffprobe.

    Returns None if ffprobe fails or duration is invalid.
    """
    try:
        result = subprocess.run(
            [
                get_ffprobe_path(), "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        dur = float(result.stdout.strip())
        return dur if dur > 0 else None
    except (subprocess.SubprocessError, ValueError):
        return None


def _detect_language(model: Model, pcm: np.ndarray) -> str:
    """Spoken language of 16 kHz audio, or "" (whisper.cpp's own auto-detect) on failure."""
    try:
        (lang, _prob), _all_probs = model.auto_detect_language(pcm)
        return lang
    except Exception:
        logger.warning("Language detection failed; leaving it to whisper.cpp", exc_info=True)
        return ""


# Whisper's language ID needs a few seconds of speech: «Ок, го.» (1 s) scores p=0.70,
# while every 2 s+ utterance of the RU/EN test clips scores 0.99+.
_MIN_LANGUAGE_ID_SECONDS = 2.0

# Screens a chunk's regions for a language switch before the main model identifies them one by
# one: ~12 ms a region instead of ~0.3 s on large-v3. On 69 RU/EN test chunks it flagged every
# chunk large-v3 cuts. base is worse: it hears 6 s of Russian as Ukrainian (p=0.95).
_SCREEN_MODEL = "tiny"
# ponytail: exact size of whisper.cpp's ggml-tiny.bin (sha1 bd577a11...). If upstream ever
# republishes it, every auto-language run downloads and rejects it (screen off) until updated.
_SCREEN_MODEL_BYTES = 77_691_713
_SCREEN_DOWNLOAD_PREFIX = "ggml-tiny-download-"


def _remove_abandoned_downloads(models_dir: Path) -> None:
    """Temp dirs of screen downloads killed outright (SIGKILL, quitting the app mid-download).

    Only those nothing has written to for an hour, so a slow but live download is never touched.
    """
    for tmp in models_dir.glob(f"{_SCREEN_DOWNLOAD_PREFIX}*"):
        try:
            if time.time() - max(p.stat().st_mtime for p in [tmp, *tmp.iterdir()]) > 3600:
                shutil.rmtree(tmp)
        except OSError:
            pass  # gone meanwhile (its download finished), or not ours to delete


def _load_screen_model() -> Model | None:
    """whisper tiny for ``_language_windows``, or None: the main model then identifies every region."""
    path = Path(MODELS_DIR) / f"ggml-{_SCREEN_MODEL}.bin"
    try:
        # download_model writes in place and never checks the HTTP status, so a download cut short
        # (Ctrl-C, quitting the app) or a saved error page would sit at the model path. whisper.cpp
        # "loads" either and segfaults on first use, which no except can catch. So download beside
        # it, load the private copy, and only then move it into place, atomically. Opening the
        # shared path after our download would race a --model tiny job whose in-place download
        # started before ours finished: its open(path, "wb") truncates the file under our load.
        _remove_abandoned_downloads(path.parent)
        if path.is_file() and path.stat().st_size == _SCREEN_MODEL_BYTES:
            return Model(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=_SCREEN_DOWNLOAD_PREFIX, dir=path.parent) as tmp:
            part = Path(download_model(_SCREEN_MODEL, download_dir=tmp))  # 78 MB; raises offline
            size = part.stat().st_size
            if size != _SCREEN_MODEL_BYTES:
                raise ValueError(f"downloaded {part.name} is {size} bytes, not {_SCREEN_MODEL_BYTES}")
            model = Model(str(part))  # reads the whole file now; the rename below can't touch it
            os.replace(part, path)
        return model
    except Exception:
        logger.warning("Language screen model unavailable; identifying every region with the "
                       "main model", exc_info=True)
        return None


def _language_windows(
    model: Model,
    audio: np.ndarray,
    chunk_start: float,
    chunk_end: float,
    raw_regions: list[tuple[float, float]],
    screen: Model | None = None,
) -> list[tuple[float, float, str]]:
    """Cut a chunk where the spoken language switches: ``[(start, end, language), ...]``.

    whisper.cpp decodes one language per ``transcribe()`` call, so in a RU, EN, RU chunk it
    skips or translates the English. When the chunk has two or more VAD regions long enough
    to identify, each gets its own language ID and the chunk is cut in the pause between
    regions of different languages. Shorter regions are too unreliable to split on and join
    the language before them. Otherwise the whole chunk is identified at once, which is
    exactly the detection whisper.cpp would run itself.

    A ``screen`` model hears those regions first: when it hears one language in all of them
    and the whole-chunk ID agrees, that is the chunk's language, so the main model's
    per-region passes are only paid where the language may switch. The main model still
    decides every window.
    """
    def detect(start: float, end: float, by: Model = model) -> str:
        return _detect_language(by, audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)])

    regions = [(max(s, chunk_start), min(e, chunk_end))
               for s, e in raw_regions if s < chunk_end and e > chunk_start]
    identifiable = [e - s >= _MIN_LANGUAGE_ID_SECONDS for s, e in regions]
    if sum(identifiable) < 2:
        return [(chunk_start, chunk_end, detect(chunk_start, chunk_end))]

    # ponytail: tiny hears noisy Russian as uk/bg/pt (3 of 14 pure-RU chunks at 12 dB SNR), and
    # those chunks still pay one main-model pass per region. A gate on the chunk-level p of the
    # second language was no better: its scale moves 10-100x between clean and noisy audio.
    if screen is not None:
        heard = {detect(s, e, screen) for (s, e), ok in zip(regions, identifiable) if ok}
        if len(heard) == 1 and "" not in heard:  # "" = the screen failed
            # Alone, the whole-chunk ID leans Russian whenever any is spoken (14 of 18 English
            # test chunks with one Russian utterance), so it stands only if the screen agrees.
            whole = detect(chunk_start, chunk_end)
            if heard == {whole}:
                return [(chunk_start, chunk_end, whole)]

    langs = [detect(s, e) if ok else "" for (s, e), ok in zip(regions, identifiable)]
    prev = next((lang for lang in langs if lang), "")  # leading short regions join the first one
    for i, lang in enumerate(langs):
        prev = lang or prev
        langs[i] = prev

    windows, start = [], chunk_start
    for i in range(1, len(regions)):
        if langs[i] != langs[i - 1]:
            cut = (regions[i - 1][1] + regions[i][0]) / 2  # middle of the pause
            windows.append((start, cut, langs[i - 1]))
            start = cut
    windows.append((start, chunk_end, langs[-1]))
    return windows


def _deduplicate_segments(segments: list[dict], max_repeats: int = 2) -> list[dict]:
    """Remove consecutive duplicate segments caused by whisper hallucination.

    When whisper encounters silence, music, or unclear audio it often
    hallucinates by repeating the same phrase over and over.  This function
    keeps at most *max_repeats* consecutive segments with identical text and
    drops the rest.

    Args:
        segments: List of segment dicts with at least a ``text`` key.
        max_repeats: Maximum allowed identical consecutive segments.

    Returns:
        Filtered list of segments.
    """
    if not segments:
        return segments

    deduped: list[dict] = [segments[0]]
    repeat_count = 1

    for seg in segments[1:]:
        if seg["text"] == deduped[-1]["text"]:
            repeat_count += 1
            if repeat_count <= max_repeats:
                deduped.append(seg)
            # else: drop this duplicate
        else:
            repeat_count = 1
            deduped.append(seg)

    return deduped


# ---------------------------------------------------------------------------
# VAD (Voice Activity Detection) preprocessing
# ---------------------------------------------------------------------------

def _extract_audio_pcm(file_path: str) -> np.ndarray:
    """Extract audio from a media file as a 16 kHz float32 mono numpy array.

    Uses ffmpeg to decode the media file and pipe raw PCM to stdout,
    then converts to the float32 format expected by pywhispercpp.

    Args:
        file_path: Path to the media file (video or audio).

    Returns:
        numpy array of float32 samples at 16 kHz, normalized to [-1, 1].

    Raises:
        RuntimeError: If ffmpeg fails to extract audio.
    """
    result = subprocess.run(
        [
            get_ffmpeg_path(), "-i", str(file_path),
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # 16-bit signed little-endian PCM
            "-ar", str(SAMPLE_RATE),  # 16 kHz
            "-ac", "1",               # mono
            "-f", "s16le",            # raw PCM format (no container header)
            "pipe:1",                 # pipe to stdout
        ],
        capture_output=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio (exit code {result.returncode}): "
            f"{result.stderr[:500]}"
        )
    if not result.stdout:
        return np.array([], dtype=np.float32)

    audio_int16 = np.frombuffer(result.stdout, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def _detect_speech_regions(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    padding_duration_ms: int = 300,
) -> list[tuple[float, float]]:
    """Detect speech regions in audio using WebRTC VAD.

    Processes audio frame-by-frame through the VAD and uses a ring-buffer
    state machine to group per-frame detections into contiguous speech
    regions with start/end timestamps.

    Args:
        audio: Float32 numpy array at the given sample_rate.
        sample_rate: Audio sample rate in Hz (must be 8000, 16000, 32000, or 48000).
        aggressiveness: VAD aggressiveness mode (0–3). Higher = more aggressive
            filtering of non-speech. 2 is recommended for transcription.
        frame_duration_ms: Frame size in milliseconds (10, 20, or 30).
        padding_duration_ms: Ring-buffer window size in ms for smoothing
            speech/silence transitions.

    Returns:
        List of (start_seconds, end_seconds) tuples for each speech region.
        Returns empty list if no speech is detected.
    """
    if len(audio) == 0:
        return []

    # Convert float32 → int16 PCM bytes for webrtcvad
    pcm_bytes = (audio * 32767).astype(np.int16).tobytes()

    vad = webrtcvad.Vad(aggressiveness)

    # Frame parameters
    frame_samples = int(sample_rate * frame_duration_ms / 1000)  # 480 for 30ms@16kHz
    frame_bytes = frame_samples * 2  # 16-bit = 2 bytes per sample
    frame_duration_s = frame_duration_ms / 1000.0

    # Ring buffer for smoothing transitions
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    speech_regions: list[tuple[float, float]] = []
    region_start = 0.0

    offset = 0
    frame_index = 0

    while offset + frame_bytes <= len(pcm_bytes):
        frame = pcm_bytes[offset:offset + frame_bytes]
        is_speech = vad.is_speech(frame, sample_rate)
        timestamp = frame_index * frame_duration_s

        if not triggered:
            ring_buffer.append((timestamp, is_speech))
            num_voiced = sum(1 for _, s in ring_buffer if s)
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # Region starts at the beginning of the ring buffer window
                region_start = ring_buffer[0][0]
                ring_buffer.clear()
        else:
            ring_buffer.append((timestamp, is_speech))
            num_unvoiced = sum(1 for _, s in ring_buffer if not s)
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                # Region ends after the current frame (same grid math as region_start, so a
                # region resuming on the very next frame touches this one exactly)
                region_end = (frame_index + 1) * frame_duration_s
                speech_regions.append((region_start, region_end))
                ring_buffer.clear()

        offset += frame_bytes
        frame_index += 1

    # Close any open region at end of audio
    if triggered:
        region_end = frame_index * frame_duration_s
        speech_regions.append((region_start, region_end))

    return speech_regions


def _merge_speech_regions(
    regions: list[tuple[float, float]],
    max_gap: float = 5.0,
    max_segment: float = 30.0,
    padding: float = 0.5,
    min_duration: float = 1.0,
    total_duration: float | None = None,
) -> list[tuple[float, float]]:
    """Merge adjacent speech regions into Whisper-friendly chunks.

    Combines regions that are close together, applies padding, filters
    very short regions, and cuts overly long stretches inside pauses.

    Args:
        regions: List of (start, end) speech region timestamps in seconds.
        max_gap: Maximum gap in seconds between regions to merge them.
        max_segment: Maximum speech span per chunk in seconds (Whisper works
            best with <=30s chunks). A longer stretch is cut in its longest
            pause within reach; only unbroken speech is cut blindly.
        padding: Padding in seconds to add before/after each chunk. At a cut
            inside a stretch it stops at the middle of the pause, so
            neighbouring chunks never overlap (no phrase is transcribed twice).
        min_duration: Minimum region duration in seconds. Shorter regions
            don't start or extend a stretch (likely noise, not meaningful
            speech), but inside one they are kept and never cut through.
        total_duration: Total audio duration for clamping padding.

    Returns:
        List of merged (start, end) tuples, sorted and non-overlapping.
    """
    regions = sorted(regions)

    # Filter out very short regions (noise, not speech)
    filtered = [(s, e) for s, e in regions if (e - s) >= min_duration]
    if not filtered:
        return []

    # Merge regions separated by less than max_gap
    merged: list[list[float]] = [list(filtered[0])]
    for start, end in filtered[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    chunks: list[tuple[float, float]] = []
    i = 0
    for group_start, group_end in merged:
        # Pauses inside the stretch: gaps between all raw regions, short ones included.
        # Touching regions count too: a VAD region ends with 300 ms of hangover silence.
        pauses: list[tuple[float, float]] = []
        reach = group_start
        while i < len(regions) and regions[i][0] < group_end:
            s, e = regions[i]
            if s >= reach:
                pauses.append((reach, s))
            reach = max(reach, e)
            i += 1

        lead = max(group_start - padding, chunks[-1][1] if chunks else 0.0)
        start = group_start  # speech onset the max_segment budget counts from
        while group_end - start > max_segment:
            limit = start + max_segment
            fits = [(a, b) for a, b in pauses if start < a <= limit]
            if fits:
                # ponytail: greedy per window, favouring pieces >= max_segment/2 — whisper pays a
                # full 30 s encoder pass per chunk however short. A DP over all cuts if placement matters.
                late = [p for p in fits if p[0] - start >= max_segment / 2] or fits
                # Longest pause, latest on ties (rounded: VAD's 30 ms frames make many pauses equal)
                a, b = max(late, key=lambda p: (round(p[1] - p[0], 2), p[0]))
                # Pad both sides of the cut, but only up to the middle of the pause
                mid = (a + b) / 2
                chunks.append((lead, min(a + padding, mid)))
                lead, start = max(b - padding, mid), b
            else:
                chunks.append((lead, limit))  # unbroken speech longer than max_segment: hard cut
                lead = start = limit
        end = group_end + padding
        chunks.append((lead, min(end, total_duration) if total_duration is not None else end))

    return chunks


# ---------------------------------------------------------------------------
# Main transcription function
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    # Video
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus",
}

# Transcription range on the 0..1 progress bar (setup takes 0..0.12; diarization,
# when enabled, takes the tail 0.85..0.99).
_TRANSCRIBE_START_FRAC = 0.12


def _transcribe_whisper_regions(
    model_size: str,
    audio_pcm: np.ndarray,
    speech_regions: list[tuple[float, float]],
    raw_regions: list[tuple[float, float]],
    language: str | None,
    diarize: bool,
    report: Callable[[float, str], None],
) -> tuple[list[dict], str]:
    """Transcribe VAD speech regions with whisper.cpp (the default engine).

    Loads the model, then transcribes each region, streaming per-segment progress. Without a
    ``language``, each region is first cut where the spoken language switches
    (``_language_windows``, screened by whisper tiny) and the reported language is the one
    spoken longest.
    Returns ``(segments, detected_language)``.
    """
    # --- Load model (Metal GPU acceleration is automatic on Apple Silicon) ---
    report(_TRANSCRIBE_START_FRAC, f"Loading model '{model_size}'...")
    t0 = time.monotonic()
    model = Model(model_size, print_realtime=False, print_progress=False)
    report(_TRANSCRIBE_START_FRAC, f"Model loaded in {_format_eta(time.monotonic() - t0)}.")
    screen = _load_screen_model() if not language and model_size != _SCREEN_MODEL else None
    report(_TRANSCRIBE_START_FRAC, f"Language: {language or 'auto-detect'}. Transcribing...")

    transcribe_start = time.monotonic()
    all_segments: list[dict] = []
    total_speech_duration = sum(end - start for start, end in speech_regions)
    cumulative_speech = 0.0
    language_seconds: collections.Counter[str] = collections.Counter()

    transcribe_kwargs = {
        # --- Anti-hallucination / anti-repetition parameters ---
        "no_context": True,
        # no_context only resets between calls. Inside a chunk, each re-seeked 30 s window
        # (max_tokens forces one roughly every 15 s of Russian) was still conditioned on the
        # previous window's text, and after a language switch it skipped whole utterances.
        "n_max_text_ctx": 0,
        "no_speech_thold": 0.3,
        "entropy_thold": 2.4,
        "max_tokens": 100,
        "suppress_blank": True,
    }

    transcribe_range = 0.73 if diarize else 0.88

    for chunk_idx, (chunk_start, chunk_end) in enumerate(speech_regions):
        windows = (
            [(chunk_start, chunk_end, language)] if language
            else _language_windows(model, audio_pcm, chunk_start, chunk_end, raw_regions, screen)
        )
        chunk_segments: list[dict] = []

        for win_start, win_end, win_language in windows:
            win_audio = audio_pcm[int(win_start * SAMPLE_RATE):int(win_end * SAMPLE_RATE)]
            win_duration = win_end - win_start

            def _on_new_segment(
                segment,
                _offset=win_start,
                _win_dur=win_duration,
                _chunk_idx=chunk_idx,
            ):
                nonlocal cumulative_speech
                try:
                    # pywhispercpp timestamps are in centiseconds (10ms units)
                    seg_start = segment.t0 / 100.0 + _offset
                    seg_end = segment.t1 / 100.0 + _offset
                    seg_text = segment.text.strip()

                    if not seg_text or seg_end <= seg_start:
                        return

                    chunk_segments.append({
                        "start": round(seg_start, 3),
                        "end": round(seg_end, 3),
                        "text": seg_text,
                    })

                    elapsed = time.monotonic() - transcribe_start
                    if total_speech_duration > 0:
                        win_progress = min(
                            (segment.t1 / 100.0) / _win_dur, 1.0
                        ) if _win_dur > 0 else 1.0
                        current_speech = cumulative_speech + _win_dur * win_progress
                        fraction = min(current_speech / total_speech_duration, 0.99)
                        scaled = _TRANSCRIBE_START_FRAC + fraction * transcribe_range
                        if elapsed > 0.5 and fraction > 0.01:
                            eta_seconds = elapsed / fraction * (1.0 - fraction)
                            report(
                                scaled,
                                f"Transcribing chunk {_chunk_idx + 1}/{len(speech_regions)}"
                                f"... {fraction:.0%} — ~{_format_eta(eta_seconds)} remaining",
                            )
                        else:
                            report(
                                scaled,
                                f"Transcribing chunk {_chunk_idx + 1}/{len(speech_regions)}...",
                            )
                except Exception:
                    # Never let a progress reporting error abort the transcription
                    pass

            model.transcribe(
                win_audio,
                new_segment_callback=_on_new_segment,
                # Always pass a language: whisper.cpp's default is "en", which translates other
                # speech. "" is its own auto-detect, the fallback when detection failed ("auto"
                # works too, but pywhispercpp logs "unknown language" for it on every call).
                language=win_language,
                **transcribe_kwargs,
            )
            language_seconds[win_language] += win_duration
            cumulative_speech += win_duration

        all_segments.extend(chunk_segments)

    if not language:
        del language_seconds[""]
        language = max(language_seconds, key=language_seconds.get, default="unknown")
    return all_segments, language


def _transcribe_qwen_regions(
    audio_pcm: np.ndarray,
    speech_regions: list[tuple[float, float]],
    language: str | None,
    qwen_context: str | None,
    qwen_model: str,
    diarize: bool,
    align: bool,
    report: Callable[[float, str], None],
) -> tuple[list[dict], str]:
    """Transcribe VAD speech regions with the Qwen3-ASR MLX engine (fallback).

    Uses the distilled domain context prompt by default. When ``align`` is True, each
    region is force-aligned into phrase-level segments (finer timestamps for diarization/
    subtitles). Returns ``(segments, detected_language)``.
    """
    if qwen_context is None:
        from transcribe.qwen_prompt import load_context_prompt

        qwen_context = load_context_prompt()

    transcribe_range = 0.73 if diarize else 0.88
    align_suffix = " (+alignment)" if align else ""
    transcribe_start = time.monotonic()

    def _progress(done: int, total: int) -> None:
        fraction = min(done / total, 0.99) if total else 0.99
        scaled = _TRANSCRIBE_START_FRAC + fraction * transcribe_range
        elapsed = time.monotonic() - transcribe_start
        if elapsed > 0.5 and fraction > 0.01:
            eta = elapsed / fraction * (1.0 - fraction)
            report(scaled, f"Transcribing chunk {done}/{total} with Qwen3-ASR{align_suffix}"
                           f"... ~{_format_eta(eta)} remaining")
        else:
            report(scaled, f"Transcribing chunk {done}/{total} with Qwen3-ASR{align_suffix}...")

    # A first-run download is multi-GB and slow — say so, so a stalled bar is explainable.
    if not qwen_engine.is_model_cached(qwen_model):
        report(_TRANSCRIBE_START_FRAC, "Downloading Qwen3-ASR weights (~3.4 GB, first run only)...")
    else:
        report(_TRANSCRIBE_START_FRAC,
               "Loading Qwen3-ASR-1.7B" + (" + ForcedAligner" if align else "") + " model...")
    return qwen_engine.transcribe_regions(
        audio_pcm,
        speech_regions,
        language=language,
        system_prompt=qwen_context,
        model_path=qwen_model,
        align=align,
        progress_callback=_progress,
    )


def transcribe_media(
    media_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    output_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    diarize: bool = False,
    num_speakers: int | None = None,
    hf_token: str | None = None,
    engine: str = "whisper",
    qwen_context: str | None = None,
    qwen_model: str = qwen_engine.DEFAULT_QWEN_MODEL,
    qwen_align: bool | None = None,
    allow_qwen_fallback: bool = True,
) -> dict:
    """Transcribe a media file (video or audio).

    Uses WebRTC VAD to detect speech regions first, then transcribes only
    those regions to avoid hallucination on silence/music/noise sections.

    Two engines are available. The default ``"whisper"`` runs whisper.cpp via
    pywhispercpp (Metal-accelerated). ``"qwen"`` runs Qwen3-ASR-1.7B via MLX and is
    also used automatically as a fallback when the whisper engine fails (unless
    ``allow_qwen_fallback`` is False). The VAD, deduplication and diarization stages
    are shared by both engines.

    Args:
        media_path: Path to the media file (video or audio).
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        language: Language code (e.g. "en"). Auto-detected if None.
        output_dir: Directory for output files. Defaults to file's directory.
        progress_callback: Optional callback ``(progress_fraction, message) -> None``
            called during transcription to report progress (0.0–1.0).
        diarize: If True, run speaker diarization after transcription.
        num_speakers: Force a specific speaker count (only used when diarize=True).
        hf_token: HuggingFace access token (required when diarize=True).
        engine: Transcription engine, ``"whisper"`` (default) or ``"qwen"``.
        qwen_context: Domain context prompt for Qwen3-ASR. Defaults to the distilled
            prompt in :mod:`transcribe.qwen_prompt` when the Qwen engine is used.
        qwen_model: HF repo id of the MLX Qwen3-ASR model to use for the Qwen engine.
        qwen_align: For the Qwen engine, force-align each region into phrase-level segments
            (finer timestamps for diarization/subtitles). None (default) auto-enables this
            only when ``diarize`` is True; True/False overrides.
        allow_qwen_fallback: If True, fall back to Qwen3-ASR when the whisper engine
            raises (and MLX is available). Set False to surface whisper errors.

    Returns:
        Dict with keys: text, segments, language.
        When diarize=True, also includes "speakers" (int) and each segment
        has a "speaker" field (e.g. "Speaker 1").
    """
    if not shutil.which(get_ffmpeg_path()):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with: brew install ffmpeg"
        )

    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    ext = media_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if output_dir is None:
        output_dir = media_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.monotonic()

    def _report(fraction: float, msg: str):
        if progress_callback is not None:
            progress_callback(fraction, msg)

    # --- Get media duration for ETA calculation ---
    # (The transcription model is loaded lazily by the selected engine below, so the
    # shared VAD preprocessing is ready before either engine — including the fallback.)
    duration = _get_media_duration(str(media_path))

    # --- Extract audio for VAD analysis ---
    _report(0.03, "Extracting audio for speech detection...")
    t0 = time.monotonic()
    audio_pcm = _extract_audio_pcm(str(media_path))
    extract_time = time.monotonic() - t0
    _report(0.06, f"Audio extracted in {_format_eta(extract_time)}.")

    # --- Run VAD to find speech regions ---
    _report(0.06, "Detecting speech regions...")
    t0 = time.monotonic()
    raw_regions = _detect_speech_regions(audio_pcm)
    speech_regions = _merge_speech_regions(raw_regions, total_duration=duration)
    vad_time = time.monotonic() - t0
    _report(0.10, f"Found {len(speech_regions)} speech regions in {_format_eta(vad_time)}.")

    # --- Edge case: no speech detected ---
    if not speech_regions:
        total_time = time.monotonic() - overall_start
        _report(1.0, f"No speech detected. Done in {_format_eta(total_time)}.")
        return {
            "text": "",
            "segments": [],
            "language": language or "unknown",
            "engine": (engine or "whisper").lower(),
        }

    # --- Transcribe via the selected engine (whisper default; Qwen3-ASR fallback) ---
    engine_name = (engine or "whisper").lower()
    # Alignment produces fine per-phrase timestamps; auto-enable it when diarizing.
    align = diarize if qwen_align is None else qwen_align
    engine_used = engine_name
    fell_back = False
    if engine_name == "qwen":
        all_segments, detected_language = _transcribe_qwen_regions(
            audio_pcm, speech_regions, language, qwen_context, qwen_model, diarize, align, _report,
        )
    else:
        try:
            all_segments, detected_language = _transcribe_whisper_regions(
                model_size, audio_pcm, speech_regions, raw_regions, language, diarize, _report,
            )
        except MemoryError:
            # Never "recover" from memory exhaustion by loading an even larger model.
            raise
        except Exception as exc:
            # Only fall back when Qwen is usable AND already downloaded — an unattended
            # job must not kick off a multi-GB download precisely when things are failing.
            can_fallback = (
                allow_qwen_fallback
                and qwen_engine.is_available()
                and qwen_engine.is_model_cached(qwen_model)
            )
            if not can_fallback:
                raise
            # Preserve the real cause: log it and free whisper's (native/Metal) buffers
            # before loading Qwen so the two model stacks don't co-reside. The in-flight
            # traceback keeps the failed Model alive via its frame locals — clearing them
            # is what actually lets Model.__del__ release the Metal context here.
            logger.exception("Whisper engine failed; falling back to Qwen3-ASR")
            traceback.clear_frames(exc.__traceback__)
            gc.collect()
            _report(
                _TRANSCRIBE_START_FRAC,
                f"Whisper engine failed ({type(exc).__name__}); falling back to Qwen3-ASR-1.7B...",
            )
            try:
                all_segments, detected_language = _transcribe_qwen_regions(
                    audio_pcm, speech_regions, language, qwen_context, qwen_model, diarize, align, _report,
                )
            except Exception as qexc:
                raise qexc from exc
            engine_used, fell_back = "qwen", True

    # --- Post-processing ---
    # VAD chunks no longer overlap, but a segment can still start a little before its
    # chunk (Qwen's prompt-echo retry moves the window 100 ms earlier) — sort by start so
    # deduplication, diarization and any subtitle output see a monotonic timeline.
    all_segments.sort(key=lambda s: (s["start"], s["end"]))
    # Deduplication remains as a second safety net after VAD filtering.
    cleaned_segments = _deduplicate_segments(all_segments, max_repeats=2)
    removed = len(all_segments) - len(cleaned_segments)

    # --- Optional speaker diarization ---
    speaker_count = 0
    diarize_error = None
    if diarize and cleaned_segments:
        from transcribe.diarize import diarize as run_diarize

        def _diarize_progress(frac: float, msg: str) -> None:
            _report(0.85 + frac * 0.14, msg)

        _report(0.85, "Starting speaker diarization...")
        try:
            cleaned_segments, speaker_count = run_diarize(
                audio_pcm,
                cleaned_segments,
                hf_token=hf_token,
                num_speakers=num_speakers,
                progress_callback=_diarize_progress,
            )
        except Exception as exc:
            # Diarization failed — keep the plain transcript intact
            diarize_error = str(exc)
            _report(0.99, f"Speaker detection failed: {diarize_error}")

    total_time = time.monotonic() - overall_start
    done_msg = f"Done in {_format_eta(total_time)}."
    if removed:
        done_msg += f" ({removed} duplicate segments removed)"
    done_msg += f" ({len(speech_regions)} speech regions transcribed)"
    _report(1.0, done_msg)

    result = {
        "text": " ".join(seg["text"] for seg in cleaned_segments),
        "segments": cleaned_segments,
        "language": detected_language,
        "engine": engine_used,
    }
    if fell_back:
        # The transcript came from a different engine/prompt than requested — make that
        # auditable in the saved output, the web view and retry-diarize.
        result["fallback"] = True
    if diarize:
        result["speakers"] = speaker_count
    if diarize_error:
        result["diarize_error"] = diarize_error
    return result


def save_txt(result: dict, output_path: str | Path) -> Path:
    """Save transcription as plain text.

    When segments contain speaker labels, formats each line as
    ``Speaker N: text``. Otherwise saves plain text.
    """
    output_path = Path(output_path)

    segments = result.get("segments", [])
    has_speakers = segments and "speaker" in segments[0]

    if has_speakers:
        lines = []
        for seg in segments:
            lines.append(f"{seg['speaker']}: {seg['text']}")
        output_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        output_path.write_text(result["text"], encoding="utf-8")

    return output_path


def save_txt_alongside(result: dict, original_media_path: str | Path) -> Path:
    """Save transcript as .txt next to the original media file.

    Derives output path from the media path (e.g. video.mp4 → video.txt).
    If the target file already exists, appends a numeric suffix (_1, _2, …).

    Raises PermissionError or OSError if the directory is not writable.
    """
    media = Path(original_media_path)
    base = media.parent / media.stem
    candidate = base.with_suffix(".txt")
    counter = 1
    while candidate.exists():
        candidate = media.parent / f"{media.stem}_{counter}.txt"
        counter += 1
    return save_txt(result, candidate)


def retry_diarize(
    media_path: str,
    result: dict,
    hf_token: str,
    num_speakers: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Retry speaker diarization on an already-transcribed result.

    Re-extracts audio via ffmpeg (fast) and runs only the diarization step,
    skipping the full whisper transcription.  Returns a new result dict with
    speaker labels applied, or with ``diarize_error`` if it fails again.
    """
    from transcribe.diarize import diarize as run_diarize

    def _report(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, msg)

    segments = result.get("segments", [])
    if not segments:
        return result

    _report(0.0, "Extracting audio for speaker detection...")
    audio_pcm = _extract_audio_pcm(media_path)

    _report(0.15, "Running speaker diarization...")

    def _diarize_progress(frac: float, msg: str) -> None:
        _report(0.15 + frac * 0.84, msg)

    try:
        labeled_segments, speaker_count = run_diarize(
            audio_pcm,
            segments,
            hf_token=hf_token,
            num_speakers=num_speakers,
            progress_callback=_diarize_progress,
        )
    except Exception as exc:
        _report(1.0, f"Speaker detection failed: {exc}")
        return {
            **result,
            "diarize_error": str(exc),
            "speakers": 0,
        }

    _report(1.0, f"Done — {speaker_count} speaker{'s' if speaker_count != 1 else ''}.")
    new_result = {
        **result,
        "segments": labeled_segments,
        "text": " ".join(seg["text"] for seg in labeled_segments),
        "speakers": speaker_count,
    }
    new_result.pop("diarize_error", None)
    return new_result
