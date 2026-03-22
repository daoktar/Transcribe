import collections
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import webrtcvad
from pywhispercpp.model import Model

from transcribe.paths import get_ffmpeg_path, get_ffprobe_path

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


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS timestamp."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


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


def _detect_language(model: Model, media_path: str) -> tuple[str, str | None]:
    """Detect the spoken language of a media file.

    Returns (language_code, error_message). error_message is None on success.
    """
    try:
        (lang, _prob), _all_probs = model.auto_detect_language(str(media_path))
        return lang, None
    except Exception as exc:
        return "unknown", f"Language detection failed: {type(exc).__name__}: {exc}"


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
                # Region ends at current position
                region_end = timestamp + frame_duration_s
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
    very short regions, and splits overly long regions.

    Args:
        regions: List of (start, end) speech region timestamps in seconds.
        max_gap: Maximum gap in seconds between regions to merge them.
        max_segment: Maximum segment duration in seconds (Whisper works
            best with <=30s chunks).
        padding: Padding in seconds to add before/after each region.
        min_duration: Minimum region duration in seconds. Shorter regions
            are filtered out (likely noise, not meaningful speech).
        total_duration: Total audio duration for clamping padding.

    Returns:
        List of merged (start, end) tuples.
    """
    if not regions:
        return []

    # Filter out very short regions (noise, not speech)
    filtered = [(s, e) for s, e in regions if (e - s) >= min_duration]
    if not filtered:
        return []

    # Sort by start time
    filtered.sort(key=lambda r: r[0])

    # Merge regions separated by less than max_gap
    merged: list[list[float]] = [list(filtered[0])]
    for start, end in filtered[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    # Split any region longer than max_segment
    split: list[tuple[float, float]] = []
    for start, end in merged:
        while (end - start) > max_segment:
            split.append((start, start + max_segment))
            start += max_segment
        split.append((start, end))

    # Apply padding (clamped to valid range)
    padded: list[tuple[float, float]] = []
    for start, end in split:
        padded_start = max(0.0, start - padding)
        padded_end = end + padding
        if total_duration is not None:
            padded_end = min(padded_end, total_duration)
        padded.append((padded_start, padded_end))

    return padded


# ---------------------------------------------------------------------------
# Main transcription function
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    # Video
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus",
}


def transcribe_media(
    media_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    output_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    diarize: bool = False,
    num_speakers: int | None = None,
    hf_token: str | None = None,
) -> dict:
    """Transcribe a media file (video or audio) using whisper.cpp via pywhispercpp.

    Uses WebRTC VAD to detect speech regions first, then transcribes only
    those regions to avoid hallucination on silence/music/noise sections.

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

    # --- Load model (Metal GPU acceleration is automatic on Apple Silicon) ---
    _report(0.0, f"Loading model '{model_size}'...")
    t0 = time.monotonic()
    model = Model(model_size, print_realtime=False, print_progress=False)
    load_time = time.monotonic() - t0
    _report(0.03, f"Model loaded in {_format_eta(load_time)}.")

    # --- Get media duration for ETA calculation ---
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
        }

    # --- Detect language if not specified (on original file) ---
    if language is None:
        _report(0.10, "Detecting language...")
        t0 = time.monotonic()
        detected_language, lang_error = _detect_language(model, str(media_path))
        detect_time = time.monotonic() - t0
        if lang_error:
            _report(0.12, f"Language: unknown (detection failed, {_format_eta(detect_time)}). Transcribing...")
        else:
            _report(0.12, f"Language: {detected_language} (detected in {_format_eta(detect_time)}). Transcribing...")
    else:
        detected_language = language
        _report(0.12, f"Language: {language}. Transcribing...")

    # --- Transcribe each speech region ---
    transcribe_start = time.monotonic()
    all_segments: list[dict] = []

    # Total speech duration for progress tracking
    total_speech_duration = sum(end - start for start, end in speech_regions)
    cumulative_speech = 0.0

    transcribe_kwargs = {
        # --- Anti-hallucination / anti-repetition parameters ---
        "no_context": True,
        "no_speech_thold": 0.3,
        "entropy_thold": 2.4,
        "max_tokens": 100,
        "suppress_blank": True,
    }
    if language is not None:
        transcribe_kwargs["language"] = language

    for chunk_idx, (chunk_start, chunk_end) in enumerate(speech_regions):
        # Slice the audio numpy array for this chunk
        start_sample = int(chunk_start * SAMPLE_RATE)
        end_sample = int(chunk_end * SAMPLE_RATE)
        chunk_audio = audio_pcm[start_sample:end_sample]

        chunk_segments: list[dict] = []
        chunk_duration = chunk_end - chunk_start

        def _on_new_segment(
            segment,
            _offset=chunk_start,
            _chunk_dur=chunk_duration,
            _chunk_idx=chunk_idx,
        ):
            nonlocal cumulative_speech
            try:
                # pywhispercpp timestamps are in centiseconds (10ms units)
                # Add chunk offset to get absolute timestamps
                seg_start = segment.t0 / 100.0 + _offset
                seg_end = segment.t1 / 100.0 + _offset
                seg_text = segment.text.strip()

                # Skip empty or malformed segments
                if not seg_text or seg_end <= seg_start:
                    return

                chunk_segments.append({
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "text": seg_text,
                })

                # Report progress across all chunks
                elapsed = time.monotonic() - transcribe_start
                if total_speech_duration > 0:
                    chunk_progress = min(
                        (segment.t1 / 100.0) / _chunk_dur, 1.0
                    ) if _chunk_dur > 0 else 1.0
                    current_speech = cumulative_speech + _chunk_dur * chunk_progress
                    fraction = min(current_speech / total_speech_duration, 0.99)
                    transcribe_range = 0.73 if diarize else 0.88
                    scaled = 0.12 + fraction * transcribe_range
                    if elapsed > 0.5 and fraction > 0.01:
                        eta_seconds = elapsed / fraction * (1.0 - fraction)
                        eta_str = _format_eta(eta_seconds)
                        _report(
                            scaled,
                            f"Transcribing chunk {_chunk_idx + 1}/{len(speech_regions)}"
                            f"... {fraction:.0%} — ~{eta_str} remaining",
                        )
                    else:
                        _report(
                            scaled,
                            f"Transcribing chunk {_chunk_idx + 1}/{len(speech_regions)}...",
                        )
            except Exception:
                # Never let a progress reporting error abort the transcription
                pass

        model.transcribe(
            chunk_audio,
            new_segment_callback=_on_new_segment,
            **transcribe_kwargs,
        )

        all_segments.extend(chunk_segments)
        cumulative_speech += chunk_duration

    # --- Post-processing: remove hallucinated repetition ---
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
    }
    if diarize:
        result["speakers"] = speaker_count
    if diarize_error:
        result["diarize_error"] = diarize_error
    return result


# Backward-compatible alias
transcribe_video = transcribe_media


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
