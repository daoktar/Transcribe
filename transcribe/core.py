import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from pywhispercpp.model import Model


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
                "ffprobe", "-v", "quiet",
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


def transcribe_video(
    video_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    output_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Transcribe a video file using whisper.cpp via pywhispercpp.

    Args:
        video_path: Path to the video file.
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        language: Language code (e.g. "en"). Auto-detected if None.
        output_dir: Directory for output files. Defaults to video's directory.
        progress_callback: Optional callback ``(progress_fraction, message) -> None``
            called during transcription to report progress (0.0–1.0).

    Returns:
        Dict with keys: text, segments, language.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with: brew install ffmpeg"
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent
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
    duration = _get_media_duration(str(video_path))

    # --- Detect language if not specified ---
    if language is None:
        _report(0.04, "Detecting language...")
        t0 = time.monotonic()
        detected_language, lang_error = _detect_language(model, str(video_path))
        detect_time = time.monotonic() - t0
        if lang_error:
            _report(0.05, f"Language: unknown (detection failed, {_format_eta(detect_time)}). Transcribing...")
        else:
            _report(0.05, f"Language: {detected_language} (detected in {_format_eta(detect_time)}). Transcribing...")
    else:
        detected_language = language
        _report(0.05, f"Language: {language}. Transcribing...")

    # --- Transcribe using new_segment_callback for real-time progress ---
    transcribe_start = time.monotonic()
    segments: list[dict] = []
    full_text_parts: list[str] = []
    seg_count = 0

    def _on_new_segment(segment):
        nonlocal seg_count
        try:
            # pywhispercpp timestamps are in centiseconds (10ms units)
            seg_start = segment.t0 / 100.0
            seg_end = segment.t1 / 100.0
            seg_text = segment.text.strip()

            # Skip empty or malformed segments
            if not seg_text or seg_end <= seg_start:
                return

            segments.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": seg_text,
            })
            full_text_parts.append(seg_text)
            seg_count += 1

            # Report progress with ETA
            elapsed = time.monotonic() - transcribe_start
            if duration and duration > 0:
                fraction = min(seg_end / duration, 0.99)
                scaled = 0.05 + fraction * 0.95
                if fraction > 0.01 and elapsed > 0.5:
                    eta_seconds = elapsed / fraction * (1.0 - fraction)
                    eta_str = _format_eta(eta_seconds)
                    _report(scaled, f"Transcribing... {fraction:.0%} — ~{eta_str} remaining")
                else:
                    _report(scaled, "Transcribing...")
            else:
                # No duration available — report segment count instead
                _report(0.05, f"Transcribing... {seg_count} segments processed ({_format_eta(elapsed)} elapsed)")
        except Exception:
            # Never let a progress reporting error abort the transcription
            pass

    transcribe_kwargs = {}
    if language is not None:
        transcribe_kwargs["language"] = language

    model.transcribe(
        str(video_path),
        new_segment_callback=_on_new_segment,
        **transcribe_kwargs,
    )

    total_time = time.monotonic() - overall_start
    _report(1.0, f"Done in {_format_eta(total_time)}.")

    return {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "language": detected_language,
    }


def save_txt(result: dict, output_path: str | Path) -> Path:
    """Save transcription as plain text."""
    output_path = Path(output_path)
    output_path.write_text(result["text"], encoding="utf-8")
    return output_path


def save_json(result: dict, output_path: str | Path) -> Path:
    """Save transcription as JSON with segment timestamps."""
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
