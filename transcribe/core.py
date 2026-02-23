import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel


def get_device_and_compute():
    """Auto-detect the best device and compute type.

    CTranslate2 supports cuda and cpu. On macOS (no CUDA), this returns cpu/int8.
    """
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def transcribe_video(
    video_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    output_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Transcribe a video file using faster-whisper.

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

    def _report(fraction: float, msg: str):
        if progress_callback is not None:
            progress_callback(fraction, msg)

    _report(0.0, f"Loading model '{model_size}'...")

    device, compute_type = get_device_and_compute()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    _report(0.05, "Model loaded. Starting transcription...")

    segments_iter, info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=5,
    )

    duration = info.duration  # total audio duration in seconds
    start_time = time.monotonic()

    segments = []
    full_text_parts = []
    for segment in segments_iter:
        segments.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        })
        full_text_parts.append(segment.text.strip())

        # Report progress with ETA
        if duration and duration > 0:
            fraction = min(segment.end / duration, 0.99)
            # Scale into the 0.05–1.0 range (first 5% was model loading)
            scaled = 0.05 + fraction * 0.95
            elapsed = time.monotonic() - start_time
            if fraction > 0.01:
                eta_seconds = elapsed / fraction * (1.0 - fraction)
                eta_str = _format_eta(eta_seconds)
                _report(scaled, f"Transcribing... {fraction:.0%} — ~{eta_str} remaining")
            else:
                _report(scaled, "Transcribing...")

    _report(1.0, "Done.")

    result = {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "language": info.language,
    }
    return result


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
