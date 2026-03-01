"""Speaker diarization using pyannote.audio.

Assigns speaker labels to transcription segments by running the
pyannote speaker-diarization pipeline on the same audio, then
mapping speaker turns to segment timestamps via majority-vote overlap.
"""

import hashlib
from collections import Counter
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16000

_pipeline = None
_pipeline_token_hash: str | None = None


def _load_pipeline(hf_token: str):
    """Load the pyannote diarization pipeline (cached after first call).

    The pipeline is re-loaded only if the token changes.  Only a SHA-256
    hash of the token is kept in memory for cache invalidation — the raw
    token is never stored beyond the ``Pipeline.from_pretrained`` call.
    """
    from pyannote.audio import Pipeline

    import torch

    global _pipeline, _pipeline_token_hash
    token_hash = hashlib.sha256(hf_token.encode()).hexdigest()
    if _pipeline is None or _pipeline_token_hash != token_hash:
        try:
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
            # Use Metal GPU on Apple Silicon for faster inference
            if torch.backends.mps.is_available():
                _pipeline.to(torch.device("mps"))
        except Exception as exc:
            # Reset cache so a subsequent call with a new token retries
            _pipeline = None
            _pipeline_token_hash = None
            # Sanitize: strip any token value from the error message
            err_msg = str(exc).replace(hf_token, "***")
            raise RuntimeError(
                f"Failed to load pyannote pipeline: {err_msg}\n"
                "Accept user conditions for BOTH required models:\n"
                "  1. https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  2. https://huggingface.co/pyannote/segmentation-3.0"
            ) from None
        _pipeline_token_hash = token_hash
    return _pipeline


def diarize(
    audio: np.ndarray,
    segments: list[dict],
    hf_token: str,
    num_speakers: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], int]:
    """Run speaker diarization and assign labels to transcription segments.

    Args:
        audio: Full audio as float32 numpy array at 16 kHz mono.
        segments: Transcription segments from core.transcribe_media().
            Each dict has keys: "start", "end", "text".
        hf_token: HuggingFace access token for pyannote model.
        num_speakers: Force a specific speaker count, or None for auto-detect.
        progress_callback: Optional (fraction, message) callback.
            Fraction range: 0.0 to 1.0 (within the diarization phase).

    Returns:
        Tuple of (labeled_segments, speaker_count).
        labeled_segments is a new list of segment dicts with an added
        "speaker" field (e.g. "Speaker 1"). Originals are not mutated.
    """
    import torch

    def _report(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, msg)

    if not segments:
        return [], 0

    # --- Load model ---
    _report(0.0, "Loading speaker diarization model...")
    pipeline = _load_pipeline(hf_token)

    # --- Run diarization ---
    _report(0.2, "Detecting speakers...")
    waveform = torch.from_numpy(audio[np.newaxis, :]).float()
    audio_dict = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    output = pipeline(audio_dict, **kwargs)

    # pyannote.audio v4+ returns DiarizeOutput; v3 returns Annotation directly
    annotation = getattr(output, "speaker_diarization", output)

    # --- Collect speaker turns ---
    _report(0.7, "Assigning speaker labels...")
    speaker_turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))

    if not speaker_turns:
        # No speakers detected — label everything as Speaker 1
        labeled = [{**seg, "speaker": "Speaker 1"} for seg in segments]
        _report(1.0, "Done — 1 speaker.")
        return labeled, 1

    # --- Build first-appearance ordering for speaker labels ---
    seen_order: list[str] = []
    for _, _, spk in speaker_turns:
        if spk not in seen_order:
            seen_order.append(spk)

    label_map = {raw: f"Speaker {i + 1}" for i, raw in enumerate(seen_order)}

    # --- Assign speakers to segments via majority-vote overlap ---
    labeled_segments: list[dict] = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find overlapping speaker turns
        overlapping: list[str] = []
        for t_start, t_end, spk in speaker_turns:
            if t_start < seg_end and t_end > seg_start:
                # Weight by overlap duration
                overlap = min(seg_end, t_end) - max(seg_start, t_start)
                overlapping.extend([spk] * max(1, int(overlap * 10)))

        if overlapping:
            winner = Counter(overlapping).most_common(1)[0][0]
        else:
            # No overlap — use nearest speaker turn by center distance
            seg_center = (seg_start + seg_end) / 2
            nearest = min(
                speaker_turns,
                key=lambda t: abs((t[0] + t[1]) / 2 - seg_center),
            )
            winner = nearest[2]

        labeled_segments.append({**seg, "speaker": label_map[winner]})

    speaker_count = len(seen_order)
    _report(1.0, f"Done — {speaker_count} speaker{'s' if speaker_count != 1 else ''}.")
    return labeled_segments, speaker_count
