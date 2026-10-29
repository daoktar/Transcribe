"""Speaker diarization using pyannote.audio.

Assigns speaker labels to transcription segments by running the
pyannote speaker-diarization pipeline on the same audio, then
mapping speaker turns to segment timestamps via majority-vote overlap.
"""

import hashlib
import os
from collections import Counter
from collections.abc import Callable

import numpy as np

from transcribe import gpu, qwen_engine

# pyannote.audio 4 ships OpenTelemetry usage metrics, on by default, posting to
# otel.pyannote.ai. This app makes no network calls beyond model downloads, so force
# them off — pyannote reads the variable when it is first imported (lazily, below).
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

SAMPLE_RATE = 16000

# pyannote 4's recommended open pipeline. One gated repo: it bundles the segmentation,
# embedding and PLDA models (loading the legacy 3.1 pipeline on pyannote 4 would need
# community-1's PLDA anyway, on top of the 3.1 and segmentation-3.0 licenses).
PIPELINE_ID = "pyannote/speaker-diarization-community-1"

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
            pipeline = Pipeline.from_pretrained(PIPELINE_ID, token=hf_token)
            if pipeline is None:
                # pyannote 4 returns None instead of raising for some download failures
                raise LookupError(f"no pipeline returned for {PIPELINE_ID}")
            # Use the GPU when there is one: Metal on Apple Silicon, CUDA on NVIDIA
            if torch.backends.mps.is_available():
                pipeline.to(torch.device("mps"))
            elif torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            _pipeline = pipeline
        except Exception as exc:
            # Reset cache so a subsequent call with a new token retries
            _pipeline = None
            _pipeline_token_hash = None
            # Sanitize: strip any token value from the error message
            err_msg = str(exc).replace(hf_token, "***")
            raise RuntimeError(
                f"Failed to load pyannote pipeline: {err_msg}\n"
                "Accept the model's user conditions at:\n"
                f"  https://huggingface.co/{PIPELINE_ID}"
            ) from None
        _pipeline_token_hash = token_hash
    return _pipeline


@gpu.exclusive_call
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
    # Qwen (with its aligner, since diarizing turns alignment on) peaks at 3.3 GB: on a 4 GB
    # card pyannote would run out of memory beside it, so drop it first.
    qwen_engine.release_gpu_models()
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

    # pyannote.audio 4 returns a DiarizeOutput wrapping the Annotation
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
