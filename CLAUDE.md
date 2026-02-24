# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# CLI transcription
python -m transcribe.cli <video_file> [--model large-v3] [--language en] [--output-dir ./out]

# Launch Gradio web UI
python -m transcribe.web

# Run tests
pytest tests/ -v
```

## Architecture

The project is a Python package (`transcribe/`) with three modules:

- **`core.py`** — Transcription engine. The main pipeline in `transcribe_video()`:
  1. Loads a whisper.cpp model via pywhispercpp (auto-detects Metal on Apple Silicon)
  2. Extracts audio as 16 kHz float32 numpy array via ffmpeg
  3. Runs WebRTC VAD (`_detect_speech_regions`) to find speech timestamps, then merges/pads them into ~30s chunks (`_merge_speech_regions`)
  4. Transcribes only speech chunks via `model.transcribe(numpy_array)` with `new_segment_callback` for progress
  5. Offsets segment timestamps by each chunk's position in the original audio
  6. Runs `_deduplicate_segments` as a final safety net against any remaining repeated segments
  7. Returns a dict with full text, timestamped segments, and detected language

  Key implementation details:
  - pywhispercpp segment timestamps (`t0`/`t1`) are in **centiseconds** — divided by 100 to get seconds
  - `model.transcribe()` accepts `np.ndarray` directly (float32, 16kHz, mono) — no temp files needed for chunks
  - Closure-in-loop: segment callbacks use default arguments (`_offset=chunk_start`) to capture per-chunk values
  - Anti-hallucination params: `no_context=True`, `no_speech_thold=0.3`, `entropy_thold=2.4`, `max_tokens=100`

- **`cli.py`** — CLI entry point via `argparse`. Calls `core.transcribe_video()` and saves TXT output.
- **`web.py`** — Gradio web UI. Uses `gr.Blocks` layout with video upload, model/language selection, custom HTML progress bar (threading + queue + generator pattern), and downloadable TXT file.

All transcription logic lives in `core.py`. Both CLI and web UI are thin wrappers around it.

## Dependencies

- `pywhispercpp` — Python bindings for whisper.cpp (requires system ffmpeg)
- `webrtcvad-wheels` — WebRTC Voice Activity Detection (lightweight C extension, no PyTorch)
- `gradio` — Web UI framework
- `numpy` — Installed as a pywhispercpp dependency
