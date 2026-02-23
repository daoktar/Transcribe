# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# CLI transcription
python -m transcribe.cli <video_file> [--model large-v3] [--language en] [--output-dir ./out] [--format all]

# Launch Gradio web UI
python -m transcribe.web

# Run tests
pytest tests/ -v
```

## Architecture

The project is a Python package (`transcribe/`) with three modules:

- **`core.py`** — Transcription engine. `transcribe_video()` loads a whisper.cpp model via pywhispercpp (auto-detects Metal on Apple Silicon), runs transcription with real-time progress via `new_segment_callback`, and returns a dict with full text, timestamped segments, and detected language. `save_txt()` and `save_json()` handle output serialization. Note: pywhispercpp segment timestamps (`t0`/`t1`) are in centiseconds — divided by 100 to get seconds.
- **`cli.py`** — CLI entry point via `argparse`. Calls `core.transcribe_video()` and saves output files.
- **`web.py`** — Gradio web UI. Uses `gr.Blocks` layout with video upload, model/language selection, progress bar with ETA, and downloadable output files.

All transcription logic lives in `core.py`. Both CLI and web UI are thin wrappers around it.

## Dependencies

- `pywhispercpp` — Python bindings for whisper.cpp (requires system ffmpeg)
- `gradio` — Web UI framework
