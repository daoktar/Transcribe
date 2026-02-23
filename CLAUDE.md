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
```

## Architecture

The project is a Python package (`transcribe/`) with three modules:

- **`core.py`** — Transcription engine. `transcribe_video()` loads a faster-whisper model with auto-detected device (CUDA/CPU), runs transcription, and returns a dict with full text, timestamped segments, and detected language. `save_txt()` and `save_json()` handle output serialization.
- **`cli.py`** — CLI entry point via `argparse`. Calls `core.transcribe_video()` and saves output files.
- **`web.py`** — Gradio web UI. Uses `gr.Blocks` layout with video upload, model/language selection, and downloadable output files.

All transcription logic lives in `core.py`. Both CLI and web UI are thin wrappers around it.

## Dependencies

- `faster-whisper` — CTranslate2-based Whisper implementation (requires system ffmpeg)
- `gradio` — Web UI framework
