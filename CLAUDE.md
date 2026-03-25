# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# CLI transcription (video or audio)
python -m transcribe.cli <media_file> [--model large-v3] [--language en] [--output-dir ./out]

# Launch web UI (standalone, browser)
python -m transcribe.web

# Launch native macOS app
python -m transcribe.app

# Run tests
pytest tests/ -v
```

## Architecture

The project is a Python package (`transcribe/`) with these modules:

- **`core.py`** — Transcription engine. Supports video (mp4, mkv, avi, mov, webm, flv, wmv) and audio (mp3, wav, ogg, flac, aac, m4a, wma, opus) files. The main pipeline in `transcribe_media()`:
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

- **`cli.py`** — CLI entry point via `argparse`. Calls `core.transcribe_media()` and saves TXT output.
- **`web.py`** — FastAPI backend. REST API + SSE for progress streaming. Serves the frontend from `static/`. Key endpoints: `/api/upload`, `/api/transcribe/{job_id}`, `/api/progress/{job_id}` (SSE), `/api/result/{job_id}/{file_index}`, `/api/download/{job_id}`. In-memory job state with `asyncio.Queue` bridged from background threads for SSE. Provides `create_app()` returning a FastAPI app.
- **`static/`** — Frontend: plain HTML/CSS/JS (no frameworks). `index.html` (3-tab SPA), `style.css` (dark navy theme with CSS variables), `app.js` (vanilla JS with SSE, drag-and-drop, native pywebview integration).
- **`app.py`** — Native macOS app. Launches the FastAPI/uvicorn server in a daemon thread and displays it inside a native WKWebView window via pywebview, with menu-bar tray integration via PyObjC.
- **`tray.py`** — macOS menu bar (system tray) integration via PyObjC.

All transcription logic lives in `core.py`. CLI and native app are thin wrappers around it.

## Dependencies

- `pywhispercpp` — Python bindings for whisper.cpp (requires system ffmpeg)
- `webrtcvad-wheels` — WebRTC Voice Activity Detection (lightweight C extension, no PyTorch)
- `fastapi` + `uvicorn` — Backend web framework and ASGI server
- `python-multipart` — File upload handling for FastAPI
- `pywebview` — Native WKWebView window for the macOS app
- `pyobjc-framework-Cocoa` — macOS menu bar integration
- `numpy` — Installed as a pywhispercpp dependency
