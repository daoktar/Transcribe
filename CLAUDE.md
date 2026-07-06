# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# CLI transcription (video or audio)
python -m transcribe.cli <media_file> [--model large-v3] [--engine whisper|qwen] [--language en] [--output-dir ./out]

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
  - **Engine dispatch:** `transcribe_media(engine=...)` selects the transcriber. The shared VAD/dedup/diarize scaffolding wraps two per-region transcribers: `_transcribe_whisper_regions` (default) and `_transcribe_qwen_regions`. If the whisper path raises and `allow_qwen_fallback` is set (and MLX is available), it falls back to Qwen3-ASR automatically. Both return `(segments, detected_language)`.

- **`qwen_engine.py`** — Qwen3-ASR-1.7B fallback engine via MLX (`mlx-audio`), Apple Silicon only. `transcribe_regions()` runs one `model.generate(chunk, language=, system_prompt=)` per VAD region. By default each region → one coarse segment. With `align=True` it also runs **Qwen3-ForcedAligner-0.6B** per region (`_align_region` + `_regroup_with_alignment`) to split into phrase-level segments with word-derived timestamps — the aligner strips punctuation, so its per-word timestamps are zipped back onto the original punctuated tokens and split on sentence punctuation / pauses / max length. `is_available()` guards the optional MLX import; both models cached in `_MODEL_CACHE`. Language names normalized to ISO codes. Core auto-enables alignment when `diarize=True` (or via `qwen_align`); `transcribe_media` sorts all segments by start afterward (VAD sub-chunks overlap when a long region is split).
- **`qwen_prompt.py`** — loads the domain context prompt (Qwen's `system_prompt`) from a plain-text file **`qwen_context_prompt.txt`** (not hardcoded), so it can be retuned without editing Python or rebuilding. `load_context_prompt()` resolves first hit of: `$TRANSCRIBE_QWEN_PROMPT_FILE` → `<user config>/qwen_context_prompt.txt` (see `paths.get_user_config_dir`) → the bundled default → a built-in fallback; read fresh per transcription. `DEFAULT_CONTEXT_PROMPT` remains as an import-time snapshot for compat. Distilled from real meeting transcripts (kontur.talk syncs); keeps English tech terms in Latin.
- **`cli.py`** — CLI entry point via `argparse` (`--engine whisper|qwen`). Calls `core.transcribe_media()` and saves TXT output.
- **`web.py`** — FastAPI backend. REST API + SSE for progress streaming. Serves the frontend from `static/`. Key endpoints: `/api/upload`, `/api/transcribe/{job_id}`, `/api/progress/{job_id}` (SSE), `/api/result/{job_id}/{file_index}`, `/api/download/{job_id}`. In-memory job state with `asyncio.Queue` bridged from background threads for SSE. Provides `create_app()` returning a FastAPI app.
- **`static/`** — Frontend: plain HTML/CSS/JS (no frameworks). `index.html` (3-tab SPA), `style.css` (dark navy theme with CSS variables), `app.js` (vanilla JS with SSE, drag-and-drop, native pywebview integration).
- **`app.py`** — Native macOS app. Launches the FastAPI/uvicorn server in a daemon thread and displays it inside a native WKWebView window via pywebview, with menu-bar tray integration via PyObjC.
- **`tray.py`** — macOS menu bar (system tray) integration via PyObjC.

All transcription logic lives in `core.py`. CLI and native app are thin wrappers around it.

## Dependencies

- `pywhispercpp` — Python bindings for whisper.cpp (requires system ffmpeg)
- `mlx-audio` — MLX speech-to-text (Qwen3-ASR fallback engine); Apple Silicon only, gated in `requirements.txt` by `platform_machine == "arm64"`. Pins `transformers>=5.5,<5.13` (mlx-lm 0.31.3 breaks on transformers 5.13+).
- `webrtcvad-wheels` — WebRTC Voice Activity Detection (lightweight C extension, no PyTorch)
- `fastapi` + `uvicorn` — Backend web framework and ASGI server
- `python-multipart` — File upload handling for FastAPI
- `pywebview` — Native WKWebView window for the macOS app
- `pyobjc-framework-Cocoa` — macOS menu bar integration
- `numpy` — Installed as a pywhispercpp dependency
