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
  3. Runs WebRTC VAD (`_detect_speech_regions`) to find speech timestamps, then merges them into ~30s chunks (`_merge_speech_regions`). A longer stretch is cut inside its longest pause (a blind cut only through >30 s of unbroken speech) and padding at a cut stops at the middle of the pause, so chunks never overlap — overlapping chunks made both engines transcribe the boundary phrase twice
  4. Transcribes only speech chunks via `model.transcribe(numpy_array)` with `new_segment_callback` for progress
  5. Offsets segment timestamps by each chunk's position in the original audio
  6. Runs `_deduplicate_segments` as a final safety net against any remaining repeated segments
  7. Returns a dict with full text, timestamped segments, and detected language

  Key implementation details:
  - pywhispercpp segment timestamps (`t0`/`t1`) are in **centiseconds** — divided by 100 to get seconds
  - `model.transcribe()` accepts `np.ndarray` directly (float32, 16kHz, mono) — no temp files needed for chunks
  - Closure-in-loop: segment callbacks use default arguments (`_offset=win_start`) to capture per-window values
  - `language` is always passed to `model.transcribe()`: whisper.cpp's own default is `en` (it translates other speech), and it decodes one language per call — in a RU, EN, RU chunk it skips or translates the English. Without `--language`, `_language_windows` cuts each chunk where the language switches: in a chunk with two or more VAD regions ≥2 s, whisper tiny (`_load_screen_model`, ~12 ms a region) hears each region first. If it hears one language and the main model's whole-chunk ID agrees, that is the chunk's language. The whole-chunk ID alone is not enough: it leans Russian whenever any Russian is spoken, and would translate an English chunk. Otherwise (tiny hears more than one language, fails, or disagrees) each region gets `model.auto_detect_language` from the main model (one encoder pass each, ~0.3 s on large-v3), and the chunk is cut in the pause between regions of different languages; shorter regions, and regions identified below p=0.5 (`_MIN_SWITCH_PROB`: Russian jargon heard as en/pt/pl), join the language before them. These regions are the level-2 VAD regions cut at the pauses a level-3 VAD hears inside them (`_split_at_pauses`, only where both sides stay ≥2 s): level 2 bridges the 0.7-1.2 s pause between a Russian and an English sentence, while level 3's own regions are trimmed too tight to identify. Chunking stays on level 2. Any other chunk is identified whole (the same detection whisper.cpp runs itself). tiny (78 MB) is downloaded on first use, and never in place. pywhispercpp's `download_model` writes straight to the model path and never checks the HTTP status. A download cut short (Ctrl-C, quitting the app) or a saved error page would then sit at the path, and whisper.cpp "loads" either and segfaults on first use, which no `except` catches. So `_load_screen_model` uses a cached file only when it is exactly `_SCREEN_MODEL_BYTES`. Otherwise it downloads into a temp dir inside the models dir, loads the `Model` from that private copy, and only then `os.replace`s it into place. Loading from the shared path instead would race a `--model tiny` job whose in-place download started first, because its `open(path, "wb")` truncates the file mid-load. It never deletes the path, since another job may be writing it. A download killed outright (quitting the app) leaves its `ggml-tiny-download-*` temp dir behind. The next load removes such a dir once nothing has written to it for an hour. Offline, or on any failure, the main model hears every region (slower, same output). Each window gets its language explicitly; `""` (whisper.cpp's auto-detect) is only the fallback when detection fails. The reported language is the one with the most window time
  - Anti-hallucination params: `no_context=True`, `n_max_text_ctx=0` (no conditioning between the 30 s windows inside a chunk; `no_context` only resets between calls), `no_speech_thold=0.3`, `entropy_thold=2.4`, `max_tokens=100`
  - pywhispercpp's `_set_params` applies only the kwargs it is given, and they persist on the Model across `transcribe()` calls: an omitted param is the C default (first call) or whatever the last call set. Never pass one conditionally; when probing params by hand, pass every compared param on each call
  - A VAD region ends with 300 ms of hangover silence, so even touching raw regions meet at a real pause — `_merge_speech_regions` cuts there; region starts and ends share the same frame-grid math so touching compares exactly
  - **Engine dispatch:** `transcribe_media(engine=...)` selects the transcriber. The shared VAD/dedup/diarize scaffolding wraps two per-region transcribers: `_transcribe_whisper_regions` (default) and `_transcribe_qwen_regions`. If the whisper path raises and `allow_qwen_fallback` is set (and MLX is available), it falls back to Qwen3-ASR automatically. Both return `(segments, detected_language)`.

- **`qwen_engine.py`** — Qwen3-ASR-1.7B fallback engine via MLX (`mlx-audio`), Apple Silicon only. `transcribe_regions()` runs one `model.generate(chunk, language=, system_prompt=)` per VAD region. By default each region → one coarse segment. With `align=True` it also runs **Qwen3-ForcedAligner-0.6B** per region (`_align_region` + `_regroup_with_alignment`) to split into phrase-level segments with word-derived timestamps — the aligner strips punctuation, so its per-word timestamps are zipped back onto the original punctuated tokens and split on sentence punctuation / pauses / max length. `is_available()` guards the optional MLX import; both models cached in `_MODEL_CACHE`. Language names normalized to ISO codes. Core auto-enables alignment when `diarize=True` (or via `qwen_align`); `transcribe_media` sorts all segments by start afterward (chunks don't overlap, but the echo retry can move a window 100 ms earlier).
  - **Prompt echo:** Qwen3-ASR occasionally decodes the `system_prompt` instead of the audio (a short prompt leaks in as a prefix; a long one replaces the transcription outright). It is unrelated to silence — reproduced on a region with 98% voiced frames — and sits on a knife edge: the same audio transcribes correctly when the window shifts by 100 ms, or when the prompt is withheld. Two layers handle it: `_is_prompt_echo` flags a region whose text repeats 8+ prompt words verbatim and `_retranscribe_echo_region` re-runs it (window 100 ms earlier, then later, then prompt-free) so the speech is recovered rather than lost; `_strip_prompt_echo` is the net under it, dropping leftover echo segments only when they cover ≥30% of the prompt across the whole file. Both log a WARNING. Without the retry, alignment smears one echoing region into dozens of word-segments interleaved with real speech.
- **`qwen_prompt.py`** — loads the domain context prompt (Qwen's `system_prompt`) from a plain-text file **`qwen_context_prompt.txt`** (not hardcoded), so it can be retuned without editing Python or rebuilding. `load_context_prompt()` resolves first hit of: `$TRANSCRIBE_QWEN_PROMPT_FILE` → `<user config>/qwen_context_prompt.txt` (see `paths.get_user_config_dir`) → the bundled default → a built-in fallback; read fresh per transcription. `DEFAULT_CONTEXT_PROMPT` remains as an import-time snapshot for compat. Distilled from real meeting transcripts (kontur.talk syncs); keeps English tech terms in Latin.
- **`cli.py`** — CLI entry point via `argparse` (`--engine whisper|qwen`). Calls `core.transcribe_media()` and saves TXT output.
- **`web.py`** — FastAPI backend. REST API + SSE for progress streaming. Serves the frontend from `static/`. Key endpoints: `/api/upload`, `/api/transcribe/{job_id}`, `/api/progress/{job_id}` (SSE), `/api/result/{job_id}/{file_index}`, `/api/download/{job_id}`. In-memory job state with `asyncio.Queue` bridged from background threads for SSE. `TrustedHostMiddleware` rejects any Host other than `127.0.0.1`/`localhost` (DNS-rebinding guard). Temp transcripts live in the job's own temp dir; `Job.txt_paths` is keyed by file index. Provides `create_app()` returning a FastAPI app.
- **`static/`** — Frontend: plain HTML/CSS/JS (no frameworks). Single-page state machine (setup → working → reading) in `index.html`; `style.css` is a warm-paper "galley proof" editorial theme (JetBrains Mono chrome + system serif content, CSS variables); `app.js` (vanilla JS with SSE, drag-and-drop, native pywebview integration). All assets self-hosted — CSP blocks external fetches and data: URIs.
- **`app.py`** — Native macOS app. Launches the FastAPI/uvicorn server in a daemon thread and displays it inside a native WKWebView window via pywebview, with menu-bar tray integration via PyObjC.
- **`tray.py`** — macOS menu bar (system tray) integration via PyObjC.

All transcription logic lives in `core.py`. CLI and native app are thin wrappers around it.

## Dependencies

- `pywhispercpp` — Python bindings for whisper.cpp (requires system ffmpeg)
- `mlx-audio` — MLX speech-to-text (Qwen3-ASR fallback engine); Apple Silicon only, gated in `requirements.txt` by `platform_machine == "arm64"`. Pins `transformers>=5.5,<5.13` (mlx-lm 0.31.3 breaks on transformers 5.13+).
- `webrtcvad-wheels` — WebRTC Voice Activity Detection (lightweight C extension, no PyTorch)
- `pyannote.audio` 4.x (+ `torch`/`torchaudio` ≥2.8) — speaker diarization with the gated `pyannote/speaker-diarization-community-1` pipeline. 3.x cannot work on current torchaudio/huggingface_hub. pyannote 4's telemetry is on by default; `diarize.py` forces `PYANNOTE_METRICS_ENABLED=false`. `lightning` and its `pytorch-lightning` mirror are floored at 2.6.6 (PYSEC-2026-3624, PYSEC-2026-3967).
- `fastapi` + `uvicorn` — Backend web framework and ASGI server
- `python-multipart` — File upload handling for FastAPI
- `pywebview` — Native WKWebView window for the macOS app
- `pyobjc-framework-Cocoa` — macOS menu bar integration
- `numpy` — Installed as a pywhispercpp dependency
