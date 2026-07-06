# Media Transcriber

Local video & audio transcription powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Runs entirely on your machine — no cloud APIs, no data leaves your device.

## Features

- **Multi-format support** — MP4, MKV, AVI, MOV, WebM, FLV, WMV, MP3, WAV, OGG, FLAC, AAC, M4A, WMA, Opus
- **Voice Activity Detection** — WebRTC VAD skips silence and music, reducing hallucinations
- **Speaker diarization** — identify and label speakers via pyannote.audio (optional, requires HuggingFace token)
- **Multiple interfaces** — CLI, standalone web UI, and native macOS app (dark navy theme with Upload/Process/Review tabs) with menu bar tray
- **5 model sizes** — tiny, base, small, medium, large-v3
- **Qwen3-ASR fallback engine** — optional Qwen3-ASR-1.7B (via MLX, Apple Silicon) with a
  domain context prompt; stronger on Russian and Russian-English code-switching, and used
  automatically if the whisper engine fails
- **Auto language detection** — or set manually with ISO 639-1 codes
- **Standalone macOS app** — PyInstaller packaging with .dmg distribution
- **Apple Silicon acceleration** — automatic Metal GPU via whisper.cpp (and MLX for Qwen3-ASR)

## Quick Start

**Prerequisites:** Python 3.10+ and ffmpeg on your PATH.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

```bash
git clone https://github.com/daoktar/Transcribe.git
cd Transcribe
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Default (large-v3 model, auto language)
python -m transcribe.cli video.mp4

# Faster model, fixed language
python -m transcribe.cli video.mp4 --model base --language en

# Qwen3-ASR-1.7B engine (Apple Silicon; better for Russian / RU-EN code-switching)
python -m transcribe.cli meeting.mp4 --engine qwen --language ru

# Custom output directory
python -m transcribe.cli video.mp4 --output-dir ./outputs

# With speaker detection
export HF_TOKEN=hf_YourTokenHere
python -m transcribe.cli recording.mp4 --speakers --num-speakers 3
```

### Web UI

```bash
python -m transcribe.web
```

Opens a standalone web interface at `http://127.0.0.1:7860` with a dark navy theme. Three-tab workflow:

- **Upload** — Drag-and-drop or browse for files, configure model/language/speaker detection
- **Process** — Real-time progress bar with SSE streaming, color-coded job queue with per-file status badges
- **Review** — File-switcher pills for multi-file navigation, compact inline summary bar, monospace transcript viewer with copy/download/retry actions

Responsive layout adapts to mobile (375px+) with stacked buttons, abbreviated labels, and 44px touch targets.

### Native macOS App

```bash
python -m transcribe.app
```

Opens a native WKWebView window running the same web UI. Includes menu bar tray icon, minimize-to-tray, and Cmd+Q to quit.

## Speaker Diarization

Identifies different speakers using pyannote.audio. Requires a free [HuggingFace token](https://huggingface.co/settings/tokens) with read access — you must first accept the [model license](https://huggingface.co/pyannote/speaker-diarization-3.1).

```
[00:01] Speaker 1: Hello, how are you?
[00:03] Speaker 2: I'm doing great, thanks for asking!
```

### Providing Your Token

The app looks for a HuggingFace token in this order (first match wins):

1. **Web UI field** — paste into the "HuggingFace Token" input (least preferred; token transits through the browser)
2. **Environment variable** — `export HF_TOKEN=hf_...` before launching
3. **`.env` file** — create a `.env` file in the project root (see `.env.example`):
   ```
   HF_TOKEN=hf_YourTokenHere
   ```

**CLI:** set `HF_TOKEN` env var + `--speakers` flag.

Token security: never stored persistently (only SHA-256 hash cached for model reuse), never logged, password-masked in UI.

## How It Works

1. ffmpeg extracts 16 kHz mono audio
2. WebRTC VAD detects speech regions, merges into ~30s chunks
3. The selected engine transcribes only speech chunks:
   - **whisper** (default) — whisper.cpp with anti-hallucination params
   - **qwen** — Qwen3-ASR-1.7B via MLX with a domain context prompt
4. Post-processing deduplicates repeated segments
5. Optional: pyannote maps speakers to segments via majority-vote overlap

## Qwen3-ASR Fallback Engine

[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (Apache-2.0) runs locally on
Apple Silicon via [MLX](https://github.com/ml-explore/mlx) — no PyTorch, no cloud. It is
stronger than whisper.cpp on Russian and, crucially, on **Russian-English code-switching**:
it accepts a free-text *context prompt* that biases recognition toward your domain vocabulary
and keeps English technical terms in Latin script instead of transliterating them.

- Select it in the web/native UI (**ASR ENGINE** dropdown), with `--engine qwen` (CLI), or
  `engine="qwen"` (`transcribe_media`).
- It is also used **automatically as a fallback** when the whisper engine fails
  (disable with `allow_qwen_fallback=False`).
- The context prompt lives in an editable text file,
  [`transcribe/qwen_context_prompt.txt`](transcribe/qwen_context_prompt.txt) (not in code), tuned
  for the maintainer's meeting domain. Retune it without touching Python: edit that file, set
  `$TRANSCRIBE_QWEN_PROMPT_FILE` to your own file, or drop a
  `qwen_context_prompt.txt` in `~/Library/Application Support/Media Transcriber/` to override the
  bundled default (survives app updates). You can also pass a per-call `qwen_context`.
- **Fine timestamps:** Qwen transcribes a whole ~30 s speech region at once, so by default each
  region is one coarse segment. When diarizing (or with `qwen_align=True`),
  [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) is run per
  region to split it into phrase-level segments with accurate word-derived timestamps —
  restoring Whisper-like granularity for speaker attribution and subtitles.
- Weights (`mlx-community/Qwen3-ASR-1.7B-bf16`, ~3.4 GB; aligner ~1.8 GB) download from Hugging
  Face on first use. bf16 is recommended: lower quantizations noticeably degrade Russian accuracy.

Requires Apple Silicon; the MLX dependencies install automatically on `arm64` macOS via
`requirements.txt`. On other platforms the app runs whisper-only.

## Building the macOS App

```bash
# Install build tools
pip install pyinstaller
brew install create-dmg   # optional, for .dmg creation

# Build .app bundle + .dmg
bash scripts/build_macos.sh
```

The build excludes unused pyannote transitive dependencies (scipy, sklearn, pandas, matplotlib, onnxruntime) and torch subpackages (CUDA, distributed, JIT, ONNX) to minimize bundle size. A post-build cleanup step removes test data, metadata, and type stubs.

The spec also collects the MLX stack (`mlx`, `mlx_audio`, `mlx_lm`) plus the dynamically-imported `mlx_audio.stt.models.qwen3_asr` / `qwen3_forced_aligner` modules so the Qwen3-ASR engine and fallback work in the packaged app (they are invisible to PyInstaller's static analysis otherwise). After any dependency change, rebuild and launch the `.app` once with `engine=qwen` **and** with a forced whisper failure to confirm the fallback path — a green `python -m transcribe.app` run does not exercise the bundle.

To override the bundled ffmpeg: `FFMPEG_BIN=/path/to/ffmpeg FFPROBE_BIN=/path/to/ffprobe bash scripts/build_macos.sh`

## Privacy

All processing is local. No external API keys required for transcription. Media files never leave your machine.

## Architecture

- **`transcribe/core.py`** — Transcription pipeline (WebRTC VAD + engine dispatch + optional pyannote diarization)
- **`transcribe/qwen_engine.py`** — Qwen3-ASR-1.7B fallback engine + ForcedAligner (MLX, Apple Silicon)
- **`transcribe/qwen_prompt.py`** — Loads the Qwen context prompt from `qwen_context_prompt.txt`
- **`transcribe/qwen_context_prompt.txt`** — Editable domain context prompt (vocabulary/names)
- **`transcribe/cli.py`** — CLI entry point
- **`transcribe/web.py`** — FastAPI backend (REST API + SSE progress streaming)
- **`transcribe/static/`** — Frontend (vanilla HTML/CSS/JS, no frameworks)
- **`transcribe/app.py`** — Native macOS app (pywebview + uvicorn)
- **`transcribe/tray.py`** — macOS menu bar integration (PyObjC)

## Development

```bash
pytest -v
```

## License

MIT (see [LICENSE](LICENSE))
