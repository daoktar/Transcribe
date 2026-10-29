# Media Transcriber

Local video & audio transcription powered by [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Runs entirely on your machine — no cloud APIs, no data leaves your device.

## Features

- **Multi-format support** — MP4, MKV, AVI, MOV, WebM, FLV, WMV, MP3, WAV, OGG, FLAC, AAC, M4A, WMA, Opus
- **Voice Activity Detection** — WebRTC VAD skips silence and music, reducing hallucinations
- **Speaker diarization** — identify and label speakers via pyannote.audio (optional, requires HuggingFace token)
- **Multiple interfaces** — CLI, standalone web UI, and native macOS app (dark navy theme with Upload/Process/Review tabs) with menu bar tray
- **5 model sizes** — tiny, base, small, medium, large-v3
- **Two engines** — Qwen3-ASR-1.7B (default on Apple Silicon via MLX and on NVIDIA GPUs via CUDA)
  with a domain context prompt, and whisper.cpp as the alternative (and the default elsewhere)
- **Auto language detection** — or set manually with ISO 639-1 codes
- **Standalone macOS app** — PyInstaller packaging with .dmg distribution
- **Apple Silicon acceleration** — automatic Metal GPU via whisper.cpp (and MLX for Qwen3-ASR)
- **NVIDIA GPU acceleration (Linux)** — `scripts/setup.sh` builds whisper.cpp with CUDA and
  installs the CUDA backend of Qwen3-ASR; fits 4 GB cards (see [NVIDIA GPU](#nvidia-gpu-linux))

## Engines

**Qwen3-ASR-1.7B is the default engine** on Apple Silicon and on Linux with an NVIDIA GPU set
up by `scripts/setup.sh` (CLI, web UI and the macOS app).
**whisper.cpp large-v3 is the alternative**: pick it with `--engine whisper` or the **ENGINE**
dropdown. Where neither MLX nor CUDA is available (Intel Macs, Linux without a GPU) whisper is the default, and if whisper
fails the app falls back to Qwen automatically.

<details>
<summary>Model comparison on real recordings — 2026-10-17</summary>

Local A/B on ~110 min of our own audio (two Russian work meetings, Russian/English street
interviews and two language lessons), all engines on the same VAD chunks with `--language ru`.
There was no human reference transcript, so the numbers are proxies plus a manual read of the
passages where the engines disagreed most.

| | Qwen3-ASR 1.7B | whisper large-v3 | whisper large-v3-turbo + prompt | Breeze-ASR-25 |
|---|---|---|---|---|
| Decoder loops (chunks) | **0** | **0** | 3 | ~200 |
| Speed (RTF, M4 Max) | 0.10 | 0.08 | **0.03** | 0.12–0.14 |
| English speech kept in English | mostly | no (translated) | yes | yes |
| Main problem | some chunks lose punctuation; occasionally translates English | translates English into Russian | drops whole chunks, invents “subtitles by …” captions | loops, glues sentences |

- **Qwen3-ASR** gave the most complete text with no loops or glued sentences, so it became the default.
- **whisper large-v3-turbo** is 3–4× faster but dropped a 30-second chunk and looped on a meeting.
- **Breeze-ASR-25** (Whisper fine-tune for English IT terms) is unusable for Russian.
- **Diarization:** NVIDIA Nemotron-3-Diarization matched pyannote on small calls (same 3 speakers)
  and was ~30× faster, but it caps at 8 speakers (pyannote found 13 on a large demo). pyannote stays.

</details>

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

Or use the bundled scripts (also shipped in each release as `MediaTranscriber-<version>-cli-web.zip`):

```bash
scripts/setup.sh                      # create .venv and install requirements
scripts/web.sh                        # web UI at http://127.0.0.1:7860
scripts/transcribe.sh meeting.mp4 --language ru --speakers   # CLI, same flags as below
```

## Usage

### CLI

```bash
# Default (Qwen3-ASR on Apple Silicon, auto language)
python -m transcribe.cli video.mp4

# whisper.cpp instead (large-v3 by default)
python -m transcribe.cli video.mp4 --engine whisper

# Faster whisper model, fixed language
python -m transcribe.cli video.mp4 --engine whisper --model base --language en

# Russian meeting with speakers (what we use day to day)
python -m transcribe.cli meeting.mp4 --language ru --speakers

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

Identifies different speakers using pyannote.audio. Requires a free [HuggingFace token](https://huggingface.co/settings/tokens) with read access — you must first accept the [model's user conditions](https://huggingface.co/pyannote/speaker-diarization-community-1). pyannote's built-in usage telemetry is switched off by the app.

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
   - **qwen** (default on Apple Silicon and NVIDIA GPUs) — Qwen3-ASR-1.7B via MLX or CUDA with a
     domain context prompt
   - **whisper** — whisper.cpp with anti-hallucination params; without `--language` it identifies
     the language per region and splits chunks at RU/EN switches
4. Post-processing deduplicates repeated segments
5. Optional: pyannote maps speakers to segments via majority-vote overlap

## Qwen3-ASR Engine

[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (Apache-2.0) runs locally on
Apple Silicon via [MLX](https://github.com/ml-explore/mlx) (no PyTorch) or on an NVIDIA GPU via
PyTorch/CUDA and the official [`qwen-asr`](https://pypi.org/project/qwen-asr/) package — no cloud. It is
stronger than whisper.cpp on Russian and, crucially, on **Russian-English code-switching**:
it accepts a free-text *context prompt* that biases recognition toward your domain vocabulary
and keeps English technical terms in Latin script instead of transliterating them.

- It is the **default engine** in the CLI, web UI and macOS app wherever MLX or CUDA is available
  (`transcribe_media` itself still defaults to `engine="whisper"` for API compatibility).
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

Requires Apple Silicon or an NVIDIA GPU. The MLX dependencies install automatically on `arm64`
macOS via `requirements.txt`; the CUDA ones come from `scripts/setup.sh` (below). On other
platforms the app runs whisper-only.

## NVIDIA GPU (Linux)

`scripts/setup.sh` detects an NVIDIA GPU (`nvidia-smi`) and then also:

1. installs the cu126 build of PyTorch **before** `requirements.txt` — the default wheel targets
   the newest CUDA and does not run on older drivers such as Debian 13's 550 (CUDA 12.4);
   override with `TORCH_INDEX_URL`;
2. installs [`requirements-cuda.txt`](requirements-cuda.txt): `qwen-asr` and `bitsandbytes`;
3. rebuilds pywhispercpp from source with CUDA (`scripts/build_whisper_cuda.sh`, ~15 min).

It needs the CUDA compiler: `sudo apt install nvidia-cuda-toolkit`. `TRANSCRIBE_CPU_ONLY=1`
skips the GPU path.

Small cards are handled automatically. On an 11-minute English video on a 4 GB RTX 3050 Laptop:

| | Loaded as | VRAM | Time |
|---|---|---|---|
| whisper large-v3, CPU (i7-11370H) | f16 | — | 48 min |
| whisper large-v3, CUDA | `large-v3-q5_0` (GPUs under 6 GB) | 1.1 GB model | 55 s |
| Qwen3-ASR-1.7B, CUDA | text decoder in 4-bit NF4 (GPUs under 8 GB) | 2.5 GB, 3.3 GB with aligner | 94 s |

The full bf16 Qwen checkpoint (4.7 GB) and f16 large-v3 (3.1 GB) do not fit in 4 GB. 8-bit Qwen
fits too but runs ~8× slower than NF4. Force a mode with `TRANSCRIBE_QWEN_CUDA_QUANT=nf4|bf16`.

## Building the macOS App

```bash
# Install build tools
pip install pyinstaller
brew install create-dmg   # optional, for .dmg creation

# Build .app bundle + .dmg
bash scripts/build_macos.sh
```

The build excludes unused extras (matplotlib, PIL, onnxruntime, gRPC) and torch subpackages (CUDA, distributed, JIT, ONNX) to minimize bundle size; scipy, sklearn and pandas stay because pyannote.audio 4 needs them, and package metadata (`*.dist-info`) stays because transformers checks dependency versions at import. A post-build cleanup step removes test data and type stubs.

The spec also collects the MLX stack (`mlx`, `mlx_audio`, `mlx_lm`) plus the dynamically-imported `mlx_audio.stt.models.qwen3_asr` / `qwen3_forced_aligner` modules so the Qwen3-ASR engine and fallback work in the packaged app (they are invisible to PyInstaller's static analysis otherwise). After any dependency change, rebuild and run one transcription through the built `.app` with the default engine (qwen), with **diarization on**, and with `engine=whisper` — a green `python -m transcribe.app` or pytest run does not exercise the bundle (the 1.5.0 build only worked after this caught missing modules).

To override the bundled ffmpeg: `FFMPEG_BIN=/path/to/ffmpeg FFPROBE_BIN=/path/to/ffprobe bash scripts/build_macos.sh`

## Privacy

All processing is local. No external API keys required for transcription. Media files never leave your machine.

## Architecture

- **`transcribe/core.py`** — Transcription pipeline (WebRTC VAD + engine dispatch + optional pyannote diarization)
- **`transcribe/qwen_engine.py`** — Qwen3-ASR-1.7B engine + ForcedAligner (MLX on Apple Silicon, CUDA on NVIDIA)
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
