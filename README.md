# Transcribe

Local video/audio transcription powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Runs entirely on your machine — no API keys, no cloud, no data leaves your computer.

Includes a **CLI** and a **web UI** (Gradio). Automatically uses **Metal GPU acceleration** on Apple Silicon Macs.

## Features

- **Voice Activity Detection (VAD)** — Preprocesses audio with WebRTC VAD to detect speech regions before transcription, eliminating hallucinated/repeated text on silence and music sections
- **Real-time progress** — ETA, model loading time, and per-chunk transcription status
- **Auto language detection** — Detects the spoken language automatically, or specify it manually
- **Multiple model sizes** — From `tiny` (fast, lower quality) to `large-v3` (slow, best quality)

## Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and available on PATH

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Transcribe.git
cd Transcribe

python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Transcribe with defaults (large-v3 model, auto-detect language)
python -m transcribe.cli video.mp4

# Specify model and language
python -m transcribe.cli video.mp4 --model base --language en

# Output to a specific directory
python -m transcribe.cli video.mp4 --output-dir ./output
```

Output is saved as a `.txt` file next to the video (or in `--output-dir`).

### Web UI

```bash
python -m transcribe.web
```

Opens a Gradio interface at [http://localhost:7860](http://localhost:7860) with:
- Drag-and-drop video upload
- Model and language selection
- Live progress bar with ETA
- Downloadable transcript file

## How It Works

```
Video file
  │
  ├─ ffmpeg ──► Extract 16kHz mono audio
  │
  ├─ WebRTC VAD ──► Detect speech regions (skip silence/music)
  │
  ├─ whisper.cpp ──► Transcribe each speech region
  │
  └─ Post-processing ──► Deduplicate, merge, output text
```

1. **Audio extraction** — ffmpeg converts the video to 16kHz mono PCM
2. **VAD preprocessing** — WebRTC VAD detects where speech actually is, merges nearby regions into ~30s chunks, and filters noise
3. **Transcription** — Only speech chunks are sent to whisper.cpp (via pywhispercpp), preventing hallucination on silent sections
4. **Post-processing** — A deduplication filter removes any remaining repeated segments

## Model Sizes

| Model | Size | Speed | Quality | Use case |
|-------|------|-------|---------|----------|
| `tiny` | ~75 MB | Fastest | Low | Quick drafts, testing |
| `base` | ~142 MB | Fast | Fair | Short clips |
| `small` | ~466 MB | Medium | Good | General use |
| `medium` | ~1.5 GB | Slow | Great | Longer content |
| `large-v3` | ~3 GB | Slowest | Best | Final transcriptions |

Models are downloaded automatically on first use.

## Project Structure

```
transcribe/
  core.py    # Transcription engine (VAD + whisper.cpp + post-processing)
  cli.py     # Command-line interface
  web.py     # Gradio web UI
tests/
  test_core.py   # Unit and integration tests
  test_cli.py    # CLI argument parsing tests
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
