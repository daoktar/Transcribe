# Transcribe

Video transcription tool using [whisper.cpp](https://github.com/ggml-org/whisper.cpp) via [pywhispercpp](https://github.com/absadiki/pywhispercpp). Supports CLI and web UI.

Automatically uses **Metal GPU acceleration** on Apple Silicon Macs for fast local transcription.

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on PATH (`brew install ffmpeg`)

## Install

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Transcribe with defaults (large-v3 model, auto-detect language, output TXT + JSON)
python -m transcribe.cli video.mp4

# Specify model and language
python -m transcribe.cli video.mp4 --model base --language en

# Output only JSON to a specific directory
python -m transcribe.cli video.mp4 --format json --output-dir ./output
```

### Web UI

```bash
python -m transcribe.web
```

Opens a Gradio interface at `http://localhost:7860` with drag-and-drop video upload.

## Output Formats

- **TXT** — Plain text transcript
- **JSON** — Full transcript with per-segment timestamps (`start`, `end`, `text`) and detected language
