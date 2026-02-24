# Transcribe

Local video/audio transcription powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

- Runs on your machine (no cloud API required)
- CLI and web UI (Gradio)
- Speech-region detection (WebRTC VAD) to reduce silence/music hallucinations
- Apple Silicon acceleration via Metal (when available)

## Quick Start

### 1. Install prerequisites

- Python 3.10+
- `ffmpeg` on your `PATH`

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 2. Install project

```bash
git clone https://github.com/YOUR_USERNAME/Transcribe.git
cd Transcribe

python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

pip install -r requirements.txt
```

### 3. Run transcription

```bash
python -m transcribe.cli /path/to/video.mp4
```

Output is a `.txt` transcript saved next to the input file (or in `--output-dir` if provided).

## CLI Usage

```bash
# Default model (large-v3), auto language detect
python -m transcribe.cli video.mp4

# Faster model + fixed language
python -m transcribe.cli video.mp4 --model base --language en

# Write output to a specific directory
python -m transcribe.cli video.mp4 --output-dir ./outputs
```

Available models: `tiny`, `base`, `small`, `medium`, `large-v3`.

## Web UI Usage

```bash
python -m transcribe.web
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860).

Security default: the web app is bound to `127.0.0.1` with `share=False`, so it is local-only unless you explicitly change code/settings.

## How It Works

1. `ffmpeg` extracts 16 kHz mono PCM audio.
2. WebRTC VAD finds speech regions.
3. whisper.cpp transcribes only speech chunks.
4. Post-processing deduplicates repeated hallucinated segments.

## Privacy and Security

- No external transcription API keys are required.
- Media is processed locally.
- See [SECURITY.md](SECURITY.md) for vulnerability reporting and hardening guidance.

Important: this project depends on external binaries/libraries (`ffmpeg`, whisper runtime). Keep your OS packages and Python dependencies up to date.

## What Is Ignored In Git

The repository is configured to avoid committing local/user artifacts:

- Media files (`*.mp4`, `*.wav`, etc.)
- Transcript outputs (`*.txt`, `*.srt`, `*.vtt`)
- Working/cache dirs (`output/`, `outputs/`, `data/`, `tmp/`, `cache/`, etc.)
- Local secrets (`.env*`, key/cert files)

If you need to version a specific artifact for docs/tests, add an explicit exception rule in `.gitignore`.

## Development

Run tests:

```bash
pytest -v
```

## License

MIT (see [LICENSE](LICENSE))
