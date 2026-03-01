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

## Speaker Detection

Optional speaker diarization identifies and labels different speakers in your transcription.

```bash
# CLI — token via environment variable only (never as CLI arg)
export HF_TOKEN=hf_YourTokenHere
python -m transcribe.cli recording.mp4 --speakers

# Auto-detect speaker count, or force it:
python -m transcribe.cli recording.mp4 --speakers --num-speakers 3
```

In the **Web UI**, check "Detect Speakers" and paste your token into the password field that appears.

Output format with speakers enabled:

```
[00:01] Speaker 1: Hello, how are you?
[00:03] Speaker 2: I'm doing great, thanks for asking!
[00:05] Speaker 1: That's wonderful to hear.
```

<details>
<summary>What is the HuggingFace token and how do I get one?</summary>

### What token do I need?

The speaker diarization feature uses **pyannote.audio** (`pyannote/speaker-diarization-3.1`), a model hosted on HuggingFace that is **gated** — you must agree to its license terms before downloading. This requires a HuggingFace access token.

### How to get the token

1. **Create a free HuggingFace account** at [huggingface.co/join](https://huggingface.co/join)
2. **Accept the model's license terms** by visiting [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and clicking "Agree and access repository"
3. **Generate an access token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a token with **read** permission

The token looks like: `hf_AbCdEfGhIjKl...`

### How the token is used

The token is used **once** to authenticate the initial model download via `Pipeline.from_pretrained(...)`. After the first run, model files are cached locally in `~/.cache/huggingface/hub/` and the download is not repeated (though the token is still needed for authentication).

### Token security

- The raw token is **never stored** in memory after use — only a SHA-256 hash is kept for cache invalidation
- The token is **never printed**, logged, or written to any output file
- In the web UI the token field is a **password input** (masked)
- On CLI there is **no `--hf-token` flag** — the `HF_TOKEN` environment variable is the only way to provide it, keeping the token out of `ps` output and shell history
- If the pyannote pipeline fails to load, the error is **re-raised without the token** in the traceback

</details>

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
