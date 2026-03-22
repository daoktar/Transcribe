# Media Transcriber

Local video & audio transcription powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Runs entirely on your machine — no cloud APIs, no data leaves your device.

## Features

- **Multi-format support** — MP4, MKV, AVI, MOV, WebM, FLV, WMV, MP3, WAV, OGG, FLAC, AAC, M4A, WMA, Opus
- **Voice Activity Detection** — WebRTC VAD skips silence and music, reducing hallucinations
- **Speaker diarization** — identify and label speakers via pyannote.audio (optional, requires HuggingFace token)
- **Multiple interfaces** — CLI and native macOS app with menu bar tray
- **5 model sizes** — tiny, base, small, medium, large-v3
- **Auto language detection** — or set manually with ISO 639-1 codes
- **Standalone macOS app** — PyInstaller packaging with .dmg distribution
- **Apple Silicon acceleration** — automatic Metal GPU via whisper.cpp

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

# Custom output directory
python -m transcribe.cli video.mp4 --output-dir ./outputs

# With speaker detection
export HF_TOKEN=hf_YourTokenHere
python -m transcribe.cli recording.mp4 --speakers --num-speakers 3
```

### Native macOS App

```bash
python -m transcribe.app
```

Runs as a native window with menu bar tray icon. Minimize-to-tray, Cmd+Q to quit.

## Speaker Diarization

Identifies different speakers using pyannote.audio. Requires a free [HuggingFace token](https://huggingface.co/settings/tokens) with read access — you must first accept the [model license](https://huggingface.co/pyannote/speaker-diarization-3.1).

```
[00:01] Speaker 1: Hello, how are you?
[00:03] Speaker 2: I'm doing great, thanks for asking!
```

**CLI:** set `HF_TOKEN` env var + `--speakers` flag. **Native app:** check "Detect Speakers" and paste token.

Token security: never stored (only SHA-256 hash cached), never logged, password-masked in UI, env-var only on CLI.

## How It Works

1. ffmpeg extracts 16 kHz mono audio
2. WebRTC VAD detects speech regions, merges into ~30s chunks
3. whisper.cpp transcribes only speech chunks (with anti-hallucination params)
4. Post-processing deduplicates repeated segments
5. Optional: pyannote maps speakers to segments via majority-vote overlap

## Building the macOS App

```bash
# Build .app bundle
bash scripts/build_macos.sh

# Build .dmg installer
bash scripts/build_macos.sh --dmg
```

Bundles ffmpeg, whisper models, and all Python dependencies into a standalone .app.

## Privacy

All processing is local. No external API keys required for transcription. Media files never leave your machine.

## Development

```bash
pytest -v
```

## License

MIT (see [LICENSE](LICENSE))
