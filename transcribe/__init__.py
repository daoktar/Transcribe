import os
from pathlib import Path

# Load .env file from project root (if present) so HF_TOKEN and other
# env vars are available across all entry points (CLI, web, native app).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: parse .env manually if python-dotenv is not installed
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.is_file():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip("'\"")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

from transcribe.core import transcribe_media, transcribe_video, save_txt, save_txt_alongside

__all__ = ["transcribe_media", "transcribe_video", "save_txt", "save_txt_alongside"]

try:
    from transcribe.diarize import diarize as speaker_diarize
    __all__.append("speaker_diarize")
except ImportError:
    pass
