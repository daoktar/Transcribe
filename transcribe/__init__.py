# Load .env from project root (if present) so HF_TOKEN and other env vars are
# available across all entry points (CLI, web, native app). python-dotenv is a
# pinned dependency, so no manual-parse fallback is needed.
from dotenv import load_dotenv

load_dotenv()

from transcribe.core import save_txt, save_txt_alongside, transcribe_media

__all__ = ["transcribe_media", "save_txt", "save_txt_alongside"]

try:
    from transcribe.diarize import diarize as speaker_diarize
    __all__.append("speaker_diarize")
except ImportError:
    pass
