"""Domain context prompt for the Qwen3-ASR fallback engine.

Qwen3-ASR accepts a free-text *context* (passed as ``system_prompt``) that biases
recognition toward the expected domain, vocabulary and speaker names. This is the single
most effective accuracy lever the model exposes — and, unlike Whisper, it can be told to
keep English technical terms in Latin script instead of transliterating them into Cyrillic.

The prompt text lives in a plain-text file (``qwen_context_prompt.txt``), NOT in this
module, so it can be retuned without editing Python or rebuilding the app. Resolution order
(first hit wins):

  1. ``$TRANSCRIBE_QWEN_PROMPT_FILE`` — explicit path override.
  2. ``<user config>/qwen_context_prompt.txt`` — user override that survives app updates
     (see :func:`transcribe.paths.get_user_config_dir`).
  3. the default file shipped next to this module / bundled in the .app.
  4. a minimal built-in fallback, only if no file can be read.

The default was distilled from the user's real meeting transcripts and Obsidian summaries
(Avito ad-moderation and duplicate-detection syncs recorded in kontur.talk).
"""
from __future__ import annotations

import os
from pathlib import Path

from transcribe.paths import get_base_dir, get_user_config_dir

PROMPT_FILENAME = "qwen_context_prompt.txt"

# Used only if no prompt file can be read — keeps the code-switch instruction alive.
_FALLBACK_PROMPT = (
    "Рабочие созвоны с русской речью и большим количеством английских IT-терминов "
    "вперемешку (код-свитчинг). Английские технические термины сохраняй латиницей, "
    "не транслитерируй в кириллицу."
)


def context_prompt_path() -> Path:
    """Return the path of the context-prompt file that will be used (may not exist)."""
    override = os.environ.get("TRANSCRIBE_QWEN_PROMPT_FILE")
    if override:
        return Path(override)
    user_file = get_user_config_dir() / PROMPT_FILENAME
    if user_file.is_file():
        return user_file
    return get_base_dir() / PROMPT_FILENAME


def load_context_prompt() -> str:
    """Read the domain context prompt from the resolved file (read fresh each call).

    Reading per call means editing the file takes effect on the next transcription
    without restarting the app. Falls back to a minimal built-in prompt if unreadable.
    """
    try:
        text = context_prompt_path().read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    return _FALLBACK_PROMPT


# Backwards-compatible module attribute — a snapshot read at import time.
DEFAULT_CONTEXT_PROMPT = load_context_prompt()
