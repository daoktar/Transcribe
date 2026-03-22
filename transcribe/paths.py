"""Resource path resolution for both development and bundled (.app) modes."""

from __future__ import annotations

import sys
from pathlib import Path


def is_bundled() -> bool:
    """Return True when running inside a PyInstaller .app bundle."""
    return getattr(sys, "frozen", False)


def get_base_dir() -> Path:
    """Return the base directory for the transcribe package."""
    if is_bundled():
        return Path(sys._MEIPASS) / "transcribe"
    return Path(__file__).parent


def get_assets_dir() -> Path:
    """Return the path to the assets directory."""
    return get_base_dir() / "assets"


def get_ffmpeg_path() -> str:
    """Return path to ffmpeg, preferring the bundled copy."""
    if is_bundled():
        bundled = Path(sys._MEIPASS) / "ffmpeg"
        if bundled.exists():
            return str(bundled)
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Return path to ffprobe, preferring the bundled copy."""
    if is_bundled():
        bundled = Path(sys._MEIPASS) / "ffprobe"
        if bundled.exists():
            return str(bundled)
    return "ffprobe"
