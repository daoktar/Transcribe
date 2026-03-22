# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Media Transcriber macOS .app bundle."""

import os
import shutil
from PyInstaller.utils.hooks import collect_all, collect_data_files

# ---------------------------------------------------------------------------
# Disable broken contrib hook for webrtcvad (incompatible with webrtcvad-wheels)
# ---------------------------------------------------------------------------
import PyInstaller.config
_hooks_dir = os.path.join(
    os.path.dirname(__import__("_pyinstaller_hooks_contrib").__file__),
    "stdhooks",
)
_bad_hook = os.path.join(_hooks_dir, "hook-webrtcvad.py")
if os.path.exists(_bad_hook):
    os.rename(_bad_hook, _bad_hook + ".disabled")

# ---------------------------------------------------------------------------
# Collect complex packages with data files
# ---------------------------------------------------------------------------
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
gradio_datas, gradio_binaries, gradio_hiddenimports = collect_all("gradio")
gradio_client_datas, _, gradio_client_hiddenimports = collect_all("gradio_client")
pyannote_datas, _, pyannote_hiddenimports = collect_all("pyannote")

# ---------------------------------------------------------------------------
# Locate bundled ffmpeg/ffprobe (Homebrew or build cache)
# ---------------------------------------------------------------------------
_ffmpeg = os.environ.get("FFMPEG_BIN", shutil.which("ffmpeg") or "ffmpeg")
_ffprobe = os.environ.get("FFPROBE_BIN", shutil.which("ffprobe") or "ffprobe")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ["transcribe/app.py"],
    pathex=[],
    datas=[
        ("transcribe/assets", "transcribe/assets"),
        *collect_data_files("safehttpx"),
        *collect_data_files("groovy"),
        *collect_data_files("gradio_client"),
        *torch_datas,
        *gradio_datas,
        *gradio_client_datas,
        *pyannote_datas,
    ],
    binaries=[
        (_ffmpeg, "."),
        (_ffprobe, "."),
        *torch_binaries,
        *gradio_binaries,
    ],
    hiddenimports=[
        "transcribe",
        "transcribe.core",
        "transcribe.web",
        "transcribe.tray",
        "transcribe.diarize",
        "transcribe.paths",
        "transcribe.cli",
        "webrtcvad",
        "pywhispercpp",
        "pywhispercpp.model",
        "PyObjCTools",
        "PyObjCTools.AppHelper",
        "AppKit",
        "Foundation",
        "objc",
        "webview",
        *torch_hiddenimports,
        *gradio_hiddenimports,
        *gradio_client_hiddenimports,
        *pyannote_hiddenimports,
    ],
    excludes=[
        "torch.cuda",
        "torch.distributed",
        "torch.testing",
    ],
    noarchive=False,
)

# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MediaTranscriber",
    debug=False,
    strip=False,       # don't strip — breaks PyObjC signatures
    upx=False,         # don't compress — breaks torch Metal libs
    console=False,     # no terminal window
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MediaTranscriber",
)

app = BUNDLE(
    coll,
    name="Media Transcriber.app",
    icon="transcribe/assets/app_icon.icns",
    bundle_identifier="com.mediatranscriber.app",
    info_plist={
        "CFBundleName": "Media Transcriber",
        "CFBundleDisplayName": "Media Transcriber",
        "CFBundleIdentifier": "com.mediatranscriber.app",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "MTSC",
        "CFBundleExecutable": "MediaTranscriber",
        "CFBundleIconFile": "app_icon.icns",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.productivity",
        "LSUIElement": False,
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Media File",
                "CFBundleTypeRole": "Viewer",
                "LSItemContentTypes": [
                    "public.movie",
                    "public.audio",
                    "public.mp3",
                    "public.mpeg-4-audio",
                    "com.microsoft.waveform-audio",
                    "org.xiph.flac",
                    "org.xiph.ogg-vorbis",
                ],
                "CFBundleTypeExtensions": [
                    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv",
                    "mp3", "wav", "ogg", "flac", "aac", "m4a", "wma", "opus",
                ],
            },
        ],
    },
)
