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
pyannote_datas, _, pyannote_hiddenimports = collect_all("pyannote")

# ---------------------------------------------------------------------------
# Locate bundled ffmpeg/ffprobe (Homebrew or env override)
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
        ("transcribe/static", "transcribe/static"),
        *torch_datas,
        *pyannote_datas,
    ],
    binaries=[
        (_ffmpeg, "."),
        (_ffprobe, "."),
        *torch_binaries,
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
        "fastapi",
        "uvicorn",
        "starlette",
        "starlette.staticfiles",
        "starlette.responses",
        "multipart",
        "PyObjCTools",
        "PyObjCTools.AppHelper",
        "AppKit",
        "Foundation",
        "objc",
        "webview",
        *torch_hiddenimports,
        *pyannote_hiddenimports,
    ],
    excludes=[
        # ---------------------------------------------------------------
        # Torch: unused backends & subpackages (~50-80MB savings)
        # ---------------------------------------------------------------
        "torch.cuda",
        "torch.distributed",
        "torch.testing",
        "torch.utils.tensorboard",
        "torch.utils.benchmark",
        "torch.onnx",
        "torch.jit",
        "caffe2",
        # ---------------------------------------------------------------
        # Pyannote transitive deps not needed at runtime (~163MB savings)
        # Only pyannote.audio is used; pyannote.metrics/database pull in
        # scipy, sklearn, pandas, matplotlib — none used by our code.
        # ---------------------------------------------------------------
        "pyannote.metrics",
        "pyannote.database",
        "scipy",
        "sklearn",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "PIL",
        "Pillow",
        "grpc",
        "grpcio",
        "onnxruntime",
        "hf_xet",
        # ---------------------------------------------------------------
        # Unused third-party
        # ---------------------------------------------------------------
        "gradio",
        "gradio_client",
        "sympy",
        "IPython",
        "notebook",
        "jupyterlab",
        # ---------------------------------------------------------------
        # Testing / dev tools (should not be in production)
        # ---------------------------------------------------------------
        "pytest",
        "_pytest",
        "unittest",
        "doctest",
        "setuptools",
        "pip",
        "distutils",
        "ensurepip",
        # ---------------------------------------------------------------
        # Unused stdlib
        # ---------------------------------------------------------------
        "tkinter",
        "_tkinter",
        "turtle",
        "idlelib",
        "xmlrpc",
        "ftplib",
        "imaplib",
        "poplib",
        "smtplib",
        "nntplib",
        "pydoc",
        "pydoc_data",
        # ---------------------------------------------------------------
        # Unused numpy subpackages
        # ---------------------------------------------------------------
        "numpy.testing",
        "numpy.tests",
        "numpy.f2py",
        "numpy.distutils",
    ],
    noarchive=False,
)

# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, optimize=2)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MediaTranscriber",
    debug=False,
    strip=False,       # don't strip — breaks PyObjC signatures
    upx=False,         # PyInstaller disables UPX on macOS anyway
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
        "CFBundleVersion": "1.1.0",
        "CFBundleShortVersionString": "1.1.0",
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
