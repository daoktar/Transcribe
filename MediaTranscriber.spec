# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Media Transcriber macOS .app bundle."""

import os
import shutil
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

# Bundle version — kept in sync with scripts/build_macos.sh via the VERSION env var.
_VERSION = os.environ.get("VERSION", "1.6.0")

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
# MLX stack for the Qwen3-ASR fallback engine (Apple Silicon only). Optional:
# if MLX isn't installed at build time the app is packaged whisper-only.
# collect_all() cannot see mlx_audio's runtime importlib.import_module of the
# model classes, so the qwen3_asr / qwen3_forced_aligner modules are named
# explicitly below.
# ---------------------------------------------------------------------------
mlx_datas, mlx_binaries, mlx_hiddenimports = [], [], []
for _mlx_pkg in ("mlx", "mlx_metal", "mlx_audio", "mlx_lm"):
    try:
        _d, _b, _h = collect_all(_mlx_pkg)
        mlx_datas += _d
        mlx_binaries += _b
        mlx_hiddenimports += _h
    except Exception as _mlx_err:  # noqa: BLE001 — build-time best effort
        print(f"[spec] MLX package '{_mlx_pkg}' not collected ({_mlx_err})")
if mlx_hiddenimports:
    mlx_hiddenimports += [
        "transcribe.qwen_engine",
        "transcribe.qwen_prompt",
        "mlx_audio.stt.models.qwen3_asr",
        "mlx_audio.stt.models.qwen3_forced_aligner",
    ]

# transformers reads these versions via importlib.metadata at import time
# (e.g. audio_utils -> version("torchcodec")); PyInstaller copies no metadata by default.
for _meta_pkg in ("torchcodec", "torchaudio", "torch", "transformers", "tqdm", "tokenizers",
                  "huggingface_hub", "safetensors", "numpy", "packaging", "regex", "pyyaml"):
    try:
        mlx_datas += copy_metadata(_meta_pkg)
    except Exception as _meta_err:  # noqa: BLE001 — build-time best effort
        print(f"[spec] metadata for '{_meta_pkg}' not copied ({_meta_err})")

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
        ("transcribe/qwen_context_prompt.txt", "transcribe"),
        *torch_datas,
        *pyannote_datas,
        *mlx_datas,
    ],
    binaries=[
        (_ffmpeg, "."),
        (_ffprobe, "."),
        *torch_binaries,
        *mlx_binaries,
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
        *mlx_hiddenimports,
    ],
    excludes=[
        # ---------------------------------------------------------------
        # Torch: unused backends & subpackages (~50-80MB savings)
        # ---------------------------------------------------------------
        "torch.cuda",
        "torch.distributed",
        # NOT torch.testing: the transformers/torch stack Qwen3-ASR loads imports it at runtime
        "torch.utils.tensorboard",
        "torch.utils.benchmark",
        "torch.onnx",
        "torch.jit",
        "caffe2",
        # ---------------------------------------------------------------
        # pyannote.audio 4 needs pyannote.metrics/database, scipy, sklearn and pandas at
        # runtime (the community-1 pipeline fails to load without them), so only the
        # plotting/ONNX extras stay excluded.
        # ---------------------------------------------------------------
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
        # NOT sympy: torch imports it when transformers loads (Qwen3-ASR path)
        "IPython",
        "notebook",
        "jupyterlab",
        # ---------------------------------------------------------------
        # Testing / dev tools (should not be in production)
        # ---------------------------------------------------------------
        "pytest",
        "_pytest",
        "doctest",
        # NOT setuptools / distutils: scipy pulls distutils in, and excluding it clashes
        # with PyInstaller's distutils alias hook ("already imported as ExcludedModule")
        "pip",
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
        "pydoc_data",
        # NOT unittest / pydoc / numpy.testing / numpy.f2py: torch.testing (imported by the Qwen3-ASR
        # transformers path) needs them at runtime
        # ---------------------------------------------------------------
        # Unused numpy subpackages
        # ---------------------------------------------------------------
        "numpy.tests",
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
        "CFBundleVersion": _VERSION,
        "CFBundleShortVersionString": _VERSION,
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
