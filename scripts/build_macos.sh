#!/usr/bin/env bash
# Build Media Transcriber as a macOS .app bundle and package into a .dmg.
#
# Prerequisites:
#   pip install pyinstaller
#   brew install create-dmg   (optional, for .dmg creation)
#
# Usage:
#   bash scripts/build_macos.sh
#
# Environment variables:
#   FFMPEG_BIN   — path to ffmpeg binary to bundle (default: $(which ffmpeg))
#   FFPROBE_BIN  — path to ffprobe binary to bundle (default: $(which ffprobe))
#   SKIP_DMG=1   — skip .dmg creation step

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="1.1.0"
APP_NAME="Media Transcriber"
DMG_NAME="MediaTranscriber-${VERSION}-arm64"

cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------
# Step 1: Generate .icns icon (if not already present)
# -----------------------------------------------------------------------
ICNS="transcribe/assets/app_icon.icns"
if [ ! -f "$ICNS" ]; then
    echo "=== Generating .icns icon ==="
    ICONSET="build/app_icon.iconset"
    mkdir -p "$ICONSET"
    sips -z 16 16     transcribe/assets/app_icon.png --out "$ICONSET/icon_16x16.png"
    sips -z 32 32     transcribe/assets/app_icon.png --out "$ICONSET/icon_16x16@2x.png"
    sips -z 32 32     transcribe/assets/app_icon.png --out "$ICONSET/icon_32x32.png"
    sips -z 64 64     transcribe/assets/app_icon.png --out "$ICONSET/icon_32x32@2x.png"
    sips -z 128 128   transcribe/assets/app_icon.png --out "$ICONSET/icon_128x128.png"
    sips -z 256 256   transcribe/assets/app_icon.png --out "$ICONSET/icon_128x128@2x.png"
    sips -z 256 256   transcribe/assets/app_icon.png --out "$ICONSET/icon_256x256.png"
    sips -z 512 512   transcribe/assets/app_icon.png --out "$ICONSET/icon_256x256@2x.png"
    sips -z 512 512   transcribe/assets/app_icon.png --out "$ICONSET/icon_512x512.png"
    sips -z 1024 1024 transcribe/assets/app_icon.png --out "$ICONSET/icon_512x512@2x.png"
    iconutil -c icns "$ICONSET" -o "$ICNS"
    rm -rf "$ICONSET"
    echo "Icon generated: $ICNS"
else
    echo "=== .icns icon already exists, skipping ==="
fi

# -----------------------------------------------------------------------
# Step 2: Verify ffmpeg is available
# -----------------------------------------------------------------------
echo "=== Checking ffmpeg ==="
export FFMPEG_BIN="${FFMPEG_BIN:-$(which ffmpeg || true)}"
export FFPROBE_BIN="${FFPROBE_BIN:-$(which ffprobe || true)}"

if [ -z "$FFMPEG_BIN" ] || [ ! -f "$FFMPEG_BIN" ]; then
    echo "ERROR: ffmpeg not found. Install with: brew install ffmpeg"
    echo "Or set FFMPEG_BIN=/path/to/ffmpeg"
    exit 1
fi
if [ -z "$FFPROBE_BIN" ] || [ ! -f "$FFPROBE_BIN" ]; then
    echo "ERROR: ffprobe not found. Install with: brew install ffmpeg"
    echo "Or set FFPROBE_BIN=/path/to/ffprobe"
    exit 1
fi
echo "ffmpeg:  $FFMPEG_BIN"
echo "ffprobe: $FFPROBE_BIN"

# -----------------------------------------------------------------------
# Step 3: Run PyInstaller
# -----------------------------------------------------------------------
echo "=== Building .app with PyInstaller ==="
if ! command -v pyinstaller &>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

pyinstaller MediaTranscriber.spec --noconfirm --clean

echo "=== Build complete ==="
ls -lh "dist/$APP_NAME.app/Contents/MacOS/MediaTranscriber"

# -----------------------------------------------------------------------
# Step 4: Post-build cleanup — remove unnecessary files from bundle
# -----------------------------------------------------------------------
echo "=== Post-build cleanup ==="
APP_CONTENTS="dist/$APP_NAME.app/Contents"
BEFORE_SIZE=$(du -sm "dist/$APP_NAME.app" | cut -f1)

# Remove test directories
find "$APP_CONTENTS" -type d \( -name "tests" -o -name "test" -o -name "__pycache__" \) -exec rm -rf {} + 2>/dev/null || true

# Remove package metadata
find "$APP_CONTENTS" -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true

# Remove type stubs and documentation
find "$APP_CONTENTS" \( -name "*.pyi" -o -name "*.pyx" \) -delete 2>/dev/null || true

# Remove leftover torch test data and benchmarks
find "$APP_CONTENTS" -path "*/torch/testing/*" -exec rm -rf {} + 2>/dev/null || true
find "$APP_CONTENTS" -path "*/torch/utils/benchmark/*" -exec rm -rf {} + 2>/dev/null || true
find "$APP_CONTENTS" -path "*/caffe2/*" -exec rm -rf {} + 2>/dev/null || true

# Remove any scipy/sklearn/pandas that survived exclude (transitive imports)
for pkg in scipy sklearn pandas matplotlib PIL grpc onnxruntime hf_xet; do
    find "$APP_CONTENTS" -type d -name "$pkg" -exec rm -rf {} + 2>/dev/null || true
done

AFTER_SIZE=$(du -sm "dist/$APP_NAME.app" | cut -f1)
echo "Cleanup saved $((BEFORE_SIZE - AFTER_SIZE))MB (${BEFORE_SIZE}MB -> ${AFTER_SIZE}MB)"
echo "App bundle: dist/$APP_NAME.app"

# -----------------------------------------------------------------------
# Step 5: Create .dmg (optional)
# -----------------------------------------------------------------------
if [ "${SKIP_DMG:-}" = "1" ]; then
    echo "=== Skipping .dmg creation (SKIP_DMG=1) ==="
else
    if command -v create-dmg &>/dev/null; then
        echo "=== Creating .dmg ==="
        rm -f "dist/${DMG_NAME}.dmg"
        create-dmg \
            --volname "$APP_NAME" \
            --volicon "$ICNS" \
            --window-pos 200 120 \
            --window-size 660 400 \
            --icon-size 100 \
            --icon "$APP_NAME.app" 180 190 \
            --hide-extension "$APP_NAME.app" \
            --app-drop-link 480 190 \
            "dist/${DMG_NAME}.dmg" \
            "dist/$APP_NAME.app"

        echo "=== Done ==="
        ls -lh "dist/${DMG_NAME}.dmg"
    else
        echo "=== Skipping .dmg (install create-dmg: brew install create-dmg) ==="
        echo "You can still run the app directly: open \"dist/$APP_NAME.app\""
    fi
fi
