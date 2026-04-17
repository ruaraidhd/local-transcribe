#!/usr/bin/env bash
# Build Verbatim.app — standalone macOS application.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Building Verbatim ==="

# Unmount any stale Verbatim volume (blocks rm -rf dist/)
for vol in /Volumes/Verbatim*; do
    [ -d "$vol" ] && hdiutil detach "$vol" -force 2>/dev/null || true
done

# Kill any running Verbatim instance holding files open
pkill -f "Verbatim.app" 2>/dev/null || true
sleep 0.5

# Clean
rm -rf dist/ build/

# PyInstaller
echo "Running PyInstaller..."
uv run pyinstaller verbatim.spec --noconfirm

# Copy models (too large for PyInstaller bundling)
echo "Copying models..."
cp -r models/ "dist/Verbatim.app/Contents/Frameworks/models"

# Copy MLX metallib to where the .so expects it
cp "dist/Verbatim.app/Contents/Frameworks/mlx/lib/mlx.metallib" \
   "dist/Verbatim.app/Contents/Frameworks/mlx.metallib" 2>/dev/null || true

# Ad-hoc code signing
echo "Signing..."
codesign --force --deep --sign - "dist/Verbatim.app" 2>&1 || true

echo "=== Build complete ==="
echo "App: dist/Verbatim.app ($(du -sh dist/Verbatim.app | cut -f1))"

# Create DMG
echo "Creating DMG..."
DMG_NAME="Verbatim-0.3.0.dmg"
rm -f "dist/$DMG_NAME"

# Create a temporary directory with the .app and a symlink to Applications
mkdir -p dist/dmg_staging
cp -r "dist/Verbatim.app" dist/dmg_staging/
ln -sf /Applications dist/dmg_staging/Applications

hdiutil create -volname "Verbatim" \
    -srcfolder dist/dmg_staging \
    -ov -format UDZO \
    "dist/$DMG_NAME"

rm -rf dist/dmg_staging
echo "DMG: dist/$DMG_NAME ($(du -sh dist/$DMG_NAME | cut -f1))"
