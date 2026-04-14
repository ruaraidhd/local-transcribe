#!/usr/bin/env bash
# Launcher wrapped by Platypus into Local Transcribe.app
# Resolves the project directory from the script's own location so the .app
# can live anywhere and still find the venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

exec uv run python gui.py
