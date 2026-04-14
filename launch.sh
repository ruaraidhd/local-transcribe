#!/usr/bin/env bash
# Launcher for Local Transcribe.app (wrapped by Platypus).
#
# Platypus copies this script into the .app bundle, so we can't locate the
# project relative to the script. Instead we search a list of known paths for
# a directory containing gui.py + pyproject.toml, and cd there.

set -euo pipefail

CANDIDATES=(
    "$HOME/Applications/local-transcribe"
    "$HOME/Applications/local_transcribe"
    "$HOME/Dropbox/Personal/local_transcribe"
)

for dir in "${CANDIDATES[@]}"; do
    if [[ -f "$dir/pyproject.toml" && -f "$dir/gui.py" ]]; then
        cd "$dir"
        export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
        exec uv run python gui.py
    fi
done

osascript <<'OSA'
display alert "Local Transcribe" ¬
    message "Couldn't find the project folder. Expected one of:
  ~/Applications/local-transcribe
  ~/Applications/local_transcribe
  ~/Dropbox/Personal/local_transcribe

See SETUP.md." ¬
    as critical
OSA
exit 1
