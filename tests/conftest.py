"""Shared pytest fixtures for the Verbatim test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable (needed when running pytest from any dir).
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
