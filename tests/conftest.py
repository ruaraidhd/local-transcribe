"""Shared pytest fixtures for the Verbatim test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from hypothesis import HealthCheck, settings

# Ensure the project root is importable (needed when running pytest from any dir).
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A deterministic profile for mutation testing.
# Enabled by setting MUTMUT=1 in the environment.
settings.register_profile(
    "mutmut",
    derandomize=True,
    max_examples=25,  # fewer examples for speed; with derandomize, they're always the same
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

if os.environ.get("MUTMUT"):
    settings.load_profile("mutmut")
