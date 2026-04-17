"""Integration smoke test: run the whole pipeline on a synthetic 1-second WAV.

Mark as slow so it's skipped in normal runs:
  uv run pytest -m slow          — run integration tests only
  uv run pytest -m "not slow"    — skip integration tests
"""
from __future__ import annotations

import numpy as np
import os
import wave
from pathlib import Path

import pytest

pytest.importorskip("parakeet_mlx")
pytest.importorskip("pyannote.audio")


def _make_synthetic_wav(path, duration_sec=1.0, sr=16000):
    """Generate a sine wave so it looks like audio."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr))
    data = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.3).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


@pytest.mark.slow
def test_pipeline_smoke(tmp_path):
    """End-to-end pipeline runs on synthetic audio without crashing.

    Doesn't check transcript content (likely empty/garbage for sine wave)
    — just that the wiring works: transcribe → diarise → assign → write.
    """
    from transcribe import load_pipeline, transcribe_one

    audio = tmp_path / "smoke.wav"
    _make_synthetic_wav(audio)

    outbox = tmp_path / "out"
    outbox.mkdir()

    # Bundled models needed; skip if not available
    models = Path(__file__).parent.parent / "models"
    if not models.exists():
        pytest.skip("models/ not populated — run download_models.py")
    os.environ["HF_HOME"] = str(models)

    pipeline = load_pipeline(hf_token="", settings={})
    try:
        result = transcribe_one(audio, outbox, pipeline, settings={})
        # Pipeline wrote outputs
        assert (outbox / "smoke.json").exists()
        assert (outbox / "smoke.txt").exists()
        assert (outbox / "smoke.srt").exists()
    finally:
        if pipeline.transcriber.is_loaded:
            pipeline.transcriber.unload()
