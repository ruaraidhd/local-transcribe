"""Contract tests for the TranscriptionBackend interface.

A MockTranscriptionBackend is defined and exercised against the interface
contract. Future backends can be tested the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from backends import Token, TranscriptionBackend, TranscriptionResult


# ---------------------------------------------------------------------------
# Minimal mock backend
# ---------------------------------------------------------------------------

class MockTranscriptionBackend(TranscriptionBackend):
    """A no-op backend that returns canned results for testing."""

    def __init__(self):
        self._loaded = False
        self._model_id: str | None = None

    def load(self, model_id: str, **kwargs) -> None:
        self._model_id = model_id
        self._loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        tokens = [
            Token(text="Hello", start=0.0, end=0.5, confidence=0.9),
            Token(text=" world", start=0.5, end=1.0, confidence=0.95),
        ]
        return TranscriptionResult(text="Hello world", tokens=tokens)

    def unload(self) -> None:
        self._loaded = False
        self._model_id = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_initially_not_loaded():
    backend = MockTranscriptionBackend()
    assert backend.is_loaded is False


def test_after_load_is_loaded():
    backend = MockTranscriptionBackend()
    backend.load("mock-model-v1")
    assert backend.is_loaded is True


def test_after_unload_not_loaded():
    backend = MockTranscriptionBackend()
    backend.load("mock-model-v1")
    backend.unload()
    assert backend.is_loaded is False


def test_transcribe_returns_transcription_result():
    backend = MockTranscriptionBackend()
    backend.load("mock-model-v1")
    result = backend.transcribe(Path("/fake/audio.wav"))
    assert isinstance(result, TranscriptionResult)


def test_transcription_result_has_text():
    backend = MockTranscriptionBackend()
    backend.load("mock-model-v1")
    result = backend.transcribe(Path("/fake/audio.wav"))
    assert isinstance(result.text, str)
    assert len(result.text) > 0


def test_transcription_result_has_token_list():
    backend = MockTranscriptionBackend()
    backend.load("mock-model-v1")
    result = backend.transcribe(Path("/fake/audio.wav"))
    assert isinstance(result.tokens, list)
    assert len(result.tokens) > 0


def test_tokens_are_token_instances():
    backend = MockTranscriptionBackend()
    backend.load("mock-model-v1")
    result = backend.transcribe(Path("/fake/audio.wav"))
    for tok in result.tokens:
        assert isinstance(tok, Token)
        assert isinstance(tok.text, str)
        assert isinstance(tok.start, float)
        assert isinstance(tok.end, float)
        assert 0.0 <= tok.confidence <= 1.0
        assert tok.end > tok.start
        assert tok.duration == pytest.approx(tok.end - tok.start)


def test_transcribe_before_load_raises():
    backend = MockTranscriptionBackend()
    with pytest.raises(RuntimeError):
        backend.transcribe(Path("/fake/audio.wav"))


def test_reload_after_unload_works():
    backend = MockTranscriptionBackend()
    backend.load("mock-model-v1")
    backend.unload()
    assert not backend.is_loaded
    backend.load("mock-model-v1")
    assert backend.is_loaded
    result = backend.transcribe(Path("/fake/audio.wav"))
    assert result.text == "Hello world"


def test_default_supported_languages_is_none():
    """The base interface returns None for supported_languages (all languages)."""
    backend = MockTranscriptionBackend()
    # MockTranscriptionBackend doesn't override supported_languages,
    # so it should return None per the ABC default.
    assert backend.supported_languages is None
