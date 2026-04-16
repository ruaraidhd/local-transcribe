"""Backend abstraction for transcription and diarisation.

Each backend implements a simple interface. transcribe.py picks the right
backend based on settings/platform and calls it through that interface.
Adding a new backend = one new class, no changes to orchestration or GUI.

Transcription backends produce tokens (text + timestamps).
Diarisation backends produce speaker turns (start, end, speaker label).
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger("local_transcribe")


# ---- Data types (backend-agnostic) ----

@dataclass
class Token:
    text: str
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class TranscriptionResult:
    text: str
    tokens: list[Token]


# ---- Transcription backend interface ----

class TranscriptionBackend(ABC):
    """Transcribes audio to text with word-level timestamps."""

    @abstractmethod
    def load(self, model_id: str, **kwargs) -> None:
        """Load the model. Called once (or after unload)."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file. Returns text + tokens with timestamps."""

    @abstractmethod
    def unload(self) -> None:
        """Free model memory. Called between stages to save RAM."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently in memory."""

    @property
    def supported_languages(self) -> list[str] | None:
        """List of supported language codes, or None for all languages."""
        return None


# ---- Diarisation backend interface ----

class DiarisationBackend(ABC):
    """Identifies who spoke when."""

    @abstractmethod
    def diarise(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[SpeakerTurn]:
        """Return speaker turns for the audio."""


# ---- Parakeet MLX backend (macOS, Apple Silicon) ----

class ParakeetMLXBackend(TranscriptionBackend):
    """Parakeet TDT via MLX — fast on Apple Silicon GPU."""

    def __init__(self):
        self._model = None

    def load(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v2", **kwargs) -> None:
        import parakeet_mlx
        log.info("Loading parakeet model %s ...", model_id)
        t0 = time.perf_counter()
        self._model = parakeet_mlx.from_pretrained(model_id)
        log.info("Model loaded (%.1fs)", time.perf_counter() - t0)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        if language is not None and language != "en":
            log.warning("parakeet-mlx is English-only; ignoring language=%r", language)

        def _chunk_progress(current_pos, total_pos):
            if on_progress and total_pos > 0:
                on_progress((current_pos / total_pos) * 100)

        result = self._model.transcribe(
            str(audio_path),
            chunk_duration=600.0,
            overlap_duration=15.0,
            chunk_callback=_chunk_progress,
        )
        tokens = [
            Token(text=t.text, start=t.start, end=t.end, confidence=t.confidence)
            for t in result.tokens
        ]
        return TranscriptionResult(text=result.text, tokens=tokens)

    def unload(self) -> None:
        import gc
        self._model = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        log.debug("Freed parakeet model")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supported_languages(self) -> list[str]:
        return ["en"]


# ---- Pyannote diarisation backend ----

class PyannoteDiarisationBackend(DiarisationBackend):
    """Pyannote speaker diarisation — MPS on Mac, CUDA on Linux/Windows, CPU fallback."""

    def __init__(self, device: str = "cpu", hf_token: str | None = None):
        self.device = device
        self.hf_token = hf_token

    def diarise(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[SpeakerTurn]:
        import numpy as np
        import subprocess as sp
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline

        # Decode audio via ffmpeg (avoids broken torchcodec loader).
        SAMPLE_RATE = 16000
        cmd = ["ffmpeg", "-i", str(audio_path), "-f", "f32le", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-loglevel", "error", "-"]
        pcm = sp.run(cmd, capture_output=True, check=True).stdout
        waveform = torch.from_numpy(
            np.frombuffer(pcm, dtype=np.float32).copy()
        ).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

        def _run(device: str) -> list[SpeakerTurn]:
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=self.hf_token,
            )
            if device == "mps":
                pipeline = pipeline.to(torch.device("mps"))
            elif device == "cuda":
                pipeline = pipeline.to(torch.device("cuda"))

            kwargs = {}
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

            result = pipeline(audio_input, **kwargs)
            annotation = getattr(result, "speaker_diarization", result)
            return [
                SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]

        if self.device in ("mps", "cuda"):
            try:
                return _run(self.device)
            except Exception as e:
                log.warning("Diarise on %s failed (%s), retrying on CPU", self.device, e)
                return _run("cpu")
        return _run("cpu")


# ---- Registry ----

TRANSCRIPTION_BACKENDS = {
    "parakeet-mlx": ParakeetMLXBackend,
    # Future: "faster-whisper", "whisper-cpp", "nemo-cuda", ...
}

DIARISATION_BACKENDS = {
    "pyannote": PyannoteDiarisationBackend,
    # Future: "nemo-sortformer", ...
}


def get_transcription_backend(name: str = "parakeet-mlx") -> TranscriptionBackend:
    cls = TRANSCRIPTION_BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown transcription backend '{name}'. "
            f"Available: {list(TRANSCRIPTION_BACKENDS)}"
        )
    return cls()


def get_diarisation_backend(
    name: str = "pyannote", device: str = "cpu", hf_token: str | None = None,
) -> DiarisationBackend:
    cls = DIARISATION_BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown diarisation backend '{name}'. "
            f"Available: {list(DIARISATION_BACKENDS)}"
        )
    if name == "pyannote":
        return cls(device=device, hf_token=hf_token)
    return cls()
