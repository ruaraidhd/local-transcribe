"""Transcribe and diarise audio locally with parakeet-mlx + pyannote.

Usable as a CLI or imported (see ``load_pipeline`` and ``transcribe_one``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path as _Path

# Resolve the app's resource directory — differs between dev and PyInstaller bundle.
_BUNDLE_DIR = _Path(getattr(sys, "_MEIPASS", _Path(__file__).parent))

# Use bundled models if available (eliminates HF token requirement).
_models_dir = _BUNDLE_DIR / "models"
if _models_dir.exists():
    os.environ["HF_HOME"] = str(_models_dir)
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from logging_setup import (
    hint_for_pyannote_error,
    log_batch_start,
    log_startup_diagnostics,
    setup_logging,
)

SETTINGS_PATH = _BUNDLE_DIR / "settings.toml"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".aac"}

log = logging.getLogger("local_transcribe")


from backends import (
    TranscriptionBackend,
    DiarisationBackend,
    get_transcription_backend,
    get_diarisation_backend,
)


@dataclass
class Pipeline:
    transcriber: TranscriptionBackend
    diariser: DiarisationBackend


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        s = tomllib.loads(SETTINGS_PATH.read_text())
    elif _models_dir.exists():
        # Bundled mode: models are shipped, no token needed.
        s = {}
    else:
        sys.exit(
            f"Missing {SETTINGS_PATH.name}. "
            "Copy settings.example.toml to settings.toml and fill in your HF token."
        )
    # HF token only required when models aren't bundled.
    token = s.get("hf_token", "").strip()
    if not _models_dir.exists() and (not token or token.startswith("hf_YOUR")):
        sys.exit(f"hf_token not set in {SETTINGS_PATH.name}")
    return s


def save_settings(settings: dict) -> None:
    """Persist hf_token, outbox, model, and device_mps to settings.toml."""
    lines = [f'hf_token = "{settings["hf_token"]}"']
    if "outbox" in settings:
        lines.append(f'outbox = "{settings["outbox"]}"')
    if "model" in settings:
        lines.append(f'model = "{settings["model"]}"')
    if "device_mps" in settings:
        val = "true" if settings["device_mps"] else "false"
        lines.append(f"device_mps = {val}")
    SETTINGS_PATH.write_text("\n".join(lines) + "\n")


def accel_device(settings: dict) -> str:
    """Return 'mps' if device_mps is enabled and available, else 'cpu'."""
    if settings.get("device_mps") is not True:
        return "cpu"
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    raise FileNotFoundError(f"Not a file or directory: {path}")


def format_ts(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(
            f"{i}\n{format_ts(seg['start'])} --> {format_ts(seg['end'])}\n"
            f"[{speaker}] {text}\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: list[dict], path: Path) -> None:
    lines, current, buf = [], None, []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if speaker != current:
            if buf:
                lines.append(f"{current}: {' '.join(buf)}")
                buf = []
            current = speaker
        buf.append(text)
    if buf:
        lines.append(f"{current}: {' '.join(buf)}")
    path.write_text("\n\n".join(lines), encoding="utf-8")


def load_pipeline(
    hf_token: str,
    model_id: str = "mlx-community/parakeet-tdt-0.6b-v2",
    settings: dict | None = None,
) -> Pipeline:
    _settings = settings or {}
    transcriber = get_transcription_backend(_settings.get("transcription_backend", "parakeet-mlx"))
    transcriber.load(model_id)
    dev = accel_device(_settings)
    diariser = get_diarisation_backend(
        _settings.get("diarisation_backend", "pyannote"),
        device=dev, hf_token=hf_token,
    )
    return Pipeline(transcriber=transcriber, diariser=diariser)


def _assign_speakers(tokens, speaker_turns) -> list[dict]:
    """Group tokens into speaker-labelled segments.

    tokens: list of backends.Token (or anything with .text, .start, .duration)
    speaker_turns: list of backends.SpeakerTurn (or anything with .start, .end, .speaker)
    """
    import math

    turns = [(t.start, t.end, t.speaker) for t in speaker_turns]

    def find_speaker(mid: float) -> str | None:
        for t_start, t_end, spk in turns:
            if t_start <= mid <= t_end:
                return spk
        return None

    def _segment_confidence(seg_tokens) -> float:
        confidences = [t.confidence for t in seg_tokens if hasattr(t, "confidence")]
        if confidences:
            return math.exp(
                sum(math.log(c + 1e-10) for c in confidences) / len(confidences)
            )
        return 1.0

    segments: list[dict] = []
    current_speaker: str | None = None
    current_text: list[str] = []
    current_tokens: list = []
    current_start: float = 0.0
    current_end: float = 0.0
    # Seed prev_speaker with the first speaker that appears, so leading
    # tokens before pyannote's first turn inherit forward rather than
    # falling to UNKNOWN.
    first_speaker = turns[0][2] if turns else "UNKNOWN"
    prev_speaker = first_speaker

    for token in tokens:
        mid = token.start + token.duration / 2
        speaker = find_speaker(mid) or prev_speaker
        prev_speaker = speaker
        if speaker != current_speaker:
            if current_text:
                segments.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "text": "".join(current_text).strip(),
                    "confidence": _segment_confidence(current_tokens),
                    "edited": False,
                    "reviewed": False,
                })
            current_speaker = speaker
            current_start = token.start
            current_text = []
            current_tokens = []
        current_text.append(token.text)
        current_tokens.append(token)
        current_end = token.end

    if current_text:
        segments.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "text": "".join(current_text).strip(),
            "confidence": _segment_confidence(current_tokens),
            "edited": False,
            "reviewed": False,
        })

    # Smooth: merge very short segments (< 1s) into their neighbour.
    # First pass: merge short segments into the previous one.
    # When merging, take the lower confidence of the two segments.
    smoothed: list[dict] = []
    for seg in segments:
        if (
            smoothed
            and (seg["end"] - seg["start"]) < 1.0
            and smoothed[-1]["speaker"] != seg["speaker"]
        ):
            smoothed[-1]["end"] = seg["end"]
            smoothed[-1]["text"] += " " + seg["text"]
            smoothed[-1]["confidence"] = min(smoothed[-1]["confidence"], seg["confidence"])
        elif smoothed and smoothed[-1]["speaker"] == seg["speaker"]:
            smoothed[-1]["end"] = seg["end"]
            smoothed[-1]["text"] += " " + seg["text"]
            smoothed[-1]["confidence"] = min(smoothed[-1]["confidence"], seg["confidence"])
        else:
            smoothed.append(seg)

    # Second pass: if the first segment is very short, merge it forward.
    if (
        len(smoothed) >= 2
        and (smoothed[0]["end"] - smoothed[0]["start"]) < 1.0
    ):
        smoothed[1]["start"] = smoothed[0]["start"]
        smoothed[1]["text"] = smoothed[0]["text"] + " " + smoothed[1]["text"]
        smoothed[1]["confidence"] = min(smoothed[0]["confidence"], smoothed[1]["confidence"])
        smoothed.pop(0)

    return smoothed


def transcribe_one(
    audio_path: Path,
    out_dir: Path,
    pipeline: Pipeline,
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    settings: dict | None = None,
    on_stage_start: Callable[[str], None] | None = None,
    on_stage_end: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    if language is not None and language not in (None, "en"):
        log.warning(
            "parakeet-mlx is English-only; ignoring language=%r", language
        )

    log.info("=== %s ===", audio_path.name)

    # Preflight: warn if disk space is low (swap needs headroom).
    import shutil
    free_gb = shutil.disk_usage(Path.home()).free / (1024 ** 3)
    if free_gb < 10:
        log.warning(
            "LOW DISK: only %.1f GB free. macOS needs free space for swap "
            "when processing large files. Risk of system freeze if memory "
            "pressure is high. Free up disk space before continuing.", free_gb
        )
    if free_gb < 5:
        raise RuntimeError(
            f"Aborting: only {free_gb:.1f} GB free disk. Processing large "
            f"audio files requires at least 10 GB free for swap headroom. "
            f"Free up disk space and try again."
        )

    t_total = time.perf_counter()

    # --- Stage 1: Transcribe ---
    if on_stage_start:
        on_stage_start("Transcribing")
    log.info("Transcribing...")
    t0 = time.perf_counter()

    result = pipeline.transcriber.transcribe(audio_path, language, on_progress)

    log.info(
        "Transcribe: %.1fs (%d tokens)",
        time.perf_counter() - t0,
        len(result.tokens),
    )
    if on_stage_end:
        on_stage_end("Transcribing")

    # Free the transcription model before diarisation to save memory.
    pipeline.transcriber.unload()

    # --- Stage 2: Diarise ---
    if on_stage_start:
        on_stage_start("Diarising")
    log.info("Diarising (speakers=%s-%s)...", min_speakers, max_speakers)
    t0 = time.perf_counter()

    try:
        speaker_turns = pipeline.diariser.diarise(
            audio_path, min_speakers, max_speakers, on_progress,
        )
    except Exception as e:
        hint = hint_for_pyannote_error(e)
        if hint:
            log.error("%s", hint)
        raise

    log.info("Diarise: %.1fs", time.perf_counter() - t0)
    if on_stage_end:
        on_stage_end("Diarising")

    # --- Stage 3: Assign speakers ---
    if on_stage_start:
        on_stage_start("Assigning speakers")
    log.info("Assigning speakers to tokens...")
    t0 = time.perf_counter()

    segments = _assign_speakers(result.tokens, speaker_turns)

    log.info("Speaker assignment: %.1fs (%d segments)", time.perf_counter() - t0, len(segments))
    if on_stage_end:
        on_stage_end("Assigning speakers")

    # --- Write outputs ---
    stem = audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "text": result.text,
        "segments": segments,
        "tokens": [
            {
                "text": t.text,
                "start": t.start,
                "end": t.end,
                "confidence": t.confidence,
            }
            for t in result.tokens
        ],
    }

    (out_dir / f"{stem}.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_srt(segments, out_dir / f"{stem}.srt")
    write_txt(segments, out_dir / f"{stem}.txt")
    log.info(
        "Wrote %s.{json,srt,txt}  (total %.1fs)",
        stem,
        time.perf_counter() - t_total,
    )
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Audio file or directory of audio files")
    ap.add_argument("-o", "--output", type=Path, default=Path("output"))
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    ap.add_argument("--language", default=None,
                    help="Ignored (parakeet-mlx is English-only); retained for compatibility")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    settings = load_settings()
    log_startup_diagnostics(settings, args.input, args.output)

    try:
        inputs = collect_inputs(args.input)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if not inputs:
        sys.exit("No audio files found")

    log_batch_start(inputs)
    pipeline = load_pipeline(
        settings["hf_token"],
        args.model,
        settings,
    )
    failures = 0
    for audio_path in inputs:
        try:
            transcribe_one(
                audio_path, args.output, pipeline,
                args.language, args.min_speakers, args.max_speakers,
                settings=settings,
            )
        except Exception:
            failures += 1
            log.exception("FAILED %s", audio_path.name)
    log.info("Batch complete: %d ok, %d failed", len(inputs) - failures, failures)


if __name__ == "__main__":
    main()
