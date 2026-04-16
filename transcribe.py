"""Transcribe and diarise audio locally with parakeet-mlx + pyannote.

Usable as a CLI or imported (see ``load_pipeline`` and ``transcribe_one``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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

SETTINGS_PATH = Path(__file__).parent / "settings.toml"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".aac"}

log = logging.getLogger("local_transcribe")


@dataclass
class Pipeline:
    model: object          # parakeet-mlx model
    hf_token: str
    diarize_device: str    # "mps" or "cpu" for pyannote


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        sys.exit(
            f"Missing {SETTINGS_PATH.name}. "
            "Copy settings.example.toml to settings.toml and fill in your HF token."
        )
    s = tomllib.loads(SETTINGS_PATH.read_text())
    token = s.get("hf_token", "").strip()
    if not token or token.startswith("hf_YOUR"):
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
    import parakeet_mlx
    log.info("Loading parakeet model %s ...", model_id)
    t0 = time.perf_counter()
    model = parakeet_mlx.from_pretrained(model_id)
    log.info("Model loaded (%.1fs)", time.perf_counter() - t0)
    dev = accel_device(settings or {})
    return Pipeline(model=model, hf_token=hf_token, diarize_device=dev)


def _assign_speakers(tokens, diarization) -> list[dict]:
    """Group parakeet tokens into speaker-labelled segments."""
    speaker_turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    def find_speaker(mid: float) -> str:
        for t_start, t_end, spk in speaker_turns:
            if t_start <= mid <= t_end:
                return spk
        return "UNKNOWN"

    segments: list[dict] = []
    current_speaker: str | None = None
    current_text: list[str] = []
    current_start: float = 0.0
    current_end: float = 0.0

    for token in tokens:
        mid = token.start + token.duration / 2
        speaker = find_speaker(mid)
        if speaker != current_speaker:
            if current_text:
                segments.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "text": "".join(current_text).strip(),
                })
            current_speaker = speaker
            current_start = token.start
            current_text = []
        current_text.append(token.text)
        current_end = token.end

    if current_text:
        segments.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "text": "".join(current_text).strip(),
        })

    return segments


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
    t_total = time.perf_counter()

    # --- Stage 1: Transcribe ---
    if on_stage_start:
        on_stage_start("Transcribing")
    log.info("Transcribing...")
    t0 = time.perf_counter()

    result = pipeline.model.transcribe(str(audio_path))

    log.info(
        "Transcribe: %.1fs (%d tokens)",
        time.perf_counter() - t0,
        len(result.tokens),
    )
    if on_stage_end:
        on_stage_end("Transcribing")

    # --- Stage 2: Diarise ---
    if on_stage_start:
        on_stage_start("Diarising")
    log.info("Diarising (speakers=%s-%s)...", min_speakers, max_speakers)
    t0 = time.perf_counter()

    from pyannote.audio import Pipeline as PyannotePipeline

    def _do_diarize(device: str) -> object:
        diarize_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=pipeline.hf_token,
        )
        if device == "mps":
            import torch
            diarize_pipeline = diarize_pipeline.to(torch.device("mps"))
        kwargs: dict = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        return diarize_pipeline(str(audio_path), **kwargs)

    try:
        dev = pipeline.diarize_device
        if dev == "mps":
            try:
                diarization = _do_diarize("mps")
            except Exception as e:
                log.warning("Diarize on MPS failed (%s), retrying on CPU", e)
                diarization = _do_diarize("cpu")
        else:
            diarization = _do_diarize("cpu")
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

    segments = _assign_speakers(result.tokens, diarization)

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
