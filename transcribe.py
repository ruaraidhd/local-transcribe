"""Transcribe and diarise audio locally with WhisperX.

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
    model: object
    device: str
    hf_token: str


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
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Pipeline:
    import whisperx
    log.info("Loading Whisper model %s on %s (%s)...", model_name, device, compute_type)
    t0 = time.perf_counter()
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    log.info("Model loaded (%.1fs)", time.perf_counter() - t0)
    return Pipeline(model=model, device=device, hf_token=hf_token)


def transcribe_one(
    audio_path: Path,
    out_dir: Path,
    pipeline: Pipeline,
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    log.info("=== %s ===", audio_path.name)
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    audio = whisperx.load_audio(str(audio_path))
    log.debug("Load audio: %.1fs", time.perf_counter() - t0)

    log.info("Transcribing...")
    t0 = time.perf_counter()
    result = pipeline.model.transcribe(audio, batch_size=8, language=language)
    lang = result["language"]
    log.info("Transcribe: %.1fs (language: %s, %d segments)",
             time.perf_counter() - t0, lang, len(result["segments"]))

    log.info("Aligning...")
    t0 = time.perf_counter()
    align_model, align_meta = whisperx.load_align_model(
        language_code=lang, device=pipeline.device
    )
    result = whisperx.align(
        result["segments"], align_model, align_meta, audio, pipeline.device,
        return_char_alignments=False,
    )
    del align_model
    log.info("Align: %.1fs", time.perf_counter() - t0)

    log.info("Diarising (speakers=%s-%s)...", min_speakers, max_speakers)
    t0 = time.perf_counter()
    try:
        diarize_model = DiarizationPipeline(token=pipeline.hf_token, device=pipeline.device)
        diarize_segments = diarize_model(
            audio, min_speakers=min_speakers, max_speakers=max_speakers
        )
    except Exception as e:
        hint = hint_for_pyannote_error(e)
        if hint:
            log.error("%s", hint)
        raise
    result = whisperx.assign_word_speakers(diarize_segments, result)
    log.info("Diarise: %.1fs", time.perf_counter() - t0)

    stem = audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_srt(result["segments"], out_dir / f"{stem}.srt")
    write_txt(result["segments"], out_dir / f"{stem}.txt")
    log.info("Wrote %s.{json,srt,txt}  (total %.1fs)",
             stem, time.perf_counter() - t_total)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Audio file or directory of audio files")
    ap.add_argument("-o", "--output", type=Path, default=Path("output"))
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--language", default=None)
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
        settings["hf_token"], args.model, args.device, args.compute_type,
    )
    failures = 0
    for audio_path in inputs:
        try:
            transcribe_one(
                audio_path, args.output, pipeline,
                args.language, args.min_speakers, args.max_speakers,
            )
        except Exception:
            failures += 1
            log.exception("FAILED %s", audio_path.name)
    log.info("Batch complete: %d ok, %d failed", len(inputs) - failures, failures)


if __name__ == "__main__":
    main()
