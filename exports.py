"""Transcript export helpers for Verbatim.

Each function writes one format to *out_dir* and returns the output Path.
The API.export_transcript method is a thin dispatcher that calls these.
"""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Time-formatting helpers
# ---------------------------------------------------------------------------

def _ts_short(seconds: float) -> str:
    """HH:MM:SS (no milliseconds) — used by MAXQDA and NVivo."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ts_srt(seconds: float) -> str:
    """HH:MM:SS,mmm — SRT timestamp format."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    """HH:MM:SS.mmm — WebVTT timestamp format."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_plain(segments: list[dict], stem: str, out_dir: Path) -> Path:
    """Plain text: speaker-grouped paragraphs separated by blank lines."""
    out = out_dir / f"{stem}_transcript.txt"
    lines: list[str] = []
    current = None
    buf: list[str] = []
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
    out.write_text("\n\n".join(lines), encoding="utf-8")
    return out


def export_srt(segments: list[dict], stem: str, out_dir: Path) -> Path:
    """SRT subtitle format."""
    out = out_dir / f"{stem}.srt"
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(
            f"{i}\n{_ts_srt(seg['start'])} --> {_ts_srt(seg['end'])}\n"
            f"[{speaker}] {text}\n"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_maxqda(segments: list[dict], stem: str, out_dir: Path) -> Path:
    """MAXQDA focus-group format: #HH:MM:SS# / Speaker: text."""
    out = out_dir / f"{stem}_maxqda.txt"
    lines: list[str] = []
    for seg in segments:
        ts = _ts_short(seg["start"])
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(f"#{ts}#\n{speaker}: {text}")
    out.write_text("\n\n".join(lines), encoding="utf-8")
    return out


def export_atlasti(segments: list[dict], stem: str, out_dir: Path) -> Path:
    """ATLAS.ti VTT format with speaker cue annotations."""
    out = out_dir / f"{stem}_atlasti.vtt"
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        start = _ts_vtt(seg["start"])
        end = _ts_vtt(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"<v {speaker}>{text}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_nvivo(segments: list[dict], stem: str, out_dir: Path) -> Path:
    """NVivo tab-delimited: timespan\\tSpeaker\\tContent."""
    out = out_dir / f"{stem}_nvivo.txt"
    lines: list[str] = []
    for seg in segments:
        start = _ts_short(seg["start"])
        end = _ts_short(seg["end"])
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        lines.append(f"{start}-{end}\t{speaker}\t{text}")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

EXPORTERS = {
    "plain": export_plain,
    "srt": export_srt,
    "maxqda": export_maxqda,
    "atlasti": export_atlasti,
    "nvivo": export_nvivo,
}
