"""Tests for transcript export functions in exports.py."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from exports import (
    export_atlasti,
    export_maxqda,
    export_nvivo,
    export_plain,
    export_srt,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SAMPLE_SEGMENTS = [
    {
        "speaker": "SPEAKER_A",
        "start": 0.0,
        "end": 5.5,
        "text": "Hello everyone thank you for joining",
        "confidence": 0.9,
        "edited": False,
        "reviewed": False,
    },
    {
        "speaker": "SPEAKER_B",
        "start": 6.0,
        "end": 12.3,
        "text": "Thanks for having me on the show",
        "confidence": 0.85,
        "edited": False,
        "reviewed": False,
    },
    {
        "speaker": "SPEAKER_A",
        "start": 13.0,
        "end": 18.0,
        "text": "Great let us dive in",
        "confidence": 0.95,
        "edited": False,
        "reviewed": False,
    },
]


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def test_export_plain_creates_file(tmp):
    out = export_plain(SAMPLE_SEGMENTS, "test", tmp)
    assert out.exists()


def test_export_plain_groups_same_speaker(tmp):
    """Consecutive same-speaker segments should be joined."""
    segs = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "Hello"},
        {"speaker": "A", "start": 1.0, "end": 2.0, "text": "world"},
    ]
    out = export_plain(segs, "test", tmp)
    content = out.read_text()
    # Should appear as one paragraph, not two
    assert content.count("A:") == 1
    assert "Hello world" in content


def test_export_plain_speaker_prefix(tmp):
    out = export_plain(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    assert "SPEAKER_A:" in content
    assert "SPEAKER_B:" in content


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------

_SRT_LINE_PATTERN = re.compile(
    r"^\d+\n"
    r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n"
    r"\[[\w_]+\] .+",
    re.MULTILINE,
)


def test_export_srt_format(tmp):
    out = export_srt(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    assert len(blocks) == len(SAMPLE_SEGMENTS)
    for block in blocks:
        lines = block.splitlines()
        # line 0: index
        assert lines[0].isdigit()
        # line 1: timestamp
        assert "-->" in lines[1]
        assert re.match(r"\d{2}:\d{2}:\d{2},\d{3}", lines[1])
        # line 2: [SPEAKER] text
        assert re.match(r"\[[\w_]+\]", lines[2])


def test_export_srt_index_sequential(tmp):
    out = export_srt(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    indices = [int(b.splitlines()[0]) for b in blocks]
    assert indices == list(range(1, len(SAMPLE_SEGMENTS) + 1))


# ---------------------------------------------------------------------------
# MAXQDA
# ---------------------------------------------------------------------------

def test_export_maxqda_timestamp_format(tmp):
    out = export_maxqda(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    # Each block should start with #HH:MM:SS#
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    assert len(blocks) == len(SAMPLE_SEGMENTS)
    for block in blocks:
        first_line = block.splitlines()[0]
        assert re.match(r"#\d{2}:\d{2}:\d{2}#", first_line), (
            f"Expected MAXQDA timestamp, got: {first_line!r}"
        )


def test_export_maxqda_speaker_colon(tmp):
    out = export_maxqda(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    for block in blocks:
        second_line = block.splitlines()[1]
        assert ":" in second_line, f"Expected Speaker: text, got: {second_line!r}"


# ---------------------------------------------------------------------------
# ATLAS.ti VTT
# ---------------------------------------------------------------------------

def test_export_atlasti_webvtt_header(tmp):
    out = export_atlasti(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    assert content.startswith("WEBVTT"), "VTT file must start with WEBVTT"


def test_export_atlasti_speaker_cues(tmp):
    out = export_atlasti(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    for seg in SAMPLE_SEGMENTS:
        speaker = seg["speaker"]
        assert f"<v {speaker}>" in content, f"Missing cue for {speaker!r}"


def test_export_atlasti_vtt_timestamps(tmp):
    out = export_atlasti(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    # Should have HH:MM:SS.mmm --> HH:MM:SS.mmm lines
    ts_pattern = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}")
    assert ts_pattern.search(content), "No VTT timestamp lines found"


# ---------------------------------------------------------------------------
# NVivo
# ---------------------------------------------------------------------------

def test_export_nvivo_tab_delimited(tmp):
    out = export_nvivo(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    lines = [l for l in content.splitlines() if l.strip()]
    assert len(lines) == len(SAMPLE_SEGMENTS)
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == 3, f"Expected 3 tab-separated fields, got {len(parts)}: {line!r}"


def test_export_nvivo_timespan_format(tmp):
    out = export_nvivo(SAMPLE_SEGMENTS, "test", tmp)
    content = out.read_text()
    lines = [l for l in content.splitlines() if l.strip()]
    for line in lines:
        timespan = line.split("\t")[0]
        assert re.match(r"\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}", timespan), (
            f"Unexpected timespan format: {timespan!r}"
        )
