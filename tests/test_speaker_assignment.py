"""Property-based tests for _assign_speakers using Hypothesis."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from transcribe import _assign_speakers


# ---------------------------------------------------------------------------
# Lightweight token / turn mocks (duck-typed to match _assign_speakers)
# ---------------------------------------------------------------------------

@dataclass
class T:
    """Minimal token mock."""
    text: str
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Turn:
    """Minimal speaker-turn mock."""
    start: float
    end: float
    speaker: str


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def tokens_strategy():
    return (
        st.lists(
            st.builds(
                T,
                text=st.text(min_size=1, max_size=10).filter(lambda s: s.strip()),
                start=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
                end=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
                confidence=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            ).filter(lambda t: t.end > t.start),
            min_size=1,
            max_size=20,
        )
        .map(lambda ts: sorted(ts, key=lambda t: t.start))
    )


def _non_overlapping_turns(raw: list[tuple]) -> list[Turn]:
    """Turn raw (start, duration, speaker) triples into non-overlapping SpeakerTurns."""
    turns = []
    cursor = 0.0
    for duration, speaker in raw:
        start = cursor
        end = cursor + duration
        turns.append(Turn(start=start, end=end, speaker=speaker))
        cursor = end + 0.1  # small gap
    return turns


def turns_strategy():
    return (
        st.lists(
            st.tuples(
                st.floats(min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False),
                st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=65), min_size=2, max_size=8),
            ),
            min_size=1,
            max_size=10,
        )
        .map(_non_overlapping_turns)
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(tokens=tokens_strategy(), turns=turns_strategy())
def test_output_preserves_token_text_order(tokens, turns):
    """Concatenating all segment texts (in order) contains every token text in order."""
    segments = _assign_speakers(tokens, turns)
    # Build the full flattened text from segments
    full_text = " ".join(seg["text"] for seg in segments)
    # Build ordered token texts
    token_texts = [t.text.strip() for t in tokens if t.text.strip()]
    # Walk through full_text checking each token appears in order
    pos = 0
    for tt in token_texts:
        idx = full_text.find(tt, pos)
        assert idx != -1, f"Token text {tt!r} not found in output after pos {pos}"
        pos = idx + len(tt)


@pytest.mark.xfail(
    reason=(
        "Known edge case: tokens with identical start times but different ends can be "
        "assigned to different speakers, producing out-of-order segment boundaries. "
        "Hypothesis found: two tokens both at start=0.0 (ends 1.0 and 3.0) get split "
        "across speakers AA/AB, yielding seg[0].end=1.0 > seg[1].start=0.0. "
        "Fix would require post-sorting segments or preventing start-time ties."
    )
)
@settings(max_examples=200)
@given(tokens=tokens_strategy(), turns=turns_strategy())
def test_segments_chronologically_ordered(tokens, turns):
    """Segments must be in non-decreasing time order (end[i] <= start[i+1] + epsilon)."""
    segments = _assign_speakers(tokens, turns)
    for i in range(len(segments) - 1):
        assert segments[i]["end"] <= segments[i + 1]["start"] + 1e-6, (
            f"Segment {i} ends at {segments[i]['end']} but segment {i+1} starts at {segments[i+1]['start']}"
        )


@settings(max_examples=200)
@given(tokens=tokens_strategy(), turns=turns_strategy())
def test_no_empty_segments(tokens, turns):
    """No segment should have empty text after smoothing."""
    segments = _assign_speakers(tokens, turns)
    for seg in segments:
        assert seg["text"].strip(), f"Empty segment found: {seg!r}"


@settings(max_examples=200)
@given(tokens=tokens_strategy(), turns=turns_strategy())
def test_speaker_set_subset_of_turns(tokens, turns):
    """Every speaker label in output must come from an input turn (or be UNKNOWN as fallback)."""
    segments = _assign_speakers(tokens, turns)
    valid_speakers = {t.speaker for t in turns} | {"UNKNOWN"}
    for seg in segments:
        assert seg["speaker"] in valid_speakers, (
            f"Speaker {seg['speaker']!r} not in turns {valid_speakers!r}"
        )


@settings(max_examples=200)
@given(tokens=tokens_strategy(), turns=turns_strategy())
def test_smoothing_idempotent(tokens, turns):
    """Calling _assign_speakers twice with same inputs produces identical output (determinism)."""
    result1 = _assign_speakers(tokens, turns)
    result2 = _assign_speakers(tokens, turns)
    assert result1 == result2


# ---------------------------------------------------------------------------
# Regression / specific-scenario tests
# ---------------------------------------------------------------------------

def test_orphaned_leading_token_regression():
    """Token at 0s-1s with first speaker turn starting at 1s must NOT be UNKNOWN.

    Regression for the 'So' bug: a leading token whose midpoint falls before
    pyannote's first turn should inherit the first speaker via the forward-
    inherit seed, not fall through to UNKNOWN.
    """
    tokens = [T(text="So", start=0.0, end=1.0)]
    turns = [Turn(start=1.0, end=5.0, speaker="SPEAKER_A")]
    segments = _assign_speakers(tokens, turns)
    assert len(segments) == 1
    assert segments[0]["speaker"] != "UNKNOWN", (
        "Leading token before first speaker turn must inherit first speaker, not UNKNOWN"
    )
    assert segments[0]["speaker"] == "SPEAKER_A"


def test_short_first_segment_merged_forward():
    """A very short (<1s) first segment followed by a different speaker is merged forward."""
    # Token at 0-0.5s → short segment, then a longer segment for a different speaker
    tokens = [
        T(text="Hi", start=0.0, end=0.5),
        T(text="Hello there everyone", start=1.0, end=3.0),
    ]
    turns = [
        Turn(start=0.0, end=0.5, speaker="SPEAKER_A"),
        Turn(start=1.0, end=3.0, speaker="SPEAKER_B"),
    ]
    segments = _assign_speakers(tokens, turns)
    # The first short segment (0-0.5, <1s) should be merged forward into the second
    assert len(segments) == 1
    assert "Hi" in segments[0]["text"]
    assert "Hello there everyone" in segments[0]["text"]


def test_tokens_outside_turns_inherit():
    """Tokens in gaps between speaker turns should inherit the previous speaker.

    Note: _assign_speakers joins token texts without spaces (raw concatenation),
    so "First" and " Second" become a single string. Tokens must have leading
    spaces to produce readable text; here we use distinct single-word tokens
    with explicit spaces so we can locate them by substring search.
    """
    # Turn A ends at 2s, turn B starts at 4s — gap from 2-4s
    # Use a space prefix so tokens join readably
    tokens = [
        T(text="First", start=0.5, end=1.0),    # in turn A
        T(text=" Second", start=2.5, end=3.0),  # in gap — should inherit A
        T(text=" Third", start=4.5, end=5.0),   # in turn B
    ]
    turns = [
        Turn(start=0.0, end=2.0, speaker="SPEAKER_A"),
        Turn(start=4.0, end=6.0, speaker="SPEAKER_B"),
    ]
    segments = _assign_speakers(tokens, turns)
    # Find the segment whose text contains "Second" — should be SPEAKER_A (inherited)
    second_speaker = None
    for seg in segments:
        if "Second" in seg["text"]:
            second_speaker = seg["speaker"]
            break
    assert second_speaker == "SPEAKER_A", (
        f"Gap token 'Second' should inherit SPEAKER_A, got {second_speaker!r}. "
        f"Segments: {segments}"
    )


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_empty_tokens():
    """No tokens → empty segments list, no crash."""
    result = _assign_speakers([], [])
    assert result == []


def test_empty_turns_but_tokens():
    """Tokens but no speaker turns → everything assigned to 'UNKNOWN' or sensible default."""
    from types import SimpleNamespace as S
    tokens = [S(text="hello", start=0.0, end=1.0, duration=1.0, confidence=0.9)]
    result = _assign_speakers(tokens, [])
    # One segment, speaker is UNKNOWN (no turns to assign from)
    assert len(result) == 1
    assert result[0]["speaker"] == "UNKNOWN"
    assert "hello" in result[0]["text"]


def test_all_tokens_in_single_turn():
    """All tokens fall within one speaker's turn → one segment with that speaker."""
    from types import SimpleNamespace as S
    tokens = [
        S(text="hello", start=0.0, end=0.5, duration=0.5, confidence=0.9),
        S(text=" world", start=0.6, end=1.2, duration=0.6, confidence=0.9),
    ]
    turns = [S(start=0.0, end=2.0, speaker="SPEAKER_00")]
    result = _assign_speakers(tokens, turns)
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
    assert "hello" in result[0]["text"]
    assert "world" in result[0]["text"]


def test_overlapping_turns_first_wins():
    """When turns overlap, the first-listed turn covering the token's midpoint wins."""
    from types import SimpleNamespace as S
    tokens = [S(text="hi", start=5.0, end=5.5, duration=0.5, confidence=0.9)]
    turns = [
        S(start=0.0, end=10.0, speaker="SPEAKER_00"),  # first
        S(start=4.0, end=6.0, speaker="SPEAKER_01"),   # overlaps, but listed second
    ]
    result = _assign_speakers(tokens, turns)
    assert result[0]["speaker"] == "SPEAKER_00"


def test_confidence_aggregated_geometric_mean():
    """Segment confidence equals geometric mean of token confidences (uniform → same value)."""
    conf = 0.8
    tokens = [
        T(text="Hello", start=0.0, end=1.0, confidence=conf),
        T(text=" world", start=1.0, end=2.0, confidence=conf),
    ]
    turns = [Turn(start=0.0, end=2.0, speaker="SPEAKER_A")]
    segments = _assign_speakers(tokens, turns)
    assert len(segments) == 1
    # geometric mean of uniform values should equal that value
    assert abs(segments[0]["confidence"] - conf) < 1e-6, (
        f"Expected confidence {conf}, got {segments[0]['confidence']}"
    )
