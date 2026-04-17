"""Tests for API.rate_transcript and API.submit_feedback in gui.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api(monkeypatch, tmp_path):
    """Instantiate API() with all side-effectful dependencies patched out."""
    # Redirect Path.home() so feedback goes into tmp_path
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    # Prevent load_settings from reading (or failing on) real settings.toml
    with patch("transcribe.load_settings", return_value={}), \
         patch("gui.load_settings", return_value={}):
        # Suppress the audio server — it would bind a real port (harmless but noisy)
        with patch("gui.API._start_audio_server", return_value=None):
            import gui
            api = gui.API()
    return api


# ---------------------------------------------------------------------------
# rate_transcript tests
# ---------------------------------------------------------------------------

class TestRateTranscript:
    def test_creates_file_on_first_call(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.rate_transcript("recording.wav", 4)
        fb_file = tmp_path / "Documents" / "Verbatim" / "feedback.json"
        assert fb_file.exists(), "feedback.json should be created on first rating"

    def test_correct_structure_on_first_call(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.rate_transcript("recording.wav", 4)
        data = json.loads((tmp_path / "Documents" / "Verbatim" / "feedback.json").read_text())
        assert "ratings" in data
        assert "recording.wav" in data["ratings"]

    def test_stars_stored(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.rate_transcript("file.m4a", 5)
        data = json.loads((tmp_path / "Documents" / "Verbatim" / "feedback.json").read_text())
        assert data["ratings"]["file.m4a"]["stars"] == 5

    def test_timestamp_written(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.rate_transcript("file.m4a", 3)
        data = json.loads((tmp_path / "Documents" / "Verbatim" / "feedback.json").read_text())
        ts = data["ratings"]["file.m4a"]["timestamp"]
        # ISO-like: YYYY-MM-DDTHH:MM:SS
        assert "T" in ts and len(ts) >= 16, f"Unexpected timestamp format: {ts!r}"

    def test_second_call_different_file_appends(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.rate_transcript("file_a.wav", 4)
        api.rate_transcript("file_b.wav", 2)
        data = json.loads((tmp_path / "Documents" / "Verbatim" / "feedback.json").read_text())
        assert "file_a.wav" in data["ratings"]
        assert "file_b.wav" in data["ratings"]
        assert len(data["ratings"]) == 2

    def test_re_rating_same_file_updates_not_duplicates(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.rate_transcript("dup.wav", 3)
        api.rate_transcript("dup.wav", 5)
        data = json.loads((tmp_path / "Documents" / "Verbatim" / "feedback.json").read_text())
        # There should still be exactly one entry for this file
        assert len(data["ratings"]) == 1
        # And the stars should have been updated to 5
        assert data["ratings"]["dup.wav"]["stars"] == 5


# ---------------------------------------------------------------------------
# submit_feedback tests
# ---------------------------------------------------------------------------

class TestSubmitFeedback:
    def _fb_file(self, tmp_path) -> Path:
        return tmp_path / "Documents" / "Verbatim" / "feedback.json"

    def test_creates_file_if_missing(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.submit_feedback("bug", "something broke", 3, False)
        assert self._fb_file(tmp_path).exists()

    def test_appends_to_existing_file(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.submit_feedback("bug", "first issue", 2, False)
        api.submit_feedback("feature", "nice idea", 4, False)
        data = json.loads(self._fb_file(tmp_path).read_text())
        assert len(data["feedback"]) == 2

    def test_does_not_overwrite_previous_entries(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.submit_feedback("bug", "original entry", 1, False)
        api.submit_feedback("general", "second entry", 5, False)
        data = json.loads(self._fb_file(tmp_path).read_text())
        texts = [e["text"] for e in data["feedback"]]
        assert "original entry" in texts
        assert "second entry" in texts

    def test_returns_mailto_url(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        result = api.submit_feedback("general", "great app", 5, False)
        assert "mailto" in result
        assert result["mailto"].startswith("mailto:")

    def test_mailto_subject_contains_category(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        result = api.submit_feedback("bug", "crash on open", 2, False)
        # The URL-encoded subject should contain the category
        import urllib.parse
        parsed = urllib.parse.urlparse(result["mailto"])
        qs = urllib.parse.parse_qs(parsed.query)
        subject = urllib.parse.unquote(qs["subject"][0])
        assert "bug" in subject

    def test_include_diagnostics_adds_diagnostics_path(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        # Patch create_diagnostics_zip so it doesn't actually zip anything
        fake_diag = tmp_path / "diag.zip"
        fake_diag.touch()
        with patch("gui.create_diagnostics_zip", return_value=fake_diag):
            api.submit_feedback("bug", "with diag", 3, True)
        data = json.loads(self._fb_file(tmp_path).read_text())
        entry = data["feedback"][0]
        assert "diagnostics_path" in entry

    def test_app_version_stamped(self, monkeypatch, tmp_path):
        api = _make_api(monkeypatch, tmp_path)
        api.submit_feedback("general", "version check", 4, False)
        data = json.loads(self._fb_file(tmp_path).read_text())
        entry = data["feedback"][0]
        assert "app_version" in entry
        assert entry["app_version"]  # non-empty
