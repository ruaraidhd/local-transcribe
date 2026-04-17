"""Tests for load_settings and save_settings in transcribe.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Patch must happen before importing transcribe
import transcribe


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    """Saving a settings dict and loading it back yields the same values."""
    settings_file = tmp_path / "settings.toml"
    monkeypatch.setattr(transcribe, "SETTINGS_PATH", settings_file)
    # Also patch the module-level _models_dir to not exist so load_settings
    # doesn't try the bundled-mode path.
    monkeypatch.setattr(transcribe, "_models_dir", tmp_path / "nonexistent_models")

    original = {
        "hf_token": "hf_testtoken123",
        "outbox": "/tmp/output",
        "model": "mlx-community/parakeet-tdt-0.6b-v2",
        "device_mps": True,
    }
    transcribe.save_settings(original)
    assert settings_file.exists()

    loaded = transcribe.load_settings()
    assert loaded["hf_token"] == original["hf_token"]
    assert loaded["outbox"] == original["outbox"]
    assert loaded["model"] == original["model"]
    assert loaded["device_mps"] is True


def test_save_writes_model_key(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.toml"
    monkeypatch.setattr(transcribe, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(transcribe, "_models_dir", tmp_path / "nonexistent_models")

    transcribe.save_settings({
        "hf_token": "hf_abc",
        "model": "my-custom-model",
    })
    content = settings_file.read_text()
    assert 'model = "my-custom-model"' in content


def test_save_writes_outbox_key(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.toml"
    monkeypatch.setattr(transcribe, "SETTINGS_PATH", settings_file)

    transcribe.save_settings({
        "hf_token": "hf_abc",
        "outbox": "/Users/test/Transcripts",
    })
    content = settings_file.read_text()
    assert "outbox" in content
    assert "/Users/test/Transcripts" in content


def test_save_writes_device_mps_false(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.toml"
    monkeypatch.setattr(transcribe, "SETTINGS_PATH", settings_file)

    transcribe.save_settings({
        "hf_token": "hf_abc",
        "device_mps": False,
    })
    content = settings_file.read_text()
    assert "device_mps = false" in content


def test_load_missing_file_exits(tmp_path, monkeypatch):
    """When settings file is absent and no bundled models, load_settings calls sys.exit."""
    missing = tmp_path / "missing_settings.toml"
    monkeypatch.setattr(transcribe, "SETTINGS_PATH", missing)
    monkeypatch.setattr(transcribe, "_models_dir", tmp_path / "nonexistent_models")

    with pytest.raises(SystemExit):
        transcribe.load_settings()


def test_load_bundled_mode_no_token(tmp_path, monkeypatch):
    """In bundled mode (models dir exists), missing settings file is OK and returns {}."""
    missing = tmp_path / "missing_settings.toml"
    fake_models = tmp_path / "models"
    fake_models.mkdir()

    monkeypatch.setattr(transcribe, "SETTINGS_PATH", missing)
    monkeypatch.setattr(transcribe, "_models_dir", fake_models)

    # Should not raise — returns empty dict (no token required in bundled mode)
    result = transcribe.load_settings()
    assert isinstance(result, dict)
