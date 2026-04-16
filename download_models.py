"""Download model weights for offline use.

Run once after cloning:
    uv run python download_models.py

Downloads parakeet-mlx and pyannote models into a local models/ directory.
When models/ exists, the app loads from there instead of the HuggingFace
cache, eliminating the need for a HuggingFace account or token.
"""
import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS = [
    "mlx-community/parakeet-tdt-0.6b-v2",
    "pyannote/speaker-diarization-community-1",
]


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR)

    from huggingface_hub import snapshot_download

    # Pyannote is gated — needs a token for the initial download.
    # After that, the cached files work without one.
    token = None
    settings_path = Path(__file__).parent / "settings.toml"
    if settings_path.exists():
        import tomllib
        s = tomllib.loads(settings_path.read_text())
        token = s.get("hf_token", "").strip() or None

    for repo_id in MODELS:
        print(f"Downloading {repo_id}...")
        try:
            snapshot_download(repo_id, token=token)
            print(f"  done.")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            if "gated" in str(e).lower() or "401" in str(e):
                print(
                    "  This model requires a HuggingFace token. "
                    "Set hf_token in settings.toml first.",
                    file=sys.stderr,
                )
            sys.exit(1)

    print(f"\nAll models downloaded to {MODELS_DIR}")
    print("The app will use these automatically — no HF token needed at runtime.")


if __name__ == "__main__":
    main()
