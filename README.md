# Verbatim

Private, on-device audio transcription and speaker diarisation on macOS (Apple Silicon),
built on [parakeet-mlx](https://github.com/senstella/parakeet-mlx) (MLX GPU
transcription) + [pyannote.audio](https://github.com/pyannote/pyannote-audio)
(diarisation). Runs entirely on-device — no audio leaves the machine.

**English only.** parakeet-mlx is an English-language model.

Ships as a macOS app (PyWebView GUI) with a queue-based workflow: add files,
click Transcribe, outputs land in a folder you choose.

## Running

```bash
uv run python gui.py                              # GUI
uv run python transcribe.py path/to/audio.mp3     # CLI
```

Outputs per input file: `{name}.txt` (speaker-labelled transcript),
`{name}.srt` (subtitles), `{name}.json` (full word-level data).

## First-time setup on a new Mac

See [SETUP.md](SETUP.md).

## Logs

`~/Library/Logs/LocalTranscribe/transcribe.log` (rotated, 5 × 1MB). The GUI
has an **Open logs** button and a **Copy diagnostics** button that zips the
current log plus a redacted `settings.toml` onto the Desktop — hand that zip
over when debugging a remote install.
