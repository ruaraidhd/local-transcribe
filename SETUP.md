# Setup on a new Mac

One-time install for a fresh Apple Silicon Mac. Expect ~30 min including the
first model download.

## 1. HuggingFace account and terms

The diarisation step uses pyannote, which requires a HuggingFace account and
per-user acceptance of the model terms.

1. Create an account at https://huggingface.co/join (or log in).
2. Accept the terms on **both** of these pages while logged in:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Create a **Read** token at https://huggingface.co/settings/tokens — copy it;
   you'll paste it into `settings.toml` below.

## 2. System dependencies

Install Homebrew if it isn't already there:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then:

```bash
brew install ffmpeg uv
```

## 3. Project files

Copy or clone this project to `~/Applications/local-transcribe/` (any path
works; the launcher uses its own location).

```bash
mkdir -p ~/Applications
# ...copy the local_transcribe folder into ~/Applications/local-transcribe/
cd ~/Applications/local-transcribe
uv sync
```

`uv sync` will download Python 3.11, PyTorch, WhisperX, pyannote etc. —
around 3 GB. Give it a few minutes.

## 4. Settings

```bash
cp settings.example.toml settings.toml
```

Open `settings.toml` and paste the HF token from step 1 into the `hf_token =`
line. Save.

## 5. Warm the model caches

First real use downloads ~3 GB of model weights. Do this once up-front so the
user doesn't hit it on first click:

```bash
mkdir -p ~/Transcripts/in ~/Transcripts/out
# Drop any short audio clip into ~/Transcripts/in first, then:
uv run python transcribe.py ~/Transcripts/in -o ~/Transcripts/out
```

If diarisation works end-to-end (a `.txt` lands in the outbox with
`SPEAKER_00`/`SPEAKER_01` labels), the install is good. If you see a 401
error, the HF token or terms acceptance is wrong — revisit step 1.

## 6. Wrap as a Mac .app (Platypus)

Install Platypus (free) from https://sveinbjorn.org/platypus or via Homebrew:

```bash
brew install --cask platypus
```

Launch Platypus and configure:

- **App Name**: `Local Transcribe`
- **Script Path**: `~/Applications/local-transcribe/launch.sh`
- **Interface**: `None` (Toga draws its own window)
- **Identifier**: `com.rdobson.local_transcribe` (or any reverse-DNS you like)
- **Remain running after execution**: ✅
- Optional: drop a custom icon onto the icon well.

Click **Create App** → save `Local Transcribe.app` to `/Applications`.
Double-click to launch.

## Updating later

Pull or copy new project files over the existing folder, then:

```bash
cd ~/Applications/local-transcribe
uv sync
```

The Platypus `.app` doesn't need rebuilding — it just runs `launch.sh`, which
always picks up the latest code.

## Troubleshooting

- Logs: `~/Library/Logs/LocalTranscribe/transcribe.log`
- In the GUI: **Copy diagnostics** button → zip on the Desktop with the log
  and a redacted settings file. Email/send that zip when reporting issues.
- Common remote-install trip-ups:
  - HF token pasted without accepting pyannote terms → 401 on diarise.
  - Different HF account used for token vs. terms acceptance → same symptom.
  - `ffmpeg` not on PATH → audio load failure. `brew install ffmpeg` fixes it.
