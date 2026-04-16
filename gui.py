"""PyWebView GUI for local_transcribe."""
from __future__ import annotations

import json
import gc
import logging
import threading
import time
from pathlib import Path

import webview

from logging_setup import (
    LOG_DIR,
    create_diagnostics_zip,
    install_asyncio_exception_handler,
    log_batch_start,
    log_startup_diagnostics,
    setup_logging,
)
from transcribe import (
    AUDIO_EXTS,
    SETTINGS_PATH,
    load_pipeline,
    load_settings,
    save_settings,
    transcribe_one,
)

log = logging.getLogger("local_transcribe")


class API:
    def __init__(self):
        self.settings = load_settings()
        self.pipeline = None
        self.window = None  # set after window created
        self._stop_flag = False
        self._busy = False
        # Results stored per-file for viewer
        self._results = {}  # filename -> {segments, tokens, speakers}

    def pick_files(self):
        """Open native file dialog, return list of file paths."""
        file_types = ("Audio Files", " ".join(f"*.{ext.lstrip('.')}" for ext in AUDIO_EXTS))
        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=True,
            file_types=(file_types,),
        )
        if result:
            return [str(p) for p in result]
        return []

    def start_transcription(self, file_paths, num_speakers):
        """Start transcription in background thread."""
        if self._busy:
            return {"error": "Already running"}
        self._busy = True
        self._stop_flag = False
        thread = threading.Thread(
            target=self._transcribe_batch,
            args=(file_paths, num_speakers),
            daemon=True,
        )
        thread.start()
        return {"status": "started"}

    def stop(self):
        self._stop_flag = True

    def get_transcript(self, filename):
        """Return transcript data for the viewer."""
        return self._results.get(filename, {})

    def rename_speaker(self, filename, old_name, new_name):
        """Rename a speaker in stored results."""
        if filename not in self._results:
            return
        for seg in self._results[filename]["segments"]:
            if seg["speaker"] == old_name:
                seg["speaker"] = new_name
        # Update the speakers list
        speakers = self._results[filename]["speakers"]
        if old_name in speakers:
            idx = speakers.index(old_name)
            speakers[idx] = new_name

    def export_transcript(self, filename, format_type):
        """Export transcript. Returns the path to the exported file."""
        data = self._results.get(filename)
        if not data:
            return {"error": "No transcript data"}
        stem = Path(filename).stem
        desktop = Path.home() / "Desktop"

        if format_type == "plain":
            out = desktop / f"{stem}_transcript.txt"
            lines, current, buf = [], None, []
            for seg in data["segments"]:
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

        elif format_type == "srt":
            from transcribe import write_srt
            out = desktop / f"{stem}.srt"
            write_srt(data["segments"], out)

        elif format_type == "maxqda":
            from transcribe import write_srt
            out = desktop / f"{stem}_maxqda.srt"
            write_srt(data["segments"], out)

        elif format_type == "atlasti":
            from transcribe import write_srt
            out = desktop / f"{stem}_atlasti.srt"
            write_srt(data["segments"], out)

        elif format_type == "nvivo":
            out = desktop / f"{stem}_nvivo.txt"
            from transcribe import format_ts
            lines = []
            for seg in data["segments"]:
                ts = format_ts(seg["start"]).split(",")[0]  # HH:MM:SS without ms
                lines.append(f"{ts}\t{seg.get('speaker', 'UNKNOWN')}")
                lines.append(f"{seg.get('text', '').strip()}\n")
            out.write_text("\n".join(lines), encoding="utf-8")

        else:
            return {"error": f"Unknown format: {format_type}"}

        return {"path": str(out)}

    def open_logs(self):
        import subprocess
        subprocess.run(["open", str(LOG_DIR)], check=False)

    def copy_diagnostics(self):
        try:
            path = create_diagnostics_zip(SETTINGS_PATH)
            import subprocess
            subprocess.run(["open", "-R", str(path)], check=False)
            return {"path": str(path)}
        except Exception as e:
            return {"error": str(e)}

    def _push(self, event, data):
        """Push an event to the JS frontend."""
        js = f"window.onPythonEvent({json.dumps(event)}, {json.dumps(data)})"
        try:
            self.window.evaluate_js(js)
        except Exception:
            pass  # window may be closing

    def _transcribe_batch(self, file_paths, num_speakers):
        """Run transcription batch on background thread."""
        try:
            # Load model if needed
            if self.pipeline is None or self.pipeline.model is None:
                self._push("status", {"message": "Loading model..."})
                model_id = self.settings.get("model", "mlx-community/parakeet-tdt-0.6b-v2")
                self.pipeline = load_pipeline(
                    self.settings["hf_token"], model_id, self.settings,
                )

            speakers = int(num_speakers) if num_speakers else None
            total = len(file_paths)

            for i, fpath in enumerate(file_paths):
                if self._stop_flag:
                    self._push("stopped", {})
                    break

                filename = Path(fpath).name
                self._push("file_start", {
                    "index": i,
                    "total": total,
                    "filename": filename,
                })

                try:
                    outbox = Path(self.settings.get("outbox", "~/Transcripts/out")).expanduser()
                    outbox.mkdir(parents=True, exist_ok=True)

                    def on_stage_start(stage):
                        self._push("stage", {"filename": filename, "stage": stage, "event": "start"})

                    def on_stage_end(stage):
                        self._push("stage", {"filename": filename, "stage": stage, "event": "end"})

                    def on_progress(pct):
                        self._push("progress", {"filename": filename, "percent": round(pct, 1)})

                    result = transcribe_one(
                        Path(fpath), outbox, self.pipeline,
                        None, speakers, speakers,
                        self.settings,
                        on_stage_start, on_stage_end, on_progress,
                    )

                    # Store results for viewer
                    segments = result["segments"]
                    speakers_list = sorted(set(s["speaker"] for s in segments))
                    self._results[filename] = {
                        "segments": segments,
                        "speakers": speakers_list,
                        "audio_path": fpath,
                    }

                    self._push("file_done", {"index": i, "filename": filename})

                except Exception as e:
                    log.exception("FAILED %s", filename)
                    self._push("file_error", {
                        "index": i,
                        "filename": filename,
                        "error": _friendly_error(str(e)),
                    })

                # transcribe_one frees the parakeet model internally.
                # Reload it if there are more files to process.
                if (
                    i < total - 1
                    and not self._stop_flag
                    and self.pipeline is not None
                    and self.pipeline.model is None
                ):
                    try:
                        self._push("status", {"message": "Reloading model..."})
                        model_id = self.settings.get("model", "mlx-community/parakeet-tdt-0.6b-v2")
                        self.pipeline = load_pipeline(
                            self.settings["hf_token"], model_id, self.settings,
                        )
                    except Exception:
                        log.exception("Failed to reload model between files")

            self._push("batch_done", {"total": total})

        except Exception as e:
            log.exception("Batch failed")
            self._push("batch_error", {"error": _friendly_error(str(e))})
        finally:
            self._busy = False


def _friendly_error(msg):
    """Convert technical errors to human-readable messages."""
    lower = msg.lower()
    if "disk" in lower or "space" in lower:
        return "Not enough free disk space. Please free up at least 10 GB and try again."
    if "401" in lower or "403" in lower or "gated" in lower:
        return "Model access denied. Please check your HuggingFace token in settings."
    if "ffmpeg" in lower:
        return "ffmpeg is not installed. Please install it with: brew install ffmpeg"
    if "metal" in lower or "allocat" in lower:
        return "Not enough GPU memory for this file. Try closing other apps."
    return f"Something went wrong: {msg}"


def main():
    setup_logging()
    api = API()

    log_startup_diagnostics(
        api.settings,
        None,
        Path(api.settings.get("outbox", "~/Transcripts/out")).expanduser(),
    )

    window = webview.create_window(
        "Local Transcribe",
        url=str(Path(__file__).parent / "web" / "index.html"),
        js_api=api,
        width=800,
        height=700,
        min_size=(600, 500),
        text_select=True,
    )
    api.window = window

    # Load model in background after window opens
    def on_loaded():
        api._push("status", {"message": "Loading model..."})
        try:
            model_id = api.settings.get("model", "mlx-community/parakeet-tdt-0.6b-v2")
            api.pipeline = load_pipeline(api.settings["hf_token"], model_id, api.settings)
            api._push("model_ready", {})
        except Exception as e:
            log.exception("Model load failed")
            api._push("model_error", {"error": str(e)})

    window.events.loaded += lambda: threading.Thread(target=on_loaded, daemon=True).start()

    webview.start()


if __name__ == "__main__":
    main()
