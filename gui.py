"""PyWebView GUI for local_transcribe."""
from __future__ import annotations

import http.server
import json
import gc
import logging
import os
import secrets
import socketserver
import sys
import threading
import time
from pathlib import Path

# Use bundled models if available (eliminates HF token requirement).
_bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
_models_dir = _bundle_dir / "models"
if _models_dir.exists():
    os.environ["HF_HOME"] = str(_models_dir)

# Ensure ffmpeg is on PATH — bundled binary first, then Homebrew fallback.
_extra_paths = str(_bundle_dir) + ":/opt/homebrew/bin:/usr/local/bin"
if str(_bundle_dir) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _extra_paths + ":" + os.environ.get("PATH", "")

import webview

from logging_setup import (
    LOG_DIR,
    create_diagnostics_zip,
    install_asyncio_exception_handler,
    log_batch_start,
    log_startup_diagnostics,
    setup_logging,
)
from exports import EXPORTERS
from transcribe import (
    AUDIO_EXTS,
    SETTINGS_PATH,
    load_pipeline,
    load_settings,
    save_settings,
    transcribe_one,
)

log = logging.getLogger("local_transcribe")


# ---------------------------------------------------------------------------
# Local HTTP server for audio streaming (with byte-range support)
# ---------------------------------------------------------------------------

class _LimitedReader:
    """Wraps a file object and stops after `n` bytes — used for range responses."""
    def __init__(self, f, n):
        self.f = f
        self.remaining = n

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        data = self.f.read(size)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


class _RangeHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Minimal handler that serves a single mapped file with HTTP range support."""

    # Set by _make_audio_handler
    _file_map: dict = {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/audio/"):
            token = path[7:]
            fspath = self._file_map.get(token, "")
        else:
            fspath = ""

        if not fspath or not os.path.isfile(fspath):
            self.send_error(404)
            return

        try:
            f = open(fspath, "rb")
        except OSError:
            self.send_error(404)
            return

        size = os.fstat(f.fileno()).st_size
        range_header = self.headers.get("Range")

        # Guess MIME type
        ext = os.path.splitext(fspath)[1].lower()
        mime = {
            ".m4a": "audio/mp4", ".mp4": "audio/mp4",
            ".mp3": "audio/mpeg", ".wav": "audio/wav",
            ".flac": "audio/flac", ".ogg": "audio/ogg",
            ".aac": "audio/aac", ".webm": "audio/webm",
        }.get(ext, "audio/octet-stream")

        if range_header:
            start, end = self._parse_range(range_header, size)
            if start is None:
                f.close()
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            f.seek(start)
            reader = _LimitedReader(f, end - start + 1)
        else:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            reader = f

        try:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            reader.close()

    @staticmethod
    def _parse_range(header, size):
        try:
            units, rng = header.split("=", 1)
            if units.strip() != "bytes":
                return None, None
            start_s, end_s = rng.split("-", 1)
            if not start_s.strip():
                # Suffix range: bytes=-N  →  last N bytes
                suffix = int(end_s)
                return max(0, size - suffix), size - 1
            start = int(start_s)
            end = int(end_s) if end_s.strip() else size - 1
            return start, min(end, size - 1)
        except Exception:
            return None, None

    def log_message(self, format, *args):
        pass  # silence server logs


def _make_audio_handler(file_map: dict):
    """Return a handler class bound to the given file_map dict."""
    class Handler(_RangeHTTPRequestHandler):
        _file_map = file_map
    return Handler


class API:
    def __init__(self):
        self.settings = load_settings()
        self.pipeline = None
        self.window = None  # set after window created
        self._stop_flag = False
        self._busy = False
        # Results stored per-file for viewer
        self._results = {}  # filename -> {segments, tokens, speakers}
        # Audio server state
        self._audio_server_port: int | None = None
        self._audio_file_map: dict = {}  # token -> filesystem path
        self._start_audio_server()

    def pick_files(self):
        """Open native file dialog, return list of file paths."""
        exts = ";".join(f"*.{ext.lstrip('.')}" for ext in AUDIO_EXTS)
        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=True,
            file_types=(f"Audio files ({exts})",),
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

    def get_audio_url(self, filename):
        """Return an HTTP URL for the audio file, served via the local audio server."""
        data = self._results.get(filename)
        if not data or "audio_path" not in data:
            return {"error": "No audio"}
        if self._audio_server_port is None:
            return {"error": "Audio server not available"}
        token = secrets.token_urlsafe(16)
        self._audio_file_map[token] = data["audio_path"]
        return {"url": f"http://127.0.0.1:{self._audio_server_port}/audio/{token}"}

    def update_segment_text(self, filename, segment_index, new_text):
        """Update a segment's text in memory and write back to the outbox JSON."""
        data = self._results.get(filename)
        if not data or segment_index >= len(data["segments"]):
            return {"error": "Segment not found"}

        data["segments"][segment_index]["text"] = new_text.strip()
        data["segments"][segment_index]["edited"] = True
        data["segments"][segment_index]["reviewed"] = True  # editing implies review

        outbox = Path(self.settings.get("outbox", "~/Transcripts/out")).expanduser()
        stem = Path(filename).stem
        json_path = outbox / f"{stem}.json"
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text())
                existing["segments"] = data["segments"]
                json_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
            except Exception as e:
                log.warning("Failed to persist edit to %s: %s", json_path, e)

        return {"status": "saved"}

    def mark_segment_reviewed(self, filename, segment_index, reviewed=True):
        """Mark a segment as reviewed without editing it."""
        data = self._results.get(filename)
        if not data or segment_index >= len(data["segments"]):
            return {"error": "Segment not found"}
        data["segments"][segment_index]["reviewed"] = reviewed
        return {"status": "ok"}

    def _start_audio_server(self):
        """Start a background HTTP server for audio streaming."""
        try:
            handler = _make_audio_handler(self._audio_file_map)
            server = socketserver.TCPServer(("127.0.0.1", 0), handler)
            server.allow_reuse_address = True
            self._audio_server_port = server.server_address[1]
            log.debug("Audio server started on port %d", self._audio_server_port)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except Exception as e:
            log.warning("Could not start audio server: %s", e)
            self._audio_server_port = None

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
        exporter = EXPORTERS.get(format_type)
        if exporter is None:
            return {"error": f"Unknown format: {format_type}"}
        stem = Path(filename).stem
        desktop = Path.home() / "Desktop"
        out = exporter(data["segments"], stem, desktop)
        return {"path": str(out)}

    def pick_output_folder(self):
        result = self.window.create_file_dialog(webview.FileDialog.FOLDER)
        if result and len(result) > 0:
            folder = str(result[0])
            self.settings["outbox"] = folder
            save_settings(self.settings)
            return {"path": folder}
        return None

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

    def rate_transcript(self, filename, stars):
        """Store a star rating (1-5) for a transcript."""
        feedback_dir = Path.home() / "Documents" / "Verbatim"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_file = feedback_dir / "feedback.json"

        data = {}
        if feedback_file.exists():
            data = json.loads(feedback_file.read_text())

        if "ratings" not in data:
            data["ratings"] = {}
        data["ratings"][filename] = {
            "stars": stars,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        feedback_file.write_text(json.dumps(data, indent=2))
        return {"status": "saved"}

    def submit_feedback(self, category, text, overall_stars, include_diagnostics):
        """Save feedback locally and optionally prepare an email."""
        import urllib.parse

        feedback_dir = Path.home() / "Documents" / "Verbatim"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_file = feedback_dir / "feedback.json"

        data = {}
        if feedback_file.exists():
            data = json.loads(feedback_file.read_text())

        if "feedback" not in data:
            data["feedback"] = []

        entry = {
            "category": category,
            "text": text,
            "overall_stars": overall_stars,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "app_version": "0.3.0",
        }

        if include_diagnostics:
            try:
                diag_path = create_diagnostics_zip(SETTINGS_PATH)
                entry["diagnostics_path"] = str(diag_path)
            except Exception:
                pass

        data["feedback"].append(entry)
        feedback_file.write_text(json.dumps(data, indent=2))

        # Return mailto link for optional email
        subject = urllib.parse.quote(f"Verbatim feedback: {category}")
        body_lines = [
            f"Category: {category}",
            f"Overall rating: {'★' * overall_stars + '☆' * (5 - overall_stars) if overall_stars else 'not rated'}",
            "",
            text,
            "",
            "---",
            "Sent from Verbatim v0.3.0",
        ]
        if include_diagnostics and "diagnostics_path" in entry:
            body_lines.append(f"Diagnostics saved to: {entry['diagnostics_path']}")
            body_lines.append("(Please attach this file to the email)")
        body = urllib.parse.quote("\n".join(body_lines))
        mailto = f"mailto:support@southlondonscientific.co.uk?subject={subject}&body={body}"

        return {"status": "saved", "mailto": mailto}

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
            if self.pipeline is None or not self.pipeline.transcriber.is_loaded:
                self._push("status", {"message": "Loading model..."})
                model_id = self.settings.get("model", "mlx-community/parakeet-tdt-0.6b-v2")
                self.pipeline = load_pipeline(
                    self.settings.get("hf_token", ""), model_id, self.settings,
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
                    and not self.pipeline.transcriber.is_loaded
                ):
                    try:
                        self._push("status", {"message": "Reloading model..."})
                        model_id = self.settings.get("model", "mlx-community/parakeet-tdt-0.6b-v2")
                        self.pipeline = load_pipeline(
                            self.settings.get("hf_token", ""), model_id, self.settings,
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
        "Verbatim",
        url=str(Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "web" / "index.html"),
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
            api.pipeline = load_pipeline(api.settings.get("hf_token", ""), model_id, api.settings)
            api._push("model_ready", {})
        except Exception as e:
            log.exception("Model load failed")
            api._push("model_error", {"error": str(e)})

    window.events.loaded += lambda: threading.Thread(target=on_loaded, daemon=True).start()

    def on_closing():
        log.info("Window closing — shutting down.")
        os._exit(0)  # Force-exit; daemon threads + ML cleanup hang otherwise.

    window.events.closing += on_closing

    webview.start()


if __name__ == "__main__":
    main()
