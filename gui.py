"""Toga GUI for local_transcribe."""
from __future__ import annotations

import asyncio
import gc
import logging
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

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

MODEL_OPTIONS = [
    ("Parakeet TDT v2 (0.6B)", "mlx-community/parakeet-tdt-0.6b-v2"),
]

log = logging.getLogger("local_transcribe")


def _fmt_size(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return "?"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    return f"{n / 1024:.0f} KB"


@dataclass
class QueueItem:
    path: Path
    size_str: str = field(default="")
    status: str = "Queued"

    def __post_init__(self) -> None:
        if not self.size_str:
            self.size_str = _fmt_size(self.path)


class UILogHandler(logging.Handler):
    """Bridges the logger into the Toga UI (log pane + status line).

    Must be attached from the main thread. Dispatches updates to the loop via
    call_soon_threadsafe so it's safe from worker threads too.
    """

    def __init__(self, app: "TranscribeApp") -> None:
        super().__init__()
        self.app = app
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            return
        # status line shows just the message at INFO+
        status = record.getMessage() if record.levelno >= logging.INFO else None
        self.app.loop.call_soon_threadsafe(self.app._ui_log, text, status)


class TranscribeApp(toga.App):
    def startup(self) -> None:
        setup_logging()
        install_asyncio_exception_handler(self.loop)

        self.settings = load_settings()
        self.outbox = Path(self.settings.get("outbox", "~/Transcripts/out")).expanduser()
        self.outbox.mkdir(parents=True, exist_ok=True)

        self.pipeline = None
        self.busy = False
        self._stop_flag = False

        # Stage ticker state
        self.current_stage: str | None = None
        self.stage_started: float | None = None
        self._last_progress_pct: float | None = None

        # Queue data model: list of QueueItem, paths used for dedup
        self.queue: list[QueueItem] = []
        self._queue_paths: set[Path] = set()

        root = toga.Box(style=Pack(direction=COLUMN, padding=12, flex=1))

        # ---- Config group ----
        config_group = toga.Box(style=Pack(direction=COLUMN))

        # Output folder row
        out_row = toga.Box(style=Pack(direction=ROW, padding_bottom=8))
        self.outbox_label = toga.Label(
            f"Output folder: {self.outbox}",
            style=Pack(flex=1, padding_top=6),
        )
        out_row.add(self.outbox_label)
        out_row.add(toga.Button("Change...", on_press=self.on_pick_outbox,
                                style=Pack(padding_right=6)))
        out_row.add(toga.Button("Open", on_press=self.on_open_outputs))
        config_group.add(out_row)

        # Settings row: Model | Speakers
        settings_row = toga.Box(style=Pack(direction=ROW, padding_bottom=8))
        settings_row.add(toga.Label("Model:", style=Pack(padding_right=6, padding_top=6)))
        default_model_id = self.settings.get("model", MODEL_OPTIONS[0][1])
        model_display_names = [label for label, _ in MODEL_OPTIONS]
        default_display = next(
            (label for label, mid in MODEL_OPTIONS if mid == default_model_id),
            model_display_names[0],
        )
        self.model_select = toga.Selection(
            items=model_display_names,
            style=Pack(width=240, padding_right=16),
        )
        self.model_select.value = default_display
        self.model_select.on_change = self.on_model_changed
        settings_row.add(self.model_select)

        settings_row.add(toga.Label("Speakers:", style=Pack(padding_right=6, padding_top=6)))
        self.speakers_input = toga.TextInput(
            placeholder="auto", style=Pack(width=60)
        )
        settings_row.add(self.speakers_input)
        config_group.add(settings_row)

        root.add(config_group)

        # ---- Queue group ----
        queue_group = toga.Box(style=Pack(direction=COLUMN, padding_top=12))

        self.table = toga.Table(
            headings=["File", "Size", "Status"],
            accessors=["file", "size", "status"],
            multiple_select=True,
            style=Pack(flex=1, padding_bottom=4),
        )
        queue_group.add(self.table)

        queue_buttons = toga.Box(style=Pack(direction=ROW, padding_bottom=4))
        queue_buttons.add(toga.Button(
            "Add files...", on_press=self.on_add_files,
            style=Pack(padding_right=6),
        ))
        queue_buttons.add(toga.Button(
            "Add folder...", on_press=self.on_add_folder,
            style=Pack(padding_right=6),
        ))
        queue_buttons.add(toga.Button(
            "Remove selected", on_press=self.on_remove_selected,
            style=Pack(padding_right=6),
        ))
        queue_buttons.add(toga.Button(
            "Clear completed", on_press=self.on_clear_completed,
        ))
        queue_group.add(queue_buttons)

        root.add(queue_group)

        # ---- Actions group ----
        actions_group = toga.Box(style=Pack(direction=ROW, padding_top=12, padding_bottom=8))
        self.go_button = toga.Button(
            "Transcribe all", on_press=self.on_transcribe,
            style=Pack(flex=1, padding_right=8),
        )
        actions_group.add(self.go_button)
        self.stop_button = toga.Button(
            "Stop", on_press=self.on_stop,
        )
        self.stop_button.enabled = False
        actions_group.add(self.stop_button)
        root.add(actions_group)

        # ---- Status group ----
        status_group = toga.Box(style=Pack(direction=COLUMN, padding_top=12, flex=1))

        self.status_label = toga.Label("Now: Ready.", style=Pack(padding_bottom=6))
        status_group.add(self.status_label)

        self.log_view = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, height=200))
        status_group.add(self.log_view)

        # Footer row: Open logs + Copy diagnostics
        footer_row = toga.Box(style=Pack(direction=ROW, padding_top=4))
        footer_row.add(toga.Box(style=Pack(flex=1)))  # spacer
        footer_row.add(toga.Button("Open logs", on_press=self.on_open_logs,
                                   style=Pack(padding_right=6)))
        footer_row.add(toga.Button("Copy diagnostics", on_press=self.on_copy_diagnostics))
        status_group.add(footer_row)

        root.add(status_group)

        self.main_window = toga.MainWindow(title=self.formal_name, size=(760, 780))
        self.main_window.content = root
        self.main_window.show()

        # Wire the logger into the UI now that widgets exist.
        ui_handler = UILogHandler(self)
        ui_handler.setLevel(logging.INFO)
        log.addHandler(ui_handler)

        log_startup_diagnostics(self.settings, None, self.outbox)

        # Start background tasks
        self.loop.create_task(self.load_model_background())
        self.loop.create_task(self._stage_ticker())

    # ---- UI helpers ----

    def _set_status(self, text: str) -> None:
        self.status_label.text = text

    def _ui_log(self, line: str, status: str | None) -> None:
        self.log_view.value = (self.log_view.value + line + "\n") if self.log_view.value else line + "\n"
        if status is not None:
            # Don't overwrite live stage ticker with log messages while a stage is active
            if self.current_stage is None:
                self._set_status(f"Now: {status}")

    def refresh_table(self) -> None:
        self.table.data = [
            {"file": item.path.name, "size": item.size_str, "status": item.status}
            for item in self.queue
        ]

    def _add_paths(self, paths: list[Path]) -> int:
        """Add paths to the queue, deduplicating. Returns count added."""
        added = 0
        for p in paths:
            if p not in self._queue_paths:
                self.queue.append(QueueItem(path=p))
                self._queue_paths.add(p)
                added += 1
        return added

    def _check_output_collision(self, item: QueueItem) -> bool:
        """Return True if all three output files already exist (skip this item)."""
        stem = item.path.stem
        return all(
            (self.outbox / f"{stem}{ext}").exists()
            for ext in (".txt", ".srt", ".json")
        )

    # ---- Stage ticker ----

    async def _stage_ticker(self) -> None:
        while True:
            await asyncio.sleep(2)
            self._update_status_ticker()

    def _update_status_ticker(self) -> None:
        if self.current_stage is None or self.stage_started is None:
            return
        elapsed = time.monotonic() - self.stage_started
        mm = int(elapsed) // 60
        ss = int(elapsed) % 60
        elapsed_str = f"{mm:02d}:{ss:02d}"
        if self._last_progress_pct is not None:
            self._set_status(
                f"Now: {self.current_stage} — {self._last_progress_pct:.0f}% ({elapsed_str} elapsed)"
            )
        else:
            self._set_status(f"Now: {self.current_stage} — {elapsed_str} elapsed")

    def _on_stage_start(self, stage: str) -> None:
        """Called from worker thread — must use call_soon_threadsafe."""
        def _set():
            self.current_stage = stage
            self.stage_started = time.monotonic()
            self._last_progress_pct = None
            self._update_status_ticker()
        self.loop.call_soon_threadsafe(_set)

    def _on_stage_end(self, stage: str) -> None:
        """Called from worker thread — must use call_soon_threadsafe."""
        def _clear():
            self.current_stage = None
            self.stage_started = None
            self._last_progress_pct = None
        self.loop.call_soon_threadsafe(_clear)

    def _on_progress(self, pct: float) -> None:
        """Called from worker thread with progress 0-100."""
        def _update():
            self._last_progress_pct = pct
            self._update_status_ticker()
        self.loop.call_soon_threadsafe(_update)

    # ---- Folder / file pickers ----

    async def _pick_folder(self, prompt: str, initial: Path) -> Path | None:
        try:
            result = await self.main_window.dialog(
                toga.SelectFolderDialog(prompt, initial_directory=initial)
            )
        except AttributeError:
            result = await self.main_window.select_folder_dialog(prompt)
        if result is None:
            return None
        return Path(result[0] if isinstance(result, list) else result)

    async def on_add_files(self, widget) -> None:
        try:
            result = await self.main_window.dialog(
                toga.OpenFileDialog(
                    "Add audio files",
                    multiple_select=True,
                    file_types=[ext.lstrip(".") for ext in AUDIO_EXTS],
                )
            )
        except Exception:
            log.exception("File dialog error")
            return
        if not result:
            return
        paths = [Path(p) for p in (result if isinstance(result, list) else [result])]
        added = self._add_paths(paths)
        if added:
            self.refresh_table()
            log.info("Added %d file(s) to queue (%d total)", added, len(self.queue))

    async def on_add_folder(self, widget) -> None:
        chosen = await self._pick_folder("Add folder", self.outbox.parent)
        if chosen is None:
            return
        paths = [p for p in chosen.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        paths.sort()
        added = self._add_paths(paths)
        if added:
            self.refresh_table()
            log.info("Added %d file(s) from %s (%d total)", added, chosen.name, len(self.queue))
        else:
            log.info("No new audio files found in %s", chosen)

    def on_remove_selected(self, widget) -> None:
        selected = self.table.selection
        if not selected:
            return
        if not isinstance(selected, list):
            selected = [selected]
        to_remove_names = set()
        for row in selected:
            for item in self.queue:
                if item.path.name == row.file and item.status != "Running":
                    to_remove_names.add(item.path)
        blocked = len(selected) - len(to_remove_names)
        if blocked:
            log.info("Cannot remove %d item(s) currently running.", blocked)
        if to_remove_names:
            self.queue = [i for i in self.queue if i.path not in to_remove_names]
            self._queue_paths -= to_remove_names
            self.refresh_table()

    def on_clear_completed(self, widget) -> None:
        done_statuses = {"Done", "Skipped", "Failed"}
        to_remove = {i.path for i in self.queue if i.status in done_statuses}
        if to_remove:
            self.queue = [i for i in self.queue if i.path not in to_remove]
            self._queue_paths -= to_remove
            self.refresh_table()

    def on_open_outputs(self, widget) -> None:
        subprocess.run(["open", str(self.outbox)], check=False)

    def on_open_logs(self, widget) -> None:
        subprocess.run(["open", str(LOG_DIR)], check=False)

    async def on_copy_diagnostics(self, widget) -> None:
        try:
            path = create_diagnostics_zip(SETTINGS_PATH)
        except Exception as e:
            log.exception("Failed to create diagnostics zip")
            await self.main_window.dialog(
                toga.ErrorDialog("Diagnostics failed", f"{e}\nSee logs.")
            )
            return
        subprocess.run(["open", "-R", str(path)], check=False)
        await self.main_window.dialog(
            toga.InfoDialog(
                "Diagnostics saved",
                f"Saved to {path}\nSettings file was included with HF token redacted.",
            )
        )

    def save_settings(self) -> None:
        self.settings["outbox"] = str(self.outbox)
        save_settings(self.settings)

    async def on_pick_outbox(self, widget) -> None:
        chosen = await self._pick_folder("Choose output folder", self.outbox)
        if chosen is None:
            return
        self.outbox = chosen
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.outbox_label.text = f"Output folder: {self.outbox}"
        log.info("Outbox changed to %s", self.outbox)
        self.save_settings()

    def on_stop(self, widget) -> None:
        self._stop_flag = True
        self.stop_button.enabled = False
        log.info("Stop requested — will finish current file then halt.")

    # ---- Model selector ----

    def _selected_model_id(self) -> str:
        display = self.model_select.value
        for label, mid in MODEL_OPTIONS:
            if label == display:
                return mid
        return MODEL_OPTIONS[0][1]

    async def on_model_changed(self, widget) -> None:
        if self.busy:
            return
        model_id = self._selected_model_id()
        if not model_id:
            return
        log.info("Switching model to %s...", model_id)
        self.go_button.enabled = False
        self.model_select.enabled = False

        # Drop old pipeline
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            gc.collect()

        # Persist new model choice
        self.settings["model"] = model_id
        self.save_settings()

        # Reload in background
        self.loop.create_task(self._reload_model_task(model_id))

    async def _reload_model_task(self, model_id: str) -> None:
        try:
            self.pipeline = await asyncio.to_thread(
                load_pipeline, self.settings["hf_token"], model_id, self.settings,
            )
            log.info("Model %s ready.", model_id)
        except Exception:
            log.exception("Model load failed for %s", model_id)
        finally:
            if not self.busy:
                self.go_button.enabled = True
            self.model_select.enabled = True

    # ---- Workflow ----

    async def load_model_background(self) -> None:
        default_id = MODEL_OPTIONS[0][1]
        model_id = self.settings.get("model", default_id)
        known_ids = {mid for _, mid in MODEL_OPTIONS}
        if model_id not in known_ids:
            model_id = default_id
        self.go_button.enabled = False
        self.model_select.enabled = False
        try:
            self.pipeline = await asyncio.to_thread(
                load_pipeline, self.settings["hf_token"], model_id, self.settings,
            )
            log.info("Model %s ready.", model_id)
        except Exception:
            log.exception("Model load failed")
        finally:
            if not self.busy:
                self.go_button.enabled = True
            self.model_select.enabled = True

    def selected_speakers(self) -> int | None:
        raw = self.speakers_input.value.strip()
        if not raw:
            return None
        try:
            n = int(raw)
            return n if n > 0 else None
        except ValueError:
            log.warning("Ignoring non-numeric speakers value: %r", raw)
            return None

    def _next_queued(self) -> QueueItem | None:
        """Return the first item with status Queued, or None.

        Failed items are only retried on a subsequent batch (see on_transcribe,
        which resets Failed -> Queued at batch start). Retrying within the same
        batch would loop forever on a persistent failure.
        """
        for item in self.queue:
            if item.status == "Queued":
                return item
        return None

    async def on_transcribe(self, widget) -> None:
        if self.busy:
            return
        if self.pipeline is None:
            log.info("Model still loading — wait a moment and try again.")
            return
        if not any(i.status in ("Queued", "Failed") for i in self.queue):
            log.info("No queued files to process.")
            return

        speakers = self.selected_speakers()

        self.busy = True
        self._stop_flag = False
        self.go_button.enabled = False
        self.stop_button.enabled = True
        self.model_select.enabled = False

        # Reset previously-Failed items so a fresh click retries them.
        for item in self.queue:
            if item.status == "Failed":
                item.status = "Queued"
        self.refresh_table()

        queued_files = [i.path for i in self.queue if i.status == "Queued"]
        log_batch_start(queued_files)
        log.info("Settings: speakers=%s model=%s",
                 speakers, self._selected_model_id())

        try:
            while not self._stop_flag:
                item = self._next_queued()
                if item is None:
                    break

                if self._check_output_collision(item):
                    item.status = "Skipped"
                    self.refresh_table()
                    log.warning("Skipped %s — outputs already exist in outbox", item.path.name)
                    continue

                item.status = "Running"
                self.refresh_table()
                log.info("Processing: %s", item.path.name)
                try:
                    await asyncio.to_thread(
                        transcribe_one,
                        item.path, self.outbox, self.pipeline,
                        None, speakers, speakers,
                        self.settings,
                        self._on_stage_start,
                        self._on_stage_end,
                        self._on_progress,
                    )
                    item.status = "Done"
                except Exception:
                    item.status = "Failed"
                    log.exception("FAILED %s", item.path.name)
                finally:
                    self.refresh_table()

            if self._stop_flag:
                log.info("Batch stopped by user. Remaining items stay Queued.")
            else:
                log.info("Batch complete. Outputs in %s", self.outbox)
        finally:
            self.busy = False
            self._stop_flag = False
            self.go_button.enabled = True
            self.stop_button.enabled = False
            self.model_select.enabled = True
            self.current_stage = None
            self.stage_started = None
            self._last_progress_pct = None
            self._set_status("Now: Ready.")


def main() -> TranscribeApp:
    return TranscribeApp("Local Transcribe", "com.rdobson.local_transcribe")


if __name__ == "__main__":
    main().main_loop()
