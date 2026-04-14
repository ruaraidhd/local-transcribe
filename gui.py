"""Toga GUI for local_transcribe."""
from __future__ import annotations

import asyncio
import logging
import subprocess
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
    transcribe_one,
)

LANGUAGES = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("Russian", "ru"),
    ("Arabic", "ar"),
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

        # Queue data model: list of QueueItem, paths used for dedup
        self.queue: list[QueueItem] = []
        self._queue_paths: set[Path] = set()

        root = toga.Box(style=Pack(direction=COLUMN, padding=12, flex=1))

        # Output folder row
        out_row = toga.Box(style=Pack(direction=ROW, padding_bottom=8))
        self.outbox_label = toga.Label(f"Output: {self.outbox}", style=Pack(flex=1, padding_top=6))
        out_row.add(self.outbox_label)
        out_row.add(toga.Button("Change...", on_press=self.on_pick_outbox))
        root.add(out_row)

        # Queue table
        self.table = toga.Table(
            headings=["File", "Size", "Status"],
            accessors=["file", "size", "status"],
            multiple_select=True,
            style=Pack(flex=1, padding_bottom=4),
        )
        root.add(self.table)

        # Queue management buttons
        queue_buttons = toga.Box(style=Pack(direction=ROW, padding_bottom=8))
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
        root.add(queue_buttons)

        # Controls row (speakers, language)
        controls = toga.Box(style=Pack(direction=ROW, padding_bottom=8))
        controls.add(toga.Label("Speakers:", style=Pack(padding_right=6, padding_top=6)))
        self.speakers_input = toga.TextInput(
            placeholder="auto", style=Pack(width=60, padding_right=16)
        )
        controls.add(self.speakers_input)
        controls.add(toga.Label("Language:", style=Pack(padding_right=6, padding_top=6)))
        self.language_select = toga.Selection(
            items=[label for label, _ in LANGUAGES], style=Pack(width=160)
        )
        controls.add(self.language_select)
        root.add(controls)

        # Action buttons
        buttons = toga.Box(style=Pack(direction=ROW, padding_bottom=8))
        self.go_button = toga.Button(
            "Transcribe all", on_press=self.on_transcribe,
            style=Pack(padding_right=8, flex=1),
        )
        buttons.add(self.go_button)
        self.stop_button = toga.Button(
            "Stop", on_press=self.on_stop,
            style=Pack(padding_right=8),
        )
        self.stop_button.enabled = False
        buttons.add(self.stop_button)
        buttons.add(toga.Button("Open outputs", on_press=self.on_open_outputs,
                                style=Pack(padding_right=8)))
        buttons.add(toga.Button("Open logs", on_press=self.on_open_logs,
                                style=Pack(padding_right=8)))
        buttons.add(toga.Button("Copy diagnostics", on_press=self.on_copy_diagnostics))
        root.add(buttons)

        self.status = toga.Label("Ready.", style=Pack(padding_bottom=6))
        root.add(self.status)

        self.log_view = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, height=200))
        root.add(self.log_view)

        self.main_window = toga.MainWindow(title=self.formal_name, size=(760, 720))
        self.main_window.content = root
        self.main_window.show()

        # Wire the logger into the UI now that widgets exist.
        ui_handler = UILogHandler(self)
        ui_handler.setLevel(logging.INFO)
        log.addHandler(ui_handler)

        log_startup_diagnostics(self.settings, None, self.outbox)
        self.loop.create_task(self.load_model_background())

    # ---- UI helpers ----

    def _ui_log(self, line: str, status: str | None) -> None:
        self.log_view.value = (self.log_view.value + line + "\n") if self.log_view.value else line + "\n"
        if status is not None:
            self.status.text = status

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
        # selection may be a single row or list
        if not isinstance(selected, list):
            selected = [selected]
        # Build set of file names to remove (only non-Running)
        to_remove_names = set()
        for row in selected:
            # Find matching queue item by file name
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
        lines = [
            f'hf_token = "{self.settings["hf_token"]}"',
            f'outbox = "{self.outbox}"',
        ]
        SETTINGS_PATH.write_text("\n".join(lines) + "\n")

    async def on_pick_outbox(self, widget) -> None:
        chosen = await self._pick_folder("Choose output folder", self.outbox)
        if chosen is None:
            return
        self.outbox = chosen
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.outbox_label.text = f"Output: {self.outbox}"
        log.info("Outbox changed to %s", self.outbox)
        self.save_settings()

    def on_stop(self, widget) -> None:
        self._stop_flag = True
        self.stop_button.enabled = False
        log.info("Stop requested — will finish current file then halt.")

    # ---- Workflow ----

    async def load_model_background(self) -> None:
        try:
            self.pipeline = await asyncio.to_thread(
                load_pipeline, self.settings["hf_token"], "large-v3", "cpu", "int8",
            )
        except Exception:
            log.exception("Model load failed")

    def selected_language(self) -> str | None:
        label = self.language_select.value
        for name, code in LANGUAGES:
            if name == label:
                return code
        return None

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
        """Return the first item with status Queued or Failed, or None."""
        for item in self.queue:
            if item.status in ("Queued", "Failed"):
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
        language = self.selected_language()

        self.busy = True
        self._stop_flag = False
        self.go_button.enabled = False
        self.stop_button.enabled = True

        queued_files = [i.path for i in self.queue if i.status in ("Queued", "Failed")]
        log_batch_start(queued_files)
        log.info("Settings: speakers=%s language=%s", speakers, language)

        try:
            while not self._stop_flag:
                item = self._next_queued()
                if item is None:
                    break

                # Check output collision before running
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
                        language, speakers, speakers,
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


def main() -> TranscribeApp:
    return TranscribeApp("Local Transcribe", "com.rdobson.local_transcribe")


if __name__ == "__main__":
    main().main_loop()
