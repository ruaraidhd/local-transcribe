"""Logging, diagnostics, and crash capture for local_transcribe."""
from __future__ import annotations

import logging
import logging.handlers
import platform
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

LOG_DIR = Path.home() / "Library" / "Logs" / "LocalTranscribe"
LOG_FILE = LOG_DIR / "transcribe.log"
LOGGER_NAME = "local_transcribe"

_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FMT)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(_FMT)
    logger.addHandler(sh)

    _install_excepthook(logger)
    return logger


def _install_excepthook(logger: logging.Logger) -> None:
    def handle(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = handle


def install_asyncio_exception_handler(loop) -> None:
    logger = logging.getLogger(LOGGER_NAME)

    def handler(loop, context):
        exc = context.get("exception")
        msg = context.get("message", "asyncio error")
        if exc:
            logger.error("Asyncio: %s", msg, exc_info=exc)
        else:
            logger.error("Asyncio: %s | %s", msg, context)
    loop.set_exception_handler(handler)


def log_startup_diagnostics(settings: dict, inbox: Path, outbox: Path) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("===== Session start =====")
    logger.info("macOS:      %s (%s)", platform.mac_ver()[0] or "?", platform.machine())
    logger.info("Python:     %s", sys.version.split()[0])
    for pkg in ("torch", "parakeet_mlx", "mlx", "pyannote.audio", "toga"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            logger.info("%-11s %s", pkg + ":", ver)
        except Exception as e:
            logger.info("%-11s (not importable: %s)", pkg + ":", e)
    try:
        free_gb = shutil.disk_usage(Path.home()).free / (1024 ** 3)
        logger.info("Free disk:  %.1f GB", free_gb)
    except Exception as e:
        logger.debug("disk usage check failed: %s", e)
    logger.info("HF token:   %s", "set" if settings.get("hf_token") else "MISSING")
    logger.info("Inbox:      %s", inbox)
    logger.info("Outbox:     %s", outbox)


def log_batch_start(files: list[Path]) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Batch: %d file(s)", len(files))
    for p in files:
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            logger.info("  %s (%.1f MB)", p.name, size_mb)
        except Exception:
            logger.info("  %s (size unknown)", p.name)


def create_diagnostics_zip(settings_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path.home() / "Desktop" / f"local-transcribe-diagnostics-{stamp}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(LOG_DIR.glob("transcribe.log*")):
            z.write(p, arcname=f"logs/{p.name}")
        if settings_path.exists():
            z.writestr("settings.redacted.toml", _redact_settings(settings_path.read_text()))
    return out


def _redact_settings(text: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("hf_token") and "=" in line:
            key = line.split("=", 1)[0]
            out.append(f'{key.rstrip()} = "<redacted>"')
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def hint_for_pyannote_error(err: Exception) -> str | None:
    msg = str(err).lower()
    gated = any(tok in msg for tok in ("401", "403", "gated", "agree", "restricted", "unauthor"))
    if gated:
        return (
            "Diarisation model access denied. The HuggingFace account whose token is in "
            "settings.toml must accept the terms on BOTH of these pages (while logged in):\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0"
        )
    return None
