# utils/session_log.py
"""
Session-wide logging based on the standard `logging` module.

A single timestamped log file is created in <app dir>/logs/ per session
(same file-name format as the previous ad-hoc logger). All messages routed
through the `logging` machinery — including the per-panel GUI logs forwarded
via `append_to_log_file()` — end up in that file with a timestamp and level.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

# Logger used for messages forwarded from the GUI log boxes.
_session_logger = logging.getLogger("interferolab.session")

_configured = False


def app_base_dir() -> str:
    """
    Return the directory the application is anchored to (NOT the cwd).

    - Frozen (PyInstaller): the directory containing the executable.
    - Development: the project root (parent of the `utils` package).
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_log_file() -> None:
    """
    Configure session logging: a FileHandler writing to
    <app dir>/logs/interferolab_<timestamp>.log at INFO level, plus a
    stderr handler for warnings and errors.

    Must be called once at application startup. Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console output for warnings/errors (replaces the old stderr prints).
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    try:
        log_dir = os.path.join(app_base_dir(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(log_dir, f"interferolab_{timestamp}.log")
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as e:
        root.warning("Could not create session log file: %s", e)

    _configured = True


def append_to_log_file(msg: str) -> None:
    """
    Record a message forwarded from a GUI log box in the session log.

    Kept for backward compatibility with the previous ad-hoc logger API;
    the message is logged at INFO level (it already carries its own tag).
    """
    _session_logger.info(msg)
