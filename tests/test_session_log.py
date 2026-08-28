"""Tests for utils/session_log.py — session file logging setup and append API."""

from __future__ import annotations

import logging
import re

import pytest

from utils import session_log

_FILENAME_RE = re.compile(r"^interferolab_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log$")
_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[INFO\] interferolab\.session: Hello GUI$"
)


@pytest.fixture
def fresh_session_log(tmp_path, monkeypatch):
    """Redirect the app dir to tmp_path, reset the setup flag and restore root handlers afterwards."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    monkeypatch.setattr(session_log, "app_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(session_log, "_configured", False)
    yield tmp_path
    for h in root.handlers[:]:
        if h not in saved_handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved_level)


def _log_files(tmp_path):
    log_dir = tmp_path / "logs"
    return sorted(log_dir.iterdir()) if log_dir.is_dir() else []


def test_setup_creates_timestamped_log_file(fresh_session_log):
    """setup_log_file creates logs/interferolab_<YYYY-MM-DD_HH-MM-SS>.log under the app dir."""
    session_log.setup_log_file()
    files = _log_files(fresh_session_log)
    assert len(files) == 1
    assert _FILENAME_RE.match(files[0].name), files[0].name


def test_append_writes_timestamp_level_and_message(fresh_session_log):
    """append_to_log_file writes 'YYYY-MM-DD HH:MM:SS [INFO] interferolab.session: <msg>' to the session file."""
    session_log.setup_log_file()
    session_log.append_to_log_file("Hello GUI")
    for h in logging.getLogger().handlers:
        h.flush()
    (log_file,) = _log_files(fresh_session_log)
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert _LINE_RE.match(lines[0]), lines[0]


def test_setup_is_idempotent(fresh_session_log):
    """A second setup_log_file call is a no-op: no extra file, no duplicate handlers."""
    session_log.setup_log_file()
    root = logging.getLogger()
    n_handlers = len(root.handlers)
    session_log.setup_log_file()
    assert len(root.handlers) == n_handlers
    assert len(_log_files(fresh_session_log)) == 1


def test_append_before_setup_does_not_raise_or_create_file(fresh_session_log):
    """Calling append_to_log_file before setup neither raises nor creates a log file (message goes to the unconfigured logging machinery)."""
    session_log.append_to_log_file("too early")
    assert _log_files(fresh_session_log) == []


def test_setup_survives_unwritable_app_dir(fresh_session_log, tmp_path, monkeypatch, capsys):
    """If the logs dir cannot be created, setup warns and continues instead of raising."""
    import os

    ro = tmp_path / "ro"
    ro.mkdir()
    monkeypatch.setattr(session_log, "app_base_dir", lambda: str(ro))
    os.chmod(ro, 0o500)
    try:
        session_log.setup_log_file()
    finally:
        os.chmod(ro, 0o700)
    assert not (ro / "logs").exists()
    assert "Could not create session log file" in capsys.readouterr().err


def test_app_base_dir_is_project_root_in_dev(project_root):
    """In development mode app_base_dir resolves to the project root (parent of utils/)."""
    assert session_log.app_base_dir() == project_root
