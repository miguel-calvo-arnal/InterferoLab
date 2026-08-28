"""Shared pytest configuration for the InterferoLab test suite.

All tests run WITHOUT real hardware: camera/piezo access must be mocked.
Qt tests run offscreen (no display needed).
"""

from __future__ import annotations

import os
import sys

import pytest

# Project root importable regardless of where pytest is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "backend"), os.path.join(_ROOT, "backend", "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Qt must never require a display in CI/headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def project_root() -> str:
    """Absolute path to the repository root."""
    return _ROOT


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run the test with cwd set to an empty temporary directory.

    Useful for code that resolves ./output or config paths relative to cwd.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
