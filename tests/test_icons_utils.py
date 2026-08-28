"""Tests for utils/icons.py — resource paths, HH:MM:SS formatting and icon loading."""

from __future__ import annotations

import os

import pytest

from utils.icons import format_hms, icon, resource_path


@pytest.fixture
def in_project_root(project_root, monkeypatch):
    """Run with cwd at the project root (resource_path resolves against cwd in dev mode)."""
    monkeypatch.chdir(project_root)
    return project_root


def test_resource_path_is_absolute_and_exists(in_project_root):
    """resource_path returns an absolute path to a real bundled resource (resources/styles.qss)."""
    path = resource_path(os.path.join("resources", "styles.qss"))
    assert os.path.isabs(path)
    assert os.path.isfile(path)


def test_resource_path_icons_dir(in_project_root):
    """resource_path resolves the icons directory and a real SVG inside it."""
    assert os.path.isdir(resource_path(os.path.join("resources", "icons")))
    assert os.path.isfile(resource_path(os.path.join("resources", "icons", "check.svg")))


def test_format_hms_zero_and_negative_are_unknown():
    """Durations <= 0 render as the unknown placeholder '--:--:--'."""
    assert format_hms(0) == "--:--:--"
    assert format_hms(-5) == "--:--:--"
    assert format_hms(-0.1) == "--:--:--"


def test_format_hms_formats_hours_minutes_seconds():
    """Positive durations are rendered as zero-padded HH:MM:SS."""
    assert format_hms(3661) == "01:01:01"
    assert format_hms(1) == "00:00:01"
    assert format_hms(59) == "00:00:59"
    assert format_hms(3600) == "01:00:00"
    assert format_hms(86399) == "23:59:59"


def test_format_hms_truncates_fractional_seconds():
    """Fractional seconds are truncated, not rounded (0.5 s -> 00:00:00)."""
    assert format_hms(0.5) == "00:00:00"
    assert format_hms(61.9) == "00:01:01"


def test_format_hms_hours_beyond_a_day():
    """Durations over 24 h keep counting hours instead of wrapping."""
    assert format_hms(90000) == "25:00:00"


def test_icon_existing_svg_is_not_null(qapp, in_project_root):
    """icon() returns a non-null QIcon with renderable pixmaps for a real SVG icon."""
    ico = icon("check.svg")
    assert not ico.isNull()
    assert not ico.pixmap(16).isNull()
    assert not ico.pixmap(32).isNull()


def test_icon_existing_svg_has_disabled_variant(qapp, in_project_root):
    """icon() embeds a themed disabled variant so greyed-out buttons keep an icon."""
    from PySide6.QtGui import QIcon

    ico = icon("check.svg")
    assert not ico.pixmap(16, QIcon.Disabled).isNull()


def test_icon_missing_name_is_null(qapp, in_project_root):
    """icon() falls back to a null QIcon (no pixmaps) for a nonexistent icon name."""
    for name in ("no_such_icon.svg", "no_such_icon.png"):
        ico = icon(name)
        assert ico.isNull()
        assert ico.pixmap(16).isNull()
