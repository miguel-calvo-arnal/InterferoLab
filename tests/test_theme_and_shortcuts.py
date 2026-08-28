"""Tests for the visual theme, keyboard shortcuts, tooltips and icons.

Covers the MainWindow status bar and QShortcut set, tooltip coverage of the
interactive controls of the three panels, the widgets/theme.py palette, and
the U+03BC (Greek mu) regression across views/ and widgets/.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
)

import views.MainWindow as mainwindow_mod
from utils.icons import icon
from views.MainWindow import MainWindow
from widgets import theme

TOOLTIP_COVERAGE_THRESHOLD = 0.90

INTERACTIVE_TYPES = (
    QPushButton,
    QToolButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QLineEdit,
)


@pytest.fixture(autouse=True)
def _project_cwd(monkeypatch, project_root):
    """Run from the project root so icons and resource paths resolve."""
    monkeypatch.chdir(project_root)


@pytest.fixture
def main_window(qtbot, monkeypatch):
    """MainWindow isolated from the real app_config.json."""
    monkeypatch.setattr(mainwindow_mod, "load_config", lambda: {})
    monkeypatch.setattr(mainwindow_mod, "save_config", lambda cfg: None)
    win = MainWindow()
    qtbot.addWidget(win)
    return win


# ======================================================================
#  Status bar
# ======================================================================
def test_statusbar_shows_ready_message(main_window):
    """The status bar greets with the shortcut summary on startup."""
    msg = main_window.statusBar().currentMessage()
    assert msg.startswith("Ready")
    for token in ("F5", "Ctrl+Return", "Esc", "Ctrl+1/2/3"):
        assert token in msg


def test_statusbar_hover_label_updates(main_window):
    """The permanent hover label mirrors ResultsPanel.hoverInfoChanged."""
    assert main_window._status_hover.text() == ""
    main_window._results_panel.hoverInfoChanged.emit("Hover: x=1, y=2, z=3.0000 µm")
    assert main_window._status_hover.text() == "Hover: x=1, y=2, z=3.0000 µm"


# ======================================================================
#  Keyboard shortcuts
# ======================================================================
def test_shortcuts_exist(main_window):
    """F5, Ctrl+Return, Esc and Ctrl+1/2/3 are registered as QShortcut objects."""
    sequences = {sc.key().toString() for sc in main_window.findChildren(QShortcut)}
    expected = {"F5", "Ctrl+Return", "Esc", "Ctrl+1", "Ctrl+2", "Ctrl+3"}
    missing = expected - sequences
    assert not missing, f"Missing shortcuts: {sorted(missing)} (found: {sorted(sequences)})"


def test_tab_shortcut_count_matches_tabs(main_window):
    """There is one Ctrl+N tab shortcut per tab."""
    sequences = [sc.key().toString() for sc in main_window.findChildren(QShortcut)]
    tab_shortcuts = [s for s in sequences if s.startswith("Ctrl+") and s[-1].isdigit()]
    assert len(tab_shortcuts) == main_window._tabs.count()


# ======================================================================
#  Tooltip coverage
# ======================================================================
def _is_hidden_subtree(widget, root) -> bool:
    """True if *widget* or any ancestor below *root* is explicitly hidden."""
    w = widget
    while w is not None and w is not root:
        if w.isHidden():
            return True
        w = w.parentWidget()
    return False


def _interactive_controls(root):
    """Yield the user-facing interactive controls under *root*.

    Skips Qt-internal children (spinbox line edits, etc.) and widgets living in
    explicitly hidden subtrees (e.g. pyqtgraph's hidden ROI/menu/norm controls).
    """
    seen: set[int] = set()
    for widget_type in INTERACTIVE_TYPES:
        for w in root.findChildren(widget_type):
            if id(w) in seen:
                continue  # QToolButton is also a QAbstractButton subclass, etc.
            seen.add(id(w))
            if w.objectName().startswith("qt_"):
                continue
            if _is_hidden_subtree(w, root):
                continue
            parent = w.parent()
            if isinstance(parent, (QSpinBox, QDoubleSpinBox, QComboBox)):
                continue  # internal editor of a compound control
            yield w


def test_tooltip_coverage_above_threshold(qtbot, main_window):
    """More than 90% of the interactive controls in the three panels have tooltips."""
    qtbot.wait(30)  # let deferred singleShot callbacks hide pyqtgraph's internal buttons
    panels = [
        main_window._acq_panel,
        main_window._proc_panel,
        main_window._results_panel,
    ]
    controls: list = []
    for panel in panels:
        controls.extend(_interactive_controls(panel))
    assert controls, "No interactive controls found — selector is broken"

    missing = [
        f"{type(w).__name__}(objectName={w.objectName()!r}, text={getattr(w, 'text', str)()!r})"
        for w in controls
        if not w.toolTip().strip()
    ]
    coverage = 1.0 - len(missing) / len(controls)
    assert coverage > TOOLTIP_COVERAGE_THRESHOLD, (
        f"Tooltip coverage {coverage:.1%} ({len(controls) - len(missing)}/{len(controls)}); "
        f"missing tooltips on: {missing}"
    )


# ======================================================================
#  Theme palette and helpers
# ======================================================================
def test_theme_palette_key_colors():
    """widgets/theme.py exposes the documented palette values."""
    assert theme.ACCENT == "#1f77b4"
    assert theme.ACCENT == theme.MATPLOTLIB_BLUE  # by design
    assert theme.DANGER == "#c62828"
    assert theme.SUCCESS == "#2e7d32"
    assert theme.MATPLOTLIB_BG == "#f0f0f0"
    assert theme.MATPLOTLIB_RED == "#d62728"
    assert theme.MATPLOTLIB_GREEN == "#2ca02c"
    assert theme.TEXT_PRIMARY == "#1a1f24"
    assert theme.NEUTRAL == theme.TEXT_SECONDARY


def test_theme_brush_applies_alpha(qtbot):
    """theme.brush returns a QBrush with the requested colour and alpha."""
    b = theme.brush("#d62728", 40)
    color = b.color()
    assert color.alpha() == 40
    assert (color.red(), color.green(), color.blue()) == (0xD6, 0x27, 0x28)


def test_set_variant_sets_dynamic_property(qtbot):
    """theme.set_variant stores the QSS 'variant' dynamic property."""
    btn = QPushButton()
    qtbot.addWidget(btn)
    theme.set_variant(btn, "primary")
    assert btn.property("variant") == "primary"
    theme.set_variant(btn, "danger")
    assert btn.property("variant") == "danger"


# ======================================================================
#  Icons
# ======================================================================
def test_icon_loader_returns_non_null_icons(qtbot):
    """The SVG icon loader produces non-null icons with rendered pixmaps."""
    for name in ("play.svg", "stop.svg", "refresh.svg", "plug.svg"):
        ico = icon(name)
        assert not ico.isNull(), f"icon({name!r}) is null"
        assert not ico.pixmap(16, 16).isNull()


def test_main_buttons_have_icons(main_window):
    """The primary action buttons of the three panels carry non-null icons."""
    buttons = [
        main_window._acq_panel.btn_connect,
        main_window._acq_panel.btn_start,
        main_window._acq_panel.btn_cancel,
        main_window._proc_panel._start_btn,
        main_window._proc_panel._cancel_btn,
        main_window._results_panel._load_btn,
        main_window._results_panel._refresh_btn,
        main_window._results_panel._tilt_btn,
    ]
    for btn in buttons:
        assert not btn.icon().isNull(), f"Button {btn.text()!r} has a null icon"


# ======================================================================
#  Unicode regression: Greek mu must not reappear
# ======================================================================
def test_no_greek_mu_in_views_and_widgets(project_root):
    """No .py file under views/ or widgets/ contains U+03BC (Greek mu); µ (U+00B5) is the standard."""
    offenders = []
    for folder in ("views", "widgets"):
        for path in Path(project_root, folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "μ" in text:
                offenders.append(os.path.relpath(path, project_root))
    assert not offenders, f"U+03BC (Greek mu) reappeared in: {offenders}"
