# widgets/theme.py
"""
Single source of truth for the application colour palette.

Two families of colours live here:

- The matplotlib-inspired plot identity (``MATPLOTLIB_*``) used by the
  pyqtgraph views. The Qt theme (resources/styles.qss) frames the plots;
  it never replaces their look.
- The semantic accents of the Qt theme (``ACCENT``, ``DANGER``, ``SUCCESS``,
  text tones) for the few places where Python code needs a theme colour
  (pyqtgraph items are not reachable from QSS).

Keep these values in sync with resources/styles.qss (the original design
rationale lives in the July 2026 audit records, kept outside the repo).
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

# ---------------------------------------------------------------------
# Matplotlib plot identity (pyqtgraph backgrounds and curves)
# ---------------------------------------------------------------------
MATPLOTLIB_BG = "#f0f0f0"
MATPLOTLIB_BLUE = "#1f77b4"
MATPLOTLIB_RED = "#d62728"
MATPLOTLIB_GREEN = "#2ca02c"

# ---------------------------------------------------------------------
# Qt theme accents (mirror of styles.qss)
# ---------------------------------------------------------------------
ACCENT = "#1f77b4"  # same blue as MATPLOTLIB_BLUE, by design
DANGER = "#c62828"
SUCCESS = "#2e7d32"
WARNING_TEXT = "#7a5200"
TEXT_PRIMARY = "#1a1f24"
TEXT_SECONDARY = "#5b6672"
TEXT_DISABLED = "#8b949d"

# Neutral curve colour for single-channel (mono) data in plots.
NEUTRAL = TEXT_SECONDARY


def brush(color: str, alpha: int) -> pg.mkBrush:
    """Return a pyqtgraph brush for *color* (hex string) with 0-255 *alpha*."""
    c = QColor(color)
    c.setAlpha(alpha)
    return pg.mkBrush(c)


def set_variant(widget: QWidget, variant: str | None) -> None:
    """
    Set the dynamic ``variant`` property ("primary" / "danger" / "success")
    used by the QSS button rules, repolishing so a runtime change is applied.
    """
    if widget.property("variant") == variant:
        return
    widget.setProperty("variant", variant)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
