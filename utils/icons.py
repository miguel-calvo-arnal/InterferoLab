import os
import sys

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

# Backward-compatible re-exports: the session log lives in utils.session_log.
from utils.session_log import append_to_log_file as append_to_log_file
from utils.session_log import setup_log_file as setup_log_file

# Theme mirror (keep in sync with widgets/theme.py; not imported to avoid
# pulling pyqtgraph into every consumer of this module).
_ICON_COLOR = "#1a1f24"  # primary text
_ICON_DISABLED_COLOR = "#8b949d"  # disabled text


def resource_path(relative_path: str) -> str:
    """
    Return the correct absolute path whether running
    frozen under PyInstaller or in development mode.
    """
    # sys._MEIPASS is the PyInstaller unpack dir when frozen
    base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.abspath(".")
    return os.path.join(base, relative_path)


def _render_svg(svg_text: str, color: str, size: int) -> QPixmap:
    """Rasterize *svg_text* at *size* px with currentColor replaced by *color*."""
    data = QByteArray(svg_text.replace("currentColor", color).encode("utf-8"))
    renderer = QSvgRenderer(data)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    return pm


def icon(name: str, color: str | None = None) -> QIcon:
    """
    Load an icon from resources/icons/.

    SVG icons use ``stroke="currentColor"``; Qt does not resolve that to the
    widget text colour on its own, so they are rasterized here with *color*
    (default: theme primary text). Pass ``color="#ffffff"`` for icons placed
    on filled primary/danger/success buttons. A themed disabled variant is
    embedded so greyed-out buttons keep a visible icon.
    """
    path = resource_path(os.path.join("resources", "icons", name))
    if not name.lower().endswith(".svg"):
        return QIcon(path)
    try:
        with open(path, encoding="utf-8") as f:
            svg_text = f.read()
        ico = QIcon()
        for size in (16, 20, 24, 32, 48):
            ico.addPixmap(_render_svg(svg_text, color or _ICON_COLOR, size), QIcon.Normal)
            ico.addPixmap(_render_svg(svg_text, _ICON_DISABLED_COLOR, size), QIcon.Disabled)
        return ico
    except Exception:
        return QIcon(path)


def format_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS, or '--:--:--' if unknown."""
    if seconds <= 0:
        return "--:--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
