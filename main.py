import os
import sys

if getattr(sys, "frozen", False):  # Frozen
    base_path = sys._MEIPASS
else:  # Normal
    base_path = os.path.abspath(".")
    if sys.platform.startswith("win"):
        # The compiled backend (.pyd) links against the vcpkg DLLs. In
        # development they live where cmake leaves them (build\Release) or
        # inside a previously frozen bundle (dist\win). Register whichever
        # exists — add_dll_directory raises on a missing path, and a fresh
        # clone has neither.
        for _dll_dir in (
            os.path.join(base_path, "backend", "analysis", "build", "Release"),
            os.path.join(base_path, "dist", "win", "InterferoLab", "_internal"),
        ):
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)

sys.path.insert(0, os.path.join(base_path, "backend"))
sys.path.insert(0, os.path.join(base_path, "backend", "analysis"))

import logging

from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import QApplication

from utils.icons import icon, resource_path
from utils.session_log import setup_log_file
from views.MainWindow import MainWindow

log = logging.getLogger(__name__)


def main():
    # HiDPI: exact (non-rounded) scale factors; must be set before QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    # Neutral base style so the QSS renders identically on all platforms
    app.setStyle("Fusion")

    # Configure session logging (file in logs/ + stderr for warnings)
    setup_log_file()

    # Enforce decimal point "."
    QLocale.setDefault(QLocale(QLocale.C))

    # Add styles
    qss_path = resource_path(os.path.join("resources", "styles.qss"))
    try:
        with open(qss_path, encoding="utf-8") as f:
            qss = f.read()
        # Anchor relative url(resources/...) references (QSS resolves them
        # against the cwd, which is wrong for frozen builds)
        qss_base = resource_path("").replace(os.sep, "/").rstrip("/")
        qss = qss.replace("url(resources/", f"url({qss_base}/resources/")
        app.setStyleSheet(qss)
    except Exception as e:
        log.warning("Could not load stylesheet: %s", e)

    win = MainWindow()
    win.setWindowIcon(icon("app_icon.png"))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
