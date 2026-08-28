# widgets/progress_info.py
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QWidget

from utils.icons import format_hms


class ProgressInfoLabel(QLabel):
    """
    Label showing "Elapsed: HH:MM:SS | ETA: HH:MM:SS".

    Shared by the acquisition and processing panels; connect the ViewModel's
    elapsedChanged/etaChanged signals directly to set_elapsed/set_eta.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._elapsed_s = 0.0
        self._eta_s = 0.0
        self._refresh()

    @Slot(float)
    def set_elapsed(self, seconds: float) -> None:
        self._elapsed_s = seconds
        self._refresh()

    @Slot(float)
    def set_eta(self, seconds: float) -> None:
        self._eta_s = seconds
        self._refresh()

    def _refresh(self) -> None:
        self.setText(f"Elapsed: {format_hms(self._elapsed_s)} | ETA: {format_hms(self._eta_s)}")
