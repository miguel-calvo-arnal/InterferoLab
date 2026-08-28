# viewmodels/analysis_vm.py
from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QObject, Signal, Slot

from services.analysis_service import AnalysisService


class AnalysisVM(QObject):
    """
    ViewModel for the processing/analysis panel.

    Wraps AnalysisService and exposes Qt‑friendly signals for the UI.

    Signals
    -------
    progressChanged(percent, stage, done, total, eta_s)
        Progress update with estimated remaining time (seconds).
    logReceived(level, message)
        Log messages from the analysis backend.
    finished(result_dict)
        Emitted when the analysis completes successfully.
    error(message)
        Emitted on unexpected errors.
    runningChanged(bool)
        Indicates whether an analysis job is currently active.
    etaChanged(float)
        ETA in seconds, emitted separately for convenience.
    elapsedChanged(float)
        Elapsed time in seconds since the analysis started.
    """

    # Signals exposed to the UI
    progressChanged = Signal(int, str, int, int, float)  # percent, stage, done, total, eta_s
    logReceived = Signal(str, str)
    finished = Signal(dict)
    error = Signal(str)
    runningChanged = Signal(bool)

    etaChanged = Signal(float)
    elapsedChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._svc = AnalysisService(self)
        self._timer = QElapsedTimer()

        self._elapsed_s: float = 0.0
        self._eta_s: float = 0.0
        self._last_percent: int = 0

        # Connect service signals
        self._svc.progressChanged.connect(self._on_progress)
        self._svc.logReceived.connect(self.logReceived)
        self._svc.finished.connect(self.finished)
        self._svc.error.connect(self.error)
        self._svc.runningChanged.connect(self._on_running_changed)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def is_running(self) -> bool:
        return self._svc.is_running()

    def start(
        self,
        dataset_folder: str,
        name: str = "",
        channel_weights: list[float] | None = None,
        method: int = 2,
        pixel_plot_grid: int = 0,
    ) -> bool:
        """
        Start analysis for the given dataset folder.

        channel_weights: normalized [wr, wg, wb] for RGB-to-grayscale conversion.
            None means use the C++ backend defaults (from config.hpp).
        method: 1 = squared-derivative centroid, 2 = FFT Fourier-filter centroid.
        pixel_plot_grid: 0 = save median pixel only; N > 0 = save N×N grid + median.
        """
        if self._svc.is_running():
            return False

        # Reset timer and internal counters
        self._timer.restart()
        self._elapsed_s = 0.0
        self._eta_s = 0.0
        self._last_percent = 0
        self.elapsedChanged.emit(0.0)
        self.etaChanged.emit(0.0)

        return self._svc.start_analysis(
            dataset_folder, name, channel_weights, method, pixel_plot_grid
        )

    def cancel(self) -> None:
        """Forward cancellation request to the service."""
        self._svc.cancel()

    def shutdown(self) -> None:
        """Cancel and stop the worker thread (application close)."""
        self._svc.shutdown()

    # --------------------------------------------------------------
    # Internal slots
    # --------------------------------------------------------------
    @Slot(int, str, int, int)
    def _on_progress(self, percent: int, stage: str, done: int, total: int) -> None:
        """
        Receives progress from AnalysisService (without ETA) and computes:

        - elapsed time (from QElapsedTimer)
        - ETA assuming approximately linear progress in percentage

        Then emits:
          - elapsedChanged
          - etaChanged
          - progressChanged(percent, stage, done, total, eta_s)
        """
        # Monotonicity guard: progress must never decrease
        percent = max(percent, self._last_percent)
        self._last_percent = percent

        # Elapsed time in seconds
        if self._timer.isValid():
            ms = self._timer.elapsed()
            self._elapsed_s = ms / 1000.0 if ms >= 0 else 0.0
        else:
            self._elapsed_s = 0.0

        # Estimate ETA (simple linear model on percent)
        if 0 < percent < 100:
            self._eta_s = self._elapsed_s * (100.0 - percent) / percent
        else:
            self._eta_s = 0.0

        # Emit separated signals
        self.elapsedChanged.emit(self._elapsed_s)
        self.etaChanged.emit(self._eta_s)

        # Emit combined progress with ETA for the UI
        self.progressChanged.emit(percent, stage, done, total, self._eta_s)

    @Slot(bool)
    def _on_running_changed(self, running: bool) -> None:
        """
        Handle running state changes from the service.

        When `running` becomes True, the timer is (re)started and
        elapsed/ETA are reset.
        """
        if running:
            self._timer.restart()
            self._elapsed_s = 0.0
            self._eta_s = 0.0
            self._last_percent = 0
            self.elapsedChanged.emit(0.0)
            self.etaChanged.emit(0.0)

        self.runningChanged.emit(running)
