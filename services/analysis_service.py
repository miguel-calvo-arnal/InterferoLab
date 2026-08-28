# services/analysis_service.py
from __future__ import annotations

try:
    import analysis_backend
except ImportError:
    # Compiled binaries are not shipped in the repo: a fresh clone has no
    # .so/.pyd until the backend is built (see README, Getting started).
    # Degrade gracefully so the app can still start in acquisition mode.
    analysis_backend = None
from PySide6.QtCore import QObject, QThread, Signal, Slot

# ---------------------------------------------------------------------
# Fallback method list used when the C++ backend is not yet compiled.
# Mirrors the entries in methods.hpp — keep in sync when adding methods.
# ---------------------------------------------------------------------
FALLBACK_METHODS = [
    {
        "id": 1,
        "name": "Squared-derivative centroid",
        "description": "Centroid of (dG_i)^2 after Gaussian baseline removal.",
    },
    {
        "id": 2,
        "name": "FFT Fourier-filter centroid",
        "description": "Bandpass around k_avg + 2*|IFFT| envelope and weighted centroid.",
    },
]


def load_reconstruction_methods() -> list[dict]:
    """
    Try to query the C++ backend for the list of registered reconstruction methods.
    Falls back to FALLBACK_METHODS if the backend is unavailable.
    """
    if analysis_backend is None:
        return FALLBACK_METHODS
    try:
        return list(analysis_backend.get_reconstruction_methods())
    except Exception:
        return FALLBACK_METHODS


# ---------------------------------------------------------------------
# Worker executed inside a dedicated QThread
# ---------------------------------------------------------------------
class _AnalysisWorker(QObject):
    """
    Worker responsible for running `analysis_backend.run_analysis(...)`
    inside a background thread.

    This object is never accessed directly by the UI. It is always
    orchestrated by `AnalysisService`.

    Signals
    -------
    progress(percent, stage, done, total)
        Progress update emitted by the C++ backend.
    log(level, message)
        Text log messages for the UI.
    finished(result_dict)
        Emitted when the analysis completes successfully.
    error(message)
        Emitted when an unexpected exception occurs.
    """

    progress = Signal(int, str, int, int)  # percent, stage, done, total
    log = Signal(str, str)  # level, message
    finished = Signal(dict)  # C++ result dict (converted to Python dict)
    error = Signal(str)  # error message

    def __init__(
        self,
        dataset_folder: str,
        name: str = "",
        channel_weights: list[float] | None = None,
        method: int = 2,
        pixel_plot_grid: int = 0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_folder = dataset_folder
        self._name = name
        self._channel_weights = channel_weights
        self._method = method
        self._pixel_plot_grid = pixel_plot_grid

    @Slot()
    def run(self) -> None:
        """
        Entry point executed inside the worker thread.

        Calls `analysis_backend.run_analysis(...)` with Python callbacks.
        These callbacks are converted to Qt signals, which are queued
        automatically across threads.
        """
        try:
            # Callbacks invoked by the C++ backend in the worker thread
            def progress_cb(percent: int, stage: str, done: int, total: int) -> None:
                self.progress.emit(percent, stage, done, total)

            def log_cb(level: str, message: str) -> None:
                self.log.emit(level, message)

            # Blocking call into C++ backend
            result = analysis_backend.run_analysis(
                self._dataset_folder,
                self._name,
                progress_cb=progress_cb,
                log_cb=log_cb,
                channel_weights=self._channel_weights,
                method=self._method,
                pixel_plot_grid=self._pixel_plot_grid,
            )

            # pybind11 returns a dict-like object; convert explicitly
            result = dict(result)
            self.finished.emit(result)

        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


# ---------------------------------------------------------------------
# High-level Service (API used by the ViewModel / UI)
# ---------------------------------------------------------------------
class AnalysisService(QObject):
    """
    High-level service that launches the C++ analysis pipeline inside
    a background QThread.

    This wraps the worker lifecycle, forwards Qt signals to the UI layer
    (via AnalysisVM), and keeps track of the running state.

    Public API
    ----------
    start_analysis(dataset_folder, name="")
        Spawns a QThread + _AnalysisWorker, wires signals, and starts analysis.
    cancel()
        Requests cooperative cancellation: the C++ backend polls the flag
        and aborts between processing chunks (see analysis_backend.cancel_analysis).
    is_running()
        Whether an analysis job is currently active.

    Signals
    -------
    progressChanged(percent, stage, done, total)
        Forwarded from the C++ backend.
    logReceived(level, message)
        Forwarded textual log messages.
    finished(result_dict)
        Emitted on successful completion.
    error(message)
        Unexpected errors.
    runningChanged(bool)
        Indicates whether analysis is active.
    """

    progressChanged = Signal(int, str, int, int)
    logReceived = Signal(str, str)
    finished = Signal(dict)
    error = Signal(str)
    runningChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._thread: QThread | None = None
        self._worker: _AnalysisWorker | None = None
        self._running: bool = False

        # Graveyard for finished worker/thread wrappers: dropping the last
        # Python reference while deleteLater destroys the C++ object in the
        # dying thread races the GC and can SIGBUS (PySide6 6.11/Py 3.14).
        # Wrappers are retained here and trimmed at the next start.
        self._retired: list[tuple[QObject, QThread]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        """Return True if an analysis job is currently running."""
        return self._running

    def start_analysis(
        self,
        dataset_folder: str,
        name: str = "",
        channel_weights: list[float] | None = None,
        method: int = 2,
        pixel_plot_grid: int = 0,
    ) -> bool:
        """
        Launch the analysis in a background thread.

        Parameters
        ----------
        dataset_folder : str
            Folder containing the raw input images.
        name : str
            Optional base name; if empty, the C++ backend infers it.
        channel_weights : list[float] | None
            Normalized [wr, wg, wb] weights for RGB-to-grayscale conversion.
            None means use the C++ backend defaults (from config.hpp).

        Returns
        -------
        bool
            True if the analysis was launched; False if another instance
                is already running.
        """
        if self._running:
            return False  # prevent concurrent jobs

        if analysis_backend is None:
            self.error.emit(
                "The C++ analysis backend is not compiled "
                "(see README, Getting started)."
            )
            return False

        # Create worker thread
        self._thread = QThread(self)
        self._worker = _AnalysisWorker(
            dataset_folder, name, channel_weights, method, pixel_plot_grid
        )

        # Move worker into thread
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        # Forward worker signals
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Clean up when worker signals completion
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        # Mark as running
        self._running = True
        self.runningChanged.emit(True)

        # Start thread
        self._thread.start()
        return True

    def cancel(self) -> None:
        """
        Request cancellation of the C++ analysis.

        This triggers cooperative cancellation in the C++ code.
        """
        try:
            analysis_backend.cancel_analysis()
            self.logReceived.emit("info", "Analysis cancellation requested.")
        except Exception as e:
            self.logReceived.emit("error", f"Failed to request cancellation: {e}")

    def shutdown(self, timeout_ms: int = 15000) -> None:
        """
        Cancel any running analysis and stop the worker thread.

        Called on application close: a QThread must never be destroyed while
        still running (Qt aborts the process), so request cooperative
        cancellation — the C++ backend polls the flag between chunks — and
        wait for the thread with a bounded timeout.
        """
        thread = self._thread
        if thread is None:
            return
        if self._running:
            self.cancel()
        thread.quit()
        if not thread.wait(timeout_ms):
            # Last resort: better a forced stop than the guaranteed qFatal
            # from destroying a live QThread during window teardown.
            self.logReceived.emit(
                "error", "Analysis thread did not stop in time; terminating it."
            )
            thread.terminate()
            thread.wait(2000)

    # ------------------------------------------------------------------
    # Internal slots: forward worker signals to the external API
    # ------------------------------------------------------------------
    @Slot(int, str, int, int)
    def _on_progress(self, percent: int, stage: str, done: int, total: int) -> None:
        self.progressChanged.emit(percent, stage, done, total)

    @Slot(str, str)
    def _on_log(self, level: str, message: str) -> None:
        self.logReceived.emit(level, message)

    @Slot(dict)
    def _on_finished(self, result: dict) -> None:
        self.finished.emit(result)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.error.emit(message)

    @Slot()
    def _on_thread_finished(self) -> None:
        """
        Slot invoked when the worker QThread terminates.
        Resets service state and clears internal references.
        """
        self._running = False
        self.runningChanged.emit(False)
        if self._worker is not None and self._thread is not None:
            self._retired.append((self._worker, self._thread))
            self._retired = self._retired[-3:]
        self._thread = None
        self._worker = None
