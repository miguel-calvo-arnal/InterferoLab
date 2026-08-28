# services/acquisition_service.py
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from backend.acquisition.acquisition_controller import AcquisitionSession, CancelFlag


# ---------------------------------------------------------------------
# Worker executed inside a dedicated QThread (SWEEP)
# ---------------------------------------------------------------------
class _AcquisitionWorker(QObject):
    """
    Worker class responsible for running the acquisition sweep in a
    background thread.

    Signals
    -------
    progress(percent, z, idx, total, elapsed, eta)
        Emitted periodically during acquisition.
    log(level, msg)
        Log messages forwarded to the UI.
    preview(frame, z)
        Raw frame preview and associated z value.
    finished(output_folder, skipped_frames, total_frames)
        Emitted when run_sweep completes normally; skipped_frames > 0
        signals an incomplete dataset (total_frames were planned).
    error(msg)
        Emitted when an unexpected exception occurs.
    """

    progress = Signal(float, float, int, int, float, float)
    log = Signal(str, str)
    preview = Signal(object, float)
    finished = Signal(str, int, int)
    error = Signal(str)

    def __init__(
        self,
        session: AcquisitionSession,
        cfg: dict,
        cancel_flag: CancelFlag,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._cfg = cfg
        self._cancel_flag = cancel_flag

    @Slot()
    def run(self) -> None:
        """
        Entry point executed inside the worker thread.

        Wraps `AcquisitionSession.run_sweep()` and bridges its
        Python callbacks to Qt signals for thread-safe delivery.
        """
        try:

            def progress_cb(percent, z, idx, total, elapsed, eta):
                self.progress.emit(percent, z, idx, total, elapsed, eta)

            def log_cb(level: str, msg: str):
                self.log.emit(level, msg)

            def preview_cb(frame, z):
                self.preview.emit(frame, z)

            output_folder = self._session.run_sweep(
                self._cfg,
                progress_cb=progress_cb,
                log_cb=log_cb,
                preview_cb=preview_cb,
                cancel_flag=self._cancel_flag,
            )
            skipped = int(getattr(self._session, "last_sweep_skipped_frames", 0))
            total = int(getattr(self._session, "last_sweep_total_frames", 0))
            self.finished.emit(output_folder, skipped, total)

        except Exception as e:  # noqa: BLE001
            # run_sweep already logged the error via log_cb; emit error signal
            # so the service/VM can update UI state accordingly.
            self.log.emit("error", f"Acquisition worker caught: {e}")
            self.error.emit(str(e))


# ---------------------------------------------------------------------
# Worker for single-frame live preview (one-shot)
# ---------------------------------------------------------------------
class _PreviewWorker(QObject):
    """
    Worker to capture a single preview frame in a background thread.

    This avoids blocking the GUI while calling AcquisitionSession.capture_preview().
    """

    frameReady = Signal(object, float)
    log = Signal(str, str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        session: AcquisitionSession,
        timeout: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._timeout = timeout

    @Slot()
    def run(self) -> None:
        try:
            frame, z = self._session.capture_preview(
                timeout=self._timeout,
                log_cb=lambda lvl, msg: self.log.emit(lvl, msg),
            )
            self.frameReady.emit(frame, z)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------
# Worker for a single manual piezo move
# ---------------------------------------------------------------------
class _MoveWorker(QObject):
    """
    Worker to move the piezo to a target position in a background thread.

    Piezo movement is blocking (pitools.waitontarget can take several
    seconds). Running it in a QThread keeps the GUI fully responsive.

    Signals
    -------
    position(z)
        Real position reached (µm), emitted once when the move succeeds.
    """

    finished = Signal()
    position = Signal(float)
    error = Signal(str)
    log = Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        z: float,
        cancel_flag: CancelFlag | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._z = z
        self._cancel_flag = cancel_flag

    @Slot()
    def run(self) -> None:
        try:
            real_pos = self._session.move_to(
                self._z,
                log_cb=lambda lvl, msg: self.log.emit(lvl, msg),
                cancel_flag=self._cancel_flag,
            )
            self.position.emit(float(real_pos))
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------
# High-level service (API consumed by the ViewModel)
# ---------------------------------------------------------------------
class AcquisitionService(QObject):
    """
    Front-end acquisition service encapsulating:

      - a Python backend session (AcquisitionSession)
      - a QThread for background execution of sweeps
      - one-shot QThreads for live preview captures
      - a QThread for non-blocking manual piezo moves
      - worker/VM/UI signal forwarding

    Signals
    -------
    progressChanged(percent, z, idx, total, elapsed, eta)
        Acquisition progress for the UI.
    logReceived(level, msg)
        Log messages forwarded from the backend.
    previewFrame(frame, z)
        Preview frames (NumPy arrays) + current Z.
    runningChanged(bool)
        True when a sweep is active.
    movingChanged(bool)
        True when a manual piezo move is in progress.
    positionChanged(z)
        Real piezo position (µm) reached by a completed manual move.
    finished(output_folder, skipped_frames, total_frames)
        Emitted when acquisition terminates successfully.
    error(msg)
        Any unexpected error.
    """

    progressChanged = Signal(float, float, int, int, float, float)
    logReceived = Signal(str, str)
    previewFrame = Signal(object, float)
    runningChanged = Signal(bool)
    movingChanged = Signal(bool)
    positionChanged = Signal(float)
    finished = Signal(str, int, int)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._session = AcquisitionSession()
        self._cfg: dict = {}

        # Sweep worker/thread
        self._thread: QThread | None = None
        self._worker: _AcquisitionWorker | None = None
        self._cancel_flag: CancelFlag | None = None

        # One-shot preview worker/thread
        self._preview_thread: QThread | None = None
        self._preview_worker: _PreviewWorker | None = None

        # Manual move worker/thread
        self._move_thread: QThread | None = None
        self._move_worker: _MoveWorker | None = None
        self._move_cancel_flag: CancelFlag | None = None

        self._running = False

        # Graveyard for finished worker/thread wrappers. Dropping the last
        # Python reference while the C++ object is being destroyed via
        # deleteLater in the dying thread races the GC and can SIGBUS
        # (observed with PySide6 6.11 / Python 3.14). Wrappers are retained
        # here and trimmed at the next start, long after deletion completed.
        self._retired: list[tuple[QObject, QThread]] = []

    def _retire(self, worker: QObject | None, thread: QThread | None) -> None:
        """Keep worker/thread wrappers alive until their deleteLater has run."""
        if worker is not None and thread is not None:
            self._retired.append((worker, thread))

    def _trim_retired(self) -> None:
        """Drop wrappers retired before the previous run (safe point)."""
        self._retired = self._retired[-3:]

    # ------------------------------------------------------------------
    # Hardware connection
    # ------------------------------------------------------------------
    def connect_piezo(self, serial: str, dll: str) -> bool:
        """
        Connect to the piezo controller using the Python backend session.

        The axis and closed-loop mode come from the currently stored config.
        """
        axis = self._cfg.get("axis", "A")
        closed = bool(self._cfg.get("closed_loop", True))
        return self._session.connect_piezo(
            serial,
            dll,
            axis=axis,
            closed_loop=closed,
            log_cb=self._emit_log_direct,
        )

    def connect_camera(self) -> bool:
        """
        Connect to the Thorlabs camera via the backend session.

        If the camera is already connected, this will simply update the
        exposure time (see AcquisitionSession.connect_camera).
        """
        exposure = float(self._cfg.get("exposure", 0.07))
        return self._session.connect_camera(
            exposure_s=exposure,
            log_cb=self._emit_log_direct,
        )

    def disconnect_all(self) -> bool:
        """Disconnect both piezo and camera."""
        return self._session.disconnect_all(log_cb=self._emit_log_direct)

    # ------------------------------------------------------------------
    # Configuration and control
    # ------------------------------------------------------------------
    def apply_config(self, cfg: dict) -> None:
        """
        Store configuration parameters (dictionary) to be used on start().
        Merges with existing cfg to allow partial updates.

        Rejected (with a log warning) while a sweep is running: changing the
        configuration mid-sweep could corrupt the dataset being acquired.
        """
        if self._running:
            self._emit_log_direct("warn", "Configuration change rejected: a sweep is running.")
            return
        self._cfg.update(cfg)
        if "color_mode" in cfg:
            self._session.color_mode = cfg["color_mode"]

    def start(self) -> bool:
        """
        Launch acquisition in a dedicated QThread.

        Returns False (and logs a reason) if:
          - a sweep is already running
          - a manual move is still in progress
          - a preview capture is still in flight
          - no configuration has been set
        """
        self._trim_retired()
        if self._running:
            self._emit_log_direct("warn", "A sweep is already running.")
            return False

        if self._move_thread is not None:
            self._emit_log_direct("warn", "Cannot start sweep while a manual move is in progress.")
            return False

        if self._preview_thread is not None:
            self._emit_log_direct(
                "warn", "Cannot start sweep while a preview capture is in progress."
            )
            return False

        if not self._cfg:
            self._emit_log_direct("error", "No configuration set before start().")
            return False

        self._cancel_flag = CancelFlag()
        self._thread = QThread(self)
        # Pass a snapshot of the config so later apply_config() calls can
        # never mutate the dict the worker is reading.
        self._worker = _AcquisitionWorker(
            self._session,
            dict(self._cfg),
            self._cancel_flag,
        )

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        # Worker → service
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.preview.connect(self._on_preview)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Thread clean-up: worker signals → quit thread
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._running = True
        self.runningChanged.emit(True)

        self._thread.start()
        return True

    def cancel(self) -> None:
        """Request cooperative cancellation from the worker thread."""
        if self._cancel_flag is not None:
            self._cancel_flag.cancel()
            self._emit_log_direct("info", "Cancel request sent.")

    def is_running(self) -> bool:
        """Return whether a sweep is currently active."""
        return self._running

    def is_moving(self) -> bool:
        """Return whether a manual piezo move is currently in progress."""
        return self._move_thread is not None

    def is_previewing(self) -> bool:
        """Return whether a one-shot preview capture is currently in flight."""
        return self._preview_thread is not None

    def shutdown(self, timeout_ms: int = 12000) -> None:
        """
        Orderly shutdown for application exit.

        Cancels any running sweep and any manual move, stops the three
        worker threads (quit + bounded wait), then disconnects the
        hardware. Safe to call even when nothing is running.

        The manual-move cancel matters for the timeout budget: a healthy
        closed-loop move may legally block up to MOVE_TIMEOUT_S (30 s),
        longer than the wait below — cancelling makes _wait_on_target
        return within one poll interval instead.
        """
        self.cancel()
        if self._move_cancel_flag is not None:
            self._move_cancel_flag.cancel()

        for thread in (self._thread, self._preview_thread, self._move_thread):
            if thread is not None:
                thread.quit()
                if not thread.wait(timeout_ms):
                    self._emit_log_direct(
                        "warn",
                        f"A worker thread did not stop within {timeout_ms / 1000.0:.0f} s; "
                        "disconnecting hardware anyway.",
                    )

        self._running = False
        self._thread = None
        self._worker = None
        self._cancel_flag = None
        self._preview_thread = None
        self._preview_worker = None
        self._move_thread = None
        self._move_worker = None
        self._move_cancel_flag = None

        self.disconnect_all()

    # ------------------------------------------------------------------
    # Manual control / live preview
    # ------------------------------------------------------------------
    def move_to(self, z: float) -> None:
        """
        Perform a manual piezo move to position z in a background thread.

        The move is non-blocking: movingChanged(True) is emitted immediately
        and movingChanged(False) is emitted when the move completes.
        Ignores the request (with a log warning) if a sweep or another
        move is already in progress.
        """
        self._trim_retired()
        if self._running:
            self._emit_log_direct("warn", "Cannot move piezo manually while a sweep is running.")
            return

        if self._move_thread is not None:
            self._emit_log_direct("warn", "A move is already in progress.")
            return

        # Also block moves while a preview capture is running to keep
        # the piezo stable during image acquisition.
        if self._preview_thread is not None:
            self._emit_log_direct(
                "warn", "Cannot move piezo while a preview capture is in progress."
            )
            return

        self._move_thread = QThread(self)
        self._move_cancel_flag = CancelFlag()
        self._move_worker = _MoveWorker(self._session, z, self._move_cancel_flag)

        self._move_worker.moveToThread(self._move_thread)
        self._move_thread.started.connect(self._move_worker.run)

        self._move_worker.log.connect(self._on_log)
        self._move_worker.position.connect(self.positionChanged)
        self._move_worker.error.connect(self._on_error)

        self._move_worker.finished.connect(self._move_thread.quit)
        self._move_worker.finished.connect(self._move_worker.deleteLater)
        self._move_thread.finished.connect(self._on_move_thread_finished)
        self._move_thread.finished.connect(self._move_thread.deleteLater)

        self.movingChanged.emit(True)
        self._move_thread.start()

    def capture_preview(self) -> bool:
        """
        Capture a single preview frame using the current camera in a
        background thread.  Emits previewFrame(frame, z) on success.

        Returns True if the capture was started, False otherwise.
        """
        self._trim_retired()
        if self._running:
            return False

        # Only one preview at a time
        if self._preview_thread is not None:
            return False

        # Do not capture while a move is in progress
        if self._move_thread is not None:
            return False

        timeout = float(self._cfg.get("timeout", 5.0))

        self._preview_thread = QThread(self)
        self._preview_worker = _PreviewWorker(self._session, timeout)

        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)

        self._preview_worker.frameReady.connect(self._on_preview)
        self._preview_worker.log.connect(self._on_log)
        self._preview_worker.error.connect(self.error)

        self._preview_worker.finished.connect(self._preview_thread.quit)
        self._preview_worker.finished.connect(self._preview_worker.deleteLater)
        self._preview_thread.finished.connect(self._on_preview_thread_finished)
        self._preview_thread.finished.connect(self._preview_thread.deleteLater)

        self._preview_thread.start()
        return True

    # ------------------------------------------------------------------
    # Internal handlers: forward worker signals to the outside
    # ------------------------------------------------------------------
    def _emit_log_direct(self, level: str, msg: str) -> None:
        self.logReceived.emit(level, msg)

    @Slot(float, float, int, int, float, float)
    def _on_progress(
        self,
        percent: float,
        z: float,
        idx: int,
        total: int,
        elapsed: float,
        eta: float,
    ) -> None:
        self.progressChanged.emit(percent, z, idx, total, elapsed, eta)

    @Slot(str, str)
    def _on_log(self, level: str, msg: str) -> None:
        self.logReceived.emit(level, msg)

    @Slot(object, float)
    def _on_preview(self, frame, z: float) -> None:
        self.previewFrame.emit(frame, z)

    @Slot(str, int, int)
    def _on_finished(self, output_folder: str, skipped: int, total: int) -> None:
        self.finished.emit(output_folder, skipped, total)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.error.emit(msg)

    @Slot()
    def _on_thread_finished(self) -> None:
        """Reset service state after the sweep worker thread terminates."""
        self._running = False
        self.runningChanged.emit(False)
        self._retire(self._worker, self._thread)
        self._thread = None
        self._worker = None
        self._cancel_flag = None

    @Slot()
    def _on_preview_thread_finished(self) -> None:
        """Reset preview thread state after one-shot capture finishes."""
        self._retire(self._preview_worker, self._preview_thread)
        self._preview_thread = None
        self._preview_worker = None

    @Slot()
    def _on_move_thread_finished(self) -> None:
        """Reset move thread state and notify observers."""
        self._retire(self._move_worker, self._move_thread)
        self._move_thread = None
        self._move_worker = None
        self._move_cancel_flag = None
        self.movingChanged.emit(False)
