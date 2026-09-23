# services/acquisition_service.py
from __future__ import annotations

import contextlib
import threading

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from backend.acquisition.acquisition_controller import AcquisitionSession, CancelFlag
from backend.acquisition.camera_owner import OwnerCallbacks, describe_error


# ---------------------------------------------------------------------
# Bridge: the camera thread reports through plain callbacks; these signals
# carry them into the GUI thread (queued connections)
# ---------------------------------------------------------------------
class _OwnerBridge(QObject):
    """Qt signals emitted from the camera thread and delivered to the GUI thread.

    Frames are NOT carried by a signal: the camera thread drops each one into
    a one-slot mailbox and only signals "a frame is available".  If the GUI
    is busy, later frames overwrite the mailbox and the single pending
    notification is reused, so the GUI always paints the newest frame and a
    queue of stale frames can never build up.
    """

    frameAvailable = Signal()
    previewState = Signal(bool)
    previewNotice = Signal(str)
    errorOccurred = Signal(str, str)  # source ("exposure"/"sweep"), message
    logged = Signal(str, str)
    sweepProgress = Signal(float, float, int, int, float, float)
    sweepAborted = Signal(str)
    sweepFinished = Signal(str, int, int)
    # from the connection thread (see AcquisitionService.connect_hardware)
    connectDone = Signal(bool)
    disconnectDone = Signal()


# ---------------------------------------------------------------------
# Worker for a single manual piezo move
# ---------------------------------------------------------------------
class _MoveWorker(QObject):
    """
    Worker to move the piezo to a target position in a background thread.

    Piezo movement is blocking (waiting for on-target can take several
    seconds). Running it in a QThread keeps the GUI fully responsive.  The
    piezo is never used by the camera thread outside a sweep, and moves are
    refused during a sweep, so this is the only thread on the piezo then.

    Signals
    -------
    position(z)
        Measured piezo position (µm; NaN = unreadable), read on this thread:
        live while a move takes longer than one poll, and once when it ends,
        also when it fails or is cancelled (the stage may have moved anyway).
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
            # The final position is reported through position_cb too.
            self._session.move_to(
                self._z,
                log_cb=lambda lvl, msg: self.log.emit(lvl, msg),
                cancel_flag=self._cancel_flag,
                position_cb=lambda z: self.position.emit(float(z)),
            )
        except Exception as e:  # noqa: BLE001
            self.error.emit(describe_error(e))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------
# High-level service (API consumed by the ViewModel)
# ---------------------------------------------------------------------
class AcquisitionService(QObject):
    """
    Front-end acquisition service encapsulating:

      - a Python backend session (AcquisitionSession)
      - the camera thread the session owns (live preview stream, snapshots,
        exposure changes, sweeps and camera close all run there)
      - a QThread for non-blocking manual piezo moves
      - worker/VM/UI signal forwarding

    Signals
    -------
    progressChanged(percent, z, idx, total, elapsed, eta)
        Acquisition progress for the UI.
    logReceived(level, msg)
        Log messages forwarded from the backend.
    previewFrame(frame, z)
        Preview frames (NumPy arrays) + the z they were labelled with.
    runningChanged(bool)
        True when a sweep is active.
    movingChanged(bool)
        True while manual piezo moves are in progress (including a queued one).
    previewingChanged(bool)
        True when the live preview stream is on; False when it stops, also
        because of a camera error.
    previewNotice(text)
        Non-blocking camera problem for the window to show (never a dialog):
        a preview failure being retried, the preview given up after repeated
        failures, or a failed exposure change.  "" means "cleared".
    positionChanged(z)
        Measured piezo position (µm; NaN = unknown): at connection, live
        during a manual move and when it ends (also after a failure).
    connectFinished(ok)
        connect_hardware() is over: both devices connected (True), or the
        attempt failed and whatever was opened has been released (False).
    disconnectFinished()
        disconnect_hardware() is over: both devices released.
    sweepAborted(summary)
        The sweep stopped early after repeated camera failures; emitted just
        before finished(), which still carries the (partial) folder.
    finished(output_folder, skipped_frames, total_frames)
        Emitted when acquisition terminates (also after an abort).
    error(msg)
        An error that needs the user's attention (failed sweep, failed move).
    """

    progressChanged = Signal(float, float, int, int, float, float)
    logReceived = Signal(str, str)
    previewFrame = Signal(object, float)
    runningChanged = Signal(bool)
    movingChanged = Signal(bool)
    previewingChanged = Signal(bool)
    previewNotice = Signal(str)
    positionChanged = Signal(float)
    sweepAborted = Signal(str)
    finished = Signal(str, int, int)
    error = Signal(str)
    connectFinished = Signal(bool)
    disconnectFinished = Signal()

    #: Bound on shutdown's wait for the move thread (a cancelled move returns
    #: within one poll period, so this is only reached if the DLL hangs).
    MOVE_THREAD_WAIT_MS = 12000

    #: On application close the piezo gets this long to report it is back at
    #: 0 (10 s on a normal disconnect): closing must not hang on a faulty
    #: controller; the servo is switched off and the link closed anyway.
    CLOSE_PARK_TIMEOUT_S = 2.0

    #: Bound on shutdown's wait for a connect/disconnect in progress: the
    #: camera close deadline plus a margin (the piezo wait is cancelled).
    CONNECTION_THREAD_WAIT_S = 10.0

    #: A connect/disconnect still running after this long is stuck in a
    #: driver call: the window is given back ("not answering") instead of
    #: showing "Disconnecting…" until the app is closed.  Above the longest
    #: legitimate case: a disconnect whose piezo never reports reaching 0
    #: (10 s) overlapping the camera close deadline (8 s).
    CONNECTION_STUCK_S = 20.0

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._session = AcquisitionSession()
        self._cfg: dict = {}

        # Sweep (runs on the camera thread)
        self._cancel_flag: CancelFlag | None = None
        self._running = False
        self._shutting_down = False

        # Manual move worker/thread + the latest target ordered meanwhile
        self._move_thread: QThread | None = None
        self._move_worker: _MoveWorker | None = None
        self._move_cancel_flag: CancelFlag | None = None
        self._pending_move: float | None = None

        # Connect / disconnect run on a Python daemon thread, never on the
        # GUI thread (findings C8, R9): a DLL call that never returns is left
        # behind at exit instead of freezing the window or making Qt abort.
        # One at a time; "" / "connecting" / "disconnecting".
        self._conn_thread: threading.Thread | None = None
        self._conn_state = ""
        self._conn_cancel: CancelFlag | None = None
        # The current connection thread was given up on (CONNECTION_STUCK_S):
        # while it lives no new connection starts (it may still be inside the
        # driver), and its late report is handled, not forwarded.
        self._conn_abandoned = False

        # Graveyard for finished move worker/thread wrappers. Dropping the
        # last Python reference while the C++ object is being destroyed via
        # deleteLater in the dying thread races the GC and can SIGBUS
        # (observed with PySide6 6.11 / Python 3.14). Wrappers are retained
        # here and trimmed at the next start, long after deletion completed.
        self._retired: list[tuple[QObject, QThread]] = []

        # Camera thread -> GUI thread plumbing
        self._bridge = _OwnerBridge(self)
        self._latest_lock = threading.Lock()
        self._latest: tuple[object, float] | None = None
        self._notify_pending = False
        queued = Qt.ConnectionType.QueuedConnection
        self._bridge.frameAvailable.connect(self._on_frame_available, queued)
        self._bridge.previewState.connect(self._on_preview_state, queued)
        self._bridge.previewNotice.connect(self.previewNotice, queued)
        self._bridge.errorOccurred.connect(self._on_owner_error, queued)
        self._bridge.logged.connect(self._on_log, queued)
        self._bridge.sweepProgress.connect(self._on_progress, queued)
        self._bridge.sweepAborted.connect(self.sweepAborted, queued)
        self._bridge.sweepFinished.connect(self._on_sweep_finished, queued)
        self._bridge.connectDone.connect(self._on_connect_done, queued)
        self._bridge.disconnectDone.connect(self._on_disconnect_done, queued)
        self._session.set_owner_callbacks(
            OwnerCallbacks(
                on_frame=self._owner_frame,
                on_preview_state=self._bridge.previewState.emit,
                on_error=self._bridge.errorOccurred.emit,
                on_log=self._bridge.logged.emit,
                on_sweep_progress=self._bridge.sweepProgress.emit,
                on_sweep_finished=self._bridge.sweepFinished.emit,
                on_preview_notice=self._bridge.previewNotice.emit,
                on_sweep_aborted=self._bridge.sweepAborted.emit,
            )
        )

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
        ok = self._session.connect_piezo(
            serial,
            dll,
            axis=axis,
            closed_loop=closed,
            log_cb=self._emit_log_direct,
        )
        if ok:
            # Read by connect_piezo itself (no extra query here): the window
            # starts from where the stage really is.
            self.positionChanged.emit(float(self._session.last_position))
        return ok

    def connect_camera(self) -> bool:
        """
        Connect to the Thorlabs camera via the backend session.

        If the camera is already connected, this only updates the exposure,
        applied by the camera thread between frames (the live preview is
        stopped and re-armed around it).  True means "accepted".
        """
        if self._conn_state and threading.current_thread() is not self._conn_thread:
            # "Apply camera settings" while the connection thread opens or
            # closes the camera: with no owner yet (or already gone) the
            # exposure would be set inline, from this thread, on a camera
            # another thread is using (review H1 of batch 4).
            self._emit_log_direct(
                "warn", f"Camera settings not applied: hardware is {self._conn_state}."
            )
            return False
        exposure = float(self._cfg.get("exposure", 0.07))
        return self._session.connect_camera(
            exposure_s=exposure,
            log_cb=self._emit_log_direct,
        )

    def disconnect_all(self) -> bool:
        """Disconnect both piezo and camera, blocking (the camera through its
        thread).  The window uses disconnect_hardware(), which does this off
        the GUI thread."""
        return self._session.disconnect_all(log_cb=self._emit_log_direct)

    # ------------------------------------------------------------------
    # Connect / disconnect off the GUI thread
    # ------------------------------------------------------------------
    def connection_busy(self) -> str:
        """ "connecting", "disconnecting" or "" (nothing in progress)."""
        return self._conn_state

    def connect_hardware(self, serial: str, dll: str) -> bool:
        """Connect piezo and camera on the connection thread (non-blocking).

        connectFinished(ok) follows.  If either device fails, whatever was
        opened is released on the same thread before reporting False.
        Refused (False) while another connect/disconnect is in progress.
        """
        return self._start_connection_thread("connecting", self._connect_worker, serial, dll)

    def disconnect_hardware(self) -> bool:
        """Release camera and piezo on the connection thread (non-blocking).

        disconnectFinished() follows.  Refused (False) while a connect or
        disconnect is in progress, during a sweep or during a manual move
        (they use the piezo: one thread at a time).
        """
        if self._running or self.is_moving():
            self._emit_log_direct("warn", "Cannot disconnect while a sweep or a move is running.")
            return False
        return self._start_connection_thread("disconnecting", self._disconnect_worker)

    def _start_connection_thread(self, state: str, target, *args) -> bool:
        if self._conn_state or self._shutting_down:
            self._emit_log_direct(
                "warn", f"Ignored: hardware is already {self._conn_state or 'closing'}."
            )
            return False
        if self._conn_thread is not None and self._conn_thread.is_alive():
            self._emit_log_direct(
                "warn",
                "The previous connection attempt is still stuck in a driver call. "
                "Power-cycle the camera and the piezo controller, then try again.",
            )
            return False
        self._conn_state = state
        self._conn_abandoned = False
        self._conn_cancel = CancelFlag()
        thread = threading.Thread(target=target, args=args, name=f"hardware-{state}", daemon=True)
        self._conn_thread = thread
        thread.start()
        QTimer.singleShot(
            int(self.CONNECTION_STUCK_S * 1000), self, lambda: self._on_connection_deadline(thread)
        )
        return True

    def _on_connection_deadline(self, thread: threading.Thread) -> None:
        """The connection thread is still running long after it should have
        finished: give the window back and leave the thread alone (it never
        gets a second thread next to it on the hardware; see
        _start_connection_thread)."""
        if thread is not self._conn_thread or not self._conn_state or not thread.is_alive():
            return  # finished meanwhile, or a newer attempt
        state = self._conn_state
        self._conn_state = ""
        self._conn_abandoned = True
        if self._conn_cancel is not None:
            self._conn_cancel.cancel()
        self._emit_log_direct(
            "warn",
            f"Hardware is not answering: {state} did not finish within "
            f"{self.CONNECTION_STUCK_S:.0f} s. Power-cycle the camera and the piezo controller.",
        )
        if self._shutting_down:
            return
        if state == "connecting":
            self.connectFinished.emit(False)
        else:
            self.disconnectFinished.emit()

    def _connect_worker(self, serial: str, dll: str) -> None:
        """Connection thread: both devices, released again if either fails."""
        ok = False
        try:
            ok_piezo = self.connect_piezo(serial, dll)
            # Not even opened when the piezo already failed: the attempt is
            # lost anyway, and the camera stays free for other programs.
            ok_camera = ok_piezo and self.connect_camera()
            ok = bool(ok_piezo and ok_camera)
            if not ok:
                self._session.disconnect_all(
                    log_cb=self._emit_log_direct, cancel_flag=self._conn_cancel
                )
        except Exception as e:  # noqa: BLE001 - reported, never lost
            self._emit_log_direct("error", f"Connection failed: {describe_error(e)}")
            with contextlib.suppress(Exception):  # best effort after a failure
                self._session.disconnect_all(
                    log_cb=self._emit_log_direct, cancel_flag=self._conn_cancel
                )
        finally:
            self._bridge.connectDone.emit(ok)

    def _disconnect_worker(self) -> None:
        try:
            self._session.disconnect_all(
                log_cb=self._emit_log_direct, cancel_flag=self._conn_cancel
            )
        except Exception as e:  # noqa: BLE001
            self._emit_log_direct("error", f"Disconnect failed: {describe_error(e)}")
        finally:
            self._bridge.disconnectDone.emit()

    @Slot(bool)
    def _on_connect_done(self, ok: bool) -> None:
        if self._late_report():
            if ok and not self._shutting_down:
                # The window already said "failed": release what the late
                # thread opened, so the next connect starts clean.
                self._emit_log_direct("info", "A late connection finished; releasing it.")
                self.disconnect_hardware()
            return
        self._conn_state = ""
        self._conn_thread = None
        if not self._shutting_down:
            self.connectFinished.emit(bool(ok))

    @Slot()
    def _on_disconnect_done(self) -> None:
        if self._late_report():
            return
        self._conn_state = ""
        self._conn_thread = None
        if not self._shutting_down:
            self.disconnectFinished.emit()

    def _late_report(self) -> bool:
        """True for the report of a thread given up on (already announced)."""
        if not self._conn_abandoned:
            return False
        self._conn_abandoned = False
        self._conn_thread = None
        self._emit_log_direct("info", "The stuck hardware connection thread has finished.")
        return True

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
        Launch the sweep on the camera thread.

        The live preview (if on) is stopped by that thread before the sweep
        and re-armed after it; a pending snapshot simply runs first.

        Returns False (and logs a reason) if:
          - a sweep is already running
          - a manual move is still in progress (the sweep drives the piezo)
          - no configuration has been set
        """
        self._trim_retired()
        if self._conn_state:
            self._emit_log_direct(
                "warn", f"Cannot start a sweep while {self._conn_state} hardware."
            )
            return False
        if self._running:
            self._emit_log_direct("warn", "A sweep is already running.")
            return False

        if self.is_moving():
            self._emit_log_direct("warn", "Cannot start sweep while a manual move is in progress.")
            return False

        if not self._cfg:
            self._emit_log_direct("error", "No configuration set before start().")
            return False

        self._cancel_flag = CancelFlag()
        self._running = True
        self.runningChanged.emit(True)
        # Pass a snapshot of the config so later apply_config() calls can
        # never mutate the dict the sweep is reading.
        self._session.camera_owner().run_sweep(dict(self._cfg), self._cancel_flag)
        return True

    def cancel(self) -> None:
        """Request cooperative cancellation of the running sweep."""
        if self._cancel_flag is not None:
            self._cancel_flag.cancel()
            self._emit_log_direct("info", "Cancel request sent.")

    def is_running(self) -> bool:
        """Return whether a sweep is currently active."""
        return self._running

    def is_moving(self) -> bool:
        """Return whether a manual piezo move is in progress or queued."""
        return self._move_thread is not None or self._pending_move is not None

    def is_previewing(self) -> bool:
        """Return whether the live preview is on or a snapshot is pending."""
        owner = self._session.camera_owner(create=False)
        return owner is not None and owner.is_preview_busy()

    def shutdown(self, timeout_ms: int = MOVE_THREAD_WAIT_MS) -> None:
        """
        Orderly shutdown for application exit.

        Cancels any running sweep and any manual move, waits (bounded) for
        the move thread and for a connect/disconnect in progress, then
        disconnects the hardware: the camera through its thread with a
        deadline (an unresponsive camera is abandoned and reported, never
        destroyed under a running thread) while the piezo is parked with a
        short deadline (CLOSE_PARK_TIMEOUT_S).  Worst case about 8 s (a hung
        camera); normally well under a second.  Safe to call even when
        nothing is running.
        """
        self._shutting_down = True
        self.cancel()
        self._pending_move = None
        if self._move_cancel_flag is not None:
            self._move_cancel_flag.cancel()

        thread = self._move_thread
        if thread is not None:
            thread.quit()
            if not thread.wait(timeout_ms):
                self._emit_log_direct(
                    "warn",
                    f"The piezo move thread did not stop within {timeout_ms / 1000.0:.0f} s; "
                    "disconnecting hardware anyway.",
                )

        self._running = False
        self._cancel_flag = None
        self._move_thread = None
        self._move_worker = None
        self._move_cancel_flag = None

        # A connect or disconnect in progress finishes first (its piezo wait
        # is cut short); both threads never use the hardware at once.
        conn = self._conn_thread
        if conn is not None and conn.is_alive():
            if self._conn_cancel is not None:
                self._conn_cancel.cancel()
            conn.join(self.CONNECTION_THREAD_WAIT_S)
            if conn.is_alive():
                # Stuck inside a driver call: never touch the hardware from a
                # second thread; the daemon thread is left behind at exit.
                self._emit_log_direct(
                    "warn",
                    "The hardware connection thread is not answering; the devices "
                    "were not released. Power-cycle them before the next session.",
                )
                return

        self._session.disconnect_all(
            log_cb=self._emit_log_direct, park_timeout_s=self.CLOSE_PARK_TIMEOUT_S
        )

    # ------------------------------------------------------------------
    # Manual control / live preview
    # ------------------------------------------------------------------
    def move_to(self, z: float) -> None:
        """
        Move the piezo to position z in a background thread, without waiting
        for the camera: the live preview keeps streaming meanwhile.

        Non-blocking: movingChanged(True) is emitted when the first move
        starts and movingChanged(False) when no move is left.  An order that
        arrives while a move is in progress replaces any earlier queued
        target (the LAST order wins; old steps are never replayed).
        Ignored (with a log warning) while a sweep is running.
        """
        self._trim_retired()
        if self._conn_state:
            self._emit_log_direct(
                "warn", f"Cannot move the piezo while {self._conn_state} hardware."
            )
            return
        if self._running:
            self._emit_log_direct("warn", "Cannot move piezo manually while a sweep is running.")
            return

        if self._move_thread is not None:
            self._pending_move = float(z)
            return

        self._start_move(float(z))

    def _start_move(self, z: float) -> None:
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

    def start_preview(self) -> bool:
        """Turn the live preview stream on (camera armed once, newest frame delivered).

        False while connecting or disconnecting: the camera owner must not be
        created while the connection thread is still opening or closing the
        camera (two threads in the SDK).
        """
        if self._conn_state:
            return False
        timeout = float(self._cfg.get("timeout", 5.0))
        self._session.camera_owner().preview_start(frame_timeout_s=timeout)
        return True

    def stop_preview(self) -> None:
        """Turn the live preview stream off (after the sweep in progress, if any)."""
        owner = self._session.camera_owner(create=False)
        if owner is not None:
            owner.preview_stop()

    def capture_preview(self) -> bool:
        """
        Ask for a single preview frame (the next stream frame, or a snap()
        when the stream is off).  Emits previewFrame(frame, z) when it lands.

        Returns True if the request was accepted, False during a sweep or
        while connecting/disconnecting (see start_preview).
        """
        if self._running or self._conn_state:
            return False
        timeout = float(self._cfg.get("timeout", 5.0))
        self._session.camera_owner().snapshot(frame_timeout_s=timeout)
        return True

    # ------------------------------------------------------------------
    # Camera thread -> GUI thread
    # ------------------------------------------------------------------
    def _owner_frame(self, frame, z: float) -> None:
        """Called on the camera thread: newest frame into the mailbox."""
        with self._latest_lock:
            self._latest = (frame, float(z))
            if self._notify_pending:
                return  # the GUI will pick this newer frame up with the pending notice
            self._notify_pending = True
        self._bridge.frameAvailable.emit()

    @Slot()
    def _on_frame_available(self) -> None:
        with self._latest_lock:
            latest = self._latest
            self._latest = None
            self._notify_pending = False
        if latest is not None:
            self.previewFrame.emit(*latest)

    @Slot(bool)
    def _on_preview_state(self, active: bool) -> None:
        self.previewingChanged.emit(bool(active))

    @Slot(str, str)
    def _on_owner_error(self, source: str, msg: str) -> None:
        if source != "sweep":
            # A failed exposure change: the preview goes on and every sweep
            # sets and verifies its own exposure, so a notice is enough.
            self.logReceived.emit("error", msg)
            if not self._shutting_down:
                self.previewNotice.emit(msg)
            return
        self._running = False
        self._cancel_flag = None
        self.runningChanged.emit(False)
        if not self._shutting_down:
            self.error.emit(msg)

    @Slot(str, int, int)
    def _on_sweep_finished(self, output_folder: str, skipped: int, total: int) -> None:
        self._running = False
        self._cancel_flag = None
        self.runningChanged.emit(False)
        if not self._shutting_down:
            self.finished.emit(output_folder, skipped, total)

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

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.error.emit(msg)

    @Slot()
    def _on_move_thread_finished(self) -> None:
        """Reset move thread state; start the latest queued target, if any."""
        self._retire(self._move_worker, self._move_thread)
        self._move_thread = None
        self._move_worker = None
        self._move_cancel_flag = None
        pending = self._pending_move
        self._pending_move = None
        if pending is not None and not self._shutting_down:
            self._start_move(pending)  # movingChanged stays True across the chain
            return
        self.movingChanged.emit(False)
