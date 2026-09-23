# backend/acquisition/camera_owner.py
"""Single owner of the camera: one thread, one request queue.

Why a thread that *owns* the camera
-----------------------------------
pylablib puts no lock around the Thorlabs SDK calls, so two threads inside
the camera object at the same time (a capture in one, ``close()`` or
``set_exposure()`` in another) is undefined behaviour on the real hardware
(findings C1, C1b, C5 and C9 of the phase-2 review).  Rather than guarding
every call site with flags, the camera is used from exactly ONE thread once
it is connected: this one.  Everything else (GUI, sweep start, exposure
change, disconnect, shutdown) *asks* the owner through the request queue
and never touches the camera object.

What runs on this thread
------------------------
* the live preview as a continuous stream: the camera is armed once and the
  newest frame is read in a loop.  ``snap()`` is never used per frame: it
  arms the camera, sleeps 0.25 s inside pylablib and disarms it every time
  (finding R1: 0.5 s and ~2 GB of buffer per preview frame);
* one-shot snapshots (``snap()``) while the stream is off;
* exposure changes, with the stream stopped around them and re-armed after
  (the SDK behaviour of ``set_exposure`` on an armed camera is unknown);
* the Z sweep (``AcquisitionSession.run_sweep``, unchanged), with the stream
  stopped before it and re-armed after it if the preview was on;
* closing the camera.

Delivery to the GUI is "newest frame wins": the owner hands every frame to
``on_frame`` without waiting for anybody; the receiver keeps only the latest
one (see ``AcquisitionService``), so a slow GUI drops intermediate frames
instead of building a queue.

Why a Python daemon thread and not a QThread
--------------------------------------------
If an SDK call never returns, whoever is waiting for the owner gives up
after a deadline, reports it and *abandons* this thread (``abandon()``): the
owner then exits as soon as the stuck call returns, without touching the
camera again.  A daemon thread stuck in a DLL is simply left behind when the
process exits, whereas Qt aborts the whole process when a running QThread
is destroyed (finding C2).
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def describe_error(e: BaseException) -> str:
    """Text of an exception for the user, never empty.

    pylablib's camera timeout is raised without a message (``str(e) == ""``,
    finding C7), which used to reach the user as an empty dialog.  Fall back
    to the exception's class name, which at least says what happened.
    """
    text = str(e).strip()
    if text:
        return text
    name = type(e).__name__
    if "timeout" in name.lower():
        return f"the camera did not deliver a frame in time ({name})"
    return f"{name} (the driver gave no details)"


class OwnerCallbacks:
    """Callbacks the owner thread invokes (from ITS thread) to report events.

    on_frame(frame, z)                  a preview or sweep frame (merged) + z (µm)
    on_preview_state(active)            live preview started (True) / stopped (False)
    on_preview_notice(text)             non-blocking live-preview problem: retrying or
                                        given up (text never empty); "" once it recovered
                                        (a failed exposure goes through on_error; its
                                        "" comes when an exposure is applied again)
    on_error(source, message)           source is "exposure" or "sweep"
    on_log(level, message)              log lines
    on_sweep_progress(percent, z, idx, total, elapsed, eta)
    on_sweep_aborted(reason)            the sweep stopped early on camera failures
                                        (sent just before on_sweep_finished)
    on_sweep_finished(folder, skipped, total)
    """

    def __init__(
        self,
        on_frame: Callable[..., None] | None = None,
        on_preview_state: Callable[[bool], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
        on_sweep_progress: Callable[..., None] | None = None,
        on_sweep_finished: Callable[[str, int, int], None] | None = None,
        on_preview_notice: Callable[[str], None] | None = None,
        on_sweep_aborted: Callable[[str], None] | None = None,
    ) -> None:
        self.on_frame = on_frame or _noop
        self.on_preview_state = on_preview_state or _noop
        self.on_error = on_error or _noop
        self.on_log = on_log or _noop
        self.on_sweep_progress = on_sweep_progress or _noop
        self.on_sweep_finished = on_sweep_finished or _noop
        self.on_preview_notice = on_preview_notice or _noop
        self.on_sweep_aborted = on_sweep_aborted or _noop


class _Abandoned(Exception):
    """Raised inside the owner thread once the outside world gave up on it."""


class CameraOwner(threading.Thread):
    """The only thread that uses the camera after it has been connected.

    The public methods are non-blocking and may be called from any thread;
    they record the *wanted* state and wake the thread, which reconciles the
    camera with it.  Reconciling (instead of executing each order in turn)
    means that the last order always wins and that an order can never be
    applied out of date.
    """

    #: Upper bound of one wait for a frame inside the stream loop: it is the
    #: reaction time to a request (stop, exposure, close) while no frame comes.
    STREAM_POLL_S: float = 0.25

    #: Ring-buffer size asked from pylablib for the stream.  It reads the newest
    #: frame only, so a few frames are enough; the SDK rounds it up to 10
    #: (about 250 MB) instead of the 85 frames (~2.1 GB) ``snap()`` allocates.
    STREAM_RING_FRAMES: int = 4

    #: Consecutive live-preview failures (arming failed, SDK error while
    #: streaming, no frame within the timeout, snapshot failed) after which
    #: the preview gives up and stops.  Each failure before that is reported
    #: as a non-blocking notice and retried; a delivered frame resets it.
    PREVIEW_MAX_FAILURES: int = 3

    #: Pause before retrying after a preview failure, so a transient USB/SDK
    #: hiccup can clear.  Waited on the request queue: a stop, a sweep or a
    #: close is still served at once during the pause.
    PREVIEW_RETRY_DELAY_S: float = 1.0

    def __init__(self, session: Any, callbacks: OwnerCallbacks | None = None) -> None:
        super().__init__(name="camera-owner", daemon=True)
        self._session = session
        self._cb = callbacks or OwnerCallbacks()
        self._requests: queue.SimpleQueue = queue.SimpleQueue()

        # Wanted state: written from any thread, read by the owner.  Plain
        # bool/float assignments are atomic under the GIL.
        self._stream_wanted = False
        self._snapshot_pending = False
        self._frame_timeout_s = 5.0
        self._abandoned = False

        # Actual state: owner thread only.
        self._stream_on = False
        self._sweep_active = False
        self._sweep_cancel: Any = None  # cancel flag of the sweep in progress
        self._last_frame_t = 0.0
        self._exposure_failed = False  # its notice is on screen until one succeeds
        self._preview_failures = 0  # consecutive, reset by a delivered frame
        self._retrying = False  # a failed preview/snapshot waits to be retried
        self._retry_at = 0.0  # monotonic time of that retry

        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # API (any thread, non-blocking unless stated)
    # ------------------------------------------------------------------
    def preview_start(self, frame_timeout_s: float = 5.0) -> None:
        """Keep the camera armed and deliver every new frame until preview_stop()."""
        self._frame_timeout_s = float(frame_timeout_s)
        self._stream_wanted = True
        self._post("wake")

    def preview_stop(self) -> None:
        """Disarm the camera (after the sweep in progress, if any)."""
        self._stream_wanted = False
        self._post("wake")

    def snapshot(self, frame_timeout_s: float = 5.0) -> None:
        """Deliver one frame: the next stream frame, or a snap() if the stream is off.

        Repeated requests before the frame arrives are coalesced into one.
        """
        self._frame_timeout_s = float(frame_timeout_s)
        self._snapshot_pending = True
        self._post("wake")

    def set_exposure(self, exposure_s: float) -> None:
        """Apply a new exposure between frames (stream stopped and re-armed around it)."""
        self._post("exposure", float(exposure_s))

    def run_sweep(self, cfg: dict, cancel_flag: Any) -> None:
        """Run AcquisitionSession.run_sweep on this thread, stream paused meanwhile."""
        self._post("sweep", (cfg, cancel_flag))

    def close(self, timeout_s: float) -> bool:
        """Close the camera and end the thread; wait up to timeout_s for it.

        Returns False when the owner did not finish in time (a camera call
        is not returning): the caller must then abandon() it and never touch
        the camera again.
        """
        self.request_close()
        return self.wait_closed(timeout_s)

    def request_close(self) -> None:
        """Ask for the close without waiting (see close()): lets the caller
        release the piezo meanwhile and wait for both at once."""
        self._stream_wanted = False
        self._snapshot_pending = False
        self._post("close")

    def wait_closed(self, timeout_s: float) -> bool:
        """Wait up to timeout_s for a requested close; False if it is not done."""
        return self._closed.wait(max(0.0, timeout_s))

    def abandon(self) -> None:
        """Give up on this thread: it exits as soon as its stuck call returns.

        From this point it neither touches the camera nor reports anything.
        A sweep in progress is cancelled too: its loop would otherwise keep
        moving the piezo (even one reconnected later) once the stuck snap()
        returns (review finding H2).
        """
        self._abandoned = True
        cancel = self._sweep_cancel
        if cancel is not None:
            cancel.cancel()

    def is_streaming(self) -> bool:
        """True while the live preview is requested."""
        return self._stream_wanted

    def is_preview_busy(self) -> bool:
        """True while the live preview is wanted OR the camera is still armed
        for it (a stop takes ~0.25 s to complete), or a snapshot is pending."""
        return self._stream_wanted or self._stream_on or self._snapshot_pending

    def is_sweeping(self) -> bool:
        return self._sweep_active

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def _post(self, kind: str, payload: Any = None) -> None:
        self._requests.put((kind, payload))

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one session primitive; bail out if we were abandoned meanwhile.

        Every camera access goes through here so that a thread abandoned
        while stuck in the SDK cannot touch a camera reconnected later.  The
        check runs BEFORE the call and in a ``finally``: a stuck call that
        finally returns with an exception (the usual outcome after the
        camera is power-cycled) must not be followed by a stop/retry on the
        new camera (review finding H1).
        """
        if self._abandoned:
            raise _Abandoned
        try:
            return fn(*args, **kwargs)
        finally:
            if self._abandoned:
                raise _Abandoned

    def _report(self, name: str, *args: Any) -> None:
        if self._abandoned:
            return
        getattr(self._cb, name)(*args)

    def run(self) -> None:  # noqa: C901 - one flat loop is easier to audit than several
        try:
            while True:
                if self._abandoned:
                    break
                self._reconcile()
                try:
                    kind, payload = self._next_request()
                except queue.Empty:
                    kind, payload = None, None
                if kind == "close":
                    self._do_close()
                    break
                if kind == "exposure":
                    self._do_exposure(payload)
                elif kind == "sweep":
                    self._do_sweep(*payload)
                elif kind is None and self._stream_on:
                    self._pump()
                # "wake" carries no action: _reconcile() at the top of the
                # loop applies whatever the wanted state now says.
        except _Abandoned:
            pass
        except Exception as e:  # noqa: BLE001 - the owner must never die silently
            self._report("on_log", "error", f"Camera thread stopped on an unexpected error: {e}")
        finally:
            self._stream_on = False
            self._closed.set()

    def _next_request(self) -> tuple[str, Any]:
        """Next request: polled while streaming, waited for until the retry
        time while a failed preview waits, blocking otherwise."""
        if self._stream_on:
            return self._requests.get_nowait()
        if self._retrying:
            return self._requests.get(timeout=max(0.0, self._retry_at - time.monotonic()))
        return self._requests.get()

    def _reconcile(self) -> None:
        """Make the camera match the wanted state (idempotent)."""
        if self._retrying and not self._stream_wanted and not self._snapshot_pending:
            # The user stopped the preview while it was waiting to retry.
            self._clear_failures(notice="")
            self._report("on_preview_state", False)
        if self._stream_wanted and not self._stream_on:
            if not self._waiting_to_retry():
                self._stream_start()
        elif self._stream_on and not self._stream_wanted:
            self._stream_stop(announce=True)
        # Asked again (not cached): an arming that just failed has scheduled a
        # retry, and the pending snapshot must wait for it too instead of
        # spending a second attempt with no pause (review H1).
        if self._snapshot_pending and not self._stream_on and not self._waiting_to_retry():
            self._do_snapshot()

    def _waiting_to_retry(self) -> bool:
        return self._retrying and time.monotonic() < self._retry_at

    # -- preview failures: notice, retry, give up ----------------------
    def _preview_failed(self, message: str) -> None:
        """One live-preview failure: disarm, then retry later or give up.

        Never a dialog: the GUI shows the notice in the window and keeps
        working.  Only after PREVIEW_MAX_FAILURES in a row does the preview
        stop, and the notice says so.
        """
        if self._stream_on:
            self._stream_stop(announce=False)  # a re-arm is the usual cure
        self._preview_failures += 1
        n = self._preview_failures
        if n >= self.PREVIEW_MAX_FAILURES:
            if self._stream_wanted:
                text = f"Live preview stopped after {n} failed attempts. Last error: {message}"
            else:
                # Only a one-shot frame was asked for (e.g. after connecting).
                text = (
                    f"Preview frame given up after {n} failed attempts; the live "
                    f"preview is stopped. Last error: {message}"
                )
            self._stream_wanted = False
            self._snapshot_pending = False
            self._clear_failures(notice=text)
            self._report("on_log", "error", text)
            self._report("on_preview_state", False)
            return
        self._retrying = True
        self._retry_at = time.monotonic() + self.PREVIEW_RETRY_DELAY_S
        text = (
            f"{message} Retrying in {self.PREVIEW_RETRY_DELAY_S:g} s "
            f"(attempt {n + 1} of {self.PREVIEW_MAX_FAILURES})."
        )
        self._report("on_log", "warn", text)
        self._report("on_preview_notice", text)

    def _preview_ok(self) -> None:
        """A frame arrived: forget earlier failures and clear the notice."""
        if self._preview_failures or self._retrying:
            n = self._preview_failures
            self._clear_failures(notice="")
            self._report("on_log", "info", f"Live preview recovered after {n} failed attempt(s).")

    def _clear_failures(self, notice: str) -> None:
        self._preview_failures = 0
        self._retrying = False
        self._retry_at = 0.0
        self._report("on_preview_notice", notice)

    # -- stream --------------------------------------------------------
    def _stream_start(self) -> None:
        try:
            self._call(self._session.stream_start, self.STREAM_RING_FRAMES)
        except _Abandoned:
            raise
        except Exception as e:  # noqa: BLE001 - retried by _preview_failed
            self._preview_failed(f"Camera could not start the live preview: {describe_error(e)}.")
            return
        self._stream_on = True
        self._last_frame_t = time.monotonic()
        self._report("on_preview_state", True)

    def _stream_stop(self, announce: bool) -> None:
        try:
            self._call(self._session.stream_stop)
        except _Abandoned:
            raise
        except Exception as e:  # noqa: BLE001
            self._report("on_log", "warn", f"Could not stop the live preview cleanly: {e}")
        self._stream_on = False
        if announce:
            self._report("on_preview_state", False)

    def _pump(self) -> None:
        """Wait for the next frame (bounded) and hand the newest one out."""
        try:
            result = self._call(self._session.stream_read, self.STREAM_POLL_S)
        except _Abandoned:
            raise
        except Exception as e:  # noqa: BLE001
            self._preview_failed(f"Live preview error: {describe_error(e)}.")
            return
        now = time.monotonic()
        if result is None:
            if now - self._last_frame_t > self._frame_timeout_s:
                self._preview_failed(f"No frame from the camera for {self._frame_timeout_s:.1f} s.")
            return
        frame, z = result
        self._last_frame_t = now
        self._snapshot_pending = False  # a stream frame satisfies a pending snapshot
        self._preview_ok()
        self._report("on_frame", frame, z)

    # -- one-shot ------------------------------------------------------
    def _do_snapshot(self) -> None:
        try:
            frame, z = self._call(
                self._session.capture_preview,
                timeout=self._frame_timeout_s,
                log_cb=lambda lvl, msg: self._report("on_log", lvl, msg),
            )
        except _Abandoned:
            raise
        except Exception as e:  # noqa: BLE001
            # Stays pending: retried after the pause, like the stream.
            self._preview_failed(f"Could not capture a preview frame: {describe_error(e)}.")
            return
        # Cleared after the capture so requests made meanwhile coalesce into it.
        self._snapshot_pending = False
        self._preview_ok()
        self._report("on_frame", frame, z)

    # -- exposure ------------------------------------------------------
    def _do_exposure(self, exposure_s: float) -> None:
        if self._stream_on:
            self._stream_stop(announce=False)  # re-armed by _reconcile() afterwards
        try:
            self._call(self._session.set_exposure, exposure_s, log_cb=lambda lvl, msg: self._report("on_log", lvl, msg))
        except _Abandoned:
            raise
        except Exception as e:  # noqa: BLE001
            self._exposure_failed = True
            self._report("on_error", "exposure", f"Could not update camera exposure: {describe_error(e)}")
            return
        if self._exposure_failed:
            # Clear the notice of the earlier failure (review H2).  Not on the
            # next frame: frames keep coming at the OLD exposure, so a frame
            # does not mean the problem went away.
            self._exposure_failed = False
            self._report("on_preview_notice", "")

    # -- sweep ---------------------------------------------------------
    def _do_sweep(self, cfg: dict, cancel_flag: Any) -> None:
        if self._stream_on:
            self._stream_stop(announce=False)  # re-armed by _reconcile() if still wanted
        self._sweep_active = True
        self._sweep_cancel = cancel_flag
        try:
            # Callbacks go through _report so an abandoned thread stays silent.
            folder = self._call(
                self._session.run_sweep,
                cfg,
                progress_cb=lambda *a: self._report("on_sweep_progress", *a),
                log_cb=lambda lvl, msg: self._report("on_log", lvl, msg),
                preview_cb=lambda frame, z: self._report("on_frame", frame, z),
                cancel_flag=cancel_flag,
            )
            skipped = int(getattr(self._session, "last_sweep_skipped_frames", 0))
            total = int(getattr(self._session, "last_sweep_total_frames", 0))
            aborted = getattr(self._session, "last_sweep_aborted", None)
            if aborted:
                self._report("on_sweep_aborted", str(aborted))
            self._report("on_sweep_finished", folder, skipped, total)
        except _Abandoned:
            raise
        except Exception as e:  # noqa: BLE001
            # run_sweep already logged the cause; report it so the UI leaves
            # the "running" state.
            self._report("on_log", "error", f"Acquisition worker caught: {describe_error(e)}")
            self._report("on_error", "sweep", describe_error(e))
        finally:
            self._sweep_active = False
            self._sweep_cancel = None

    # -- close ---------------------------------------------------------
    def _do_close(self) -> None:
        if self._stream_on:
            self._stream_stop(announce=False)
        try:
            self._call(self._session.close_camera, log_cb=lambda lvl, msg: self._report("on_log", lvl, msg))
        except _Abandoned:
            raise
        except Exception as e:  # noqa: BLE001
            self._report("on_log", "warn", f"Could not close camera: {e}")
