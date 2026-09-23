"""Shared fake hardware and helpers for the acquisition test modules.

No real camera/piezo is ever touched: FakeCamera / FakePiezo implement the
minimal duck-typed interface AcquisitionSession expects, and FakeSession
replaces AcquisitionSession inside AcquisitionService tests (the real
CameraOwner thread runs on top of FakeSession's synchronous primitives).
"""

from __future__ import annotations

import atexit
import threading
import time

import numpy as np

from backend.acquisition.acquisition_controller import AcquisitionSession
from backend.acquisition.camera_owner import CameraOwner, OwnerCallbacks
from utils.camera_constants import BAYER_W_B, BAYER_W_G1, BAYER_W_G2, BAYER_W_R


def make_session() -> AcquisitionSession:
    """Create an AcquisitionSession without leaving its atexit hook behind."""
    session = AcquisitionSession()
    atexit.unregister(session.disconnect_all)
    return session


def make_rggb_frame(h: int = 8, w: int = 8, seed: int = 0) -> np.ndarray:
    """Random 12-bit RGGB Bayer frame of shape (h, w), dtype uint16."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4096, size=(h, w), dtype=np.uint16)


def reference_superpixel(frame: np.ndarray) -> np.ndarray:
    """Naive float32 reference for the weighted mono superpixel merge."""
    f = frame.astype(np.float32)
    r = f[0::2, 0::2]
    g1 = f[0::2, 1::2]
    g2 = f[1::2, 0::2]
    b = f[1::2, 1::2]
    sp = (
        np.float32(BAYER_W_R) * r
        + np.float32(BAYER_W_G1) * g1
        + np.float32(BAYER_W_G2) * g2
        + np.float32(BAYER_W_B) * b
    )
    return np.clip(np.round(sp), 0, 4095).astype(np.uint16)


def reference_superpixel_binned(frame: np.ndarray) -> np.ndarray:
    """Naive reference for the 'mono_superpixel' mode: weighted merge + 2x2 bin.

    Mirrors the documented contract of _bayer_to_superpixel_binned: odd
    intermediate dimensions are cropped before binning and the output width
    is cropped to an even value (bin12 compatibility).
    """
    mono = reference_superpixel(frame).astype(np.float32)
    h, w = mono.shape
    m = mono[: (h // 2) * 2, : (w // 2) * 2]
    binned = (m[0::2, 0::2] + m[1::2, 0::2] + m[0::2, 1::2] + m[1::2, 1::2]) * 0.25
    out = np.clip(np.round(binned), 0, 4095).astype(np.uint16)
    if out.shape[1] % 2 != 0 and out.shape[1] > 1:
        out = out[:, :-1]
    return out


class FakeCameraTimeoutError(RuntimeError):
    """Stands in for pylablib's ThorlabsTLCameraTimeoutError (empty message)."""


class FakeCamera:
    """Minimal Thorlabs-camera stand-in.

    Covers snap()/set_exposure()/get_exposure()/close() as before, plus the
    continuous-acquisition surface the camera thread uses: start_acquisition
    (armed state + ring size), wait_for_frame, read_newest_image and
    stop_acquisition.  Every public call records the calling thread in
    ``call_threads`` and its name in ``calls`` so tests can assert that a
    single thread ever touches the camera, and in what order.

    Stream frames: ``stream_frames`` (cycled) or ``frame``; each delivered
    frame gets an increasing index (``delivered``).  Knobs:
    ``frame_period`` (s per frame), ``hang_event`` (wait_for_frame blocks
    on it: a hung SDK call), ``wait_error`` (raised by wait_for_frame),
    ``no_frames`` (wait_for_frame always times out).
    """

    TimeoutError = FakeCameraTimeoutError

    def __init__(
        self,
        frame: np.ndarray | None = None,
        fail_snap_at: tuple[int, ...] = (),
        exposure_scale: float = 1.0,
        set_exposure_error: Exception | None = None,
        get_exposure_error: Exception | None = None,
        stop_error: Exception | None = None,
        close_error: Exception | None = None,
        stream_frames: list[np.ndarray] | None = None,
        frame_period: float = 0.002,
    ) -> None:
        self.frame = frame if frame is not None else make_rggb_frame()
        self.fail_snap_at = set(fail_snap_at)
        self.exposure_scale = exposure_scale
        self.set_exposure_error = set_exposure_error
        self.get_exposure_error = get_exposure_error
        self.stop_error = stop_error
        self.close_error = close_error
        self.start_error: Exception | None = None
        self.snap_count = 0
        self.exposure = 0.0
        self.stopped = False
        self.closed = False

        # continuous acquisition
        self.stream_frames = stream_frames
        self.frame_period = frame_period
        self.armed = False
        self.ring_frames: int | None = None
        self.start_count = 0
        self.stop_count = 0
        self.acquired = 0
        self.last_read = 0
        self.delivered: list[int] = []
        self.hang_event: threading.Event | None = None
        self.wait_error: Exception | None = None
        self.no_frames = False

        # observation
        self.calls: list[str] = []
        self.call_threads: set[int] = set()
        self.exposure_calls: list[tuple[float, bool]] = []  # (exposure, armed at call)
        self.close_thread: int | None = None

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.call_threads.add(threading.get_ident())

    def set_exposure(self, exposure_s: float) -> None:
        self._record("set_exposure")
        self.exposure_calls.append((float(exposure_s), self.armed))
        if self.set_exposure_error is not None:
            raise self.set_exposure_error
        self.exposure = float(exposure_s)

    def get_exposure(self) -> float:
        self._record("get_exposure")
        if self.get_exposure_error is not None:
            raise self.get_exposure_error
        return self.exposure * self.exposure_scale

    def snap(self, timeout: float | None = None) -> np.ndarray:
        self._record("snap")
        idx = self.snap_count
        self.snap_count += 1
        if idx in self.fail_snap_at:
            raise RuntimeError(f"fake snap failure at frame {idx}")
        return self.frame

    # -- continuous acquisition (what the camera thread uses) --------------
    def start_acquisition(self, frames_per_trigger="default", auto_start=True, nframes=None):
        self._record("start_acquisition")
        if self.start_error is not None:
            raise self.start_error
        self.armed = True
        self.ring_frames = nframes
        self.start_count += 1
        self.acquired = 0
        self.last_read = 0

    def acquisition_in_progress(self) -> bool:
        return self.armed

    def wait_for_frame(self, since="lastread", nframes=1, timeout=20.0, error_on_stopped=False):
        self._record("wait_for_frame")
        if self.hang_event is not None:
            self.hang_event.wait()  # a hung SDK call: ignores its timeout
        if self.wait_error is not None:
            raise self.wait_error
        if not self.armed:
            return False
        if self.no_frames:
            time.sleep(min(timeout, 0.01))
            raise self.TimeoutError()
        time.sleep(self.frame_period)
        self.acquired += 1
        return True

    def read_newest_image(self, peek=False, return_info=False):
        self._record("read_newest_image")
        if not self.armed or self.acquired <= self.last_read:
            return None
        idx = self.acquired - 1
        self.last_read = self.acquired
        self.delivered.append(idx)
        if self.stream_frames:
            return self.stream_frames[idx % len(self.stream_frames)]
        return self.frame

    def stop_acquisition(self) -> None:
        self._record("stop_acquisition")
        self.stopped = True
        self.armed = False
        self.stop_count += 1
        if self.stop_error is not None:
            raise self.stop_error

    def close(self) -> None:
        self._record("close")
        self.close_thread = threading.get_ident()
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakePiezo:
    """Minimal GCSDevice stand-in: MOV/SVA/qONT/qVOL/SVO/CloseConnection."""

    def __init__(self, ont_polls_needed: int = 0) -> None:
        #: qONT returns False this many times after each MOV, then True.
        self.ont_polls_needed = ont_polls_needed
        #: optional hook called on every qONT poll (e.g. to trigger a cancel).
        self.qont_hook = None
        self.moves: list[tuple[str, float]] = []
        self.sva_calls: list[tuple[str, float]] = []
        self.svo_calls: list[tuple[str, int]] = []
        self.closed = False
        self.mov_error: Exception | None = None
        #: qPOS answer: the last MOV target unless set; an Exception is raised
        self.position: float | Exception | None = None
        self.qpos_calls = 0
        self._target = 0.0
        self._polls = 0
        self._last_voltage = 0.0

    def MOV(self, axis: str, z: float) -> None:  # noqa: N802 (GCS API name)
        if self.mov_error is not None:
            raise self.mov_error
        self.moves.append((axis, float(z)))
        self._target = float(z)
        self._polls = 0

    def qONT(self, axis: str) -> dict:  # noqa: N802
        if self.qont_hook is not None:
            self.qont_hook(self)
        self._polls += 1
        return {axis: self._polls > self.ont_polls_needed}

    def qPOS(self, axis: str) -> dict:  # noqa: N802
        self.qpos_calls += 1
        if isinstance(self.position, Exception):
            raise self.position
        return {axis: self._target if self.position is None else self.position}

    def SVA(self, axis: str, z: float) -> None:  # noqa: N802
        self.sva_calls.append((axis, float(z)))
        self._last_voltage = float(z)

    def qVOL(self, axis: str) -> dict:  # noqa: N802
        return {axis: self._last_voltage}

    def SVO(self, axis: str, value: int) -> None:  # noqa: N802
        self.svo_calls.append((axis, int(value)))

    def CloseConnection(self) -> None:  # noqa: N802
        self.closed = True


class LogCollector:
    """Callable log_cb that records (level, message) tuples."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def __call__(self, level: str, msg: str) -> None:
        self.records.append((level, msg))

    def messages(self, level: str | None = None) -> list[str]:
        return [m for lvl, m in self.records if level is None or lvl == level]

    def contains(self, fragment: str, level: str | None = None) -> bool:
        return any(fragment in m for m in self.messages(level))


class FakeSession:
    """AcquisitionSession stand-in for AcquisitionService tests.

    The real CameraOwner thread runs on top of these synchronous primitives
    (run_sweep, capture_preview, stream_*, set_exposure, close_camera), so
    the service tests exercise the actual request queue and mailbox.
    Timing is controlled with threading.Event flags instead of sleeps so the
    tests can hold the camera thread 'busy' deterministically and release it
    fast.
    """

    CLOSE_TIMEOUT_S = 3.0

    def __init__(self) -> None:
        self.color_mode = "mono"
        self.last_sweep_skipped_frames = 0
        self.last_sweep_total_frames = 0
        self.last_sweep_aborted: str | None = None
        self.abort_sweep_with: str | None = None  # run_sweep "aborts" with this summary
        self.exposure_error: Exception | None = None
        self.preview_error: Exception | None = None  # capture_preview raises it

        self.block_sweep = False
        self.fail_sweep: Exception | None = None
        self.sweep_started = threading.Event()
        self.release_sweep = threading.Event()

        self.block_preview = False
        self.preview_started = threading.Event()
        self.release_preview = threading.Event()

        # live stream: one numbered frame per stream_read (value = sequence)
        self.stream_period = 0.002
        self.stream_running = False
        self.stream_starts = 0
        self.stream_stops = 0
        self.stream_reads = 0
        self.stream_frames = 0
        self.stream_error: Exception | None = None
        self.block_move = False
        self.release_move = threading.Event()

        self.exposure: float | None = None
        self.camera_closed = False
        self.order: list[str] = []  # "preview" / "sweep" / "stream_start" ... in call order

        self.sweep_calls = 0
        self.disconnect_calls = 0
        self.output_folder = "/fake/output/folder"

        self._owner: CameraOwner | None = None
        self._owner_callbacks = OwnerCallbacks()
        self.move_calls: list[float] = []

        # connect / disconnect (run on the service's connection thread)
        self.piezo_connects = True
        self.camera_connects = True
        self.block_connect = False
        self.connect_started = threading.Event()
        self.release_connect = threading.Event()
        self.connect_threads: list[str] = []
        self.camera_connect_calls: list[str] = []  # thread names
        self.block_disconnect = False
        self.ignore_cancel = False
        self.disconnect_started = threading.Event()
        self.release_disconnect = threading.Event()
        self.disconnect_log: list[tuple[str, float | None]] = []  # (thread name, park timeout)

    # -- camera thread (same surface as AcquisitionSession) --------------
    def set_owner_callbacks(self, callbacks: OwnerCallbacks) -> None:
        self._owner_callbacks = callbacks

    def camera_owner(self, create: bool = True) -> CameraOwner | None:
        owner = self._owner
        if owner is not None and owner.is_alive():
            return owner
        if not create:
            return None
        owner = CameraOwner(self, self._owner_callbacks)
        self._owner = owner
        owner.start()
        return owner

    @property
    def camera_connected(self) -> bool:
        return True

    def stream_start(self, ring_frames: int) -> None:
        self.order.append("stream_start")
        self.stream_starts += 1
        self.stream_running = True

    def stream_read(self, timeout: float):
        self.stream_reads += 1
        if self.stream_error is not None:
            raise self.stream_error
        time.sleep(self.stream_period)
        self.stream_frames += 1
        return np.full((2, 2), self.stream_frames, dtype=np.uint16), 1.5

    def stream_stop(self) -> None:
        self.order.append("stream_stop")
        self.stream_stops += 1
        self.stream_running = False

    def set_exposure(self, exposure_s: float, log_cb=None) -> None:
        self.order.append("set_exposure")
        if self.exposure_error is not None:
            raise self.exposure_error
        self.exposure = float(exposure_s)
        if log_cb:
            log_cb("info", f"Camera exposure updated to {exposure_s:.4g} s")

    def close_camera(self, log_cb=None) -> None:
        self.order.append("close_camera")
        self.camera_closed = True

    # -- sweep -----------------------------------------------------------
    def run_sweep(self, cfg, progress_cb=None, log_cb=None, preview_cb=None, cancel_flag=None):
        self.order.append("sweep")
        self.sweep_calls += 1
        self.sweep_started.set()
        if log_cb:
            log_cb("info", "fake sweep started")
        if self.fail_sweep is not None:
            raise self.fail_sweep
        if self.block_sweep:
            # Cooperative wait: exits on release event OR cancellation.
            while not self.release_sweep.is_set():
                if cancel_flag is not None and cancel_flag.is_cancelled():
                    break
                time.sleep(0.005)
        if progress_cb:
            progress_cb(100.0, 0.0, 1, 1, 0.01, 0.0)
        self.last_sweep_skipped_frames = 1
        self.last_sweep_total_frames = 5
        self.last_sweep_aborted = self.abort_sweep_with
        return self.output_folder

    # -- preview / move --------------------------------------------------
    def capture_preview(self, timeout: float = 5.0, log_cb=None):
        self.order.append("preview")
        self.preview_started.set()
        if self.block_preview:
            self.release_preview.wait(timeout=5.0)
        if self.preview_error is not None:
            raise self.preview_error
        return np.zeros((2, 2), dtype=np.uint16), 1.5

    def move_to(self, z: float, log_cb=None, cancel_flag=None, position_cb=None) -> float:
        self.move_calls.append(float(z))
        if self.block_move:
            while not self.release_move.is_set():
                if cancel_flag is not None and cancel_flag.is_cancelled():
                    break
                time.sleep(0.005)
        if position_cb is not None:
            position_cb(float(z))  # the real session reports the measured end position
        return float(z)

    # -- connection ------------------------------------------------------
    def connect_piezo(self, serial, dll_path, axis="A", closed_loop=True, log_cb=None) -> bool:
        self.connect_threads.append(threading.current_thread().name)
        if self.block_connect:
            self.connect_started.set()
            self.release_connect.wait(timeout=5.0)
        return self.piezo_connects

    last_position = 0.0

    def connect_camera(self, exposure_s, log_cb=None) -> bool:
        self.camera_connect_calls.append(threading.current_thread().name)
        return self.camera_connects

    def disconnect_all(self, log_cb=None, park_timeout_s=None, cancel_flag=None) -> bool:
        self.disconnect_calls += 1
        self.disconnect_log.append((threading.current_thread().name, park_timeout_s))
        if self.block_disconnect:
            self.disconnect_started.set()
            # like the piezo park: cut short by the cancel flag (or ignores it)
            while not self.release_disconnect.is_set():
                if (
                    cancel_flag is not None
                    and cancel_flag.is_cancelled()
                    and not self.ignore_cancel
                ):
                    break
                time.sleep(0.005)
        owner = self._owner
        if owner is not None and owner.is_alive():
            self._owner = None
            if not owner.close(self.CLOSE_TIMEOUT_S):
                owner.abandon()
                if log_cb:
                    log_cb("warn", "Camera is not answering (fake): camera thread abandoned.")
        return True
