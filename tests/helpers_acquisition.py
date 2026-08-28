"""Shared fake hardware and helpers for the acquisition test modules.

No real camera/piezo is ever touched: FakeCamera / FakePiezo implement the
minimal duck-typed interface AcquisitionSession expects, and FakeSession
replaces AcquisitionSession inside AcquisitionService tests.
"""

from __future__ import annotations

import atexit
import threading
import time

import numpy as np

from backend.acquisition.acquisition_controller import AcquisitionSession
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


class FakeCamera:
    """Minimal Thorlabs-camera stand-in: snap/set_exposure/get_exposure/close."""

    def __init__(
        self,
        frame: np.ndarray | None = None,
        fail_snap_at: tuple[int, ...] = (),
        exposure_scale: float = 1.0,
        set_exposure_error: Exception | None = None,
        get_exposure_error: Exception | None = None,
        stop_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.frame = frame if frame is not None else make_rggb_frame()
        self.fail_snap_at = set(fail_snap_at)
        self.exposure_scale = exposure_scale
        self.set_exposure_error = set_exposure_error
        self.get_exposure_error = get_exposure_error
        self.stop_error = stop_error
        self.close_error = close_error
        self.snap_count = 0
        self.exposure = 0.0
        self.stopped = False
        self.closed = False

    def set_exposure(self, exposure_s: float) -> None:
        if self.set_exposure_error is not None:
            raise self.set_exposure_error
        self.exposure = float(exposure_s)

    def get_exposure(self) -> float:
        if self.get_exposure_error is not None:
            raise self.get_exposure_error
        return self.exposure * self.exposure_scale

    def snap(self, timeout: float | None = None) -> np.ndarray:
        idx = self.snap_count
        self.snap_count += 1
        if idx in self.fail_snap_at:
            raise RuntimeError(f"fake snap failure at frame {idx}")
        return self.frame

    def stop_acquisition(self) -> None:
        self.stopped = True
        if self.stop_error is not None:
            raise self.stop_error

    def close(self) -> None:
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
        self._polls = 0
        self._last_voltage = 0.0

    def MOV(self, axis: str, z: float) -> None:  # noqa: N802 (GCS API name)
        if self.mov_error is not None:
            raise self.mov_error
        self.moves.append((axis, float(z)))
        self._polls = 0

    def qONT(self, axis: str) -> dict:  # noqa: N802
        if self.qont_hook is not None:
            self.qont_hook(self)
        self._polls += 1
        return {axis: self._polls > self.ont_polls_needed}

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

    Timing is controlled with threading.Event flags instead of sleeps so the
    tests can hold a worker 'running' deterministically and release it fast.
    """

    def __init__(self) -> None:
        self.color_mode = "mono"
        self.last_sweep_skipped_frames = 0
        self.last_sweep_total_frames = 0

        self.block_sweep = False
        self.fail_sweep: Exception | None = None
        self.sweep_started = threading.Event()
        self.release_sweep = threading.Event()

        self.block_preview = False
        self.preview_started = threading.Event()
        self.release_preview = threading.Event()

        self.sweep_calls = 0
        self.disconnect_calls = 0
        self.output_folder = "/fake/output/folder"

    # -- sweep -----------------------------------------------------------
    def run_sweep(self, cfg, progress_cb=None, log_cb=None, preview_cb=None, cancel_flag=None):
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
        return self.output_folder

    # -- preview / move --------------------------------------------------
    def capture_preview(self, timeout: float = 5.0, log_cb=None):
        self.preview_started.set()
        if self.block_preview:
            self.release_preview.wait(timeout=5.0)
        return np.zeros((2, 2), dtype=np.uint16), 1.5

    def move_to(self, z: float, log_cb=None, cancel_flag=None) -> float:
        return float(z)

    # -- connection ------------------------------------------------------
    def connect_piezo(self, serial, dll_path, axis="A", closed_loop=True, log_cb=None) -> bool:
        return True

    def connect_camera(self, exposure_s, log_cb=None) -> bool:
        return True

    def disconnect_all(self, log_cb=None) -> bool:
        self.disconnect_calls += 1
        return True
