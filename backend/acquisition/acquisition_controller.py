# backend/acquisition/acquisition_controller.py
from __future__ import annotations

import atexit
import contextlib
import csv
import math
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime

import cv2
import numpy as np
from pipython import GCSDevice, GCSError
from pylablib.devices import Thorlabs

from backend.acquisition.camera_owner import CameraOwner, OwnerCallbacks, describe_error
from utils import bin12
from utils.camera_constants import (
    BAYER_W_B,
    BAYER_W_G1,
    BAYER_W_G2,
    BAYER_W_R,
    DEFAULT_BAYER_PHASE,
    bayer_sites,
)

# --------------------------------------------------------
# Callback type aliases
# --------------------------------------------------------
ProgressCallback = Callable[[float, float, int, int, float, float], None]
LogCallback = Callable[[str, str], None]
PreviewCallback = Callable[[np.ndarray, float], None]
# Measured piezo position (µm), NaN when it could not be read
PositionCallback = Callable[[float], None]


# --------------------------------------------------------
# Cooperative cancellation flag
# --------------------------------------------------------
class CancelFlag:
    """
    Simple cooperative cancellation flag.

    cancel() is called from the main thread; is_cancelled() is polled
    from the worker thread.  Plain bool read/write is GIL-safe in CPython.
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


# --------------------------------------------------------
# Main acquisition controller
# --------------------------------------------------------
class AcquisitionSession:
    """
    Controls the piezo and camera, performing Z-sweep acquisitions.

    This class has **no dependency on Qt**.
    The Qt Services/VMs connect to it exclusively via callbacks.
    """

    # Maximum time (s) to wait for the piezo to reach its target position.
    # Prevents infinite hangs when hardware is unresponsive.
    MOVE_TIMEOUT_S: float = 30.0

    # Polling period (s) used while waiting for the piezo to settle.
    MOVE_POLL_S: float = 0.1

    # How long disconnect/shutdown wait for the camera thread to close the
    # camera before declaring it "not answering" and abandoning it.  Must
    # cover a queued snapshot (~0.5 s) or a sweep being cancelled (one snap
    # timeout, 5 s by default, plus pylablib's 0.25 s of sleeps).
    CAMERA_CLOSE_TIMEOUT_S: float = 8.0

    # How long disconnect_all waits for the piezo to report it is back at 0
    # before switching the servo off anyway.  The app's close passes a
    # shorter value (AcquisitionService.CLOSE_PARK_TIMEOUT_S).
    PIEZO_PARK_TIMEOUT_S: float = 10.0

    # Consecutive failed captures (or, separately, failed saves: a full disk)
    # after which a sweep aborts instead of walking every remaining position
    # (finding C4: a lost camera cost 2 min, a mute one 47 min, for a dataset
    # with no frames).  A single failure is still skipped, as before: a lone
    # timeout should not waste a sweep.
    SWEEP_MAX_CONSECUTIVE_FAILURES: int = 3

    # File written into the output folder of an aborted sweep, so the partial
    # dataset is recognisable later without the log.
    ABORT_MARKER: str = "SWEEP_ABORTED.txt"

    # One row per SAVED frame, written as the sweep goes (never kept in
    # memory): index, image file, commanded z, measured z, timestamp.
    POSITIONS_CSV: str = "positions.csv"
    POSITIONS_HEADER: tuple[str, ...] = (
        "index",
        "filename",
        "z_commanded_um",
        # Read with qPOS when the MOVE ENDS, before the exposure, not while
        # the frame is being taken: good enough to check the sweep, NOT a
        # per-frame z for the reconstruction (see README.md).
        "z_measured_after_move_um",
        "timestamp",
    )

    # Consecutive qONT errors tolerated while waiting for the piezo; a move
    # whose arrival cannot be confirmed is a failure, not a success (C11).
    QONT_MAX_CONSECUTIVE_FAILURES: int = 3

    # GCS error "Unknown command": the controller has no qONT at all, the
    # only case where a fixed settle-time wait is the right answer.
    GCS_UNKNOWN_COMMAND: int = 2

    def __init__(self) -> None:
        self._piezo: GCSDevice | None = None
        self._camera: Thorlabs.ThorlabsTLCamera | None = None
        self._closed_loop: bool = True
        self._axis: str = "A"
        self._settle_time: float = 0.2

        # The one thread that uses the camera once connected (see
        # camera_owner.py).  Created on demand by camera_owner(); None or a
        # dead thread means nobody but the caller is using the camera.
        self._camera_owner: CameraOwner | None = None
        self._owner_callbacks: OwnerCallbacks = OwnerCallbacks()

        # Last MEASURED piezo position (qPOS, µm; volts in open loop), NaN when
        # unknown.  Labels the preview frames; only the thread that uses the
        # piezo writes it (a manual move, a sweep, the connection).
        self._last_real_pos: float = 0.0

        # "mono" → weighted single channel; "color" → R/G/B channels separate;
        # "mono_superpixel" → weighted single channel + extra 2×2 binning
        self._color_mode: str = "mono"

        # Bayer mosaic phase of the connected sensor, read from the camera in
        # connect_camera().  The superpixel merges need it to know which corner
        # of each 2x2 block is red: the LP126CU is BGGR, not RGGB.
        self._bayer_phase: str = DEFAULT_BAYER_PHASE

        # Statistics of the last completed sweep (frames skipped vs planned)
        self.last_sweep_skipped_frames: int = 0
        self.last_sweep_total_frames: int = 0
        # Why the last sweep stopped early on camera failures (None if it did not)
        self.last_sweep_aborted: str | None = None

        # Safety net: release hardware if the process exits without a clean
        # shutdown path. disconnect_all() is idempotent, so a normal shutdown
        # followed by this atexit hook is harmless.
        atexit.register(self.disconnect_all)

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @color_mode.setter
    def color_mode(self, value: str) -> None:
        self._color_mode = value.strip().lower()

    # --------------------------------------------------------
    # Hardware connections
    # --------------------------------------------------------
    def connect_piezo(
        self,
        serial: str,
        dll_path: str,
        axis: str = "A",
        closed_loop: bool = True,
        log_cb: LogCallback | None = None,
    ) -> bool:
        """Connect PI controller."""
        try:
            dev = GCSDevice(devname="E-625", gcsdll=dll_path)
            dev.ConnectUSB(serialnum=serial)

            if log_cb:
                log_cb("info", f"Connected to PI controller: {dev.qIDN().strip()}")

            err = dev.qERR()
            if log_cb:
                log_cb("info", f"Initial PI ERR: {err}")

            self._piezo = dev
            self._axis = axis
            self._closed_loop = closed_loop

            dev.SVO(axis, 1 if closed_loop else 0)
            if log_cb:
                log_cb("info", f"Servo {'ON' if closed_loop else 'OFF'}")

            # Start from where the stage really is, not from 0 (C3/U4).
            self._last_real_pos = self._read_position(dev, log_cb)
            return True

        except GCSError as e:
            if log_cb:
                log_cb("error", f"PI GCS Error: {e}")
            return False

        except Exception as e:
            if log_cb:
                log_cb("error", f"Error connecting piezo: {e}")
            return False

    def connect_camera(
        self,
        exposure_s: float,
        log_cb: LogCallback | None = None,
    ) -> bool:
        """Connect Thorlabs camera, apply RAW/Binning and set exposure.

        With the camera already connected this only updates the exposure:
        through the camera thread when it is alive (applied between frames,
        asynchronously), inline otherwise.
        """
        try:
            if self._camera is not None:
                if self._owner_alive():
                    self._camera_owner.set_exposure(exposure_s)
                    return True
                try:
                    self.set_exposure(exposure_s, log_cb=log_cb)
                    return True
                except Exception as e:
                    if log_cb:
                        log_cb("error", f"Could not update camera exposure: {e}")
                    return False

            cam = Thorlabs.ThorlabsTLCamera()
            try:
                info = cam.get_device_info()
                if log_cb:
                    log_cb("info", f"Camera opened: {info.model} (serial {info.serial_number})")

                # ---------------------------------------------------------
                # Force RAW mono output so the full Bayer pattern is preserved.
                # Hardware binning is intentionally NOT used: the 2×2 Bayer
                # superpixel merge is done in software with physically-derived
                # per-channel weights (_bayer_to_superpixel).
                # ---------------------------------------------------------
                # pylablib debayers in software by default on color cameras
                # (color_output="auto" -> "rgb"), which makes snap() return an
                # (H, W, 3) array and breaks the superpixel merge. Ask for the
                # raw sensor data explicitly.
                try:
                    fmt = cam.set_color_format(color_output="raw", color_space="linear")
                    if log_cb:
                        log_cb(
                            "info",
                            f"Camera color format set to {fmt.color_format} (raw Bayer output).",
                        )
                except Exception as e:
                    if log_cb:
                        log_cb("error", f"Could not force raw Bayer output: {e}")
                    raise

                # The superpixel merges need the mosaic phase to place the red
                # and blue weights (and, in colour mode, the R and B channels).
                # It is read from the camera instead of assumed: this sensor
                # reports "blue" (BGGR), and assuming RGGB swapped R with B.
                try:
                    cinfo = cam.get_color_info()
                except Exception:
                    cinfo = None
                if cinfo is None:
                    self._bayer_phase = DEFAULT_BAYER_PHASE
                    if log_cb:
                        log_cb("info", "Monochrome sensor: no Bayer mosaic reported.")
                else:
                    phase = cinfo.filter_array_phase
                    try:
                        bayer_sites(phase)
                    except ValueError:
                        # Unknown phase: fall back to RGGB rather than refusing
                        # to acquire, but say so — colours may be wrong.
                        if log_cb:
                            log_cb(
                                "warn",
                                f"Unrecognised sensor Bayer phase {phase!r}; assuming "
                                f"{DEFAULT_BAYER_PHASE} (RGGB). Colour channels may be swapped.",
                            )
                        phase = DEFAULT_BAYER_PHASE
                    self._bayer_phase = phase
                    if log_cb:
                        log_cb("info", f"Sensor Bayer phase: {phase}.")

                # Ensure no hardware binning is active (reset to 1×1 explicitly).
                try:
                    cam.set_roi(hbin=1, vbin=1)
                except Exception as e:
                    if log_cb:
                        log_cb("warn", f"Could not reset hardware binning to 1×1: {e}")
                # ---------------------------------------------------------

                cam.set_exposure(exposure_s)

                # The camera is NOT armed here any more: arming belongs to the
                # camera thread (live preview stream or snap()), which retries
                # it and reports a failure as a preview error.  Arming at
                # connect only made the first capture pay an extra disarm.
                self._camera = cam
                return True

            except Exception:
                # Setup failed after the device was opened: close the handle
                # so the camera does not stay locked ("device in use").
                try:
                    cam.close()
                except Exception as close_err:
                    if log_cb:
                        log_cb("warn", f"Could not close camera after failed setup: {close_err}")
                raise

        except Exception as e:
            if log_cb:
                log_cb("error", f"Error connecting camera: {e}")
            return False

    def close_camera(self, log_cb: LogCallback | None = None) -> None:
        """Stop and close the camera object (inline, in the calling thread).

        Only the camera thread may call this while it is alive; everybody
        else goes through disconnect_all(), which asks that thread.  The
        attribute is cleared BEFORE the SDK calls so that a call that hangs
        can never be followed by a close of a camera reconnected later.
        """
        cam = self._camera
        if cam is None:
            return
        self._camera = None
        try:
            cam.stop_acquisition()
        except Exception as e:
            if log_cb:
                log_cb("warn", f"Could not stop camera acquisition: {e}")
        try:
            cam.close()
            if log_cb:
                log_cb("info", "Camera closed")
        except Exception as e:
            if log_cb:
                log_cb("warn", f"Could not close camera: {e}")

    def disconnect_all(
        self,
        log_cb: LogCallback | None = None,
        park_timeout_s: float | None = None,
        cancel_flag: CancelFlag | None = None,
    ) -> bool:
        """
        Close camera and piezo safely.

        The camera is closed by its owner thread when one is alive (bounded
        wait; if the thread does not answer, it is abandoned and reported and
        the camera is never touched again from here), inline otherwise.  The
        close is only REQUESTED first and waited for after the piezo has
        been released, so the two waits overlap instead of adding up (a hung
        camera and a piezo that never reports on-target cost max(8, 10) s,
        not 18 s).  The piezo is sent to 0 and waited for up to
        park_timeout_s (the app's close uses a shorter one); cancel_flag cuts
        that wait short (the servo is still switched off and the link closed).
        Each device is shut down independently, so a failure on one never
        prevents releasing the other. Idempotent: safe to call repeatedly
        (also used as an atexit safety net).  Blocking: the service runs it
        on its connection thread, never on the GUI thread (finding C8).
        """
        if park_timeout_s is None:
            park_timeout_s = self.PIEZO_PARK_TIMEOUT_S
        t0 = time.monotonic()
        owner = self._camera_owner
        closing_owner = None
        if owner is not None and owner.is_alive() and threading.current_thread() is not owner:
            self._camera_owner = None
            owner.request_close()
            closing_owner = owner
        elif self._camera is not None:
            self.close_camera(log_cb)

        if self._piezo is not None:
            dev = self._piezo
            self._piezo = None
            try:
                if self._closed_loop:
                    dev.MOV(self._axis, 0.0)
                    try:
                        self._wait_on_target(dev, timeout_s=park_timeout_s, cancel_flag=cancel_flag)
                    except TimeoutError:
                        # Not fatal: the MOV was sent; servo off and close anyway.
                        if log_cb:
                            log_cb(
                                "warn",
                                f"Piezo did not report reaching 0 within {park_timeout_s:.0f} s; "
                                "closing it anyway.",
                            )
                dev.SVO(self._axis, 0)
                dev.SVA(self._axis, 0)
            except Exception as e:
                if log_cb:
                    log_cb("warn", f"Could not reset piezo before closing: {e}")

            try:
                dev.CloseConnection()
                if log_cb:
                    log_cb("info", "Piezo connection closed")
            except Exception as e:
                if log_cb:
                    log_cb("warn", f"Could not close piezo: {e}")

        if closing_owner is not None:
            left = self.CAMERA_CLOSE_TIMEOUT_S - (time.monotonic() - t0)
            if not closing_owner.wait_closed(left):
                # A camera call is not returning.  Never touch that camera
                # from another thread (undefined behaviour in the SDK): let
                # the stuck thread go and forget the handle.
                closing_owner.abandon()
                self._camera = None
                if log_cb:
                    log_cb(
                        "warn",
                        f"Camera is not answering: a camera call did not return within "
                        f"{self.CAMERA_CLOSE_TIMEOUT_S:.0f} s. The camera thread has been "
                        "abandoned; power-cycle the camera before reconnecting.",
                    )

        return True

    # --------------------------------------------------------
    # Camera owner thread and the primitives it runs
    # --------------------------------------------------------
    @property
    def last_position(self) -> float:
        """Last measured piezo position (µm), NaN when unknown.  No query."""
        return self._last_real_pos

    @property
    def camera_connected(self) -> bool:
        return self._camera is not None

    def set_owner_callbacks(self, callbacks: OwnerCallbacks) -> None:
        """Callbacks the camera thread reports through (set once by the service)."""
        self._owner_callbacks = callbacks

    def _owner_alive(self) -> bool:
        owner = self._camera_owner
        return owner is not None and owner.is_alive() and threading.current_thread() is not owner

    def camera_owner(self, create: bool = True) -> CameraOwner | None:
        """The live camera thread; started on demand when `create` is True."""
        owner = self._camera_owner
        if owner is not None and owner.is_alive():
            return owner
        if not create:
            return None
        owner = CameraOwner(self, self._owner_callbacks)
        self._camera_owner = owner
        owner.start()
        return owner

    def stream_start(self, ring_frames: int) -> None:
        """Arm the camera once, in continuous mode (camera thread only)."""
        if self._camera is None:
            raise RuntimeError("Camera not connected.")
        self._camera.start_acquisition(nframes=int(ring_frames))

    def stream_read(self, timeout: float) -> tuple[np.ndarray, float] | None:
        """Wait up to `timeout` s for a new frame and return the NEWEST one, merged.

        Returns None when no frame arrived in time.  Frames that piled up
        meanwhile are skipped, never queued: the preview shows the present.
        """
        cam = self._camera
        if cam is None:
            raise RuntimeError("Camera not connected.")
        timeout_cls = getattr(cam, "TimeoutError", TimeoutError)
        try:
            got = cam.wait_for_frame(since="lastread", nframes=1, timeout=timeout)
        except timeout_cls:
            return None
        if not got:
            # pylablib returns False (no wait) when the camera is not armed.
            raise RuntimeError("Camera acquisition is not running.")
        raw = cam.read_newest_image()
        if raw is None:
            return None
        raw = np.asarray(raw)
        if raw.dtype != np.uint16:
            raw = raw.astype(np.uint16)
        return self._merge(raw), float(self._last_real_pos)

    def stream_stop(self) -> None:
        """Disarm the camera (camera thread only)."""
        if self._camera is not None:
            self._camera.stop_acquisition()

    def set_exposure(self, exposure_s: float, log_cb: LogCallback | None = None) -> None:
        """Apply the exposure to the connected camera (raises on failure)."""
        if self._camera is None:
            raise RuntimeError("Camera not connected.")
        self._camera.set_exposure(exposure_s)
        if log_cb:
            log_cb("info", f"Camera exposure updated to {exposure_s:.4g} s")

    def _merge(self, raw: np.ndarray) -> np.ndarray:
        """Bayer superpixel merge according to the current colour mode."""
        if self._color_mode == "color":
            return self._bayer_to_color_superpixel(raw, self._bayer_phase)
        if self._color_mode == "mono_superpixel":
            return self._bayer_to_superpixel_binned(raw, self._bayer_phase)
        return self._bayer_to_superpixel(raw, self._bayer_phase)

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------
    @staticmethod
    def _save_bin12(frame: np.ndarray, filepath: str) -> None:
        """
        Save a uint16 frame (12-bit values, [0..4095]) as a packed .bin12 file.

        3-channel frames are expected in BGR order (see utils.bin12.pack_frame).
        """
        with open(filepath, "wb") as f:
            f.write(bin12.pack_frame(frame))

    @staticmethod
    def _bayer_to_color_superpixel(
        frame: np.ndarray, phase: str = DEFAULT_BAYER_PHASE
    ) -> np.ndarray:
        """
        Convert a full-resolution Bayer frame to a 3-channel superpixel image.

        Each 2×2 Bayer block  [ R  G1 ]  is collapsed to (R, G, B) where
                               [ G2  B ]
            G = (G1 + G2) / 2

        Which corner holds which colour depends on the sensor's mosaic phase
        (``phase``), so the block above is the RGGB case; see
        utils.camera_constants.bayer_sites.

        No spectral weighting is applied so each channel retains its independent
        signal.  This lets the caller inspect or discard individual channels when
        one of them is known to be corrupted (e.g. partial CFA damage or heavy
        noise on a single colour).

        Parameters
        ----------
        frame : uint16 ndarray, shape (H, W).  Values in [0, 4095] (12-bit).
                H and W must be even.
        phase : Bayer mosaic phase reported by the camera.

        Returns
        -------
        uint16 ndarray, shape (H//2, W//2, 3), channel order [R, G, B].
        Values in [0, 4095].
        """
        if frame.ndim != 2:
            raise ValueError(
                f"_bayer_to_color_superpixel expects a 2-D Bayer frame, got shape {frame.shape}"
            )
        H, W = frame.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"_bayer_to_color_superpixel requires even dimensions, got {H}×{W}")

        (r_y, r_x), (g1_y, g1_x), (g2_y, g2_x), (b_y, b_x) = bayer_sites(phase)
        f = frame.astype(np.float32)
        R = f[r_y::2, r_x::2]
        G1 = f[g1_y::2, g1_x::2]
        G2 = f[g2_y::2, g2_x::2]
        B = f[b_y::2, b_x::2]

        G = (G1 + G2) * 0.5
        out = np.stack([R, G, B], axis=2)
        return np.clip(np.round(out), 0, 4095).astype(np.uint16)

    @staticmethod
    def _bayer_to_superpixel(frame: np.ndarray, phase: str = DEFAULT_BAYER_PHASE) -> np.ndarray:
        """
        Convert a full-resolution Bayer frame to a monochrome superpixel image.

        Each 2×2 Bayer block  [ R  G1 ]  is collapsed to a single value:
                               [ G2  B ]

            superpixel = w_R·R + w_G1·G1 + w_G2·G2 + w_B·B

        The block above is the RGGB case; the actual corner of each colour comes
        from the sensor's mosaic phase (``phase``).  Getting it wrong swaps w_R
        with w_B, which is a real (if subtle) photometric error.

        Weights come from utils.camera_constants (LP126CU QE curves, IR-filter
        transmission, 3200 K blackbody spectrum — Olympus U-LH100IR).

        Parameters
        ----------
        frame : uint16 ndarray, shape (H, W).  Values in [0, 4095] (12-bit).
                H and W must be even.

        Returns
        -------
        uint16 ndarray, shape (H//2, W//2).  Values in [0, 4095].
        """
        if frame.ndim != 2:
            raise ValueError(
                f"_bayer_to_superpixel expects a 2-D Bayer frame, got shape {frame.shape}"
            )
        H, W = frame.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"_bayer_to_superpixel requires even dimensions, got {H}×{W}")

        # O-3: preallocated float32 accumulator + out= keyword everywhere.
        # This is BIT-EXACT with the previous expression
        #     sp = W_R*R + W_G1*G1 + W_G2*G2 + W_B*B   (on frame.astype(float32))
        # because: the uint16->float32 element conversion inside each multiply
        # is exact; np.float32(w) applies the same rounding numpy used for the
        # Python-float scalars; and the left-to-right sum association is kept.
        # It avoids the full-resolution float32 copy of the frame and the seven
        # (H/2, W/2) temporaries the old expression materialised (~2x faster on
        # 4096x3000 frames; runs per sweep frame AND per preview frame).
        # Slices come from the mosaic phase; with the default RGGB phase these
        # are the historical [0::2, 0::2] ... [1::2, 1::2], so the sum stays
        # bit-exact with the previous code.
        (r_y, r_x), (g1_y, g1_x), (g2_y, g2_x), (b_y, b_x) = bayer_sites(phase)
        R = frame[r_y::2, r_x::2]
        G1 = frame[g1_y::2, g1_x::2]
        G2 = frame[g2_y::2, g2_x::2]
        B = frame[b_y::2, b_x::2]

        acc = np.empty((H // 2, W // 2), dtype=np.float32)
        tmp = np.empty((H // 2, W // 2), dtype=np.float32)
        np.multiply(R, np.float32(BAYER_W_R), out=acc)
        np.multiply(G1, np.float32(BAYER_W_G1), out=tmp)
        np.add(acc, tmp, out=acc)
        np.multiply(G2, np.float32(BAYER_W_G2), out=tmp)
        np.add(acc, tmp, out=acc)
        np.multiply(B, np.float32(BAYER_W_B), out=tmp)
        np.add(acc, tmp, out=acc)
        np.round(acc, out=acc)
        np.clip(acc, 0, 4095, out=acc)
        return acc.astype(np.uint16)

    @staticmethod
    def _bayer_to_superpixel_binned(
        frame: np.ndarray, phase: str = DEFAULT_BAYER_PHASE
    ) -> np.ndarray:
        """
        Convert a full-resolution Bayer frame to a small monochrome image.

        Two-stage reduction producing the same kind of output as
        scripts/process_superpixel.py, but directly at acquisition time:

        1. Weighted RGGB superpixel merge (``_bayer_to_superpixel``):
           (H, W) Bayer -> (H/2, W/2) mono.
        2. 2x2 average binning of the mono result:
           (H/2, W/2) -> (H/4, W/4).

        The result is 4x smaller than the "mono" mode and 12x smaller than
        the "color" mode, ready for analysis without any post-processing.

        If the intermediate mono image has odd dimensions they are cropped
        to even values before binning, and an odd binned width (when wider
        than one column) is cropped to an even value as well, so any
        realistic sensor size stays bin12-compatible (the packed 12-bit
        format requires an even row width).

        Parameters
        ----------
        frame : uint16 ndarray, shape (H, W).  Values in [0, 4095] (12-bit).
                H and W must be even (validated by ``_bayer_to_superpixel``).

        Returns
        -------
        uint16 ndarray, shape (~H//4, ~W//4) with even width.
        Values in [0, 4095].
        """
        mono = AcquisitionSession._bayer_to_superpixel(frame, phase)
        h, w = mono.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        m = mono[:h2, :w2].astype(np.float32)
        binned = (m[0::2, 0::2] + m[1::2, 0::2] + m[0::2, 1::2] + m[1::2, 1::2]) * np.float32(0.25)
        np.round(binned, out=binned)
        np.clip(binned, 0, 4095, out=binned)
        out = binned.astype(np.uint16)
        if out.shape[1] % 2 != 0 and out.shape[1] > 1:
            out = out[:, :-1]  # keep >= 1 column: never crop away the whole image
        return out

    @staticmethod
    def create_output_folder(base_folder: str, log_cb: LogCallback | None) -> str:
        """Create timestamped output folder."""
        os.makedirs(base_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(base_folder, timestamp)
        os.makedirs(folder, exist_ok=True)

        if log_cb:
            log_cb("info", f"Saving images to: {folder}")

        return folder

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------
    def run_sweep(
        self,
        cfg: dict,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
        preview_cb: PreviewCallback | None = None,
        cancel_flag: CancelFlag | None = None,
    ) -> str:
        """
        Perform Z-sweep.
        """
        positions_file = None
        try:
            # -------- VALIDATION --------
            if self._piezo is None or self._camera is None:
                msg = "Piezo or camera not connected."
                if log_cb:
                    log_cb("error", msg)
                raise RuntimeError(msg)

            start = float(cfg["start"])
            end = float(cfg["end"])
            step = float(cfg["step"])
            exposure = float(cfg["exposure"])
            timeout = float(cfg["timeout"])
            fmt = str(cfg.get("format", "tiff")).strip().lower()
            base_output = cfg.get("output_folder", "data")
            # Capture the colour mode ONCE for the whole sweep so a config
            # change mid-sweep can never produce a mixed mono/colour dataset.
            color_mode = str(cfg.get("color_mode", "mono")).strip().lower()
            self._color_mode = color_mode

            self._closed_loop = bool(cfg.get("closed_loop", True))
            self._axis = str(cfg.get("axis", "A"))
            self._settle_time = float(cfg.get("settle", 0.2))

            if step <= 0:
                msg = "STEP must be > 0"
                if log_cb:
                    log_cb("error", msg)
                raise ValueError(msg)

            if fmt not in ("tiff", "png", "bin12"):
                if log_cb:
                    log_cb("warn", f"Unknown format '{fmt}', defaulting to TIFF")
                fmt = "tiff"
            ext = {
                "tiff": ".tiff",
                "png": ".png",
                "bin12": ".bin12",
            }[fmt]

            # The camera object is captured ONCE for the whole sweep: if a
            # call hangs and the camera thread is abandoned, a later step of
            # this loop must never reach a camera reconnected meanwhile.
            cam = self._camera

            # -------- EXPOSURE --------
            # A wrong exposure would silently corrupt the whole dataset,
            # so failure to set (or verify) it aborts the sweep.
            try:
                cam.set_exposure(exposure)
            except Exception as e:
                msg = f"Could not set sweep exposure to {exposure:.6g} s: {e}"
                if log_cb:
                    log_cb("error", msg)
                raise RuntimeError(msg) from e

            try:
                actual = float(cam.get_exposure())
            except Exception as e:
                if log_cb:
                    log_cb("warn", f"Could not read back camera exposure: {e}")
            else:
                if abs(actual - exposure) > max(0.05 * exposure, 1e-4):
                    msg = (
                        f"Camera exposure verification failed: "
                        f"requested {exposure:.6g} s, camera reports {actual:.6g} s"
                    )
                    if log_cb:
                        log_cb("error", msg)
                    raise RuntimeError(msg)

            # -------- OUTPUT FOLDER --------
            output_folder = self.create_output_folder(base_output, log_cb)
            positions_file, positions_csv = self._open_positions_csv(output_folder, log_cb)

            n_steps = max(1, int(round((end - start) / step)) + 1)
            positions = np.linspace(start, end, n_steps)
            total = len(positions)
            t0 = time.time()
            save_errors = 0
            skipped_frames = 0
            saved_frames = 0
            consecutive_failures = 0
            consecutive_save_failures = 0  # disk full, folder gone (review H4)
            self.last_sweep_skipped_frames = 0
            self.last_sweep_total_frames = total
            self.last_sweep_aborted = None

            if log_cb:
                log_cb("info", f"Starting sweep: {total} steps from {start} to {end} µm")

            # -------- MAIN LOOP --------
            cancelled = False
            cancelled_at = 0
            aborted_reason: str | None = None
            aborted_at = 0
            for idx, z in enumerate(positions):
                if cancel_flag and cancel_flag.is_cancelled():
                    if log_cb:
                        log_cb("info", "Sweep cancelled by user.")
                    cancelled = True
                    cancelled_at = idx
                    break

                real_pos = self._move_piezo(z, log_cb=log_cb, cancel_flag=cancel_flag)

                # Re-check cancellation between moving and capturing: the move
                # wait polls the flag, so react here instead of snapping again.
                if cancel_flag and cancel_flag.is_cancelled():
                    if log_cb:
                        log_cb("info", "Sweep cancelled by user.")
                    cancelled = True
                    cancelled_at = idx
                    break

                # SNAP
                try:
                    # copy=False: snap() already returns uint16 on this camera,
                    # so avoid duplicating the full-resolution frame (O-3).
                    raw = np.asarray(cam.snap(timeout=timeout)).astype(np.uint16, copy=False)
                    t_frame = datetime.now()
                except Exception as e:
                    skipped_frames += 1
                    consecutive_failures += 1
                    if log_cb:
                        log_cb(
                            "error", f"Camera timeout/error at z={z:.4f} µm: {describe_error(e)}"
                        )
                    if consecutive_failures >= self.SWEEP_MAX_CONSECUTIVE_FAILURES:
                        aborted_reason = (
                            f"{consecutive_failures} consecutive camera failures "
                            f"(last at z={z:.4f} µm: {describe_error(e)})"
                        )
                        aborted_at = idx + 1  # positions attempted so far
                        break
                    continue
                consecutive_failures = 0

                # BAYER SUPERPIXEL MERGE (software, weighted or per-channel)
                try:
                    if color_mode == "color":
                        frame = self._bayer_to_color_superpixel(raw, self._bayer_phase)
                    elif color_mode == "mono_superpixel":
                        frame = self._bayer_to_superpixel_binned(raw, self._bayer_phase)
                    else:
                        frame = self._bayer_to_superpixel(raw, self._bayer_phase)
                except Exception as e:
                    skipped_frames += 1
                    if log_cb:
                        log_cb("error", f"Bayer superpixel conversion failed at z={z:.4f}: {e}")
                    continue

                # SAVE
                # Closed loop: the file keeps the COMMANDED z, as always (the
                # analysis reads z from the name; switching it to the sensor
                # reading would change results and is Miguel's call).  The
                # measured z goes to the progress / position indicator.
                tag = f"{z:.4f}um" if self._closed_loop else f"{real_pos:.4f}V"
                filename = f"piezo_{tag}_{idx:04d}{ext}"
                filepath = os.path.join(output_folder, filename)

                try:
                    if fmt == "bin12":
                        # _save_bin12 expects BGR for 3-channel input; our color
                        # superpixel is RGB, so reverse the last axis before passing.
                        save_frame = frame[:, :, ::-1] if frame.ndim == 3 else frame
                        self._save_bin12(save_frame, filepath)
                    else:
                        # cv2.imwrite expects BGR; mono frames need no conversion.
                        save_frame = frame[:, :, ::-1] if frame.ndim == 3 else frame
                        ok = cv2.imwrite(filepath, save_frame)
                        if not ok:
                            raise RuntimeError("cv2.imwrite returned False")
                    saved_frames += 1
                    consecutive_save_failures = 0
                    # One row per saved frame, flushed immediately: an
                    # aborted or cancelled sweep leaves a csv that matches
                    # the files on disk.
                    self._write_position_row(
                        positions_file,
                        positions_csv,
                        idx,
                        filename,
                        z,
                        real_pos,
                        t_frame,
                        log_cb,
                    )

                except Exception as e:
                    save_errors += 1
                    consecutive_save_failures += 1
                    if log_cb:
                        log_cb("error", f"Failed to save {filename}: {describe_error(e)}")
                    # Same rule as the camera: a full disk would otherwise
                    # walk every position without saving anything.
                    if consecutive_save_failures >= self.SWEEP_MAX_CONSECUTIVE_FAILURES:
                        aborted_reason = (
                            f"{consecutive_save_failures} consecutive save failures "
                            f"(last {filename}: {describe_error(e)})"
                        )
                        aborted_at = idx + 1
                        break

                # PREVIEW  (frame is RGB or mono — VM handles both)
                if preview_cb:
                    try:
                        preview_cb(frame, float(real_pos))
                    except Exception as e:
                        if log_cb:
                            log_cb("warn", f"Preview callback error: {e}")

                # PROGRESS
                if progress_cb:
                    elapsed = time.time() - t0
                    done = idx + 1
                    percent = (done / total) * 100.0
                    eta = (elapsed / done) * (total - done) if done > 0 else 0.0
                    try:
                        progress_cb(percent, float(real_pos), done, total, elapsed, eta)
                    except Exception as e:
                        if log_cb:
                            log_cb("warn", f"Progress callback error: {e}")

            # -------- POST-LOOP SUMMARY --------
            if cancelled:
                # Frames never attempted because of the cancel count as
                # skipped, so the completion report (and the UI's
                # incomplete-dataset warning) reflect the partial dataset
                # instead of looking like a normal, complete finish.
                skipped_frames += total - cancelled_at
                if log_cb:
                    log_cb(
                        "warn",
                        f"Sweep cancelled — partial dataset "
                        f"({cancelled_at}/{total} frames captured).",
                    )
            if aborted_reason is not None:
                # Positions never attempted count as skipped, like a cancel.
                skipped_frames += total - aborted_at
                summary = (
                    f"Sweep aborted after {aborted_reason}. "
                    f"{saved_frames} of {total} frame(s) saved in {output_folder}."
                )
                self.last_sweep_aborted = summary
                if log_cb:
                    log_cb("error", summary)
                self._write_abort_marker(output_folder, summary, saved_frames, total, log_cb)
            self.last_sweep_skipped_frames = skipped_frames
            self.last_sweep_total_frames = total

            if not cancelled and aborted_reason is None and log_cb:
                log_cb("info", "Sweep finished.")

            if skipped_frames > 0 and log_cb:
                log_cb(
                    "warn",
                    f"Incomplete dataset: {skipped_frames} of {total} frame(s) "
                    f"could not be acquired.",
                )

            if save_errors > 0 and log_cb:
                log_cb("warn", f"{save_errors} frame(s) could not be saved.")

            return output_folder

        except Exception as e:
            msg = f"Sweep aborted: {e}"
            if log_cb:
                log_cb("error", msg)
            raise
        finally:
            if positions_file is not None:
                with contextlib.suppress(Exception):
                    positions_file.close()

    def _open_positions_csv(self, folder: str, log_cb: LogCallback | None):
        """Open <folder>/positions.csv and write its header.

        Returns (file, csv.writer), or (None, None) if it cannot be opened:
        the log of the positions must never stop a sweep.
        """
        try:
            fh = open(  # noqa: SIM115 - closed in run_sweep's finally
                os.path.join(folder, self.POSITIONS_CSV), "w", newline="", encoding="utf-8"
            )
            writer = csv.writer(fh)
            writer.writerow(self.POSITIONS_HEADER)
            fh.flush()
        except OSError as e:
            if log_cb:
                log_cb("warn", f"Could not write {self.POSITIONS_CSV}: {describe_error(e)}")
            return None, None
        return fh, writer

    def _write_position_row(
        self,
        fh,
        writer,
        idx: int,
        filename: str,
        z_commanded: float,
        z_measured: float,
        when: datetime,
        log_cb: LogCallback | None,
    ) -> None:
        """Append one row and flush it (a few µs; the frame is already saved)."""
        if writer is None:
            return
        try:
            writer.writerow(
                [
                    idx,
                    filename,
                    f"{z_commanded:.4f}",
                    "nan" if math.isnan(z_measured) else f"{z_measured:.4f}",
                    when.isoformat(timespec="milliseconds"),
                ]
            )
            fh.flush()
        except (OSError, ValueError) as e:
            if log_cb:
                log_cb("warn", f"Could not append to {self.POSITIONS_CSV}: {describe_error(e)}")

    def _write_abort_marker(
        self,
        folder: str,
        summary: str,
        saved: int,
        total: int,
        log_cb: LogCallback | None,
    ) -> None:
        """Leave a plain-text note in the folder of an aborted sweep."""
        path = os.path.join(folder, self.ABORT_MARKER)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "INCOMPLETE DATASET: this sweep was aborted by InterferoLab.\n"
                    f"{summary}\n"
                    f"frames_saved = {saved}\n"
                    f"frames_planned = {total}\n"
                    f"aborted_at = {datetime.now().isoformat(timespec='seconds')}\n"
                )
        except OSError as e:
            if log_cb:
                log_cb("warn", f"Could not write {self.ABORT_MARKER}: {e}")

    # --------------------------------------------------------
    # Piezo movement
    # --------------------------------------------------------
    def _wait_on_target(
        self,
        dev: GCSDevice,
        timeout_s: float,
        cancel_flag: CancelFlag | None = None,
        position_cb: PositionCallback | None = None,
    ) -> bool:
        """
        Wait until the piezo reports "on target", polling qONT every 100 ms
        (fixed rate: measured from the start of one poll to the next).

        Cancellation-aware replacement for pitools.waitontarget: the cancel
        flag is checked on every poll, so a cancel request takes effect within
        one polling period instead of blocking for the full move timeout.

        Returns True when the target is reached, False if cancelled.
        Raises TimeoutError if the target is not reached within timeout_s,
        and RuntimeError if qONT keeps failing (the arrival cannot be
        confirmed: finding C11).  Only a controller without qONT at all
        (GCS "unknown command") falls back to a fixed settle-time wait.

        position_cb, when given, receives the measured position (qPOS) on
        every poll after the first one, so the window can follow a move that
        takes longer than one poll.  Not on the first poll: it comes right
        after MOV, when the stage has hardly moved, and a healthy small step
        is on target by the second one, so it costs nothing then.
        """
        deadline = time.monotonic() + timeout_s
        qont_failures = 0
        polls = 0
        while True:
            if cancel_flag is not None and cancel_flag.is_cancelled():
                return False
            # Fixed-rate polling: the next poll is due MOVE_POLL_S after this
            # one STARTS, so the time spent in the queries (qONT, and qPOS
            # since batch 3) comes out of the pause instead of adding to it.
            # That keeps a move as long as before although it now reads the
            # position at the end (sim: 153 ms both ways).
            next_poll = time.monotonic() + self.MOVE_POLL_S
            try:
                ont = dev.qONT(self._axis)
            except Exception as e:  # noqa: BLE001 - GCS, link and DLL errors
                if isinstance(e, GCSError) and e.val == self.GCS_UNKNOWN_COMMAND:
                    # No qONT on this controller: fall back to a time-based wait.
                    settle_deadline = min(deadline, time.monotonic() + self._settle_time)
                    while time.monotonic() < settle_deadline:
                        if cancel_flag is not None and cancel_flag.is_cancelled():
                            return False
                        time.sleep(self.MOVE_POLL_S)
                    return True
                qont_failures = self._qont_failed(qont_failures, e)
                ont = False  # unknown: poll again (bounded by the deadline)
            else:
                qont_failures = 0

            if isinstance(ont, dict):
                on_target = all(bool(v) for v in ont.values())
            elif isinstance(ont, (list, tuple)):
                on_target = all(bool(v) for v in ont)
            else:
                on_target = bool(ont)
            if on_target:
                return True

            polls += 1
            if position_cb is not None and polls > 1:
                position_cb(self._read_position(dev))

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Piezo did not reach target within {timeout_s:.1f} s")
            time.sleep(max(0.0, next_poll - time.monotonic()))

    def _qont_failed(self, failures: int, e: Exception) -> int:
        """Count one failed qONT; raise once too many came in a row.

        A transient comms error (e.g. GCS -7, DLL timeout) is polled again;
        a link that stays down means the arrival cannot be confirmed.
        """
        failures += 1
        if failures >= self.QONT_MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"Could not confirm that the piezo reached its target: "
                f"qONT failed {failures} times in a row ({describe_error(e)})"
            ) from e
        return failures

    def _read_position(self, dev: GCSDevice, log_cb: LogCallback | None = None) -> float:
        """Measured position of the axis (qPOS, µm), or NaN if it cannot be read.

        Never raises: a position that cannot be read is shown as unknown,
        which is the truth, instead of stopping whatever asked for it.
        """
        try:
            pos = dev.qPOS(self._axis)
            if isinstance(pos, dict):
                return float(pos[self._axis])
            if isinstance(pos, (list, tuple)):
                return float(pos[0])
            return float(pos)
        except Exception as e:  # noqa: BLE001 - link down, DLL error
            if log_cb:
                log_cb("warn", f"Could not read the piezo position: {describe_error(e)}")
            return float("nan")

    def _move_piezo(
        self,
        z: float,
        log_cb: LogCallback | None = None,
        cancel_flag: CancelFlag | None = None,
        position_cb: PositionCallback | None = None,
    ) -> float:
        """Move and return the MEASURED position (NaN if unreadable).

        Closed loop: the position is read with qPOS when the move ends, also
        when it fails or is cancelled: the MOV may have been executed even if
        the arrival was never confirmed (finding C3), so the window must show
        where the stage really is, not the target and not the old position.
        """
        if self._piezo is None:
            raise RuntimeError("_move_piezo called but piezo is not connected")
        dev = self._piezo

        if self._closed_loop:
            try:
                dev.MOV(self._axis, float(z))
                self._wait_on_target(
                    dev, self.MOVE_TIMEOUT_S, cancel_flag=cancel_flag, position_cb=position_cb
                )
            except Exception as e:
                if log_cb:
                    log_cb("error", f"Closed-loop move to {z:.4f} failed: {e}")
                self._last_real_pos = self._read_position(dev, log_cb)
                if position_cb is not None:
                    position_cb(self._last_real_pos)
                raise
            real = self._read_position(dev, log_cb)
            self._last_real_pos = real
            if position_cb is not None:
                position_cb(real)
            return real

        try:
            dev.SVA(self._axis, float(z))
            time.sleep(self._settle_time)

            try:
                v = dev.qVOL(self._axis)
                if isinstance(v, dict):
                    real = float(v[self._axis])
                elif isinstance(v, (list, tuple)):
                    real = float(v[0])
                else:
                    real = float(z)
            except Exception:
                real = float(z)

            self._last_real_pos = real
            return real

        except Exception as e:
            if log_cb:
                log_cb("error", f"Open-loop move to {z:.4f} failed: {e}")
            raise

    def move_to(
        self,
        z: float,
        log_cb: LogCallback | None = None,
        cancel_flag: CancelFlag | None = None,
        position_cb: PositionCallback | None = None,
    ) -> float:
        # cancel_flag makes manual moves abortable like sweep moves:
        # _wait_on_target polls it, so shutdown() can cancel a move that
        # would otherwise legally block up to MOVE_TIMEOUT_S (30 s).
        # position_cb follows the measured position live, from this thread.
        return self._move_piezo(z, log_cb, cancel_flag=cancel_flag, position_cb=position_cb)

    # --------------------------------------------------------
    # Single preview frame (snapshot while the stream is off)
    # --------------------------------------------------------
    def capture_preview(
        self,
        timeout: float = 5.0,
        log_cb: LogCallback | None = None,
    ) -> tuple[np.ndarray, float]:
        """One frame through snap(): arm, expose, read, disarm (~0.5 s).

        Used for the single picture after connecting; the live preview uses
        the stream primitives instead.  Camera thread only when it is alive.
        """
        if self._camera is None:
            msg = "Camera not connected."
            if log_cb:
                log_cb("error", msg)
            raise RuntimeError(msg)

        try:
            raw = np.asarray(self._camera.snap(timeout=timeout)).astype(np.uint16)
            frame = self._merge(raw)
            z = float(self._last_real_pos)
            return frame, z
        except Exception as e:
            if log_cb:
                log_cb("error", f"Error capturing preview frame: {describe_error(e)}")
            raise
