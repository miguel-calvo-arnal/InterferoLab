# backend/acquisition/acquisition_controller.py
from __future__ import annotations

import atexit
import os
import time
from collections.abc import Callable
from datetime import datetime

import cv2
import numpy as np
from pipython import GCSDevice, GCSError
from pylablib.devices import Thorlabs

from utils import bin12
from utils.camera_constants import BAYER_W_B, BAYER_W_G1, BAYER_W_G2, BAYER_W_R

# --------------------------------------------------------
# Callback type aliases
# --------------------------------------------------------
ProgressCallback = Callable[[float, float, int, int, float, float], None]
LogCallback = Callable[[str, str], None]
PreviewCallback = Callable[[np.ndarray, float], None]


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

    def __init__(self) -> None:
        self._piezo: GCSDevice | None = None
        self._camera: Thorlabs.ThorlabsTLCamera | None = None
        self._closed_loop: bool = True
        self._axis: str = "A"
        self._settle_time: float = 0.2

        # Remember last real position/voltage for live preview
        self._last_real_pos: float = 0.0

        # "mono" → weighted single channel; "color" → R/G/B channels separate;
        # "mono_superpixel" → weighted single channel + extra 2×2 binning
        self._color_mode: str = "mono"

        # Statistics of the last completed sweep (frames skipped vs planned)
        self.last_sweep_skipped_frames: int = 0
        self.last_sweep_total_frames: int = 0

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
        """Connect Thorlabs camera, apply RAW/Binning and set exposure."""
        try:
            if self._camera is not None:
                try:
                    self._camera.set_exposure(exposure_s)
                    if log_cb:
                        log_cb("info", f"Camera exposure updated to {exposure_s:.4g} s")
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
                try:
                    available_formats = cam.get_all_pixel_formats()
                    if "Mono16" in available_formats:
                        cam.set_pixel_format("Mono16")
                    elif "Mono12" in available_formats:
                        cam.set_pixel_format("Mono12")
                    elif "Mono8" in available_formats:
                        cam.set_pixel_format("Mono8")
                    if log_cb:
                        fmt_name = cam.get_pixel_format()
                        log_cb("info", f"Camera pixel format set to {fmt_name} (raw Bayer output).")
                except Exception as e:
                    if log_cb:
                        log_cb("warn", f"Could not set mono pixel format: {e}")

                # Ensure no hardware binning is active (reset to 1×1 explicitly).
                try:
                    cam.set_roi(hbin=1, vbin=1)
                except Exception as e:
                    if log_cb:
                        log_cb("warn", f"Could not reset hardware binning to 1×1: {e}")
                # ---------------------------------------------------------

                cam.set_exposure(exposure_s)

                for _ in range(3):
                    try:
                        cam.start_acquisition()
                        break
                    except Exception:
                        time.sleep(0.2)
                else:
                    msg = "Camera failed to start acquisition"
                    if log_cb:
                        log_cb("error", msg)
                    raise RuntimeError(msg)

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

    def disconnect_all(self, log_cb: LogCallback | None = None) -> bool:
        """
        Close camera and piezo safely.

        Each device is shut down independently, so a failure on one never
        prevents releasing the other. Idempotent: safe to call repeatedly
        (also used as an atexit safety net).
        """
        if self._camera is not None:
            cam = self._camera
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

        if self._piezo is not None:
            dev = self._piezo
            self._piezo = None
            try:
                if self._closed_loop:
                    dev.MOV(self._axis, 0.0)
                    # Use a conservative timeout to avoid infinite hang on disconnect
                    self._wait_on_target(dev, timeout_s=10.0)
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

        return True

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
    def _bayer_to_color_superpixel(frame: np.ndarray) -> np.ndarray:
        """
        Convert a full-resolution Bayer RGGB frame to a 3-channel superpixel image.

        Each 2×2 Bayer block  [ R  G1 ]  is collapsed to (R, G, B) where
                               [ G2  B ]
            G = (G1 + G2) / 2

        No spectral weighting is applied so each channel retains its independent
        signal.  This lets the caller inspect or discard individual channels when
        one of them is known to be corrupted (e.g. partial CFA damage or heavy
        noise on a single colour).

        Parameters
        ----------
        frame : uint16 ndarray, shape (H, W).  Values in [0, 4095] (12-bit).
                H and W must be even.

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

        f = frame.astype(np.float32)
        R = f[0::2, 0::2]
        G1 = f[0::2, 1::2]
        G2 = f[1::2, 0::2]
        B = f[1::2, 1::2]

        G = (G1 + G2) * 0.5
        out = np.stack([R, G, B], axis=2)
        return np.clip(np.round(out), 0, 4095).astype(np.uint16)

    @staticmethod
    def _bayer_to_superpixel(frame: np.ndarray) -> np.ndarray:
        """
        Convert a full-resolution Bayer RGGB frame to a monochrome superpixel image.

        Each 2×2 Bayer block  [ R  G1 ]  is collapsed to a single value:
                               [ G2  B ]

            superpixel = w_R·R + w_G1·G1 + w_G2·G2 + w_B·B

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
        R = frame[0::2, 0::2]  # RGGB: even-row, even-col
        G1 = frame[0::2, 1::2]  # even-row, odd-col
        G2 = frame[1::2, 0::2]  # odd-row,  even-col
        B = frame[1::2, 1::2]  # odd-row,  odd-col

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
    def _bayer_to_superpixel_binned(frame: np.ndarray) -> np.ndarray:
        """
        Convert a full-resolution Bayer RGGB frame to a small monochrome image.

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
        mono = AcquisitionSession._bayer_to_superpixel(frame)
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

            # -------- EXPOSURE --------
            # A wrong exposure would silently corrupt the whole dataset,
            # so failure to set (or verify) it aborts the sweep.
            try:
                self._camera.set_exposure(exposure)
            except Exception as e:
                msg = f"Could not set sweep exposure to {exposure:.6g} s: {e}"
                if log_cb:
                    log_cb("error", msg)
                raise RuntimeError(msg) from e

            try:
                actual = float(self._camera.get_exposure())
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

            n_steps = max(1, int(round((end - start) / step)) + 1)
            positions = np.linspace(start, end, n_steps)
            total = len(positions)
            t0 = time.time()
            save_errors = 0
            skipped_frames = 0
            self.last_sweep_skipped_frames = 0
            self.last_sweep_total_frames = total

            if log_cb:
                log_cb("info", f"Starting sweep: {total} steps from {start} to {end} µm")

            # -------- MAIN LOOP --------
            cancelled = False
            cancelled_at = 0
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
                    raw = np.asarray(self._camera.snap(timeout=timeout)).astype(
                        np.uint16, copy=False
                    )
                except Exception as e:
                    skipped_frames += 1
                    if log_cb:
                        log_cb("error", f"Camera timeout/error at z={z:.4f} µm: {e}")
                    continue

                # BAYER SUPERPIXEL MERGE (software, weighted or per-channel)
                try:
                    if color_mode == "color":
                        frame = self._bayer_to_color_superpixel(raw)
                    elif color_mode == "mono_superpixel":
                        frame = self._bayer_to_superpixel_binned(raw)
                    else:
                        frame = self._bayer_to_superpixel(raw)
                except Exception as e:
                    skipped_frames += 1
                    if log_cb:
                        log_cb("error", f"Bayer superpixel conversion failed at z={z:.4f}: {e}")
                    continue

                # SAVE
                tag = f"{real_pos:.4f}um" if self._closed_loop else f"{real_pos:.4f}V"
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

                except Exception as e:
                    save_errors += 1
                    if log_cb:
                        log_cb("error", f"Failed to save {filename}: {e}")

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
            self.last_sweep_skipped_frames = skipped_frames
            self.last_sweep_total_frames = total

            if not cancelled and log_cb:
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

    # --------------------------------------------------------
    # Piezo movement
    # --------------------------------------------------------
    def _wait_on_target(
        self,
        dev: GCSDevice,
        timeout_s: float,
        cancel_flag: CancelFlag | None = None,
    ) -> bool:
        """
        Wait until the piezo reports "on target", polling qONT every ~100 ms.

        Cancellation-aware replacement for pitools.waitontarget: the cancel
        flag is checked on every poll, so a cancel request takes effect within
        one polling period instead of blocking for the full move timeout.

        Returns True when the target is reached, False if cancelled.
        Raises TimeoutError if the target is not reached within timeout_s.
        If qONT is unavailable, falls back to a fixed settle-time wait.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            if cancel_flag is not None and cancel_flag.is_cancelled():
                return False
            try:
                ont = dev.qONT(self._axis)
            except Exception:
                # qONT not supported/failing: fall back to a time-based wait.
                settle_deadline = min(deadline, time.monotonic() + self._settle_time)
                while time.monotonic() < settle_deadline:
                    if cancel_flag is not None and cancel_flag.is_cancelled():
                        return False
                    time.sleep(self.MOVE_POLL_S)
                return True

            if isinstance(ont, dict):
                on_target = all(bool(v) for v in ont.values())
            elif isinstance(ont, (list, tuple)):
                on_target = all(bool(v) for v in ont)
            else:
                on_target = bool(ont)
            if on_target:
                return True

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Piezo did not reach target within {timeout_s:.1f} s")
            time.sleep(self.MOVE_POLL_S)

    def _move_piezo(
        self,
        z: float,
        log_cb: LogCallback | None = None,
        cancel_flag: CancelFlag | None = None,
    ) -> float:
        if self._piezo is None:
            raise RuntimeError("_move_piezo called but piezo is not connected")
        dev = self._piezo

        if self._closed_loop:
            try:
                dev.MOV(self._axis, float(z))
                self._wait_on_target(dev, self.MOVE_TIMEOUT_S, cancel_flag=cancel_flag)
                real = float(z)
                self._last_real_pos = real
                return real
            except Exception as e:
                if log_cb:
                    log_cb("error", f"Closed-loop move to {z:.4f} failed: {e}")
                raise

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
    ) -> float:
        # cancel_flag makes manual moves abortable like sweep moves:
        # _wait_on_target polls it, so shutdown() can cancel a move that
        # would otherwise legally block up to MOVE_TIMEOUT_S (30 s).
        return self._move_piezo(z, log_cb, cancel_flag=cancel_flag)

    # --------------------------------------------------------
    # Single preview frame
    # --------------------------------------------------------
    def capture_preview(
        self,
        timeout: float = 5.0,
        log_cb: LogCallback | None = None,
    ) -> tuple[np.ndarray, float]:

        if self._camera is None:
            msg = "Camera not connected."
            if log_cb:
                log_cb("error", msg)
            raise RuntimeError(msg)

        try:
            raw = np.asarray(self._camera.snap(timeout=timeout)).astype(np.uint16)
            if self._color_mode == "color":
                frame = self._bayer_to_color_superpixel(raw)
            elif self._color_mode == "mono_superpixel":
                frame = self._bayer_to_superpixel_binned(raw)
            else:
                frame = self._bayer_to_superpixel(raw)
            z = float(self._last_real_pos)
            return frame, z
        except Exception as e:
            if log_cb:
                log_cb("error", f"Error capturing preview frame: {e}")
            raise
