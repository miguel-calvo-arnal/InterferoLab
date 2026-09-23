"""
[SIMULATION] Fake of pylablib.devices.Thorlabs.TLCamera (pylablib 1.4.5).

Same public names, signatures, return types and exceptions as the real
module for everything InterferoLab uses, and the same *control flow* for
acquisition, including pylablib's fixed sleeps:

* ``start_acquisition``: stop (sleep 0.2 s + disarm if armed) -> setup the
  ring buffer (clamped to 85 frames at 4096x3000) -> arm -> sleep 0.05 s ->
  software trigger.  Continuous mode: frames keep arriving at the frame
  period until disarm, and each one is copied into the buffer like the
  SDK callback does (``np.ctypeslib.as_array(...).copy()`` in pylablib).
* ``snap`` = ``grab(1)``: start_acquisition, wait for the first frame,
  read it, and ALWAYS ``stop_acquisition`` (sleep 0.2 s + disarm).
* A timeout raises ``ThorlabsTLCameraTimeoutError()`` with an empty message.

SDK costs (open, arm, disarm, readout...) come from the profile
(sim/profiles/*.toml), never from constants here; the two sleeps and the
buffer clamp below are pylablib's own code, reproduced verbatim.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import threading
import time

import numpy as np

from sim.world import get_world

from .tl_camera_sdk_lib import ThorlabsTLCameraError, ThorlabsTLCameraLibError, sim_lib_error

log = logging.getLogger("sim.camera")

# --- pylablib 1.4.5 code constants (TLCamera.py) ---------------------------
_SLEEP_BEFORE_TRIGGER_S = 0.05  # TLCamera.start_acquisition, line 493
_SLEEP_BEFORE_DISARM_S = 0.2  # TLCamera.stop_acquisition, line 497
_MAX_FRAME_BYTES = 2000 * 2**20  # TLCamera._max_frame_bytes
_ADDED_FRAME_SIZE = 2**15  # TLCamera._added_frame_size
_DEFAULT_NFRAMES = 100  # grab()/setup_acquisition default buffer size


class ThorlabsTLCameraTimeoutError(ThorlabsTLCameraError):
    "TLCamera frame timeout error"


TDeviceInfo = collections.namedtuple(
    "TDeviceInfo", ["model", "name", "serial_number", "firmware_version"]
)
TSensorInfo = collections.namedtuple("TSensorInfo", ["sensor_type", "bit_depth"])
TColorInfo = collections.namedtuple(
    "TColorInfo", ["filter_array_phase", "correction_matrix", "default_white_balance_matrix"]
)
TColorFormat = collections.namedtuple("TColorFormat", ["color_format", "color_space"])
TFrameInfo = collections.namedtuple(
    "TFrameInfo", ["frame_index", "framestamp", "pixelclock", "pixeltype", "offset"]
)
TAcqTimings = collections.namedtuple("TAcqTimings", ["exposure", "frame_period"])

#: One entry of ThorlabsTLCamera.frame_log (simulator-only observation point):
#: seq = frames delivered since the camera was opened (never resets; use it to
#: order frames), index = frame index inside the current acquisition (what
#: read_multiple_images reports; resets at every setup), t_* = time.monotonic()
#: of the exposure start / mid-exposure / frame ready in the buffer, z_um =
#: stage position sampled at mid-exposure, thumb = small float32 summary or None.
FrameRecord = collections.namedtuple(
    "FrameRecord", ["seq", "index", "t_start", "t_mid", "t_ready", "z_um", "exposure_s", "thumb"]
)

THUMB_SHAPE = (48, 64)


def area_resize(img: np.ndarray, shape=THUMB_SHAPE) -> np.ndarray:
    """rows x cols float32 area average of the WHOLE image (grid aligned to its edges).

    Both sides of a content match (raw frame here, displayed image in the
    perf gate) must use this same function so their grids coincide.
    """
    rows, cols = shape
    src = np.ascontiguousarray(img, dtype=np.float32)
    try:
        import cv2

        return cv2.resize(src, (cols, rows), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001 - numpy fallback: block means of the largest aligned crop
        h = (src.shape[0] // rows) * rows
        w = (src.shape[1] // cols) * cols
        return src[:h, :w].reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))


def frame_thumbnail(frame: np.ndarray, shape=THUMB_SHAPE) -> np.ndarray:
    """Small float32 summary of a raw frame (all four photosites mixed).

    Cheap (strided reads + one small resize, ~1 ms): meant for matching what
    a display shows with the frame the camera delivered, not for photometry.
    """
    f = frame
    sub = (
        f[0::8, 0::8].astype(np.float32)
        + f[1::8, 0::8]
        + f[0::8, 1::8]
        + f[1::8, 1::8]
    )
    return area_resize(sub, shape)

_COLOR_OUTPUTS = ("raw", "rgb", "grayscale", "auto")
_COLOR_SPACES = ("srgb", "linear")


def list_cameras():
    """List connected TLCamera cameras"""
    w = get_world()
    if w.faults.camera_unplugged():
        return []
    return [str(w.profile["camera.serial"])]


def get_cameras_number():
    """Get number of connected TLCamera cameras"""
    return len(list_cameras())


class _Buffer:
    """Ring buffer + frame counter (pylablib's RingBuffer/FrameNotifier)."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.frames: list | None = None
        self.size = 0
        self.acquired = 0  # frames received since the last start
        self.last_read = 0

    def setup(self, size: int) -> None:
        with self.cond:
            self.frames = []
            self.size = max(size, 1)
            self.acquired = 0
            self.last_read = 0
            self.cond.notify_all()

    def cleanup(self) -> None:
        with self.cond:
            self.frames = None
            self.size = 0
            self.acquired = 0
            self.last_read = 0
            self.cond.notify_all()

    def append(self, frame) -> None:
        with self.cond:
            if self.frames is None:
                return
            self.frames.append(frame)
            if len(self.frames) > self.size:
                del self.frames[: len(self.frames) - self.size]
            self.acquired += 1
            self.cond.notify_all()

    def get(self, idx: int):
        """Frame with absolute index idx (0-based) or None if overwritten."""
        with self.cond:
            if self.frames is None:
                return None
            back = self.acquired - idx
            if 0 < back <= len(self.frames):
                return self.frames[-back]
            return None


class ThorlabsTLCamera:
    """
    Thorlabs TSI camera.

    Args:
        serial(str): camera serial number; can be either a string obtained using :func:`list_cameras` function,
            or ``None``, which means connecting to the first available camera (not recommended unless only one camera is connected)
    """

    Error = ThorlabsTLCameraError
    TimeoutError = ThorlabsTLCameraTimeoutError
    _TFrameInfo = TFrameInfo
    __simulated__ = True

    def __init__(self, serial=None):
        self.serial = str(serial) if isinstance(serial, int) else serial
        self.handle = None
        self._world = get_world()
        p = self._world.profile
        self._p = p
        t = "camera.timing."
        self._t_open = float(p[t + "open_s"])
        self._t_close = float(p[t + "close_s"])
        self._t_call = float(p[t + "sdk_call_s"])
        self._t_arm = float(p[t + "arm_s"])
        self._t_disarm = float(p[t + "disarm_s"])
        self._t_trig = float(p[t + "trigger_latency_s"])
        self._t_readout = float(p[t + "readout_transfer_s"])
        self._t_min_period = float(p[t + "min_frame_period_s"])
        self._exp_min = float(p["camera.sensor.exposure_min_s"])
        self._exp_max = float(p["camera.sensor.exposure_max_s"])
        self._height = int(p["camera.sensor.height"])
        self._width = int(p["camera.sensor.width"])
        self._bit_depth = int(p["camera.sensor.bit_depth"])
        self._regenerate = bool(p.get("camera.regenerate_every_frame", False))
        self._buffer = _Buffer()
        self._armed = False
        self._acq_nframes = None
        self._producer: threading.Thread | None = None
        self._stop_evt: threading.Event | None = None
        self._next_acq_times_out = False
        self._in_grab = False  # start_acquisition called by grab()/snap(), not a stream
        self._exposure_us = 10000  # 10 ms until set (camera default unknown)
        self._roi = (0, self._width, 0, self._height, 1, 1)
        self._color_output = "raw"
        self._color_space = "linear"
        self._white_balance_matrix = np.eye(3)
        self._color_info = None
        self._gen_lock = threading.Lock()
        self._warned_slow = False
        # Stable observation point for tools (perf gate, tests): every frame
        # delivered while armed, newest last.  Thumbnails cost ~1 ms each and
        # are only computed when a tool sets frame_log_thumbnails = True.
        self.frame_log: collections.deque = collections.deque(maxlen=512)
        self.frame_log_thumbnails = False
        self.frames_delivered = 0  # since open, never resets
        # frames whose synthetic generation took longer than the readout budget
        # (the SIMULATOR delayed them): the perf gate reports this
        self.frames_delayed_by_generation = 0
        self._world.cameras.append(self)
        self.open()

    # ------------------------------------------------------------------
    # SDK plumbing
    # ------------------------------------------------------------------
    def _sdk(self, func: str, cost: float | None = None) -> None:
        """One SDK call: costs time, fails if the camera was unplugged."""
        if self.handle is None:
            raise sim_lib_error(func, 1, "camera handle is closed [SIMULATION]")
        if self._world.faults.camera_unplugged():
            raise sim_lib_error(func, 1004, "device disconnected [SIMULATION]")
        time.sleep(self._t_call if cost is None else cost)

    def _get_connection_parameters(self):
        return self.serial

    def open(self):
        """Open connection to the camera"""
        if self.handle is not None:
            return
        lst = list_cameras()
        if self.serial is None:
            self.serial = lst[0] if lst else ""
        elif self.serial not in lst:
            raise ThorlabsTLCameraError(
                f"camera with serial number {self.serial} isn't present among available cameras: {lst}"
            )
        if not self.serial:
            raise sim_lib_error("tl_camera_open_camera", 1004, "no camera found [SIMULATION]")
        t0 = time.monotonic()
        self._world.source()  # build the image model inside the open time
        remaining = self._t_open - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)
        self.handle = id(self)
        self.set_color_format()
        self.set_white_balance_matrix()
        log.warning(
            "[SIMULATION] Fake Thorlabs camera connected: model %s, serial %s (profile %s)",
            self._p["camera.model"],
            self.serial,
            ", ".join(self._p.files),
        )

    def close(self):
        """Close connection to the camera"""
        if self.handle is not None:
            try:
                with contextlib.suppress(ThorlabsTLCameraError):
                    self.clear_acquisition()
                time.sleep(self._t_close)
            finally:
                self.handle = None
                log.warning(
                    "[SIMULATION] Fake camera %s closed (%s)",
                    self.serial,
                    self._world.frame_stats(),
                )

    def is_opened(self):
        """Check if the device is connected"""
        return self.handle is not None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_device_info(self):
        """
        Get camera model data.

        Return tuple ``(model, name, serial_number, firmware_version)``.
        """
        self._sdk("tl_camera_get_model", 4 * self._t_call)
        return TDeviceInfo(str(self._p["camera.model"]), "SIM camera", str(self.serial), "SIM-1.0")

    def get_sensor_info(self):
        """Get camera sensor info: ``(sensor_type, bit_depth)``."""
        return TSensorInfo("bayer", self._bit_depth)

    def get_color_info(self):
        """
        Get camera color info.

        Return tuple ``(filter_array_phase, correction_matrix, default_white_balance_matrix)``, or ``None`` if the sensor type is not ``"bayer"``.
        """
        if self.get_sensor_info().sensor_type != "bayer":
            return None
        if self._color_info is None:
            self._sdk("tl_camera_get_color_filter_array_phase", 3 * self._t_call)
            self._color_info = TColorInfo(self._world.phase, np.eye(3), np.eye(3))
        return self._color_info

    def get_white_balance_matrix(self):
        """Get the white balance matrix"""
        return self._white_balance_matrix

    def set_white_balance_matrix(self, matrix=None):
        """Set the white balance matrix (``None``, 3 multipliers, or 3x3)."""
        if matrix is None:
            matrix = self.get_color_info().default_white_balance_matrix
        elif np.ndim(matrix) == 1:
            matrix = np.diag(matrix)
        self._white_balance_matrix = matrix

    def set_color_format(self, color_output="auto", color_space="linear"):
        """
        Set camera color format.

        `color_output` is ``"raw"``, ``"rgb"``, ``"grayscale"`` or ``"auto"`` (``"rgb"`` on color cameras);
        `color_space` is ``"linear"`` or ``"srgb"``.
        """
        # pylablib 1.4.5 does NOT validate these values (unknown ones are stored as is).
        cinfo = self.get_color_info()
        if color_output == "rgb" and cinfo is None:
            raise ValueError("'rgb' color mode is only supported on color cameras")
        if color_output == "auto":
            color_output = "rgb" if cinfo is not None else "raw"
        self._color_output = color_output
        self._color_space = color_space
        return self.get_color_format()

    def get_color_format(self):
        """Get camera color format as a tuple ``(color_output, color_space)``"""
        return TColorFormat(self._color_output, self._color_space)

    def get_trigger_mode(self):
        """Trigger mode: always ``"int"`` (software) in the simulator."""
        return "int"

    def get_timestamp_clock_frequency(self):
        return None

    # ------------------------------------------------------------------
    # Exposure and timings
    # ------------------------------------------------------------------
    def set_exposure(self, exposure):
        """Set camera exposure"""
        self._sdk("tl_camera_set_exposure_time")
        f = self._world.faults.p
        if not bool(f.get("faults.exposure_ignored", False)):
            us = int(exposure / 1e-6)  # pylablib truncates to integer microseconds
            us = min(max(us, int(round(self._exp_min / 1e-6))), int(self._exp_max / 1e-6))
            factor = float(f.get("faults.exposure_apply_factor", 1.0))
            self._exposure_us = max(1, int(us * factor))
        return self.get_exposure()

    def get_exposure(self):
        """Get current exposure"""
        return self.get_frame_timings().exposure

    def _frame_period(self) -> float:
        return max(self._t_min_period, self._exposure_us * 1e-6)

    def get_frame_timings(self):
        self._sdk("tl_camera_get_exposure_time", 2 * self._t_call)
        return TAcqTimings(self._exposure_us * 1e-6, self._frame_period())

    def get_frame_period(self):
        return self.get_frame_timings().frame_period

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------
    def get_detector_size(self):
        return self._width, self._height

    def get_roi(self):
        return self._roi

    def set_roi(self, hstart=0, hend=None, vstart=0, vend=None, hbin=1, vbin=1):
        # @camera.acqcleared: stop + clear, apply, restart if it was running
        was_running = self.acquisition_in_progress()
        params = self._acq_nframes
        self.stop_acquisition()
        self.clear_acquisition()
        try:
            self._sdk("tl_camera_set_roi", 6 * self._t_call)
            hend = self._width if hend is None else min(int(hend), self._width)
            vend = self._height if vend is None else min(int(vend), self._height)
            hstart = max(0, min(int(hstart), hend - 2)) & ~1
            vstart = max(0, min(int(vstart), vend - 2)) & ~1
            hend -= (hend - hstart) % 2
            vend -= (vend - vstart) % 2
            # binning is not simulated: the fake keeps 1x1 (the app asks for 1x1)
            self._roi = (hstart, hend, vstart, vend, 1, 1)
            return self.get_roi()
        finally:
            if was_running:
                self.start_acquisition(nframes=params)

    def _get_data_dimensions_rc(self):
        h0, h1, v0, v1 = self._roi[:4]
        return v1 - v0, h1 - h0

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def acquisition_in_progress(self):
        return self._armed

    def get_acquisition_parameters(self):
        return {"nframes": self._acq_nframes} if self._acq_nframes else None

    def setup_acquisition(self, nframes=100):
        """Setup acquisition: ring buffer of `nframes` (clamped by frame size, like pylablib)."""
        r, c = self._get_data_dimensions_rc()
        frame_nbytes = r * c * 2 + _ADDED_FRAME_SIZE
        nframes = min(int(_MAX_FRAME_BYTES / frame_nbytes), nframes)
        self._acq_nframes = nframes
        self._buffer.setup(nframes + 10)

    def clear_acquisition(self):
        self.stop_acquisition()
        self._buffer.cleanup()
        self._acq_nframes = None

    def start_acquisition(self, frames_per_trigger="default", auto_start=True, nframes=None):
        """
        Start camera acquisition.

        Args:
            frames_per_trigger: frames per trigger; ``None`` means unlimited (default for the software trigger)
            auto_start: if ``True``, send the software trigger right away (after pylablib's 0.05 s sleep)
            nframes: number of frames in the ring buffer
        """
        self.stop_acquisition()
        self.setup_acquisition(nframes=nframes or _DEFAULT_NFRAMES)
        if frames_per_trigger == "default":
            frames_per_trigger = None if self.get_trigger_mode() == "int" else 1
        self._frames_per_trigger = frames_per_trigger or 0
        # tl_camera_arm(handle, max(nframes, 10))
        self._sdk("tl_camera_arm", self._t_arm)
        self._armed = True
        self._stop_evt = threading.Event()
        if not self._in_grab:
            self._world.faults.stream_armed()
        if auto_start:
            time.sleep(_SLEEP_BEFORE_TRIGGER_S)
            self.send_software_trigger()

    def send_software_trigger(self):
        """Send software trigger signal"""
        if not self._armed:
            return
        self._sdk("tl_camera_issue_software_trigger")
        if self._producer is not None and self._producer.is_alive():
            return  # continuous mode already running
        fail = self._next_acq_times_out
        self._next_acq_times_out = False
        self._producer = threading.Thread(
            target=self._produce,
            args=(time.monotonic(), self._exposure_us * 1e-6, self._stop_evt, fail, not self._in_grab),
            name="sim-camera-sdk",
            daemon=True,
        )
        self._producer.start()

    def stop_acquisition(self):
        if self.acquisition_in_progress():
            time.sleep(_SLEEP_BEFORE_DISARM_S)  # pylablib: "seems to improve code stability"
            try:
                self._sdk("tl_camera_disarm", self._t_disarm)
            finally:
                self._armed = False
                if self._stop_evt is not None:
                    self._stop_evt.set()
                if self._producer is not None:
                    self._producer.join(timeout=5.0)
                self._producer = None

    def _produce(
        self,
        t_trigger: float,
        exposure: float,
        stop: threading.Event,
        fail: bool,
        stream: bool = False,
    ) -> None:
        """The camera + SDK callback thread: delivers frames while armed."""
        if fail:
            stop.wait()  # simulated fault: the frame never arrives
            return
        world = self._world
        period = max(self._t_min_period, exposure)
        h0, h1, v0, v1 = self._roi[:4]
        full = (v1 - v0, h1 - h0) == world.sensor_shape
        last = None
        z = None
        k = 0
        try:
            while True:
                if stream and world.faults.stream_frame_blocked():
                    stop.wait()  # simulated stream stall: armed, but no more frames
                    return
                t_start = t_trigger + self._t_trig + k * period
                t_mid = t_start + 0.5 * exposure
                t_ready = t_start + exposure + self._t_readout
                if stop.wait(max(0.0, t_mid - time.monotonic())):
                    return
                if last is None or self._regenerate:
                    z = world.stage.position(t_mid)
                    t_gen = time.monotonic()
                    with self._gen_lock:
                        frame = world.source().generate(z, exposure)
                    t_gen = time.monotonic() - t_gen
                    if t_gen > 0.5 * exposure + self._t_readout:
                        self.frames_delayed_by_generation += 1
                    if t_gen > 0.5 * exposure + self._t_readout and not self._warned_slow:
                        self._warned_slow = True
                        log.warning(
                            "[SIMULATION] frame generation took %.0f ms, longer than the camera's "
                            "readout budget (%.0f ms): the SIMULATOR is delaying this frame",
                            t_gen * 1000,
                            (0.5 * exposure + self._t_readout) * 1000,
                        )
                    if not full:
                        frame = np.ascontiguousarray(frame[v0:v1, h0:h1])
                else:
                    frame = last.copy()  # the SDK callback copies every frame
                if stop.wait(max(0.0, t_ready - time.monotonic())):
                    return
                self._buffer.append(frame)
                self.frames_delivered += 1
                self.frame_log.append(
                    FrameRecord(
                        self.frames_delivered,
                        self._buffer.acquired - 1,
                        t_start,
                        t_mid,
                        time.monotonic(),
                        z,
                        exposure,
                        frame_thumbnail(frame) if self.frame_log_thumbnails else None,
                    )
                )
                last = frame
                k += 1
        except Exception:  # noqa: BLE001 - a dead producer shows up as a snap timeout
            log.exception("[SIMULATION] camera frame producer failed")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def _get_acquired_frames(self):
        return self._buffer.acquired if self._buffer.frames is not None else None

    def wait_for_frame(self, since="lastread", nframes=1, timeout=20.0, error_on_stopped=False):
        """
        Wait for one or several new camera frames.

        If the call times out, raise ``TimeoutError`` (the camera's, with no message).
        """
        if isinstance(timeout, tuple):
            timeout = timeout[0]
        if not self.acquisition_in_progress():
            if error_on_stopped:
                raise self.Error("waiting for a frame while acquisition is stopped")
            return False
        buf = self._buffer
        with buf.cond:
            if since == "lastread":
                target = buf.last_read + nframes
            elif since in ("now", "lastwait"):
                target = buf.acquired + nframes
            else:  # "start"
                target = nframes
            deadline = None if timeout is None else time.monotonic() + timeout
            while buf.acquired < target:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise self.TimeoutError
                buf.cond.wait(left)
        return True

    def get_new_images_range(self):
        buf = self._buffer
        if buf.frames is None:
            return None
        oldest = max(buf.last_read, buf.acquired - len(buf.frames))
        return (oldest, buf.acquired) if buf.acquired > oldest else None

    def read_multiple_images(
        self, rng=None, peek=False, missing_frame="skip", return_info=False, return_rng=False
    ):
        """
        Read multiple images specified by `rng` (by default, all un-read images).

        If no new frames are available, return an empty list; if no acquisition is set up, return ``None``.
        """
        buf = self._buffer
        if buf.frames is None:
            result = tuple(None for inc in (True, return_info, return_rng) if inc)
            return result[0] if len(result) == 1 else result
        with buf.cond:
            if rng is None:
                rng = (buf.last_read, buf.acquired)
            frames, infos, idxs = [], [], []
            for n in range(*rng):
                f = buf.get(n)
                if f is None:
                    if missing_frame == "skip":
                        continue
                    f = (
                        np.zeros_like(buf.frames[-1])
                        if missing_frame == "zero" and buf.frames
                        else None
                    )
                frames.append(self._color_convert(f) if f is not None else None)
                infos.append(TFrameInfo(n, n, None, None, None))
                idxs.append(n)
            if not peek:
                buf.last_read = max(buf.last_read, rng[1])
        out_rng = (idxs[0], idxs[-1] + 1) if idxs else (rng[0], rng[0])
        result = [frames]
        if return_info:
            result.append(infos)
        if return_rng:
            result.append(out_rng)
        return result[0] if len(result) == 1 else tuple(result)

    def read_newest_image(self, peek=False, return_info=False):
        rng = self.get_new_images_range()
        if rng is None:
            return None
        res = self.read_multiple_images(rng=(rng[1] - 1, rng[1]), peek=peek, return_info=True)
        frames, infos = res
        if not frames:
            return None
        return (frames[0], infos[0]) if return_info else frames[0]

    def read_oldest_image(self, peek=False, return_info=False):
        rng = self.get_new_images_range()
        if rng is None:
            return None
        res = self.read_multiple_images(rng=(rng[0], rng[0] + 1), peek=peek, return_info=True)
        frames, infos = res
        if not frames:
            return None
        return (frames[0], infos[0]) if return_info else frames[0]

    def _color_convert(self, img):
        """raw: as is.  rgb/grayscale: CHEAP superpixel demosaic (approximation)."""
        if self._color_output == "raw":
            return img
        from sim.signal_model import bayer_channel_pattern

        pat = bayer_channel_pattern(self._world.phase)
        chans = []
        for c in (0, 1, 2):
            sites = [(y, x) for y in (0, 1) for x in (0, 1) if pat[y, x] == c]
            acc = sum(img[y::2, x::2].astype(np.float32) for y, x in sites) / len(sites)
            chans.append(np.repeat(np.repeat(acc, 2, axis=0), 2, axis=1))
        rgb = np.stack(chans, axis=-1).astype(img.dtype)
        if self._color_output == "grayscale":
            return np.mean(rgb, axis=-1, dtype=rgb.dtype)
        return rgb

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------
    def grab(
        self, nframes=1, frame_timeout=5.0, missing_frame="skip", return_info=False, buff_size=None
    ):
        """
        Snap `nframes` images (with preset image read mode parameters)

        Timeout is specified for a single-frame acquisition, not for the whole acquisition time.
        """
        if buff_size is None:
            buff_size = _DEFAULT_NFRAMES
        self._next_acq_times_out = self._world.faults.next_snap_times_out()
        frames, info, nacq = [], [], 0
        self._in_grab = True
        try:
            self.start_acquisition(nframes=buff_size, frames_per_trigger=None, auto_start=True)
        finally:
            self._in_grab = False
        try:
            while nacq < nframes:
                self.wait_for_frame(timeout=frame_timeout)
                new_frames, new_info, rng = self.read_multiple_images(
                    missing_frame=missing_frame, return_info=True, return_rng=True
                )
                frames += new_frames
                info += new_info
                nacq += rng[1] - rng[0]
            frames, info = frames[:nframes], info[:nframes]
            return (frames, info) if return_info else frames
        finally:
            self.stop_acquisition()

    def snap(self, timeout=5.0, return_info=False):
        """Snap a single frame"""
        res = self.grab(frame_timeout=timeout, return_info=return_info)
        if return_info:
            return res[0][0], res[1][0]
        return res[0]


__all__ = [
    "ThorlabsTLCamera",
    "ThorlabsTLCameraError",
    "ThorlabsTLCameraLibError",
    "ThorlabsTLCameraTimeoutError",
    "list_cameras",
    "get_cameras_number",
    "TDeviceInfo",
    "TSensorInfo",
    "TColorInfo",
    "TColorFormat",
    "TFrameInfo",
]
