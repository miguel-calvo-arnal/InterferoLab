#!/usr/bin/env python
r"""
Lab timing probe: measure, with the REAL camera and piezo, every timing and
sensor figure the simulator had to estimate, and write a profile that
sim/run_simulated.py can load (``--profile``).

Lab PC (Windows, only the packaged app, no Python): use the STANDALONE
executable built from sim/lab_timing_probe.spec (see sim/README.md):

    lab_timing_probe\lab_timing_probe.exe                 (double-click, ~6-10 min)
    lab_timing_probe\lab_timing_probe.exe --interactive   (+ dark frames)

It needs what the app needs and nothing else: the ThorCam software
installed (pylablib loads thorlabs_tsi_camera_sdk.dll from
"Program Files/Thorlabs/Scientific Imaging/ThorCam", exactly as the app
does) and the PI DLL, which is bundled inside the probe like in the app
(API/PI/E816_DLL_x64.dll).  The profile, the raw JSON, the raw stack and a
log are written NEXT TO THE EXECUTABLE.  Bring that folder back and load:

    .venv/bin/python sim/run_simulated.py --profile <folder>/measured_<date>_<host>.toml

From a source checkout (any OS, repo + venv):

    .venv/bin/python sim/lab_timing_probe.py                 # real hardware
    .venv-win\Scripts\python.exe sim\lab_timing_probe.py   # Windows venv
    .venv/bin/python sim/lab_timing_probe.py --dry-run --quick   # against the fakes (source only)

Close InterferoLab first (the devices can only be opened once).

Robust by design: every step runs in its own thread with a timeout; a step
that fails is recorded and the probe goes on.  A step that HANGS (timeout)
marks its device(s) as dead: no later step touches that device again (the
hung SDK/DLL call may still be running), it is not closed, and the probe
tells you to power-cycle it.  The other device keeps being measured.

The probe is self-contained (no import from sim/): the standalone build
must not carry the simulator.

Output (in --out, default sim/profiles/):
  measured_<date>_<host>.toml   figures with origin "measured" (merge over default)
  measured_<date>_<host>.json   every raw number, error and sample
A --dry-run writes dryrun_*.toml with origin "estimated" and a loud source:
it measures the SIMULATOR, not the hardware.

What it measures (gaps listed in _agentes/_trabajo/1_hechos.md s.5-6):
  camera: open time, arm and disarm, first frame after the software trigger
  (at several exposures -> readout+transfer intercept), frame period, full
  snap(), SDK call cost, default trigger mode / gain / black level / frame
  period as the camera opens, the Bayer phase (reported and from the image),
  gain e-/DN (photon transfer), black level and read noise (dark, optional).
  piezo: connect, GCS query round trip (qERR), MOV cost (write + ERR?),
  time to ONT for 0.02/0.1/1/10 um steps, position error at ONT, error codes
  for an out-of-range MOV and a MOV with servo off.
  both: a coarse z scan that finds the coherence peak and a DEFOCUSED
  position (the photon transfer is measured there: fringes plus stage
  vibration would inflate the variance and bias the gain low), a short RAW
  stack through focus (256x256 crop, 20 nm steps: raw visibility, channel
  levels, noise, NA), and 20 steps of the app's sweep loop.
"""

from __future__ import annotations

import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))
if FROZEN:  # standalone lab executable: outputs next to the .exe
    EXE_DIR = os.path.dirname(os.path.abspath(sys.executable))
    BUNDLE_DIR = getattr(sys, "_MEIPASS", EXE_DIR)
    SIM_DIR = ROOT = FAKES = None
else:
    SIM_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(SIM_DIR)
    FAKES = os.path.join(SIM_DIR, "fakes")
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) not in (SIM_DIR, FAKES)]

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402

PYLABLIB_SLEEP_BEFORE_DISARM_S = 0.2  # pylablib 1.4.5 TLCamera.stop_acquisition
PYLABLIB_SLEEP_BEFORE_TRIGGER_S = 0.05  # TLCamera.start_acquisition


ORIGINS = ("published", "measured", "derived", "estimated")


def toml_figure(value, origin: str, source: str) -> str:
    """One figure as a TOML inline table (same format as sim/profiles/*.toml)."""
    if origin not in ORIGINS:
        raise ValueError(f"bad origin {origin!r}")
    if isinstance(value, (bool, np.bool_)):
        v = "true" if value else "false"
    elif isinstance(value, (int, np.integer)):
        v = str(int(value))
    elif isinstance(value, (float, np.floating)):
        v = repr(float(value))  # never "np.float64(...)"
    else:
        v = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
    src = source.replace("\\", "\\\\").replace('"', '\\"')
    return f'{{ value = {v}, origin = "{origin}", source = "{src}" }}'


class _Tee:
    """Copy stdout/stderr into the log file next to the outputs."""

    def __init__(self, stream, fh) -> None:
        self.stream, self.fh = stream, fh

    def write(self, text):
        with contextlib.suppress(Exception):  # a closed console must not stop the probe
            self.stream.write(text)
        self.fh.write(text)
        self.fh.flush()
        return len(text)

    def flush(self):
        with contextlib.suppress(Exception):
            self.stream.flush()
        self.fh.flush()


class Probe:
    def __init__(self, args) -> None:
        self.args = args
        self.raw: dict = {}
        self.errors: dict = {}
        self.figures: dict = {}  # dotted key -> (value, source)
        self.cam = None
        self.dev = None
        self.reps = 3 if args.quick else 10
        self.dead: set[str] = set()  # devices left unusable by a hung step
        self.focus_z = None
        self.defocus_z = None
        self.stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    def usable(self, *devices: str) -> bool:
        have = {"camera": self.cam is not None, "piezo": self.dev is not None}
        return all(have[d] and d not in self.dead for d in devices)

    # ------------------------------------------------------------------
    def run_step(self, name: str, fn, timeout: float | None = None, devices=()) -> None:
        """Run fn in a thread; on TIMEOUT the devices it used are declared dead."""
        if any(d in self.dead for d in devices):
            self.errors[name] = (
                "SKIPPED: device(s) " + ", ".join(sorted(self.dead)) + " hung earlier"
            )
            print(f"  [SKIPPED] {name} (device hung earlier)", flush=True)
            return
        timeout = timeout or self.args.step_timeout
        box: dict = {}

        def target():
            try:
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001
                box["error"] = f"{type(e).__name__}: {e}"
                box["tb"] = traceback.format_exc()

        t0 = time.perf_counter()
        th = threading.Thread(target=target, name=f"probe-{name}", daemon=True)
        th.start()
        th.join(timeout)
        dt = time.perf_counter() - t0
        if th.is_alive():
            self.errors[name] = (
                f"TIMEOUT after {timeout:.1f} s (thread left running); "
                f"device(s) {', '.join(devices) or '-'} no longer used"
            )
            self.dead.update(devices)
            status = "TIMEOUT"
        elif "error" in box:
            self.errors[name] = box["error"]
            self.raw.setdefault("tracebacks", {})[name] = box["tb"]
            status = "FAILED: " + box["error"]
        else:
            status = "ok"
        print(f"  [{status:>6.40}] {name} ({dt:.1f} s)", flush=True)

    def fig(self, key: str, value, source: str) -> None:
        self.figures[key] = (value, source)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def cam_open(self):
        from pylablib.devices import Thorlabs

        t0 = time.perf_counter()
        self.cam = Thorlabs.ThorlabsTLCamera()
        dt = time.perf_counter() - t0
        self.raw["camera_open_s"] = dt
        self.fig("camera.timing.open_s", round(dt, 4), "ThorlabsTLCamera() constructor, 1 sample")

    def cam_info(self):
        cam = self.cam
        info = cam.get_device_info()
        self.raw["device_info"] = list(info)
        si = cam.get_sensor_info()
        self.raw["sensor_info"] = list(si)
        ci = cam.get_color_info()
        self.raw["color_info"] = (
            None
            if ci is None
            else {
                "filter_array_phase": ci.filter_array_phase,
                "correction_matrix": np.asarray(ci.correction_matrix).tolist(),
                "default_white_balance_matrix": np.asarray(
                    ci.default_white_balance_matrix
                ).tolist(),
            }
        )
        if ci is not None:
            self.fig(
                "camera.sensor.filter_array_phase",
                ci.filter_array_phase,
                "get_color_info().filter_array_phase",
            )
        self.fig("camera.sensor.bit_depth", int(si.bit_depth), "get_sensor_info().bit_depth")
        w, h = cam.get_detector_size()
        self.fig("camera.sensor.width", int(w), "get_detector_size()")
        self.fig("camera.sensor.height", int(h), "get_detector_size()")

    def cam_defaults(self):
        """What the camera has when opened (the app never sets these)."""
        cam = self.cam
        out = {}
        for name in (
            "get_trigger_mode",
            "get_exposure",
            "get_frame_period",
            "get_gain",
            "get_gain_range",
            "get_black_level",
            "get_black_level_range",
            "get_pixel_correction_parameters",
            "get_roi",
            "get_frame_timings",
            "get_color_format",
        ):
            try:
                v = getattr(cam, name)()
                out[name] = (
                    v
                    if isinstance(v, (int, float, str, type(None)))
                    else list(v)
                    if isinstance(v, tuple)
                    else str(v)
                )
            except Exception as e:  # noqa: BLE001
                out[name] = f"n/a ({type(e).__name__}: {e})"
        self.raw["camera_defaults_at_open"] = out

    def cam_configure(self):
        """Same calls as connect_camera() + SDK call cost."""
        cam = self.cam
        cam.set_color_format(color_output="raw", color_space="linear")
        cam.set_roi(hbin=1, vbin=1)
        ts = []
        for _ in range(self.reps * 2):
            t0 = time.perf_counter()
            cam.set_exposure(self.args.exposure_ms / 1000)
            ts.append(time.perf_counter() - t0)
        self.raw["set_exposure_s"] = ts
        # set_exposure = set + get (get goes through get_frame_timings: 2 SDK calls)
        self.fig(
            "camera.timing.sdk_call_s",
            round(statistics.median(ts) / 3, 5),
            f"median(set_exposure) / 3 SDK calls, {len(ts)} samples",
        )
        self.raw["exposure_readback_s"] = cam.get_exposure()

    def _first_frame_and_period(self, exposure_s: float, nper: int):
        cam = self.cam
        cam.set_exposure(exposure_s)
        t0 = time.perf_counter()
        cam.start_acquisition(auto_start=False)
        t_arm = time.perf_counter() - t0
        t1 = time.perf_counter()
        cam.send_software_trigger()
        cam.wait_for_frame(timeout=5.0)
        t_first = time.perf_counter() - t1
        cam.read_multiple_images()
        stamps = []
        for _ in range(nper):
            cam.wait_for_frame(timeout=5.0)
            stamps.append(time.perf_counter())
            cam.read_multiple_images()
        t2 = time.perf_counter()
        cam.stop_acquisition()
        t_stop = time.perf_counter() - t2
        period = statistics.median(np.diff(stamps)) if len(stamps) > 2 else None
        return t_arm, t_first, period, t_stop

    def cam_arm_trigger(self):
        cam = self.cam
        cam.stop_acquisition()
        exposures = [0.001, self.args.exposure_ms / 1000, 0.030]
        rows = []
        for e in exposures:
            for _ in range(max(2, self.reps // 2)):
                rows.append((e, *self._first_frame_and_period(e, 4 if self.args.quick else 10)))
        self.raw["arm_trigger_rows"] = [
            {
                "exposure_s": r[0],
                "arm_s": r[1],
                "first_frame_s": r[2],
                "frame_period_s": r[3],
                "stop_s": r[4],
            }
            for r in rows
        ]
        self.fig(
            "camera.timing.arm_s",
            round(statistics.median(r[1] for r in rows), 4),
            f"start_acquisition(auto_start=False) from disarmed, {len(rows)} samples",
        )
        self.fig(
            "camera.timing.disarm_s",
            round(
                max(0.0, statistics.median(r[4] for r in rows) - PYLABLIB_SLEEP_BEFORE_DISARM_S), 4
            ),
            f"stop_acquisition() minus pylablib's 0.2 s sleep, {len(rows)} samples",
        )
        ex = np.array([r[0] for r in rows])
        ff = np.array([r[2] for r in rows])
        slope, icpt = np.polyfit(ex, ff, 1)
        self.raw["first_frame_fit"] = {"slope": slope, "intercept_s": icpt}
        self.fig(
            "camera.timing.readout_transfer_s",
            round(float(icpt), 4),
            f"intercept of first-frame latency vs exposure (slope {slope:.2f}); includes the trigger latency",
        )
        self.fig(
            "camera.timing.trigger_latency_s", 0.0, "folded into readout_transfer_s by the probe"
        )
        short = [r[3] for r in rows if r[0] == exposures[0] and r[3]]
        if short:
            self.fig(
                "camera.timing.min_frame_period_s",
                round(statistics.median(short), 5),
                "frame period in continuous mode at 1 ms exposure",
            )

    def cam_snap(self):
        cam = self.cam
        cam.set_exposure(self.args.exposure_ms / 1000)
        ts = []
        for _ in range(self.reps * 2):
            t0 = time.perf_counter()
            frame = cam.snap(timeout=5.0)
            ts.append(time.perf_counter() - t0)
        self.raw["snap_s"] = ts
        self.raw["snap_frame"] = {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "max": int(frame.max()),
            "min": int(frame.min()),
        }
        print(f"        snap(): median {statistics.median(ts):.3f} s over {len(ts)}")

    def cam_bayer_from_image(self):
        f = np.asarray(self.cam.snap(timeout=5.0)).astype(np.float64)
        h, w = f.shape
        c = f[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        means = {f"{y}{x}": float(c[y::2, x::2].mean()) for y in (0, 1) for x in (0, 1)}
        self.raw["bayer_site_means"] = means
        # Under the halogen lamp blue is by far the weakest channel (weights: B 0.12, R 0.26, G 0.31)
        lowest = min(means, key=means.get)
        phase = {"00": "blue", "11": "red", "01": "green_left_or_blue", "10": "green_left_or_red"}[
            lowest
        ]
        self.raw["bayer_phase_from_image"] = phase
        print(
            f"        site means {means} -> weakest site {lowest} -> phase '{phase}' if it is blue"
        )

    def cam_photon_transfer(self):
        """Gain e-/DN from pairs of frames (difference removes fixed patterns).

        Measured at the DEFOCUSED z found by the focus scan: with fringes in
        the window, stage vibration between the two frames adds variance
        and the gain comes out low.  Saturation is checked in the window.
        """
        cam = self.cam
        where = "no piezo: NOT defocused, gain may be biased low"
        if self.defocus_z is not None and self.usable("piezo"):
            self.dev.MOV("A", float(self.defocus_z))
            self._wait_ont(5.0)
            where = f"defocused at z = {self.defocus_z:.2f} um (focus {self.focus_z:.2f} um)"
        self.raw["photon_transfer_where"] = where
        print(f"        photon transfer {where}")
        pts = []
        for e in (0.002, self.args.exposure_ms / 1000, 0.015, 0.030):
            cam.set_exposure(e)
            a = np.asarray(cam.snap(timeout=5.0)).astype(np.float64)
            b = np.asarray(cam.snap(timeout=5.0)).astype(np.float64)
            h, w = a.shape
            sl = (slice(h // 2 - 256, h // 2 + 256), slice(w // 2 - 256, w // 2 + 256))
            # green sites of either phase: (0,1) and (1,0) are green for red/blue phases
            ga, gb = a[sl][0::2, 1::2], b[sl][0::2, 1::2]
            mean = 0.5 * (ga.mean() + gb.mean())
            var = np.var(ga - gb) / 2
            pts.append((e, mean, var, int(max(ga.max(), gb.max()))))
        cam.set_exposure(self.args.exposure_ms / 1000)
        self.raw["photon_transfer"] = [
            {"exposure_s": p[0], "mean_dn": p[1], "var_dn2": p[2], "max_dn_window": p[3]}
            for p in pts
        ]
        ok = [p for p in pts if p[3] < 4000]
        if len(ok) >= 2:
            slope, icpt = np.polyfit([p[1] for p in ok], [p[2] for p in ok], 1)
            if slope > 0:
                self.fig(
                    "camera.sensor.gain_e_per_dn",
                    round(1.0 / slope, 4),
                    f"photon transfer, G sites of a 512x512 window, {len(ok)} exposures "
                    f"(var = mean/g + c), {where}",
                )
            self.raw["photon_transfer_fit"] = {"slope": slope, "intercept": icpt}
            e7 = min(ok, key=lambda p: abs(p[0] - self.args.exposure_ms / 1000))
            self.raw["g_level_at_exposure"] = {"exposure_s": e7[0], "mean_dn": e7[1]}

    def cam_dark(self):
        if not self.args.interactive:
            self.raw["dark"] = (
                "skipped (run with --interactive to measure black level and read noise)"
            )
            return
        input("\n  >>> Block ALL light to the camera, then press Enter... ")
        cam = self.cam
        cam.set_exposure(self.args.exposure_ms / 1000)
        a = np.asarray(cam.snap(timeout=5.0)).astype(np.float64)
        b = np.asarray(cam.snap(timeout=5.0)).astype(np.float64)
        black = float(0.5 * (a.mean() + b.mean()))
        read_dn = float(np.std(a - b) / np.sqrt(2))
        self.raw["dark"] = {"black_dn": black, "read_noise_dn": read_dn}
        self.fig(
            "camera.sensor.black_level_dn",
            round(black, 3),
            f"dark frames at {self.args.exposure_ms} ms",
        )
        g = self.figures.get("camera.sensor.gain_e_per_dn", (None,))[0]
        if g:
            self.fig(
                "camera.sensor.read_noise_e",
                round(read_dn * g, 3),
                "dark frame difference x measured gain",
            )
        input("  >>> Restore the light, then press Enter... ")

    def cam_close(self):
        t0 = time.perf_counter()
        self.cam.close()
        self.fig("camera.timing.close_s", round(time.perf_counter() - t0, 4), "close(), 1 sample")
        self.cam = None

    # ------------------------------------------------------------------
    # Piezo
    # ------------------------------------------------------------------
    def pz_connect(self):
        from pipython import GCSDevice

        t0 = time.perf_counter()
        dev = GCSDevice(devname="E-625", gcsdll=self.args.dll)
        dev.ConnectUSB(serialnum=self.args.piezo_serial)
        self.dev = dev  # only a connected controller is used by later steps
        self.fig(
            "piezo.timing.connect_s",
            round(time.perf_counter() - t0, 4),
            "GCSDevice + ConnectUSB, 1 sample",
        )
        self.raw["piezo_idn"] = self.dev.qIDN().strip()
        self.raw["piezo_initial_err"] = self.dev.qERR()
        try:
            self.raw["piezo_initial_pos"] = dict(self.dev.qPOS("A"))
            self.raw["piezo_initial_svo"] = dict(self.dev.qSVO("A"))
        except Exception as e:  # noqa: BLE001
            self.raw["piezo_initial_state_error"] = str(e)

    def pz_query(self):
        ts = []
        for _ in range(self.reps * 5):
            t0 = time.perf_counter()
            self.dev.qERR()
            ts.append(time.perf_counter() - t0)
        self.raw["qERR_s"] = ts
        self.fig(
            "piezo.timing.gcs_query_s",
            round(statistics.median(ts), 5),
            f"median qERR() (one query, no errcheck), {len(ts)} samples",
        )

    def pz_mov_and_settle(self):
        dev = self.dev
        dev.SVO("A", 1)
        base = 50.0
        dev.MOV("A", base)
        self._wait_ont(5.0)
        q = self.figures.get("piezo.timing.gcs_query_s", (0.0,))[0]
        rows = []
        for step in (0.02, 0.1, 1.0, 10.0):
            for i in range(self.reps):
                target = base + step * (1 if i % 2 == 0 else 0)
                t0 = time.perf_counter()
                dev.MOV("A", target)
                t_mov = time.perf_counter() - t0
                # Each call ends with pipython's ERR? query (~q), so the
                # controller executed MOV at ~t_mov - q and answered each
                # ONT? at ~(return time - q).
                t_cmd = t_mov - q
                last_false = 0.0
                polls = 0
                while True:
                    polls += 1
                    ont = dev.qONT("A")["A"]
                    t_ans = time.perf_counter() - t0 - q
                    if ont:
                        break
                    last_false = t_ans - t_cmd
                    if time.perf_counter() - t0 > 5:
                        break
                t_ont = time.perf_counter() - t0
                first_true = t_ans - t_cmd
                try:
                    err = float(dev.qPOS("A")["A"]) - target
                except Exception:  # noqa: BLE001
                    err = None
                rows.append(
                    {
                        "step_um": step,
                        "mov_s": t_mov,
                        "ont_s": t_ont,
                        "polls": polls,
                        "settle_lo_s": max(0.0, last_false),
                        "settle_hi_s": first_true,
                        "pos_error_at_ont_um": err,
                        "reached": bool(ont),
                    }
                )
        self.raw["settle_rows"] = rows
        movs = [r["mov_s"] for r in rows]
        self.fig(
            "piezo.timing.gcs_write_s",
            round(max(0.0, statistics.median(movs) - q), 5),
            "median MOV (write + ERR?) minus gcs_query_s",
        )
        small = [
            0.5 * (r["settle_lo_s"] + r["settle_hi_s"])
            for r in rows
            if r["step_um"] == 0.02 and r["reached"]
        ]
        if small:
            self.fig(
                "piezo.timing.settle_s",
                round(statistics.median(small), 4),
                "MOV of 0.02 um: controller time from MOV to ONT, midpoint between the last false and "
                "first true qONT answers (resolution ~2 x gcs_query_s)",
            )
        for step in (0.02, 0.1, 1.0, 10.0):
            s = [r["ont_s"] for r in rows if r["step_um"] == step and r["reached"]]
            if s:
                print(
                    f"        time to ONT, {step:>5} um step: median {statistics.median(s) * 1000:.1f} ms"
                )

    def _wait_ont(self, timeout: float) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.dev.qONT("A")["A"]:
                return True
            time.sleep(0.01)
        return False

    def pz_error_codes(self):
        from pipython import GCSError

        dev = self.dev
        codes = {}
        try:
            dev.MOV("A", 1000.0)
            codes["mov_out_of_range"] = "no error raised"
        except GCSError as e:
            codes["mov_out_of_range"] = [e.val, str(e)]
        try:
            dev.SVO("A", 0)
            try:
                dev.MOV("A", 50.0)
                codes["mov_servo_off"] = "no error raised"
            except GCSError as e:
                codes["mov_servo_off"] = [e.val, str(e)]
        finally:
            dev.SVO("A", 1)
            dev.MOV("A", 50.0)
            self._wait_ont(5.0)
        self.raw["gcs_error_codes"] = codes

    # ------------------------------------------------------------------
    def _snap_center(self, half: int = 256) -> np.ndarray:
        f = np.asarray(self.cam.snap(timeout=5.0))
        h, w = f.shape
        y0, x0 = (h // 2 - half) & ~1, (w // 2 - half) & ~1
        return f[y0 : y0 + 2 * half, x0 : x0 + 2 * half].astype(np.float64)

    def _goto(self, z: float) -> None:
        self.dev.MOV("A", float(z))
        self._wait_ont(5.0)

    def focus_scan(self):
        """Coarse z scan: fringe activity = mean |frame(z) - frame(z + 70 nm)| on G sites.

        70 nm is about a quarter of the G fringe period, so the difference is
        large inside the coherence envelope and just noise outside it.
        """
        self.cam.set_exposure(self.args.exposure_ms / 1000)
        step = 2.0 if self.args.quick else 0.5
        zs = np.arange(self.args.scan_from, self.args.scan_to + 1e-9, step)
        act = []
        for z in zs:
            self._goto(z)
            a = self._snap_center()
            self._goto(z + 0.07)
            b = self._snap_center()
            act.append(float(np.mean(np.abs(a[0::2, 1::2] - b[0::2, 1::2]))))
        act = np.array(act)
        i = int(np.argmax(act))
        self.focus_z = float(zs[i])
        far = np.abs(zs - self.focus_z) >= 4.0
        cand = (
            np.where(far)[0] if far.any() else np.array([int(np.argmax(np.abs(zs - self.focus_z)))])
        )
        self.defocus_z = float(zs[cand[np.argmin(act[cand])]])
        self.raw["focus_scan"] = {
            "z_um": zs.tolist(),
            "activity_dn": act.tolist(),
            "focus_z_um": self.focus_z,
            "defocus_z_um": self.defocus_z,
        }
        print(
            f"        fringe activity peak at z = {self.focus_z:.2f} um "
            f"({act[i]:.1f} DN vs {np.median(act):.1f} median); defocused z = {self.defocus_z:.2f} um"
        )
        if act[i] < 3 * np.median(act):
            print("        WARNING: weak peak - is there a sample in focus within the scan range?")

    def raw_stack(self):
        """Short RAW stack through focus (crop), saved as .npz next to the profile."""
        if self.focus_z is None:
            raise RuntimeError("no focus found (focus scan failed or was skipped)")
        n = 20 if self.args.quick else self.args.raw_frames
        dz = 0.02
        zs = self.focus_z + (np.arange(n) - n // 2) * dz
        self.cam.set_exposure(self.args.exposure_ms / 1000)
        frames = np.empty((n, 256, 256), dtype=np.uint16)
        t = np.empty(n)
        for k, z in enumerate(zs):
            self._goto(z)
            t[k] = time.perf_counter()
            frames[k] = self._snap_center(128).astype(np.uint16)
        ci = self.raw.get("color_info") or {}
        name = f"{'dryrun' if self.args.dry_run else 'raw'}_stack_{self.stamp}.npz"
        path = os.path.join(self.args.out, name)
        os.makedirs(self.args.out, exist_ok=True)
        np.savez_compressed(
            path,
            frames=frames,
            z_setpoint_um=zs,
            t_s=t - t[0],
            exposure_s=self.args.exposure_ms / 1000,
            filter_array_phase=str(ci.get("filter_array_phase", "")),
            correction_matrix=np.asarray(ci.get("correction_matrix", np.eye(3))),
            white_balance_matrix=np.asarray(ci.get("default_white_balance_matrix", np.eye(3))),
            note="raw Bayer 256x256 crop at the sensor centre (even-aligned); z = MOV set-points",
        )
        self.raw["raw_stack"] = {"file": name, "frames": n, "dz_um": dz, "z0_um": float(zs[0])}
        print(f"        raw stack: {n} frames of 256x256 -> {name}")

    def sweep_like_app(self):
        """20 steps of AcquisitionSession.run_sweep's loop (without saving)."""
        cam, dev = self.cam, self.dev
        cam.set_exposure(self.args.exposure_ms / 1000)
        steps = []
        z0 = 50.0
        n = 5 if self.args.quick else 20
        for i in range(n):
            z = z0 + 0.02 * i
            t0 = time.perf_counter()
            dev.MOV("A", z)
            polls = 0
            while True:
                polls += 1
                if dev.qONT("A")["A"]:
                    break
                time.sleep(0.1)  # AcquisitionSession.MOVE_POLL_S
            t_move = time.perf_counter() - t0
            cam.snap(timeout=5.0)
            steps.append({"move_s": t_move, "polls": polls, "step_s": time.perf_counter() - t0})
        self.raw["sweep_like_app"] = steps
        print(
            f"        app-like sweep step: median {statistics.median(s['step_s'] for s in steps):.3f} s "
            f"(move {statistics.median(s['move_s'] for s in steps):.3f} s)"
        )

    def pz_close(self):
        dev = self.dev
        try:
            init = self.raw.get("piezo_initial_pos", {}).get("A")
            if init is not None:
                dev.MOV("A", float(init))
                self._wait_ont(5.0)
            svo = self.raw.get("piezo_initial_svo", {}).get("A")
            if svo is not None and not svo:
                dev.SVO("A", 0)
        finally:
            t0 = time.perf_counter()
            dev.CloseConnection()
            self.fig(
                "piezo.timing.close_s",
                round(time.perf_counter() - t0, 4),
                "CloseConnection(), 1 sample",
            )
            self.dev = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        a = self.args
        C, P, CP = ("camera",), ("piezo",), ("camera", "piezo")
        if not a.skip_camera:
            print("\n== Camera")
            self.run_step("camera open", self.cam_open, timeout=60, devices=C)
            if self.usable("camera"):
                self.run_step("camera info", self.cam_info, devices=C)
                self.run_step("camera defaults at open", self.cam_defaults, devices=C)
                self.run_step("camera configure like the app", self.cam_configure, devices=C)
                self.run_step(
                    "arm / trigger / first frame / period / disarm",
                    self.cam_arm_trigger,
                    timeout=300,
                    devices=C,
                )
                self.run_step("snap() timing", self.cam_snap, timeout=120, devices=C)
                self.run_step("Bayer phase from the image", self.cam_bayer_from_image, devices=C)
        if not a.skip_piezo:
            print("\n== Piezo")
            self.run_step("piezo connect", self.pz_connect, devices=P)
            if self.usable("piezo"):
                self.run_step("GCS query round trip", self.pz_query, devices=P)
                self.run_step(
                    "MOV cost and time to on-target", self.pz_mov_and_settle, timeout=300, devices=P
                )
                self.run_step("GCS error codes", self.pz_error_codes, devices=P)
        if self.usable("camera", "piezo"):
            print("\n== Camera + piezo")
            self.run_step(
                "focus scan (find coherence peak)", self.focus_scan, timeout=400, devices=CP
            )
        if self.usable("camera"):
            devs = CP if self.usable("piezo") else C
            self.run_step(
                "photon transfer (gain)", self.cam_photon_transfer, timeout=180, devices=devs
            )
        if self.usable("camera", "piezo"):
            if not a.no_raw_stack:
                self.run_step("raw stack through focus", self.raw_stack, timeout=900, devices=CP)
            self.run_step("sweep loop like the app", self.sweep_like_app, timeout=300, devices=CP)
        if self.usable("camera"):
            self.run_step(
                "dark frames", self.cam_dark, timeout=None if a.interactive else 60, devices=C
            )
        print("\n== Closing")
        if self.usable("camera"):
            self.run_step("camera close", self.cam_close, devices=C)
        if self.usable("piezo"):
            self.run_step("piezo close", self.pz_close, devices=P)
        for d in sorted(self.dead):
            print(
                f"  !! the {d} HUNG during the probe and was left open: close this window, "
                f"power-cycle / re-plug the {d} before using InterferoLab."
            )

    # ------------------------------------------------------------------
    def write(self) -> tuple[str, str]:
        dry = self.args.dry_run
        stamp = self.stamp
        host = platform.node() or "host"
        base = f"{'dryrun' if dry else 'measured'}_{stamp}_{host}"
        os.makedirs(self.args.out, exist_ok=True)
        toml_path = os.path.join(self.args.out, base + ".toml")
        json_path = os.path.join(self.args.out, base + ".json")
        origin = "estimated" if dry else "measured"
        prefix = (
            "DRY RUN against the simulator, NOT a hardware measurement: "
            if dry
            else f"lab_timing_probe {stamp} on {host}: "
        )
        sections: dict[str, list[str]] = {}
        for key, (value, src) in sorted(self.figures.items()):
            sect, name = key.rsplit(".", 1)
            sections.setdefault(sect, []).append(
                f"{name} = {toml_figure(value, origin, prefix + src)}"
            )
        lines = [
            "# Written by sim/lab_timing_probe.py -- merge over sim/profiles/default.toml:",
            f"#   python sim/run_simulated.py --profile <path to>/{os.path.basename(toml_path)}",
            f"# Raw numbers, samples and errors: {os.path.basename(json_path)}",
            "",
            "[meta]",
            f'name = "{base}"',
            f'description = "{"DRY RUN (simulator)" if dry else "Measured in the lab"} {stamp}; '
            f'{len(self.figures)} figures, {len(self.errors)} failed steps"',
            "",
        ]
        for sect in sorted(sections):
            lines.append(f"[{sect}]")
            lines += sections[sect]
            lines.append("")
        with open(toml_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        with open(toml_path, "rb") as fh:  # validate: TOML syntax, origins, sources
            tree = tomllib.load(fh)
        for sect, items in tree.items():
            if sect == "meta":
                continue
            for key, node in _flat(items, sect):
                if node.get("origin") not in ORIGINS or not node.get("source"):
                    raise ValueError(f"bad figure written: {key}")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dry_run": dry,
                    "args": vars(self.args),
                    "figures": self.figures,
                    "errors": self.errors,
                    "raw": self.raw,
                },
                fh,
                indent=2,
                default=str,
            )
        return toml_path, json_path


def _flat(tree: dict, path: str):
    for key, node in tree.items():
        here = f"{path}.{key}"
        if isinstance(node, dict) and "value" in node:
            yield here, node
        elif isinstance(node, dict):
            yield from _flat(node, here)


def _resolve_piezo_serial(args) -> str:
    # The real serial is kept out of the repo (public mirror): take it from the
    # app_config.json the app already keeps on the lab machine, or ask for it.
    if args.dry_run or args.skip_piezo:
        return "000000000"
    base = BUNDLE_DIR if FROZEN else ROOT
    for folder in dict.fromkeys((EXE_DIR if FROZEN else ROOT, os.getcwd(), base)):
        path = os.path.join(folder, "app_config.json")
        try:
            with open(path, encoding="utf-8") as fh:
                serial = str(json.load(fh).get("piezo_serial") or "").strip()
        except (OSError, ValueError):
            continue
        if serial:
            print(f"Piezo serial {serial} read from {path}")
            return serial
    return input("Piezo serial number (see the label on the E-625 controller): ").strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure real camera/piezo timings and write a simulator profile."
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="run against the simulator fakes (tests this script)"
    )
    ap.add_argument("--quick", action="store_true", help="fewer repetitions")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="also take dark frames (asks you to block the light)",
    )
    ap.add_argument("--skip-camera", action="store_true")
    ap.add_argument("--skip-piezo", action="store_true")
    ap.add_argument("--exposure-ms", type=float, default=7.0, help="the app default is 7 ms")
    ap.add_argument(
        "--piezo-serial",
        default=None,
        help="default: read 'piezo_serial' from the app's app_config.json, else ask",
    )
    base = BUNDLE_DIR if FROZEN else ROOT
    ap.add_argument("--dll", default=os.path.join(base, "API", "PI", "E816_DLL_x64.dll"))
    ap.add_argument(
        "--out",
        default=EXE_DIR if FROZEN else os.path.join(SIM_DIR, "profiles"),
        help="output folder (default: next to the executable / sim/profiles)",
    )
    ap.add_argument("--step-timeout", type=float, default=60.0)
    ap.add_argument("--scan-from", type=float, default=35.0, help="focus scan start (um)")
    ap.add_argument("--scan-to", type=float, default=65.0, help="focus scan end (um)")
    ap.add_argument("--raw-frames", type=int, default=300, help="raw stack length (20 nm steps)")
    ap.add_argument("--no-raw-stack", action="store_true")
    ap.add_argument("--no-pause", action="store_true", help="do not wait for Enter at the end")
    args = ap.parse_args(argv)
    if args.piezo_serial is None:
        args.piezo_serial = _resolve_piezo_serial(args)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.out, f"lab_timing_probe_{stamp}.log")
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - lives for the whole run
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    try:
        return _main(args, log_path)
    except Exception:  # noqa: BLE001 - always leave the traceback in the log
        traceback.print_exc()
        return 3
    finally:
        if FROZEN and not args.no_pause and sys.stdin and sys.stdin.isatty():
            with contextlib.suppress(EOFError):
                input("\nPress Enter to close this window...")


def _main(args, log_path: str) -> int:
    print(f"InterferoLab lab timing probe - {datetime.now().isoformat(timespec='seconds')}")
    print(f"  output folder: {args.out}\n  log: {log_path}\n  PI DLL: {args.dll}")
    if args.dry_run and FROZEN:
        print(
            "ABORT: --dry-run only exists in a source checkout (the executable has no simulator)."
        )
        return 2
    if args.dry_run:
        sys.path.insert(0, ROOT)
        sys.path.insert(0, FAKES)
        print("[SIMULATION] DRY RUN: the probe talks to the simulator fakes, not to hardware.")
    import pipython
    import pylablib

    fake = getattr(pylablib, "__simulated__", False) or getattr(pipython, "__simulated__", False)
    if fake != args.dry_run:
        print(
            "ABORT: the fakes are loaded without --dry-run (or the other way round).",
            file=sys.stderr,
        )
        return 2

    t0 = time.perf_counter()
    probe = Probe(args)
    try:
        probe.run()
    finally:
        toml_path, json_path = probe.write()
    print(
        f"\nDone in {time.perf_counter() - t0:.0f} s: {len(probe.figures)} figures, {len(probe.errors)} failed steps."
    )
    for name, err in probe.errors.items():
        print(f"  FAILED {name}: {err}")
    print(f"  profile: {toml_path}\n  raw    : {json_path}\n  log    : {log_path}")
    return 1 if probe.dead else 0


if __name__ == "__main__":
    sys.exit(main())
