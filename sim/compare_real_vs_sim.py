#!/usr/bin/env python
"""
Fidelity gate: compare the hardware simulator against the real libraries and
the real LP126CU stacks.  Run it whenever sim/profiles/*.toml, sim/fakes or
sim/signal_model.py change.

    .venv/bin/python sim/compare_real_vs_sim.py                       # all layers
    .venv/bin/python sim/compare_real_vs_sim.py --layers timing,signal
    .venv/bin/python sim/compare_real_vs_sim.py --profile sim/profiles/measured_X.toml
    .venv/bin/python sim/compare_real_vs_sim.py --real /mnt/datos/data/2026-03-13_11-21-39

Exit code: 0 = every check passed, 1 = at least one check failed,
2 = nothing could be compared (e.g. no real stack found for the signal layer).

Layers
------
interface  Signatures, exception hierarchies, namedtuples and error messages
           of the fakes vs the REAL pylablib / pipython in .venv (each side in
           its own subprocess, because the fakes carry the real package names),
           plus the return types the app relies on (1_hechos s.1.2-1.3).
timing     Stopwatch on the fakes (subprocess, fakes first on sys.path) vs the
           time the profile predicts for the same call sequence (pylablib
           1.4.5's own sleeps included).  Also checks that every "published"
           figure cites a link and that derived readout = 1/fps.
signal     Per-pixel z series sampled from real stacks (read-only; a grid of
           pixels, never whole stacks kept in memory) vs simulated frames at the
           SAME z positions, passed through the same kind of colour fusion:
           (a) the app's current superpixel fusion (AcquisitionSession.
           _bayer_to_color_superpixel, phase from the profile) and (b) pylablib's
           own debayer (color.bayer_interpolate, identity colour matrices),
           which is how the pre-2026-09-08 stacks were made (color_output
           "auto" -> "rgb").  Real channels are identified by fringe period
           (longest = R), never by file order.
           Metrics per channel: background level, visibility, envelope FWHM,
           fringe period (phase slope of the analytic signal), temporal noise
           (first differences far from focus) and level/noise^2.

LZW-compressed stacks need ``imagecodecs``.  It is not in .venv on purpose:
sample those stacks with another interpreter that has numpy + tifffile +
imagecodecs, then pass the cache:

    /path/to/other/python sim/compare_real_vs_sim.py sample --real DIR --cache DIR
    .venv/bin/python sim/compare_real_vs_sim.py --cache DIR

The ``sample`` sub-command only needs numpy and tifffile (+ imagecodecs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SIM_DIR)
FAKES = os.path.join(SIM_DIR, "fakes")
# "python sim/compare_real_vs_sim.py" puts sim/ at sys.path[0]; drop it (same as the launcher).
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) not in (SIM_DIR, FAKES)]

DEFAULT_REAL = [
    "/mnt/datos/data/2026-03-13_11-21-39",
    "/mnt/datos/data/2026-03-13_11-50-20",
    "/mnt/datos/data/2026-03-13_13-14-36",
    "/mnt/datos/data/2026-04-22_12-41-58",
]

# Tolerances of the gate (sim vs real median).  Chosen from the spread
# between the real LP126CU stacks themselves (see _agentes/_trabajo/3_verificacion.md).
TOL = {
    "period_rel": 0.05,  # R and G fringe period
    "fwhm_rel": 0.25,  # R and G envelope FWHM
    "level_g_rel": 0.35,  # G background (exposure of the real stacks unknown)
    "timing_rel": 0.10,
    "timing_abs_s": 0.015,
}

_Z_RE = re.compile(r"piezo_([+-]?\d+(?:\.\d+)?)um", re.IGNORECASE)


# ======================================================================
# Small utilities
# ======================================================================
class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, layer: str, name: str, ok: bool | None, detail: str = "", **data) -> None:
        """ok=None -> informational (never fails the gate)."""
        self.rows.append({"layer": layer, "name": name, "ok": ok, "detail": detail, **data})
        tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["ok"] is False]

    @property
    def checked(self) -> list[dict]:
        return [r for r in self.rows if r["ok"] is not None]


def _run_json(code: str, fakes: bool, timeout: float = 600) -> dict:
    """Run ``code`` in a fresh interpreter (same python) and parse its last stdout line as JSON."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["QT_QPA_PLATFORM"] = "offscreen"
    paths = [FAKES, ROOT] if fakes else [ROOT]
    prelude = f"import sys; sys.path[:0] = {paths!r}\n"
    proc = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=ROOT,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(
            f"subprocess failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}\n{proc.stdout[-1000:]}"
        )
    return json.loads(lines[-1])


# ======================================================================
# Layer 1: interface
# ======================================================================
_API_DUMP = r"""
import inspect, json, collections
import pylablib, pipython
from pylablib.devices import Thorlabs
from pylablib.devices.Thorlabs import TLCamera as tlc
from pylablib.devices.Thorlabs import tl_camera_sdk_lib as sdklib
from pipython import GCSDevice, GCSError

def sig(obj, name):
    f = getattr(obj, name, None)
    if f is None:
        return None
    try:
        s = inspect.signature(f)
    except (TypeError, ValueError):
        return "<no signature>"
    return [(p.name, repr(p.default) if p.default is not inspect.Parameter.empty else None,
             str(p.kind)) for p in s.parameters.values()]

def mro(cls):
    return [c.__name__ for c in cls.__mro__ if c.__name__ != "object"]

cam_methods = %(cam)r
gcs_methods = %(gcs)r
# pipython's GCSDevice resolves most commands through __getattr__ onto GCSCommands
try:
    from pipython.pidevice.gcscommands import GCSCommands
except Exception:
    GCSCommands = None
out = {
    "pylablib_version": getattr(pylablib, "__version__", None),
    "pipython_version": getattr(pipython, "__version__", None),
    "simulated": bool(getattr(pylablib, "__simulated__", False) or getattr(pipython, "__simulated__", False)),
    "camera": {m: sig(Thorlabs.ThorlabsTLCamera, m) for m in cam_methods},
    "gcs": {m: (sig(GCSDevice, m) if getattr(GCSDevice, m, None) is not None
                else (sig(GCSCommands, m) if GCSCommands is not None else None)) for m in gcs_methods},
    "mro": {
        "ThorlabsTLCameraTimeoutError": mro(tlc.ThorlabsTLCameraTimeoutError),
        "ThorlabsTLCameraError": mro(Thorlabs.ThorlabsTLCameraError),
        "ThorlabsTLCameraLibError": mro(sdklib.ThorlabsTLCameraLibError),
        "GCSError": mro(GCSError),
    },
    "timeout_is_runtimeerror": issubclass(tlc.ThorlabsTLCameraTimeoutError, RuntimeError),
    "timeout_str": str(tlc.ThorlabsTLCameraTimeoutError()),
    "camera_TimeoutError_attr": getattr(Thorlabs.ThorlabsTLCamera, "TimeoutError", None).__name__,
    "gcserror_is_exception": issubclass(GCSError, Exception),
    "gcserror_is_runtimeerror": issubclass(GCSError, RuntimeError),
    "gcserror_msgs": {str(c): str(GCSError(c)) for c in (-1, -2, -3, -7, -9, 0, 1, 2, 5, 7, 10, 15, 17)},
    "gcserror_val": GCSError(7).val,
    "gcserror_msg_extra": str(GCSError(7, "extra")),
    "thorlabs_names": sorted(n for n in dir(Thorlabs) if n.startswith("ThorlabsTLCam")),
    "namedtuples": {n: list(getattr(tlc, n)._fields) for n in ("TDeviceInfo", "TColorInfo", "TColorFormat")},
}
# Missing DLL: which exception does ConnectUSB raise? (no hardware touched)
try:
    GCSDevice(devname="E-625", gcsdll="/nonexistent/E816_DLL_x64.dll").ConnectUSB(serialnum="1")
    out["missing_dll"] = "no error"
except BaseException as e:
    out["missing_dll"] = [type(e).__name__, isinstance(e, OSError), isinstance(e, GCSError)]
print(json.dumps(out))
"""

_FAKE_BEHAVIOUR = r"""
import json, collections, numpy as np
from sim.world import configure
configure(None, {"camera": {"timing": {k: {"value": 0.0, "origin": "estimated", "source": "test"}
                 for k in ("open_s", "close_s", "arm_s", "disarm_s")}}})
from pylablib.devices import Thorlabs
from pipython import GCSDevice, GCSError
out = {}
cam = Thorlabs.ThorlabsTLCamera()
info = cam.get_device_info()
out["device_info_type"] = type(info).__name__
out["serial"] = info.serial_number
fmt = cam.set_color_format(color_output="raw", color_space="linear")
out["color_format"] = fmt.color_format
ci = cam.get_color_info()
out["phase"] = ci.filter_array_phase
out["set_roi"] = list(cam.set_roi(hbin=1, vbin=1))
cam.set_exposure(0.0070004)
out["exposure_readback"] = cam.get_exposure()
out["exposure_type"] = type(cam.get_exposure()).__name__
cam.start_acquisition()
f = cam.snap(timeout=5.0)
out["snap"] = [type(f).__name__, list(f.shape), str(f.dtype), int(f.max()), bool(f.flags["C_CONTIGUOUS"]),
               bool(f.flags["WRITEABLE"])]
out["armed_after_snap"] = cam.acquisition_in_progress()
# invalid color_output: the real pylablib 1.4.5 does NOT validate it
try:
    cam.set_color_format(color_output="bogus")
    out["bogus_color_output"] = "accepted"
except Exception as e:
    out["bogus_color_output"] = type(e).__name__
cam.set_color_format(color_output="raw")
cam.close()
dev = GCSDevice(devname="E-625", gcsdll="API/PI/E816_DLL_x64.dll")
dev.ConnectUSB(serialnum="000000000")
idn = dev.qIDN()
out["qIDN"] = [type(idn).__name__, idn.endswith("\n"), "SIM" in idn]
out["qERR"] = type(dev.qERR()).__name__
dev.SVO("A", 1)
out["SVO_ret"] = repr(dev.SVO("A", 1))
out["MOV_ret"] = repr(dev.MOV("A", 50.0))
ont = dev.qONT("A")
out["qONT"] = [type(ont).__name__, list(ont.keys()), type(list(ont.values())[0]).__name__]
vol = dev.qVOL("A")
out["qVOL"] = [type(vol).__name__, type(list(vol.values())[0]).__name__]
try:
    dev.MOV("A", 1000.0)
    out["mov_out_of_range"] = "no error"
except GCSError as e:
    out["mov_out_of_range"] = e.val
dev.CloseConnection()
print(json.dumps(out))
"""

# What the app calls (1_hechos s.1.2-1.3) plus what pylablib's snap() calls internally.
APP_CAMERA_CALLS = [
    "__init__",
    "get_device_info",
    "set_color_format",
    "get_color_info",
    "set_roi",
    "set_exposure",
    "get_exposure",
    "start_acquisition",
    "snap",
    "stop_acquisition",
    "close",
]
INTERNAL_CAMERA_CALLS = [
    "open",
    "grab",
    "wait_for_frame",
    "read_multiple_images",
    "setup_acquisition",
    "clear_acquisition",
    "acquisition_in_progress",
    "send_software_trigger",
    "get_sensor_info",
    "get_frame_timings",
    "get_trigger_mode",
    "set_white_balance_matrix",
    "get_color_format",
]
APP_GCS_CALLS = [
    "__init__",
    "ConnectUSB",
    "qIDN",
    "qERR",
    "SVO",
    "MOV",
    "qONT",
    "SVA",
    "qVOL",
    "CloseConnection",
]


def _norm_sig(s):
    """Parameter names + defaults; the kinds of bound methods are compared too."""
    if s is None or isinstance(s, str):
        return s
    return [(n, d) for n, d, _k in s]


def layer_interface(rep: Report) -> None:
    code = _API_DUMP % {"cam": APP_CAMERA_CALLS + INTERNAL_CAMERA_CALLS, "gcs": APP_GCS_CALLS}
    real = _run_json(code, fakes=False)
    fake = _run_json(code, fakes=True)
    L = "interface"
    rep.check(L, "real libraries loaded in the reference process", not real["simulated"])
    rep.check(L, "fakes loaded in the simulator process", fake["simulated"])
    rep.check(
        L,
        "pylablib/pipython versions the fakes mirror",
        real["pylablib_version"] == "1.4.5" and str(real["pipython_version"]).startswith("2.13.0"),
        f"real pylablib {real['pylablib_version']}, pipython {real['pipython_version']} "
        "(fakes mirror 1.4.5 / 2.13.0.2: re-verify the fakes if this changes)",
    )
    for group, names in (
        ("camera", APP_CAMERA_CALLS + INTERNAL_CAMERA_CALLS),
        ("gcs", APP_GCS_CALLS),
    ):
        bad = []
        for m in names:
            r, f = _norm_sig(real[group].get(m)), _norm_sig(fake[group].get(m))
            if r is None:
                continue  # not in the real library either
            if f is None or (r != f and r != "<no signature>"):
                bad.append(f"{m}: real {r} / fake {f}")
        rep.check(
            L, f"{group} signatures (app + internal calls)", not bad, "; ".join(bad) or "all match"
        )
    for k in real["mro"]:
        rep.check(
            L,
            f"MRO {k}",
            real["mro"][k] == fake["mro"][k],
            f"real {real['mro'][k]} / fake {fake['mro'][k]}",
        )
    for k in (
        "timeout_is_runtimeerror",
        "timeout_str",
        "camera_TimeoutError_attr",
        "gcserror_is_exception",
        "gcserror_is_runtimeerror",
        "gcserror_val",
        "gcserror_msg_extra",
        "namedtuples",
    ):
        rep.check(L, k, real[k] == fake[k], f"real {real[k]!r} / fake {fake[k]!r}")
    extra = sorted(set(fake["thorlabs_names"]) - set(real["thorlabs_names"]))
    missing = sorted(set(real["thorlabs_names"]) - set(fake["thorlabs_names"]))
    rep.check(
        L,
        "TLCamera names in pylablib.devices.Thorlabs",
        not missing,
        f"missing in fake {missing}; extra in fake {extra} (extra names are harmless)",
    )
    for k in ():
        rep.check(L, k, real[k] == fake[k], f"real {real[k]!r} / fake {fake[k]!r}")
    diff = {
        c: (real["gcserror_msgs"][c], fake["gcserror_msgs"][c])
        for c in real["gcserror_msgs"]
        if real["gcserror_msgs"][c] != fake["gcserror_msgs"][c]
    }
    rep.check(
        L,
        "GCSError messages for the codes the simulator raises",
        not diff,
        str(diff) if diff else "all match",
    )
    rep.check(
        L,
        "missing DLL -> same exception",
        real["missing_dll"] == fake["missing_dll"],
        f"real {real['missing_dll']} / fake {fake['missing_dll']}",
    )

    b = _run_json(_FAKE_BEHAVIOUR, fakes=True)
    expect = {
        "device_info_type": "TDeviceInfo",
        "color_format": "raw",
        "phase": "blue",
        "exposure_type": "float",
        "armed_after_snap": False,
        "qERR": "int",
        "SVO_ret": "None",
        "MOV_ret": "None",
    }
    for k, v in expect.items():
        rep.check(L, f"fake returns {k}", b[k] == v, f"{b[k]!r} (expected {v!r})")
    rep.check(
        L,
        "exposure truncated to integer us like pylablib",
        abs(b["exposure_readback"] - 0.007) < 1e-12,
        f"set 7.0004 ms -> {b['exposure_readback'] * 1e3:.4f} ms",
    )
    rep.check(
        L,
        "snap(): ndarray (3000, 4096) uint16 <= 4095",
        b["snap"][:3] == ["ndarray", [3000, 4096], "uint16"] and b["snap"][3] <= 4095,
        str(b["snap"]),
    )
    rep.check(
        L,
        "qIDN: str ending in newline, marked SIM",
        b["qIDN"] == ["str", True, True],
        str(b["qIDN"]),
    )
    rep.check(
        L,
        "qONT: OrderedDict keyed by the axis string, bool",
        b["qONT"] == ["OrderedDict", ["A"], "bool"],
        str(b["qONT"]),
    )
    rep.check(
        L, "qVOL: OrderedDict of float", b["qVOL"] == ["OrderedDict", "float"], str(b["qVOL"])
    )
    rep.check(L, "serial marked SIM-", str(b["serial"]).startswith("SIM-"), b["serial"])
    rep.check(
        L,
        "set_color_format('bogus') (real 1.4.5 accepts it without error)",
        None,
        b["bogus_color_output"],
    )
    rep.check(
        L,
        "MOV outside the travel -> GCS error code (E-625 code unverified)",
        None,
        str(b["mov_out_of_range"]),
    )


# ======================================================================
# Layer 2: timing
# ======================================================================
_TIMING = r"""
import json, time, statistics
from sim.world import configure
w = configure(%(profile)r, %(overrides)r)
p = w.profile
from pylablib.devices import Thorlabs
from pipython import GCSDevice
out = {}
t0 = time.perf_counter(); cam = Thorlabs.ThorlabsTLCamera(); out["open"] = time.perf_counter() - t0
cam.get_device_info(); cam.set_color_format(color_output="raw", color_space="linear"); cam.get_color_info()
cam.set_roi(hbin=1, vbin=1)
ts = []
for _ in range(5):
    t0 = time.perf_counter(); cam.set_exposure(0.007); ts.append(time.perf_counter() - t0)
out["set_exposure"] = statistics.median(ts)
t0 = time.perf_counter(); cam.start_acquisition(); out["start_acquisition"] = time.perf_counter() - t0
t0 = time.perf_counter(); cam.snap(timeout=5.0); out["snap_first_armed"] = time.perf_counter() - t0
ts = []
for _ in range(%(n)d):
    t0 = time.perf_counter(); cam.snap(timeout=5.0); ts.append(time.perf_counter() - t0)
out["snap"] = statistics.median(ts); out["snap_max"] = max(ts)
cam.set_exposure(0.030)
ts = []
for _ in range(3):
    t0 = time.perf_counter(); cam.snap(timeout=5.0); ts.append(time.perf_counter() - t0)
out["snap_30ms"] = statistics.median(ts)
t0 = time.perf_counter(); cam.close(); out["close"] = time.perf_counter() - t0
dev = GCSDevice(devname="E-625", gcsdll="API/PI/E816_DLL_x64.dll")
t0 = time.perf_counter(); dev.ConnectUSB(serialnum="000000000"); out["connect"] = time.perf_counter() - t0
dev.SVO("A", 1)
dev.MOV("A", 50.0); time.sleep(0.2)
movs, steps, polls = [], [], []
for i in range(%(n)d):
    z = 50.0 + 0.02 * (i + 1)
    t0 = time.perf_counter(); dev.MOV("A", z); movs.append(time.perf_counter() - t0)
    k = 0
    while True:   # AcquisitionSession._wait_on_target
        k += 1
        if dev.qONT("A")["A"]:
            break
        time.sleep(0.1)
    steps.append(time.perf_counter() - t0); polls.append(k)
out["mov"] = statistics.median(movs); out["move_step"] = statistics.median(steps); out["polls"] = statistics.median(polls)
dev.CloseConnection()
print(json.dumps(out))
"""


def predicted_timings(prof) -> dict:
    """What the profile + pylablib 1.4.5 code say each call should cost."""
    g = lambda k: float(prof[k])  # noqa: E731
    call, arm, disarm = (
        g("camera.timing.sdk_call_s"),
        g("camera.timing.arm_s"),
        g("camera.timing.disarm_s"),
    )
    trig, rd = g("camera.timing.trigger_latency_s"), g("camera.timing.readout_transfer_s")
    q, wr, settle = (
        g("piezo.timing.gcs_query_s"),
        g("piezo.timing.gcs_write_s"),
        g("piezo.timing.settle_s"),
    )

    def snap(exp, armed=False):
        # [stop: 0.2 + disarm if armed] + arm + 0.05 + trigger call + latency + exposure + readout
        # + stop (0.2 s sleep + disarm)
        return (0.2 + disarm if armed else 0.0) + arm + 0.05 + call + trig + exp + rd + 0.2 + disarm

    # MOV = write + ERR?; first qONT answered after one query from MOV's apply
    mov = wr + q
    t_apply = wr
    t = mov
    polls = 0
    while True:
        polls += 1
        t_answer = t + q
        t = t_answer + q  # + ERR?
        if t_answer - t_apply >= settle:
            break
        t += 0.1
    return {
        "open": g("camera.timing.open_s"),
        "set_exposure": 3 * call,
        "start_acquisition": arm + 0.05 + call,
        "snap_first_armed": snap(0.007, armed=True),
        "snap": snap(0.007),
        "snap_30ms": snap(0.030),
        "close": g("camera.timing.close_s"),
        "connect": g("piezo.timing.connect_s"),
        "mov": mov,
        "move_step": t,
        "polls": polls,
    }


def layer_timing(rep: Report, profile: str | None, overrides: dict) -> None:
    from sim.hwprofile import load_profile

    L = "timing"
    prof = load_profile(profile, overrides)
    # --- labels -----------------------------------------------------------
    bad = [
        k
        for k, _v, o, s in prof.figures()
        if o == "published"
        and "http" not in s
        and "page" not in s
        and not s.lower().startswith("same")
    ]
    rep.check(L, "every 'published' figure cites a link or page", not bad, ", ".join(bad) or "ok")
    fps = 21.7
    rd = float(prof["camera.timing.readout_transfer_s"])
    if prof.origin("camera.timing.readout_transfer_s") == "derived":
        rep.check(
            L,
            "readout_transfer_s = 1/21.7 fps (derived)",
            abs(rd - 1 / fps) < 5e-4,
            f"{rd * 1e3:.2f} ms",
        )
    for k in (
        "camera.timing.arm_s",
        "camera.timing.disarm_s",
        "camera.timing.open_s",
        "piezo.timing.gcs_query_s",
        "piezo.timing.settle_s",
    ):
        rep.check(L, f"origin of {k}", None, f"{prof[k]} s, {prof.origin(k)}")
    # --- stopwatch ----------------------------------------------------------
    pred = predicted_timings(prof)
    meas = _run_json(_TIMING % {"profile": profile, "overrides": overrides, "n": 8}, fakes=True)
    for k in (
        "open",
        "set_exposure",
        "start_acquisition",
        "snap_first_armed",
        "snap",
        "snap_30ms",
        "close",
        "connect",
        "mov",
        "move_step",
    ):
        m, p = meas[k], pred[k]
        tol = max(TOL["timing_rel"] * p, TOL["timing_abs_s"])
        rep.check(
            L,
            f"{k}: stopwatch vs profile",
            abs(m - p) <= tol,
            f"measured {m * 1e3:.1f} ms, predicted {p * 1e3:.1f} ms",
            measured_s=m,
            predicted_s=p,
        )
    rep.check(
        L,
        "qONT polls per 0.02 um step",
        meas["polls"] == pred["polls"],
        f"measured {meas['polls']}, predicted {pred['polls']} (each extra poll costs 0.1 s + 2 queries)",
    )
    rep.check(
        L, "snap max over repeats (jitter / generator)", None, f"{meas['snap_max'] * 1e3:.1f} ms"
    )


# ======================================================================
# Layer 3: signal
# ======================================================================
def stack_files(folder: str) -> tuple[np.ndarray, list[str]]:
    items = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in (".tif", ".tiff"):
            m = _Z_RE.search(f)
            if m:
                items.append((float(m.group(1)), os.path.join(folder, f)))
    items.sort()
    return np.array([z for z, _ in items]), [p for _, p in items]


def pixel_grid(h: int, w: int, nr: int = 24, nc: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) grid avoiding a 5 % border.  Row and column parity alternate, so the
    four Bayer sites are sampled equally (a debayered pixel's noise depends on its site)."""
    rows = (np.linspace(0.05 * h, 0.95 * h, nr).astype(int) // 2) * 2 + np.arange(nr) % 2
    cols = (np.linspace(0.05 * w, 0.95 * w, nc).astype(int) // 2) * 2 + np.arange(nc) % 2
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.ravel(), cc.ravel()


def sample_real_stack(folder: str, cache_dir: str, max_frames: int | None = None) -> str:
    """Per-pixel z series of a real stack -> npz in cache_dir (read-only on the stack)."""
    import tifffile

    name = os.path.basename(os.path.normpath(folder))
    out = os.path.join(cache_dir, f"real_{name}_{CACHE_VERSION}.npz")
    if os.path.exists(out):
        return out
    z, files = stack_files(folder)
    if max_frames and len(files) > max_frames:
        keep = np.arange(max_frames)  # contiguous: keeps the z step
        z, files = z[keep], [files[i] for i in keep]
    with tifffile.TiffFile(files[0]) as t:
        page = t.pages[0]
        h, w = page.shape[:2]
        compression = page.compression.name
        software = str(page.tags["Software"].value) if "Software" in page.tags else ""
    rr, cc = pixel_grid(h, w)
    series = np.empty((len(files), len(rr), 3), dtype=np.float32)
    t0 = time.perf_counter()
    for i, f in enumerate(files):
        if compression == "NONE":
            a = tifffile.memmap(f, mode="r")
            series[i] = a[rr, cc, :]
            del a
        else:
            series[i] = tifffile.imread(f)[rr, cc, :]
        if i % 200 == 0:
            print(
                f"    {name}: frame {i}/{len(files)} ({time.perf_counter() - t0:.0f} s)", flush=True
            )
    # one full frame for the histogram (first frame; far from focus in these stacks)
    full = tifffile.imread(files[0])
    pct = np.percentile(full.reshape(-1, 3), [1, 5, 50, 95, 99, 99.9], axis=0)
    sat = (full >= 4095).mean(axis=(0, 1))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        out,
        z=z,
        series=series,
        rows=rr,
        cols=cc,
        shape=np.array([h, w]),
        hist_pct=pct,
        sat=sat,
        compression=compression,
        software=software,
    )
    return out


def analytic_series(
    x: np.ndarray, dz: float, pmin: float = 0.12, pmax: float = 0.6, pbase: float = 1.0
):
    """(baseline, analytic fringe signal) of a z series (last axis), FFT band-pass."""
    n = x.shape[-1]
    k = np.arange(n)
    # remove a linear trend so the FFT wrap-around does not ring
    a = (x[..., -1:] - x[..., :1]) / max(n - 1, 1)
    trend = x[..., :1] + a * k
    y = x - trend
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    Y = np.fft.fft(y, nfft, axis=-1)
    f = np.fft.fftfreq(nfft, dz)
    band = ((f > 1.0 / pmax) & (f < 1.0 / pmin)).astype(float)
    low = (np.abs(f) < 1.0 / pbase).astype(float)
    analytic = np.fft.ifft(2 * Y * band, axis=-1)[..., :n]
    base = np.fft.ifft(Y * low, axis=-1)[..., :n].real + trend
    return base, analytic


def _half_cross(z, e, A, i0, i1):
    """Linear interpolation of the half-maximum crossing between samples i0 (below) and i1 (above)."""
    e0, e1 = e[i0], e[i1]
    return z[i0] + (0.5 * A - e0) / (e1 - e0) * (z[i1] - z[i0])


def series_metrics(z: np.ndarray, S: np.ndarray) -> dict:
    """S: (nz, npix) one channel.  Returns per-pixel metric arrays (NaN = rejected)."""
    dz = float(np.median(np.diff(z)))
    base, an = analytic_series(S.T.astype(np.float64), dz)
    env = np.abs(an)
    npix = S.shape[1]
    out = {
        k: np.full(npix, np.nan) for k in ("level", "vis", "fwhm", "period", "noise", "z0", "amp")
    }
    for j in range(npix):
        e = env[j]
        k0 = int(np.argmax(e))
        A = e[k0]
        half = e >= 0.5 * A
        lo = k0
        while lo > 0 and half[lo - 1]:
            lo -= 1
        hi = k0
        while hi < len(e) - 1 and half[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(e) - 1 or hi - lo < 4:
            continue

        fwhm = _half_cross(z, e, A, hi + 1, hi) - _half_cross(z, e, A, lo - 1, lo)
        ph = np.unwrap(np.angle(an[j, lo : hi + 1]))
        wts = e[lo : hi + 1] ** 2
        slope = np.polyfit(z[lo : hi + 1], ph, 1, w=np.sqrt(wts))[0]
        far = np.abs(z - z[k0]) > max(2.5 * fwhm, 3.0)
        # noise from first differences of consecutive FAR frames (robust MAD)
        idx = np.where(far[:-1] & far[1:])[0]
        if len(idx) < 20:
            continue
        d = S[idx + 1, j].astype(np.float64) - S[idx, j]
        d = d - np.median(d)
        mad = 1.4826 * np.median(np.abs(d)) + 1e-9
        noise = float(np.std(d[np.abs(d) <= 5 * mad])) / np.sqrt(
            2
        )  # clipped std: MAD is quantised on DN
        lvl = float(base[j, k0])
        out["level"][j] = lvl
        out["amp"][j] = A
        out["vis"][j] = A / lvl if lvl > 0 else np.nan
        out["fwhm"][j] = fwhm
        out["period"][j] = 2 * np.pi / abs(slope) if slope else np.nan
        out["noise"][j] = noise
        out["z0"][j] = z[k0]
    # quality: clear fringe well inside the range
    ok = (out["amp"] > 6 * np.maximum(out["noise"], 0.5)) & np.isfinite(out["fwhm"])
    for k in out:
        out[k] = np.where(ok, out[k], np.nan)
    return out


def frame_z_jitter_nm(z: np.ndarray, S: np.ndarray) -> float:
    """RMS z error shared by all pixels of a frame (piezo position error or vibration), in nm.

    Each pixel's fringe is modelled by its band-passed analytic signal f_j(z); for every
    frame k the common shift d_k that best explains the residuals of all pixels near focus
    is the least-squares solution of r_jk = f_j'(z_k) d_k.  Pixel noise averages out over
    the pixels, a common displacement does not.  The band-pass keeps part of the jitter in
    the model, so the estimate is slightly low (~0.8x, calibrated on simulated frames with
    a known Gaussian stage error: see 3_verificacion.md)."""
    dz = float(np.median(np.diff(z)))
    base, an = analytic_series(S.T.astype(np.float64), dz)
    env = np.abs(an)
    # local fringe frequency per pixel from the phase slope near focus
    num = np.zeros(len(z))
    den = np.zeros(len(z))
    cnt = np.zeros(len(z), dtype=int)
    for j in range(S.shape[1]):
        e = env[j]
        k0 = int(np.argmax(e))
        near = e >= 0.5 * e[k0]
        idx = np.where(near)[0]
        if len(idx) < 8 or e[k0] <= 0:
            continue
        ph = np.unwrap(np.angle(an[j, idx]))
        slope = np.polyfit(z[idx], ph, 1)[0]  # rad / um
        model = base[j] + an[j].real
        grad = -slope * an[j].imag  # d/dz of Re(A e^{i phi}) with phi' = slope (envelope slow)
        r = S[:, j] - model
        num[idx] += r[idx] * grad[idx]
        den[idx] += grad[idx] ** 2
        cnt[idx] += 1
    ok = cnt >= 20
    if not ok.any():
        return float("nan")
    d = num[ok] / den[ok] * 1000.0  # nm
    return float(np.sqrt(np.mean((d - np.mean(d)) ** 2)))


def slow_z_wander_nm(z: np.ndarray, S: np.ndarray) -> float:
    """RMS of the SLOW z error common to all pixels (within the fringe band: over a few
    frames), in nm: per-pixel phase residuals of the band-passed analytic signal after a
    linear fit near focus, median over pixels per frame.  Frame-to-frame white jitter is
    mostly filtered out here (8 nm white reads ~1.3 nm); drift, vibration at low frequency
    or an uneven piezo step show up in full.  This is what corrupts period/FWHM estimates."""
    dz = float(np.median(np.diff(z)))
    _base, an = analytic_series(S.T.astype(np.float64), dz)
    env = np.abs(an)
    R = np.full((len(z), S.shape[1]), np.nan)
    for j in range(S.shape[1]):
        e = env[j]
        k0 = int(np.argmax(e))
        idx = np.where(e >= 0.6 * e[k0])[0]
        idx = idx[np.abs(idx - k0) <= 40]
        if len(idx) < 8:
            continue
        ph = np.unwrap(np.angle(an[j, idx]))
        c = np.polyfit(z[idx], ph, 1)
        if not c[0]:
            continue
        R[idx, j] = (ph - np.polyval(c, z[idx])) / c[0] * 1000.0
    ok = np.sum(np.isfinite(R), axis=1) >= 20
    if not ok.any():
        return float("nan")
    cm = np.nanmedian(R[ok], axis=1)
    return float(np.sqrt(np.mean(cm**2)))


#: Above this slow z wander a stack is not used by the gate (informational only).
MAX_WANDER_NM = 20.0
#: Bump when the sampling (pixel grid, stored fields) changes, so old caches are ignored.
CACHE_VERSION = "v2"


def summarize(m: dict) -> dict:
    s = {}
    for k, v in m.items():
        v = v[np.isfinite(v)]
        s[k] = (
            (
                float(np.median(v)),
                float(np.percentile(v, 25)),
                float(np.percentile(v, 75)),
                int(v.size),
            )
            if v.size
            else (np.nan, np.nan, np.nan, 0)
        )
    lv, nz = m["level"], m["noise"]
    g = lv / nz**2
    g = g[np.isfinite(g)]
    s["level_per_noise2"] = (
        (float(np.median(g)), float(np.percentile(g, 25)), float(np.percentile(g, 75)), int(g.size))
        if g.size
        else (np.nan, np.nan, np.nan, 0)
    )
    return s


def identify_channels(summ: list[dict]) -> dict:
    """Map R/G/B -> channel index by fringe period: longest = R, then G, shortest = B."""
    per = [s["period"][0] for s in summ]
    order = np.argsort(per)[::-1]
    return {"R": int(order[0]), "G": int(order[1]), "B": int(order[2])}


def simulate_like(npz: dict, profile: str | None, overrides: dict, window_um: float = 6.0) -> dict:
    """Simulated series at the real stack's z positions, through both fusions."""
    from pylablib.devices.utils import color as plcolor

    from backend.acquisition.acquisition_controller import AcquisitionSession
    from sim.hwprofile import load_profile
    from sim.signal_model import RED_SITE, SyntheticSource

    z = npz["z"]
    zc = 0.5 * (z[0] + z[-1])
    over = json.loads(json.dumps(overrides)) if overrides else {}
    over.setdefault("surface", {})["base_height_um"] = float(round(zc, 3))
    prof = load_profile(profile, over)
    H, W = int(prof["camera.sensor.height"]), int(prof["camera.sensor.width"])
    phase = str(prof["camera.sensor.filter_array_phase"])
    src = SyntheticSource(
        prof, H, W, phase, rng=np.random.default_rng(int(prof.get("signal.seed", 0)))
    )
    sel = np.where(np.abs(z - zc) <= window_um)[0]
    zs = z[sel]
    rr, cc = pixel_grid(H, W)
    exp = float(prof["signal.reference_exposure_s"])
    sp = np.empty((len(zs), len(rr), 3), np.float32)
    pl = np.empty((len(zs), len(rr), 3), np.float32)
    off = RED_SITE[phase]
    # the camera samples the STAGE, which lands on each set-point with a random error
    zerr = np.random.default_rng(1).normal(
        0.0, float(prof["piezo.stage.position_noise_um"]), len(zs)
    )
    t0 = time.perf_counter()
    for i, zz in enumerate(zs):
        raw = src.generate(float(zz + zerr[i]), exp)
        sp[i] = AcquisitionSession._bayer_to_color_superpixel(raw, phase)[rr // 2, cc // 2, :]
        # pylablib 1.4.5 _debayer with identity matrices (colour matrices of the real camera unknown)
        pl[i] = np.clip(plcolor.bayer_interpolate(raw.astype(float), off=off), 0, None)[rr, cc, :]
    full_sp = AcquisitionSession._bayer_to_color_superpixel(src.generate(float(z[0]), exp), phase)
    print(f"    simulated {len(zs)} frames in {time.perf_counter() - t0:.0f} s", flush=True)
    return {
        "z": zs,
        "superpixel": sp,
        "pylablib": pl,
        "hist_pct": np.percentile(full_sp.reshape(-1, 3), [1, 5, 50, 95, 99, 99.9], axis=0),
        "sat": (full_sp >= 4095).mean(axis=(0, 1)),
    }


def layer_signal(rep: Report, profile, overrides, real_dirs, cache_dir, max_frames) -> dict:
    from sim.hwprofile import load_profile

    L = "signal"
    prof = load_profile(profile, overrides)
    results = {}
    found = False
    sim_cache: dict = {}
    for d in real_dirs:
        name = os.path.basename(os.path.normpath(d))
        cached = os.path.join(cache_dir, f"real_{name}_{CACHE_VERSION}.npz")
        if not os.path.exists(cached):
            if not os.path.isdir(d):
                rep.check(L, f"{name}: real stack", None, "not found, skipped")
                continue
            try:
                sample_real_stack(d, cache_dir, max_frames)
            except Exception as e:  # noqa: BLE001 - e.g. LZW without imagecodecs
                rep.check(
                    L,
                    f"{name}: real stack",
                    None,
                    f"cannot sample ({type(e).__name__}: {e}); use the 'sample' sub-command with an "
                    "interpreter that has imagecodecs",
                )
                continue
        found = True
        npz = dict(np.load(cached, allow_pickle=False))
        z = npz["z"]
        dz = float(np.median(np.diff(z)))
        real_s = [summarize(series_metrics(z, npz["series"][:, :, c])) for c in range(3)]
        ch = identify_channels(real_s)
        jit = frame_z_jitter_nm(z, npz["series"][:, :, ch["G"]])
        wander = slow_z_wander_nm(z, npz["series"][:, :, ch["G"]])
        key = round(dz, 4)
        if key not in sim_cache:
            sim = simulate_like(npz, profile, overrides)
            sim_cache[key] = {
                fusion: [
                    summarize(series_metrics(sim["z"], sim[fusion][:, :, c])) for c in range(3)
                ]
                for fusion in ("superpixel", "pylablib")
            } | {
                "hist_pct": sim["hist_pct"],
                "sat": sim["sat"],
                "jitter_nm": frame_z_jitter_nm(sim["z"], sim["superpixel"][:, :, 1]),
                "wander_nm": slow_z_wander_nm(sim["z"], sim["superpixel"][:, :, 1]),
            }
        sim_s = sim_cache[key]
        if not sim_cache.get("selfchecked"):
            sim_cache["selfchecked"] = True
            # the analysis must recover the profile's own figures from simulated frames
            # WITHOUT stage error (z jitter narrows the measured envelope: that is physics,
            # not an analysis error, and it is compared against the real stacks below)
            quiet = json.loads(json.dumps(overrides)) if overrides else {}
            quiet.setdefault("piezo", {}).setdefault("stage", {})["position_noise_um"] = {
                "value": 0.0,
                "origin": "estimated",
                "source": "self-check",
            }
            zq = 50.0 + np.arange(
                -12.0, 12.0 + 1e-9, 0.02
            )  # long enough for the far-field noise of R
            sq = simulate_like({"z": zq}, profile, quiet, window_um=12.0)
            for i, c in enumerate("rg"):
                mm = summarize(series_metrics(sq["z"], sq["superpixel"][:, :, i]))
                for metric, fig, tol in (
                    ("period", f"signal.period_{c}_um", 0.02),
                    ("fwhm", f"signal.envelope_fwhm_{c}_um", 0.10),
                    ("vis", f"signal.visibility_{c}", 0.10),
                ):
                    got, want = mm[metric][0], float(prof[fig])
                    rep.check(
                        L,
                        f"metric self-check: {fig} recovered from noise-free simulated frames",
                        abs(got - want) <= tol * want,
                        f"{got:.3f} vs profile {want:.3f}",
                    )
        res = {
            "dz": dz,
            "n": len(z),
            "jitter_nm": jit,
            "sim_jitter_nm": sim_s["jitter_nm"],
            "wander_nm": wander,
            "sim_wander_nm": sim_s["wander_nm"],
            "z_range": [float(z[0]), float(z[-1])],
            "channels": ch,
            "real": {c: real_s[i] for c, i in ch.items()},
            "sim_superpixel": dict(zip("RGB", sim_s["superpixel"], strict=True)),
            "sim_pylablib": dict(zip("RGB", sim_s["pylablib"], strict=True)),
            "real_hist_pct": {c: npz["hist_pct"][:, i].tolist() for c, i in ch.items()},
            "real_sat": {c: float(npz["sat"][i]) for c, i in ch.items()},
            "sim_hist_pct": {c: sim_s["hist_pct"][:, i].tolist() for i, c in enumerate("RGB")},
            "software": str(npz.get("software", "")),
            "compression": str(npz.get("compression", "")),
        }
        results[name] = res
        usable = bool(np.isfinite(wander) and wander <= MAX_WANDER_NM)
        tag = f"{name} (slow z wander {wander:.0f} nm{'' if usable else f' > {MAX_WANDER_NM:.0f}: NOT gated'}; "
        tag += f"dz {dz * 1000:.0f} nm, file channels "
        tag += f"R={ch['R']} G={ch['G']} B={ch['B']})"
        print(f"\n  -- {tag}")
        print(
            f"     z error estimates (nm): frame-to-frame real {jit:.1f} / sim {sim_s['jitter_nm']:.1f}; "
            f"slow wander real {wander:.1f} / sim {sim_s['wander_nm']:.1f}"
        )
        print(
            f"     G histogram p1/p50/p99 real "
            f"{[round(v) for v in np.array(res['real_hist_pct']['G'])[[0, 2, 4]]]} sim "
            f"{[round(v) for v in np.array(res['sim_hist_pct']['G'])[[0, 2, 4]]]}; saturated real "
            f"{res['real_sat']} "
        )
        fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}-{t[2]:.3f}] n={t[3]}"  # noqa: E731
        for c in "RGB":
            r, s = res["real"][c], res["sim_superpixel"][c]
            print(f"     {c}: period real {fmt(r['period'])} | sim {fmt(s['period'])}")
            print(f"        fwhm   real {fmt(r['fwhm'])} | sim {fmt(s['fwhm'])}")
            print(f"        vis    real {fmt(r['vis'])} | sim {fmt(s['vis'])}")
            print(
                f"        level  real {r['level'][0]:.0f} | sim {s['level'][0]:.0f};"
                f" noise real {r['noise'][0]:.2f} | sim-sp {s['noise'][0]:.2f} |"
                f" sim-pl {res['sim_pylablib'][c]['noise'][0]:.2f};"
                f" level/noise^2 real {r['level_per_noise2'][0]:.1f} | sim-sp {s['level_per_noise2'][0]:.1f}"
                f" | sim-pl {res['sim_pylablib'][c]['level_per_noise2'][0]:.1f}"
            )
        gate = (lambda ok: ok) if usable else (lambda ok: None)  # noqa: E731
        rep.check(
            L,
            f"{name} frame-to-frame z error (common to all pixels)",
            gate(abs(sim_s["jitter_nm"] - jit) <= max(0.5 * jit, 3.0)),
            f"real {jit:.1f} nm vs sim {sim_s['jitter_nm']:.1f} nm (estimator reads ~0.8x the true rms)",
        )
        for c in ("R", "G"):
            r, s = res["real"][c]["period"][0], res["sim_superpixel"][c]["period"][0]
            rep.check(
                L,
                f"{name} period {c}",
                gate(abs(s - r) <= TOL["period_rel"] * r),
                f"sim {s:.3f} vs real {r:.3f} um ({(s / r - 1) * 100:+.1f} %)",
            )
            r, s = res["real"][c]["fwhm"][0], res["sim_superpixel"][c]["fwhm"][0]
            rep.check(
                L,
                f"{name} envelope FWHM {c}",
                gate(abs(s - r) <= TOL["fwhm_rel"] * r),
                f"sim {s:.2f} vs real {r:.2f} um ({(s / r - 1) * 100:+.1f} %)",
            )
        r, s = res["real"]["G"]["level"][0], res["sim_superpixel"]["G"]["level"][0]
        rep.check(
            L,
            f"{name} G background level",
            gate(abs(s - r) <= TOL["level_g_rel"] * r),
            f"sim {s:.0f} vs real {r:.0f} DN ({(s / r - 1) * 100:+.1f} %; real exposure unknown)",
        )
        rep.check(
            L,
            f"{name} B period / visibility (real B is colour-corrected: informational)",
            None,
            f"real {res['real']['B']['period'][0]:.3f} um / V {res['real']['B']['vis'][0]:.2f}; "
            f"sim {res['sim_superpixel']['B']['period'][0]:.3f} um / V {res['sim_superpixel']['B']['vis'][0]:.2f}",
        )
    if not found:
        rep.check(L, "real stacks", None, "none available: signal layer not compared")
    return results


# ======================================================================
# CLI
# ======================================================================
def _overrides_from_set(items) -> dict:
    over: dict = {}
    for item in items or []:
        key, val = item.split("=", 1)
        node = over
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        node[parts[-1]] = {"value": v, "origin": "estimated", "source": "compare_real_vs_sim --set"}
    return over


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", nargs="?", default="compare", choices=("compare", "sample"))
    ap.add_argument("--layers", default="interface,timing,signal")
    ap.add_argument(
        "--profile", default=None, help="profile TOML merged over sim/profiles/default.toml"
    )
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a profile figure")
    ap.add_argument(
        "--real",
        action="append",
        help="real stack folder (repeatable); default: the LP126CU stacks",
    )
    ap.add_argument(
        "--cache",
        default=os.path.join(tempfile.gettempdir(), "interferolab_compare_cache"),
        help="where the sampled real series are cached (never inside the stacks)",
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--json", help="write every number to this JSON file")
    args = ap.parse_args(argv)
    real_dirs = args.real or DEFAULT_REAL
    os.makedirs(args.cache, exist_ok=True)

    if args.command == "sample":
        for d in real_dirs:
            if os.path.isdir(d):
                print(f"sampling {d}")
                print("  ->", sample_real_stack(d, args.cache, args.max_frames))
        return 0

    sys.path[:0] = [p for p in (ROOT,) if p not in sys.path]
    overrides = _overrides_from_set(args.set)
    rep = Report()
    out: dict = {"profile": args.profile, "overrides": overrides}
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    if "interface" in layers:
        print("\n== interface")
        layer_interface(rep)
    if "timing" in layers:
        print("\n== timing")
        layer_timing(rep, args.profile, overrides)
    if "signal" in layers:
        print("\n== signal")
        out["signal"] = layer_signal(
            rep, args.profile, overrides, real_dirs, args.cache, args.max_frames
        )
    out["checks"] = rep.rows
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
            )
    n_fail, n_chk = len(rep.failed), len(rep.checked)
    print(
        f"\n{n_chk - n_fail}/{n_chk} checks passed"
        + (f"; FAILED: {[r['name'] for r in rep.failed]}" if n_fail else "")
    )
    if n_chk == 0:
        return 2
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
