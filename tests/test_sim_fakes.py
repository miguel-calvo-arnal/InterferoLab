"""Simulator fakes (sim/fakes/pylablib, sim/fakes/pipython): API surface, timing, faults.

The fakes carry the SAME package names as the real drivers, so they are
only ever imported in a subprocess with sim/fakes first on sys.path; this
test process keeps the real pylablib/pipython from .venv and compares.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKES = os.path.join(ROOT, "sim", "fakes")

CAMERA_METHODS = [
    "__init__",
    "open",
    "close",
    "is_opened",
    "get_device_info",
    "get_sensor_info",
    "get_color_info",
    "set_color_format",
    "get_color_format",
    "set_white_balance_matrix",
    "get_white_balance_matrix",
    "set_roi",
    "get_roi",
    "get_detector_size",
    "set_exposure",
    "get_exposure",
    "get_frame_timings",
    "get_frame_period",
    "get_trigger_mode",
    "start_acquisition",
    "stop_acquisition",
    "acquisition_in_progress",
    "setup_acquisition",
    "clear_acquisition",
    "send_software_trigger",
    "wait_for_frame",
    "read_multiple_images",
    "read_newest_image",
    "read_oldest_image",
    "get_new_images_range",
    "grab",
    "snap",
]
PIEZO_METHODS = [
    "ConnectUSB",
    "qIDN",
    "qERR",
    "SVO",
    "qSVO",
    "MOV",
    "qMOV",
    "qPOS",
    "qONT",
    "SVA",
    "qSVA",
    "qVOL",
    "CloseConnection",
    "close",
]
NAMEDTUPLES = ["TDeviceInfo", "TSensorInfo", "TColorInfo", "TColorFormat", "TFrameInfo"]
GCS_CODES = [-9, -3, -2, -1, 0, 1, 2, 5, 7, 10, 15, 17]

_PRELUDE = f"""
import sys, json, time
sys.path.insert(0, {FAKES!r})
sys.path.insert(1, {ROOT!r})
"""


def run_with_fakes(code: str, timeout: float = 120, setup: str = "") -> dict:
    """Run *code* with the fakes first on sys.path; it must print one JSON line last."""
    proc = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(setup) + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------
# API surface vs the real libraries in .venv
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake_api():
    methods = json.dumps(CAMERA_METHODS)
    pmethods = json.dumps(PIEZO_METHODS)
    tuples = json.dumps(NAMEDTUPLES)
    codes = json.dumps(GCS_CODES)
    return run_with_fakes(f"""
        import inspect
        import pylablib, pipython
        from pylablib.devices import Thorlabs
        from pylablib.devices.Thorlabs import TLCamera as T
        from pylablib.devices.Thorlabs.tl_camera_sdk_lib import ThorlabsTLCameraLibError
        from pipython import GCSDevice, GCSError
        out = {{
            "files": [pylablib.__file__, pipython.__file__],
            "cam": {{m: str(inspect.signature(getattr(Thorlabs.ThorlabsTLCamera, m))) for m in {methods}}},
            "piezo": {{m: str(inspect.signature(getattr(GCSDevice, m))) for m in {pmethods}}},
            "gcsdevice_init": str(inspect.signature(GCSDevice.__init__)),
            "gcserror_init": str(inspect.signature(GCSError.__init__)),
            "liberror_init": str(inspect.signature(ThorlabsTLCameraLibError.__init__)),
            "tuples": {{n: list(getattr(T, n)._fields) for n in {tuples}}},
            "timeout_mro": [c.__name__ for c in Thorlabs.ThorlabsTLCameraTimeoutError.__mro__],
            "gcserror_mro": [c.__name__ for c in GCSError.__mro__],
            "gcs_msgs": {{str(c): str(GCSError(c)) for c in {codes}}},
            "timeout_str": str(Thorlabs.ThorlabsTLCameraTimeoutError()),
        }}
        print(json.dumps(out))
    """)


def test_fakes_are_the_ones_loaded(fake_api):
    for f in fake_api["files"]:
        assert os.path.abspath(f).startswith(FAKES + os.sep)


def test_camera_signatures_match_real_pylablib(fake_api):
    from pylablib.devices.Thorlabs import TLCamera as real

    assert not getattr(__import__("pylablib"), "__simulated__", False)
    for m in CAMERA_METHODS:
        assert fake_api["cam"][m] == str(inspect.signature(getattr(real.ThorlabsTLCamera, m))), m
    for n in NAMEDTUPLES:
        assert fake_api["tuples"][n] == list(getattr(real, n)._fields), n
    from pylablib.devices.Thorlabs.tl_camera_sdk_lib import ThorlabsTLCameraLibError

    assert fake_api["liberror_init"] == str(inspect.signature(ThorlabsTLCameraLibError.__init__))


def test_piezo_signatures_match_real_pipython(fake_api):
    import pipython
    from pipython.pidevice.gcs2.gcs2device import GCS2Device

    for m in PIEZO_METHODS:
        assert fake_api["piezo"][m] == str(inspect.signature(getattr(GCS2Device, m))), m
    assert fake_api["gcsdevice_init"] == str(inspect.signature(pipython.GCSDevice.__init__))
    assert fake_api["gcserror_init"] == str(inspect.signature(pipython.GCSError.__init__))


def test_exception_hierarchies_and_messages_match(fake_api):
    import pipython
    from pylablib.devices import Thorlabs

    assert fake_api["timeout_mro"] == [
        c.__name__ for c in Thorlabs.ThorlabsTLCameraTimeoutError.__mro__
    ]
    assert fake_api["gcserror_mro"] == [c.__name__ for c in pipython.GCSError.__mro__]
    for code, msg in fake_api["gcs_msgs"].items():
        assert msg == str(pipython.GCSError(int(code))), code
    assert fake_api["timeout_str"] == str(Thorlabs.ThorlabsTLCameraTimeoutError()) == ""


# ----------------------------------------------------------------------
# Behaviour and timing (default profile = the app's real call sequence)
# ----------------------------------------------------------------------
def test_camera_connect_and_snap_like_the_app():
    r = run_with_fakes("""
        from pylablib.devices import Thorlabs
        t = time.monotonic(); cam = Thorlabs.ThorlabsTLCamera(); t_open = time.monotonic() - t
        info = cam.get_device_info()
        fmt = cam.set_color_format(color_output="raw", color_space="linear")
        phase = cam.get_color_info().filter_array_phase
        roi = cam.set_roi(hbin=1, vbin=1)
        cam.set_exposure(0.007)
        exp = cam.get_exposure()
        cam.set_exposure(0.0070009)
        exp_trunc = cam.get_exposure()
        cam.set_exposure(0.007)
        cam.start_acquisition()
        armed_after_connect = cam.acquisition_in_progress()
        snaps = []
        for i in range(3):
            t = time.monotonic(); f = cam.snap(timeout=5.0); snaps.append(time.monotonic() - t)
        print(json.dumps({"t_open": t_open, "serial": info.serial_number, "model": info.model,
                          "fmt": list(fmt), "phase": phase, "roi": list(roi), "exp": exp,
                          "exp_trunc": exp_trunc, "armed": armed_after_connect,
                          "armed_after_snap": cam.acquisition_in_progress(),
                          "snaps": snaps, "shape": list(f.shape), "dtype": str(f.dtype),
                          "max": int(f.max())}))
        cam.close()
    """)
    assert r["serial"].startswith("SIM-") and "SIMULATION" in r["model"]
    assert r["fmt"] == ["raw", "linear"]
    assert r["phase"] == "blue"
    assert r["roi"] == [0, 4096, 0, 3000, 1, 1]
    assert r["exp"] == 7000 * 1e-6  # same float as pylablib: int(us) * 1e-6
    assert r["exp_trunc"] == 7000 * 1e-6  # truncated to whole microseconds
    assert r["armed"] and not r["armed_after_snap"]
    assert r["shape"] == [3000, 4096] and r["dtype"] == "uint16" and r["max"] <= 4095
    assert r["t_open"] == pytest.approx(0.8, abs=0.15)
    # every snap: arm 0.15 + 0.05 + 1 ms + 7 ms + 46.1 ms + 0.2 + disarm 0.05 = 0.504 s;
    # the first one also disarms what connect_camera left armed (+0.2 +0.05)
    assert r["snaps"][0] == pytest.approx(0.754, abs=0.08)
    for s in r["snaps"][1:]:
        assert s == pytest.approx(0.504, abs=0.06)


_FAST = """
from sim.world import configure
def fig(v): return {"value": v, "origin": "estimated", "source": "test"}
FAST = {"camera": {"timing": {k: fig(0.0) for k in ("open_s", "close_s", "sdk_call_s", "arm_s", "disarm_s", "trigger_latency_s")},
                   "sensor": {"height": fig(32), "width": fig(48)}},
        "piezo": {"timing": {k: fig(0.0) for k in ("connect_s", "close_s", "gcs_write_s", "gcs_query_s")}}}
def world(**faults):
    over = dict(FAST); over["faults"] = faults
    return configure(None, over)
"""


def test_camera_fault_injection():
    r = run_with_fakes(
        setup=_FAST,
        code="""
        from pylablib.devices import Thorlabs
        from pylablib.devices.Thorlabs.tl_camera_sdk_lib import ThorlabsTLCameraLibError
        out = {}
        world(snap_timeout_every=2)
        cam = Thorlabs.ThorlabsTLCamera(); cam.set_color_format(color_output="raw")
        cam.snap(timeout=1.0)
        t = time.monotonic()
        try:
            cam.snap(timeout=0.3); out["timeout"] = None
        except Thorlabs.ThorlabsTLCameraTimeoutError as e:
            out["timeout"] = [type(e).__name__, str(e), isinstance(e, RuntimeError), time.monotonic() - t]
        out["after_timeout_ok"] = cam.snap(timeout=1.0).shape == (32, 48)
        cam.close()

        world(exposure_apply_factor=1.2)
        cam = Thorlabs.ThorlabsTLCamera(); cam.set_exposure(0.01)
        out["exposure_applied"] = cam.get_exposure(); cam.close()

        world(exposure_ignored=True)
        cam = Thorlabs.ThorlabsTLCamera(); cam.set_exposure(0.002)
        out["exposure_ignored"] = cam.get_exposure(); cam.close()

        world(camera_disconnect_after_snaps=2)
        cam = Thorlabs.ThorlabsTLCamera(); cam.snap(); cam.snap()
        try:
            cam.snap(); out["unplug"] = None
        except ThorlabsTLCameraLibError as e:
            out["unplug"] = [str(e), isinstance(e, RuntimeError)]
        try:
            Thorlabs.ThorlabsTLCamera(); out["reopen"] = None
        except Exception as e:
            out["reopen"] = type(e).__name__
        print(json.dumps(out))
    """,
    )
    name, msg, is_rt, dt = r["timeout"]
    assert name == "ThorlabsTLCameraTimeoutError" and msg == "" and is_rt
    assert dt == pytest.approx(0.3 + 0.2, abs=0.15)  # plazo + pylablib's disarm sleep
    assert r["after_timeout_ok"]
    assert r["exposure_applied"] == pytest.approx(0.012)
    assert r["exposure_ignored"] == pytest.approx(0.010)  # still the 10 ms initial value
    assert r["unplug"] is not None and "disconnected" in r["unplug"][0] and r["unplug"][1]
    assert r["reopen"] is not None


def test_piezo_gcs_behaviour_and_errors():
    r = run_with_fakes("""
        from pipython import GCSDevice, GCSError
        out = {}
        try:
            GCSDevice(devname="E-625", gcsdll="/nonexistent/E816_DLL_x64.dll").ConnectUSB(serialnum="1")
        except OSError as e:
            out["missing_dll"] = str(e)
        dev = GCSDevice(devname="E-625", gcsdll="API/PI/E816_DLL_x64.dll")
        dev.ConnectUSB(serialnum="000000000")
        out["idn"] = dev.qIDN(); out["err"] = dev.qERR()
        try:
            dev.MOV("A", 50.0)
        except GCSError as e:
            out["servo_off"] = e.val
        dev.SVO("A", 1)
        try:
            dev.MOV("A", 150.0)
        except GCSError as e:
            out["out_of_range"] = e.val
        try:
            dev.MOV("Z", 1.0)
        except GCSError as e:
            out["bad_axis"] = e.val
        t = time.monotonic(); dev.MOV("A", 50.0); out["t_mov"] = time.monotonic() - t
        t = time.monotonic(); o = dev.qONT("A"); out["t_ont"] = time.monotonic() - t
        out["ont_first"] = [type(o).__name__, list(o.keys()), o["A"]]
        time.sleep(0.05)
        out["ont_later"] = dev.qONT("A")["A"]
        out["pos"] = dev.qPOS("A")["A"]
        out["vol_keys"] = list(dev.qVOL("A").keys())
        dev.CloseConnection()
        try:
            dev.qONT("A")
        except GCSError as e:
            out["closed"] = e.val
        print(json.dumps(out))
    """)
    assert "not found" in r["missing_dll"]
    assert "SIM-000000000" in r["idn"] and r["idn"].endswith("\n") and r["err"] == 0
    assert r["servo_off"] == 5 and r["out_of_range"] == 7 and r["bad_axis"] == 15
    # MOV = write 2 ms + ERR? 10 ms; qONT = query 10 ms + ERR? 10 ms (pipython errcheck)
    assert r["t_mov"] == pytest.approx(0.012, abs=0.006)
    assert r["t_ont"] == pytest.approx(0.020, abs=0.008)
    assert r["ont_first"] == ["OrderedDict", ["A"], False]
    assert r["ont_later"] is True
    assert r["pos"] == pytest.approx(50.0, abs=0.01)
    assert r["vol_keys"] == ["A"]
    assert r["closed"] == -9


def test_piezo_fault_injection():
    r = run_with_fakes(
        setup=_FAST,
        code="""
        from pipython import GCSDevice, GCSError
        out = {}
        def dev():
            d = GCSDevice(devname="E-625", gcsdll=""); d.ConnectUSB(serialnum="1"); d.SVO("A", 1); return d
        world(mov_error_probability=1.0, mov_error_code=-1)
        try:
            dev().MOV("A", 10.0); out["mov"] = None
        except GCSError as e:
            out["mov"] = [e.val, str(e)]
        world(piezo_never_on_target=True)
        d = dev(); d.MOV("A", 10.0); time.sleep(0.1)
        out["never"] = d.qONT("A")["A"]
        world(piezo_disconnect_after_commands=4)
        d = dev(); d.MOV("A", 10.0)
        try:
            d.qONT("A"); d.qONT("A"); out["unplug"] = None
        except GCSError as e:
            out["unplug"] = e.val
        print(json.dumps(out))
    """,
    )
    assert r["mov"][0] == -1 and "com operation" in r["mov"][1]
    assert r["never"] is False
    assert r["unplug"] in (-2, -3)


def test_camera_sees_the_piezo_position():
    """Frames at the surface height show fringes; 8 um away they do not."""
    r = run_with_fakes(
        setup=_FAST,
        code="""
        world()
        from pylablib.devices import Thorlabs
        from pipython import GCSDevice
        cam = Thorlabs.ThorlabsTLCamera(); cam.set_color_format(color_output="raw")
        d = GCSDevice(devname="E-625", gcsdll=""); d.ConnectUSB(serialnum="1"); d.SVO("A", 1)
        out = {}
        for z in (42.0, 50.0):
            d.MOV("A", z); time.sleep(0.05)
            f = cam.snap().astype(float)[0::2, 1::2]  # G sites
            out[str(z)] = float(f.std())
        print(json.dumps(out))
    """,
    )
    assert r["50.0"] > 5 * r["42.0"]
