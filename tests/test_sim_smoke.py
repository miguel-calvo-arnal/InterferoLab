"""End-to-end smoke test: the real app, started through the simulator launcher.

In a subprocess (the fakes shadow the real drivers there): bootstrap the
launcher with the DEFAULT profile (faithful timings), build MainWindow
offscreen, connect camera + piezo through the panel, take previews (a single
snapshot and the continuous stream), move the piezo, run a short sweep,
disconnect.
Also runs the lab probe in --dry-run against the fakes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")

DRIVER = r"""
import sys, os, json, time
sys.path.insert(0, SIM_DIR)
import run_simulated
args, world = run_simulated.bootstrap([])

import views.MainWindow as mw
mw.load_config = lambda: {}          # never touch the real app_config.json
mw.save_config = lambda cfg: None
import views.AcquisitionPanel as ap
dialogs = []
class _NoModal:
    @staticmethod
    def critical(parent, title, text, *a, **k): dialogs.append(["critical", title, text])
    @staticmethod
    def warning(parent, title, text, *a, **k): dialogs.append(["warning", title, text])
ap.QMessageBox = _NoModal            # a modal dialog would block offscreen

from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
win = mw.MainWindow()
win.show()
res = {"title": win.windowTitle(),
       "banner": win.menuWidget().objectName() if win.menuWidget() else None,
       "banner_text": win.menuWidget().text() if win.menuWidget() else ""}
win.setWindowTitle("InterferoLab - renamed")
app.processEvents(); app.processEvents()
res["title_after_rename"] = win.windowTitle()

panel = win._acq_panel
vm = panel._vm
frames, positions, finished, errors = [], [], [], []
vm.previewFrame.connect(lambda f, z: frames.append([list(f.shape), str(f.dtype), float(z), time.monotonic()]))
vm.positionChanged.connect(lambda z: positions.append(float(z)))
vm.finished.connect(lambda folder, skipped, total: finished.append([folder, skipped, total]))
vm.error.connect(lambda msg: errors.append(msg))

def spin(cond, timeout):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    return cond()

panel.le_piezo_serial.setText("000000000")  # batch 5: no hardcoded fallback any more
t0 = time.monotonic()
panel._connect_hw()  # since batch 4 on the connection thread: wait for it
spin(lambda: not vm.connection_busy(), 30)
res["t_connect"] = time.monotonic() - t0
res["connected"] = panel._hardware_connected
res["first_preview"] = spin(lambda: len(frames) >= 1 and not vm.is_previewing(), 20)
res["t_first_preview_after_connect"] = frames[0][3] - (t0 + res["t_connect"]) if frames else None

n = len(frames); t0 = time.monotonic()
vm.request_preview()
spin(lambda: len(frames) > n and not vm.is_previewing(), 20)
res["t_single_preview"] = frames[-1][3] - t0 if len(frames) > n else None

# continuous live preview (camera armed once, newest frame delivered)
n = len(frames)
cam = panel._vm.svc._session._camera
arms = {"n": 0}
orig_start = cam.start_acquisition
def counting_start(*a, **k):
    arms["n"] += 1
    return orig_start(*a, **k)
cam.start_acquisition = counting_start
panel._toggle_preview()
spin(lambda: False, 3.2)
panel._toggle_preview()
spin(lambda: not vm.is_previewing(), 10)
res["live_frames_in_3s"] = len(frames) - n
res["arms_during_live"] = arms["n"]
res["armed_after_stop"] = cam.acquisition_in_progress()

t0 = time.monotonic()
vm.move_to(47.5)
# the position is MEASURED since batch 3 (qPOS: target + ~20 nm of stage noise)
res["moved"] = spin(lambda: positions and abs(positions[-1] - 47.5) < 0.1 and not vm.is_moving(), 20)
res["t_move"] = time.monotonic() - t0

out = OUT_DIR
cfg = {"closed_loop": True, "start": 50.0, "end": 50.1, "step": 0.05, "exposure": 0.007,
       "timeout": 5.0, "format": "bin12", "color_mode": "mono",
       "axis": "A", "settle": 0.2, "output_folder": out}
t0 = time.monotonic()
res["sweep_started"] = vm.apply_and_start(cfg)
res["sweep_done"] = spin(lambda: bool(finished), 60)
res["t_sweep"] = time.monotonic() - t0
if finished:
    folder = finished[0][0]
    res["folder"] = folder
    res["skipped_total"] = finished[0][1:]
    res["files"] = sorted(os.listdir(folder))
    with open(os.path.join(folder, "sim_metadata.json"), encoding="utf-8") as fh:
        res["meta_simulated"] = json.load(fh)["simulated"]

res["frames"] = frames
res["errors"] = errors
res["dialogs"] = dialogs
res["log"] = panel.log_box.toPlainText()
panel._connect_hw()                   # disconnect (connection thread)
spin(lambda: not vm.connection_busy(), 30)
res["disconnected"] = not panel._hardware_connected
res["frame_stats"] = world.frame_stats()
win.close()
app.processEvents()
print(json.dumps(res))
"""


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim_data")
    code = DRIVER.replace("SIM_DIR", repr(SIM)).replace("OUT_DIR", repr(str(out)))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-5000:]
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    r["stderr"] = proc.stderr
    return r


def test_window_is_marked_as_simulation(smoke):
    assert smoke["title"].startswith("[SIMULATION]")
    assert smoke["title_after_rename"].startswith("[SIMULATION]")
    assert smoke["banner"] == "simBanner" and "SIMULATION" in smoke["banner_text"]


def test_connect_logs_sim_serials(smoke):
    assert smoke["connected"], smoke["log"]
    # piezo qIDN in the UI log; the serial itself comes from the local app_config.json
    assert re.search(r"E-625 \[SIMULATION\], SIM-\d+", smoke["log"])
    assert "serial SIM-LP126CU" in smoke["log"]  # camera info in the UI log
    assert "Sensor Bayer phase: blue." in smoke["log"]
    assert "[SIMULATION] Fake Thorlabs camera connected" in smoke["stderr"]
    assert "[SIMULATION] Fake PI controller connected" in smoke["stderr"]
    assert smoke["dialogs"] == [] and smoke["errors"] == []


def test_preview_frames(smoke):
    assert smoke["first_preview"]
    shape, dtype, z, _t = smoke["frames"][0]
    assert shape == [1500, 2048] and dtype == "uint16"  # mono superpixel of the 3000x4096 raw
    # one snapshot = one snap(): ~0.5 s of pylablib sleeps + arm/disarm
    assert 0.4 < smoke["t_single_preview"] < 2.0
    # continuous preview: the camera is armed ONCE for the whole stream (no
    # snap() per frame), disarmed on stop, and more than one frame arrives
    assert smoke["arms_during_live"] == 1 and not smoke["armed_after_stop"]
    assert smoke["live_frames_in_3s"] >= 2
    if os.environ.get("INTERFEROLAB_TIMING_TESTS"):
        # load-dependent (about 60 here; the simulator synthesises ~20 frames/s): on demand
        assert smoke["live_frames_in_3s"] >= 20


def test_move_and_sweep(smoke):
    assert smoke["moved"] and smoke["t_move"] < 2.0
    assert smoke["sweep_started"] and smoke["sweep_done"]
    assert os.path.basename(smoke["folder"]).startswith("SIM_")
    assert smoke["meta_simulated"] is True
    frames = [f for f in smoke["files"] if f.endswith(".bin12")]
    assert len(frames) == 3 and smoke["skipped_total"] == [0, 3]
    assert "sim_ground_truth_height_superpixel_um.npy" in smoke["files"]
    assert smoke["disconnected"]


def test_lab_probe_dry_run(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(SIM, "lab_timing_probe.py"),
            "--dry-run",
            "--quick",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 failed steps" in proc.stdout
    tomls = list(tmp_path.glob("dryrun_*.toml"))
    assert len(tomls) == 1
    from sim.hwprofile import load_profile

    prof = load_profile(str(tomls[0]))
    # a dry run must never pass for a measurement
    assert prof.origin("camera.timing.arm_s") == "estimated"
    assert "DRY RUN" in [f for f in prof.figures() if f[0] == "camera.timing.arm_s"][0][3]
    assert prof["camera.timing.arm_s"] == pytest.approx(0.15, abs=0.03)
    assert prof["camera.sensor.gain_e_per_dn"] == pytest.approx(2.6, rel=0.15)
    assert (
        "defocused at z"
        in [f for f in prof.figures() if f[0] == "camera.sensor.gain_e_per_dn"][0][3]
    )
    assert list(tmp_path.glob("dryrun_stack_*.npz"))


def test_lab_probe_dry_run_focus_defocus_and_raw_stack(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(SIM, "lab_timing_probe.py"),
            "--dry-run",
            "--quick",
            "--skip-piezo",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # without the piezo the gain is still measured, but flagged as not defocused
    assert "NOT defocused" in proc.stdout
    assert list(tmp_path.glob("lab_timing_probe_*.log"))  # log next to the outputs


def test_lab_probe_stops_using_a_hung_device(tmp_path):
    """A step that times out marks its device dead: later steps skip it, it is not closed."""
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(SIM, "lab_timing_probe.py"),
            "--dry-run",
            "--quick",
            "--no-raw-stack",
            "--step-timeout",
            "0.5",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=ROOT,
    )
    out = proc.stdout
    assert proc.returncode == 1, out + proc.stderr
    # snap() takes ~0.5 s, so the first single-snap step ("Bayer phase") hangs past 0.5 s
    assert "[TIMEOUT] Bayer phase from the image" in out
    after = out.split("[TIMEOUT] Bayer phase from the image")[1]
    for step in ("focus scan", "photon transfer", "sweep loop", "dark frames", "camera close"):
        assert step not in after, step  # the hung camera is never touched again
    assert "the camera HUNG" in out
    # the piezo is still measured and closed
    assert "[    ok] GCS query round trip" in after
    assert "[    ok] piezo close" in after
    raw = json.loads(next(tmp_path.glob("dryrun_*.json")).read_text())
    assert "no longer used" in raw["errors"]["Bayer phase from the image"]
    assert len(list(tmp_path.glob("dryrun_*.toml"))) == 1
