"""The simulator side of the performance gate (sim/perf_preview.py).

* the synthetic generator produces the SAME frame in parallel chunks as in one
  serial pass, fast enough for a camera that stays armed;
* the fake camera records every delivered frame (frame_log) with a monotonic
  sequence number, the stage z at mid-exposure and, on request, a thumbnail;
* the fake controller records every MOV and ONT answer (command_log);
* a displayed frame is matched by content to the frame it came from, and not
  to a frame taken 50 nm away;
* the gate itself runs to the end against whatever acquisition code the app
  has, credits only paints matched to a delivered frame, and its verdict is
  consistent with its own numbers (it fails on the app of da3c70d and passes
  on a continuous-stream app: both are right).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")
FAKES = os.path.join(SIM, "fakes")


@pytest.fixture(scope="module")
def fakes_first():
    """Fake drivers first on sys.path for this module only (like the launcher)."""
    saved = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if k.split(".")[0] in ("pylablib", "pipython")}
    for k in saved_mods:
        del sys.modules[k]
    sys.path.insert(0, FAKES)
    try:
        import pipython
        import pylablib

        assert getattr(pylablib, "__simulated__", False) and getattr(pipython, "__simulated__", False)
        yield
    finally:
        sys.path[:] = saved
        for k in [k for k in sys.modules if k.split(".")[0] in ("pylablib", "pipython")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)


def test_generator_chunks_are_bit_exact_and_fast():
    from sim.hwprofile import load_profile
    from sim.signal_model import SyntheticSource

    serial = SyntheticSource(load_profile(None, {"signal": {"generate_threads": 1}}), 3000, 4096, "blue", rng=np.random.default_rng(3))
    auto = SyntheticSource(load_profile(None, {"signal": {"generate_threads": 0}}), 3000, 4096, "blue", rng=np.random.default_rng(3))
    assert serial.threads == 1
    a = serial.generate(50.123, 0.007)
    b = auto.generate(50.123, 0.007)
    assert a.dtype == np.uint16 and a.shape == (3000, 4096)
    assert np.array_equal(a, b)
    if auto.threads > 1 and (os.environ.get("INTERFEROLAB_TIMING_TESTS") or os.environ.get("INTERFEROLAB_PERF_GATE_TEST")):
        for _ in range(5):
            auto.generate(50.2, 0.007)
        # a timing bound (8-12 ms here): only outside the battery, like the gate run
        assert np.median(auto.gen_times_s[-5:]) < 0.046


def test_frame_log_and_command_log(fakes_first):
    from pipython import GCSDevice
    from pylablib.devices import Thorlabs

    from sim.world import configure

    world = configure(None, {})
    dev = GCSDevice(devname="E-625", gcsdll="")
    dev.ConnectUSB(serialnum="0")
    dev.SVO("A", 1)
    dev.MOV("A", 50.0)
    time.sleep(0.05)
    assert world.piezo is dev
    movs = [c for c in dev.command_log if c.command == "MOV"]
    assert movs and movs[-1].value == 50.0 and movs[-1].accepted
    assert dev.qONT("A")["A"] is True
    assert dev.command_log[-1].command == "ONT" and dev.command_log[-1].value is True

    cam = Thorlabs.ThorlabsTLCamera()
    assert world.camera is cam
    cam.set_color_format(color_output="raw", color_space="linear")
    cam.set_exposure(0.007)
    cam.frame_log_thumbnails = True
    raw = cam.snap(timeout=5.0)
    recs = list(cam.frame_log)
    assert len(recs) >= 2  # frames keep arriving until pylablib's disarm
    seqs = [r.seq for r in recs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and seqs[0] == 1
    assert recs[0].index == 0  # buffer index restarts at every acquisition
    assert all(abs(r.z_um - 50.0) < 0.1 for r in recs)  # stage z at mid-exposure
    assert all(r.exposure_s == pytest.approx(0.007) for r in recs)
    assert all(r.t_start < r.t_mid < r.t_ready for r in recs)
    assert recs[0].thumb is not None and recs[0].thumb.shape == Thorlabs.TLCamera.THUMB_SHAPE
    # the snapped frame is the first one after the trigger
    assert np.allclose(Thorlabs.TLCamera.frame_thumbnail(raw), recs[0].thumb)
    # a second snap keeps counting
    cam.snap(timeout=5.0)
    assert cam.frame_log[-1].seq > seqs[-1] and cam.frame_log[-1].index >= 0
    cam.close()
    dev.CloseConnection()


def test_display_matches_its_own_frame_not_a_neighbour(fakes_first):
    from pylablib.devices.Thorlabs.TLCamera import area_resize, frame_thumbnail

    from backend.acquisition.acquisition_controller import AcquisitionSession
    from sim.perf_preview import MATCH_MIN_CORR, correlation
    from sim.world import configure

    src = configure(None, {}).source()
    frames = {z: src.generate(z, 0.007) for z in (50.0, 50.05, 50.1)}
    thumbs = {z: frame_thumbnail(f) for z, f in frames.items()}
    for z, f in frames.items():
        mono = AcquisitionSession._bayer_to_superpixel(f, "blue")
        shown = area_resize((mono >> 4).astype(np.float32))  # what the app paints, 8-bit
        own = correlation(shown, thumbs[z])
        others = [correlation(shown, thumbs[o]) for o in thumbs if o != z]
        assert own >= MATCH_MIN_CORR, own
        assert all(c < MATCH_MIN_CORR for c in others), others


def _frame(FrameRecord, seq, t_ready, z, thumb):
    return FrameRecord(seq, seq - 1, t_ready - 0.053, t_ready - 0.0495, t_ready, z, 0.007, thumb)


def _pixmap_of(raw):
    """What the app paints for a raw frame: mono superpixel, 8-bit, scaled to the label."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap

    from backend.acquisition.acquisition_controller import AcquisitionSession

    mono = AcquisitionSession._bayer_to_superpixel(raw, "blue")
    img8 = (mono >> 4).astype(np.uint8)
    h, w = img8.shape
    pix = QPixmap.fromImage(QImage(img8.data, w, h, img8.strides[0], QImage.Format_Grayscale8).copy())
    return pix.scaled(947, 437, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def test_paint_crediting_cannot_be_fooled(fakes_first):
    """Controlled frames and paints (no timing): repeated pixels, a re-shown old
    frame and a paint with no matching frame are never credited; the count can
    never exceed what the camera delivered."""
    from collections import deque

    from pylablib.devices.Thorlabs.TLCamera import FrameRecord, frame_thumbnail
    from PySide6.QtWidgets import QApplication, QLabel

    from sim import perf_preview as pp
    from sim.world import configure

    QApplication.instance() or QApplication([])
    src = configure(None, {}).source()
    raw_a, raw_b = src.generate(50.0, 0.007), src.generate(50.05, 0.007)

    class Cam:
        frame_log = deque()
        frames_delayed_by_generation = 0

    class World:
        camera = Cam()

    log = World.camera.frame_log
    log.append(_frame(FrameRecord, 1, 1.00, 50.0, frame_thumbnail(raw_a)))
    log.append(_frame(FrameRecord, 2, 1.05, 50.0, frame_thumbnail(raw_a)))  # same z, fresh noise
    log.append(_frame(FrameRecord, 3, 1.10, 50.05, frame_thumbnail(raw_b)))
    pp.PROBE.__init__()
    pp.PROBE.scenario = "t"
    label = QLabel()

    def paint(pix, t):
        label.setPixmap(pix)
        pp.on_preview_painted(label, World, t)
        return pp.PROBE.paints[-1]

    pix_a, pix_b = _pixmap_of(raw_a), _pixmap_of(raw_b)
    p = paint(pix_a, 2.0)
    assert p["changed"] and p["new_frame"] and p["seq"] == 1 and p["z"] == 50.0
    p = paint(pix_a, 2.1)  # the very same pixmap painted again
    assert not p["changed"] and not p["new_frame"]
    p = paint(_pixmap_of(raw_a), 2.2)  # rebuilt, identical pixels
    assert not p["changed"] and not p["new_frame"]
    p = paint(pix_b, 2.3)  # a newer frame at another z
    assert p["new_frame"] and p["seq"] == 3 and p["z"] == 50.05
    p = paint(pix_a, 2.4)  # back to the OLD frame: pixels changed, but it is stale
    assert p["changed"] and not p["new_frame"] and p.get("stale")
    log.append(_frame(FrameRecord, 4, 2.5, 50.05, frame_thumbnail(raw_b)))
    p = paint(pix_b, 2.6)  # frame 4 arrived, but the pixels shown are frame 3's again
    assert p["changed"] and p["new_frame"] and p["seq"] == 4  # same z: indistinguishable, bounded by delivery
    p = paint(pix_b, 2.7)  # nothing new delivered, same pixels
    assert not p["changed"] and not p["new_frame"]
    credited = [q for q in pp.PROBE.paints if q["new_frame"]]
    assert len(credited) == 3 <= len(log)
    # frames delivered while the app rearms (same z, old exposure) are never
    # read: a paint with the NEW exposure must credit the new-exposure frame,
    # not the oldest tied one, and the offset must not accumulate
    raw_b20 = src.generate(50.05, 0.020)
    for seq in range(5, 10):
        log.append(_frame(FrameRecord, seq, 2.8 + 0.05 * seq, 50.05, frame_thumbnail(raw_b)))
    log.append(_frame(FrameRecord, 10, 3.5, 50.05, frame_thumbnail(raw_b20))._replace(exposure_s=0.020))
    p = paint(_pixmap_of(raw_b20), 3.6)
    assert p["new_frame"] and p["seq"] == 10 and p["exposure"] == 0.020
    log.append(_frame(FrameRecord, 11, 3.7, 50.05, frame_thumbnail(raw_b20))._replace(exposure_s=0.020))
    log.append(_frame(FrameRecord, 12, 3.8, 50.05, frame_thumbnail(raw_b))._replace(exposure_s=0.007))
    p = paint(pix_b, 3.9)  # back to 7 ms: the 7 ms frame, not the 20 ms one
    assert p["new_frame"] and p["seq"] == 12 and p["exposure"] == 0.007
    # a paint that matches no delivered frame (flat image) is unmatched
    log.clear()
    log.append(_frame(FrameRecord, 5, 3.0, 50.1, frame_thumbnail(src.generate(50.1, 0.007))))
    p = paint(pix_a, 3.1)
    assert p["changed"] and not p["new_frame"] and not p.get("stale") and p["corr"] < pp.MATCH_MIN_CORR


def test_move_latency_counts_lost_orders_as_infinite(fakes_first):
    from collections import deque

    from pipython.pidevice.gcsdevice import CommandRecord
    from pylablib.devices.Thorlabs.TLCamera import FrameRecord

    from sim import perf_preview as pp

    settle = 0.025

    class Piezo:
        command_log = deque()

    class World:
        piezo = Piezo()
        camera = None

    pp.PROBE.__init__()
    pp.PROBE.windows["moving"] = (10.0, 20.0)
    # order 1: MOV 3 ms later, first frame at the new z ready at +0.10, painted at +0.13
    # order 2: MOV, frame at the new z, but the screen never shows it inside the window
    # order 3: rejected by the app and no later MOV either (no MOV at all)
    pp.PROBE.orders = [{"t": 10.0, "z": 50.05}, {"t": 12.0, "z": 50.15}, {"t": 13.0, "z": 50.20}]
    World.piezo.command_log.extend(
        [
            CommandRecord(10.003, "MOV", "A", 50.05, True),
            CommandRecord(12.003, "MOV", "A", 50.15, True),
        ]
    )
    frames = [
        _frame(FrameRecord, 1, 10.05, 50.02, None),  # exposed before settling: not at the new z
        _frame(FrameRecord, 2, 10.10, 50.05, None),
        _frame(FrameRecord, 3, 12.10, 50.15, None),
    ]
    pp.PROBE.window_frames["moving"] = frames
    pp.PROBE.paints = [
        {"t": 10.08, "scenario": "moving", "changed": True, "new_frame": True, "seq": 1, "t_mid": frames[0].t_mid, "t_ready": frames[0].t_ready, "t_ready_newest": frames[0].t_ready, "z": 50.02, "exposure": 0.007},
        {"t": 10.13, "scenario": "moving", "changed": True, "new_frame": True, "seq": 2, "t_mid": frames[1].t_mid, "t_ready": frames[1].t_ready, "t_ready_newest": frames[1].t_ready, "z": 50.05, "exposure": 0.007},
    ]
    m = pp.summarize_moves("moving", World, settle)
    assert m["orders"] == 3 and m["orders_with_their_own_mov"] == 2
    assert m["orders_lost_no_mov"] == 1 and m["orders_lost_no_image"] == 1  # orders 3 and 2
    lat = m["order_to_image"]
    assert lat["n"] == 3 and lat["n_infinite"] == 2 and lat["min_ms"] == pytest.approx(130, abs=1)
    assert lat["median_ms"] is None  # two of three are infinite: the median is infinite -> no PASS possible
    assert m["order_to_camera_frame_at_new_z"]["n"] == 2  # camera side: orders 1 and 2
    # with the lost orders fixed, the median is finite
    pp.PROBE.orders = pp.PROBE.orders[:1]
    assert pp.summarize_moves("moving", World, settle)["order_to_image"]["median_ms"] == pytest.approx(130, abs=1)


@pytest.mark.skipif(
    not os.environ.get("INTERFEROLAB_PERF_GATE_TEST"),
    reason="runs the whole gate (about 20 s, timing-sensitive); set INTERFEROLAB_PERF_GATE_TEST=1",
)
def test_gate_runs_and_cannot_be_fooled(tmp_path):
    """Whatever the app's acquisition code does, the gate must (a) run to the end,
    (b) credit only paints matched to a delivered frame and never more frames
    than the camera delivered, and (c) give a verdict consistent with its own
    numbers.  Against the app of da3c70d it fails (about 1.5 fps, ~0.8 s);
    against a continuous-stream app it passes: both are right."""
    out = tmp_path / "gate.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(SIM, "perf_preview.py"), "--quick", "--quiet", "--json", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert proc.returncode in (0, 1), proc.stdout[-2000:] + proc.stderr[-3000:]
    r = json.loads(out.read_text())
    assert r["notes"] == [], r["notes"]
    for name in ("still_fast", "moving_fast", "exposure"):
        s = r["scenarios"][name]
        assert s["paints_unmatched"] == 0 and s["paints_stale_or_repeated"] == 0, (name, s)
        assert s["frames_shown"] <= s["camera_frames_delivered"], name
        assert s["fps_on_screen"] <= 1.0 / 0.046 * 1.05, name  # never above the camera
    fps_ok, lat_ok = (c["ok"] for c in r["gate"])
    st, mv = r["scenarios"]["still_fast"], r["scenarios"]["moving_fast"]["moves"]
    assert fps_ok == (st["fps_on_screen"] >= 15.0)
    med = mv["order_to_image"].get("median_ms")
    assert lat_ok == (med is not None and med < 250.0)
    assert mv["order_to_image"]["n"] == mv["orders"]  # every order counted, lost ones as infinite
    assert proc.returncode == (0 if fps_ok and lat_ok else 1)
