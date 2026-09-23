#!/usr/bin/env python
"""
[SIMULATION] Live-preview performance gate for InterferoLab.

    .venv/bin/python sim/perf_preview.py                        # gate at 15 fps / 250 ms
    .venv/bin/python sim/perf_preview.py --json out.json --quick
    .venv/bin/python sim/perf_preview.py --profile sim/profiles/perf_sdk_half.toml

Starts the UNMODIFIED app against the simulated hardware (same bootstrap as
sim/run_simulated.py), offscreen, drives it through its own widgets (the
buttons and spin boxes a user would click) and measures at two STABLE
boundaries only, so it works with the current app and with any redesign of
its acquisition code:

  camera side   every frame the fake camera delivers while armed is recorded
                by the fake itself (ThorlabsTLCamera.frame_log: index, exposure
                start / mid / ready times, the stage position sampled at
                mid-exposure, the exposure, a thumbnail); every MOV that reached
                the fake controller is in GCSDevice.command_log;
  screen side   every paint event of the preview widget is captured (its
                pixmap), compared with the previous one and MATCHED BY CONTENT
                (thumbnail correlation) to one delivered frame.

What counts:
  * a frame is "on screen" only when the painted pixels changed AND they match
    a frame the camera delivered AFTER the one shown before.  Repainting the
    same frame, or showing an older one, does not count;
  * latency of a move order = from the click on "Move piezo" to the paint of
    the first frame whose exposure started after the stage settled on the
    first MOV the controller accepted at or after that order.  An order that
    never leads to a MOV, or never to such a frame, is LOST and counts as an
    infinite latency in the median.

Gate (exit 0 = pass, 1 = fail, 2 = crash), thresholds set by Miguel on
2026-09-22:  on-screen frame rate >= --min-fps (15) with the piezo still and the
fastest preview interval the UI allows;  median move-order -> image latency
< --max-latency-ms (250) while the piezo is moved by hand in small steps.
With the app as of da3c70d it FAILS (about 1.5 fps; 0.75 s for the orders it
accepts and several seconds once the lost ones count): expected.

Which figures are real and which are simulated: the fake camera's frame
times come from the profile (readout 1/21.7 fps derived, SDK costs mostly
ESTIMATED); everything between the camera's buffer and the screen (the app,
numpy, Qt) runs for real on this machine.  The simulator's own frame
generation cost is reported: when it exceeds the readout period the SIMULATOR,
not the app, limits the frame rate ("simulator-limited" in the report).
"""

from __future__ import annotations

import os
import sys

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SIM_DIR)
# "python sim/perf_preview.py" puts sim/ first on sys.path: drop it (as the
# launcher does) so nothing in sim/ can shadow a top-level module.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != SIM_DIR]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_now = time.perf_counter  # same clock as time.monotonic on Linux (checked at start)

#: Minimum thumbnail correlation to accept that the screen shows a given frame.
MATCH_MIN_CORR = 0.97


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------
class Probe:
    """Everything observed, with the scenario it belongs to."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.paints: list[dict] = []  # every paint of the preview widget
        self.orders: list[dict] = []  # move orders given through the UI
        self.applies: list[dict] = []  # exposure changes given through the UI
        self.stalls: list[dict] = []  # GUI events longer than the threshold
        self.windows: dict[str, tuple[float, float]] = {}
        self.scenario = "setup"
        self.main_thread = threading.get_ident()
        self.last_shown_seq = 0
        self.level_ratios: list[float] = []  # display mean / thumbnail mean, credited frames
        self.last_display: np.ndarray | None = None
        self.gen_marks: dict[str, tuple[int, int]] = {}
        self.window_frames: dict[str, list] = {}  # FrameRecords delivered inside each window (no thumbs)
        self.delay_marks: dict[str, tuple[int, int]] = {}
        self.debug_dir: str | None = None
        self.notes: list[str] = []
        self.debug_dumps = 0


PROBE = Probe()


# ----------------------------------------------------------------------
# Screen side: capture the preview widget and match it to a delivered frame
# ----------------------------------------------------------------------
def display_to_array(widget):
    """Grayscale uint8 array of what the widget shows (its pixmap, else a grab)."""
    from PySide6.QtGui import QImage

    pm = None
    if hasattr(widget, "pixmap"):
        pm = widget.pixmap()
        if pm is None or pm.isNull():
            return None  # a label without an image yet: nothing to match
    if pm is None:
        if getattr(widget, "_perf_grabbing", False):
            return None
        widget._perf_grabbing = True
        try:
            pm = widget.grab()
        finally:
            widget._perf_grabbing = False
    img = pm.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    if w == 0 or h == 0:
        return None
    arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, :w]
    return arr.copy()


def crop_letterbox(arr: np.ndarray) -> np.ndarray:
    """Drop uniform borders (a grabbed label may pad the image with background)."""
    bg = arr[0, 0]
    rows = np.where((arr != bg).any(axis=1))[0]
    cols = np.where((arr != bg).any(axis=0))[0]
    if rows.size < 8 or cols.size < 8:
        return arr
    return arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def thumbnail(arr: np.ndarray) -> np.ndarray | None:
    """Same grid as the fake camera's frame_thumbnail (area average of the whole image)."""
    from pylablib.devices.Thorlabs.TLCamera import area_resize  # the FAKE

    if arr.shape[0] < 8 or arr.shape[1] < 8:
        return None
    return area_resize(arr)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


#: Frames considered when matching a paint (about 3 s at 21.7 fps).
MATCH_WINDOW = 64
#: Correlations within this margin of the best are ties: the OLDEST wins, so
#: the next frame at the same z can still be credited afterwards.
MATCH_TIE = 0.002


def match_frame(thumb: np.ndarray, records, level_ratio: float | None = None) -> tuple[object | None, float, object | None]:
    """(oldest near-equal match, its correlation, newest near-equal match).

    Correlation is scale-invariant, so frames at the same z but another
    exposure tie.  Ties are broken by brightness when ``level_ratio``
    (display mean / thumbnail mean of frames already credited) is known: only
    candidates whose brightness agrees within LEVEL_TOL stay.  Among those the
    oldest one is credited (so the following same-z frames stay creditable);
    the newest one bounds the camera-to-screen latency from below.
    """
    recs = [r for r in records if r.thumb is not None][-MATCH_WINDOW:]
    if not recs:
        return None, -2.0, None
    corrs = [correlation(thumb, r.thumb) for r in recs]
    best_c = max(corrs)
    tied = [(r, c) for r, c in zip(recs, corrs, strict=True) if c >= best_c - MATCH_TIE]
    if level_ratio and len(tied) > 1:
        mean = float(thumb.mean())
        bright = [(r, c) for r, c in tied if abs(float(r.thumb.mean()) * level_ratio - mean) <= LEVEL_TOL * mean]
        if bright:
            tied = bright
    return tied[0][0], tied[0][1], tied[-1][0]


#: Brightness agreement (relative) that keeps a tied candidate; an exposure
#: change of 7 -> 20 ms scales the level by 2.9, far outside.
LEVEL_TOL = 0.25


def on_preview_painted(widget, world, t_paint: float) -> None:
    """Called after every paint event of the preview widget (GUI thread).

    A paint is credited as a NEW frame on screen when its pixels changed and
    its content matches (by thumbnail correlation) a frame the camera
    delivered after the last credited one.  Frames at the same z differ only
    in noise, invisible at thumbnail scale, so the match cannot tell two
    same-z frames apart: content decides the z (what matters for latency),
    and the "newer than the last credited" rule bounds the count by what the
    camera delivered.  A display whose best match over ALL frames is an older
    frame at a clearly different z is stale and is not credited.
    """
    arr = display_to_array(widget)
    if arr is None:
        return
    changed = PROBE.last_display is None or arr.shape != PROBE.last_display.shape or not np.array_equal(arr, PROBE.last_display)
    rec = {"t": t_paint, "scenario": PROBE.scenario, "changed": changed, "new_frame": False}
    if changed:
        PROBE.last_display = arr
        cam = world.camera
        th = thumbnail(crop_letterbox(arr))
        if cam is not None and th is not None:
            records = list(cam.frame_log)[-MATCH_WINDOW:]
            newer = [r for r in records if r.seq > PROBE.last_shown_seq]
            ratio = statistics.median(PROBE.level_ratios[-9:]) if PROBE.level_ratios else None
            frame, corr, newest = match_frame(th, newer, ratio)
            _any, corr_any, _n = match_frame(th, records, ratio)
            rec["corr"] = corr
            rec["corr_any"] = corr_any
            if frame is not None and corr >= MATCH_MIN_CORR and corr_any - corr < 0.01:
                rec.update(
                    {
                        "seq": frame.seq,
                        "index": frame.index,
                        "t_ready": frame.t_ready,
                        "t_ready_newest": newest.t_ready,
                        "t_mid": frame.t_mid,
                        "z": frame.z_um,
                        "exposure": frame.exposure_s,
                        "new_frame": True,
                    }
                )
                PROBE.last_shown_seq = frame.seq
                if float(frame.thumb.mean()) > 0:
                    PROBE.level_ratios.append(float(th.mean()) / float(frame.thumb.mean()))
            elif corr_any >= MATCH_MIN_CORR:
                rec["stale"] = True
            if PROBE.debug_dir and not rec["new_frame"] and PROBE.debug_dumps < 3:
                PROBE.debug_dumps += 1
                np.savez(
                    os.path.join(PROBE.debug_dir, f"perf_unmatched_{PROBE.debug_dumps}.npz"),
                    display=arr,
                    thumb=th,
                    cam_thumbs=np.array([r.thumb for r in records[-12:]]),
                    cam_z=np.array([r.z_um for r in records[-12:]]),
                    cam_exp=np.array([r.exposure_s for r in records[-12:]]),
                )
    PROBE.paints.append(rec)


# ----------------------------------------------------------------------
# QApplication with a notify() override: preview paints and GUI stalls
# ----------------------------------------------------------------------
def make_app_class():
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    class PerfApplication(QApplication):
        preview_widget = None
        world = None
        stall_ms = 5.0

        def notify(self, obj, event):  # noqa: N802 - Qt API
            t0 = _now()
            r = super().notify(obj, event)
            t1 = _now()
            if event.type() == QEvent.Type.Paint and obj is self.preview_widget:
                if hasattr(obj, "pixmap"):
                    on_preview_painted(obj, self.world, t1)
                else:  # grabbing a widget inside its own paint event would recurse
                    from PySide6.QtCore import QTimer

                    QTimer.singleShot(0, lambda o=obj, t=t1: on_preview_painted(o, self.world, t))
            elif t1 - t0 > self.stall_ms / 1000 and threading.get_ident() == PROBE.main_thread:
                PROBE.stalls.append(
                    {
                        "t": t0,
                        "ms": 1000 * (t1 - t0),
                        "scenario": PROBE.scenario,
                        "event": str(event.type()).rsplit(".", 1)[-1],
                        "receiver": type(obj).__name__,
                    }
                )
            return r

    return PerfApplication


# ----------------------------------------------------------------------
# UI driving: the widgets a user would click, found by text, not by attribute
# ----------------------------------------------------------------------
class UI:
    def __init__(self, win) -> None:
        self.win = win

    def button(self, text: str):
        from PySide6.QtWidgets import QPushButton

        for b in self.win.findChildren(QPushButton):
            if b.text().strip().lower().startswith(text.lower()):
                return b
        return None

    def click(self, text: str) -> bool:
        b = self.button(text)
        if b is None or not b.isEnabled():
            return False
        b.click()
        return True

    def spin_after_label(self, label_text: str):
        """The spin box placed right after a label in a grid layout."""
        from PySide6.QtWidgets import QAbstractSpinBox, QGridLayout, QLabel

        for grid in self.win.findChildren(QGridLayout):
            for i in range(grid.count()):
                w = grid.itemAt(i).widget()
                if isinstance(w, QLabel) and label_text.lower() in w.text().lower():
                    r, c, _rs, _cs = grid.getItemPosition(i)
                    item = grid.itemAtPosition(r, c + 1)
                    if item is not None and isinstance(item.widget(), QAbstractSpinBox):
                        return item.widget()
        return None

    def preview_widget(self, object_name: str):
        from PySide6.QtWidgets import QLabel, QWidget

        for w in self.win.findChildren(QWidget):
            if w.objectName() == object_name:
                return w
        # fallback: the label showing the biggest pixmap
        best, area = None, 0
        for w in self.win.findChildren(QLabel):
            pm = w.pixmap()
            if pm is not None and not pm.isNull() and pm.width() * pm.height() > area:
                best, area = w, pm.width() * pm.height()
        return best


# ----------------------------------------------------------------------
# Event-loop helpers (real event loop, no sleeps in the GUI thread)
# ----------------------------------------------------------------------
def run_loop_ms(ms: int) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec()


def wait_until(cond, timeout_s: float, poll_ms: int = 5) -> bool:
    from PySide6.QtCore import QEventLoop, QTimer

    if cond():
        return True
    loop = QEventLoop()
    deadline = _now() + timeout_s
    timer = QTimer()
    timer.setInterval(poll_ms)

    def check():
        if cond() or _now() > deadline:
            loop.quit()

    timer.timeout.connect(check)
    timer.start()
    loop.exec()
    timer.stop()
    return cond()


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def _stats(values) -> dict:
    finite = sorted(v for v in values if v is not None and math.isfinite(v))
    n_inf = sum(1 for v in values if v is not None and not math.isfinite(v))
    if not finite and not n_inf:
        return {"n": 0}
    allv = sorted((v for v in values if v is not None), key=lambda v: (math.isinf(v), v))
    med = allv[len(allv) // 2] if len(allv) % 2 else (allv[len(allv) // 2 - 1] + allv[len(allv) // 2]) / 2
    out = {"n": len(allv), "n_infinite": n_inf, "median_ms": 1000 * med if math.isfinite(med) else None}
    if finite:
        p95 = finite[min(len(finite) - 1, int(round(0.95 * (len(finite) - 1))))]
        out.update(
            {
                "mean_ms": 1000 * statistics.fmean(finite),
                "p95_ms": 1000 * p95,
                "min_ms": 1000 * finite[0],
                "max_ms": 1000 * finite[-1],
            }
        )
    return out


def frames_in(name: str, t0: float, t1: float) -> list:
    return [r for r in PROBE.window_frames.get(name, []) if t0 <= r.t_ready <= t1]


def summarize_window(name: str, world, settle_s: float, readout_budget_s: float) -> dict:
    t0, t1 = PROBE.windows[name]
    dur = t1 - t0
    paints = [p for p in PROBE.paints if t0 <= p["t"] <= t1]
    shown = [p for p in paints if p["new_frame"]]
    changed = [p for p in paints if p["changed"]]
    stale = [p for p in changed if p.get("stale")]
    unmatched = [p for p in changed if not p["new_frame"] and not p.get("stale")]
    delivered = frames_in(name, t0, t1)
    stalls = [s for s in PROBE.stalls if t0 <= s["t"] <= t1]
    gen = []
    src = world.source_if_built()
    if src is not None and name in PROBE.gen_marks:
        a, b = PROBE.gen_marks[name]
        gen = list(src.gen_times_s[a:b])
    gen_stats = _stats(gen) if gen else {"n": 0}
    d0, d1 = PROBE.delay_marks.get(name, (0, 0))
    delayed = max(0, d1 - d0)
    out = {
        "duration_s": dur,
        "frames_shown": len(shown),
        "fps_on_screen": len(shown) / dur if dur > 0 else 0.0,
        "paints": len(paints),
        "paints_changed": len(changed),
        "paints_unmatched": len(unmatched),
        "paints_stale_or_repeated": len(stale),
        "match_corr_min": min([p["corr"] for p in shown], default=None),
        "camera_frames_delivered": len(delivered),
        "camera_fps": len(delivered) / dur if dur > 0 else 0.0,
        "camera_frames_discarded": len(delivered) - len(shown),
        "camera_ready_to_screen_lower_bound": _stats([p["t"] - p["t_ready_newest"] for p in shown]),
        "gui_longest_stall_ms": max([s["ms"] for s in stalls], default=0.0),
        "gui_stalls": sorted(stalls, key=lambda s: -s["ms"])[:5],
        "sim_generation": gen_stats,
        "sim_frames_delayed_by_generation": delayed,
        "sim_limited": delayed > 0.02 * max(1, len(delivered)),
        "sim_ceiling_fps": (1.0 / max(readout_budget_s, gen_stats.get("median_ms", 0) / 1000.0)) if gen else None,
    }
    return out


def summarize_moves(name: str, world, settle_s: float) -> dict:
    t0, t1 = PROBE.windows[name]
    orders = [o for o in PROBE.orders if t0 <= o["t"] <= t1]
    piezo = world.piezo
    movs = [c for c in (piezo.command_log if piezo else []) if c.command == "MOV" and c.accepted and t0 <= c.t <= t1 + 5]
    shown = [p for p in PROBE.paints if p["new_frame"] and p["t"] >= t0]
    delivered = [r for r in PROBE.window_frames.get(name, []) if r.t_ready >= t0]
    lat_screen, lat_camera, own_mov, lost_no_mov, lost_no_image, order_to_mov = [], [], 0, 0, 0, []
    for o in orders:
        mov = next((m for m in movs if m.t >= o["t"] - 1e-6), None)
        if mov is None:
            lost_no_mov += 1
            lat_screen.append(math.inf)
            continue
        if abs(mov.value - o["z"]) < 1e-9:
            own_mov += 1
        order_to_mov.append(mov.t - o["t"])
        t_settled = mov.t + settle_s
        cam_frame = next((r for r in delivered if r.t_mid >= t_settled), None)
        if cam_frame is not None:
            lat_camera.append(cam_frame.t_ready - o["t"])
        img = next((p for p in shown if p["t_mid"] >= t_settled and p["t"] > o["t"]), None)
        if img is None:
            lost_no_image += 1
            lat_screen.append(math.inf)
        else:
            lat_screen.append(img["t"] - o["t"])
    return {
        "orders": len(orders),
        "orders_with_their_own_mov": own_mov,
        "orders_lost_no_mov": lost_no_mov,
        "orders_lost_no_image": lost_no_image,
        "order_to_mov": _stats(order_to_mov),
        "order_to_image": _stats(lat_screen),
        "order_to_camera_frame_at_new_z": _stats(lat_camera),
        "movs_accepted_by_controller": len([m for m in movs if t0 <= m.t <= t1]),
    }


def summarize_exposure(name: str) -> dict:
    t0, t1 = PROBE.windows[name]
    applies = [a for a in PROBE.applies if t0 <= a["t"] <= t1]
    shown = [p for p in PROBE.paints if p["new_frame"] and p["t"] >= t0]
    lat = []
    for a in applies:
        img = next((p for p in shown if p["t"] > a["t"] and abs(p["exposure"] - a["exposure_s"]) < 1e-7), None)
        lat.append((img["t"] - a["t"]) if img else math.inf)
    return {
        "applies": len(applies),
        "apply_blocks_gui": _stats([a["dt"] for a in applies]),
        "apply_to_image": _stats(lat),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="InterferoLab live-preview performance gate (simulator).")
    ap.add_argument("--profile", help="profile TOML merged over sim/profiles/default.toml")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="profile override (repeatable)")
    ap.add_argument("--min-fps", type=float, default=15.0, help="gate: on-screen fps, piezo still, fastest UI interval")
    ap.add_argument("--max-latency-ms", type=float, default=250.0, help="gate: median move order -> painted frame at the new position")
    ap.add_argument("--still-s", type=float, default=6.0, help="seconds of the 'still' scenario")
    ap.add_argument("--moves", type=int, default=30, help="manual moves in the 'moving' scenario")
    ap.add_argument("--move-period-ms", type=int, default=300, help="mean time between manual move orders")
    ap.add_argument("--move-jitter-ms", type=int, default=150, help="uniform +/- jitter of that period (seeded)")
    ap.add_argument("--move-step-um", type=float, default=0.05, help="size of each manual step")
    ap.add_argument("--exposure-changes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--color-mode", choices=("mono", "color", "mono_superpixel"), default="mono")
    ap.add_argument("--preview-widget", default="previewLabel", help="objectName of the widget that shows the preview")
    ap.add_argument("--window", default="1600x1000", help="window size (the preview scales with it)")
    ap.add_argument("--also-default-interval", action="store_true", help="repeat still/moving at the interval the UI starts with")
    ap.add_argument("--quick", action="store_true", help="short scenarios (about 20 s): for tests")
    ap.add_argument("--json", help="write the full results here")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--debug", action="store_true", help="print the UI state at every scenario boundary")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.still_s, args.moves, args.exposure_changes = 3.0, 8, 2
    if args.debug:
        PROBE.debug_dir = os.path.dirname(os.path.abspath(args.json)) if args.json else os.getcwd()
    t_wall0 = _now()
    launcher_argv = []
    if args.profile:
        launcher_argv += ["--profile", args.profile]
    for s in args.set or []:
        launcher_argv += ["--set", s]

    from sim import run_simulated

    _args, world = run_simulated.bootstrap(launcher_argv)
    assert abs(time.perf_counter() - time.monotonic()) < 0.01, "perf_counter and monotonic differ"
    profile = world.profile
    settle_s = float(profile["piezo.timing.settle_s"])
    readout_budget_s = float(profile["camera.timing.readout_transfer_s"])

    import views.MainWindow as mw

    mw.load_config = lambda: {}  # deterministic UI defaults, never the user's config
    mw.save_config = lambda cfg: None
    dialogs: list = []
    try:  # a modal dialog would block offscreen; the panel module may change name
        import views.AcquisitionPanel as ap

        class _NoModal:
            @staticmethod
            def critical(parent, title, text, *a, **k):
                dialogs.append(["critical", title, text])

            @staticmethod
            def warning(parent, title, text, *a, **k):
                dialogs.append(["warning", title, text])

            @staticmethod
            def information(parent, title, text, *a, **k):
                dialogs.append(["information", title, text])

        ap.QMessageBox = _NoModal
    except Exception:  # noqa: BLE001
        pass

    from PySide6.QtCore import QTimer

    App = make_app_class()
    app = App([])
    app.setStyle("Fusion")
    App.world = world
    win = mw.MainWindow()
    w, h = (int(x) for x in args.window.lower().split("x"))
    win.resize(w, h)
    win.show()
    # Batch 5: no hardcoded fallback serial any more, and load_config is {}
    # (deterministic UI defaults) -- the field is empty unless set here.
    win._acq_panel.le_piezo_serial.setText("000000000")
    ui = UI(win)
    run_loop_ms(50)

    def scenario(name: str) -> None:
        PROBE.scenario = name
        if args.debug:
            from PySide6.QtWidgets import QComboBox, QPushButton

            print(f"[debug] {name}: buttons", [(b.text(), b.isEnabled()) for b in win.findChildren(QPushButton) if b.text()],
                  "combos", [c.currentText() for c in win.findChildren(QComboBox)],
                  "frames delivered", world.camera.frames_delivered if world.camera else None,
                  "paints", len(PROBE.paints), "notes", PROBE.notes[-3:], file=sys.stderr)
            try:  # app-specific, debug only
                panel = win._acq_panel
                print(f"[debug]   app: moving={panel._vm.is_moving()} previewing={panel._vm.is_previewing()} "
                      f"active={panel._preview_active} log tail={panel.log_box.toPlainText().splitlines()[-4:]}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print("[debug]   app state unavailable:", e, file=sys.stderr)
        src = world.source_if_built()
        PROBE.gen_marks[name] = (len(src.gen_times_s) if src else 0, 0)
        PROBE.delay_marks[name] = (world.camera.frames_delayed_by_generation if world.camera else 0, 0)

    def close_window(name: str, t0: float, t1: float) -> None:
        PROBE.windows[name] = (t0, t1)
        src = world.source_if_built()
        a, _ = PROBE.gen_marks[name]
        PROBE.gen_marks[name] = (a, len(src.gen_times_s) if src else 0)
        d, _ = PROBE.delay_marks[name]
        PROBE.delay_marks[name] = (d, world.camera.frames_delayed_by_generation if world.camera else 0)
        cam = world.camera
        PROBE.window_frames[name] = [
            r._replace(thumb=None) for r in (cam.frame_log if cam else []) if t0 <= r.t_ready <= t1 + 5
        ]

    # sim-side conditions ---------------------------------------------------
    def connected() -> bool:
        return world.camera is not None and world.camera.is_opened() and world.piezo is not None and world.piezo.connected

    def last_frame_t() -> float:
        cam = world.camera
        return cam.frame_log[-1].t_ready if cam is not None and cam.frame_log else -1.0

    def no_new_frames(for_s: float) -> bool:
        return _now() - last_frame_t() > for_s

    def order_done(t_order: float, give_up_s: float = 2.0):
        """Condition: the MOV for an order given at t_order reached the controller
        and the stage settled -- or no MOV came within give_up_s (the app
        rejected the order, as the current one does during a capture)."""

        def cond() -> bool:
            piezo = world.piezo
            movs = [c.t for c in (piezo.command_log if piezo else []) if c.command == "MOV" and c.t >= t_order - 1e-6]
            if not movs:
                return _now() - t_order > give_up_s
            return world.stage.on_target() and _now() - max(movs) > settle_s + 0.3

        return cond

    # --- connect through the UI -------------------------------------------
    scenario("connect")
    t0 = _now()
    if not ui.click("Connect hardware"):
        print("FAIL: no 'Connect hardware' button", file=sys.stderr)
        return 2
    t_connected = _now()
    if not wait_until(connected, 30):
        print("FAIL: hardware did not connect", dialogs, file=sys.stderr)
        return 2
    world.camera.frame_log_thumbnails = True
    preview = ui.preview_widget(args.preview_widget)
    App.preview_widget = preview
    wait_until(lambda: len(PROBE.paints) > 0 or (last_frame_t() > 0 and no_new_frames(1.0)), 20)
    run_loop_ms(300)
    close_window("connect", t0, _now())
    if preview is None:
        print(f"FAIL: preview widget {args.preview_widget!r} not found", file=sys.stderr)
        return 2

    spin_z = ui.spin_after_label("Manual Z")
    spin_int = ui.spin_after_label("Preview interval")
    spin_exp = ui.spin_after_label("Exposure")
    if spin_z is None or spin_exp is None:
        print("FAIL: 'Manual Z' or 'Exposure' spin box not found", file=sys.stderr)
        return 2

    def order_move(z: float) -> None:
        spin_z.setValue(z)
        t = _now()
        ok = ui.click("Move piezo")
        PROBE.orders.append({"t": t, "z": float(spin_z.value()), "clicked": ok, "scenario": PROBE.scenario})

    # park mid-travel so the moving scenario starts from a realistic position
    scenario("park")
    order_move(50.0)
    wait_until(order_done(PROBE.orders[-1]["t"]), 35)
    run_loop_ms(100)

    def set_interval(ms: int | None) -> int | None:
        if spin_int is None:
            return None
        if ms is None:
            ms = spin_int.minimum()
        spin_int.setValue(ms)
        ui.click("Apply preview interval")
        return int(spin_int.value())

    def preview_on(on: bool) -> None:
        b = ui.button("Stop preview") if on else ui.button("Start preview")
        # b is None: not yet in the wanted state -> click the toggle
        if b is None and not ui.click("Start preview" if on else "Stop preview"):
            PROBE.notes.append(f"{PROBE.scenario}: preview button missing or disabled (wanted on={on})")
        if not wait_until(lambda: (ui.button("Stop preview") is not None) == on, 2):
            PROBE.notes.append(f"{PROBE.scenario}: preview did not switch to on={on}")

    def still(name: str, interval_ms: int | None) -> None:
        scenario(name)
        iv = set_interval(interval_ms)
        preview_on(True)
        t0 = _now()
        run_loop_ms(int(args.still_s * 1000))
        t1 = _now()
        preview_on(False)
        wait_until(lambda: no_new_frames(0.5), 15)
        close_window(name, t0, t1)
        PROBE.windows[name + "_meta"] = (iv, iv)

    def moving(name: str, interval_ms: int | None) -> None:
        scenario(name)
        set_interval(interval_ms)
        preview_on(True)
        rng = random.Random(args.seed)
        state = {"n": 0, "z": 50.0, "dir": 1.0}
        timer = QTimer()
        timer.setSingleShot(True)

        def next_period() -> int:
            return max(20, args.move_period_ms + rng.randint(-args.move_jitter_ms, args.move_jitter_ms))

        def step():
            if state["n"] >= args.moves:
                return
            state["n"] += 1
            state["z"] += state["dir"] * args.move_step_um
            if state["z"] > 55 or state["z"] < 45:
                state["dir"] *= -1
            order_move(round(state["z"], 4))
            timer.start(next_period())

        timer.timeout.connect(step)
        t0 = _now()
        timer.start(next_period())
        run_loop_ms((args.move_period_ms + args.move_jitter_ms) * (args.moves + 1))
        timer.stop()
        wait_until(order_done(PROBE.orders[-1]["t"]), 35)
        run_loop_ms(1500)  # let the last order show up on screen
        t1 = _now()
        preview_on(False)
        wait_until(lambda: no_new_frames(0.5), 15)
        close_window(name, t0, t1)

    def exposure(name: str) -> None:
        scenario(name)
        set_interval(None)
        preview_on(True)
        values = [7.0, 20.0]
        t0 = _now()
        for i in range(args.exposure_changes):
            run_loop_ms(1500)
            ms = values[(i + 1) % 2]
            spin_exp.setValue(ms)
            t = _now()
            ui.click("Apply camera settings")
            PROBE.applies.append({"t": t, "dt": _now() - t, "exposure_s": ms / 1000.0, "scenario": name})
        run_loop_ms(1500)
        t1 = _now()
        preview_on(False)
        wait_until(lambda: no_new_frames(0.5), 15)
        spin_exp.setValue(7.0)
        ui.click("Apply camera settings")
        close_window(name, t0, t1)

    default_interval = int(spin_int.value()) if spin_int is not None else None
    if args.also_default_interval and spin_int is not None:
        still("still_default", default_interval)
        moving("moving_default", default_interval)
    still("still_fast", None)
    moving("moving_fast", None)
    exposure("exposure")

    scenario("shutdown")
    ui.click("Disconnect hardware")
    win.close()
    run_loop_ms(50)

    # --- analysis -----------------------------------------------------------
    results = {
        "profile_files": profile.files,
        "color_mode": args.color_mode,
        "window": args.window,
        "preview_widget": preview.objectName() or type(preview).__name__,
        "interval_ms": {"default": default_interval, "fastest": int(spin_int.minimum()) if spin_int else None},
        "wall_time_s": _now() - t_wall0,
        "connect": {
            "connect_click_blocks_gui_s": t_connected - PROBE.windows["connect"][0],
            "first_paint_after_connect_s": (next((p["t"] for p in PROBE.paints if p["changed"]), None) or float("nan")) - PROBE.windows["connect"][0],
        },
        "scenarios": {},
        "dialogs": dialogs,
        "notes": PROBE.notes,
    }
    for name in list(PROBE.windows):
        if name in ("connect",) or name.endswith("_meta"):
            continue
        s = summarize_window(name, world, settle_s, readout_budget_s)
        if name.startswith("moving"):
            s["moves"] = summarize_moves(name, world, settle_s)
        if name.startswith("exposure"):
            s["exposure"] = summarize_exposure(name)
        results["scenarios"][name] = s
    results["paints"] = [p for p in PROBE.paints if p["changed"]]
    results["orders"] = PROBE.orders

    # --- gate -----------------------------------------------------------------
    st = results["scenarios"]["still_fast"]
    mv = results["scenarios"]["moving_fast"]["moves"]
    fps = st["fps_on_screen"]
    lat = mv["order_to_image"]
    lat_ms = lat.get("median_ms")
    fps_label = "on-screen fps, piezo still, fastest UI interval"
    if st["sim_limited"]:
        fps_label += f" [SIMULATOR-LIMITED: {st['sim_frames_delayed_by_generation']} frames delayed by the synthetic generation]"
    lat_label = "move order -> painted frame at the new position, median (ms)"
    lost = mv["orders_lost_no_mov"] + mv["orders_lost_no_image"]
    if lost:
        lat_label += f" [{lost} of {mv['orders']} orders lost: {mv['orders_lost_no_mov']} never reached the controller, {mv['orders_lost_no_image']} never shown]"
    checks = [
        (fps_label, fps, ">=", args.min_fps, fps >= args.min_fps),
        (lat_label, lat_ms, "<", args.max_latency_ms, lat_ms is not None and lat_ms < args.max_latency_ms),
    ]
    results["gate"] = [{"check": c, "value": v, "op": op, "threshold": th, "ok": ok} for c, v, op, th, ok in checks]
    ok_all = all(c[4] for c in checks)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, default=float)
    if not args.quiet:
        print_report(results, checks, readout_budget_s)
    print("PERF GATE:", "PASS" if ok_all else "FAIL", f"({results['wall_time_s']:.0f} s)")
    return 0 if ok_all else 1


def _ms(st: dict, key: str = "median_ms") -> str:
    v = st.get(key)
    return "n/a" if v is None else f"{v:.0f}"


def print_report(results: dict, checks, readout_budget_s: float) -> None:
    print("=" * 78)
    print("[SIMULATION] InterferoLab live-preview performance (stable-boundary gate)")
    print("profile:", ", ".join(os.path.basename(p) for p in results["profile_files"]))
    print(f"colour mode: {results['color_mode']}; window {results['window']}; preview widget {results['preview_widget']}; "
          f"preview interval default {results['interval_ms']['default']} ms, fastest {results['interval_ms']['fastest']} ms")
    c = results["connect"]
    print(f"connect: the click blocked the GUI {1000 * c['connect_click_blocks_gui_s']:.0f} ms; first image painted after {1000 * c['first_paint_after_connect_s']:.0f} ms")
    for name, s in results["scenarios"].items():
        print("-" * 78)
        print(f"{name}: {s['duration_s']:.1f} s; camera delivered {s['camera_frames_delivered']} frames ({s['camera_fps']:.1f} fps); "
              f"shown {s['frames_shown']} -> {s['fps_on_screen']:.2f} fps on screen; discarded {s['camera_frames_discarded']}; "
              f"paints {s['paints']} (changed {s['paints_changed']}, stale/repeated {s['paints_stale_or_repeated']}, unmatched {s['paints_unmatched']})")
        print(f"  camera frame ready -> on screen (lower bound, newest same-z frame): median {_ms(s['camera_ready_to_screen_lower_bound'])} ms, "
              f"p95 {_ms(s['camera_ready_to_screen_lower_bound'], 'p95_ms')} ms; "
              f"longest GUI stall {s['gui_longest_stall_ms']:.1f} ms; match corr min {s['match_corr_min']}")
        g = s["sim_generation"]
        if g.get("n"):
            print(f"  simulator: generation median {g['median_ms']:.1f} ms, p95 {g['p95_ms']:.1f} ms vs readout {1000 * readout_budget_s:.0f} ms "
                  f"-> ceiling {s['sim_ceiling_fps']:.1f} fps; {s['sim_frames_delayed_by_generation']} frames delayed by generation"
                  f"{' (SIMULATOR-LIMITED)' if s['sim_limited'] else ''}")
        if "moves" in s:
            m = s["moves"]
            print(f"  moves: {m['orders']} orders, {m['orders_with_their_own_mov']} reached the controller as their own MOV, "
                  f"{m['orders_lost_no_mov']} never reached it, {m['orders_lost_no_image']} never shown; "
                  f"order -> MOV median {_ms(m['order_to_mov'])} ms")
            print(f"  order -> image at new z: median {_ms(m['order_to_image'])} ms, p95 {_ms(m['order_to_image'], 'p95_ms')} ms, "
                  f"min {_ms(m['order_to_image'], 'min_ms')} ms ({m['order_to_image'].get('n_infinite', 0)} infinite); "
                  f"order -> camera frame at new z: median {_ms(m['order_to_camera_frame_at_new_z'])} ms")
        if "exposure" in s:
            e = s["exposure"]
            print(f"  exposure: {e['applies']} applies, GUI blocked median {_ms(e['apply_blocks_gui'])} ms, "
                  f"apply -> image with new exposure median {_ms(e['apply_to_image'])} ms ({e['apply_to_image'].get('n_infinite', 0)} never shown)")
    print("-" * 78)
    for c, v, op, th, ok in checks:
        vs = "n/a" if v is None else f"{v:.2f}"
        print(f"[{'ok' if ok else 'FAIL'}] {c}: {vs} {op} {th}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(2)
