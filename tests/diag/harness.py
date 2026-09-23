"""
Shared harness for the concurrency reproductions (tests/diag/).

Boots the app through the simulator launcher (sim/run_simulated.py)
offscreen, builds the real MainWindow and records everything the GUI would
show: preview frames, position updates, errors, modal dialogs and the panel
log.  Every reproduction script prints one JSON line at the end; the
diagnostic tests (tests/test_diag_concurrency.py) run the scripts in a
subprocess and parse that line.

The scripts were written for the phase-2 diagnosis (role A2, 2026-09-21,
originally outside git) and moved here on 2026-09-22 because the tests that
retire the findings must live in the repository.

Nothing in the app is edited by them.  Instrumentation is applied from
outside, the same way the launcher does it (module attributes replaced at
runtime).
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "sim")


class Dialogs:
    """Replacement for QMessageBox: records instead of blocking."""

    calls: list[list[str]] = []

    @classmethod
    def critical(cls, parent, title, text, *a, **k):
        cls.calls.append(["critical", title, text])

    @classmethod
    def warning(cls, parent, title, text, *a, **k):
        cls.calls.append(["warning", title, text])

    @classmethod
    def information(cls, parent, title, text, *a, **k):
        cls.calls.append(["information", title, text])

    @classmethod
    def question(cls, parent, title, text, *a, **k):
        cls.calls.append(["question", title, text])
        from PySide6.QtWidgets import QMessageBox

        return QMessageBox.Yes


class App:
    def __init__(self, sets=(), out_dir=None):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        sys.path.insert(0, SIM)
        import run_simulated

        argv = []
        for s in sets:
            argv += ["--set", s]
        self.args, self.world = run_simulated.bootstrap(argv)

        import views.MainWindow as mw

        mw.load_config = lambda: {}  # never touch app_config.json
        mw.save_config = lambda cfg: None
        import views.AcquisitionPanel as ap

        ap.QMessageBox = Dialogs
        mw.QMessageBox = Dialogs

        from PySide6.QtWidgets import QApplication

        self.qt = QApplication.instance() or QApplication([])
        self.win = mw.MainWindow()
        self.win.show()
        self.panel = self.win._acq_panel
        # Batch 5: no hardcoded fallback serial any more, and mw.load_config
        # is patched to {} above so nothing fills the field from a config
        # file either -- every script needs one to get past _connect_hw().
        self.panel.le_piezo_serial.setText("000000000")
        self.vm = self.panel._vm
        self.svc = self.vm.svc
        self.session = self.svc._session
        self.out_dir = (
            out_dir or os.environ.get("A2_OUT") or os.path.join(ROOT, "logs", "diag_repro_data")
        )
        # Sweeps started from the UI go to "data": redirect them (from outside,
        # on top of the launcher's own SIM_ patch) so no folder lands in data/.
        from backend.acquisition import acquisition_controller as ac

        sim_create = ac.AcquisitionSession.create_output_folder
        out_dir_ = self.out_dir

        def create_output_folder(base_folder, log_cb):
            return sim_create(os.path.join(out_dir_, base_folder), log_cb)

        ac.AcquisitionSession.create_output_folder = staticmethod(create_output_folder)

        self.frames: list[dict] = []
        self.positions: list[tuple[float, float]] = []
        self.errors: list[tuple[float, str]] = []
        self.finished: list[list] = []
        self.moving: list[tuple[float, bool]] = []
        self.running: list[tuple[float, bool]] = []
        self.notices: list[tuple[float, str]] = []  # non-blocking camera notices
        self.aborted: list[str] = []  # sweep abort summaries
        self.t0 = time.monotonic()
        self.vm.previewFrame.connect(
            lambda f, z: self.frames.append(
                {"t": self.now(), "z": float(z), "shape": list(f.shape), "mean": float(f.mean())}
            )
        )
        self.vm.positionChanged.connect(lambda z: self.positions.append((self.now(), float(z))))
        self.vm.error.connect(lambda m: self.errors.append((self.now(), m)))
        self.vm.finished.connect(lambda folder, s, t: self.finished.append([folder, s, t]))
        self.vm.movingChanged.connect(lambda b: self.moving.append((self.now(), bool(b))))
        self.vm.runningChanged.connect(lambda b: self.running.append((self.now(), bool(b))))
        # getattr: the same harness also runs against older app versions
        if hasattr(self.vm, "previewNotice"):
            self.vm.previewNotice.connect(lambda t: self.notices.append((self.now(), t)))
        if hasattr(self.vm, "sweepAborted"):
            self.vm.sweepAborted.connect(self.aborted.append)

    # ------------------------------------------------------------------
    def now(self) -> float:
        return round(time.monotonic() - self.t0, 4)

    def spin(self, cond=lambda: False, timeout: float = 1.0) -> bool:
        """Pump the event loop until cond() or timeout (like a user waiting)."""
        t_end = time.monotonic() + timeout
        while not cond() and time.monotonic() < t_end:
            self.qt.processEvents()
            time.sleep(0.005)
        self.qt.processEvents()
        return bool(cond())

    def connect(self) -> bool:
        n = len(self.frames)
        self.panel._connect_hw()
        # since batch 4 the connection runs on a thread: wait for it, then for
        # the first preview it requests (a NEW frame: this may be a reconnect)
        busy = getattr(self.vm, "connection_busy", lambda: "")
        self.spin(lambda: not busy(), 30)
        self.spin(lambda: len(self.frames) > n and not self.vm.is_previewing(), 20)
        # since batch 5 a successful connect also parks the piezo (non-blocking):
        # wait for it too, so scripts start from a settled, known state exactly
        # as before (a script that wants to race the park move issues its own
        # connect_hw()/connectFinished directly instead of this helper).
        if self.panel._hardware_connected:
            self.spin(lambda: not self.vm.is_moving(), 35)
        return self.panel._hardware_connected

    def disconnect(self, timeout: float = 30.0) -> bool:
        """Click "Disconnect hardware" and wait until the connection thread is
        done (since batch 4 disconnecting no longer blocks the GUI thread)."""
        self.panel._connect_hw()
        busy = getattr(self.vm, "connection_busy", lambda: "")
        self.spin(lambda: not busy() and not self.panel._hardware_connected, timeout)
        return not self.panel._hardware_connected

    def log_text(self) -> str:
        return self.panel.log_box.toPlainText()

    def buttons(self) -> dict:
        p = self.panel
        return {
            "connect": [p.btn_connect.isEnabled(), p.btn_connect.text()],
            "preview": [p.btn_preview.isEnabled(), p.btn_preview.text()],
            "start": p.btn_start.isEnabled(),
            "cancel": p.btn_cancel.isEnabled(),
            "move": p.btn_move_piezo.isEnabled(),
            "apply_cam": p.btn_apply_cam.isEnabled(),
            "slider": p.sl_z.isEnabled(),
            "preview_active": p._preview_active,
            # no preview timer since B1 (continuous preview); kept for old JSON readers
            "timer_active": bool(getattr(p, "_live_timer", None) and p._live_timer.isActive()),
            "hardware_connected": p._hardware_connected,
            "connection_busy": p._vm.connection_busy() if hasattr(p._vm, "connection_busy") else "",
        }

    def sweep_cfg_ui(self, start=50.0, end=50.1, step=0.05):
        """Set the sweep range through the UI spinboxes (as the user would)."""
        p = self.panel
        p.sb_end.setValue(100.0)
        p.sb_start.setValue(start)
        p.sb_end.setValue(end)
        p.sb_step.setValue(step)

    def finish(self, res: dict) -> None:
        res["dialogs"] = Dialogs.calls
        res["errors"] = self.errors
        res["notices"] = self.notices
        res["aborted"] = self.aborted
        res["status_label"] = getattr(self.panel, "lbl_camera_status", None) and [
            self.panel.lbl_camera_status.isVisible(),
            self.panel.lbl_camera_status.text(),
        ]
        res["log"] = self.log_text()
        res["stage_position_um"] = self.world.stage.position()
        print(json.dumps(res, default=str))
        sys.stdout.flush()


class CameraMonitor:
    """Wraps every public method of a (fake) camera INSTANCE to record which
    threads call it and whether two threads are ever inside it at once.

    Instrumentation from outside, like the launcher: the app is not edited.
    Nested calls from the same thread (start_acquisition -> send_software_trigger)
    count once; the fake's own producer thread uses private methods only.
    """

    def __init__(self, cam) -> None:
        import threading

        self.cam = cam
        self.lock = threading.Lock()
        self.inside: dict[int, int] = {}
        self.threads: dict[int, str] = {}
        self.calls: list[tuple[float, str, int]] = []
        self.overlaps: list[tuple[str, str]] = []
        self.main_thread = threading.get_ident()
        self._threading = threading
        for name in dir(cam):
            if name.startswith("_"):
                continue
            attr = getattr(cam, name)
            # methods only: the exception classes (TimeoutError, Error) are
            # callable too and must stay classes for `except` to work
            if callable(attr) and not isinstance(attr, type):
                setattr(cam, name, self._wrap(name, attr))

    def _wrap(self, name, fn):
        def wrapped(*a, **k):
            tid = self._threading.get_ident()
            with self.lock:
                others = [t for t, d in self.inside.items() if t != tid and d > 0]
                if others:
                    self.overlaps.append((name, f"while thread {others[0]} is inside"))
                self.inside[tid] = self.inside.get(tid, 0) + 1
                self.threads[tid] = self._threading.current_thread().name
                self.calls.append((time.monotonic(), name, tid))
            try:
                return fn(*a, **k)
            finally:
                with self.lock:
                    self.inside[tid] -= 1

        return wrapped

    def summary(self) -> dict:
        return {
            "camera_threads": sorted(set(self.threads.values())),
            "n_camera_threads": len(self.threads),
            "gui_thread_calls": sum(1 for _, _, t in self.calls if t == self.main_thread),
            "n_calls": len(self.calls),
            "overlaps": self.overlaps,
        }


def emit(res: dict) -> None:
    print(json.dumps(res, default=str))
    sys.stdout.flush()
