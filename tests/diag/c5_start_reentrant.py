"""
C5 - "Start sweep" re-enters itself through _wait_preview_idle().

_start() spins the event loop (processEvents) while a preview capture is in
flight, WITHOUT disabling the buttons first.  Any click (or Ctrl+Return, or
"Disconnect hardware") during that wait runs inside the wait.

Variant "start":      second Start during the wait.
Variant "disconnect": Disconnect hardware during the wait.

Usage: .venv/bin/python c5_start_reentrant.py [start|disconnect]
"""

import sys

from harness import App
from PySide6.QtCore import QTimer

variant = sys.argv[1] if len(sys.argv) > 1 else "start"

app = App()
res = {"connected": app.connect(), "variant": variant}
app.sweep_cfg_ui(50.0, 50.1, 0.05)  # 3 frames

app.vm.request_preview()
app.spin(lambda: False, 0.05)  # capture in flight (~0.5 s)
res["previewing_before_start"] = app.vm.is_previewing()

inner = {}


def during_wait():
    inner["previewing"] = app.vm.is_previewing()
    inner["buttons"] = app.buttons()
    if variant == "start":
        inner["start_enabled"] = app.panel.btn_start.isEnabled()
        app.panel.trigger_start()  # the Ctrl+Return path: clicks only if enabled
    else:
        inner["connect_enabled"] = app.panel.btn_connect.isEnabled()
        app.panel._connect_hw()  # "Disconnect hardware"
        inner["hardware_connected_after"] = app.panel._hardware_connected


QTimer.singleShot(30, during_wait)  # fires inside _wait_preview_idle's processEvents

app.panel._start()  # first click on Start sweep
res["inner"] = inner
res["running_after_start"] = app.vm.is_running()
res["hardware_connected_after_start"] = app.panel._hardware_connected
app.spin(lambda: bool(app.finished) or bool(app.errors), 60)
app.spin(lambda: not app.vm.is_running(), 10)
log = app.log_text()
res["n_sweep_started"] = log.count("Sweep started")
res["n_could_not_start"] = log.count("Sweep could not be started")
res["n_already_running"] = log.count("A sweep is already running")
res["finished"] = app.finished
res["running_events"] = app.running
res["ui_end"] = app.buttons()
app.finish(res)
