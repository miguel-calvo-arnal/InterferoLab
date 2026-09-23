"""
C4 - Camera lost mid-sweep: the sweep keeps moving the piezo through every
remaining position, logging one error per step, and the UI still believes the
hardware is connected afterwards.

faults.camera_disconnect_after_snaps=3: snap 1 is the preview after connect,
snaps 2-3 are the first two sweep frames, from the 4th on every snap fails.
Since batch 2 the sweep aborts after 3 failures in a row and marks the folder.
"""

import os
import time

from harness import App

app = App(sets=["faults.camera_disconnect_after_snaps=3"])
app.panel.sb_timeout.setValue(0.5)
res = {"connected": app.connect()}
app.sweep_cfg_ui(50.0, 52.0, 0.1)  # 21 positions

t0 = time.monotonic()
app.panel._start()
app.spin(lambda: bool(app.finished) or bool(app.errors), 120)
app.spin(lambda: not app.vm.is_running(), 10)
res["t_sweep_s"] = round(time.monotonic() - t0, 2)
res["finished"] = app.finished
log = app.log_text()
res["n_camera_errors"] = log.count("Camera timeout/error")
res["sweep_aborted"] = "Sweep aborted" in log
res["ui_after"] = app.buttons()
res["stage_end_um"] = round(app.world.stage.position(), 2)

res["sweep_files"] = sorted(os.listdir(app.finished[0][0])) if app.finished else []

# the UI offers preview and sweep again, on a camera that is gone: the
# snapshot is retried and given up with a notice (batch 2), never a dialog
n_err = len(app.errors)
n_notices = len(app.notices)
app.vm.request_preview()
app.spin(
    lambda: len(app.errors) > n_err or any("stopped" in t for _, t in app.notices[n_notices:]),
    20,
)
res["preview_after"] = [e[1] for e in app.errors[n_err:]]
res["preview_after_notices"] = [t for _, t in app.notices[n_notices:]]
res["ui_after_preview"] = app.buttons()
app.finish(res)
