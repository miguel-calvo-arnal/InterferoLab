"""
C4b - Camera stops delivering frames (every snap times out) but stays
"present": the sweep used to pay the full camera timeout at EVERY remaining
position and never abort.

faults.snap_timeout_probability=1.0 from the start; a 5-position sweep with
the UI minimum timeout (0.5 s).  The snapshot after connect fails too: since
batch 2 it is retried and then given up with a non-blocking notice (no
dialog).  The per-step cost measured here scales to the default sweep (501
positions, 5 s timeout) as a derived figure.
"""

import time

from harness import App

app = App(sets=["faults.snap_timeout_probability=1.0"])
app.panel.sb_timeout.setValue(0.5)
app.panel._connect_hw()
# the snapshot after connect times out; wait until it is given up (or, in an
# older app, until its error dialog)
app.spin(
    lambda: any("stopped" in t for _, t in app.notices) or len(app.errors) > 0,
    30,
)
app.spin(lambda: not app.vm.is_previewing(), 10)
res = {
    "connected": app.panel._hardware_connected,
    "connect_preview_error": len(app.errors),
    "connect_preview_notices": [t for _, t in app.notices],
}
app.sweep_cfg_ui(50.0, 50.2, 0.05)  # 5 positions

t0 = time.monotonic()
app.panel._start()
app.spin(lambda: bool(app.finished) or len(app.errors) > res["connect_preview_error"], 120)
app.spin(lambda: not app.vm.is_running(), 10)
res["t_sweep_s"] = round(time.monotonic() - t0, 2)
res["finished"] = app.finished
res["n_camera_errors"] = app.log_text().count("Camera timeout/error")
res["n_steps_attempted"] = res["n_camera_errors"]
res["s_per_failed_step"] = round(res["t_sweep_s"] / max(1, res["n_camera_errors"]), 2)
# derived: default UI sweep (45-55 um, 0.02 um -> 501 positions) with the
# default 5 s timeout: 501 x (5 s + 0.25 s pylablib sleeps + move) without an
# abort; 3 x the same with the batch-2 abort
res["derived_default_sweep_min_without_abort"] = round(
    501 * (5.0 + (res["s_per_failed_step"] - 0.5)) / 60, 1
)
res["ui_after"] = app.buttons()
app.finish(res)
