"""
C6 - With the live preview on, manual moves are rejected whenever a capture is
in flight, and the slider/spinbox snap back to the old position on the next
frame.  Preview ticks that land on a busy thread are dropped in silence.

Usage: .venv/bin/python c6_move_vs_preview.py [interval_ms] [n_moves]

interval_ms is kept for the tick bookkeeping only: since the continuous
preview (B1, 2026-09-22) there is no preview timer in the UI.
"""

import sys
import time

from harness import App

interval = int(sys.argv[1]) if len(sys.argv) > 1 else 200
n_moves = int(sys.argv[2]) if len(sys.argv) > 2 else 10

app = App()
res = {"connected": app.connect(), "interval_ms": interval}
app.panel._toggle_preview()
app.spin(lambda: False, 1.0)

attempts = []
warn_before = app.log_text().count("Cannot move piezo while a preview capture is in progress")
t_start = time.monotonic()
for i in range(n_moves):
    z = 40.0 + i  # 40, 41, ... um
    app.panel.sb_manual_z.setValue(z)
    n_warn = app.log_text().count("Cannot move piezo while a preview capture is in progress")
    app.panel._move_piezo_manual()  # = clicking "Move piezo" / releasing the slider
    accepted = app.vm.is_moving()
    rejected = (
        app.log_text().count("Cannot move piezo while a preview capture is in progress") > n_warn
    )
    app.spin(lambda: not app.vm.is_moving(), 5)
    app.spin(lambda: False, 0.7)  # let the next preview frame arrive
    attempts.append(
        {
            "target": z,
            "accepted": accepted,
            "rejected_preview_busy": rejected,
            "stage_after_um": round(app.world.stage.position(), 2),
            "slider_after_um": app.panel.sl_z.value() / 100.0,
            "spinbox_after_um": app.panel.sb_manual_z.value(),
        }
    )
elapsed = time.monotonic() - t_start
res["attempts"] = attempts
res["n_rejected"] = sum(a["rejected_preview_busy"] for a in attempts)
res["n_accepted"] = sum(a["accepted"] for a in attempts)
res["n_snapback"] = sum(
    1 for a in attempts if a["rejected_preview_busy"] and a["spinbox_after_um"] != a["target"]
)
# silent tick drops: ticks fired vs frames delivered while the preview ran
res["elapsed_s"] = round(elapsed, 2)
res["ticks_expected"] = int(elapsed * 1000 / interval)
res["frames_delivered"] = len(app.frames) - 1
app.panel._toggle_preview()
app.spin(lambda: not app.vm.is_previewing(), 10)
app.finish(res)
