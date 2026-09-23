"""
C7 - One lost preview frame used to stop the live preview and open a modal
(with an empty text: pylablib's timeout has no message).

Since batch 2 the continuous stream is armed once, so the fault is a stream
stall: faults.stream_stall_after_frames=15 makes the armed camera stop
delivering after 15 frames, for that arm only (faults.stream_stall_arms=1).
Desired: a non-blocking notice with a text in the window, the camera re-armed
by itself, frames flowing again, and no dialog at all.

Optional argv[1]: stream_stall_arms (e.g. 1000 = the camera never recovers:
the preview must give up after a few attempts and say so, still no dialog).
"""

import sys

from harness import App

arms = sys.argv[1] if len(sys.argv) > 1 else "1"
app = App(sets=["faults.stream_stall_after_frames=15", f"faults.stream_stall_arms={arms}"])
app.panel.sb_timeout.setValue(0.5)  # shortest UI timeout: keeps the run short
res = {"connected": app.connect()}
app.panel._toggle_preview()
res["preview_on"] = app.buttons()

app.spin(lambda: len(app.notices) > 0, 15)
res["frames_before_error"] = len(app.frames)
res["t_notice"] = app.notices[0][0] if app.notices else None
res["status_label_during"] = [
    app.panel.lbl_camera_status.isVisible(),
    app.panel.lbl_camera_status.text(),
]
if arms == "1":
    # recovery: frames flow again and the notice is cleared
    app.spin(lambda: len(app.frames) > res["frames_before_error"] + 5, 10)
    app.spin(lambda: app.notices and app.notices[-1][1] == "", 5)
else:
    # never recovers: the preview gives up
    app.spin(lambda: not app.panel._preview_active, 30)
    app.spin(lambda: False, 0.5)
res["frames_after_error"] = len(app.frames)
res["ui_after_error"] = app.buttons()
app.finish(res)
