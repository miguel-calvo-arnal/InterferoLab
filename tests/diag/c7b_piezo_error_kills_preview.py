"""
C7b - A piezo error (GCSError on MOV) stops the CAMERA preview.

All worker errors share AcquisitionPanel._on_error, which sets
_preview_active=False, stops the timer and opens a modal.  A failed manual
move therefore kills the live view the user is focusing with.
"""

from harness import App

app = App(sets=["faults.mov_error_probability=1.0"])
res = {"connected": app.connect()}
app.panel._toggle_preview()
app.spin(lambda: len(app.frames) >= 3, 10)
res["frames_before"] = len(app.frames)

# a move is rejected while a capture is in flight: retry like an insistent user
app.panel.sb_manual_z.setValue(55.0)
for _ in range(50):
    app.panel._move_piezo_manual()
    if app.vm.is_moving():
        break
    app.spin(lambda: False, 0.05)
res["move_accepted"] = app.vm.is_moving()
app.spin(lambda: len(app.errors) > 0, 10)
res["move_errors"] = [e[1] for e in app.errors]
n = len(app.frames)
app.spin(lambda: False, 1.5)
res["frames_after_error"] = len(app.frames) - n
res["ui_after_error"] = app.buttons()
app.finish(res)
