"""
B1 - The camera has a single owner thread (2026-09-22).

After connecting, every call on the camera object is watched (CameraMonitor)
through a whole session: continuous preview, manual moves during the
preview, a snapshot, an exposure change, a short sweep, and the disconnect.
Expected: one thread (the camera owner) makes every call, the GUI thread
makes none, and two threads are never inside the camera at the same time.
Also reports frames delivered per second to the GUI and whether any move was
rejected.
"""

import time

from harness import App, CameraMonitor

app = App()
res = {"connected": app.connect()}
mon = CameraMonitor(app.session._camera)
app.sweep_cfg_ui(50.0, 50.1, 0.05)  # 3 frames

app.panel._toggle_preview()
app.spin(lambda: len(app.frames) >= 3, 10)
n0, t0 = len(app.frames), time.monotonic()
rejected = 0
for i in range(5):
    app.panel.sb_manual_z.setValue(48.0 + 0.5 * i)
    before = app.log_text().count("Cannot move")
    app.panel._move_piezo_manual()
    rejected += app.log_text().count("Cannot move") - before
    app.spin(lambda: not app.vm.is_moving(), 5)
    app.spin(lambda: False, 0.2)
res["moves_rejected"] = rejected
res["fps_delivered_during_moves"] = round((len(app.frames) - n0) / (time.monotonic() - t0), 1)

app.vm.request_preview()  # snapshot while streaming: served by the stream
app.panel.sb_exposure.setValue(14.0)
app.panel._apply_camera_params()
app.spin(lambda: "exposure updated" in app.log_text(), 10)

app.panel._start()  # sweep with the preview on: paused, then resumed
app.spin(lambda: bool(app.finished) or bool(app.errors), 60)
app.spin(lambda: not app.vm.is_running(), 10)
res["sweep_finished"] = app.finished
n1 = len(app.frames)
res["preview_resumed_after_sweep"] = app.spin(lambda: len(app.frames) > n1 + 3, 10)
res["preview_active_after_sweep"] = app.panel._preview_active

t1 = time.monotonic()
app.disconnect()  # disconnect with the stream running (connection thread, batch 4)
res["t_disconnect_s"] = round(time.monotonic() - t1, 2)
res["previewing_after_disconnect"] = app.vm.is_previewing()
# "Camera closed" is logged by the camera thread through a queued signal:
# give the event loop a moment to deliver it (an abandonment is logged
# synchronously by disconnect_all itself)
app.spin(lambda: "Camera closed" in app.log_text(), 2.0)
res["disconnect_answered"] = (
    "not answering" not in app.log_text() and "Camera closed" in app.log_text()
)
res["monitor"] = mon.summary()
app.finish(res)
