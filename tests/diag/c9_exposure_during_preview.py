"""
C9 - "Apply camera settings" while the live preview runs.

Before B1 (2026-09-22): set_exposure ran on the GUI thread while the preview
worker was inside snap() on the same camera object (two threads in the
driver at once; pylablib has no lock around SDK calls).

Now the exposure is applied by the camera-owner thread between frames, with
the stream stopped and re-armed around it.  This script watches every call
on the camera object after connecting (CameraMonitor) and reports which
threads made them, whether two threads ever overlapped inside the camera,
and whether the frames after the change carry the new exposure.
"""

from harness import App, CameraMonitor

app = App()
res = {"connected": app.connect()}
ref_mean = app.frames[-1]["mean"]  # exposure 7 ms
mon = CameraMonitor(app.session._camera)

app.panel._toggle_preview()  # continuous preview
app.spin(lambda: len(app.frames) >= 4, 10)
res["previewing"] = app.vm.is_previewing()
n = len(app.frames)
app.panel.sb_exposure.setValue(70.0)  # 10x
app.panel._apply_camera_params()
res["apply_button_was_enabled"] = app.panel.btn_apply_cam.isEnabled()
res["log_says_applying"] = "Applying camera parameters" in app.log_text()
app.spin(lambda: "exposure updated" in app.log_text(), 10)
res["log_says_updated"] = "exposure updated" in app.log_text()
# read the fake's state directly (an attribute, not an SDK call): calling
# get_exposure() here would be the GUI thread entering the camera
res["camera_exposure_s_now"] = app.session._camera._exposure_us * 1e-6
app.spin(lambda: len(app.frames) > n + 6, 10)
new_mean = app.frames[-1]["mean"]
res["mean_ref_7ms"] = round(ref_mean, 1)
res["mean_after_70ms"] = round(new_mean, 1)
res["exposure_visible_in_preview"] = new_mean > 3 * ref_mean
res["preview_still_running"] = app.vm.is_previewing() and app.panel._preview_active
app.panel._toggle_preview()
app.spin(lambda: not app.vm.is_previewing(), 10)
res["monitor"] = mon.summary()
app.finish(res)
