"""
C2 - A camera call that never returns.

Before B1 (2026-09-22): the preview QThread never finished, previews were
dropped silently, moves rejected, Start waited and gave up; on close,
shutdown() waited 12 s, disconnected the hardware under the hung thread and
Qt destroyed the running QThread -> abort.

Now: the camera-owner thread (a daemon) is abandoned after the close deadline
with a "Camera is not answering" message; nothing touches the camera again
and the process exits normally.

The fake camera's wait_for_frame is replaced (from outside) by a wait that
ignores its timeout, like an SDK call that hangs.
Exit code and stderr are the evidence (see tests/test_diag_concurrency.py).
"""

import json
import sys
import threading
import time

from harness import App

app = App()
res = {"connected": app.connect()}

cam = app.session._camera
hang = threading.Event()


def wait_for_frame_hung(*a, **k):
    hang.wait()  # never set: the SDK call never returns
    return True


cam.wait_for_frame = wait_for_frame_hung  # instrumentation on the FAKE camera object

app.vm.request_preview()
app.spin(lambda: False, 0.5)
res["previewing"] = app.vm.is_previewing()
res["ui_while_hung"] = app.buttons()

# what the user can still do while the thread hangs
n = len(app.frames)
app.vm.request_preview()
res["second_preview_started"] = app.svc.capture_preview()
app.vm.move_to(55.0)
res["move_accepted"] = app.vm.is_moving()
app.panel.sb_timeout.setValue(0.5)
t0 = time.monotonic()
app.panel._start()
res["t_start_gave_up_s"] = round(time.monotonic() - t0, 2)
res["start_rejected"] = "did not finish in time" in app.log_text()

# closing the window: shutdown waits 12 s then disconnects anyway
t0 = time.monotonic()
app.win.close()
app.qt.processEvents()
res["t_close_s"] = round(time.monotonic() - t0, 2)
res["camera_thread_abandoned"] = "not answering" in app.log_text()
res["owner_thread_alive_and_detached"] = (
    app.session.camera_owner(create=False) is None  # the session forgot it
)
res["log"] = app.log_text()
print(json.dumps(res, default=str))
sys.stdout.flush()
# Normal interpreter exit follows.  If Qt aborts ("QThread: Destroyed while
# thread is still running"), the exit code is not 0.
del app
