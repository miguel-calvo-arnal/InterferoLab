"""
C1 - "Disconnect hardware" while a preview capture is in flight.

The GUI thread closes the camera (stop_acquisition + close) while the preview
worker thread is inside snap() on the same object.  What the user sees:
the log says "Hardware disconnected." and, seconds later, an
"Acquisition error" dialog from the orphaned capture.  The UI already offers
"Connect hardware" while the old capture thread is still alive.

Usage: .venv/bin/python c1_disconnect_during_preview.py [delay_s]
"""

import sys
import time

from harness import App

delay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05

app = App()
res = {"connected": app.connect(), "delay_s": delay}

app.vm.request_preview()
app.spin(lambda: False, delay)  # the worker is now inside snap()
res["previewing_before_disconnect"] = app.vm.is_previewing()

t0 = time.monotonic()
app.disconnect()  # the "Disconnect hardware" click, then wait for the connection thread
res["t_disconnect_call_s"] = round(time.monotonic() - t0, 3)
res["ui_after_disconnect"] = app.buttons()
res["previewing_after_disconnect"] = app.vm.is_previewing()
res["errors_at_disconnect"] = len(app.errors)
res["camera_object_is_none"] = app.session._camera is None

# the orphaned worker: how long does it live after the UI said "disconnected"?
t1 = time.monotonic()
app.spin(lambda: not app.vm.is_previewing(), 12)
res["orphan_thread_lifetime_s"] = round(time.monotonic() - t1, 3)
res["errors_after"] = [e[1] for e in app.errors]
res["ui_after_orphan"] = app.buttons()
app.finish(res)
