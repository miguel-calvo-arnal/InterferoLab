"""
C1b - Disconnect and reconnect while the orphaned preview capture is alive.

Follows C1: right after "Disconnect hardware" the user clicks
"Connect hardware" again (the button is enabled).  A second camera object is
opened while the first capture thread is still running against the closed
one; the orphan's error then lands on the freshly connected session.
"""

import time

from harness import App

app = App()
res = {"connected": app.connect()}

app.vm.request_preview()
app.spin(lambda: False, 0.05)
app.disconnect()  # click + wait for the connection thread (batch 4)
res["previewing_after_disconnect"] = app.vm.is_previewing()
res["connect_button_enabled"] = app.panel.btn_connect.isEnabled()

t0 = time.monotonic()
app.panel._connect_hw()  # reconnect immediately
app.spin(lambda: not app.vm.connection_busy(), 20)
res["t_reconnect_s"] = round(time.monotonic() - t0, 3)
res["reconnected"] = app.panel._hardware_connected
res["previewing_right_after_reconnect"] = app.vm.is_previewing()
res["first_preview_after_reconnect_requested"] = "request" in app.log_text()

app.spin(lambda: len(app.errors) > 0, 12)
res["errors"] = [e[1] for e in app.errors]
res["ui"] = app.buttons()
n = len(app.frames)
app.vm.request_preview()
res["preview_after_error_works"] = app.spin(lambda: len(app.frames) > n, 10)
app.finish(res)
