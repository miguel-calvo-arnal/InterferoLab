"""
C8 - Disconnect (and app close) used to block the GUI for up to 10 s when the
piezo does not report on-target: disconnect_all() ran on the GUI thread (MOV 0
and a 10 s wait for on-target, no cancel flag).

Since batch 4 the disconnect runs on the service's connection thread: the
window keeps painting and answering, the Connect button says
"Disconnecting…" and everything else is frozen until it is over.

Measured here with faults.piezo_never_on_target=true:
* how long the click itself blocks, and when a 100 ms single-shot timer fires
  (it can only fire if the event loop runs);
* whether that timer fired WHILE the disconnect was still in progress (a
  load-independent proof that the GUI was not frozen);
* the buttons during the disconnect, and that a second click is ignored;
* then the app is closed while a second disconnect-like wait would apply:
  how long closing takes with the faulty piezo (shorter park deadline).

Since batch 5 a successful connect also parks the piezo (non-blocking), which
the fault above makes fail after MOVE_TIMEOUT_S too: shortened from outside
(30 s -> 1 s), like c3_position_lie.py, so App.connect()'s wait for it to
settle does not add 30 s to this script.  Unrelated to the deadlines measured
below (PIEZO_PARK_TIMEOUT_S / CLOSE_PARK_TIMEOUT_S, a different constant).
"""

import time

from harness import App
from PySide6.QtCore import QTimer

app = App(sets=["faults.piezo_never_on_target=true"])
from backend.acquisition import acquisition_controller as ac  # noqa: E402

ac.AcquisitionSession.MOVE_TIMEOUT_S = 1.0  # instrumentation, not an app change

res = {"connected": app.connect()}

fired = {}
t0 = time.monotonic()
QTimer.singleShot(
    100,
    lambda: fired.update(
        after_s=round(time.monotonic() - t0, 2), busy_then=app.vm.connection_busy()
    ),
)
app.panel._connect_hw()  # "Disconnect hardware" click
res["t_disconnect_blocking_s"] = round(time.monotonic() - t0, 2)
app.spin(lambda: "after_s" in fired, 2)
res["timer_100ms_fired_after_s"] = fired.get("after_s")
res["timer_fired_while_disconnecting"] = fired.get("busy_then") == "disconnecting"
res["ui_during"] = app.buttons()
app.panel._connect_hw()  # a second click while disconnecting: ignored
res["second_click_ignored"] = app.vm.connection_busy() == "disconnecting"
app.spin(lambda: not app.vm.connection_busy(), 20)
res["t_disconnect_total_s"] = round(time.monotonic() - t0, 2)
res["ui"] = app.buttons()
res["piezo_closed"] = "Piezo connection closed" in app.log_text()

# R9: reconnect -- the click must not block either (opening the camera and
# the piezo takes ~1.4 s estimated), and the window says "Connecting…"
fired2 = {}
t2 = time.monotonic()
QTimer.singleShot(
    100,
    lambda: fired2.update(busy_then=app.vm.connection_busy(), ui=app.buttons()),
)
app.panel._connect_hw()
res["t_connect_blocking_s"] = round(time.monotonic() - t2, 2)
app.spin(lambda: not app.vm.connection_busy(), 30)
res["t_connect_total_s"] = round(time.monotonic() - t2, 2)
res["timer_fired_while_connecting"] = fired2.get("busy_then") == "connecting"
res["ui_while_connecting"] = fired2.get("ui")
app.spin(lambda: bool(app.frames), 10)
res["reconnected"] = app.panel._hardware_connected

# then close the window: the close parks the piezo with a short deadline
t1 = time.monotonic()
app.win.close()
res["t_close_s"] = round(time.monotonic() - t1, 2)
res["closed_piezo_on_close"] = app.log_text().count("Piezo connection closed") >= 2
app.finish(res)
