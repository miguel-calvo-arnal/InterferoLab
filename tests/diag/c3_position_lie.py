"""
C3 - After a failed move the UI shows the OLD position while the stage sits
at the NEW one.

_move_piezo sends MOV and only updates _last_real_pos when _wait_on_target
returns.  If the controller never reports on-target (TimeoutError), the MOV
was executed anyway: the stage is at the target, but every later preview
frame carries the stale _last_real_pos, and the VM re-emits it as
positionChanged, so the slider and the Manual Z spinbox snap back to the
old value.

MOVE_TIMEOUT_S is shortened from outside (30 s -> 1 s) to keep the run short.
Since batch 3 the app reads qPOS during and after the move (also a failed
one): the window must show where the stage is, and follow it while it moves.
"""

from harness import App

app = App(sets=["faults.piezo_never_on_target=true"])
from backend.acquisition import acquisition_controller as ac  # noqa: E402

ac.AcquisitionSession.MOVE_TIMEOUT_S = 1.0  # instrumentation, not an app change

res = {"connected": app.connect()}
res["pos_before"] = app.session._last_real_pos
res["stage_before_um"] = app.world.stage.position()

n_pos = len(app.positions)
t_move = app.now()
app.vm.move_to(60.0)
app.spin(lambda: not app.vm.is_moving(), 10)
t_done = app.now()
app.spin(lambda: False, 0.2)
# batch 3: the indicator follows the measured position while the move lasts
res["positions_during_move"] = [
    [round(t - t_move, 3), z] for t, z in app.positions[n_pos:] if t <= t_done
]
res["measured_label"] = (
    app.panel.lbl_z_measured.text() if hasattr(app.panel, "lbl_z_measured") else None
)
res["move_errors"] = [e[1] for e in app.errors]
res["stage_after_move_um"] = round(app.world.stage.position(), 3)
res["last_real_pos_after_move"] = app.session._last_real_pos
res["ui_after_move"] = app.buttons()

n = len(app.frames)
app.vm.request_preview()
app.spin(lambda: len(app.frames) > n and not app.vm.is_previewing(), 10)
res["preview_z"] = app.frames[-1]["z"] if len(app.frames) > n else None
res["slider_um"] = app.panel.sl_z.value() / 100.0
res["spinbox_um"] = app.panel.sb_manual_z.value()
res["stage_now_um"] = round(app.world.stage.position(), 3)
app.finish(res)
