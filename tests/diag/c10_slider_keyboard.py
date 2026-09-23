"""
C10 - Moving the piezo slider with the keyboard now moves the piezo for real
(batch 5): the slider's key actions (PageUp/Down, arrows, Home/End -- never a
plain drag) go through _PositionSlider.moveRequested, which reuses the same
manual-move path as a mouse release or "Move piezo".

Before batch 5: only sliderReleased (mouse) sent a move; arrow keys and
PageUp/PageDown changed the value (and the Manual Z spinbox) without any
move, and the live preview then re-emitted the stale position and snapped
both widgets back.
"""

from harness import App
from PySide6.QtWidgets import QAbstractSlider

app = App()
res = {"connected": app.connect()}
app.panel._toggle_preview()
app.spin(lambda: len(app.frames) >= 2, 10)

# Known baseline: batch 5 already parks at 50 um on connect, which app.connect()
# waits out, but starting the +5 um relative check from there would make it
# depend on exactly where the park landed.
app.vm.move_to(0.0)
app.spin(lambda: not app.vm.is_moving(), 10)

app.panel.sl_z.setFocus()
for _ in range(5):  # five PageUp presses = +5 um
    app.panel.sl_z.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd)
app.qt.processEvents()
res["slider_after_keys_um"] = app.panel.sl_z.value() / 100.0
res["spinbox_after_keys_um"] = app.panel.sb_manual_z.value()
res["move_started"] = app.vm.is_moving()
app.spin(lambda: False, 1.0)
res["slider_1s_later_um"] = app.panel.sl_z.value() / 100.0
res["spinbox_1s_later_um"] = app.panel.sb_manual_z.value()
res["stage_um"] = round(app.world.stage.position(), 2)
res["any_move_logged"] = "Moving piezo to" in app.log_text()
app.panel._toggle_preview()
app.spin(lambda: not app.vm.is_previewing(), 10)
app.finish(res)
