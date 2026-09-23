"""Tests for the piezo Z slider (Feature: slider synced with the piezo position).

Covers the loop-free slider<->spinbox sync, the move-on-release-only rule, the
indicator behaviour (position updates from moves and sweep progress; preview
frames deliberately do not drive it), the enable/disable states, and the
positionChanged plumbing through worker -> service -> ViewModel. No hardware:
the VM/service are stubbed or run against FakeSession.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers_acquisition import FakeSession
from PySide6.QtWidgets import QAbstractSlider

import services.acquisition_service as acquisition_service_module
from services.acquisition_service import AcquisitionService
from viewmodels.acquisition_vm import AcquisitionVM
from views.AcquisitionPanel import AcquisitionPanel


@pytest.fixture(autouse=True)
def _project_cwd(monkeypatch, project_root):
    """Run from the project root so icons/data/output paths resolve."""
    monkeypatch.chdir(project_root)


@pytest.fixture
def acq_panel(qtbot):
    """AcquisitionPanel with a recorded move_to and stubbable VM state."""
    panel = AcquisitionPanel()
    qtbot.addWidget(panel)

    panel.moves = []
    panel._vm.move_to = panel.moves.append

    def set_state(connected=False, running=False, moving=False, preview=False):
        panel._hardware_connected = connected
        panel._preview_active = preview
        panel._vm.is_running = lambda: running
        panel._vm.is_moving = lambda: moving
        panel._update_button_states()

    panel.set_state = set_state
    return panel


@pytest.fixture
def acq_vm(qtbot):
    """AcquisitionVM instance (service constructed but never connected)."""
    vm = AcquisitionVM()
    yield vm
    vm.deleteLater()


# ======================================================================
#  Slider geometry and slider -> spinbox sync (no hardware moves)
# ======================================================================
class TestSliderSpinboxSync:
    def test_slider_covers_physical_range_at_10nm_resolution(self, acq_panel):
        """0-100 um at 0.01 um resolution -> integer range 0..10000."""
        assert (acq_panel.sl_z.minimum(), acq_panel.sl_z.maximum()) == (0, 10000)
        assert acq_panel.sl_z.toolTip().strip()

    def test_slider_change_updates_spinbox_live_without_moving(self, acq_panel):
        """Changing the slider mirrors the value into Manual Z; no move is sent."""
        acq_panel.set_state(connected=True)
        acq_panel.sl_z.setValue(3750)
        assert acq_panel.sb_manual_z.value() == pytest.approx(37.5)
        assert acq_panel.moves == []

    def test_spinbox_edit_does_not_drag_the_indicator(self, acq_panel):
        """Typing a target in Manual Z leaves the slider on the real position."""
        acq_panel.set_state(connected=True)
        acq_panel._on_position_changed(20.0)
        acq_panel.sb_manual_z.setValue(80.0)
        assert acq_panel.sl_z.value() == 2000  # still the real position
        assert acq_panel.moves == []

    def test_no_signal_loop_between_slider_and_spinbox(self, acq_panel):
        """One slider change produces exactly one valueChanged, not an echo loop."""
        acq_panel.set_state(connected=True)
        slider_events: list[int] = []
        acq_panel.sl_z.valueChanged.connect(slider_events.append)
        acq_panel.sl_z.setValue(1234)
        assert slider_events == [1234]
        assert acq_panel.sb_manual_z.value() == pytest.approx(12.34)


# ======================================================================
#  Physical move only on slider release
# ======================================================================
class TestMoveOnRelease:
    def test_release_sends_exactly_one_move_with_slider_value(self, acq_panel):
        """sliderReleased reuses the manual-move path once, with the synced Z."""
        acq_panel.set_state(connected=True)
        acq_panel.sl_z.setValue(1234)
        acq_panel._on_slider_released()
        assert acq_panel.moves == [pytest.approx(12.34)]
        assert "Moving piezo to 12.340" in acq_panel.log_box.toPlainText()

    def test_drag_ticks_never_move_hardware(self, acq_panel):
        """Many intermediate slider values queue zero moves until release."""
        acq_panel.set_state(connected=True)
        for v in (100, 500, 900, 4200):
            acq_panel.sl_z.setValue(v)
        assert acq_panel.moves == []
        acq_panel._on_slider_released()
        assert acq_panel.moves == [pytest.approx(42.0)]

    def test_release_without_hardware_is_rejected(self, acq_panel):
        """Not connected: release logs a warning and sends nothing."""
        acq_panel.set_state(connected=False)
        acq_panel.sl_z.setValue(500)
        acq_panel._on_slider_released()
        assert acq_panel.moves == []
        assert "[WARN] Hardware not connected." in acq_panel.log_box.toPlainText()

    def test_release_during_sweep_is_rejected(self, acq_panel):
        """Sweep running: release is refused by the existing manual-move guard."""
        acq_panel.set_state(connected=True, running=True)
        acq_panel._on_slider_released()
        assert acq_panel.moves == []
        assert "Cannot move piezo while sweep is running" in acq_panel.log_box.toPlainText()


# ======================================================================
#  Keyboard also moves the real piezo (batch 5: U1/C10, U7)
# ======================================================================
class TestKeyboardMovesTheRealPiezo:
    """The slider's own step actions (arrow/page/home/end -- never a plain
    drag) and Enter/focus-out on Manual Z reuse the exact manual-move path
    as a mouse release or the "Move piezo" button."""

    def test_slider_page_step_key_moves_the_piezo(self, acq_panel, qtbot):
        """PageUp/PageDown (also a groove click): +1 um, the slider's page step."""
        acq_panel.set_state(connected=True)
        acq_panel.sl_z.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd)
        qtbot.wait(20)  # the move is requested one event-loop tick later
        assert acq_panel.moves == [pytest.approx(1.0)]

    def test_slider_single_step_key_uses_the_configured_fine_step(self, acq_panel, qtbot):
        """Arrow keys use the configurable keyboard step (U8), not a fixed 0.1 um."""
        acq_panel.set_state(connected=True)
        acq_panel.sb_keyboard_step_nm.setValue(50.0)  # 0.05 um
        acq_panel.sl_z.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
        qtbot.wait(20)
        assert acq_panel.moves == [pytest.approx(0.05)]

    def test_slider_home_and_end_keys_also_move(self, acq_panel, qtbot):
        acq_panel.set_state(connected=True)
        acq_panel.sl_z.triggerAction(QAbstractSlider.SliderAction.SliderToMaximum)
        qtbot.wait(20)
        assert acq_panel.moves == [pytest.approx(100.0)]

    def test_a_burst_of_key_presses_collapses_to_the_last_value(self, acq_panel, qtbot):
        """Several presses before the event loop turns: every one asks for a
        move (C10's own repro relies on this), but they all carry the SAME
        settled target, and the service's own move_to collapses repeats
        into "the last order wins" -- nothing here builds a queue."""
        acq_panel.set_state(connected=True)
        for _ in range(5):
            acq_panel.sl_z.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd)
        qtbot.wait(20)
        # Every press asked for a move, but the slider had already reached
        # 5.0 by the time any of them ran, so every one carries that value.
        assert acq_panel.moves == [pytest.approx(5.0)] * 5

    def test_slider_drag_is_not_a_keyboard_step(self, acq_panel, qtbot):
        """A drag emits actionTriggered too (SliderMove); it must not be
        treated as a keyboard step (releasing it is its own, existing path:
        test_drag_ticks_never_move_hardware -- not repeated here)."""
        acq_panel.set_state(connected=True)
        acq_panel.sl_z.setSliderDown(True)
        acq_panel.sl_z.setSliderPosition(4000)
        qtbot.wait(20)
        assert acq_panel.moves == []

    def test_keyboard_move_without_hardware_is_rejected_like_the_button(self, acq_panel, qtbot):
        acq_panel.set_state(connected=False)
        acq_panel.sl_z.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd)
        qtbot.wait(20)
        assert acq_panel.moves == []
        assert "[WARN] Hardware not connected." in acq_panel.log_box.toPlainText()

    def test_manual_z_enter_moves_the_piezo(self, acq_panel):
        """returnPressed (Enter) -- U7: previously nothing was connected to
        it at all."""
        acq_panel.set_state(connected=True)
        acq_panel.sb_manual_z.setValue(12.5)
        acq_panel.sb_manual_z.lineEdit().returnPressed.emit()
        assert acq_panel.moves == [pytest.approx(12.5)]
        assert "Moving piezo to 12.500" in acq_panel.log_box.toPlainText()

    def test_manual_z_losing_focus_without_enter_does_not_move(self, acq_panel):
        """H4 (batch 5 review): editingFinished (which also fires on a plain
        Tab/focus-out) used to send an unasked-for move; returnPressed only
        fires on Enter."""
        acq_panel.set_state(connected=True)
        acq_panel.sb_manual_z.setValue(12.5)
        acq_panel.sb_manual_z.editingFinished.emit()  # focus lost, no Enter
        assert acq_panel.moves == []


# ======================================================================
#  Configurable keyboard step (batch 5, U8)
# ======================================================================
class TestKeyboardStepControl:
    def test_default_step_matches_the_documented_default(self, acq_panel):
        from views.AcquisitionPanel import _Z_SLIDER_SCALE, DEFAULT_KEYBOARD_STEP_NM

        assert acq_panel.sb_keyboard_step_nm.value() == DEFAULT_KEYBOARD_STEP_NM
        expected_ticks = round(DEFAULT_KEYBOARD_STEP_NM / 1000.0 * _Z_SLIDER_SCALE)
        assert acq_panel.sl_z.singleStep() == expected_ticks
        assert acq_panel.sb_manual_z.singleStep() == pytest.approx(DEFAULT_KEYBOARD_STEP_NM / 1000.0)

    def test_changing_the_step_updates_both_widgets_live(self, acq_panel):
        acq_panel.sb_keyboard_step_nm.setValue(100.0)  # 0.1 um
        assert acq_panel.sl_z.singleStep() == 10
        assert acq_panel.sb_manual_z.singleStep() == pytest.approx(0.1)

    def test_tiny_step_still_moves_the_slider_by_at_least_one_tick(self, acq_panel):
        acq_panel.sb_keyboard_step_nm.setValue(1.0)  # 0.001 um -> rounds to 0 ticks
        assert acq_panel.sl_z.singleStep() >= 1

    def test_step_round_trips_through_get_and_apply_saved_config(self, acq_panel):
        acq_panel.sb_keyboard_step_nm.setValue(35.0)  # 0.035 um
        cfg = acq_panel.get_config()
        assert cfg["keyboard_step_um"] == pytest.approx(0.035)
        acq_panel.sb_keyboard_step_nm.setValue(20.0)
        acq_panel.apply_saved_config(cfg)
        assert acq_panel.sb_keyboard_step_nm.value() == pytest.approx(35.0)
        assert acq_panel.sl_z.singleStep() == round(0.035 * 100)


# ======================================================================
#  Indicator behaviour (position updates from the VM)
# ======================================================================
class TestPositionIndicator:
    def test_position_update_moves_slider_and_spinbox_without_moves(self, acq_panel):
        """_on_position_changed syncs both widgets and never triggers hardware."""
        acq_panel.set_state(connected=True)
        acq_panel._on_position_changed(42.37)
        assert acq_panel.sl_z.value() == 4237
        assert acq_panel.sb_manual_z.value() == pytest.approx(42.37)
        assert acq_panel.moves == []

    def test_position_update_emits_no_widget_signals(self, acq_panel):
        """Indicator updates are silent: no valueChanged echo from either widget."""
        acq_panel.set_state(connected=True)
        slider_events: list[int] = []
        spin_events: list[float] = []
        acq_panel.sl_z.valueChanged.connect(slider_events.append)
        acq_panel.sb_manual_z.valueChanged.connect(spin_events.append)
        acq_panel._on_position_changed(7.5)
        assert slider_events == []
        assert spin_events == []

    def test_disabled_slider_still_tracks_position_during_sweep(self, acq_panel):
        """During a sweep the slider is disabled for dragging but keeps updating."""
        acq_panel.set_state(connected=True, running=True)
        assert not acq_panel.sl_z.isEnabled()
        acq_panel._on_position_changed(10.0)
        assert acq_panel.sl_z.value() == 1000
        acq_panel._on_position_changed(55.55)
        assert acq_panel.sl_z.value() == 5555

    def test_vm_signal_reaches_the_panel_slot(self, acq_panel):
        """The panel is wired to vm.positionChanged (not just the slot itself)."""
        acq_panel.set_state(connected=True)
        acq_panel._vm.positionChanged.emit(31.25)
        assert acq_panel.sl_z.value() == 3125

    def test_live_update_does_not_reset_the_slider_while_it_has_keyboard_focus(self, acq_panel):
        """H3 (batch 5 review): a measured position landing while the user is
        mid a keyboard advance on the slider must not reset it -- or Manual
        Z, which is what the keyboard path actually reads as its target
        (_move_piezo_manual uses sb_manual_z.value(), not sl_z.value()).
        hasFocus() is stubbed the same way the existing Manual Z guard test
        does, below: real focus is unreliable offscreen."""
        acq_panel.set_state(connected=True)
        acq_panel.sl_z.setValue(300)  # as if 3 page-steps already landed
        acq_panel.sl_z.hasFocus = lambda: True
        acq_panel._on_position_changed(1.0)  # a stale/earlier move's position_cb
        assert acq_panel.sl_z.value() == 300
        assert acq_panel.sb_manual_z.value() == pytest.approx(3.0)
        assert acq_panel.moves == []
        # The reading itself keeps updating -- only the two target widgets freeze
        assert acq_panel.lbl_z_measured.text() == "Measured Z: 1.000 µm"

    def test_slider_drag_still_wins_over_keyboard_focus_check(self, acq_panel):
        """isSliderDown() keeps working as its own, independent guard."""
        acq_panel.set_state(connected=True)
        acq_panel.sl_z.setSliderDown(True)
        acq_panel.sl_z.setValue(500)
        acq_panel._on_position_changed(99.0)
        assert acq_panel.sl_z.value() == 500  # untouched: still mid-drag


# ======================================================================
#  VM: positionChanged fan-in from the three service sources
# ======================================================================
class TestVMPositionSources:
    def test_progress_z_feeds_position(self, acq_vm):
        """Sweep progress reuses its z as a position update."""
        positions: list[float] = []
        acq_vm.positionChanged.connect(positions.append)
        acq_vm.svc.progressChanged.emit(50.0, 12.5, 5, 10, 1.0, 1.0)
        assert positions == [12.5]

    def test_preview_frames_do_not_drive_the_position_indicator(self, acq_vm):
        """Preview frames keep flowing to the UI but never feed the position:
        at ~20 fps that would snap the slider/spinbox back to the set-point
        while the user edits them (C6, C10)."""
        positions: list[float] = []
        frames: list[tuple] = []
        acq_vm.positionChanged.connect(positions.append)
        acq_vm.previewFrame.connect(lambda f, z: frames.append((f, z)))
        frame = np.zeros((2, 2), dtype=np.uint16)
        acq_vm.svc.previewFrame.emit(frame, 3.25)
        assert positions == []
        assert len(frames) == 1 and frames[0][1] == 3.25

    def test_service_move_position_forwarded(self, acq_vm):
        """svc.positionChanged (manual move result) is forwarded unchanged."""
        positions: list[float] = []
        acq_vm.positionChanged.connect(positions.append)
        acq_vm.svc.positionChanged.emit(7.5)
        assert positions == [7.5]


# ======================================================================
#  Service: the move worker reports the reached position
# ======================================================================
def test_move_worker_emits_reached_position(qtbot, monkeypatch):
    """A completed manual move emits positionChanged with the real position."""
    monkeypatch.setattr(acquisition_service_module, "AcquisitionSession", FakeSession)
    svc = AcquisitionService()
    try:
        with qtbot.waitSignal(svc.positionChanged, timeout=3000) as blocker:
            svc.move_to(41.5)
        assert blocker.args == [41.5]
        qtbot.waitUntil(lambda: not svc.is_moving(), timeout=3000)
    finally:
        svc.shutdown(timeout_ms=3000)
        qtbot.wait(10)  # drain deferred deletions (see test_acquisition_service)


# ======================================================================
#  Measured position (batch 3)
# ======================================================================
class TestMeasuredPosition:
    def test_label_shows_the_measured_value(self, acq_panel):
        acq_panel._on_position_changed(60.00406)
        assert acq_panel.lbl_z_measured.text() == "Measured Z: 60.004 µm"
        assert acq_panel.sl_z.value() == 6000

    def test_unknown_position_is_said_and_never_drawn_as_a_number(self, acq_panel):
        acq_panel._on_position_changed(12.5)
        acq_panel._on_position_changed(float("nan"))
        assert "unknown" in acq_panel.lbl_z_measured.text()
        assert acq_panel.sl_z.value() == 1250  # left where it was, not 0
        assert acq_panel.sb_manual_z.value() == 12.5

    def test_live_update_does_not_overwrite_a_target_being_typed(self, acq_panel):
        acq_panel.sb_manual_z.setValue(33.0)
        acq_panel.sb_manual_z.hasFocus = lambda: True
        acq_panel._on_position_changed(20.0)
        assert acq_panel.sb_manual_z.value() == 33.0
        assert acq_panel.sl_z.value() == 2000
        assert acq_panel.lbl_z_measured.text() == "Measured Z: 20.000 µm"


def test_connect_piezo_emits_the_measured_start_position(qtbot, monkeypatch):
    monkeypatch.setattr(acquisition_service_module, "AcquisitionSession", FakeSession)
    svc = AcquisitionService()
    try:
        svc._session.last_position = 47.25
        with qtbot.waitSignal(svc.positionChanged, timeout=1000) as blocker:
            assert svc.connect_piezo("123", "dll") is True
        assert blocker.args == [47.25]
    finally:
        svc.shutdown(timeout_ms=3000)
        qtbot.wait(10)


# ======================================================================
#  Holding a key: end-to-end regression for H3 (batch 5 review)
# ======================================================================
def test_held_key_does_not_lose_steps_to_a_slower_in_flight_move(qtbot, monkeypatch):
    """Reproduces the reviewer's repro: 8 page-step presses ~30 ms apart
    while each move takes ~100 ms (real service + FakeSession, a real
    QThread per move). Before the H3 fix, the belated positionChanged of an
    earlier move reset the slider/Manual Z mid-sequence and 8 presses landed
    at 5 um instead of 8. The slider keeps keyboard focus throughout (Qt
    needs it to deliver the key actions in the first place); stubbed like
    the other focus-guard tests since real focus is unreliable offscreen.
    """
    monkeypatch.setattr(acquisition_service_module, "AcquisitionSession", FakeSession)
    panel = AcquisitionPanel()
    qtbot.addWidget(panel)
    panel._hardware_connected = True
    panel.sl_z.hasFocus = lambda: True

    session = panel._vm.svc._session
    real_move_to = session.move_to

    def slow_move_to(z, log_cb=None, cancel_flag=None, position_cb=None):
        import time

        time.sleep(0.1)  # a move that takes longer than one key press
        return real_move_to(z, log_cb=log_cb, cancel_flag=cancel_flag, position_cb=position_cb)

    session.move_to = slow_move_to

    try:
        for _ in range(8):
            panel.sl_z.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd)
            qtbot.wait(30)

        qtbot.waitUntil(lambda: not panel._vm.is_moving(), timeout=5000)
        qtbot.wait(50)  # let the last positionChanged land

        assert panel.sl_z.value() == 800  # 8.0 um: none of the 8 presses lost
        assert panel.sb_manual_z.value() == pytest.approx(8.0, abs=0.01)
        targets = session.move_calls
        assert targets == sorted(targets)  # every order sent was >= the previous one
        assert targets[-1] == pytest.approx(8.0)
    finally:
        panel._vm.shutdown()
        qtbot.wait(10)
