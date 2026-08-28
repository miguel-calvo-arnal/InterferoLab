"""Tests for the piezo Z slider (Feature: slider synced with the piezo position).

Covers the loop-free slider<->spinbox sync, the move-on-release-only rule, the
indicator behaviour (position updates from moves, sweep progress and preview
frames), the enable/disable states, and the positionChanged plumbing through
worker -> service -> ViewModel. No hardware: the VM/service are stubbed or run
against FakeSession.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers_acquisition import FakeSession

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

    def test_preview_z_feeds_position_and_frame_still_forwarded(self, acq_vm):
        """Preview frames keep flowing to the UI and feed the position too."""
        positions: list[float] = []
        frames: list[tuple] = []
        acq_vm.positionChanged.connect(positions.append)
        acq_vm.previewFrame.connect(lambda f, z: frames.append((f, z)))
        frame = np.zeros((2, 2), dtype=np.uint16)
        acq_vm.svc.previewFrame.emit(frame, 3.25)
        assert positions == [3.25]
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
