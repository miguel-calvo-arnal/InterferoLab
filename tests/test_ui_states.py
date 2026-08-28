"""Tests for UI state matrices: enable/disable logic and modal notifications.

State is driven by stubbing the ViewModel query methods and by invoking the
panel slots directly; no hardware and no real analysis are ever involved.
QMessageBox dialogs are monkeypatched so nothing modal ever opens.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from views.AcquisitionPanel import AcquisitionPanel
from views.ProcessingPanel import ProcessingPanel
from views.ResultsPanel import ResultsPanel


@pytest.fixture(autouse=True)
def _project_cwd(monkeypatch, project_root):
    """Run from the project root so icons/data/output paths resolve."""
    monkeypatch.chdir(project_root)


@pytest.fixture
def acq_panel(qtbot):
    """AcquisitionPanel with helpers to stub the VM state queries."""
    panel = AcquisitionPanel()
    qtbot.addWidget(panel)

    def set_state(connected=False, running=False, moving=False, preview=False):
        panel._hardware_connected = connected
        panel._preview_active = preview
        panel._vm.is_running = lambda: running
        panel._vm.is_moving = lambda: moving
        panel._update_button_states()

    panel.set_state = set_state
    return panel


@pytest.fixture
def proc_panel(qtbot):
    """ProcessingPanel instance (analysis never started)."""
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def results_panel(qtbot):
    """ResultsPanel instance (no dataset loaded)."""
    panel = ResultsPanel()
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def message_boxes(monkeypatch):
    """Capture QMessageBox static calls instead of opening modal dialogs."""
    calls: dict[str, list[tuple]] = {"warning": [], "critical": [], "information": []}

    def _recorder(kind):
        def _fake(*args, **kwargs):
            calls[kind].append(args)
            return QMessageBox.StandardButton.Ok

        return _fake

    for kind in calls:
        monkeypatch.setattr(QMessageBox, kind, staticmethod(_recorder(kind)))
    return calls


# ======================================================================
#  AcquisitionPanel._update_button_states matrix
# ======================================================================
class TestAcquisitionButtonMatrix:
    def test_disconnected_state(self, acq_panel):
        """Disconnected: only 'Connect hardware' is available; the rest is frozen."""
        acq_panel.set_state(connected=False)
        assert acq_panel.btn_connect.isEnabled()
        assert acq_panel.btn_connect.text() == "Connect hardware"
        assert acq_panel.le_piezo_serial.isEnabled()
        assert not acq_panel.btn_preview.isEnabled()
        assert not acq_panel.btn_start.isEnabled()
        assert not acq_panel.btn_cancel.isEnabled()
        assert not acq_panel.btn_move_piezo.isEnabled()
        assert not acq_panel.btn_apply_cam.isEnabled()
        assert not acq_panel.btn_apply_preview.isEnabled()
        assert not acq_panel.sl_z.isEnabled()

    def test_connected_idle_state(self, acq_panel):
        """Connected and idle: start/preview/move enabled, cancel disabled, serial locked."""
        acq_panel.set_state(connected=True)
        assert acq_panel.btn_connect.isEnabled()
        assert acq_panel.btn_connect.text() == "Disconnect hardware"
        assert not acq_panel.le_piezo_serial.isEnabled()
        assert acq_panel.btn_start.isEnabled()
        assert not acq_panel.btn_cancel.isEnabled()
        assert acq_panel.btn_preview.isEnabled()
        assert acq_panel.btn_preview.text() == "Start preview"
        assert acq_panel.btn_move_piezo.isEnabled()
        assert acq_panel.btn_apply_cam.isEnabled()
        assert acq_panel.btn_apply_preview.isEnabled()
        assert acq_panel.sl_z.isEnabled()
        # Sweep inputs editable while idle
        for sweep_input in (
            acq_panel.sb_start,
            acq_panel.sb_end,
            acq_panel.sb_step,
            acq_panel.cmb_format,
            acq_panel.cmb_channels,
        ):
            assert sweep_input.isEnabled()

    def test_running_state_freezes_sweep_inputs(self, acq_panel):
        """During a sweep: only Cancel is active; sweep spinboxes and combos freeze."""
        acq_panel.set_state(connected=True, running=True)
        assert acq_panel.btn_cancel.isEnabled()
        assert not acq_panel.btn_start.isEnabled()
        assert not acq_panel.btn_connect.isEnabled()
        assert not acq_panel.btn_preview.isEnabled()
        assert not acq_panel.btn_move_piezo.isEnabled()
        assert not acq_panel.btn_apply_cam.isEnabled()
        assert not acq_panel.btn_apply_preview.isEnabled()
        # Slider becomes a read-only position indicator during the sweep
        assert not acq_panel.sl_z.isEnabled()
        # Bloque A: sweep parameters and channel combo frozen while running
        for sweep_input in (
            acq_panel.sb_start,
            acq_panel.sb_end,
            acq_panel.sb_step,
            acq_panel.cmb_format,
            acq_panel.cmb_channels,
        ):
            assert not sweep_input.isEnabled()

    def test_moving_state_is_busy_but_not_cancellable(self, acq_panel):
        """During a manual move: busy buttons disabled, but Cancel stays off (no sweep)."""
        acq_panel.set_state(connected=True, moving=True)
        assert not acq_panel.btn_start.isEnabled()
        assert not acq_panel.btn_cancel.isEnabled()
        assert not acq_panel.btn_connect.isEnabled()
        assert not acq_panel.btn_preview.isEnabled()
        assert not acq_panel.btn_move_piezo.isEnabled()
        assert not acq_panel.sl_z.isEnabled()
        # Not a sweep: sweep inputs stay editable
        assert acq_panel.sb_start.isEnabled()
        assert acq_panel.cmb_channels.isEnabled()

    def test_preview_active_toggles_button_text(self, acq_panel):
        """With the live preview on (idle), the preview button reads 'Stop preview'."""
        acq_panel.set_state(connected=True, preview=True)
        assert acq_panel.btn_preview.isEnabled()
        assert acq_panel.btn_preview.text() == "Stop preview"
        acq_panel.set_state(connected=True, preview=False)
        assert acq_panel.btn_preview.text() == "Start preview"


# ======================================================================
#  AcquisitionPanel channel-mode combo (mono / color / mono_superpixel)
# ======================================================================
class TestChannelModeCombo:
    def test_combo_lists_three_modes_in_schema_order(self, acq_panel):
        """The combo items map 1:1 (by index) onto the persisted config values."""
        from views.AcquisitionPanel import CHANNEL_MODES

        assert acq_panel.cmb_channels.count() == len(CHANNEL_MODES) == 3
        assert CHANNEL_MODES == ("mono", "color", "mono_superpixel")

    @pytest.mark.parametrize(("index", "mode"), [(0, "mono"), (1, "color"), (2, "mono_superpixel")])
    def test_mode_persists_and_restores(self, acq_panel, index, mode):
        """get_config reports the selected mode; apply_saved_config restores it."""
        acq_panel.cmb_channels.setCurrentIndex(index)
        assert acq_panel.get_config()["color_mode"] == mode
        acq_panel.cmb_channels.setCurrentIndex((index + 1) % 3)
        acq_panel.apply_saved_config({"color_mode": mode})
        assert acq_panel.cmb_channels.currentIndex() == index

    def test_unknown_saved_mode_falls_back_to_mono(self, acq_panel):
        """A stale/unknown persisted mode selects 'Mono (weighted)' (index 0)."""
        acq_panel.cmb_channels.setCurrentIndex(1)
        acq_panel.apply_saved_config({"color_mode": "sepia"})
        assert acq_panel.cmb_channels.currentIndex() == 0

    def test_superpixel_mode_forwards_config_and_uses_mono_legend(self, acq_panel):
        """Selecting the superpixel mode applies its config value and the mono legend."""
        applied: list[dict] = []
        acq_panel._vm.apply_config = applied.append
        acq_panel.cmb_channels.setCurrentIndex(2)
        assert applied == [{"color_mode": "mono_superpixel"}]
        assert acq_panel._hist_mode == "mono"


# ======================================================================
#  AcquisitionPanel._on_finished: incomplete dataset warning
# ======================================================================
class TestAcquisitionFinished:
    def test_incomplete_dataset_shows_warning(self, acq_panel, message_boxes):
        """skipped > 0 pops the 'Incomplete dataset' QMessageBox with the counters."""
        acq_panel._on_finished("data/run1", 2, 10)
        assert len(message_boxes["warning"]) == 1
        _parent, title, text = message_boxes["warning"][0][:3]
        assert title == "Incomplete dataset"
        assert "8/10" in text and "2 skipped" in text

    def test_complete_dataset_shows_no_warning(self, acq_panel, message_boxes):
        """skipped == 0 logs completion without any modal dialog."""
        acq_panel._on_finished("data/run1", 0, 10)
        assert message_boxes["warning"] == []
        assert "data/run1" in acq_panel.log_box.toPlainText()


# ======================================================================
#  ProcessingPanel._on_running_changed
# ======================================================================
class TestProcessingRunningState:
    def test_running_freezes_inputs_and_swaps_buttons(self, proc_panel):
        """running=True disables dataset/method/weights/reload and enables Cancel."""
        proc_panel._on_running_changed(True)
        assert not proc_panel._start_btn.isEnabled()
        assert proc_panel._cancel_btn.isEnabled()
        assert not proc_panel._datasets_combo.isEnabled()
        assert not proc_panel._reload_btn.isEnabled()
        assert not proc_panel._method_combo.isEnabled()
        assert not proc_panel._pixel_plots_cb.isEnabled()
        assert not proc_panel._ch_box.isEnabled()

    def test_idle_restores_inputs(self, proc_panel):
        """running=False re-enables the inputs and resets the status line."""
        proc_panel._ch_weights_applicable = True
        proc_panel._on_running_changed(True)
        proc_panel._on_running_changed(False)
        assert proc_panel._start_btn.isEnabled()
        assert not proc_panel._cancel_btn.isEnabled()
        assert proc_panel._datasets_combo.isEnabled()
        assert proc_panel._reload_btn.isEnabled()
        assert proc_panel._method_combo.isEnabled()
        assert proc_panel._pixel_plots_cb.isEnabled()
        assert proc_panel._ch_box.isEnabled()
        assert proc_panel._status_label.text() == "Status: idle"

    def test_idle_keeps_weights_frozen_for_mono_dataset(self, proc_panel):
        """After a run on a mono dataset the weights box stays disabled."""
        proc_panel._ch_weights_applicable = False
        proc_panel._on_running_changed(True)
        proc_panel._on_running_changed(False)
        assert not proc_panel._ch_box.isEnabled()
        assert proc_panel._start_btn.isEnabled()


# ======================================================================
#  ResultsPanel tilt-correction UI
# ======================================================================
class TestResultsTiltUi:
    def test_tilt_ui_on(self, results_panel):
        """_set_tilt_ui(True) checks the button, says 'on' and uses the success variant."""
        results_panel._set_tilt_ui(True)
        assert results_panel._tilt_btn.isChecked()
        assert results_panel._tilt_btn.text() == "Tilt correction: on"
        assert results_panel._tilt_btn.property("variant") == "success"

    def test_tilt_ui_off(self, results_panel):
        """_set_tilt_ui(False) unchecks the button, says 'off' and uses the danger variant."""
        results_panel._set_tilt_ui(True)
        results_panel._set_tilt_ui(False)
        assert not results_panel._tilt_btn.isChecked()
        assert results_panel._tilt_btn.text() == "Tilt correction: off"
        assert results_panel._tilt_btn.property("variant") == "danger"

    def test_vm_error_shows_warning_box(self, results_panel, message_boxes):
        """ResultsVM.error routes to a (patched) QMessageBox.warning."""
        results_panel._vm.error.emit("something went wrong")
        assert len(message_boxes["warning"]) == 1
        assert "something went wrong" in message_boxes["warning"][0][2]

    def test_controls_disabled_while_loading(self, results_panel):
        """_set_controls_enabled toggles the top bar; tilt needs a loaded heightmap."""
        results_panel._set_controls_enabled(False)
        assert not results_panel._load_btn.isEnabled()
        assert not results_panel._refresh_btn.isEnabled()
        assert not results_panel._datasets_combo.isEnabled()
        assert not results_panel._toggle_markers_btn.isEnabled()
        results_panel._set_controls_enabled(True)
        assert results_panel._load_btn.isEnabled()
        # No heightmap loaded: tilt stays disabled even when controls re-enable
        assert not results_panel._tilt_btn.isEnabled()
