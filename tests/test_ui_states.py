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
#  AcquisitionPanel._apply_camera_params (TODO.md item 1 / U5: the timeout
#  used to be silently dropped until the next sweep rebuilt the config)
# ======================================================================
class TestApplyCameraParams:
    def test_applying_camera_params_sends_exposure_and_timeout(self, acq_panel):
        applied: list[dict] = []
        acq_panel._vm.apply_config = applied.append
        acq_panel._vm.connect_camera = lambda: True
        acq_panel.set_state(connected=True)
        acq_panel.sb_exposure.setValue(25.0)
        acq_panel.sb_timeout.setValue(9.0)
        acq_panel._apply_camera_params()
        assert applied == [{"exposure": pytest.approx(0.025), "timeout": 9.0}]
        log = acq_panel.log_box.toPlainText()
        assert "exposure=25.000 ms" in log and "timeout=9.00 s" in log

    def test_applying_camera_params_without_hardware_sends_nothing(self, acq_panel):
        applied: list[dict] = []
        acq_panel._vm.apply_config = applied.append
        acq_panel.set_state(connected=False)
        acq_panel._apply_camera_params()
        assert applied == []


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


# ======================================================================
#  AcquisitionPanel: live preview state follows the camera thread
# ======================================================================
class TestPreviewStateFollowsCameraThread:
    def test_error_does_not_switch_the_preview_off(self, acq_panel, message_boxes):
        """An error (e.g. from the piezo) shows the dialog but leaves the live
        preview flag alone: only the camera thread says when the stream stops."""
        acq_panel.set_state(connected=True, preview=True)
        acq_panel._on_error("MOV failed")
        assert acq_panel._preview_active is True
        assert acq_panel.btn_preview.text() == "Stop preview"
        assert len(message_boxes["critical"]) == 1

    def test_previewing_changed_false_resets_the_button(self, acq_panel):
        """previewingChanged(False) from the VM (camera error) turns the button back."""
        acq_panel.set_state(connected=True, preview=True)
        acq_panel._vm.previewingChanged.emit(False)
        assert acq_panel._preview_active is False
        assert acq_panel.btn_preview.text() == "Start preview"

    def test_toggle_asks_the_vm_to_start_and_stop_the_stream(self, acq_panel):
        """The button drives start_preview/stop_preview, nothing else."""
        calls: list[str] = []
        acq_panel._vm.start_preview = lambda: calls.append("start") or True
        acq_panel._vm.stop_preview = lambda: calls.append("stop")
        acq_panel.set_state(connected=True)
        acq_panel._toggle_preview()
        acq_panel._toggle_preview()
        assert calls == ["start", "stop"]
        assert "continuous" in acq_panel.log_box.toPlainText()

    def test_saved_preview_interval_is_ignored(self, acq_panel):
        """Older configs carry preview_interval: accepted, not persisted any more."""
        acq_panel.apply_saved_config({"preview_interval": 1000, "exposure": 5.0})
        assert "preview_interval" not in acq_panel.get_config()
        assert acq_panel.get_config()["exposure"] == 5.0


# ======================================================================
#  AcquisitionPanel: camera problems without modal dialogs (batch 2)
# ======================================================================
class TestCameraProblemsAreNotModal:
    def test_preview_notice_is_shown_in_the_window_without_a_dialog(self, acq_panel, message_boxes):
        """vm.previewNotice fills the visible status label; nothing modal opens."""
        acq_panel.set_state(connected=True, preview=True)
        assert acq_panel.lbl_camera_status.isHidden()
        text = "No frame from the camera for 5.0 s. Retrying in 1 s (attempt 2 of 3)."
        acq_panel._vm.previewNotice.emit(text)
        assert acq_panel.lbl_camera_status.text() == text
        assert not acq_panel.lbl_camera_status.isHidden()
        assert acq_panel._preview_active is True
        assert all(calls == [] for calls in message_boxes.values())

    def test_empty_notice_hides_the_label(self, acq_panel):
        acq_panel._vm.previewNotice.emit("Live preview error: usb gone.")
        acq_panel._vm.previewNotice.emit("")
        assert acq_panel.lbl_camera_status.isHidden()
        assert acq_panel.lbl_camera_status.text() == ""

    def test_starting_the_preview_again_clears_the_last_problem(self, acq_panel):
        acq_panel._vm.start_preview = lambda: True
        acq_panel.set_state(connected=True)
        acq_panel._vm.previewNotice.emit("Live preview stopped after 3 failed attempts.")
        acq_panel._toggle_preview()
        assert acq_panel.lbl_camera_status.isHidden()

    def test_error_without_text_still_says_something(self, acq_panel, message_boxes):
        """A dialog is never empty (C7): an empty message gets a fallback text."""
        acq_panel._on_error("")
        _parent, _title, text = message_boxes["critical"][0][:3]
        assert text.strip()

    def test_aborted_sweep_reports_the_summary_once(self, acq_panel, message_boxes):
        """sweepAborted + finished: one 'Sweep aborted' warning with the summary,
        not the generic 'Incomplete dataset' one."""
        summary = (
            "Sweep aborted after 3 consecutive camera failures (last at z=50.1000 µm: "
            "usb gone). 2 of 21 frame(s) saved in data/run1."
        )
        acq_panel._vm.sweepAborted.emit(summary)
        acq_panel._on_finished("data/run1", 19, 21)
        assert len(message_boxes["warning"]) == 1
        _parent, title, text = message_boxes["warning"][0][:3]
        assert title == "Sweep aborted"
        assert "2 of 21 frame(s) saved" in text and "SWEEP_ABORTED.txt" in text
        # the next sweep is reported normally again
        acq_panel._on_finished("data/run2", 0, 5)
        assert len(message_boxes["warning"]) == 1


# ======================================================================
#  AcquisitionPanel: connect / disconnect on the connection thread (batch 4)
# ======================================================================
class TestConnectionInProgress:
    def test_connecting_freezes_everything_and_says_so(self, acq_panel):
        acq_panel._vm.connection_busy = lambda: "connecting"
        acq_panel.set_state(connected=False)
        assert acq_panel.btn_connect.text() == "Connecting…"
        for w in (
            acq_panel.btn_connect,
            acq_panel.le_piezo_serial,
            acq_panel.btn_preview,
            acq_panel.btn_start,
            acq_panel.btn_move_piezo,
            acq_panel.btn_apply_cam,
            acq_panel.sl_z,
        ):
            assert not w.isEnabled()

    def test_disconnecting_says_so(self, acq_panel):
        acq_panel._vm.connection_busy = lambda: "disconnecting"
        acq_panel.set_state(connected=True)
        assert acq_panel.btn_connect.text() == "Disconnecting…"
        assert not acq_panel.btn_connect.isEnabled()

    def test_click_while_busy_does_nothing(self, acq_panel):
        calls: list[str] = []
        acq_panel._vm.connection_busy = lambda: "connecting"
        acq_panel._vm.connect_hardware = lambda *a: calls.append("connect") or True
        acq_panel._vm.disconnect_hardware = lambda: calls.append("disconnect") or True
        acq_panel._connect_hw()
        acq_panel._hardware_connected = True
        acq_panel._connect_hw()
        assert calls == []

    def test_click_starts_the_connection_without_waiting(self, acq_panel):
        calls: list[tuple] = []
        acq_panel.le_piezo_serial.setText("123456")
        acq_panel._vm.connect_hardware = lambda serial, dll: calls.append((serial, dll)) or True
        acq_panel._connect_hw()
        assert len(calls) == 1
        assert acq_panel._hardware_connected is False  # only when connectFinished says so
        assert "Connecting hardware" in acq_panel.log_box.toPlainText()

    def test_empty_serial_cancels_the_connect_with_a_warning(self, acq_panel, message_boxes):
        """U9: no hardcoded fallback any more -- an empty field must say so,
        clearly, instead of silently connecting to whatever serial used to
        be baked into the code."""
        calls: list[tuple] = []
        acq_panel.le_piezo_serial.setText("   ")  # blank after stripping
        acq_panel._vm.connect_hardware = lambda serial, dll: calls.append((serial, dll)) or True
        acq_panel._connect_hw()
        assert calls == []
        assert acq_panel._hardware_connected is False
        assert "[WARN] Piezo serial is empty" in acq_panel.log_box.toPlainText()
        assert message_boxes["warning"][0][1] == "Piezo serial required"

    def test_connect_finished_ok_requests_a_snapshot_and_parks_at_50(
        self, acq_panel, message_boxes
    ):
        from views.AcquisitionPanel import PARK_POSITION_UM

        asked: list[str] = []
        acq_panel.moves = []
        acq_panel._vm.request_preview = lambda: asked.append("snap")
        acq_panel._vm.move_to = acq_panel.moves.append
        acq_panel._vm.connectFinished.emit(True)
        assert acq_panel._hardware_connected is True and asked == ["snap"]
        # Item 4/8 of TODO.md: park at a known Z through the normal move
        # path (never the hardware directly), non-blocking.
        assert acq_panel.moves == [PARK_POSITION_UM]
        assert acq_panel._auto_park_pending is True
        assert all(v == [] for v in message_boxes.values())

    def test_connect_failed_shows_the_connection_dialog(self, acq_panel, message_boxes):
        acq_panel._vm.connectFinished.emit(False)
        assert acq_panel._hardware_connected is False
        assert message_boxes["critical"][0][1] == "Hardware connection failed"

    def test_disconnect_finished_resets_the_panel(self, acq_panel):
        acq_panel.set_state(connected=True, preview=True)
        acq_panel._vm.disconnectFinished.emit()
        assert acq_panel._hardware_connected is False and acq_panel._preview_active is False
        assert acq_panel.btn_connect.text() == "Connect hardware"

    def test_disconnect_finished_clears_a_pending_park(self, acq_panel):
        acq_panel._auto_park_pending = True
        acq_panel._vm.disconnectFinished.emit()
        assert acq_panel._auto_park_pending is False


class TestAutoParkAfterConnect:
    """Item 4/8 of TODO.md (Miguel's decision): a failed park move is a log
    warning, never a modal -- the user just connected and did not ask for
    this move themselves."""

    def test_failed_park_is_a_warning_not_a_dialog(self, acq_panel, message_boxes):
        acq_panel._auto_park_pending = True
        acq_panel._vm.error.emit("GCSError: -7 (parameter out of range)")
        assert acq_panel._auto_park_pending is False
        assert message_boxes["critical"] == []
        log = acq_panel.log_box.toPlainText()
        assert "[WARN] Could not park the piezo" in log
        assert "GCSError: -7" in log

    def test_a_real_move_error_still_shows_the_dialog(self, acq_panel, message_boxes):
        """Once the flag is clear (park finished, or the user moved first),
        errors go back to being modal (unchanged behaviour)."""
        acq_panel._auto_park_pending = False
        acq_panel._vm.error.emit("Piezo did not reach target within 30.0 s")
        assert message_boxes["critical"][0][1] == "Acquisition error"

    def test_a_user_move_cancels_a_pending_park_notice(self, acq_panel):
        """The user takes over (button/slider/Enter): a later error is a
        real one again, not attributed to the still-settling park."""
        acq_panel.moves = []
        acq_panel._vm.move_to = acq_panel.moves.append
        acq_panel.set_state(connected=True)
        acq_panel._auto_park_pending = True
        acq_panel.sb_manual_z.setValue(10.0)
        acq_panel._move_piezo_manual()
        assert acq_panel._auto_park_pending is False
        assert acq_panel.moves == [10.0]

    def test_a_successful_park_clears_the_flag_so_the_next_error_is_modal(
        self, acq_panel, message_boxes
    ):
        """H2 (batch 5 review): a park that finishes WITHOUT ever erroring
        never went through _on_error, so the flag used to survive it. A
        later, unrelated failure (a sweep, say) was then wrongly swallowed
        as if it were the park's -- silently, with no modal at all."""
        acq_panel._auto_park_pending = True
        acq_panel._vm.movingChanged.emit(True)  # the park move starts
        assert acq_panel._auto_park_pending is True  # still settling
        acq_panel._vm.movingChanged.emit(False)  # ... and lands cleanly
        assert acq_panel._auto_park_pending is False
        # A later sweep failure must reach the user, not be eaten as a
        # non-existent "park failed" warning.
        acq_panel._vm.error.emit("Camera exposure verification failed")
        assert message_boxes["critical"][0][1] == "Acquisition error"
        assert "Could not park" not in acq_panel.log_box.toPlainText()

    def test_starting_a_sweep_clears_a_pending_park_flag_too(self, acq_panel, message_boxes):
        """Defensive backstop (H2): start() already refuses a sweep while
        is_moving(), so this should not happen in practice, but a running
        sweep must never inherit the park's non-blocking treatment."""
        acq_panel._auto_park_pending = True
        acq_panel._vm.runningChanged.emit(True)
        assert acq_panel._auto_park_pending is False
        acq_panel._vm.error.emit("Sweep failed: camera timeout")
        assert message_boxes["critical"][0][1] == "Acquisition error"


def test_refused_preview_start_never_shows_stop_preview(acq_panel):
    """Review H2 (batch 4): start_preview refused -> the button stays 'Start'."""
    acq_panel._vm.start_preview = lambda: False
    acq_panel.set_state(connected=True)
    acq_panel._toggle_preview()
    assert acq_panel._preview_active is False
    assert acq_panel.btn_preview.text() == "Start preview"
    assert "Preview not started" in acq_panel.log_box.toPlainText()
