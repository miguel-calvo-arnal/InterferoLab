# views/AcquisitionPanel.py
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.icons import append_to_log_file, icon, resource_path
from viewmodels.acquisition_vm import AcquisitionVM
from widgets import theme
from widgets.progress_info import ProgressInfoLabel

# Channel-mode config values, ordered like the items of cmb_channels.
# Persisted in app_config.json ("color_mode") and validated by the schema in
# utils/config_manager.py — keep both in sync.
CHANNEL_MODES: tuple[str, ...] = ("mono", "color", "mono_superpixel")

# QSlider is integer-only: the piezo slider stores hundredths of a µm
# (0–100 µm -> 0–10000 ticks, 0.01 µm resolution).
_Z_SLIDER_SCALE = 100

# Z position the piezo is sent to right after a successful connection
# (Miguel's decision, item 4/8 of TODO.md): the centre of the default sweep
# range (45-55 µm), a reasonable place to start looking for fringes.
PARK_POSITION_UM = 50.0

# Default fine step for keyboard moves (arrow keys on the slider, Manual Z's
# own spin arrows): 20 nm, the same value as the default sweep step (0.02 µm)
# -- fine enough to walk through CSI fringes without the old 0.1-1 µm jumps.
DEFAULT_KEYBOARD_STEP_NM = 20.0


class _PositionSlider(QSlider):
    """QSlider whose keyboard actions also ask for a real piezo move.

    Qt's ``sliderReleased`` is mouse-only: arrow keys, Page Up/Down, Home and
    End change the value (``triggerAction`` -> ``setValue``) but never emit
    it, so a bare QSlider only showed a new number on keyboard input without
    ever moving the hardware (U1/C10).  ``actionTriggered`` fires for every
    ``triggerAction`` call -- including a plain mouse drag, reported as
    ``SliderMove`` -- so only the discrete, non-drag actions (single step,
    page step, home/end) ask for a move here; a drag keeps moving nothing
    until release, exactly as before.
    """

    # actionTriggered delivers a plain int (the signal's C++ signature is
    # `void actionTriggered(int)`); QSlider.SliderAction is a plain Enum in
    # PySide6, not an IntEnum, so it does NOT compare equal to that int
    # (verified: `events[0] == QSlider.SliderAction.SliderPageStepAdd` is
    # False even though both are 3) -- compare against `.value` instead.
    _STEP_ACTIONS = frozenset(
        a.value
        for a in (
            QSlider.SliderAction.SliderSingleStepAdd,
            QSlider.SliderAction.SliderSingleStepSub,
            QSlider.SliderAction.SliderPageStepAdd,
            QSlider.SliderAction.SliderPageStepSub,
            QSlider.SliderAction.SliderToMinimum,
            QSlider.SliderAction.SliderToMaximum,
        )
    )

    moveRequested = Signal()

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.actionTriggered.connect(self._on_action_triggered)

    def _on_action_triggered(self, action: int) -> None:
        if action not in self._STEP_ACTIONS:
            return
        # actionTriggered fires BEFORE the slider's own value() reflects the
        # step (verified against PySide6: value() inside this handler is
        # still the OLD one; valueChanged, with the new value, follows right
        # after) -- deferred one tick so Manual Z (fed by valueChanged) has
        # already caught up by the time the move reads it.  self as context:
        # cancelled automatically if the slider is destroyed meanwhile.
        QTimer.singleShot(0, self, self.moveRequested.emit)


class AcquisitionPanel(QWidget):
    """
    Acquisition panel with:
            - Full camera/piezo control
            - Manual movement (never blocked by the live preview)
            - Sweep + continuous live preview (paused by the sweep, resumed after)
            - Efficient histogram
            - Real-time progress (elapsed + ETA)
            - Logging
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._vm = AcquisitionVM(self)
        # Backward-compatible public alias (e.g. MainWindow.closeEvent uses it)
        self.vm = self._vm
        self._hardware_connected = False
        # Mirrors the live preview stream: set optimistically on the button
        # and corrected by vm.previewingChanged (e.g. stopped by a camera error).
        self._preview_active = False
        # Summary of the last sweep that stopped early (set by sweepAborted,
        # consumed by the finished() that always follows it).
        self._sweep_abort_summary = ""
        # True from the moment the post-connect park move (PARK_POSITION_UM)
        # is sent until it reports back; a failure then is logged, not shown
        # as a modal (item 4/8 of TODO.md).  Any move the user asks for
        # meanwhile clears it: a later error is then a real one again.
        self._auto_park_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ============================================================
        #  Servo block
        # ============================================================
        servo_box = QGroupBox("Servo parameters")
        servo_layout = QGridLayout(servo_box)

        r = 0
        servo_layout.addWidget(QLabel("Piezo serial:"), r, 0)
        # No hardcoded default: the serial is device-specific and persists in
        # app_config.json once entered.
        self.le_piezo_serial = QLineEdit("")
        self.le_piezo_serial.setPlaceholderText("controller serial, e.g. 0123456789")
        self.le_piezo_serial.setToolTip(
            "Serial number of the PI piezo controller (locked while hardware is connected)."
        )
        servo_layout.addWidget(self.le_piezo_serial, r, 1)
        r += 1

        servo_layout.addWidget(QLabel("Start (µm):"), r, 0)
        self.sb_start = QDoubleSpinBox()
        self.sb_start.setRange(0, 100)
        self.sb_start.setDecimals(3)
        self.sb_start.setValue(45)
        self.sb_start.setToolTip("Sweep start position, in µm (0–100).")
        servo_layout.addWidget(self.sb_start, r, 1)
        r += 1

        servo_layout.addWidget(QLabel("End (µm):"), r, 0)
        self.sb_end = QDoubleSpinBox()
        self.sb_end.setRange(0, 100)
        self.sb_end.setDecimals(3)
        self.sb_end.setValue(55)
        self.sb_end.setToolTip("Sweep end position, in µm (0–100, always ≥ start).")
        servo_layout.addWidget(self.sb_end, r, 1)
        r += 1

        servo_layout.addWidget(QLabel("Step (µm):"), r, 0)
        self.sb_step = QDoubleSpinBox()
        self.sb_step.setRange(0, 1)
        self.sb_step.setDecimals(3)
        self.sb_step.setSingleStep(0.001)
        self.sb_step.setValue(0.02)
        self.sb_step.setToolTip("Z increment between consecutive frames, in µm (0.001–1).")
        servo_layout.addWidget(self.sb_step, r, 1)
        r += 1

        servo_layout.addWidget(QLabel("Manual Z (µm):"), r, 0)
        self.sb_manual_z = QDoubleSpinBox()
        self.sb_manual_z.setRange(0, 100)
        self.sb_manual_z.setDecimals(3)
        self.sb_manual_z.setToolTip(
            "Target position for a manual piezo move, in µm (0–100).\n"
            "Press Enter to move the piezo there; its own arrow keys step\n"
            "by the keyboard step set below."
        )
        servo_layout.addWidget(self.sb_manual_z, r, 1)
        r += 1

        servo_layout.addWidget(QLabel("Keyboard step (nm):"), r, 0)
        self.sb_keyboard_step_nm = QDoubleSpinBox()
        self.sb_keyboard_step_nm.setRange(1.0, 2000.0)
        self.sb_keyboard_step_nm.setDecimals(1)
        self.sb_keyboard_step_nm.setValue(DEFAULT_KEYBOARD_STEP_NM)
        self.sb_keyboard_step_nm.setToolTip(
            "Step size for a single arrow-key press on the Z slider or on\n"
            "Manual Z, in nm. Fine enough to walk through fringes; coarser\n"
            "searches still use the slider's page step (1 µm) or a typed\n"
            "target. Saved in the configuration."
        )
        servo_layout.addWidget(self.sb_keyboard_step_nm, r, 1)
        r += 1

        # Piezo position slider: control + live indicator of the real position
        self.sl_z = _PositionSlider(Qt.Horizontal)
        self.sl_z.setRange(0, 100 * _Z_SLIDER_SCALE)  # 0–100 µm, 0.01 µm ticks
        self.sl_z.setPageStep(_Z_SLIDER_SCALE)  # 1 µm per groove click
        self.sl_z.setToolTip(
            "Piezo Z position, 0–100 µm (0.01 µm resolution).\n"
            "Drag to pick a target (Manual Z follows live); the piezo moves\n"
            "when the handle is released, also while the live preview runs.\n"
            "Arrow keys, Page Up/Down, Home and End move it for real too, by\n"
            "the keyboard step set above (Page/Home/End use the page step).\n"
            "Shows the MEASURED position read from the controller: live during\n"
            "moves, after each move (also a failed one) and during sweeps\n"
            "(read-only while busy)."
        )
        servo_layout.addWidget(self.sl_z, r, 0, 1, 2)
        r += 1

        # The measured position as a number: the slider alone cannot show
        # nanometres, and "unknown" must be said, not drawn as 0.
        self.lbl_z_measured = QLabel("Measured Z: —")
        self.lbl_z_measured.setToolTip(
            "Piezo position read from the controller (qPOS), not the target."
        )
        servo_layout.addWidget(self.lbl_z_measured, r, 0, 1, 2)
        r += 1

        self.btn_move_piezo = QPushButton("Move piezo")
        self.btn_move_piezo.setIcon(icon("move-vertical.svg"))
        self.btn_move_piezo.setToolTip("Move the piezo to the manual Z position now.")
        servo_layout.addWidget(self.btn_move_piezo, r, 0, 1, 2)
        r += 1

        # ============================================================
        #  Camera block
        # ============================================================
        cam_box = QGroupBox("Camera parameters")
        cam_layout = QGridLayout(cam_box)
        r = 0

        cam_layout.addWidget(QLabel("Exposure (ms):"), r, 0)
        self.sb_exposure = QDoubleSpinBox()
        self.sb_exposure.setRange(0.01, 10000)
        self.sb_exposure.setDecimals(3)
        self.sb_exposure.setValue(7.0)
        self.sb_exposure.setToolTip(
            "Camera exposure time per frame, in ms (0.01–10000).\n"
            "Applied on connect, or with 'Apply camera settings'."
        )
        cam_layout.addWidget(self.sb_exposure, r, 1)
        r += 1

        cam_layout.addWidget(QLabel("Timeout (s):"), r, 0)
        self.sb_timeout = QDoubleSpinBox()
        self.sb_timeout.setRange(0.5, 10)
        self.sb_timeout.setValue(5.0)
        self.sb_timeout.setToolTip("Maximum wait for a camera frame, in s (0.5–10).")
        cam_layout.addWidget(self.sb_timeout, r, 1)
        r += 1

        self.btn_apply_cam = QPushButton("Apply camera settings")
        self.btn_apply_cam.setIcon(icon("check.svg"))
        self.btn_apply_cam.setToolTip("Apply the exposure setting to the connected camera.")
        cam_layout.addWidget(self.btn_apply_cam, r, 0, 1, 2)

        # ============================================================
        #  Output block
        # ============================================================
        out_box = QGroupBox("Scan / Output")
        out_layout = QGridLayout(out_box)
        r = 0

        out_layout.addWidget(QLabel("Format:"), r, 0)
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["bin12", "tiff", "png"])
        self.cmb_format.setToolTip(
            "File format for saved frames.\n"
            "bin12: packed 12-bit raw (smallest, fastest). tiff/png: 16-bit images."
        )
        out_layout.addWidget(self.cmb_format, r, 1)
        r += 1

        out_layout.addWidget(QLabel("Channels:"), r, 0)
        self.cmb_channels = QComboBox()
        self.cmb_channels.addItems(["Mono (weighted)", "Color (R, G, B)", "Mono (superpixel)"])
        self.cmb_channels.setToolTip(
            "Mono: weighted 2×2 Bayer merge → single channel (half resolution).\n"
            "Color: 2×2 Bayer merge → R, G, B channels saved separately.\n"
            "Mono (superpixel): weighted merge + extra 2×2 binning → quarter\n"
            "resolution per axis; smallest stacks, ready to analyze (same output\n"
            "as scripts/process_superpixel.py).\n"
            "Use Color when one channel may be corrupted."
        )
        out_layout.addWidget(self.cmb_channels, r, 1)
        # No "preview interval" any more: the live preview is a continuous
        # stream (camera armed once) that always shows the newest frame.

        layout.addLayout(self._make_hbox(servo_box, cam_box, out_box))

        # ============================================================
        #  Preview + histogram
        # ============================================================
        preview_row = QHBoxLayout()

        self.preview_label = QLabel("Live preview")
        self.preview_label.setObjectName("previewLabel")  # styled in styles.qss
        # Keep the floor low so the whole window still fits small laptop
        # screens; the preview grows with the layout stretch on larger ones.
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_row.addWidget(self.preview_label, stretch=3)

        self._hist_plot = pg.PlotWidget()
        self._hist_plot.setBackground(theme.MATPLOTLIB_BG)
        self._hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self._hist_plot.setLabel("bottom", "Intensity")
        self._hist_plot.setLabel("left", "Count")
        for axis in ("left", "bottom"):
            ax = self._hist_plot.getAxis(axis)
            ax.setPen("k")
            ax.setTextPen("k")

        self._hist_legend = self._hist_plot.addLegend(
            offset=(-10, 10),
            pen=pg.mkPen("#dfe3e7"),
            brush=theme.brush("#ffffff", 190),
            labelTextColor=theme.TEXT_PRIMARY,
        )
        self._hist_r = self._hist_plot.plot(pen=pg.mkPen(theme.MATPLOTLIB_RED, width=1))
        self._hist_g = self._hist_plot.plot(pen=pg.mkPen(theme.MATPLOTLIB_GREEN, width=1))
        self._hist_b = self._hist_plot.plot(pen=pg.mkPen(theme.MATPLOTLIB_BLUE, width=1))
        self._hist_mono = self._hist_plot.plot(pen=pg.mkPen(theme.NEUTRAL, width=1))
        self._hist_mode: str | None = None  # "mono" | "color" (drives the legend)
        # Initial legend matches the selected channel mode (frames may not
        # have arrived yet, but an empty legend box would look broken)
        self._set_histogram_mode("color" if self._current_channel_mode() == "color" else "mono")

        preview_row.addWidget(self._hist_plot, stretch=2)
        layout.addLayout(preview_row)

        # Camera problems that must not interrupt the user (live preview
        # retrying or given up, exposure not applied): shown here, in the
        # window, never as a modal dialog.  Hidden while there is nothing to say.
        self.lbl_camera_status = QLabel("")
        self.lbl_camera_status.setObjectName("cameraStatus")  # styled in styles.qss
        self.lbl_camera_status.setWordWrap(True)
        self.lbl_camera_status.setVisible(False)
        layout.addWidget(self.lbl_camera_status)

        # ============================================================
        #  Progress + ETA
        # ============================================================
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_row.addWidget(self.progress, stretch=3)

        self._progress_info = ProgressInfoLabel()
        progress_row.addWidget(self._progress_info)
        layout.addLayout(progress_row)

        # ============================================================
        #  Buttons
        # ============================================================
        btns = QHBoxLayout()
        # Icons for the connect/preview toggles (white where the button is filled)
        self._icon_plug = icon("plug.svg", "#ffffff")
        self._icon_plug_off = icon("plug-off.svg", "#ffffff")
        self._icon_eye = icon("eye.svg")
        self._icon_eye_off = icon("eye-off.svg")

        self.btn_connect = QPushButton("Connect hardware")
        theme.set_variant(self.btn_connect, "primary")
        self.btn_connect.setIcon(self._icon_plug)
        self.btn_connect.setToolTip("Connect or disconnect the piezo and the camera.")

        self.btn_preview = QPushButton("Start preview")
        self.btn_preview.setIcon(self._icon_eye)
        self.btn_preview.setToolTip(
            "Start or stop the live camera preview (continuous, newest frame shown)."
        )

        self.btn_start = QPushButton("Start sweep")
        theme.set_variant(self.btn_start, "primary")
        self.btn_start.setIcon(icon("play.svg", "#ffffff"))
        self.btn_start.setToolTip(
            "Acquire one frame per Z step from start to end position (Ctrl+Return)."
        )

        self.btn_cancel = QPushButton("Cancel")
        theme.set_variant(self.btn_cancel, "danger")
        self.btn_cancel.setIcon(icon("stop.svg", "#ffffff"))
        self.btn_cancel.setToolTip("Cancel the running sweep (Esc).")
        self.btn_cancel.setEnabled(False)

        btns.addWidget(self.btn_connect)
        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

        # Logs
        self.log_box = QTextEdit()
        self.log_box.setObjectName("logView")  # monospace log styling in styles.qss
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        # ============================================================
        #  Connections
        # ============================================================
        self.sb_start.valueChanged.connect(self._on_start_changed)
        self.sb_end.valueChanged.connect(self._on_end_changed)

        self.btn_connect.clicked.connect(self._connect_hw)
        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_move_piezo.clicked.connect(self._move_piezo_manual)
        self.btn_apply_cam.clicked.connect(self._apply_camera_params)
        self.cmb_channels.currentIndexChanged.connect(self._on_channel_mode_changed)

        # Piezo slider <-> spinbox <-> real position (loop-free: the guard
        # flag plus blockSignals in _on_position_changed break every cycle).
        self._z_sync_guard = False
        self.sl_z.valueChanged.connect(self._on_slider_value_changed)
        self.sl_z.sliderReleased.connect(self._on_slider_released)
        self._vm.positionChanged.connect(self._on_position_changed)

        # Keyboard moves the piezo for real (U1/C10, U7): the slider's own
        # step actions (arrow/page/home/end, not a drag) and Enter on Manual Z
        # both reuse the normal manual-move path.  returnPressed, not
        # editingFinished (H4, batch 5 review): editingFinished also fires on
        # a plain focus loss (e.g. Tab, or clicking "Move piezo" itself),
        # which used to send an extra, unasked-for move to whatever the field
        # happened to show.
        self.sl_z.moveRequested.connect(self._move_piezo_manual)
        self.sb_manual_z.lineEdit().returnPressed.connect(self._move_piezo_manual)
        self.sb_keyboard_step_nm.valueChanged.connect(self._on_keyboard_step_changed)
        self._on_keyboard_step_changed(self.sb_keyboard_step_nm.value())

        self._vm.progressChanged.connect(self._on_progress)
        self._vm.etaChanged.connect(self._progress_info.set_eta)
        self._vm.elapsedChanged.connect(self._progress_info.set_elapsed)
        self._vm.statusTextChanged.connect(self._append_log)
        self._vm.previewFrame.connect(self._on_preview)
        self._vm.runningChanged.connect(self._on_running)
        self._vm.movingChanged.connect(self._on_moving)
        self._vm.previewingChanged.connect(self._on_previewing_changed)
        self._vm.previewNotice.connect(self._on_preview_notice)
        self._vm.sweepAborted.connect(self._on_sweep_aborted)
        self._vm.finished.connect(self._on_finished)
        self._vm.connectFinished.connect(self._on_connect_finished)
        self._vm.disconnectFinished.connect(self._on_disconnect_finished)
        self._vm.error.connect(self._on_error)

        self._update_button_states()

    # ============================================================
    #  Helpers
    # ============================================================
    def _make_hbox(self, *widgets):
        h = QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        return h

    def _exposure_s(self) -> float:
        """Return the camera exposure from the UI, converted from ms to seconds."""
        return self.sb_exposure.value() / 1000.0

    def _current_channel_mode(self) -> str:
        """Return the config value ("mono"/"color"/"mono_superpixel") of the combo."""
        index = self.cmb_channels.currentIndex()
        return CHANNEL_MODES[index] if 0 <= index < len(CHANNEL_MODES) else "mono"

    def _update_button_states(self):
        running = self._vm.is_running()
        moving = self._vm.is_moving()
        busy = running or moving  # any background activity

        # Freeze the channel mode during a sweep: changing it mid-sweep
        # would be rejected by the service anyway (mixed-dataset guard).
        self.cmb_channels.setEnabled(not running)

        # Freeze the sweep parameters and output format too: they are captured
        # once at start, so editing them mid-sweep would only mislead.
        for sweep_input in (self.sb_start, self.sb_end, self.sb_step, self.cmb_format):
            sweep_input.setEnabled(not running)

        # Piezo slider: dragging needs hardware and no background activity.
        # While busy (sweep/move) it stays visible as a read-only position
        # indicator — setValue still works on a disabled QSlider.
        self.sl_z.setEnabled(self._hardware_connected and not busy)

        connection = self._vm.connection_busy()  # "connecting" / "disconnecting" / ""
        if connection:
            # Everything frozen until the connection thread reports back:
            # no second connect, no disconnect half-way through a connect.
            self.btn_connect.setText(
                "Connecting…" if connection == "connecting" else "Disconnecting…"
            )
            for w in (
                self.btn_connect,
                self.le_piezo_serial,
                self.btn_preview,
                self.btn_start,
                self.btn_cancel,
                self.btn_move_piezo,
                self.btn_apply_cam,
                self.sl_z,
            ):
                w.setEnabled(False)
            return

        if not self._hardware_connected:
            self.btn_connect.setText("Connect hardware")
            self.btn_connect.setIcon(self._icon_plug)
            self.btn_connect.setEnabled(True)
            self.le_piezo_serial.setEnabled(True)

            self.btn_preview.setEnabled(False)
            self.btn_preview.setText("Start preview")
            self.btn_preview.setIcon(self._icon_eye)

            self.btn_start.setEnabled(False)
            self.btn_cancel.setEnabled(False)

            self.btn_move_piezo.setEnabled(False)
            self.btn_apply_cam.setEnabled(False)
            return

        # Hardware connected: lock the serial field to prevent accidental changes
        self.le_piezo_serial.setEnabled(False)
        self.btn_connect.setText("Disconnect hardware")
        self.btn_connect.setIcon(self._icon_plug_off)
        self.btn_connect.setEnabled(not busy)

        self.btn_start.setEnabled(not busy)
        self.btn_cancel.setEnabled(running)  # cancel only meaningful during sweep
        self.btn_move_piezo.setEnabled(not busy)
        # Exposure changes are applied by the camera thread between frames,
        # so they are allowed while the live preview runs.
        self.btn_apply_cam.setEnabled(not busy)

        # Preview
        if busy:
            self.btn_preview.setEnabled(False)
            self.btn_preview.setText("Start preview")
            self.btn_preview.setIcon(self._icon_eye)
        else:
            self.btn_preview.setEnabled(True)
            self.btn_preview.setText("Stop preview" if self._preview_active else "Start preview")
            self.btn_preview.setIcon(self._icon_eye_off if self._preview_active else self._icon_eye)

    # ============================================================
    #  Histogram
    # ============================================================
    def _set_histogram_mode(self, mode: str) -> None:
        """Rebuild the legend when the histogram switches between mono and RGB."""
        if mode == self._hist_mode:
            return
        self._hist_mode = mode
        self._hist_legend.clear()
        if mode == "mono":
            self._hist_legend.addItem(self._hist_mono, "Mono")
        else:
            self._hist_legend.addItem(self._hist_r, "R")
            self._hist_legend.addItem(self._hist_g, "G")
            self._hist_legend.addItem(self._hist_b, "B")

    def _update_histogram(self, frame: Any):
        """Paint the preview histogram; the data is computed by the VM."""
        histograms = self._vm.compute_preview_histogram(frame)
        if len(histograms) == 1:
            # Mono frame: single neutral curve; hide the RGB ones
            self._set_histogram_mode("mono")
            centres, counts = histograms[0]
            self._hist_mono.setData(centres, counts)
            for curve in (self._hist_r, self._hist_g, self._hist_b):
                curve.setData([], [])
        elif len(histograms) == 3:
            self._set_histogram_mode("color")
            self._hist_mono.setData([], [])
            curves = [self._hist_r, self._hist_g, self._hist_b]
            for (centres, counts), curve in zip(histograms, curves, strict=False):
                curve.setData(centres, counts)

    # ============================================================
    #  Slots - Hardware, Preview, Sweep
    # ============================================================
    @Slot()
    def _connect_hw(self) -> None:
        """
        Connect or disconnect hardware depending on current state.

        Both run on the service's connection thread (the window keeps
        painting and answering, findings C8/R9); the button says
        "Connecting…" / "Disconnecting…" and everything else is frozen until
        connectFinished / disconnectFinished arrive.  A second click while
        one is in progress does nothing.
        """
        if self._vm.connection_busy():
            return

        if self._hardware_connected:
            # Never in the middle of a sweep or a move (they use the piezo;
            # the button is disabled then anyway).  The camera is closed by
            # its own thread; the GUI never touches it.
            if self._vm.is_running() or self._vm.is_moving():
                return
            if self._vm.disconnect_hardware():
                self._preview_active = False
                self._on_preview_notice("")
                self._append_log("Disconnecting hardware…")
            self._update_button_states()
            return

        # 1) Apply current UI parameters to backend configuration BEFORE connecting
        self._vm.apply_config(
            {
                "exposure": self._exposure_s(),
                "timeout": self.sb_timeout.value(),
                "closed_loop": True,
                "axis": "A",
            }
        )

        # 2) Connect hardware using that configuration (connection thread)
        serial = self.le_piezo_serial.text().strip()
        if not serial:
            # No hardcoded fallback (U9): an empty field used to connect
            # silently to whichever controller happened to be in the code.
            self._append_log("[WARN] Piezo serial is empty; connect cancelled.")
            QMessageBox.warning(
                self,
                "Piezo serial required",
                "Enter the piezo controller's serial number before connecting.\n"
                "It is saved to your local configuration once entered.",
            )
            return
        dll_path = resource_path(os.path.join("API", "PI", "E816_DLL_x64.dll"))
        if self._vm.connect_hardware(serial, dll_path):
            self._append_log("Connecting hardware…")
        self._update_button_states()

    @Slot(bool)
    def _on_connect_finished(self, ok: bool) -> None:
        """The connection thread is done: connected, or failed and released."""
        self._hardware_connected = bool(ok)
        self._preview_active = False
        if ok:
            # Preview always starts OFF after connecting; the user enables it
            # explicitly with the Start preview button.  One snapshot shows
            # the field of view meanwhile.
            self._vm.request_preview()
            self._append_log(
                f"Hardware connected successfully. Parameters applied "
                f"(exposure={self.sb_exposure.value():.3f} ms, "
                f"timeout={self.sb_timeout.value():.2f} s)."
            )
            # Item 4/8 of TODO.md (Miguel's decision): park at a known,
            # centred Z after connecting.  Same path as any manual move (the
            # service, never the hardware directly) and non-blocking: the
            # window is usable immediately, the preview above keeps running,
            # and _on_error shows a log warning instead of a modal if it fails.
            self._auto_park_pending = True
            self._append_log(f"Parking piezo at {PARK_POSITION_UM:.1f} µm…")
            self._vm.move_to(PARK_POSITION_UM)
            self._update_button_states()
            return
        # Whatever was opened has already been released by the service.
        self._update_button_states()
        self._append_log("[ERROR] Could not connect hardware. See log for details.")
        QMessageBox.critical(
            self,
            "Hardware connection failed",
            "Could not connect to piezo or camera.\n\n"
            "Check that the hardware is powered on, the serial number is correct, "
            "and no other application is using the devices.",
        )

    @Slot()
    def _on_disconnect_finished(self) -> None:
        self._hardware_connected = False
        self._preview_active = False
        self._auto_park_pending = False
        self._append_log("Hardware disconnected.")
        self._update_button_states()

    @Slot()
    def _toggle_preview(self):
        if not self._hardware_connected:
            self._append_log("[WARN] Hardware not connected.")
            return

        if self._vm.is_running():
            self._append_log("[WARN] Cannot start preview during sweep.")
            return

        self._preview_active = not self._preview_active

        if self._preview_active:
            self._on_preview_notice("")  # a fresh start: forget the last problem
            if not self._vm.start_preview():
                # Refused (hardware connecting/disconnecting): never show
                # "Stop preview" without a stream (review H2 of batch 4).
                self._preview_active = False
                self._append_log("[WARN] Preview not started: hardware is busy.")
                self._update_button_states()
                return
            self._append_log("Preview started (continuous).")
        else:
            self._vm.stop_preview()
            self._append_log("Preview stopped.")

        self._update_button_states()

    @Slot(bool)
    def _on_previewing_changed(self, active: bool) -> None:
        """The camera thread says the live preview is on/off (also after an error)."""
        self._preview_active = bool(active)
        self._update_button_states()

    @Slot(str)
    def _on_preview_notice(self, text: str) -> None:
        """Show (or clear, with "") a camera problem without interrupting the user."""
        self.lbl_camera_status.setText(text)
        self.lbl_camera_status.setVisible(bool(text))

    @Slot(int)
    def _on_channel_mode_changed(self, index: int) -> None:
        mode = CHANNEL_MODES[index] if 0 <= index < len(CHANNEL_MODES) else "mono"
        self._vm.apply_config({"color_mode": mode})
        self._set_histogram_mode("color" if mode == "color" else "mono")

    @Slot()
    def _start(self):
        # No waiting for the preview here: the sweep runs on the camera
        # thread, which stops the live stream itself before the first step
        # and re-arms it afterwards if it was on.  Nothing spins the event
        # loop, so no click can sneak in while a sweep is being started.
        cfg = {
            "closed_loop": True,
            "start": self.sb_start.value(),
            "end": self.sb_end.value(),
            "step": self.sb_step.value(),
            "exposure": self._exposure_s(),
            "timeout": self.sb_timeout.value(),
            "format": self.cmb_format.currentText(),
            "color_mode": self._current_channel_mode(),
            "axis": "A",
            "settle": 0.2,
            "output_folder": "data",
        }

        if self._vm.apply_and_start(cfg):
            self._append_log("Sweep started…")
        else:
            self._append_log("[ERROR] Sweep could not be started")

        self._update_button_states()

    @Slot()
    def _cancel(self):
        self._vm.cancel()
        self._append_log("Cancel requested…")
        self._update_button_states()

    @Slot()
    def _move_piezo_manual(self):
        # The user is taking over (button, slider release/keys, or Enter on
        # Manual Z): a later move error is a real one again, not the
        # post-connect park's (which may still be settling in the background).
        self._auto_park_pending = False
        if not self._hardware_connected:
            self._append_log("[WARN] Hardware not connected.")
            return
        if self._vm.is_running():
            self._append_log("[WARN] Cannot move piezo while sweep is running.")
            return

        z = self.sb_manual_z.value()
        self._append_log(f"Moving piezo to {z:.3f} µm…")
        self._vm.move_to(z)

    # ============================================================
    #  Piezo slider (control + position indicator)
    # ============================================================
    @Slot(float)
    def _on_keyboard_step_changed(self, value_nm: float) -> None:
        """Apply the fine keyboard step (nm) to the slider's arrow-key step
        and to Manual Z's own spin arrows (U8: the Qt default -- 0.1 µm on
        the slider, 1 µm on the spinbox -- was too coarse for walking
        through fringes)."""
        step_um = value_nm / 1000.0
        self.sl_z.setSingleStep(max(1, round(step_um * _Z_SLIDER_SCALE)))
        self.sb_manual_z.setSingleStep(step_um)

    @Slot(int)
    def _on_slider_value_changed(self, value: int) -> None:
        """Mirror slider drags into the Manual Z spinbox. Never moves hardware."""
        if self._z_sync_guard:
            return
        self._z_sync_guard = True
        try:
            self.sb_manual_z.setValue(value / _Z_SLIDER_SCALE)
        finally:
            self._z_sync_guard = False

    @Slot()
    def _on_slider_released(self) -> None:
        """Send the physical move once, when the slider handle is released.

        Moves are deliberately NOT sent on every tick while dragging (that
        would flood the piezo); the spinbox already tracked the drag, so the
        existing manual-move path (with its guards and logging) is reused.
        """
        self._move_piezo_manual()

    @Slot(float)
    def _on_position_changed(self, z: float) -> None:
        """Reflect the MEASURED piezo position (NaN = unknown).

        Fed by the connection, manual moves (live and at the end, also a
        failed one) and sweep progress; never by preview frames.  Widget
        signals are blocked so an indicator update can never trigger a move.
        The slider is skipped while the user drags it OR while it has
        keyboard focus (H3, batch 5 review: arrow/page/home/end keys need
        that focus, and Qt computes each step from the slider's OWN current
        value -- overwriting it mid-sequence lost steps on a held key, 8
        presses landing at 5 µm instead of 8).  Skipping the slider here also
        leaves Manual Z alone (it mirrors the slider via valueChanged, not
        this method), which matters just as much: it is what a keyboard move
        actually sends.  The Manual Z field is separately skipped while IT
        has the keyboard focus (typing the next target during a followed
        move).
        """
        if math.isnan(z):
            self.lbl_z_measured.setText("Measured Z: unknown (could not be read)")
            return
        self.lbl_z_measured.setText(f"Measured Z: {z:.3f} µm")
        if self.sl_z.isSliderDown() or self.sl_z.hasFocus():
            return  # the user is choosing a target on the slider right now
        self._z_sync_guard = True
        try:
            self.sl_z.blockSignals(True)
            self.sl_z.setValue(round(z * _Z_SLIDER_SCALE))
            self.sl_z.blockSignals(False)
            if not self.sb_manual_z.hasFocus():
                self.sb_manual_z.blockSignals(True)
                self.sb_manual_z.setValue(z)
                self.sb_manual_z.blockSignals(False)
        finally:
            self._z_sync_guard = False

    @Slot()
    def _apply_camera_params(self):
        if not self._hardware_connected:
            self._append_log("[WARN] Hardware not connected.")
            return

        # The timeout is not a camera-hardware setting (nothing to send to
        # the SDK): it only needs to reach the service's config, which
        # preview_start()/capture_preview() re-read on every call. Without
        # this it silently kept the value from connect (or the last sweep)
        # until the next sweep rebuilt the whole config (TODO.md item 1, U5).
        self._vm.apply_config({"exposure": self._exposure_s(), "timeout": self.sb_timeout.value()})
        if self._vm.connect_camera():
            # Exposure applied by the camera thread between frames; it logs
            # the confirmation ("Camera exposure updated to ...") when done.
            # The timeout takes effect immediately (no camera call needed).
            self._append_log(
                f"Applying camera parameters (exposure={self.sb_exposure.value():.3f} ms, "
                f"timeout={self.sb_timeout.value():.2f} s)…"
            )
        else:
            self._append_log("[ERROR] Failed to apply camera parameters.")

    @Slot(float)
    def _on_start_changed(self, value: float) -> None:
        """Ensure 'End' is always >= 'Start'."""
        self.sb_end.setMinimum(value)

    @Slot(float)
    def _on_end_changed(self, value: float) -> None:
        """Ensure 'Start' is always <= 'End'."""
        self.sb_start.setMaximum(value)

    @Slot(str)
    def _on_sweep_aborted(self, summary: str) -> None:
        """Remember why the sweep stopped early; _on_finished (next) reports it."""
        self._sweep_abort_summary = summary

    @Slot(str, int, int)
    def _on_finished(self, output_folder: str, skipped: int, total: int) -> None:
        """Report sweep completion (and warn about incomplete datasets)."""
        aborted = self._sweep_abort_summary
        self._sweep_abort_summary = ""
        if aborted:
            # Modal on purpose: the sweep ended without the dataset the user
            # asked for, and nobody may analyse the folder as if it were whole.
            msg = (
                f"{aborted}\n\nThe folder is marked with SWEEP_ABORTED.txt. "
                "Check the camera (cable, power) and the free disk space before starting another sweep."
            )
            self._append_log(f"[WARN] {aborted}")
            QMessageBox.warning(self, "Sweep aborted", msg)
            return
        self._append_log(f"Sweep finished. Images saved to: {output_folder}")
        if skipped > 0:
            acquired = total - skipped
            msg = (
                f"Incomplete dataset: {acquired}/{total} frames acquired "
                f"({skipped} skipped). Z sampling is not uniform; "
                f"review the log before analyzing this dataset."
            )
            self._append_log(f"[WARN] {msg}")
            QMessageBox.warning(self, "Incomplete dataset", msg)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Errors that need the user's attention: a failed sweep or a failed
        piezo move (the stage may not be where the window says).  Live
        preview and exposure problems never come here: they are shown in
        lbl_camera_status (see _on_preview_notice).

        The live preview state is NOT touched here: a piezo error must not
        stop the camera stream (finding C7b).
        """
        message = message.strip() or "Unknown error (no details were given)."
        if self._auto_park_pending:
            # The automatic post-connect move failed (item 4/8 of TODO.md):
            # non-blocking by design, a log line is enough -- the user just
            # connected and did not ask for this move themselves.
            self._auto_park_pending = False
            self._append_log(
                f"[WARN] Could not park the piezo at {PARK_POSITION_UM:.1f} µm "
                f"after connecting: {message}"
            )
            self._update_button_states()
            return
        self._append_log(f"[ERROR] {message}")
        self._update_button_states()
        QMessageBox.critical(self, "Acquisition error", message)

    # ============================================================
    #  Progress (elapsed/ETA go straight to the ProgressInfoLabel)
    # ============================================================
    @Slot(float)
    def _on_progress(self, percent):
        self.progress.setValue(int(percent))

    @Slot(bool)
    def _on_running(self, running):
        # The camera thread pauses and resumes the live preview around the
        # sweep by itself; only the button matrix changes here.
        if running:
            # Defensive backstop for H2 (batch 5 review): a sweep cannot
            # legitimately start while the park move is still in progress
            # (start() refuses it), but if it ever did, its own failure must
            # not be swallowed as "the park's".
            self._auto_park_pending = False
        self._update_button_states()

    @Slot(bool)
    def _on_moving(self, moving: bool) -> None:
        """Update button states whenever a manual move starts or finishes."""
        if not moving:
            # H2 (batch 5 review): the park move is over. A failure already
            # cleared the flag itself in _on_error; a SUCCESSFUL park never
            # went through _on_error, so without this the flag would still
            # be set and swallow the next, unrelated error (e.g. a failed
            # sweep) as if it were the park's, with no modal.
            self._auto_park_pending = False
        self._update_button_states()

    # ============================================================
    #  Preview frames
    # ============================================================
    @Slot(object, float)
    def _on_preview(self, frame, z):
        arr = np.asarray(frame)

        try:
            pix = self._vm.numpy_to_pixmap(arr)
        except ValueError:
            return
        if pix.isNull():
            return

        pix = pix.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)
        self._update_histogram(arr)

    # ============================================================
    #  Public helpers (keyboard shortcuts in MainWindow)
    # ============================================================
    def trigger_start(self) -> None:
        """Start the sweep if the button state matrix currently allows it."""
        if self.btn_start.isEnabled():
            self.btn_start.click()

    def trigger_cancel(self) -> None:
        """Cancel the sweep if the button state matrix currently allows it."""
        if self.btn_cancel.isEnabled():
            self.btn_cancel.click()

    # ============================================================
    #  Persistent configuration
    # ============================================================
    def get_config(self) -> dict:
        """Return current UI settings as a serialisable dict."""
        return {
            "piezo_serial": self.le_piezo_serial.text(),
            "start": self.sb_start.value(),
            "end": self.sb_end.value(),
            "step": self.sb_step.value(),
            "manual_z": self.sb_manual_z.value(),
            "keyboard_step_um": self.sb_keyboard_step_nm.value() / 1000.0,
            "exposure": self.sb_exposure.value(),
            "timeout": self.sb_timeout.value(),
            "format": self.cmb_format.currentText(),
            "color_mode": self._current_channel_mode(),
        }

    def apply_saved_config(self, cfg: dict) -> None:
        """Apply a previously saved config dict to the UI controls."""
        if "piezo_serial" in cfg:
            self.le_piezo_serial.setText(str(cfg["piezo_serial"]))
        if "start" in cfg:
            self.sb_start.setValue(float(cfg["start"]))
        if "end" in cfg:
            self.sb_end.setValue(float(cfg["end"]))
        if "step" in cfg:
            self.sb_step.setValue(float(cfg["step"]))
        if "manual_z" in cfg:
            self.sb_manual_z.setValue(float(cfg["manual_z"]))
        if "keyboard_step_um" in cfg:
            self.sb_keyboard_step_nm.setValue(float(cfg["keyboard_step_um"]) * 1000.0)
        if "exposure" in cfg:
            self.sb_exposure.setValue(float(cfg["exposure"]))
        if "timeout" in cfg:
            self.sb_timeout.setValue(float(cfg["timeout"]))
        if "format" in cfg:
            idx = self.cmb_format.findText(str(cfg["format"]))
            if idx >= 0:
                self.cmb_format.setCurrentIndex(idx)
        if "color_mode" in cfg:
            mode = str(cfg["color_mode"])
            self.cmb_channels.setCurrentIndex(
                CHANNEL_MODES.index(mode) if mode in CHANNEL_MODES else 0
            )
        # "preview_interval" (older configs) is accepted and ignored: the
        # live preview no longer has a timer.

    # ============================================================
    #  Close
    # ============================================================
    def closeEvent(self, event):
        self._preview_active = False
        try:
            # Cancel any sweep, stop worker threads and disconnect hardware.
            self._vm.shutdown()
            if self._hardware_connected:
                self._append_log("Hardware disconnected on close.")
        except Exception:
            pass
        self._hardware_connected = False

        super().closeEvent(event)

    def _append_log(self, msg: str) -> None:
        """Append a log line to the GUI log box and to the session log file."""
        self.log_box.append(msg)
        self.log_box.ensureCursorVisible()
        append_to_log_file(f"[ACQ] {msg}")
