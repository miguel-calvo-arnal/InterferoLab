# views/AcquisitionPanel.py
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QCoreApplication, QEventLoop, Qt, QTimer, Slot
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
    QSpinBox,
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


class AcquisitionPanel(QWidget):
    """
    Acquisition panel with:
            - Full camera/piezo control
            - Manual movement
            - Sweep + live preview (before/during/after)
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
        self._preview_active = False

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
        self.sb_manual_z.setToolTip("Target position for a manual piezo move, in µm (0–100).")
        servo_layout.addWidget(self.sb_manual_z, r, 1)
        r += 1

        # Piezo position slider: control + live indicator of the real position
        self.sl_z = QSlider(Qt.Horizontal)
        self.sl_z.setRange(0, 100 * _Z_SLIDER_SCALE)  # 0–100 µm, 0.01 µm ticks
        self.sl_z.setSingleStep(10)  # 0.1 µm per arrow key
        self.sl_z.setPageStep(_Z_SLIDER_SCALE)  # 1 µm per groove click
        self.sl_z.setToolTip(
            "Piezo Z position, 0–100 µm (0.01 µm resolution).\n"
            "Drag to pick a target (Manual Z follows live); the piezo moves\n"
            "when the handle is released. Tracks the real position during\n"
            "moves, previews and sweeps (read-only while busy)."
        )
        servo_layout.addWidget(self.sl_z, r, 0, 1, 2)
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
        r += 1

        out_layout.addWidget(QLabel("Preview interval (ms):"), r, 0)
        self.sb_preview_int = QSpinBox()
        self.sb_preview_int.setRange(200, 10000)
        self.sb_preview_int.setValue(1000)
        self.sb_preview_int.setToolTip("Time between live preview captures, in ms (200–10000).")
        out_layout.addWidget(self.sb_preview_int, r, 1)
        r += 1

        self.btn_apply_preview = QPushButton("Apply preview interval")
        self.btn_apply_preview.setIcon(icon("check.svg"))
        self.btn_apply_preview.setToolTip("Apply the interval to the live preview timer.")
        out_layout.addWidget(self.btn_apply_preview, r, 0, 1, 2)

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
        self.btn_preview.setToolTip("Start or stop the live camera preview.")

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
        self.btn_apply_preview.clicked.connect(self._apply_preview_interval)
        self.cmb_channels.currentIndexChanged.connect(self._on_channel_mode_changed)

        # Piezo slider <-> spinbox <-> real position (loop-free: the guard
        # flag plus blockSignals in _on_position_changed break every cycle).
        self._z_sync_guard = False
        self.sl_z.valueChanged.connect(self._on_slider_value_changed)
        self.sl_z.sliderReleased.connect(self._on_slider_released)
        self._vm.positionChanged.connect(self._on_position_changed)

        self._vm.progressChanged.connect(self._on_progress)
        self._vm.etaChanged.connect(self._progress_info.set_eta)
        self._vm.elapsedChanged.connect(self._progress_info.set_elapsed)
        self._vm.statusTextChanged.connect(self._append_log)
        self._vm.previewFrame.connect(self._on_preview)
        self._vm.runningChanged.connect(self._on_running)
        self._vm.movingChanged.connect(self._on_moving)
        self._vm.finished.connect(self._on_finished)
        self._vm.error.connect(self._on_error)

        # Idle preview timer (only fires when preview_active and not running)
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(self.sb_preview_int.value())
        self._live_timer.stop()
        self._live_timer.timeout.connect(self._request_live_preview)

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
            self.btn_apply_preview.setEnabled(False)
            return

        # Hardware connected: lock the serial field to prevent accidental changes
        self.le_piezo_serial.setEnabled(False)
        self.btn_connect.setText("Disconnect hardware")
        self.btn_connect.setIcon(self._icon_plug_off)
        self.btn_connect.setEnabled(not busy)

        self.btn_start.setEnabled(not busy)
        self.btn_cancel.setEnabled(running)  # cancel only meaningful during sweep
        self.btn_move_piezo.setEnabled(not busy)
        self.btn_apply_cam.setEnabled(not busy)
        self.btn_apply_preview.setEnabled(not busy)

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

        - If already connected and no sweep is running: disconnect everything.
        - If not connected: apply current UI parameters and connect camera
        (and piezo when the bypass is removed).
        """
        # If already connected and not running -> disconnect
        if self._hardware_connected and not self._vm.is_running():
            self._vm.disconnect_all()
            self._hardware_connected = False
            self._preview_active = False
            self._live_timer.stop()
            self._append_log("Hardware disconnected.")
            self._update_button_states()
            return

        # If connected but running, do nothing (button should be disabled anyway)
        if self._hardware_connected and self._vm.is_running():
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

        # 2) Connect hardware using that configuration
        dll_path = resource_path(os.path.join("API", "PI", "E816_DLL_x64.dll"))
        serial = self.le_piezo_serial.text().strip() or "125056199"
        ok1 = self._vm.connect_piezo(serial, dll_path)
        ok2 = self._vm.connect_camera()

        self._hardware_connected = bool(ok1 and ok2)

        if self._hardware_connected:
            # Preview always starts OFF after connecting; the user enables it
            # explicitly with the Start preview button.
            self._preview_active = False
            self._live_timer.stop()
            # Request one picture to show and avoid start_acquisition() to wait forever and
            # freeze the GUI
            self._vm.request_preview()

            self._append_log(
                f"Hardware connected successfully. Parameters applied "
                f"(exposure={self.sb_exposure.value():.3f} ms, "
                f"timeout={self.sb_timeout.value():.2f} s)."
            )
        else:
            self._preview_active = False
            self._live_timer.stop()
            self._vm.disconnect_all()
            self._append_log("[ERROR] Could not connect hardware. See log for details.")
            QMessageBox.critical(
                self,
                "Hardware connection failed",
                "Could not connect to piezo or camera.\n\n"
                "Check that the hardware is powered on, the serial number is correct, "
                "and no other application is using the devices.",
            )

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
            interval = self.sb_preview_int.value()
            self._live_timer.setInterval(interval)
            self._live_timer.start()
            self._append_log(f"Preview started (interval={interval} ms).")
        else:
            self._live_timer.stop()
            self._append_log("Preview stopped.")

        self._update_button_states()

    @Slot(int)
    def _on_channel_mode_changed(self, index: int) -> None:
        mode = CHANNEL_MODES[index] if 0 <= index < len(CHANNEL_MODES) else "mono"
        self._vm.apply_config({"color_mode": mode})
        self._set_histogram_mode("color" if mode == "color" else "mono")

    def _wait_preview_idle(self, extra_timeout_s: float = 2.0) -> bool:
        """
        Wait for any in-flight preview capture to finish before a sweep.

        Spins the event loop (bounded by the camera timeout plus a margin)
        so the preview thread can deliver its finished signal. Returns True
        when no preview capture is in flight.
        """
        if not self._vm.is_previewing():
            return True

        self._append_log("Waiting for in-flight preview capture to finish…")
        deadline = time.monotonic() + self.sb_timeout.value() + extra_timeout_s
        while self._vm.is_previewing() and time.monotonic() < deadline:
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
            time.sleep(0.02)
        return not self._vm.is_previewing()

    @Slot()
    def _start(self):
        # Stop the live preview BEFORE starting the sweep so the preview
        # thread and the sweep thread can never use the camera concurrently.
        self._preview_active = False
        self._live_timer.stop()
        if not self._wait_preview_idle():
            self._append_log("[ERROR] Preview capture did not finish in time; sweep not started.")
            self._update_button_states()
            return

        cfg = {
            "closed_loop": True,
            "start": self.sb_start.value(),
            "end": self.sb_end.value(),
            "step": self.sb_step.value(),
            "exposure": self._exposure_s(),
            "timeout": self.sb_timeout.value(),
            "format": self.cmb_format.currentText(),
            "color_mode": self._current_channel_mode(),
            "preview_interval": self.sb_preview_int.value(),
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
        """Reflect the real piezo position on the slider and the spinbox.

        Fed by manual-move completion, sweep progress and preview frames.
        Widget signals are blocked so an indicator update can never trigger
        a move; skipped while the user is dragging the handle.
        """
        if self.sl_z.isSliderDown():
            return
        self._z_sync_guard = True
        try:
            self.sl_z.blockSignals(True)
            self.sl_z.setValue(round(z * _Z_SLIDER_SCALE))
            self.sl_z.blockSignals(False)
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

        self._vm.apply_config({"exposure": self._exposure_s()})
        if self._vm.connect_camera():
            self._append_log(
                f"Camera parameters applied (exposure={self.sb_exposure.value():.3f} ms)."
            )
        else:
            self._append_log("[ERROR] Failed to apply camera parameters.")

    @Slot()
    def _apply_preview_interval(self):
        interval = self.sb_preview_int.value()
        self._live_timer.setInterval(interval)
        self._append_log(f"Preview interval set to {interval} ms.")
        # If preview active, timer is already running.

    @Slot(float)
    def _on_start_changed(self, value: float) -> None:
        """Ensure 'End' is always >= 'Start'."""
        self.sb_end.setMinimum(value)

    @Slot(float)
    def _on_end_changed(self, value: float) -> None:
        """Ensure 'Start' is always <= 'End'."""
        self.sb_start.setMaximum(value)

    @Slot(str, int, int)
    def _on_finished(self, output_folder: str, skipped: int, total: int) -> None:
        """Report sweep completion (and warn about incomplete datasets)."""
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
        self._append_log(f"[ERROR] {message}")
        self._preview_active = False
        self._live_timer.stop()
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
        if running:
            self._live_timer.stop()
        else:
            if self._hardware_connected and self._preview_active:
                self._live_timer.start()
        self._update_button_states()

    @Slot(bool)
    def _on_moving(self, moving: bool) -> None:
        """Update button states whenever a manual move starts or finishes."""
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
    #  Timer-driven request
    # ============================================================
    @Slot()
    def _request_live_preview(self):
        if self._hardware_connected and self._preview_active and not self._vm.is_running():
            self._vm.request_preview()

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
            "exposure": self.sb_exposure.value(),
            "timeout": self.sb_timeout.value(),
            "format": self.cmb_format.currentText(),
            "color_mode": self._current_channel_mode(),
            "preview_interval": self.sb_preview_int.value(),
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
        if "preview_interval" in cfg:
            self.sb_preview_int.setValue(int(cfg["preview_interval"]))

    # ============================================================
    #  Close
    # ============================================================
    def closeEvent(self, event):
        self._preview_active = False
        self._live_timer.stop()
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
