# views/ProcessingPanel.py
from __future__ import annotations

import os

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.analysis_service import load_reconstruction_methods
from services.image_formats import detect_dataset_format
from utils.camera_constants import GRAY_W_B, GRAY_W_G, GRAY_W_R
from utils.icons import append_to_log_file, icon
from viewmodels.analysis_vm import AnalysisVM
from widgets import theme
from widgets.progress_info import ProgressInfoLabel


class ProcessingPanel(QWidget):
    """
    Processing panel for launching the C++ analysis pipeline.

    Responsibilities
    ----------------
    - Lists experiment folders under ./data/ as analysis inputs.
    - Allows the user to start the analysis via AnalysisVM.
    - Displays progress (%, stage, counters, ETA).
    - Streams real-time logs.
    - Offers 'Cancel' support (cooperative cancellation of the C++ backend).

    Signals
    -------
    analysisFinishedAndAccepted(result_dict)
            Emitted when analysis completes and the user chooses to load
            the resulting heightmap into the Results tab.
    """

    analysisFinishedAndAccepted = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._vm = AnalysisVM(self)
        # Whether channel weights make sense for the selected dataset (RGB vs mono)
        self._ch_weights_applicable = True

        main_layout = QVBoxLayout(self)

        # -----------------------------------------------------
        #  Dataset selection & (future) YAML config
        # -----------------------------------------------------
        cfg_box = QGroupBox("Processing configuration", self)
        cfg_layout = QFormLayout(cfg_box)

        # Dataset selector (subfolders under ./data)
        self._datasets_combo = QComboBox(self)
        self._datasets_combo.setToolTip("Acquired dataset to analyze (subfolder of ./data).")
        self._reload_btn = QPushButton("Refresh datasets", self)
        self._reload_btn.setIcon(icon("refresh.svg"))
        self._reload_btn.setToolTip("Rescan ./data for dataset folders (F5).")
        self._reload_btn.clicked.connect(self._reload_datasets)

        ds_row = QHBoxLayout()
        ds_row.addWidget(self._datasets_combo)
        ds_row.addWidget(self._reload_btn)

        cfg_layout.addRow("Dataset (folder in ./data):", ds_row)

        # Reconstruction method selector — populated from the C++ backend at runtime
        self._method_combo = QComboBox(self)
        _methods = load_reconstruction_methods()
        for m in _methods:
            self._method_combo.addItem(m["name"], m["id"])
        # Default to Method 1 (squared-derivative centroid): robust, calibration-free,
        # best general-purpose choice for high-SNR acquisitions with broadband sources.
        for i in range(self._method_combo.count()):
            if self._method_combo.itemData(i) == 1:
                self._method_combo.setCurrentIndex(i)
                break
        tooltip_lines = [f"  {m['id']}: {m['description']}" for m in _methods]
        self._method_combo.setToolTip("Reconstruction algorithm:\n" + "\n".join(tooltip_lines))
        cfg_layout.addRow("Reconstruction method:", self._method_combo)

        # Pixel plots option
        self._pixel_plots_cb = QCheckBox("Save 40×40 grid of pixel plots", self)
        self._pixel_plots_cb.setChecked(False)
        self._pixel_plots_cb.setToolTip(
            "When enabled, saves 1600 evenly-spaced pixel plots (40×40 grid) plus the\n"
            "median pixel plot. When disabled, only the median pixel plot is saved."
        )
        cfg_layout.addRow("Pixel plots:", self._pixel_plots_cb)

        main_layout.addWidget(cfg_box)

        # -----------------------------------------------------
        #  Color channel weights (only for RGB acquisitions)
        # -----------------------------------------------------
        self._ch_box = QGroupBox("Color channel weights (RGB images only)", self)
        ch_layout = QHBoxLayout(self._ch_box)

        # Default weights come from utils.camera_constants (single source
        # of truth shared with the acquisition code and the C++ backend).
        self._cb_r, self._sb_wr = self._make_channel_row(ch_layout, "R", GRAY_W_R)
        ch_layout.addSpacing(16)
        self._cb_g, self._sb_wg = self._make_channel_row(ch_layout, "G", GRAY_W_G)
        ch_layout.addSpacing(16)
        self._cb_b, self._sb_wb = self._make_channel_row(ch_layout, "B", GRAY_W_B)
        ch_layout.addSpacing(24)

        self._weights_label = QLabel("", self)
        ch_layout.addWidget(self._weights_label, stretch=1)

        main_layout.addWidget(self._ch_box)

        # Wire up weight label updates
        for cb in (self._cb_r, self._cb_g, self._cb_b):
            cb.toggled.connect(self._update_weights_label)
        for sb in (self._sb_wr, self._sb_wg, self._sb_wb):
            sb.valueChanged.connect(self._update_weights_label)

        # Keep spinboxes in sync with their checkboxes
        self._cb_r.toggled.connect(self._sb_wr.setEnabled)
        self._cb_g.toggled.connect(self._sb_wg.setEnabled)
        self._cb_b.toggled.connect(self._sb_wb.setEnabled)

        # Sync initial enabled state (setChecked was called before connections were wired)
        self._sb_wg.setEnabled(self._cb_g.isChecked())
        self._sb_wb.setEnabled(self._cb_b.isChecked())

        # -----------------------------------------------------
        #  Control buttons
        # -----------------------------------------------------
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("Start analysis", self)
        theme.set_variant(self._start_btn, "primary")
        self._start_btn.setIcon(icon("play.svg", "#ffffff"))
        self._start_btn.setToolTip(
            "Run the height reconstruction on the selected dataset (Ctrl+Return)."
        )
        self._cancel_btn = QPushButton("Cancel", self)
        theme.set_variant(self._cancel_btn, "danger")
        self._cancel_btn.setIcon(icon("stop.svg", "#ffffff"))
        self._cancel_btn.setToolTip("Cancel the running analysis (Esc).")
        self._cancel_btn.setEnabled(False)

        btn_layout.addWidget(self._start_btn)
        btn_layout.addWidget(self._cancel_btn)
        main_layout.addLayout(btn_layout)

        # -----------------------------------------------------
        #  Progress & textual status
        # -----------------------------------------------------
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        main_layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Status: idle", self)
        main_layout.addWidget(self._status_label)

        self._progress_info = ProgressInfoLabel(self)
        main_layout.addWidget(self._progress_info)

        # -----------------------------------------------------
        #  Live logs
        # -----------------------------------------------------
        self._logs_edit = QTextEdit(self)
        self._logs_edit.setObjectName("logView")  # monospace log styling in styles.qss
        self._logs_edit.setReadOnly(True)
        main_layout.addWidget(self._logs_edit, stretch=1)

        # -----------------------------------------------------
        #  VM -> UI connections
        # -----------------------------------------------------
        self._vm.progressChanged.connect(self._on_progress_changed)
        self._vm.logReceived.connect(self._on_log_received)
        self._vm.runningChanged.connect(self._on_running_changed)
        self._vm.finished.connect(self._on_finished)
        self._vm.error.connect(self._on_error)
        self._vm.elapsedChanged.connect(self._progress_info.set_elapsed)
        self._vm.etaChanged.connect(self._progress_info.set_eta)

        # -----------------------------------------------------
        #  UI -> VM connections
        # -----------------------------------------------------
        self._start_btn.clicked.connect(self._on_start_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._datasets_combo.currentTextChanged.connect(self._on_dataset_changed)

        # Initial dataset population (fires currentTextChanged → _on_dataset_changed)
        self._reload_datasets()

        # Sync label with actual initial checkbox/spinbox state
        self._update_weights_label()

    # =========================================================
    # Format detection
    # =========================================================
    @Slot(str)
    def _on_dataset_changed(self, dataset_name: str) -> None:
        """
        Update the channel-weights group box whenever the selected dataset changes.
        Mono datasets: weights are irrelevant — disable the box and say so.
        Color datasets: enable the box for normal use.
        Unknown (empty folder, unreadable): enable so the user can set weights manually.
        """
        if not dataset_name:
            return
        folder = self._current_dataset_folder()
        if folder is None or not os.path.isdir(folder):
            return
        fmt = detect_dataset_format(folder)
        if fmt == "mono":
            self._ch_box.setTitle(
                "Color channel weights — not applicable (mono/superpixel dataset)"
            )
            self._ch_weights_applicable = False
        else:
            self._ch_box.setTitle("Color channel weights (RGB images only)")
            self._ch_weights_applicable = True
        self._ch_box.setEnabled(self._ch_weights_applicable and not self._vm.is_running())

    # =========================================================
    # Helpers
    # =========================================================
    def _make_channel_row(
        self, layout: QHBoxLayout, label: str, default_weight: float
    ) -> tuple[QCheckBox, QDoubleSpinBox]:
        """Create and lay out a checkbox + weight spinbox pair for one colour channel."""
        channel = {"R": "red", "G": "green", "B": "blue"}.get(label, label)
        cb = QCheckBox(label, self)
        cb.setChecked(True)
        cb.setToolTip(
            f"Include the {channel} channel in the gray-scale merge.\n"
            "Uncheck to exclude a corrupted channel."
        )
        sb = QDoubleSpinBox(self)
        sb.setRange(0.0, 100.0)
        sb.setDecimals(6)
        sb.setSingleStep(0.05)
        sb.setValue(default_weight)
        sb.setToolTip(
            f"Relative weight of the {channel} channel (0–100).\n"
            "Weights are normalized to sum 1 before use."
        )
        layout.addWidget(cb)
        layout.addWidget(sb)
        return cb, sb

    def _data_root(self) -> str:
        """Return the absolute path to the ./data folder."""
        return os.path.join(os.getcwd(), "data")

    def _get_channel_weights(self) -> list[float]:
        """Return normalized [wr, wg, wb] based on the current UI state."""
        raw = [
            self._sb_wr.value() if self._cb_r.isChecked() else 0.0,
            self._sb_wg.value() if self._cb_g.isChecked() else 0.0,
            self._sb_wb.value() if self._cb_b.isChecked() else 0.0,
        ]
        total = sum(raw)
        if total <= 0.0:
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        return [w / total for w in raw]

    def _update_weights_label(self) -> None:
        """Refresh the normalized-weights display."""
        wr, wg, wb = self._get_channel_weights()
        self._weights_label.setText(f"Effective:  R = {wr:.3f}   G = {wg:.3f}   B = {wb:.3f}")

    @Slot()
    def _reload_datasets(self) -> None:
        """
        Scan ./data/ for subfolders and populate the dataset selector.
        """
        root = self._data_root()
        self._datasets_combo.clear()

        try:
            if not os.path.isdir(root):
                self._append_log(f"[WARN] Data folder does not exist: {root}")
                return

            entries = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

            if not entries:
                self._append_log(f"[INFO] No subfolders found in {root}")
            else:
                self._datasets_combo.addItems(entries)

        except Exception as e:  # noqa: BLE001
            self._append_log(f"[ERROR] Error listing datasets: {e}")

    def _current_dataset_folder(self) -> str | None:
        """
        Return the full path to data/<selected_dataset>, or None if nothing is selected.
        """
        dataset_name = self._datasets_combo.currentText().strip()
        if not dataset_name:
            return None
        return os.path.join(self._data_root(), dataset_name)

    # =========================================================
    # UI Slots
    # =========================================================
    @Slot()
    def _on_start_clicked(self) -> None:
        """Start the analysis for the currently selected dataset."""
        folder = self._current_dataset_folder()
        if not folder:
            QMessageBox.warning(
                self,
                "No dataset selected",
                "Please select a dataset under ./data first.",
            )
            return

        name = os.path.basename(folder)
        channel_weights = self._get_channel_weights()
        method: int = self._method_combo.currentData()
        pixel_plot_grid = 40 if self._pixel_plots_cb.isChecked() else 0

        if not self._vm.start(folder, name, channel_weights, method, pixel_plot_grid):
            QMessageBox.information(
                self,
                "Analysis already running",
                "An analysis job is currently in progress.",
            )
            return

        wr, wg, wb = channel_weights
        method_name = self._method_combo.currentText()
        grid_info = (
            f"  |  pixel grid: {pixel_plot_grid}×{pixel_plot_grid}"
            if pixel_plot_grid
            else "  |  pixel plots: median only"
        )
        self._append_log(
            f"[INFO] Launching analysis for: {folder}  "
            f"(weights R={wr:.3f} G={wg:.3f} B={wb:.3f}  |  {method_name}{grid_info})"
        )
        self._status_label.setText(f"Status: analyzing '{name}'…")

    @Slot()
    def _on_cancel_clicked(self) -> None:
        """Request cancellation from the ViewModel."""
        self._vm.cancel()

    # =========================================================
    # VM -> UI Slots
    # =========================================================
    @Slot(int, str, int, int, float)
    def _on_progress_changed(
        self,
        percent: int,
        stage: str,
        done: int,
        total: int,
        eta_s: float,
    ) -> None:
        """Update the progress bar and the textual status line."""
        self._progress_bar.setValue(percent)

        self._status_label.setText(f"Status: {stage} — {percent}% ({done}/{total})")

    @Slot(str, str)
    def _on_log_received(self, level: str, message: str) -> None:
        """Append a formatted log line."""
        self._append_log(f"[{level.upper()}] {message}")

    @Slot(bool)
    def _on_running_changed(self, running: bool) -> None:
        """Enable or disable controls depending on running state."""
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        # Freeze the analysis inputs while running: parameters are captured at
        # start, so mid-run edits would only mislead.
        self._datasets_combo.setEnabled(not running)
        self._reload_btn.setEnabled(not running)
        self._method_combo.setEnabled(not running)
        self._pixel_plots_cb.setEnabled(not running)
        self._ch_box.setEnabled(self._ch_weights_applicable and not running)
        if not running:
            self._status_label.setText("Status: idle")

    @Slot(dict)
    def _on_finished(self, result: dict) -> None:
        """
        Called when the C++ backend signals completion.

        If the result contains no output folder, the analysis was cancelled
        before producing any output (C++ returns empty paths on cancellation).
        """
        if not result.get("output_folder"):
            self._append_log("[INFO] Analysis cancelled.")
            return

        self._append_log("[INFO] Analysis completed.")
        self._append_log(f"[INFO] Output folder: {result.get('output_folder')}")
        self._append_log(f"[INFO] Heightmap:     {result.get('heightmap')}")

        reply = QMessageBox.question(
            self,
            "Analysis finished",
            f"The analysis completed successfully.\n\n"
            f"Load this heightmap in the Results tab?\n\n"
            f"{result.get('heightmap', '')}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            self.analysisFinishedAndAccepted.emit(result)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Display an analysis error and log it."""
        self._append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Analysis error", message)

    # =========================================================
    # Public helpers (keyboard shortcuts in MainWindow)
    # =========================================================
    def refresh_datasets(self) -> None:
        """Rescan ./data and repopulate the dataset selector."""
        self._reload_datasets()

    def trigger_start(self) -> None:
        """Start the analysis if the controls currently allow it."""
        if self._start_btn.isEnabled():
            self._start_btn.click()

    def trigger_cancel(self) -> None:
        """Cancel the analysis if the controls currently allow it."""
        if self._cancel_btn.isEnabled():
            self._cancel_btn.click()

    def is_running(self) -> bool:
        """Whether an analysis job is currently active (used by MainWindow)."""
        return self._vm.is_running()

    def closeEvent(self, event) -> None:
        # Stop the analysis worker thread before the widget tree is torn
        # down: a QThread destroyed while still running aborts the process.
        self._vm.shutdown()
        super().closeEvent(event)

    # =========================================================
    # Logging helper
    # =========================================================
    def _append_log(self, text: str) -> None:
        """Append a log line to the QTextEdit and to the session log file."""
        self._logs_edit.append(text)
        self._logs_edit.ensureCursorVisible()
        append_to_log_file(f"[PROC] {text}")

    # =========================================================
    # Persistent configuration
    # =========================================================
    def get_config(self) -> dict:
        """Return current UI settings as a serialisable dict."""
        return {
            "method_id": self._method_combo.currentData(),
            "cb_r": self._cb_r.isChecked(),
            "cb_g": self._cb_g.isChecked(),
            "cb_b": self._cb_b.isChecked(),
            "wr": self._sb_wr.value(),
            "wg": self._sb_wg.value(),
            "wb": self._sb_wb.value(),
            "pixel_plots": self._pixel_plots_cb.isChecked(),
        }

    def apply_saved_config(self, cfg: dict) -> None:
        """Apply a previously saved config dict to the UI controls."""
        if "method_id" in cfg:
            # Restore by stable method ID, not by combo position
            target_id = int(cfg["method_id"])
            for i in range(self._method_combo.count()):
                if self._method_combo.itemData(i) == target_id:
                    self._method_combo.setCurrentIndex(i)
                    break
        elif "method_index" in cfg:
            # Legacy fallback: old configs stored a raw combo index
            idx = int(cfg["method_index"])
            if 0 <= idx < self._method_combo.count():
                self._method_combo.setCurrentIndex(idx)
        if "cb_r" in cfg:
            self._cb_r.setChecked(bool(cfg["cb_r"]))
        if "cb_g" in cfg:
            self._cb_g.setChecked(bool(cfg["cb_g"]))
        if "cb_b" in cfg:
            self._cb_b.setChecked(bool(cfg["cb_b"]))
        if "wr" in cfg:
            self._sb_wr.setValue(float(cfg["wr"]))
        if "wg" in cfg:
            self._sb_wg.setValue(float(cfg["wg"]))
        if "wb" in cfg:
            self._sb_wb.setValue(float(cfg["wb"]))
        if "pixel_plots" in cfg:
            self._pixel_plots_cb.setChecked(bool(cfg["pixel_plots"]))
