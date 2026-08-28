# views/ResultsPanel.py
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.heightmap_processing import fit_and_subtract_plane
from utils.icons import icon
from viewmodels.results_vm import ResultsVM
from widgets import theme
from widgets.HeightmapView import HeightmapView


class ResultsPanel(QWidget):
    """
    Results panel:
      - Lists datasets in ./output/
      - Loads heightmap via ResultsVM
      - Loads representative pixels
      - Shows/hides markers
      - Shows hover information and Δz between two points (measure)
    """

    # Formatted hover readout, re-emitted for the MainWindow status bar
    hoverInfoChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._vm = ResultsVM(self)
        self._representative_points = []  # list[(x, y, npy_path)]
        self._markers_visible = False  # markers hidden by default

        # Original (un-corrected) heightmap kept for idempotent re-application
        self._original_arr: np.ndarray | None = None
        self._original_path: str | None = None

        main_layout = QVBoxLayout(self)

        # ----------------------------------------------------------
        #  Top bar: dataset selector + buttons
        # ----------------------------------------------------------
        top_bar = QHBoxLayout()
        main_layout.addLayout(top_bar)

        top_bar.addWidget(QLabel("Dataset:", self))

        self._datasets_combo = QComboBox(self)
        self._datasets_combo.setToolTip("Analyzed dataset to display (subfolder of ./output).")
        top_bar.addWidget(self._datasets_combo, stretch=1)

        self._refresh_btn = QPushButton("Refresh datasets", self)
        self._refresh_btn.setIcon(icon("refresh.svg"))
        self._refresh_btn.setToolTip("Rescan ./output for analyzed datasets (F5).")
        top_bar.addWidget(self._refresh_btn)

        self._load_btn = QPushButton("Load heightmap", self)
        theme.set_variant(self._load_btn, "primary")
        self._load_btn.setIcon(icon("folder-open.svg", "#ffffff"))
        self._load_btn.setToolTip(
            "Load the heightmap and representative pixels of the selected dataset."
        )
        top_bar.addWidget(self._load_btn)

        self._toggle_markers_btn = QPushButton("Show markers", self)
        self._toggle_markers_btn.setIcon(icon("crosshair.svg"))
        self._toggle_markers_btn.setToolTip(
            "Show or hide the representative-pixel markers on the heightmap."
        )
        top_bar.addWidget(self._toggle_markers_btn)

        self._tilt_btn = QPushButton("Tilt correction: on", self)
        self._tilt_btn.setCheckable(True)
        self._tilt_btn.setChecked(True)
        self._tilt_btn.setIcon(icon("level.svg", "#ffffff"))
        self._tilt_btn.setToolTip(
            "Toggle linear-plane tilt correction of the displayed heightmap.\n"
            "Green = applied; red = not applied."
        )
        self._tilt_btn.setEnabled(False)
        theme.set_variant(self._tilt_btn, "success")
        top_bar.addWidget(self._tilt_btn)

        # ----------------------------------------------------------
        #  Lower bar: hover + delta z (just shows info)
        # ----------------------------------------------------------
        info_bar = QHBoxLayout()
        main_layout.addLayout(info_bar)

        self._hover_label = QLabel("Hover: (x=?, y=?, z=?)", self)
        self._hover_label.setObjectName("hoverLabel")  # styled in styles.qss
        info_bar.addWidget(self._hover_label)

        self._delta_label = QLabel("Δz: ?", self)
        self._delta_label.setObjectName("deltaLabel")  # To add style from styles.qss
        self._delta_label.setTextFormat(Qt.TextFormat.RichText)
        info_bar.addWidget(self._delta_label)

        # ----------------------------------------------------------
        #  Heightmap view
        # ----------------------------------------------------------
        self._heightmap_view = HeightmapView(self)
        main_layout.addWidget(self._heightmap_view, stretch=1)

        # ----------------------------------------------------------
        # Connections VM <-> UI
        # ----------------------------------------------------------
        self._vm.datasetsChanged.connect(self._on_datasets_changed)
        self._vm.heightmapLoaded.connect(self._on_heightmap_loaded)
        self._vm.error.connect(self._on_error)
        self._vm.representativePixelsLoaded.connect(self._on_pixels_loaded)

        # ----------------------------------------------------------
        # Connections UI <-> VM
        # ----------------------------------------------------------
        self._refresh_btn.clicked.connect(self._vm.refresh_datasets)
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._toggle_markers_btn.clicked.connect(self._on_toggle_markers)
        self._tilt_btn.clicked.connect(self._on_tilt_correction_clicked)

        # ----------------------------------------------------------
        # Connections HeightmapView <-> UI
        # ----------------------------------------------------------
        self._heightmap_view.hoverValueChanged.connect(self._on_hover_value)
        self._heightmap_view.measurementChanged.connect(self._on_measurement_changed)

        # Load initial dataset list
        self._vm.refresh_datasets()

    # ======================================================================
    #  Public methods: used from MainWindow
    # ======================================================================
    def refresh_datasets(self) -> None:
        """Refresh the dataset list shown in the combo box."""
        self._vm.refresh_datasets()

    def load_heightmap(self, dataset_name: str) -> None:
        """Loads heightmap + pixels plots (busy cursor, controls disabled)."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._set_controls_enabled(False)
        try:
            self._vm.load_heightmap_for_dataset(dataset_name)
            self._vm.load_representative_pixels(dataset_name)
        finally:
            self._set_controls_enabled(True)
            QApplication.restoreOverrideCursor()

    def select_dataset(self, name: str) -> None:
        """Select *name* in the dataset combo, if it exists."""
        idx = self._datasets_combo.findText(name)
        if idx >= 0:
            self._datasets_combo.setCurrentIndex(idx)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable the top control bar (used while loading a heightmap)."""
        self._datasets_combo.setEnabled(enabled)
        self._refresh_btn.setEnabled(enabled)
        self._load_btn.setEnabled(enabled)
        self._toggle_markers_btn.setEnabled(enabled)
        self._tilt_btn.setEnabled(enabled and self._original_arr is not None)

    # ======================================================================
    #  SLOTS — VM <-> UI
    # ======================================================================
    @Slot(list)
    def _on_datasets_changed(self, datasets: list[str]) -> None:
        self._datasets_combo.clear()
        self._datasets_combo.addItems(datasets)

    @Slot(object, tuple, str, str)
    def _on_heightmap_loaded(self, arr, shape, dtype_str: str, path: str) -> None:
        """Receives heightmap from VM, auto-applies tilt correction, and displays it."""
        self._original_arr = arr
        self._original_path = path
        self._tilt_btn.setEnabled(self._original_arr is not None)

        try:
            display_arr = fit_and_subtract_plane(arr)
            self._set_tilt_ui(True)
        except Exception:
            display_arr = arr
            self._set_tilt_ui(False)

        self._heightmap_view.set_heightmap(display_arr, shape, dtype_str, path)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Error", msg)

    @Slot(list)
    def _on_pixels_loaded(self, points: list) -> None:
        """
        points = [(x, y, npy_path), ...]
        """
        self._representative_points = points

        # Clear previous markers
        self._heightmap_view.clear_markers()

        # Add new ones
        for x, y, npy_path in points:
            self._heightmap_view.add_marker(x, y, userdata=npy_path)

        # Respect current marker visibility choice
        self._heightmap_view.set_markers_visible(self._markers_visible)

    # ======================================================================
    #  SLOTS — UI <-> VM
    # ======================================================================
    @Slot()
    def _on_load_clicked(self) -> None:
        dataset_name = self._datasets_combo.currentText()
        self.load_heightmap(dataset_name)

    @Slot()
    def _on_toggle_markers(self) -> None:
        self._markers_visible = not self._markers_visible
        self._heightmap_view.set_markers_visible(self._markers_visible)
        self._toggle_markers_btn.setText(
            "Hide markers" if self._markers_visible else "Show markers"
        )

    @Slot()
    def _on_tilt_correction_clicked(self) -> None:
        """Toggle linear-plane tilt correction."""
        if self._original_arr is None:
            self._tilt_btn.setChecked(False)
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self._tilt_btn.isChecked():
                # Apply correction
                try:
                    corrected = fit_and_subtract_plane(self._original_arr)
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(self, "Tilt correction failed", str(exc))
                    self._set_tilt_ui(False)
                    return
                self._set_tilt_ui(True)
                self._display(corrected)
            else:
                # Revert to original
                self._set_tilt_ui(False)
                self._display(self._original_arr)

            # Re-apply markers (set_heightmap clears them internally)
            self._reapply_markers()
        finally:
            QApplication.restoreOverrideCursor()

    def _set_tilt_ui(self, on: bool) -> None:
        """Sync the tilt button (checked state, text and colour) with *on*."""
        self._tilt_btn.setChecked(on)
        self._tilt_btn.setText("Tilt correction: on" if on else "Tilt correction: off")
        theme.set_variant(self._tilt_btn, "success" if on else "danger")

    def _display(self, arr: np.ndarray) -> None:
        """Show *arr* in the heightmap view, tagged with the original file path."""
        self._heightmap_view.set_heightmap(
            arr,
            shape=arr.shape,
            dtype=str(arr.dtype),
            path=self._original_path,
        )

    def _reapply_markers(self) -> None:
        """Re-add representative markers after a display refresh."""
        for x, y, npy_path in self._representative_points:
            self._heightmap_view.add_marker(x, y, userdata=npy_path)
        self._heightmap_view.set_markers_visible(self._markers_visible)

    # ======================================================================
    #  SLOTS — events from HeightmapView
    # ======================================================================
    @Slot(int, int, float)
    def _on_hover_value(self, x: int, y: int, z: float) -> None:
        text = f"Hover: x={x}, y={y}, z={z:.4f} µm"
        self._hover_label.setText(text)
        self.hoverInfoChanged.emit(text)

    @Slot(int, int, float, int, int, float, float)
    def _on_measurement_changed(
        self,
        x1: int,
        y1: int,
        z1: float,
        x2: int,
        y2: int,
        z2: float,
        dz: float,
    ) -> None:
        detail = (
            f"(P1: x={x1}, y={y1}, z={z1:.4f} µm"
            f"&nbsp;&nbsp;→&nbsp;&nbsp;"
            f"P2: x={x2}, y={y2}, z={z2:.4f} µm)"
        )
        self._delta_label.setText(
            f'Δz: {dz:+.4f} µm&nbsp;&nbsp;<span style="font-size: 9pt;">{detail}</span>'
        )
