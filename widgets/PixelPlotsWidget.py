# widgets/PixelPlotsWidget.py
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.pixel_signal import baseline_sigma_for, gaussian_smooth, refined_centroid
from widgets.theme import MATPLOTLIB_BG, MATPLOTLIB_BLUE, MATPLOTLIB_RED, brush


def _title_span(text: str) -> str:
    """Wrap a pyqtgraph plot title in the black-text span used across the app."""
    return f'<span style="color:black">{text}</span>'


class PixelPlotsWidget(QWidget):
    """
    Left panel of the results view with:
      - Two pixel plot selectors (text inputs + Update Plots button)
      - Two per-pixel interferogram plots
      - Height distribution histogram of the loaded heightmap
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ------------------------------------------------------------------
        # Input row: pixel selectors + Update button
        # ------------------------------------------------------------------
        input_row = QHBoxLayout()
        input_row.setSpacing(4)

        input_row.addWidget(QLabel("Pixel 1:"))
        self._input1 = QLineEdit("1")
        self._input1.setFixedWidth(45)
        self._input1.setToolTip("Marker number for the first pixel plot")
        input_row.addWidget(self._input1)

        input_row.addSpacing(6)
        input_row.addWidget(QLabel("Pixel 2:"))
        self._input2 = QLineEdit("2")
        self._input2.setFixedWidth(45)
        self._input2.setToolTip("Marker number for the second pixel plot")
        input_row.addWidget(self._input2)

        input_row.addSpacing(6)
        self._update_btn = QPushButton("Update plots")
        self._update_btn.setToolTip("Refresh the plots for the entered pixel numbers")
        input_row.addWidget(self._update_btn)
        input_row.addStretch()

        layout.addLayout(input_row)

        # ------------------------------------------------------------------
        # Two pixel plots
        # ------------------------------------------------------------------
        self._plot_pixel1 = pg.PlotWidget()
        self._plot_pixel2 = pg.PlotWidget()
        self._setup_pixel_plot(self._plot_pixel1, "Pixel 1 (no data)")
        self._setup_pixel_plot(self._plot_pixel2, "Pixel 2 (no data)")
        layout.addWidget(self._plot_pixel1)
        layout.addWidget(self._plot_pixel2)

        # ------------------------------------------------------------------
        # Height distribution histogram
        # ------------------------------------------------------------------
        self._hist_widget = pg.PlotWidget()
        self._setup_hist_plot()
        layout.addWidget(self._hist_widget)

        # Internal state
        self._markers: dict[int, tuple[int, int, Any]] = {}
        self._heightmap_data: np.ndarray | None = None

        # Connect button
        self._update_btn.clicked.connect(self._on_update_plots)
        # Also update on Enter in either input
        self._input1.returnPressed.connect(self._on_update_plots)
        self._input2.returnPressed.connect(self._on_update_plots)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Clear pixel plots and reset inputs to defaults."""
        self._markers = {}
        self._plot_pixel1.clear()
        self._plot_pixel2.clear()
        self._plot_pixel1.setTitle(_title_span("Pixel 1 (no data)"))
        self._plot_pixel2.setTitle(_title_span("Pixel 2 (no data)"))
        self._input1.setText("1")
        self._input2.setText("2")

    def set_heightmap_data(self, arr: np.ndarray | None) -> None:
        """Pass the current heightmap for the distribution histogram."""
        self._heightmap_data = arr
        self._update_histogram()

    def set_histogram_xlim(self, xmin: float, xmax: float) -> None:
        """Sync the histogram X-axis to the colorbar level range."""
        self._hist_widget.setXRange(xmin, xmax, padding=0)

    def set_from_markers(
        self,
        markers: dict[int, tuple[int, int, Any]],
    ) -> None:
        """
        Update plots based on a markers dict: marker_id -> (x, y, userdata).
        Sets the inputs to the first two marker IDs and refreshes the plots.
        """
        self._markers = dict(markers)
        if not markers:
            self._plot_pixel1.clear()
            self._plot_pixel2.clear()
            self._plot_pixel1.setTitle(_title_span("Pixel 1 (no data)"))
            self._plot_pixel2.setTitle(_title_span("Pixel 2 (no data)"))
            return

        ids = sorted(markers.keys())
        self._input1.setText(str(ids[0]))
        self._input2.setText(str(ids[1]) if len(ids) > 1 else "")
        self._on_update_plots()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _on_update_plots(self) -> None:
        """Read the input fields and refresh both pixel plots."""
        id1 = self._parse_id(self._input1.text())
        id2 = self._parse_id(self._input2.text())

        if id1 is not None and id1 in self._markers:
            x, y, npy_path = self._markers[id1]
            self._plot_pixel_to_widget(x, y, npy_path, self._plot_pixel1)
        else:
            self._plot_pixel1.clear()
            self._plot_pixel1.setTitle(_title_span("Pixel 1 (no data)"))

        if id2 is not None and id2 in self._markers:
            x, y, npy_path = self._markers[id2]
            self._plot_pixel_to_widget(x, y, npy_path, self._plot_pixel2)
        else:
            self._plot_pixel2.clear()
            self._plot_pixel2.setTitle(_title_span("Pixel 2 (no data)"))

    @staticmethod
    def _parse_id(text: str) -> int | None:
        try:
            return int(text.strip())
        except ValueError:
            return None

    def _update_histogram(self) -> None:
        """Recompute and display the height distribution histogram."""
        self._hist_widget.clear()
        if self._heightmap_data is None:
            self._hist_widget.setTitle(_title_span("Height Distribution (no data)"))
            return

        arr = np.asarray(self._heightmap_data)
        valid = arr[np.isfinite(arr)].flatten()
        if valid.size == 0:
            self._hist_widget.setTitle(_title_span("Height Distribution (no finite values)"))
            return

        counts, edges = np.histogram(valid, bins=200)
        log_counts = np.log10(counts.astype(float) + 1)
        self._hist_widget.setTitle(_title_span("Height Distribution"))
        self._hist_widget.plot(
            edges,
            np.append(log_counts, 0),
            stepMode="right",
            fillLevel=0,
            brush=brush(MATPLOTLIB_BLUE, 90),
            pen=pg.mkPen(MATPLOTLIB_BLUE, width=1.5),
        )

    def _setup_pixel_plot(self, pw: pg.PlotWidget, title: str) -> None:
        """Configure a PlotWidget with a simple matplotlib-like style."""
        pw.setBackground(MATPLOTLIB_BG)
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("bottom", "Pos (µm)")
        pw.setLabel("left", "Intensity")
        pw.setTitle(_title_span(title))

        vb = pw.getViewBox()
        if isinstance(vb, pg.ViewBox):
            vb.setMouseMode(pg.ViewBox.RectMode)

        for axis in ("left", "bottom"):
            ax = pw.getAxis(axis)
            ax.setPen("k")
            ax.setTextPen("k")

    def _setup_hist_plot(self) -> None:
        """Configure the histogram PlotWidget."""
        self._hist_widget.setBackground(MATPLOTLIB_BG)
        self._hist_widget.showGrid(x=True, y=True, alpha=0.3)
        self._hist_widget.setLabel("bottom", "Height (µm)")
        self._hist_widget.setLabel("left", "log₁₀(Count + 1)")
        self._hist_widget.setTitle(_title_span("Height Distribution (no data)"))

        vb = self._hist_widget.getViewBox()
        if isinstance(vb, pg.ViewBox):
            vb.setMouseMode(pg.ViewBox.RectMode)

        for axis in ("left", "bottom"):
            ax = self._hist_widget.getAxis(axis)
            ax.setPen("k")
            ax.setTextPen("k")

    def _plot_pixel_to_widget(
        self,
        x: int,
        y: int,
        npy_path: str,
        pw: pg.PlotWidget,
    ) -> None:
        """Load a pixel .npy file and display its signal/envelope/peak."""
        pw.clear()

        if not npy_path or not os.path.isfile(npy_path):
            pw.setTitle(_title_span(f"Pixel (x={x}, y={y}) — no data"))
            return

        # A truncated/corrupt file (e.g. from a cancelled or killed analysis)
        # must degrade to an error title, not raise out of the Qt slot.
        try:
            arr = np.load(npy_path)
        except Exception:
            pw.setTitle(_title_span(f"Pixel (x={x}, y={y}) — unreadable file"))
            return
        if arr.ndim != 2 or arr.shape[1] not in (3, 4) or arr.shape[0] == 0:
            pw.setTitle(_title_span(f"Pixel (x={x}, y={y}) — invalid format"))
            return

        pos = arr[:, 0].astype(np.float64)
        sig = arr[:, 1].astype(np.float64)
        env = arr[:, 2].astype(np.float64)
        # Col 3 carries the height estimate produced by the same estimator used
        # in the main reconstruction (peak locator for M3, centroid otherwise).
        h_est = float(arr[0, 3]) if arr.shape[1] == 4 else None

        pw.setTitle(_title_span(f"Pixel (x={x}, y={y})"))

        dc = gaussian_smooth(sig, baseline_sigma_for(npy_path))

        sig_osc_amp = np.max(np.abs(sig - dc))
        env_peak = np.max(env) if env.size > 0 else 0.0
        display_scale = (sig_osc_amp / env_peak) if env_peak > 1e-30 else 1.0
        env_disp = env * display_scale

        upper = dc + env_disp
        lower = dc - env_disp

        # Raw interferogram
        pw.plot(
            pos,
            sig,
            pen=pg.mkPen(MATPLOTLIB_BLUE, width=1.0),
            symbol="o",
            symbolSize=3,
            symbolBrush=MATPLOTLIB_BLUE,
            symbolPen=None,
        )

        # Envelope band: DC ± env
        c_upper = pw.plot(
            pos,
            upper,
            pen=pg.mkPen(MATPLOTLIB_RED, width=1.2, style=Qt.DashLine),
        )
        c_lower = pw.plot(
            pos,
            lower,
            pen=pg.mkPen(MATPLOTLIB_RED, width=1.2, style=Qt.DashLine),
        )

        fill = pg.FillBetweenItem(
            c_upper,
            c_lower,
            brush=brush(MATPLOTLIB_RED, 40),
        )
        pw.getPlotItem().addItem(fill)

        # Height marker — use the value saved by C++ when available (col 3),
        # which matches the actual estimator (peak locator or centroid) used in
        # the main reconstruction.  Fall back to Python centroid for old files.
        if env.size > 0:
            if h_est is not None:
                z_c = h_est
            else:
                z_c = refined_centroid(pos, env)
                if z_c is None:
                    pw.autoRange()
                    return

            pw.addItem(
                pg.InfiniteLine(
                    pos=z_c,
                    angle=90,
                    pen=pg.mkPen(MATPLOTLIB_RED, width=1.0, style=Qt.DashLine),
                    label=f"z={z_c:.3f} µm",
                    labelOpts={"color": MATPLOTLIB_RED, "position": 0.92},
                )
            )

            marker_y = float(np.interp(z_c, pos, upper))
            pw.plot(
                [z_c],
                [marker_y],
                pen=None,
                symbol="t",
                symbolSize=8,
                symbolBrush=MATPLOTLIB_RED,
                symbolPen=MATPLOTLIB_RED,
            )

        pw.autoRange()
