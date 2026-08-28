# widgets/HeightmapView.py
from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from utils.icons import icon
from widgets.PixelPlotsWidget import PixelPlotsWidget
from widgets.theme import MATPLOTLIB_BG, MATPLOTLIB_RED

# ---------------------------------------------------------------------
# Global pyqtgraph configuration
# ---------------------------------------------------------------------
# Row-major to match NumPy / matplotlib:
#   self._data_img[y, x] == height[y, x]
pg.setConfigOptions(imageAxisOrder="row-major")


class NeutralViewBox(pg.ViewBox):
    """
    ViewBox with a neutral background and a simplified interaction mode.

    The mode can be:
      - "zoom": left-drag performs rectangular zoom/pan (standard ViewBox).
      - "measure": left-drag is ignored (used by HeightmapView to measure
        distances with simple left-clicks instead of drag-zoom).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setBackgroundColor(MATPLOTLIB_BG)
        self._interaction_mode: str = "zoom"

    def setInteractionMode(self, mode: str) -> None:
        """Set interaction mode: 'zoom', 'drag', or 'measure'."""
        self._interaction_mode = mode

    def mouseDragEvent(self, ev, axis=None) -> None:
        # In measure mode, ignore left-drag (no zoom/pan)
        if ev.button() == Qt.MouseButton.LeftButton and self._interaction_mode == "measure":
            ev.ignore()
            return
        super().mouseDragEvent(ev, axis=axis)


class HeightmapView(QWidget):
    """
    Combined 2D heightmap viewer + pixel-plots side panel.

    Layout:
      - Left column: PixelPlotsWidget (two pixel plots + height histogram).
      - Right column: main heightmap image with a small toolbar:
            [ Clear Δz ] [ Home ] [ Zoom ] [ Measure ]

    Interaction:
      - Right-click on image: pyqtgraph's context menu.
      - Left-click on image:
          * In Zoom mode   -> standard rectangular zoom.
          * In Measure mode -> two-point measurement (Δz).
    """

    # Public signals
    hoverValueChanged = Signal(int, int, float)
    measurementChanged = Signal(
        int,
        int,
        float,
        int,
        int,
        float,
        float,
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Image data (2D heightmap)
        self._data_img: np.ndarray | None = None
        self._lut_full_min: float = 0.0
        self._lut_full_max: float = 1.0

        # Measurement state: p1 and p2 are (x, y, z)
        self._p1: tuple[int, int, float] | None = None
        self._p2: tuple[int, int, float] | None = None
        self._meas_scatter: pg.ScatterPlotItem | None = None
        self._meas_line: pg.PlotDataItem | None = None

        # Markers: marker_id -> (x, y, userdata)
        # userdata is typically a path to a pixel .npy file
        self._markers: dict[int, tuple[int, int, object]] = {}
        self._marker_layer: pg.ScatterPlotItem | None = None
        self._marker_labels: dict[int, pg.TextItem] = {}
        self._markers_visible: bool = False  # hidden by default
        self._next_marker_id: int = 1
        self._marker_update_pending: bool = False

        # Interaction mode: "zoom" or "measure"
        self._mode: str = "measure"

        # ------------------------------------------------------------------
        # Layout: left pixel-plots panel + right heightmap panel
        # ------------------------------------------------------------------
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ------ Left panel: pixel plots widget ------
        self._pixel_plots = PixelPlotsWidget(self)
        root.addWidget(self._pixel_plots, stretch=2)

        # ------ Right panel (heightmap) ------
        right_panel = QWidget(self)
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header: label + small toolbar
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 4, 4, 4)

        self._label = QLabel("Heightmap (no data)")
        header_layout.addWidget(self._label)
        header_layout.addStretch()

        # Reset view
        self._home_btn = self._make_tool_button(
            header_layout,
            "Reset",
            "reset.svg",
            "Reset view to full image",
            self._reset_view,
        )

        # Zoom mode
        self._zoom_btn = self._make_tool_button(
            header_layout,
            "Zoom",
            "zoom.svg",
            "Zoom mode (left drag = zoom region)",
            self._set_mode_zoom,
            checkable=True,
        )

        # Drag / pan mode
        self._drag_btn = self._make_tool_button(
            header_layout,
            "Drag",
            "drag.svg",
            "Pan mode (left drag = pan)",
            self._set_mode_drag,
            checkable=True,
        )

        # Measure mode
        self._measure_btn = self._make_tool_button(
            header_layout,
            "Measure",
            "measure.svg",
            "Measure mode (left click = pick points)",
            self._set_mode_measure,
            checkable=True,
        )

        # Clear Δz (action for Measure mode, placed next to it)
        self._clear_btn = self._make_tool_button(
            header_layout,
            "Clear Δz",
            "clear.svg",
            "Clear measurement points",
            self.clear_measurement,
        )

        layout.addLayout(header_layout)

        # ViewBox + ImageView (mode is set at end of __init__ via _set_mode_drag)
        vb = NeutralViewBox(lockAspect=True)

        self._image_view = pg.ImageView(view=vb)
        self._image_view.setStyleSheet(f"background-color: {MATPLOTLIB_BG};")
        layout.addWidget(self._image_view, stretch=3)

        root.addWidget(right_panel, stretch=3)

        # Hide ROI/menu buttons; set up LUT histogram after widget is shown.
        # `self` is passed as context object so Qt drops the pending call if the
        # widget is destroyed before the timer fires; otherwise the callback runs
        # against an already-deleted C++ object ("libshiboken: Internal C++ object
        # (PySide6.QtWidgets.QPushButton) already deleted" under PySide6 6.11.2).
        QTimer.singleShot(0, self, self._sanitize_imageview_menu)
        QTimer.singleShot(0, self, self._setup_lut)

        # Connect mouse events from the image scene
        scene = self._image_view.getView().scene()
        scene.sigMouseMoved.connect(self._on_mouse_moved)
        scene.sigMouseClicked.connect(self._on_scene_mouse_clicked)

        # Start in drag/pan mode by default
        self._set_mode_drag()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _make_tool_button(
        self,
        layout: QHBoxLayout,
        text: str,
        icon_name: str,
        tooltip: str,
        on_clicked,
        checkable: bool = False,
    ) -> QToolButton:
        """Create and lay out one toolbar QToolButton (styled by the app QSS)."""
        btn = QToolButton()
        btn.setCheckable(checkable)
        btn.setText(text)
        btn.setIcon(icon(icon_name))
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setAutoRaise(True)
        btn.setToolTip(tooltip)
        btn.clicked.connect(on_clicked)
        layout.addWidget(btn)
        return btn

    def _get_viewbox(self) -> pg.ViewBox | None:
        view = self._image_view.getView()
        return view if isinstance(view, pg.ViewBox) else None

    def _get_image_item(self) -> pg.ImageItem | None:
        return getattr(self._image_view, "imageItem", None)

    def _remove_item_safely(self, item: Any) -> None:
        if item is None:
            return
        scene = item.scene()
        if scene is not None:
            with contextlib.suppress(Exception):
                scene.removeItem(item)

    # ------------------------------------------------------------------
    # Interaction modes
    # ------------------------------------------------------------------
    def _set_mode_zoom(self) -> None:
        self._mode = "zoom"
        self._zoom_btn.setChecked(True)
        self._drag_btn.setChecked(False)
        self._measure_btn.setChecked(False)
        vb = self._get_viewbox()
        if isinstance(vb, NeutralViewBox):
            vb.setMouseMode(pg.ViewBox.RectMode)
            vb.setInteractionMode("zoom")

    def _set_mode_drag(self) -> None:
        self._mode = "drag"
        self._zoom_btn.setChecked(False)
        self._drag_btn.setChecked(True)
        self._measure_btn.setChecked(False)
        vb = self._get_viewbox()
        if isinstance(vb, NeutralViewBox):
            vb.setMouseMode(pg.ViewBox.PanMode)
            vb.setInteractionMode("drag")

    def _set_mode_measure(self) -> None:
        self._mode = "measure"
        self._zoom_btn.setChecked(False)
        self._drag_btn.setChecked(False)
        self._measure_btn.setChecked(True)
        vb = self._get_viewbox()
        if isinstance(vb, NeutralViewBox):
            vb.setInteractionMode("measure")

    # ------------------------------------------------------------------
    # View reset
    # ------------------------------------------------------------------
    def _reset_view(self) -> None:
        vb = self._get_viewbox()
        if vb is not None:
            vb.autoRange()

    # ------------------------------------------------------------------
    # Menu / LUT histogram
    # ------------------------------------------------------------------

    def _sanitize_imageview_menu(self) -> None:
        ui = self._image_view.ui
        ui.roiBtn.hide()
        ui.menuBtn.hide()

    def _setup_lut(self) -> None:
        """One-time LUT setup: style, log-histogram connection, level-sync."""
        try:
            hist = self._image_view.ui.histogram
            lut = hist.item

            # Set the whole widget background so the axis label area matches
            hist.setBackground(MATPLOTLIB_BG)
            lut.vb.setBackgroundColor(MATPLOTLIB_BG)
            lut.axis.setPen("k")
            lut.axis.setTextPen("k")
            lut.axis.setStyle(showValues=True, tickTextOffset=5)

            # Sync left-panel histogram X range when colorbar levels change
            hist.sigLevelsChanged.connect(self._on_lut_levels_changed)

            # Recompute log histogram whenever a new image is set
            self._image_view.imageItem.sigImageChanged.connect(
                lambda: QTimer.singleShot(0, self, self._refresh_lut_log)
            )
        except Exception:
            pass

    def _on_lut_levels_changed(self) -> None:
        try:
            mn, mx = self._image_view.ui.histogram.getLevels()
            self._pixel_plots.set_histogram_xlim(mn, mx)
        except Exception:
            pass

    def _hide_gradient_ticks(self) -> None:
        """Hide the triangle handles on the gradient colorbar."""
        try:
            grad = self._image_view.ui.histogram.item.gradient
            for tick in list(grad.ticks):
                tick.hide()
        except Exception:
            pass

    def _refresh_lut_log(self) -> None:
        """Redraw the LUT histogram in log10(count+1) with linear height ticks."""
        if self._data_img is None:
            return
        try:
            hist = self._image_view.ui.histogram
            lut = hist.item
            bins, counts = self._image_view.imageItem.getHistogram()
            if bins is None or counts is None:
                return

            lo, hi = self._lut_full_min, self._lut_full_max
            span = hi - lo
            decimals = max(0, min(4, int(np.ceil(-np.log10(span))) + 2)) if span > 0 else 3
            tick_vals = np.linspace(lo, hi, 10)
            major_ticks = [(float(v), f"{v:.{decimals}f}") for v in tick_vals]
            fm = QFontMetrics(self.font())
            ax_w = max(55, max(fm.horizontalAdvance(t[1]) for t in major_ticks) + 10)

            # Apply axis size and force layout BEFORE setData triggers a repaint.
            # Root cause: with maxTickLength=-10 the axis bounding rect only extends
            # rightward; text at x<0 is clipped unless the column has non-zero width
            # allocated by the time paint() fires.
            lut.axis.setWidth(ax_w)
            lut.axis.setTicks([major_ticks, []])
            lut.axis.picture = None  # discard cached QPicture so it redraws with new width
            if hasattr(lut, "layout"):
                lut.layout.setColumnMinimumWidth(0, ax_w)
                lut.layout.invalidate()
                lut.layout.activate()  # synchronous re-layout before paint
            hist.setMinimumWidth(ax_w + 45)

            # Update histogram data and pin the height (Y) range
            lut.plot.setData(x=bins, y=np.log10(np.maximum(counts, 0).astype(float) + 1))
            lut.vb.setYRange(lo, hi, padding=0.02)
            lut.axis.update()
        except Exception:
            pass
        self._hide_gradient_ticks()

    # ------------------------------------------------------------------
    # Scene position -> (x,y) indices
    # ------------------------------------------------------------------
    def _scene_pos_to_xy(self, scene_pos) -> tuple[int, int] | None:
        item = self._get_image_item()
        if item is None or self._data_img is None:
            return None
        if not item.sceneBoundingRect().contains(scene_pos):
            return None
        local = item.mapFromScene(scene_pos)
        x = int(local.x())
        y = int(local.y())
        H, W = self._data_img.shape
        if 0 <= x < W and 0 <= y < H:
            return x, y
        return None

    # ------------------------------------------------------------------
    # LUT / colormap (fixed to viridis)
    # ------------------------------------------------------------------
    def _apply_current_lut(self) -> None:
        if self._data_img is None:
            return
        cmap = pg.colormap.get("viridis")
        self._image_view.setColorMap(cmap)
        item = self._get_image_item()
        if item is not None:
            item.setLookupTable(cmap.getLookupTable(alpha=False))
        self._hide_gradient_ticks()

    # ------------------------------------------------------------------
    # Heightmap loading
    # ------------------------------------------------------------------
    @Slot(object, tuple, str, str)
    def set_heightmap(
        self,
        arr: Any,
        shape: tuple | None = None,
        dtype: str | None = None,
        path: str | None = None,
    ) -> None:
        """
        Set the heightmap array to be displayed.

        Parameters
        ----------
        arr : np.ndarray
            2D array (H, W) with height values.
        shape : tuple, optional
            Shape reported by the loader (for text only).
        dtype : str, optional
            Dtype reported by the loader (for text only).
        path : str, optional
            Source file path (for display only).
        """
        self._clear_measurement()
        self.clear_markers()
        self._pixel_plots.clear()

        if arr is None:
            self._data_img = None
            self._image_view.clear()
            self._label.setText("Heightmap: no data")
            self._pixel_plots.set_heightmap_data(None)
            return

        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError("Heightmap must be 2D (H, W)")

        self._data_img = arr

        vals = self._data_img[np.isfinite(self._data_img)]
        if vals.size > 0:
            vmin, vmax = np.percentile(vals, [2, 98])
            self._lut_full_min = float(vals.min())
            self._lut_full_max = float(vals.max())
        else:
            vmin = vmax = 0.0
            self._lut_full_min, self._lut_full_max = 0.0, 1.0

        self._apply_current_lut()

        self._image_view.setImage(
            self._data_img,
            autoLevels=False,
            levels=(vmin, vmax),
        )

        item = self._get_image_item()
        if item is not None:
            item.setLevels((vmin, vmax))

        with contextlib.suppress(Exception):
            self._image_view.ui.histogram.setLevels(vmin, vmax)

        txt = os.path.basename(path) if path else "Heightmap"
        self._label.setText(txt)

        self._pixel_plots.set_heightmap_data(arr)
        self._pixel_plots.set_histogram_xlim(vmin, vmax)
        self._recreate_marker_layer()
        self._pixel_plots.set_from_markers(self._markers)
        self._reset_view()

    # ------------------------------------------------------------------
    # Measurement tools
    # ------------------------------------------------------------------
    @Slot()
    def clear_measurement(self) -> None:
        """Clear measurement points and overlay graphics."""
        self._clear_measurement()

    def _clear_measurement(self) -> None:
        self._p1 = None
        self._p2 = None
        self._remove_item_safely(self._meas_scatter)
        self._remove_item_safely(self._meas_line)
        self._meas_scatter = None
        self._meas_line = None

    def _update_measurement_graphics(self) -> None:
        self._remove_item_safely(self._meas_scatter)
        self._remove_item_safely(self._meas_line)

        if self._p1 is None:
            self._meas_scatter = None
            self._meas_line = None
            return

        vb = self._get_viewbox()
        if vb is None:
            return

        xs = [self._p1[0]]
        ys = [self._p1[1]]
        if self._p2 is not None:
            xs.append(self._p2[0])
            ys.append(self._p2[1])

        self._meas_scatter = pg.ScatterPlotItem(
            x=xs,
            y=ys,
            symbol="+",
            size=12,
            pen=pg.mkPen(MATPLOTLIB_RED, width=2),
        )
        vb.addItem(self._meas_scatter)

        if self._p2 is not None:
            self._meas_line = pg.PlotDataItem(
                x=[self._p1[0], self._p2[0]],
                y=[self._p1[1], self._p2[1]],
                pen=pg.mkPen(MATPLOTLIB_RED, width=1, style=Qt.DashLine),
            )
            vb.addItem(self._meas_line)

    # ------------------------------------------------------------------
    # Markers (used to show representative pixels)
    # ------------------------------------------------------------------
    def _recreate_marker_layer(self) -> None:
        vb = self._get_viewbox()
        if vb is None:
            return
        self._remove_item_safely(self._marker_layer)
        # _update_marker_layer will recreate text labels; pre-clear any lingering refs
        for ti in self._marker_labels.values():
            self._remove_item_safely(ti)
        self._marker_labels.clear()

        self._marker_layer = pg.ScatterPlotItem(
            symbol="+",
            size=18,
            pen=pg.mkPen(MATPLOTLIB_RED, width=2),
            brush=None,
        )
        vb.addItem(self._marker_layer)
        self._update_marker_layer()
        self._marker_layer.setVisible(self._markers_visible)

    def add_marker(self, x: int, y: int, userdata=None) -> None:
        """Add a marker at (x,y) with optional userdata (e.g. pixel .npy path)."""
        marker_id = self._next_marker_id
        self._next_marker_id += 1
        self._markers[marker_id] = (x, y, userdata)
        # Defer both layer redraw and plot update so N batched calls cost O(N) not O(N²)
        if not self._marker_update_pending:
            self._marker_update_pending = True
            QTimer.singleShot(0, self, self._flush_marker_updates)

    def _flush_marker_updates(self) -> None:
        self._marker_update_pending = False
        self._update_marker_layer()
        self._pixel_plots.set_from_markers(self._markers)

    def clear_markers(self) -> None:
        """Remove all markers and clear left plots."""
        self._markers.clear()
        self._next_marker_id = 1
        self._marker_update_pending = False
        self._update_marker_layer()
        self._pixel_plots.clear()

    def set_markers_visible(self, visible: bool) -> None:
        """Show or hide the marker overlay."""
        self._markers_visible = visible
        if self._marker_layer is not None:
            self._marker_layer.setVisible(visible)
        for ti in self._marker_labels.values():
            ti.setVisible(visible)

    def _update_marker_layer(self) -> None:
        # Remove existing text labels
        vb = self._get_viewbox()
        for ti in self._marker_labels.values():
            self._remove_item_safely(ti)
        self._marker_labels.clear()

        if self._marker_layer is None:
            return

        # System font (matches the app theme) instead of hardcoded Arial
        _label_font = QFont(self.font())
        _label_font.setPointSize(8)

        spots = []
        for mid, (x, y, _userdata) in self._markers.items():
            spots.append({"pos": (x, y), "data": mid})
            # Numbered text label next to the '+' symbol
            ti = pg.TextItem(str(mid), color=MATPLOTLIB_RED, anchor=(0.0, 1.0))
            ti.setFont(_label_font)
            ti.setPos(x, y)
            if vb is not None:
                vb.addItem(ti)
            ti.setVisible(self._markers_visible)
            self._marker_labels[mid] = ti

        self._marker_layer.setData(spots)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def _on_mouse_moved(self, pos) -> None:
        """Emit hoverValueChanged when the mouse moves over the image."""
        if self._data_img is None:
            return
        xy = self._scene_pos_to_xy(pos)
        if xy is None:
            return
        x, y = xy
        z = float(self._data_img[y, x])
        if not np.isfinite(z):
            return
        self.hoverValueChanged.emit(x, y, z)

    def _on_scene_mouse_clicked(self, event) -> None:
        """
        Mouse click handler:

          - Right-click: leave untouched (pyqtgraph context menu).
          - Left-click: used for measurement only when _mode == "measure".
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._mode != "measure":
            return

        pos = event.scenePos()
        xy = self._scene_pos_to_xy(pos)
        if xy is None:
            return

        x, y = xy
        z = float(self._data_img[y, x])

        if self._p1 is None:
            self._p1 = (x, y, z)
            self._p2 = None
        elif self._p2 is None:
            self._p2 = (x, y, z)
            dz = self._p2[2] - self._p1[2]
            self.measurementChanged.emit(
                self._p1[0],
                self._p1[1],
                self._p1[2],
                self._p2[0],
                self._p2[1],
                self._p2[2],
                dz,
            )
        else:
            self._p1 = (x, y, z)
            self._p2 = None

        self._update_measurement_graphics()
