"""Construction smoke tests for the UI layer (panels, MainWindow, QSS, widgets).

Everything runs offscreen. Building panels is safe (no hardware is touched
until the connect/start slots are invoked, which these tests never do).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import views.MainWindow as mainwindow_mod
from views.AcquisitionPanel import AcquisitionPanel
from views.MainWindow import MainWindow
from views.ProcessingPanel import ProcessingPanel
from views.ResultsPanel import ResultsPanel
from widgets.HeightmapView import HeightmapView
from widgets.PixelPlotsWidget import PixelPlotsWidget


@pytest.fixture(autouse=True)
def _project_cwd(monkeypatch, project_root):
    """Run from the project root so icons/data/output paths resolve as in main.py."""
    monkeypatch.chdir(project_root)


@pytest.fixture
def _isolated_config(monkeypatch):
    """Keep MainWindow from reading/writing the real app_config.json."""
    saved: list[dict] = []
    monkeypatch.setattr(mainwindow_mod, "load_config", lambda: {})
    monkeypatch.setattr(mainwindow_mod, "save_config", saved.append)
    return saved


def _assert_grab_ok(widget):
    """Assert that grab() renders a non-empty pixmap for *widget*."""
    pix = widget.grab()
    assert not pix.isNull()
    assert pix.width() > 0 and pix.height() > 0


# ======================================================================
#  Panel / MainWindow construction
# ======================================================================
def test_acquisition_panel_constructs(qtbot):
    """AcquisitionPanel builds offscreen without exceptions and renders via grab()."""
    panel = AcquisitionPanel()
    qtbot.addWidget(panel)
    _assert_grab_ok(panel)
    # Initial state: nothing connected, nothing running
    assert not panel._hardware_connected
    assert not panel.btn_start.isEnabled()


def test_processing_panel_constructs(qtbot):
    """ProcessingPanel builds offscreen, populates methods, and renders via grab()."""
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    _assert_grab_ok(panel)
    assert panel._method_combo.count() > 0
    assert panel._start_btn.isEnabled()
    assert not panel._cancel_btn.isEnabled()


def test_results_panel_constructs(qtbot):
    """ResultsPanel builds offscreen without exceptions and renders via grab()."""
    panel = ResultsPanel()
    qtbot.addWidget(panel)
    _assert_grab_ok(panel)
    assert not panel._tilt_btn.isEnabled()  # no heightmap loaded yet


def test_main_window_constructs(qtbot, _isolated_config):
    """MainWindow builds with its three tabs and closes cleanly on teardown."""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.windowTitle() == "InterferoLab"
    assert win._tabs.count() == 3
    labels = [win._tabs.tabText(i) for i in range(3)]
    assert labels == ["Acquisition", "Processing", "Results"]
    _assert_grab_ok(win)


def test_main_window_close_saves_config(qtbot, _isolated_config):
    """Closing the MainWindow persists the acq/proc config dict (patched sink)."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.close()
    assert len(_isolated_config) == 1
    assert set(_isolated_config[0]) == {"acq", "proc"}


# ======================================================================
#  Global stylesheet
# ======================================================================
def test_qss_loads_without_warnings(qtbot, qapp, qtlog, project_root):
    """resources/styles.qss parses cleanly when applied the way main.py applies it."""
    qss_path = os.path.join(project_root, "resources", "styles.qss")
    with open(qss_path, encoding="utf-8") as f:
        qss = f.read()
    assert qss.strip(), "styles.qss must not be empty"

    # Same anchoring of relative url(resources/...) refs as main.py
    qss_base = project_root.replace(os.sep, "/").rstrip("/")
    qss = qss.replace("url(resources/", f"url({qss_base}/resources/")

    old = qapp.styleSheet()
    try:
        qapp.setStyleSheet(qss)
        qapp.processEvents()
    finally:
        qapp.setStyleSheet(old)

    suspicious = [
        r.message
        for r in qtlog.records
        if any(k in r.message.lower() for k in ("parse", "stylesheet", "css"))
    ]
    assert not suspicious, f"QSS produced Qt warnings: {suspicious}"


# ======================================================================
#  HeightmapView / PixelPlotsWidget with synthetic data
# ======================================================================
def _nan_heightmap(shape=(24, 32)) -> np.ndarray:
    """Synthetic float32 heightmap with a gradient and some NaN holes."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    arr = (xx * 0.1 + yy * 0.05).astype(np.float32)
    arr[0, 0] = np.nan
    arr[5, 5] = np.nan
    return arr


def test_heightmap_view_accepts_nan_heightmap(qtbot):
    """HeightmapView.set_heightmap accepts a float32 array with NaNs and updates ranges."""
    view = HeightmapView()
    qtbot.addWidget(view)
    arr = _nan_heightmap()
    view.set_heightmap(arr, shape=arr.shape, dtype=str(arr.dtype), path="/tmp/fake_height.npy")
    qtbot.wait(50)  # let the deferred LUT/marker singleShot callbacks run

    finite = arr[np.isfinite(arr)]
    assert view._data_img is arr or np.array_equal(view._data_img, arr, equal_nan=True)
    assert view._lut_full_min == pytest.approx(float(finite.min()))
    assert view._lut_full_max == pytest.approx(float(finite.max()))
    assert view._label.text() == "fake_height.npy"


def test_heightmap_view_clears_with_none(qtbot):
    """set_heightmap(None) clears the view and reports 'no data'."""
    view = HeightmapView()
    qtbot.addWidget(view)
    view.set_heightmap(_nan_heightmap())
    view.set_heightmap(None)
    qtbot.wait(20)
    assert view._data_img is None
    assert "no data" in view._label.text()


def test_heightmap_view_rejects_non_2d(qtbot):
    """A non-2D heightmap raises ValueError instead of corrupting the view."""
    view = HeightmapView()
    qtbot.addWidget(view)
    with pytest.raises(ValueError, match="must be 2D"):
        view.set_heightmap(np.zeros((4, 4, 3), dtype=np.float32))


def test_pixel_plots_accept_nan_heightmap(qtbot):
    """PixelPlotsWidget.set_heightmap_data handles NaN data and syncs its X range."""
    widget = PixelPlotsWidget()
    qtbot.addWidget(widget)
    arr = _nan_heightmap()
    widget.set_heightmap_data(arr)  # must not raise despite NaNs
    assert widget._heightmap_data is arr

    widget.set_histogram_xlim(1.5, 4.5)
    qtbot.wait(20)
    (xmin, xmax), _y = widget._hist_widget.getViewBox().viewRange()
    assert xmin == pytest.approx(1.5, abs=1e-6)
    assert xmax == pytest.approx(4.5, abs=1e-6)


def test_pixel_plots_degenerate_heightmaps(qtbot):
    """All-NaN and None heightmaps are handled without exceptions."""
    widget = PixelPlotsWidget()
    qtbot.addWidget(widget)
    widget.set_heightmap_data(np.full((8, 8), np.nan, dtype=np.float32))
    widget.set_heightmap_data(None)
    widget.clear()
    assert widget._heightmap_data is None
