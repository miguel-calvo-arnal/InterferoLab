"""Tests for the ViewModel layer (AcquisitionVM, ResultsVM, AnalysisVM).

No hardware is touched: only pure helpers are called and service-level Qt
signals are emitted directly to verify the VM wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from viewmodels.acquisition_vm import AcquisitionVM
from viewmodels.analysis_vm import AnalysisVM
from viewmodels.results_vm import ResultsVM
from widgets.progress_info import ProgressInfoLabel


@pytest.fixture
def acq_vm(qtbot):
    """AcquisitionVM instance (service constructed but never connected)."""
    vm = AcquisitionVM()
    yield vm
    vm.deleteLater()


@pytest.fixture
def analysis_vm(qtbot):
    """AnalysisVM instance (service constructed but never started)."""
    vm = AnalysisVM()
    yield vm
    vm.deleteLater()


class _FakeElapsedTimer:
    """Deterministic stand-in for QElapsedTimer (elapsed() in ms)."""

    def __init__(self, elapsed_ms: int) -> None:
        self._elapsed_ms = elapsed_ms

    def isValid(self) -> bool:  # noqa: N802 (Qt camelCase API)
        return True

    def elapsed(self) -> int:
        return self._elapsed_ms

    def restart(self) -> None:
        pass


# ======================================================================
#  AcquisitionVM — numpy_to_pixmap
# ======================================================================
class TestNumpyToPixmap:
    def test_mono_12bit_pixmap_size(self, acq_vm):
        """A 12-bit mono frame converts to a non-null QPixmap of the frame size."""
        frame = np.random.randint(0, 4096, size=(32, 48), dtype=np.uint16)
        pix = acq_vm.numpy_to_pixmap(frame)
        assert not pix.isNull()
        assert (pix.width(), pix.height()) == (48, 32)

    def test_mono_8bit_pixmap_size(self, acq_vm):
        """An 8-bit mono frame converts without errors to the correct size."""
        frame = np.random.randint(0, 256, size=(20, 30), dtype=np.uint8)
        pix = acq_vm.numpy_to_pixmap(frame)
        assert not pix.isNull()
        assert (pix.width(), pix.height()) == (30, 20)

    def test_rgb_pixmap_size(self, acq_vm):
        """A 12-bit RGB frame converts to a non-null QPixmap of the frame size."""
        frame = np.random.randint(0, 4096, size=(16, 24, 3), dtype=np.uint16)
        pix = acq_vm.numpy_to_pixmap(frame)
        assert not pix.isNull()
        assert (pix.width(), pix.height()) == (24, 16)

    def test_rgba_extra_channels_ignored(self, acq_vm):
        """Frames with more than 3 channels keep only RGB and still convert."""
        frame = np.random.randint(0, 4096, size=(8, 10, 4), dtype=np.uint16)
        pix = acq_vm.numpy_to_pixmap(frame)
        assert not pix.isNull()
        assert (pix.width(), pix.height()) == (10, 8)

    @pytest.mark.parametrize("value", [0, 4095])
    def test_extreme_values_do_not_crash(self, acq_vm, value):
        """All-zero and all-4095 frames (mono and RGB) convert without errors."""
        mono = np.full((12, 12), value, dtype=np.uint16)
        rgb = np.full((12, 12, 3), value, dtype=np.uint16)
        assert not acq_vm.numpy_to_pixmap(mono).isNull()
        assert not acq_vm.numpy_to_pixmap(rgb).isNull()

    def test_unsupported_shape_raises(self, acq_vm):
        """A 1-D array is not a valid preview frame and raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported preview frame shape"):
            acq_vm.numpy_to_pixmap(np.arange(10, dtype=np.uint16))


# ======================================================================
#  AcquisitionVM — compute_preview_histogram
# ======================================================================
class TestPreviewHistogram:
    def test_mono_structure_and_counts(self, acq_vm):
        """A mono frame yields one (centres, counts) pair covering all sampled pixels."""
        frame = np.random.randint(0, 4096, size=(64, 64), dtype=np.uint16)
        result = acq_vm.compute_preview_histogram(frame)
        assert len(result) == 1
        centres, counts = result[0]
        assert len(centres) == len(counts) == 128
        # 4x4 subsampling of a 64x64 frame samples 16x16 = 256 pixels
        assert counts.sum() == frame[::4, ::4].size == 256

    def test_color_structure_and_counts(self, acq_vm):
        """An RGB frame yields three channel histograms, each summing to the sample count."""
        frame = np.random.randint(0, 4096, size=(64, 32, 3), dtype=np.uint16)
        result = acq_vm.compute_preview_histogram(frame)
        assert len(result) == 3
        expected = frame[::4, ::4, 0].size
        for centres, counts in result:
            assert len(centres) == len(counts) == 128
            assert counts.sum() == expected

    def test_uint16_uses_12bit_bins(self, acq_vm):
        """uint16 frames are binned over the full 12-bit range (0-4095)."""
        frame = np.zeros((16, 16), dtype=np.uint16)
        centres, _counts = acq_vm.compute_preview_histogram(frame)[0]
        assert centres[-1] > 255
        assert centres[-1] <= 4095

    def test_uint8_uses_8bit_bins(self, acq_vm):
        """uint8 frames are binned over the 8-bit range (0-255)."""
        frame = np.zeros((16, 16), dtype=np.uint8)
        centres, _counts = acq_vm.compute_preview_histogram(frame)[0]
        assert centres[-1] <= 255

    def test_extreme_values_do_not_crash(self, acq_vm):
        """All-zero and all-4095 frames produce valid histogram structures."""
        for value in (0, 4095):
            frame = np.full((16, 16), value, dtype=np.uint16)
            (centres, counts) = acq_vm.compute_preview_histogram(frame)[0]
            assert counts.sum() == 16

    def test_unsupported_ndim_returns_empty(self, acq_vm):
        """Frames that are neither 2-D nor 3-D yield an empty list."""
        assert acq_vm.compute_preview_histogram(np.arange(5)) == []


# ======================================================================
#  AcquisitionVM — service signal wiring (Bloque A)
# ======================================================================
class TestAcquisitionVMWiring:
    def test_service_error_forwarded(self, qtbot, acq_vm):
        """svc.error is wired to vm.error and forwards the message unchanged."""
        with qtbot.waitSignal(acq_vm.error, timeout=1000) as blocker:
            acq_vm.svc.error.emit("boom")
        assert blocker.args == ["boom"]

    def test_service_finished_forwarded(self, qtbot, acq_vm):
        """svc.finished is wired to vm.finished with (folder, skipped, total)."""
        with qtbot.waitSignal(acq_vm.finished, timeout=1000) as blocker:
            acq_vm.svc.finished.emit("data/run1", 2, 10)
        assert blocker.args == ["data/run1", 2, 10]

    def test_service_progress_forwarded(self, qtbot, acq_vm):
        """svc.progressChanged fans out to progressChanged/etaChanged/elapsedChanged."""
        percents, etas, elapseds = [], [], []
        acq_vm.progressChanged.connect(percents.append)
        acq_vm.etaChanged.connect(etas.append)
        acq_vm.elapsedChanged.connect(elapseds.append)
        acq_vm.svc.progressChanged.emit(50.0, 1.5, 5, 10, 12.0, 8.0)
        assert percents == [50.0]
        assert etas == [8.0]
        assert elapseds == [12.0]

    def test_service_log_formatted(self, qtbot, acq_vm):
        """svc.logReceived is reformatted as '[LEVEL] message' on statusTextChanged."""
        with qtbot.waitSignal(acq_vm.statusTextChanged, timeout=1000) as blocker:
            acq_vm.svc.logReceived.emit("warn", "low light")
        assert blocker.args == ["[WARN] low light"]


# ======================================================================
#  ResultsVM
# ======================================================================
class TestResultsVM:
    @staticmethod
    def _make_dataset(root, name: str, shape=(8, 6)):
        """Create output/<name>/<name>_height.npy under *root* and return the array."""
        folder = root / "output" / name
        folder.mkdir(parents=True)
        arr = np.random.rand(*shape).astype(np.float32)
        np.save(folder / f"{name}_height.npy", arr)
        return arr

    def test_refresh_datasets_lists_directories(self, qtbot, in_tmp_cwd):
        """refresh_datasets lists only the directories under ./output, sorted."""
        (in_tmp_cwd / "output" / "ds_b").mkdir(parents=True)
        (in_tmp_cwd / "output" / "ds_a").mkdir()
        (in_tmp_cwd / "output" / "not_a_dataset.txt").write_text("x")
        vm = ResultsVM()
        with qtbot.waitSignal(vm.datasetsChanged, timeout=1000) as blocker:
            vm.refresh_datasets()
        assert blocker.args == [["ds_a", "ds_b"]]

    def test_refresh_datasets_missing_output_is_empty(self, qtbot, in_tmp_cwd):
        """A missing ./output folder yields an empty dataset list, not an error."""
        vm = ResultsVM()
        with qtbot.waitSignal(vm.datasetsChanged, timeout=1000) as blocker:
            vm.refresh_datasets()
        assert blocker.args == [[]]

    def test_load_heightmap_success(self, qtbot, in_tmp_cwd):
        """Loading an existing heightmap emits heightmapLoaded with array metadata."""
        arr = self._make_dataset(in_tmp_cwd, "ds1", shape=(8, 6))
        vm = ResultsVM()
        with qtbot.waitSignal(vm.heightmapLoaded, timeout=1000) as blocker:
            vm.load_heightmap_for_dataset("ds1")
        loaded, shape, dtype_str, path = blocker.args
        assert shape == (8, 6)
        assert dtype_str == "float32"
        assert path.endswith("ds1_height.npy")
        np.testing.assert_allclose(np.asarray(loaded), arr)

    def test_load_heightmap_missing_emits_error(self, qtbot, in_tmp_cwd):
        """A dataset folder without a height file emits error, not heightmapLoaded."""
        (in_tmp_cwd / "output" / "empty_ds").mkdir(parents=True)
        vm = ResultsVM()
        with qtbot.waitSignal(vm.error, timeout=1000) as blocker:
            vm.load_heightmap_for_dataset("empty_ds")
        assert "Height file not found" in blocker.args[0]

    def test_load_heightmap_empty_name_emits_error(self, qtbot, in_tmp_cwd):
        """An empty dataset name is rejected with a clear error message."""
        vm = ResultsVM()
        with qtbot.waitSignal(vm.error, timeout=1000) as blocker:
            vm.load_heightmap_for_dataset("")
        assert blocker.args == ["No dataset selected."]

    def test_representative_pixels_regex_filtering(self, qtbot, in_tmp_cwd):
        """Valid *_pixel_y<Y>_x<X>.npy names are parsed; invalid names are ignored."""
        folder = in_tmp_cwd / "output" / "ds1"
        folder.mkdir(parents=True)
        valid = ["ds1_pixel_y0_x5.npy", "ds1_pixel_y1256_x3824.npy"]
        invalid = [
            "ds1_pixel_yA_x3.npy",  # non-numeric y
            "ds1_pixel_y1_x2.txt",  # wrong extension
            "ds1_height.npy",  # not a pixel file
            "ds1_pixel_x2_y1.npy",  # swapped coordinate order
        ]
        for name in valid + invalid:
            (folder / name).write_bytes(b"")
        vm = ResultsVM()
        with qtbot.waitSignal(vm.representativePixelsLoaded, timeout=1000) as blocker:
            vm.load_representative_pixels("ds1")
        points = blocker.args[0]
        assert [(x, y) for x, y, _path in points] == [(5, 0), (3824, 1256)]
        for _x, _y, path in points:
            assert path.startswith(str(folder))

    def test_representative_pixels_missing_folder(self, qtbot, in_tmp_cwd):
        """A non-existent dataset folder emits error instead of a pixel list."""
        vm = ResultsVM()
        with qtbot.waitSignal(vm.error, timeout=1000) as blocker:
            vm.load_representative_pixels("nope")
        assert "does not exist" in blocker.args[0]


# ======================================================================
#  AnalysisVM
# ======================================================================
class TestAnalysisVM:
    def test_progress_is_monotonic(self, qtbot, analysis_vm):
        """Out-of-order percentages from the service never make progress go backwards."""
        vm = analysis_vm
        percents: list[int] = []
        vm.progressChanged.connect(lambda p, _s, _d, _t, _eta: percents.append(p))
        vm._svc.runningChanged.emit(True)
        for p in (30, 10, 45, 44, 100):
            vm._svc.progressChanged.emit(p, "stage", p, 100)
        assert percents == [30, 30, 45, 45, 100]

    def test_eta_linear_model(self, qtbot, analysis_vm):
        """ETA follows elapsed * (100 - p) / p, and is 0 at 0% and 100%."""
        vm = analysis_vm
        vm._timer = _FakeElapsedTimer(elapsed_ms=2000)  # 2.0 s elapsed
        etas: list[float] = []
        elapseds: list[float] = []
        vm.etaChanged.connect(etas.append)
        vm.elapsedChanged.connect(elapseds.append)

        vm._svc.progressChanged.emit(50, "stage", 50, 100)
        assert elapseds[-1] == pytest.approx(2.0)
        assert etas[-1] == pytest.approx(2.0)  # 2.0 * 50/50

        vm._svc.progressChanged.emit(80, "stage", 80, 100)
        assert etas[-1] == pytest.approx(2.0 * 20 / 80)

        vm._svc.progressChanged.emit(100, "stage", 100, 100)
        assert etas[-1] == 0.0

    def test_running_true_resets_counters(self, qtbot, analysis_vm):
        """runningChanged(True) resets elapsed/ETA to 0 and re-allows low percentages."""
        vm = analysis_vm
        vm._timer = _FakeElapsedTimer(elapsed_ms=5000)
        vm._svc.progressChanged.emit(60, "stage", 60, 100)
        assert vm._last_percent == 60

        etas: list[float] = []
        elapseds: list[float] = []
        running: list[bool] = []
        vm.etaChanged.connect(etas.append)
        vm.elapsedChanged.connect(elapseds.append)
        vm.runningChanged.connect(running.append)
        vm._svc.runningChanged.emit(True)
        assert running == [True]
        assert elapseds == [0.0]
        assert etas == [0.0]
        assert vm._last_percent == 0

    def test_progress_info_label_format(self, qtbot):
        """ProgressInfoLabel renders elapsed/ETA as HH:MM:SS ('--:--:--' when unknown)."""
        label = ProgressInfoLabel()
        qtbot.addWidget(label)
        assert label.text() == "Elapsed: --:--:-- | ETA: --:--:--"
        label.set_elapsed(3661.0)
        label.set_eta(59.0)
        assert label.text() == "Elapsed: 01:01:01 | ETA: 00:00:59"
