# viewmodels/results_vm.py
from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal, Slot

from services.npy_loader import get_base_folder, list_datasets, load_height
from utils.analysis_paths import PIXEL_FILE_MARKER, PIXEL_FILE_REGEX


class ResultsVM(QObject):
    """
    ViewModel responsible for managing all result-loading operations,
    including heightmaps and representative pixel files under ./output/.

    Signals
    -------
    heightmapLoaded(arr, shape, dtype_str, path)
        Emitted when a heightmap has been successfully loaded.
        Parameters:
            arr       : NumPy array (typically mmap-backed)
            shape     : tuple (H, W)
            dtype_str : string representation of dtype
            path      : absolute path to the .npy file

    datasetsChanged(list[str])
        Emitted whenever the list of available datasets under ./output/
        is refreshed.

    representativePixelsLoaded(list[(x, y, npy_path)])
        Emits a list of representative pixel files extracted from the dataset.
        Each entry is a tuple: (x, y, full_path).

    error(str)
        Human‑readable error message for the UI.
    """

    # Signals
    heightmapLoaded = Signal(object, tuple, str, str)
    error = Signal(str)
    datasetsChanged = Signal(list)
    representativePixelsLoaded = Signal(list)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._heightmap = None
        self._heightmap_path: str = ""
        self._datasets: list[str] = []

    # ==================================================================
    #   DATASET LISTING
    # ==================================================================
    @Slot()
    def refresh_datasets(self) -> None:
        """
        Refresh the list of available datasets under ./output/
        and emit datasetsChanged.

        Any unexpected exception is routed to the error signal.
        """
        try:
            datasets = list_datasets()
            self._datasets = datasets
            self.datasetsChanged.emit(datasets)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Error listing datasets: {e}")

    # ==================================================================
    #   HEIGHTMAP LOADING
    # ==================================================================
    @Slot(str)
    def load_heightmap_for_dataset(self, dataset_name: str) -> None:
        """
        Load the heightmap for a dataset (usually mmap-backed via NumPy)
        and emit heightmapLoaded.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset directory under ./output.
        """
        if not dataset_name:
            self.error.emit("No dataset selected.")
            return

        try:
            arr, path = load_height(dataset_name)
            self._heightmap = arr
            self._heightmap_path = path

            self.heightmapLoaded.emit(arr, arr.shape, str(arr.dtype), path)

        except FileNotFoundError as e:
            self.error.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Error loading heightmap: {e}")

    # ==================================================================
    #   REPRESENTATIVE PIXELS LOADING
    # ==================================================================
    @Slot(str)
    def load_representative_pixels(self, dataset_name: str) -> None:
        """
        Scan output/<dataset_name>/ for representative pixel files matching:

            <name>_pixel_y<Y>_x<X>.npy

        Extract (x, y) from the filenames and emit representativePixelsLoaded
        with a list of tuples: (x, y, full_path).
        """
        folder = os.path.join(get_base_folder(), dataset_name)
        if not os.path.isdir(folder):
            self.error.emit(f"Dataset folder does not exist: {folder}")
            return

        try:
            files = sorted(
                f for f in os.listdir(folder) if PIXEL_FILE_MARKER in f and f.endswith(".npy")
            )

            points: list[tuple[int, int, str]] = []

            for fname in files:
                # Expected format: <prefix>_pixel_y1256_x3824.npy
                m = PIXEL_FILE_REGEX.match(fname)
                if m:
                    y = int(m.group(1))
                    x = int(m.group(2))
                    full_path = os.path.join(folder, fname)
                    points.append((x, y, full_path))

            self.representativePixelsLoaded.emit(points)

        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Error loading representative pixels: {e}")
