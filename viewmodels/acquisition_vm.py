# viewmodels/acquisition_vm.py
from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from services.acquisition_service import AcquisitionService


class AcquisitionVM(QObject):
    """
    ViewModel for the acquisition system (QThread-based sweep + threaded preview).

    Responsibilities:
    - Bridge signals between AcquisitionService and UI
    - Provide simple API (connect, start, cancel, request_preview, move_to)
    - Convert NumPy preview frames to QPixmap
    - Compute preview histogram data (the view only paints it)
    """

    # Signals exposed to the UI ------------------------------------------
    progressChanged = Signal(float)  # percent [0..100]
    statusTextChanged = Signal(str)  # logs
    previewFrame = Signal(object, float)  # numpy frame + z
    runningChanged = Signal(bool)  # sweep currently running?
    movingChanged = Signal(bool)  # manual piezo move in progress?
    positionChanged = Signal(float)  # real piezo Z position, µm
    finished = Signal(str, int, int)  # output folder, skipped frames, total frames
    error = Signal(str)

    etaChanged = Signal(float)  # seconds
    elapsedChanged = Signal(float)  # seconds

    # ------------------------------------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)

        self.svc = AcquisitionService(self)

        # Precomputed histogram bin edges (129 edges → 128 bins)
        self._bins_8 = np.linspace(0, 255, 129)
        self._bins_12 = np.linspace(0, 4095, 129)

        # Forward signals from service -> VM -> UI
        self.svc.progressChanged.connect(self._on_progress)
        self.svc.previewFrame.connect(self._on_preview_frame)
        self.svc.logReceived.connect(self._on_log)
        self.svc.runningChanged.connect(self.runningChanged)
        self.svc.movingChanged.connect(self.movingChanged)
        self.svc.positionChanged.connect(self.positionChanged)
        self.svc.finished.connect(self.finished)
        self.svc.error.connect(self.error)

    # ------------------------------------------------------------------
    # Public API for UI
    # ------------------------------------------------------------------
    def connect_piezo(self, serial: str, dll: str) -> bool:
        return self.svc.connect_piezo(serial, dll)

    def connect_camera(self) -> bool:
        return self.svc.connect_camera()

    def disconnect_all(self) -> bool:
        return self.svc.disconnect_all()

    def apply_config(self, cfg: dict) -> None:
        self.svc.apply_config(cfg)

    def apply_and_start(self, cfg: dict) -> bool:
        self.apply_config(cfg)
        return self.start()

    def start(self) -> bool:
        """Start sweep in background thread."""
        return self.svc.start()

    def cancel(self) -> None:
        self.svc.cancel()

    def is_running(self) -> bool:
        return self.svc.is_running()

    def is_moving(self) -> bool:
        return self.svc.is_moving()

    def is_previewing(self) -> bool:
        return self.svc.is_previewing()

    def shutdown(self) -> None:
        """Cancel any activity, stop worker threads and disconnect hardware."""
        self.svc.shutdown()

    def move_to(self, z: float) -> None:
        self.svc.move_to(z)

    def request_preview(self) -> None:
        """Ask service to capture a single preview frame in background."""
        self.svc.capture_preview()

    # ------------------------------------------------------------------
    # Preview histogram
    # ------------------------------------------------------------------
    def compute_preview_histogram(self, frame: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Compute the preview histogram for a frame (subsampled 4×4 for speed).

        Returns one (bin_centres, counts) tuple per channel: a single tuple
        for mono frames, three (R, G, B) for colour frames, and an empty list
        for unsupported shapes.
        """
        arr = np.asarray(frame)
        if arr.ndim == 2:
            channels = [arr[::4, ::4]]
        elif arr.ndim == 3:
            sub = arr[::4, ::4, :3]
            channels = [sub[..., i] for i in range(3)]
        else:
            return []

        result: list[tuple[np.ndarray, np.ndarray]] = []
        for ch in channels:
            bins = self._bins_12 if (ch.dtype == np.uint16 or ch.max() > 255) else self._bins_8
            counts, edges = np.histogram(ch, bins=bins)
            centres = 0.5 * (edges[:-1] + edges[1:])
            result.append((centres, counts))
        return result

    # ------------------------------------------------------------------
    # NumPy -> QPixmap conversion
    # ------------------------------------------------------------------
    def numpy_to_pixmap(self, frame: Any) -> QPixmap:
        arr = np.asarray(frame)

        # ---------- FAST MONO PREVIEW ----------
        if arr.ndim == 2:
            # Convert 12-bit (0-4095) directly to 8-bit by shifting
            img8 = (arr >> 4).astype(np.uint8)

            h, w = img8.shape
            qimg = QImage(img8.data, w, h, img8.strides[0], QImage.Format_Grayscale8).copy()
            return QPixmap.fromImage(qimg)

        # ---------- FAST RGB PREVIEW ----------
        if arr.ndim == 3 and arr.shape[2] >= 3:
            # Take RGB only
            rgb = arr[..., :3]

            # Normalize 12-bit to 8-bit PER CHANNEL by shift
            img8 = (rgb >> 4).astype(np.uint8)

            img8 = np.ascontiguousarray(img8)
            h, w, _ = img8.shape
            qimg = QImage(img8.data, w, h, img8.strides[0], QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)

        raise ValueError(f"Unsupported preview frame shape: {arr.shape}")

    # ------------------------------------------------------------------
    # Callbacks from service
    # ------------------------------------------------------------------
    @Slot(float, float, int, int, float, float)
    def _on_progress(self, percent, z, idx, total, elapsed, eta):
        self.progressChanged.emit(percent)
        self.etaChanged.emit(eta)
        self.elapsedChanged.emit(elapsed)
        # z is the real position of the last acquired frame: reuse it so the
        # position indicator keeps tracking the piezo during a sweep.
        self.positionChanged.emit(float(z))

    @Slot(object, float)
    def _on_preview_frame(self, frame, z: float) -> None:
        """Forward a preview frame and reuse its z as a position update."""
        self.previewFrame.emit(frame, z)
        self.positionChanged.emit(float(z))

    @Slot(str, str)
    def _on_log(self, level, message):
        self.statusTextChanged.emit(f"[{level.upper()}] {message}")
