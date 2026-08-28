# views/MainWindow.py
import logging
import os

from PySide6.QtCore import Slot
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget

from utils.config_manager import load_config, save_config

from .AcquisitionPanel import AcquisitionPanel
from .ProcessingPanel import ProcessingPanel
from .ResultsPanel import ResultsPanel

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    App main window.

    It consists of three tabs:
      - Data acquisition
      - Data processing
      - Results analysis

    When processing data it prompts to see the results automatically
    """

    # Fraction of the available screen (width and height) the window takes at
    # startup. Adaptive: on small displays the window shrinks accordingly
    # (bounded below by the layout's own minimum size).
    _SCREEN_FRACTION = 0.80

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Size the window to a fraction of the AVAILABLE screen area (excludes
        # taskbars/docks), so it adapts to small laptop displays as well.
        screen = QGuiApplication.primaryScreen()
        geom = screen.availableGeometry()

        w = int(geom.width() * self._SCREEN_FRACTION)
        h = int(geom.height() * self._SCREEN_FRACTION)
        self.resize(w, h)

        # Centre window
        rect = self.frameGeometry()
        rect.moveCenter(geom.center())
        self.move(rect.topLeft())

        self.setWindowTitle("InterferoLab")

        # Tabs
        self._tabs = QTabWidget(self)
        self.setCentralWidget(self._tabs)
        # Acquisition tab
        self._acq_panel = AcquisitionPanel(self)
        self._tabs.addTab(self._acq_panel, "Acquisition")
        # Processing tab
        self._proc_panel = ProcessingPanel(self)
        self._tabs.addTab(self._proc_panel, "Processing")
        # Results tab
        self._results_panel = ResultsPanel(self)
        self._tabs.addTab(self._results_panel, "Results")

        # Connect signal to load results after analysis
        self._proc_panel.analysisFinishedAndAccepted.connect(
            self._on_analysis_finished_and_accepted
        )

        # Status bar: transient messages + permanent hover x/y/z readout
        self._status_hover = QLabel("", self)
        self.statusBar().addPermanentWidget(self._status_hover)
        self._results_panel.hoverInfoChanged.connect(self._status_hover.setText)
        self.statusBar().showMessage(
            "Ready — F5 refresh datasets, Ctrl+Return start, Esc cancel, Ctrl+1/2/3 switch tab",
            8000,
        )

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Load and apply persistent configuration. A bad config must never
        # prevent the application from starting, so each panel is guarded.
        cfg = load_config()
        if "acq" in cfg:
            try:
                self._acq_panel.apply_saved_config(cfg["acq"])
            except Exception as e:  # noqa: BLE001
                log.warning("Could not apply saved acquisition config: %s", e)
        if "proc" in cfg:
            try:
                self._proc_panel.apply_saved_config(cfg["proc"])
            except Exception as e:  # noqa: BLE001
                log.warning("Could not apply saved processing config: %s", e)

    # ------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------
    def _setup_shortcuts(self) -> None:
        """F5 refresh, Ctrl+Return start, Esc cancel, Ctrl+1/2/3 tabs."""
        QShortcut(QKeySequence("F5"), self).activated.connect(self._on_refresh_shortcut)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._on_start_shortcut)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self._on_cancel_shortcut)
        for i in range(self._tabs.count()):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda i=i: self._tabs.setCurrentIndex(i))

    @Slot()
    def _on_refresh_shortcut(self) -> None:
        """F5: refresh the dataset list of the active tab."""
        w = self._tabs.currentWidget()
        if w is self._proc_panel:
            self._proc_panel.refresh_datasets()
        elif w is self._results_panel:
            self._results_panel.refresh_datasets()
        else:
            return
        self.statusBar().showMessage("Dataset list refreshed", 3000)

    @Slot()
    def _on_start_shortcut(self) -> None:
        """Ctrl+Return: start the sweep or the analysis, depending on the tab."""
        w = self._tabs.currentWidget()
        if w is self._acq_panel:
            self._acq_panel.trigger_start()
        elif w is self._proc_panel:
            self._proc_panel.trigger_start()

    @Slot()
    def _on_cancel_shortcut(self) -> None:
        """Esc: cancel the running operation of the active tab."""
        w = self._tabs.currentWidget()
        if w is self._acq_panel:
            self._acq_panel.trigger_cancel()
        elif w is self._proc_panel:
            self._proc_panel.trigger_cancel()

    # ------------------------------------------------------------
    # Load heightmap after processing
    # ------------------------------------------------------------
    @Slot(dict)
    def _on_analysis_finished_and_accepted(self, result: dict) -> None:
        """
        Gets dictionary with the results and loads the heightmap in the
        results tab.
        """
        heightmap_path = result.get("heightmap")
        if not heightmap_path:
            QMessageBox.warning(
                self,
                "Heightmap not found",
                "Data processing did not output a heightmap (?)",
            )
            return

        # Extract dataset_name --> output/<dataset>/<dataset>_height.npy
        dataset_name = os.path.basename(os.path.dirname(heightmap_path))

        # Switch to results tab
        self._tabs.setCurrentWidget(self._results_panel)

        # Select dataset in the combo box (after reloading list, so it is found)
        self._results_panel.refresh_datasets()
        self._results_panel.select_dataset(dataset_name)

        # Load heightmap
        try:
            self._results_panel.load_heightmap(dataset_name)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Error loading heightmap",
                f"Failed to load '{dataset_name}'.\n\n{e}",
            )

    # ------------------------------------------------------------
    # Persistent configuration
    # ------------------------------------------------------------
    def closeEvent(self, event) -> None:
        """Save configuration and ensure child panels are cleaned up."""
        # Ask for confirmation if a sweep is still running: closing will
        # cancel it and shut down the acquisition worker threads.
        if self._acq_panel.vm.is_running():
            reply = QMessageBox.question(
                self,
                "Sweep in progress",
                "An acquisition sweep is still running.\n\nCancel it and quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        # Same guard for a running C++ analysis: closing would otherwise
        # destroy its live QThread (Qt aborts) without ever cancelling it.
        if self._proc_panel.is_running():
            reply = QMessageBox.question(
                self,
                "Analysis in progress",
                "An analysis is still running.\n\nCancel it and quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        cfg = {
            "acq": self._acq_panel.get_config(),
            "proc": self._proc_panel.get_config(),
        }
        save_config(cfg)
        # Explicitly close both panels so worker threads are stopped and
        # hardware is disconnected (each panel's closeEvent → shutdown)
        self._acq_panel.close()
        self._proc_panel.close()
        super().closeEvent(event)
