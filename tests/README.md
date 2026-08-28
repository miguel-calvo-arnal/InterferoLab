# InterferoLab test suite

**307 tests**, all runnable **without hardware** (camera and piezo are mocked;
Qt runs offscreen). Total duration: ~4–8 s.

## Running

```bash
.venv/bin/python -m pytest            # full suite
.venv/bin/python -m pytest tests/test_backend_errors.py -v   # one file
.venv/bin/python -m pytest -k cancel  # by keyword
```

Configuration lives in `pyproject.toml` (`[tool.pytest.ini_options]`) and
`tests/conftest.py` (project sys.path, `QT_QPA_PLATFORM=offscreen`, fixtures
`project_root` and `in_tmp_cwd`).

## Structure

| File | Tests | Covers |
|---|---|---|
| `test_bin12.py` | 14 | bin12 format: bit-exact pack/unpack round-trip, header, corruption |
| `test_camera_constants.py` | 6 | Bayer weights; anti-divergence regression against `config.hpp` |
| `test_analysis_paths.py` | 6 | Analysis naming convention (pixel regex, height suffix) |
| `test_config_manager.py` | 17 | Schema/sanitizing, corrupt JSON → `.bak`, atomic save |
| `test_session_log.py` | 6 | Session logging (format, idempotency) |
| `test_icons_utils.py` | 9 | resource_path, format_hms, icon loading |
| `test_image_formats.py` | 27 | bin12/TIFF/PNG channel parsers and format detection |
| `test_heightmap_processing.py` | 10 | Plane fit/subtraction (IQR, NaN, no mutation) |
| `test_pixel_signal.py` | 16 | Gaussian smoothing, refined centroid, baseline sigma |
| `test_npy_loader.py` | 8 | Dataset listing and mmap heightmap loading |
| `test_acquisition_controller.py` | 50 | Session with FAKE camera/piezo: sweep, validations, cancellation, skipped frames, `_wait_on_target`, compartmentalized disconnect |
| `test_acquisition_service.py` | 15 | Qt service: exclusions, signals, shutdown, double start |
| `test_piezo_slider.py` | 16 | Piezo slider: loop-free sync, move on release only, indicator, states |
| `test_superpixel_end_to_end.py` | 3 | Mocked sweep in superpixel mode → C++ analysis reconstructs (bin12 and png) |
| `test_backend_api.py` | 6 | pybind API: methods, errors, progress, raising callback |
| `test_backend_reconstruction.py` | 13 | M1–M4 on synthetic interferograms of known height; PNG vs bin12; metadata; pixel plots |
| `test_backend_errors.py` | 15 | Insufficient frames, corrupt headers, mixed dimensions, cancellation, atomic rename |
| `test_viewmodels.py` | 28 | numpy→pixmap, histogram, signal wiring, progress monotonicity, dataset regex |
| `test_panels_construction.py` | 11 | Offscreen construction of panels/MainWindow, warning-free QSS, heightmap with NaN |
| `test_ui_states.py` | 20 | Button state matrix, freezing during sweep/analysis, incomplete-dataset warning, tilt variants |
| `test_theme_and_shortcuts.py` | 11 | StatusBar, shortcuts, tooltip coverage (100%), palette, icons, unicode regression |

Shared helpers: `helpers_acquisition.py` (FakeCamera/FakePiezo/FakeSession) and
`helpers_backend.py` (generator of synthetic CSI interferograms of known height).

## Backend conventions documented by the tests (not bugs)

- The backend reports heights **inverted** with respect to the Z axis:
  `h_reported = z_max − h_real`.
- Progress percentages may jump during parallel reading (always monotonically
  non-decreasing).

## Real bugs caught by this suite

Both are **PySide6 lifecycle** failures, invisible until the library version
changes: recreating the venv now and then and re-running the suite is part of
the safety net, not a chore.

1. **GC race vs `deleteLater`** in the QThread/worker lifecycle
   (non-deterministic SIGBUS with PySide6 6.11): fixed in
   `services/acquisition_service.py` and `services/analysis_service.py` by
   retaining worker/thread references in a "graveyard" until the next start.
   If you touch that lifecycle, run the suite several times in a row (the
   failure was non-deterministic).

2. **`QTimer.singleShot(0, self._method)` on an already-destroyed widget**
   (August 2026): deferred calls from `widgets/HeightmapView.py` stayed queued
   after the widget died, and the callback operated on an already-deleted C++
   object (`libshiboken: Internal C++ object ... already deleted`). Surfaced
   in `test_results_panel_constructs` when moving from PySide6 6.11.0 to
   6.11.2. Fixed by passing `self` as the context object, which makes Qt
   cancel the pending call when the widget is destroyed.
