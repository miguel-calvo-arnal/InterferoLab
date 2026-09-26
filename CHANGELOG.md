# Changelog

## Audit and fixes (July 2026)

Executive summary of everything applied after a full project audit (blocks
landed on 2026-07-24; tests, environment and packaging over the following
days). The detailed audit records are kept outside the repo; the general
entry point to the project is `README.md`. Git history: one commit per block.

```
486c18d Bloque C: rendimiento — reconstrucción 3.5-4.2x más rápida
fab23ab Bloque E: tema visual y UX
23bddda Bloque B: robustez del backend C++
2c85db1 Bloque D: claridad y mantenibilidad Python
1d759c0 Bloque A: robustez crítica Python
37fc5d5 Bloque D1: ruff + normalización de indentación
8965e5f Estado inicial pre-auditoría
```

---

### Block A — Python robustness (acquisition)

- Preview and piezo-motion errors now reach the UI (the service's `error`
  signal was never connected: a failing preview froze silently).
- Mutual exclusion between sweep and preview: two threads can no longer enter
  the camera SDK at once.
- Orderly shutdown: cancels the sweep, joins the QThreads and disconnects the
  hardware; asks for confirmation if a sweep is running. `atexit` as backup.
- `connect_camera` no longer leaks the handle when a connection fails halfway
  ("device in use").
- `disconnect_all` compartmentalized: a camera-close failure no longer leaves
  the piezo with servo ON; no silent `except: pass`.
- Config validated against a type/range schema; corrupt JSON is preserved as
  `app_config.json.bak` (it used to be lost and could prevent startup); atomic
  saves; path anchored to the project, not the cwd.
- Config frozen during a sweep (switching Mono/Color mid-sweep used to produce
  a silently unusable mixed dataset).
- Dropped frames counted and reported: "Incomplete dataset: N/M" warning
  before analysis.
- Piezo motion cancellation: ~0.1 s (previously up to 30 s blocked in
  waitontarget).
- If the requested exposure is not actually applied by the camera, the sweep
  aborts (related to item 1 of `TODO.md`).
- Standard `logging` (`utils/session_log.py`) replaces prints and a
  hand-rolled logger.

### Block B — C++ robustness (analysis backend)

- **Crashes eliminated** (all reachable from the GUI before): datasets of 1–45
  images (out-of-bounds read in the Gaussian filter; now per-method
  minimum-frame validation + iterative reflect), method 4 with a single image
  (wild write), cancelling during filename parsing (3 UB spots), a Python
  callback throwing with a joinable thread, and a full disk inside an OpenMP
  region (`std::terminate`).
- Safe re-analysis: output is written to `output/<name>.partial` and renamed
  on completion — a failure no longer destroys the previous result.
- `save_npy` verifies writes (a full disk no longer produces a truncated
  `.npy` "successfully").
- bin12 header validated and dimensions checked file by file (mixed datasets
  produce a clear error instead of garbage data).
- RAII for `FILE*`/`TIFF*` (no descriptor leaks on error paths).
- pybind callbacks by `const&` (no more incref/decref without the GIL) and a
  thread-safe FFTW planner.
- Portable binary: `-march=x86-64-v2` by default (previously v3 → SIGILL on
  CPUs without AVX2); CMake option `INTERFEROLAB_NATIVE_ARCH=ON` to build with
  `-march=native` locally.
- **Method 4 fixed**: the band is now estimated after baseline removal (with
  intensity drift it produced false heights: 0.315 µm where the true value was
  0.900 → now 0.885). Saturated/flat pixels fall to the centre of the range in
  all 4 methods (M3/M4 used to push them to the extreme, contaminating
  statistics).
- Correct analytic Hilbert filter for odd N; `"cancelled"` key in the result.
- Backend recompiled for the project venv (cp313 at the time; see "Python
  environment").

### Block C — Performance

Measured on 100 PNGs of S1F5 (1936×1216, 8 threads), compared bit-for-bit
against the baseline:

| Method | Before | After | Speedup |
|---|---|---|---|
| M1 | 30.1 s | 7.1 s | 4.2× |
| M2 | 24.5 s | 7.0 s | 3.5× |
| M3 | 23.3 s | 6.1 s | 3.8× |
| M4 | 31.3 s | 8.2 s | 3.8× |

- Vectorized Gaussian convolution (SIMD confined to the loop, no global
  fast-math).
- One full memory pass removed (baseline fused into the compute phase; M2/M3
  skip the gather entirely) — bit-exact.
- Parallel read/decode across files + block reads for bin12 — bit-exact.
- Bayer merge (`_bayer_to_superpixel`) 1.7× faster with bit-exact output
  (benefits both sweep and preview); `process_superpixel.py` vectorized
  (identical bytes).
- Single tolerance: SIMD reassociation (p99.9 ≤ 2·10⁻⁵ µm ≈ 0.02 nm,
  documented).

### Block D — Clarity and maintainability

- `ruff` configured in `pyproject.toml`; all code formatted (end of the
  tabs/spaces mix that corrupted external edits).
- Single sources of truth: `utils/camera_constants.py` (Bayer weights,
  previously in 4 places), `utils/bin12.py` (format, previously 4
  implementations; extraction verified bit-for-bit), `utils/analysis_paths.py`
  (naming convention shared with the C++).
- Business logic out of the views: `services/image_formats.py`,
  `services/heightmap_processing.py`, `services/pixel_signal.py`; preview
  histogram in the ViewModel; reconstruction methods in
  `services/analysis_service.py`.
- Encapsulation: public `ResultsPanel.refresh_datasets()`; `__init__.py` in
  packages; normal imports (no `importlib`); dead `BASE_FOLDER` removed.
- New `scripts/` with `process_superpixel.py` and `flat_noise_analysis.py`
  (restructured into functions + `main()`, same CLI).
- False docstrings fixed (C++ cancellation does work); deduplications
  (`ProgressInfoLabel`, button/row factories, tilt and title helpers).

### Block E — Visual theme and UX

- Complete light theme (Fusion + `resources/styles.qss`), palette derived from
  matplotlib blue `#1f77b4` with WCAG AA contrast verified (including the Tilt
  button: 2.78:1 before, ≥5:1 now). All 14 inline styles migrated to QSS
  (`primary/danger/success` variants via dynamic properties).
- `widgets/theme.py`: single palette for pyqtgraph (no more duplicated
  constants); plots keep their matplotlib/viridis identity.
- Tooltips on the ~32 controls that lacked them (units, ranges, shortcuts).
- Feedback: wait cursor and control locking during loads, analysis and sweeps.
- StatusBar with hover coordinates; F5 / Ctrl+Return / Esc / Ctrl+1/2/3
  shortcuts.
- 21 coherent SVG icons (24×24 outline) on the main buttons; custom chevrons;
  legend in the preview histogram (neutral curve in mono, previously green);
  `app_icon.png` 967 KB → 25 KB; unified copy (single µ, sentence case,
  "analyzing").

### Verification performed

- M1–M4 heightmaps compared bit-for-bit against baselines at every C++ step
  (except the documented SIMD tolerance); non-regression tests (1 image, 10
  images, cancellation) on every build; bit-exact round-trips for bin12 and
  the Bayer merge.
- `ruff check` clean; `py_compile` of every module; QSS with no Qt warnings.
- Final integrated check: full application launched (offscreen) with the new
  `.so` and all blocks together, screenshot reviewed.

### Test suite (added 2026-07-25)

`tests/` with **266 tests** runnable without hardware
(`.venv/bin/python -m pytest`, ~13 s; full structure in `tests/README.md`):

- **utils/config (56)**: bit-exact bin12 round-trip, camera constants
  (anti-divergence regression against config.hpp), corrupt config → .bak,
  atomic saves, session_log.
- **services/acquisition (112)**: format parsers, plane fitting, pixel signal,
  npy_loader, and full AcquisitionSession/Service with **mocked** camera and
  piezo (sweep validations, skipped frames, cancellation, _wait_on_target,
  compartmentalized disconnect, exclusions and the service's Qt signals).
- **C++ backend (34)**: synthetic interferograms with known height → M1–M4
  recover the plateaus (error ≤ 0.011 µm); PNG vs bin12 bit-identical; clean
  errors (too few frames, corrupt headers, mixed dimensions); cancellation and
  atomic rename.
- **UI/viewmodels (64)**: signal wiring, progress monotonicity, offscreen
  construction of all 4 containers, QSS without warnings, button state matrix,
  theme variants, shortcuts, and 100% tooltip coverage (42/42) as a test.

The tests found a **real bug**: a GC vs deleteLater race in the QThread/worker
lifecycle (non-deterministic SIGBUS with PySide6 6.11 + Python 3.14). Fixed in
both services by retaining worker/thread references until the next start
("graveyard"); suite stable across repeated runs.

### Python environment (2026-07-25)

Target version at the time: **Python 3.13** (the original binaries and the
Windows `.pyd` were cp313). See `README.md` for the current, simpler setup —
since `requirements.txt` pins exact versions, no resolver workarounds are
needed.

**PyInstaller packaging** (2026-07-28): `interferolab.spec` updated after the
refactor (adds the `utils` package; only `backend/acquisition` is copied as
data — editable without repackaging — instead of all of `backend/`, which
would drag the CMake tree along). Distributable builds are archived in
`releases/` as `InterferoLab-<date>-<platform>` archives (outside git;
`build/` and `dist/` are regenerable). Launch the app from the folder where
`output/` and `data/` should live (they resolve against the cwd).

### Documentation reorganization (2026-07-28)

Full inventory and cleanup of the project's documentation folder (369 →
224 MB, outside git). Highlights relevant to the repo:

- `docs/Quantum efficiency and IR filter/` (the scientific derivation of
  the Bayer weights, cited from `utils/camera_constants.py` and `config.hpp`)
  became **versioned in git**; the rest of the documents stay outside.
- The TIFF→bin12 converter was ported to `scripts/convert_to_bin12.py` on top
  of `utils.bin12` (byte-identical output verified).
- The LaTeX reports were checked against the code by independent reviewers,
  updated where stale (M3/M4 descriptions, signal names) and recompiled; the
  camera-diagnostic TIFFs were losslessly recompressed and verified
  bit-perfect. These documents live outside the repo.

### Features + slimmer package (2026-07-28)

- **Piezo slider**: next to Manual Z, 0–100 µm with 0.01 resolution; dragging
  moves the piezo (on release), and as an indicator it tracks the real
  position (new positionChanged signal + the progress and preview z values).
  Disabled when disconnected; read-only indicator during a sweep.
- **"Mono (superpixel)"**: a third channel mode that directly saves the
  weighted merge + 2×2 binning (mono at quarter resolution: stacks 4× smaller
  than mono, 12× smaller than color) — the acquisition-side equivalent of
  `scripts/process_superpixel.py`. Preview converted, config validated, and
  verified end-to-end against the C++ backend (bin12 and png, bit-identical).
- Test suite: 266 → **307** (41 new).
- **PyInstaller 1.4 GB → 663 MB** (tar.gz 516 → 259 MB): removed the
  indiscriminate PySide6/pyqtgraph collect_* calls, ~45 excludes verified one
  by one (WebEngine, Qml, 3D, Multimedia, ...), Windows-compatible
  post-Analysis pattern filter, GTK chain out, strip on POSIX, and
  opencv-python → opencv-python-headless in requirements (cv2 only does image
  I/O). What remains are real dependencies that pylablib requires at import
  time (llvmlite, scipy, pandas). Executable verified (SVG plugins, stable
  startup, clean log).

### Left out of scope of this round

1. **Lab validation** of the real-hardware paths: camera/piezo
   connect/disconnect, sweep/preview exclusion, motion cancellation, exposure
   verification. Verified by static analysis, not in execution.
2. ~~Recompile the **Windows** `.pyd` with the updated CMake~~ — resolved: see
   "Windows/MSVC build support".
3. Ideas noted but not applied on risk/benefit grounds: pixel-major chunk
   layout, `FFTW_MEASURE`/r2c (notes in the code), exposing the Bayer weights
   via pybind.
4. ~~No automated test suite~~ — resolved: see "Test suite".

---

## Windows/MSVC build support (2026-08-26)

Closes item 2 above. The C++ backend now builds and packages on Windows.

- **OpenMP**: MSVC implements OpenMP 2.0 with `/openmp`, which rejects
  `#pragma omp simd` (error C7660 in `filters.cpp`). `CMakeLists.txt` now uses
  `/openmp:experimental` on MSVC only — same `vcomp` runtime, auto-linked, and
  without linking `OpenMP::OpenMP_CXX`, which would also inject `/openmp` and
  trigger a D9025 in every translation unit.
- **PyInstaller**: the `.pyd` ships as a DATA file and PyInstaller does not
  analyze binary dependencies of datas, so nothing pulled in the vcpkg DLLs
  (OpenCV, FFTW, libtiff and their chain) or the MSVC runtime (`MSVCP140.dll`,
  `VCOMP140.DLL`). Without them the frozen app died with "ImportError: DLL
  load failed while importing analysis_backend". The `.spec` now copies them
  into `_internal/`: the package is self-contained and target machines do not
  need the VC++ Redistributable.
- The globs cover both `build/Release/` (MSVC multi-config generators) and
  `build/` (Makefiles/Ninja), and are guarded per platform so sharing the
  project folder between a Linux host and a Windows VM breaks neither build.
- `.pyd` recompiled (cp313): 341,504 → 355,328 bytes.
- Operational detail and failure diagnostics in
  `backend/analysis/compile_notes.txt`.

Verified on Windows: the backend compiles, the frozen executable starts, and
analysis works.

---

## Environment, packaging and Qt lifecycle (2026-08-26)

**Lifecycle bug in `HeightmapView`.** The four
`QTimer.singleShot(0, self._method)` calls stayed queued even if the widget
died before the timer fired, and the callback then operated on an already
destroyed C++ object. Unnoticeable with PySide6 6.11.0; moving to 6.11.2 broke
`test_results_panel_constructs` (`libshiboken: Internal C++ object ... already
deleted`). Fixed by passing `self` as the context object, which makes Qt
cancel the pending call. Applied to all four, not just the two that failed.
Suite verified with 6.11.2 **and** 6.11.0: 307 green on both.

**`requirements.txt` with pinned versions.** Open ranges (`PySide6>=6.6`,
`numpy>=1.26`, ...) meant every reinstall brought whatever PyPI had that day:
recreating the venv jumped PySide6 6.11.0 → 6.11.2, numpy 2.4.4 → 2.5.2 and
numba 0.65 → 0.67 without warning — and that jump is what surfaced the bug
above. The pinned set was validated by installing from scratch into a clean
venv (Python 3.13.11): 307 tests green. Direct dependencies are pinned;
transitive ones remain free.

**Linux builds in `build-linux/`.** The project folder is shared between a
Linux host and a Windows VM, and a single `backend/analysis/build/` cannot
hold both CMake caches: cmake rejects a cache generated for another source
path and another generator. Moreover `build/Release/` holds the 14 vcpkg DLLs
that `interferolab.spec` puts into the Windows bundle, so wiping `build/` to
reconfigure was not an option. Windows now uses `build/` and Linux
`build-linux/`; the spec looks for the binary in both and `.gitignore` covers
both.

**Release scripts** (`scripts/build_release.sh` and
`scripts/build_release.ps1`). They run the whole chain in one command: build
the backend, copy the binary next to `backend/analysis/`, **verify the module
imports before** launching PyInstaller (an ABI mismatch fails in seconds
instead of five minutes in), freeze with the spec and package the app into
`releases/` (`.tar.gz` on Linux, `.zip` on Windows). They strip the runtime
residue (`app_config.json` and `logs/`) that the July `.tar.gz` accidentally
carried. The Linux script is tested end to end (663 MB in `dist/`, 259 MB
compressed).

---

## Python 3.14, Windows venv and release-chain fixes (2026-08-27)

**Rebuild for Python 3.14.** The system Python (rolling distro) moved from
3.13 to 3.14 and the backend `.so` (`cpython-313`) stopped importing
(`ModuleNotFoundError: analysis_backend` while collecting 4 test modules).
The venv was recreated with 3.14.7 — the pinned `requirements.txt` installs
unchanged, with no resolver workarounds — and the backend recompiled
(`cpython-314`). Suite verified: 307 tests, 306 green + 1 pre-existing skip
(`test_pixel_regex_matches_real_output_files`, which only runs when pixel
files exist in `output/`), ~5 s. A stale CMake cache in `build-linux/`
pointing at a previous project location was cleared. The README now documents
that the backend binary is tied to the venv's Python version.

**Windows venv in `.venv-win\`.** The first real run of `build_release.ps1`
on the VM showed that `.venv\` cannot be both the Linux and the Windows venv
(the folder is shared). The script now looks for `.venv\Scripts\python.exe`
and falls back to `.venv-win\Scripts\python.exe`, with `-VenvDir <path>` as an
explicit alternative (e.g. a venv on the VM's local disk if the shared drive
is slow). The error message explains how to create it.

**`scripts/build_release.bat`.** Double-click launcher for Windows: calls the
`.ps1` with `-ExecutionPolicy Bypass`, works from any directory (`%~dp0`) and
ends with `pause` so the output stays readable. Avoids the relative-path error
of launching the `.ps1` by hand from inside `scripts\`.

**`.ps1` import check with the vcpkg DLLs.** The first full run on the VM
reached step 3 and failed with `ImportError: DLL load failed`: the check did a
bare `import`, and since Python 3.8 PATH no longer resolves DLLs, so the
`.pyd` could not find the vcpkg DLLs in `build\Release\`. The check now
declares `build\Release\` and `dist\win\InterferoLab\_internal\` with
`os.add_dll_directory`, just like `main.py` does in development. (The cmake
warning about `CMAKE_TOOLCHAIN_FILE` being unused is harmless: the `build\`
cache already carried the vcpkg toolchain from its first configuration.)

**Windows PyInstaller in `dist\win\` and `build\win\`.** Step 4 failed on the
VM while trying to empty `dist\InterferoLab`: it held the **Linux** bundle,
whose symlinks cannot be deleted from Windows through the vboxsf shared folder
(`Remove-Item: incorrect parameter`). Mirroring the `build/` vs `build-linux/`
split for the CMake caches, Windows PyInstaller now uses
`--distpath dist\win` and `--workpath build\win`: Windows-only subfolders that
still hang from `dist/` and `build/`. The final zip still lands in
`releases\`.

**Reorganization for the public GitHub mirror.** Every folder reviewed and
classified (project / documentation / regenerable) ahead of publishing:

- **Out of the git index** (still on disk): the compiled backend binaries
  (`.so`/`.pyd` — distributed via `releases/`, not git), the proprietary
  Physik Instrumente SDK (`API/PI/` — not redistributable; setup instructions
  in the new `API/README.md`), `app_config.json` (runtime residue: the app
  starts with defaults and regenerates it, and it carried the piezo serial
  number), the July 2026 audit records, and the local noise-analysis results
  (large PDFs).
- **`.gitignore` rewritten**: obsolete blocks removed, `data/` ignored
  entirely, and `.venv-win/`, the backend binaries, `API/*`,
  `app_config.json` and the personal project-management files added.
- **Into the index**: `scripts/build_release.bat`, `API/README.md` and
  `install/README.md` (MSVC runtime deployment notes, cited from
  `compile_notes.txt`).
- **Renamed to English conventions**: `CAMBIOS.md` → `CHANGELOG.md`,
  `PENDIENTE.txt` → `TODO.md`, `instalacion/` → `install/`,
  `scripts/procesar_superpixel.py` → `scripts/process_superpixel.py`, and the
  "Quantum Eficiency" typo fixed to "Quantum efficiency" (code citations
  updated). Documentation translated to English.
- **README**: the repository section no longer publishes internal
  infrastructure details.

## Multi-agent review fixes (2026-08-28)

A 27-agent review (4 reviewers + adversarial verification per finding) ran
before publishing the mirror; 20 findings were confirmed and all were fixed:

**Robustness (Python)**
- Closing the app during a running C++ analysis no longer aborts the process:
  `MainWindow.closeEvent` now guards the analysis like it guards the sweep
  (confirmation dialog), and a new `AnalysisService.shutdown()` cancels the
  backend and stops the worker thread with a bounded wait.
- `analysis_backend` is now imported guardedly: on a fresh clone without the
  compiled backend the app starts in acquisition mode with a clear error on
  analysis start, instead of a startup traceback (the documented
  `FALLBACK_METHODS` path is now actually reachable).
- A cancelled sweep now reports the unattempted frames as skipped, so the
  incomplete-dataset warning fires and the log says "partial dataset (N/M)"
  instead of a normal finish.
- Manual piezo moves are now cancellable (a `CancelFlag` threads through
  `move_to`), and `AcquisitionService.shutdown()` cancels them — previously a
  legal 30 s move could outlive the 12 s shutdown wait and the hardware was
  closed under the worker's feet.
- The displayed heightmap is loaded eagerly instead of as a live mmap: on
  Windows the mmap kept the file locked and broke re-analysis of the same
  dataset (the backend's publish step could not replace the folder).
- The pixel-plot loader tolerates corrupt/empty `.npy` files (error title
  instead of an exception escaping the Qt slot).

**Robustness (C++ backend)**
- The Z-position regex now matches the FILENAME only: a digit + "um" in any
  parent folder name (e.g. `scan_20um/`) used to hijack every frame's
  position and produce a silently flat height map. A sanity check now also
  rejects stacks where all parsed positions are identical.
- The publish step uses two renames instead of remove_all+rename: a locked
  previous output (Windows) can no longer leave the old result half-deleted
  with the new one stranded; the error now says what to close and the old
  result stays intact.
- The progress-weight sampling phase validates each sampled file's
  dimensions like the main reader does, closing a heap over-read (UB) on
  datasets with mixed image sizes.
- `result["cancelled"]` is derived from the authoritative outcome (empty
  output path) instead of re-reading the racy global flag.
- The per-chunk compute thread's misleading "thread_locals survive across
  chunks" comment now documents the real behaviour (OpenMP team and FFTW
  plans are rebuilt per chunk; correct but suboptimal, fix direction noted).

**Packaging and tooling**
- `main.py` no longer crashes on a clean Windows clone: the DLL directory
  registration is guarded with `isdir` and points at the current locations
  (`backend\analysis\build\Release`, `dist\win\...\_internal`).
- `interferolab.spec` aborts loudly when no compiled `analysis_backend` is
  found (a bundle without it froze "green" and failed at first launch), and
  takes the project root from `SPECPATH` instead of the cwd.
- `requirements.txt` now truly pins `numba`/`llvmlite` (pylablib declares
  them with no version bound; the previous comment claimed an indirect pin
  that did not exist).
- The vestigial `[project]` table (invalid per PEP 621 — no `version`) was
  removed from `pyproject.toml`; the file is tool configuration only and the
  header now says so.
- `quantum eficiency.png` renamed to fix the typo, with its two references
  updated.

Verified: backend rebuilt, full suite green (307 tests, ~5 s), ruff at
baseline. The `FakeSession.move_to` test helper gained the `cancel_flag`
parameter to match the real session.

Refuted by verification (not applied): "missing CI" and "missing
CITATION.cff" (generic advice, not defects) and a claimed stale-binary
shadowing issue in the spec (documented workflows already handle it).

**Publication decisions (2026-08-28).** License: GPL-3.0-or-later (LICENSE
added; the C++ backend links FFTW, which is GPL, so the distributed bundles
must be GPL-compatible as a whole). Mirror policy in README: issues welcome,
pull requests not merged on GitHub (changes land on the private Forgejo).
The Bayer-weights LaTeX report was translated to English and renamed
`bayer_weights_report.tex`/`.pdf` (PDF regenerated). `Documentos/` renamed to
`docs/` (references updated). Both platform releases regenerated with all the
review fixes (2026-08-28 linux tar.gz + windows zip); the 08-27 artifacts were
retired.

---

## Raw Bayer output on color cameras (2026-09-08)

**Sweep failure in the lab.** Every frame was dropped with
`_bayer_to_superpixel expects a 2-D Bayer frame, got shape (3000, 4096, 3)`.
Cause: `pylablib`'s `ThorlabsTLCamera` calls `set_color_format()` with
`color_output="auto"` on construction, which resolves to `"rgb"` on a color
sensor, so `snap()` returned a software-debayered `(H, W, 3)` array. The
existing guard in `connect_camera()` tried `get_all_pixel_formats()` /
`set_pixel_format("Mono16")`, methods that do not exist in `pylablib`'s
TLCamera class: the call raised `AttributeError`, was swallowed by the
`except` branch as a warning, and the camera stayed in RGB mode.

`connect_camera()` now calls `cam.set_color_format(color_output="raw",
color_space="linear")` and **raises** if that fails (silently continuing means
a whole sweep of dropped frames). It also reads `get_color_info()` and warns
if the sensor's Bayer phase is not `red` (RGGB), which is what the superpixel
weights and `_bayer_to_color_superpixel` assume.

## Bayer mosaic phase read from the sensor (2026-09-09)

**Wrong colours in the live preview.** With raw output working, the colour
preview showed a wrong cast: the three superpixel merges hardcoded an RGGB
mosaic (red at `[0, 0]`), but this camera reports
`get_color_info().filter_array_phase == "blue"` — a **BGGR** sensor. The guard
added the day before did detect it and logged a warning, but nothing acted on
it.

The consequences were not limited to the preview:

- Colour mode: the R and B channels were swapped, in the preview *and* in the
  saved `.bin12`/TIFF frames.
- Mono and `mono_superpixel` modes: `BAYER_W_R` (0.257) was applied to the blue
  photosite and `BAYER_W_B` (0.120) to the red one. Silent, but a real
  photometric error — the greens were unaffected, since `W_G1 == W_G2`.

`utils/camera_constants.py` now owns the mosaic phase alongside the weights:
`BAYER_RED_SITE` maps each phase name to the position of the red photosite, and
`bayer_sites(phase)` returns the four `(row, col)` offsets (both pylablib
spellings of the green phases are accepted — its enum says `green_left_or_red`
and its own debayering code says `green_left_of_red`). `connect_camera()` reads
the phase, validates it and stores it; the three `_bayer_to_*` merges take it as
an argument and default to RGGB, so the historical path stays **bit-exact**.
An unrecognised phase name falls back to RGGB with a warning instead of
aborting the connection.

The ROI is never moved (only `hbin`/`vbin` are reset), so the sensor-level phase
stays valid for the acquired frames.

12 new tests (319 total): phase→offset mapping, both spellings, R/B swap under
BGGR, bit-exactness of the default path, and phase propagation through the
binned mono merge.

**Data taken on 2026-09-08 is affected**: colour stacks have R and B swapped,
and mono stacks were merged with the red and blue weights exchanged.

## The analysis no longer depends on the machine's free memory (2026-09-24)

Same corrections batch, second block. Found by the numeric reviewer while
checking the Method 3 and 4 changes above; the defect itself is **older than
this batch** and was never introduced by it.

### Why

The backend streams a dataset in row-chunks whose size `auto_row_chunk()`
derives from `MemAvailable` at that instant. That is harmless for every
per-pixel computation — they are independent of how the image is split — but
one value is not per-pixel: Methods 2 and 4 estimate a single bandpass
`{k_avg, dk}` from a sample of Z-traces and then apply it to **every** pixel of
the image. The sample was "256 pixels of the first chunk, with a stride of
(R·Nx)/256". Different free RAM → different R → a different set of 256 pixels →
a different mean amplitude spectrum → `dk` one bin away → a different band for
the whole image.

Measured on `data/S1F1` (1216×1936, 453 frames) by forcing the chunk size, and
independently by faking `/proc/meminfo` in a user namespace: the band moved
from {45, 103} to {45, 104} and **99.95 % of the Method 2 heights changed**, by
up to 22.7 nm. On `data/S1F5`, 99.97 % of them, by up to 58.6 nm. Methods 1 and
3 never read the band and were bit-identical throughout. It was not OpenMP and
not FFTW: with a fixed chunk size the backend is already deterministic
(`FFTW_ESTIMATE`, `schedule(static)`, independent pixels).

Two runs of the same dataset on the same machine could therefore disagree —
and did, which is how it surfaced: an analysis launched right after another one
saw a different `MemAvailable` because of the page cache.

### What changed

- `reconstruction.cpp`: the bandpass sample is now a fixed grid,
  `cfg::BAND_SAMPLE_ROWS` × `cfg::BAND_SAMPLE_COLS` = 16 × 16 = 256 pixels,
  over the first 16 rows of the image and across its full width. 16 rows is the
  floor `auto_row_chunk()` is allowed to return, so the grid is always fully
  present in the first chunk whatever the memory, and the sample is identical
  on every machine and every run. The count of samples and the estimator are
  unchanged. A band of rows samples the mean *amplitude* spectrum as well as a
  scattered set does, because a height difference only moves the phase of the
  transform, not its modulus.
- `analysis_api` / `run_analysis`: the result now carries `k_avg` and `dk`
  (−1 for Methods 1 and 3). It is the only quantity the whole image shares, so
  reporting it is what lets a caller — or a test — see that two runs did the
  same analysis.
- `utils.cpp`: `auto_row_chunk()` honours `INTERFEROLAB_ROW_CHUNK`, a
  test/diagnostic override for the chunk size. It is the knob that lets a test
  vary the chunking without changing the machine's memory; nothing in normal
  use sets it. The forced value goes through the same limits as the RAM
  estimate (16 ≤ R ≤ 4096, R ≤ Ny), which now live in one function: a value
  below the floor would otherwise shrink the sampling grid (8 rows → 128
  traces) and the knob meant to prove chunk-independence would be the one way
  to break it. The two RAM-less fallbacks are clamped to Ny as well, which they
  were not.

### What it changes in the results

Methods **1 and 3 do not move at all** (bit-identical on both real stacks and
on the whole synthetic bench): they have no global band. Methods 2 and 4 move
wherever the fixed grid picks a different band from the old sample:

| Stack | M2 | M4 |
|---|---|---|
| `data/S1F1` | unchanged (bit-identical) | 99.93 % of pixels, std 0.76 nm, max 39.7 nm; S_q 6.797 → 6.813 |
| `data/S1F5` | 99.97 % of pixels, std 1.32 nm, max 58.6 nm; S_q 28.235 → 28.154 | unchanged |
| synthetic bench | `clean`, `noisy`, `centre` unchanged; `asym` 62 % of pixels, max 3.2 nm | `clean`, `noisy`, `centre` unchanged; `asym` 97 % of pixels, max 2.4 nm |

(S_q with `scripts/flat_noise_analysis.py`, the script behind every published
table.) The height maps of Methods 2 and 4 must therefore be recomputed. Over
the **eight datasets re-run for this change** their S_q moves **between 0.000
and 1.19 nm**: the largest is 163.94 → 165.13 nm on the 2026-05-21(a) raw
dataset with Method 2, i.e. 0.7 %. The two stacks in the table above, which are
the smallest of the set, move 0.02 and 0.08 nm — do not take those as the
bound. Methods 1 and 3 need no re-run.

### What this does NOT fix

The fixed grid buys reproducibility, **not** independence from the sample.
`dk = ceil(2·sqrt(var))` and, on both real stacks, `2·sqrt(var)` sits within
±0.3 bins of an integer while its spread over different 256-trace samples is
±0.3–0.5 bins, so no sample of that size pins `dk` down. On `data/S1F1` the
grid gives `dk` = 103 and 64 % of 200 random samples agree; on `data/S1F5` the
grid gives 53 and the **majority** of random samples (62 %) give 54. The grid
fixes the answer by convention, and one bin of `dk` is worth 1.3–1.4 nm of
height dispersion in Method 2. The grid also reads the top 16 rows, whose DC is
~4 % lower on S1F1, which puts its estimate about 2σ below the mean of random
samples. Making Methods 2 and 4 independent of that convention needs a
different `dk` estimator (a coarser grain, or many more traces), not a
different sample. That is a design decision and it is not taken here; it is
recorded in `config.hpp` next to `BAND_SAMPLE_ROWS`.

### Tests

`tests/test_reproducibility.py`: the same scan reconstructed with several
forced chunk sizes must give the same band and a bit-identical height map, on
two synthetic fields whose spectrum varies down the image and — when
`data/S1F1` is present — on the real stack, which is the case that actually
failed before the fix. Plus two guards: that `BAND_SAMPLE_ROWS` never exceeds
the 16-row floor which makes the grid available in the first chunk, and that a
forced chunk size below that floor is clamped up to it instead of shrinking the
grid.

## Methods 3 and 4 fixed; reference wavelength set to 570 nm (2026-09-24)

Corrections batch, backend block. They come from the physical-mathematical
verification of 2026-09-23 (findings D1-09/10/11, D1-12 and D1-18), and all
three were decided by Miguel. **The published results of Methods 3 and 4
change; Methods 1 and 2 do not move at all.**

### Why

- **Method 4 averaged angles.** The envelope position appears in the spectrum
  as a linear phase ramp, and the method read its slope as the
  amplitude-weighted mean of `arg(FFT[k+1]·conj(FFT[k]))` bin by bin. That is
  wrong in two measured ways. When the envelope sits near the middle of the
  scan the true per-bin step is exactly ±π, so `atan2` returns +π for some
  bins and −π for others and their arithmetic mean collapses to ≈ 0, placing
  the surface at the bottom of the scan (error Nz·Δz/2 ≈ 2.7 µm on a
  300-frame, 20 nm scan). And the band half-width `dk/2` comes from the
  second moment of the whole amplitude spectrum, which the noise floor
  inflates — on a real stack, 315 bins around a peak only ~12 bins wide — so
  hundreds of signal-free bins, each with a uniformly random angle, got a
  vote. Together they explain the S_q of 431–1188 nm published for Method 4.
- **Method 3 returned grid positions.** `find_envelope_peak()` reported the
  scan position of the discrete envelope maximum, so every height was
  quantised to the axial step: a uniform ±Δz/2 error, i.e. an rms floor of
  Δz/√12 = 5.8 nm at 20 nm and 8.7 nm at 30 nm. That floor is inside the
  published S_q of Method 3 (8.1 nm on the 2026b dataset contains 5.8 nm of
  pure quantisation) and is what the "multiple peaks" of its height histogram
  really are: the Δz levels, not a phase-step calibration problem.
- **`LAMBDA0_NM` was 3–4 % low.** 550 nm was a rounded nominal value. The
  Bayer-weighted spectral centroid of the detected light is 566 nm, and the
  carrier the interferometer actually produces is longer still — the measured
  fringe period on a raw superpixel scan is ≈ 0.287 µm (λ_eff ≈ 575 nm),
  because the finite NA of the Mirau objective stretches the period by
  (1 + cos θ_max)/2 ≈ +2.4 % at NA ≈ 0.3.

### What changed

- `reconstruction.cpp`, Method 4: the estimate is now
  `arg(Σ_k FFT[k+1]·conj(FFT[k]))` over the same band — the cross products are
  summed as complex numbers and the angle is taken once, at the end. The two
  ±π contributions then add coherently instead of cancelling, and each bin
  contributes a vector of length |FFT[k+1]|·|FFT[k]|, so noise bins weigh
  quadratically less and cancel against one another. Same band, same loop, one
  `atan2` instead of one per bin: the cost is unchanged.
- `reconstruction.cpp`, `find_envelope_peak()` (used only by Method 3): the
  maximum is refined by a parabola through it and its two neighbours, giving
  sub-step resolution. The refinement is skipped when the maximum is at either
  end of the scan or when the three samples are not concave, and the offset is
  clamped to ±½ step. **Known and accepted cost, documented in the function:**
  a parabola is symmetric, so on a skewed envelope its vertex is pulled
  towards the wider side. Measured on a split-normal envelope (trailing side
  1.6× wider): the discrete locator read −103.60 nm low and the parabola reads
  −103.62 nm. So those ~104 nm are **not** the price of the refinement — they
  are produced by the envelope smoothing (`ENVELOPE_SIGMA` = 15 samples) acting
  on a skewed envelope, a symmetric filter dragging the maximum of an
  asymmetric curve towards its wide side. The kernel's own, unsmoothed maximum
  is only −16 nm off at that skew, −35 nm at 2.5× and −43 nm at 3.5×; after the
  smoothing it becomes −100, −200 and −276 nm. The bias is therefore shared
  with Method 1 and is reduced by lowering σ_e, not by changing the peak
  locator. What the parabola itself adds stays below 0.5 nm for skews from 1.0
  to 5.0, with and without noise, while it removes 5.8 nm of quantisation. A
  systematic offset common to every pixel cancels in height differences, which
  is what this instrument measures.
- `config.hpp`: `LAMBDA0_NM` 550 → **570 nm**, between the 566 nm centroid and
  the ≈ 575 nm measured carrier, within 1 % of either, with the derivation in
  the comment. It is read only by Method 3's phase step α (now 0.441 rad
  instead of 0.457). Measured effect on real heights: median −0.2 nm, and 13
  pixels out of 2.35 million move by more than 100 nm (noisy pixels whose
  global maximum flips). `NOMINAL_DZ_NM` keeps its value, with a measured note
  that a 1.5× mismatch in α moves the heights by < 0.1 nm.
- `methods.hpp`: the two method descriptions shown in the GUI now say what the
  methods do.

### Measured (synthetic bench, rms height error in nm)

| Case | M1 | M2 | M3 before → after | M4 before → after |
|---|---|---|---|---|
| clean (12-bit only) | 1.3 | 0.7 | 5.78 → **0.23** | 88.5 → **0.85** |
| noisy (σ = 5 % of the fringe) | 10.3 | 17.3 | 14.5 → **13.6** | 808 → **8.9** |
| envelope at the scan centre | 9.4 | — | — | 2327 → **8.6** |
| asymmetric envelope | 0.8 | — | 5.79 → **0.34** | — |

"rms" above is the dispersion of the height error with its mean removed; on
the asymmetric case the mean itself is −103.6 nm (previous paragraph).

M1 and M2 are bit-for-bit identical before and after, on the synthetic bench
and on the 2.35-Mpixel real stack `data/S1F5` (max |diff| = 0.00 nm) — with the
same row-chunking on both runs, which before the fix below was not something a
caller could take for granted. On that real stack the spread of M4 against M1
falls from 83 nm to 30 nm (robust standard deviation) and its 1–99 % range from
[−204, +132] nm to [−18, +62] nm. Its texture parameter S_q goes 43.4 → 13.1 nm,
next to the 11.2 nm of Method 1. With `scripts/flat_noise_analysis.py`, which is
what produced every published table, Method 3 on that stack goes
**11.80 → 7.97 nm**, and √(11.80² − 8.66²) = 8.01 with 8.66 nm = 30/√12: the
whole improvement is the quantisation floor of its 30 nm step, nothing else.

### Tests

New: `scripts/method_accuracy_bench.py`, a standalone reproducible bench that
builds synthetic interferograms of known height (heights deliberately off the
axial grid, one fringe phase per pixel), measures the error of every method
and exits non-zero if any exceeds its budget. `tests/test_method_accuracy.py`
runs it inside the suite and adds the invariance test: M1 and M2 are compared
bit-for-bit against reference height maps in `tests/refdata/`, produced by the
backend **before** this change.

### What this invalidates

Every published Method 3 and Method 4 number was produced by the previous
estimators and has to be recomputed; the report must state which backend
version produced each table. Methods 1 and 2 need no re-run.

## A positions.csv next to every sweep (2026-09-23)

Phase-2 batch 6 (asked for by Miguel).

### Why

The dataset folder said nothing about what actually happened during the
sweep: the file names carry the COMMANDED z (the analysis reads it from
there) and the measured position only existed in the window. A partial
dataset (batch 2) was recognisable but not readable: which frames did it
have, and where was the stage?

### What changed

- Every sweep writes `positions.csv` in its dataset folder: a header plus one
  row per SAVED frame — `index`, `filename`, `z_commanded_um`,
  `z_measured_after_move_um` (`nan` when the position could not be read) and
  `timestamp` (local, milliseconds, taken when the frame came out of the
  camera).
- Written as the sweep goes and flushed per row (measured: 2 µs median per
  row, 1.1 ms for a whole 501-step sweep), never kept in memory: a sweep that
  aborts or is cancelled leaves the rows of the frames it did save, matching
  the files on disk and the counters in `SWEEP_ABORTED.txt`. A frame that was
  not saved (failed capture, failed save) has no row.
- The images, their names and their contents are exactly what they were; the
  csv is a record, not an input to the analysis. `README.md` (and the column
  name) say that the measured z is read when the MOVE ENDS, before the
  exposure, so it is not a per-frame z for the reconstruction.
- A folder that cannot take the csv (read-only, full) is warned about once
  and the sweep goes on.

### Tests

Content and order of the rows, the unknown-position case, frames that were
not saved, an aborted sweep (csv consistent with `SWEEP_ABORTED.txt`), a
cancelled one, proof that each row is on disk before the next frame is
captured, and a csv that cannot be opened. Two existing tests that counted
every file in the folder now count the images.

### Unrelated fix

`tests/test_no_hardcoded_serial.py` (batch 5) failed on every run: the guard
flagged itself, because the serial it searches for was a literal in that same
tracked file. It is now assembled from pieces.

## Keyboard piezo moves, fine step, camera timeout apply, park on connect (2026-09-22)

Phase-2 batch 5 (fix 4/8/9 of the verified diagnosis: findings U1/C10, U8,
U5/TODO.md item 1, and Miguel's decision to park the piezo on connect; plus
U9, the piezo serial still in the code).

### Why

- Arrow keys, Page Up/Down, Home and End on the Z slider changed the number
  shown but never sent a move: `sliderReleased` is mouse-only, so the
  keyboard looked like it worked and silently did nothing (U1/C10). Enter on
  Manual Z had no connection at all (U7). Both are exactly the fine control
  needed to walk into a fringe position while watching the preview.
- The slider's own arrow-key step (0.1 µm) and Manual Z's default spin step
  (1 µm, Qt's default -- `setSingleStep` was never called) were both too
  coarse for that (U8).
- "Apply camera settings" only ever sent the exposure; the timeout typed in
  the same box did nothing until the next sweep rebuilt the whole config
  (U5, TODO.md item 1).
- The piezo never moved to a known position after connecting -- the code for
  it did not exist (TODO.md item 1; A3's finding was that only *disconnect*
  parks at 0). Miguel decided it should go to 50 µm on a successful connect.
- `views/AcquisitionPanel.py` still had a hardcoded serial number as a silent
  fallback when the serial field was left empty (U9): the real lab serial
  stayed in the repository (and its public mirror) despite an earlier commit
  meant to remove it.

### What changed

- New `views.AcquisitionPanel._PositionSlider` (what `sl_z` now is): listens
  to `actionTriggered` and asks for a move on every discrete step action
  (single/page step, home/end) but never on a drag (`SliderMove`, the same
  action a mouse release already handles). `actionTriggered` fires *before*
  the slider's own `value()` updates (verified against PySide6), so the
  request is deferred one event-loop tick (`QTimer.singleShot(0, self, ...)`,
  the same pattern as `widgets/HeightmapView.py`) so it reads the settled
  target, not the one it is about to leave. A burst of key presses before
  the loop turns asks for a move once per key, all with the same final
  value -- `move_to`'s own "last order wins" already absorbs that; nothing
  new is queued.
- `sb_manual_z.lineEdit().returnPressed` (Enter) now triggers a move, reusing
  `_move_piezo_manual` exactly like the slider release and the "Move piezo"
  button.
- New "Keyboard step (nm):" spin box (`sb_keyboard_step_nm`, default 20 nm --
  the same value as the default sweep step, 0.02 µm, a reasonable step for
  CSI fringe hunting): sets both the slider's arrow-key step and Manual Z's
  own spin step live, and is persisted as `keyboard_step_um` in
  `app_config.json` (schema entry added to `utils/config_manager.py`). The
  slider's page step (1 µm, a groove click or Page Up/Down) and pageStep are
  unchanged. Tooltips corrected on the slider and Manual Z.
- `_apply_camera_params` now sends `{"exposure": ..., "timeout": ...}` to
  `apply_config` instead of only the exposure. Nothing else needed to
  change: `start_preview()`/`capture_preview()` already re-read
  `self._cfg["timeout"]` on every call, so the value only had to reach the
  config, not the camera hardware.
- `_on_connect_finished(True)` now also sends a move to `PARK_POSITION_UM`
  (50 µm) through `AcquisitionVM.move_to` -- the exact same service path as
  any manual move, never the hardware directly, and non-blocking like
  everything else in this window. A new `_auto_park_pending` flag makes a
  failed park a log warning (`_on_error`) instead of the usual modal: the
  user just connected and did not ask for this move. The flag is cleared by
  any move the user asks for afterwards (button, slider, Enter), when the
  park move itself finishes (success or failure, `movingChanged(False)`),
  and when a sweep starts (defensive backstop), so a later real error goes
  back to being a dialog instead of being swallowed as a fake park failure.
- `le_piezo_serial`'s hardcoded fallback serial is gone. An empty field
  at connect time cancels the attempt with a log warning and a
  `QMessageBox.warning` ("Piezo serial required") instead of silently
  reaching a specific controller; the real serial is no longer anywhere in
  a versioned file (a test walks every git-tracked file and fails if it
  reappears). The serial still comes from `app_config.json` (local, not
  versioned) or the user's own typing, as before.
- After review (`_agentes/_trabajo/B2_revision_lote5.md`): the real piezo
  serial had leaked into this changelog entry itself (three lines describing
  the fix quoted the digits) -- replaced with a description, and the new
  regression test above catches it in any tracked file, not just this one
  (H1). The `_auto_park_pending` flag was never cleared on a SUCCESSFUL
  park, so the next unrelated failure (e.g. a sweep) was silently eaten with
  no modal at all -- fixed as described above (H2). Holding a key (arrow/
  page) lost steps: the measured-position indicator (batch 3) reset the
  slider -- and Manual Z, which the keyboard path reads as its move target
  -- mid-sequence, so 8 presses could land at 5 µm instead of 8; the
  indicator now skips both widgets while the slider has keyboard focus,
  same guard already used for Manual Z while typing (H3). `editingFinished`
  also fired on a plain Tab/focus-out with nothing typed, silently sending a
  move to whatever the field showed; switched to `returnPressed` (Enter
  only) (H4).

### Tests

Diagnostic test C10 retired from `xfail(strict)` to passing (its repro now
starts from a known Z=0 baseline before the 5-PageUp check, since connecting
already parks at 50 and the check is relative). `tests/diag/harness.py`:
`App.__init__` sets a placeholder serial (no more hardcoded fallback for the
scripts to fall back on) and `App.connect()` also waits for the park move to
settle, so every script starts from the same idle state as before this
batch; `c8_disconnect_freeze.py` shortens `MOVE_TIMEOUT_S` like
`c3_position_lie.py` already did, so its `piezo_never_on_target` fault does
not add 30 s to that wait. New unit tests: the slider's step actions and
Manual Z's Enter each request a move with the settled value (drag and a
burst of key presses included); the keyboard step updates both widgets and
round-trips through `get_config`/`apply_saved_config`; the config schema
accepts/rejects `keyboard_step_um`; `_apply_camera_params` sends both
values; connecting requests the park move and a failed one logs a warning
without a dialog, cleared by a user move; an empty serial is refused with
the warning dialog and never reaches `connect_hardware`. After review: a
new `tests/test_no_hardcoded_serial.py` walks every git-tracked file and
fails if the real serial reappears anywhere (H1); a successful park clears
`_auto_park_pending` so the next error is modal again, and starting a sweep
clears it too (H2); an end-to-end test with a real (slowed) move reproduces
8 held-key presses landing at the correct 8 µm, not 5 (H3); Manual Z's
focus-loss-without-Enter no longer sends a move (H4).

### Measured on the simulator

No change to the continuous-preview or move-vs-camera figures (this batch
touches the piezo path and the config only): 20-21 fps, ~120 ms order-to-
image median, both perf-gate criteria still PASS.

## Connect and disconnect without freezing the window (2026-09-22)

Phase-2 batch 4 (fix 6 of the verified diagnosis: findings C8 and R9; the
re-entrancy of C5 was already gone with batch 1).

### Why

- Connecting (opening the camera and the piezo, about 1.4 s estimated) and
  disconnecting ran on the GUI thread. With a piezo that never reports it is
  back at 0, a disconnect or the app's close froze the window for 10 s
  (measured: 10.2 s, a 100 ms timer fired after 10.2 s); with a driver call
  that does not answer it would be longer (7 s per read in the PI DLL).

### What changed

- `AcquisitionService.connect_hardware()` / `disconnect_hardware()` run the
  connection on a Python daemon thread (never a QThread: a driver call that
  never returns is left behind at exit instead of making Qt abort) and report
  `connectFinished(ok)` / `disconnectFinished()`. One at a time
  (`connection_busy()`); while one is in progress a second connect, a
  disconnect, a sweep, a manual move and the camera (preview, snapshot) are
  refused, so the camera owner is never created while the camera is being
  opened or closed. A failed connect releases what was opened on the same
  thread.
- Panel: the button says "Connecting…" / "Disconnecting…" and everything is
  frozen until the thread reports back; a second click does nothing; the
  "Hardware connection failed" dialog stays (the user must act).
- `disconnect_all`: the camera close is requested from the owner first and
  waited for after the piezo is released, so both waits overlap (a hung
  camera and a faulty piezo cost max(8, 10) s, not 18 s); a piezo that never
  reports reaching 0 is warned about and its servo is still switched off (it
  used to skip SVO/SVA); `park_timeout_s` and `cancel_flag` parameters.
- After review: "Apply camera settings" (`connect_camera`) is refused by the
  service itself while connecting/disconnecting (it would have set the
  exposure from the GUI thread on a camera being opened or closed); a refused
  `start_preview` never leaves the button on "Stop preview"; a connection
  thread still running after `CONNECTION_STUCK_S` (20 s, above the longest
  legitimate disconnect) gives the window back with "not answering", no new
  connection starts while it lives, and whatever it opens when it finally
  returns is released; the camera is not opened when the piezo already
  failed.
- App close (`shutdown`): a connect/disconnect in progress has its piezo wait
  cancelled and is joined (bounded, 10 s; a thread stuck in a driver is left
  alone and reported, never raced from a second thread); the final release
  parks the piezo with a 2 s deadline (`CLOSE_PARK_TIMEOUT_S`) instead of
  10 s. The camera keeps its 8 s deadline and `abandon()`.

### Measured on the simulator

Faulty piezo (never on target): the Disconnect click returns at once and a
100 ms timer fires at 0.1 s, while the window shows "Disconnecting…" (the
release itself still waits its 10 s, off the GUI thread); closing the app with
the same piezo takes 2.1 s instead of 10+ s. Connect: the click returns at
once and the window says "Connecting…".

### Tests

Diagnostic test C8 retired from `xfail(strict)` (now also checks connect,
the frozen buttons, the ignored second click and the close); its wall-clock
bounds only with `INTERFEROLAB_TIMING_TESTS`. New tests for the connection
thread (non-blocking, refusals while busy, failed connect released on the
thread, shutdown cutting a slow disconnect short, a stuck thread left alone),
for `disconnect_all` (servo off after a park timeout, cancel, camera closed
while the piezo parks) and for the panel states. Reproduction scripts and the
smoke test wait for the connection thread (`App.disconnect()` in the harness).

## Measured piezo position, followed live (2026-09-22)

Phase-2 batch 3 (fix 5 of the verified diagnosis: findings C3, U3, U4).

### Why

- The window showed the TARGET as the "real" position: the app never read
  the piezo (`qPOS`). After a failed move (no on-target within the timeout)
  the stage had executed the MOV and sat at 60 µm while the slider said 0
  (C3); a cancelled move showed the target it never reached (U4).
- The position indicator jumped once at the end of a move and did not follow
  it (U3); since batch 1 it no longer moved with the preview either, and
  Miguel decided it must follow the piezo, with the measured position.

### What changed

- `AcquisitionSession._read_position` reads `qPOS` (NaN = unknown, never
  raises). `_move_piezo` returns the measured position and reads it also
  when the move fails or is cancelled, before raising; `connect_piezo` reads
  the starting position. `_last_real_pos` (the preview frames' z) is now
  always a measurement.
- Live following: `move_to(..., position_cb=...)` reports `qPOS` on every
  on-target poll after the first one, from the move thread (the only thread
  on the piezo then: no query from the GUI). A healthy small step is on
  target by the second poll, so it costs one query more than before (the
  final one); a slow or failed move is followed every ~0.12 s.
- Sweep: progress (and so the indicator) carries the measured z; the FILE
  NAMES keep the commanded z as before, because the analysis reads z from
  them and switching to the sensor reading would change results.
- Panel: a "Measured Z" label (3 decimals; "unknown" when unreadable, never
  drawn as 0); slider and Manual Z follow the measured position, but Manual
  Z is not overwritten while it has the keyboard focus (the user may be
  typing the next target during a followed move); corrected slider tooltip.

- `_wait_on_target` polls at a fixed rate (100 ms from the start of one
  poll to the next) instead of sleeping 100 ms after each query, so the
  final `qPOS` comes out of the pause and moves are not longer than before.

### Cost (simulator, default profile; GCS figures estimated)

One `qPOS` is 20 ms (query + pipython's `ERR?`). With the extra read alone a
1 µm manual move went from 153 to 173 ms and the perf gate lost one order of
30 (the Move button stays disabled longer); with fixed-rate polling it is
153 ms again and the gate is back to 28 own MOVs, 0 lost. Sweep steps keep
their length for the same reason. The preview is untouched (the camera
thread never queries the piezo).

### Tests

Diagnostic test C3 retired from `xfail(strict)` and extended: the window
follows the stage live during a failed 1 s move and ends on the measured
position. New unit tests: measured return value, one extra query per poll
after the first, failed / cancelled / rejected moves report where the stage
is, unreadable position is NaN, sweep file names vs measured progress,
starting position at connection, panel label / NaN / focus guard. The smoke
test compares the reported position with a tolerance (it is measured now).

## Camera errors without modal dialogs; sweeps abort on a lost camera (2026-09-22)

Phase-2 batch 2 (fix 2 of the verified diagnosis: findings C7, C4, C4b and
C11), on top of the single camera-owner thread of batch 1.

### Why

- Any camera hiccup during the live preview (one lost frame, an SDK error)
  stopped the preview and opened a modal "Acquisition error" dialog, often
  **empty**: pylablib raises its frame timeout without a message (C7). While
  focusing, the user had to close a blank dialog and restart the preview
  without knowing why.
- A sweep never gave up: a failed capture was skipped and the loop went on.
  With the camera unplugged it walked all 501 positions (about 2 min); with a
  camera that stays present but never delivers, each step paid the full
  timeout (derived: about 47 min), for a dataset with no frames (C4, C4b).
- A failing `qONT` counted as "on target" after a fixed wait, so a move whose
  arrival could not be confirmed was treated as a success (C11).

### What changed

- Live preview and snapshot failures (arming failed, SDK error while
  streaming, no frame within the timeout, snapshot failed) are **retried by
  the camera thread**: the camera is disarmed, and re-armed after 1 s
  (`CameraOwner.PREVIEW_RETRY_DELAY_S`; the pause never delays a stop, a
  sweep or a close). Each failure is shown as a non-blocking notice in a
  status line under the preview (`previewNotice` signal, `lbl_camera_status`),
  with the real text; the notice clears when frames flow again. After 3
  failures in a row (`PREVIEW_MAX_FAILURES`) the preview stops and the status
  line says so. No dialog in any of these cases. The inner 3-attempt arming
  loop of batch 1 is folded into this retry.
- A failed exposure change is also a notice plus a log line, not a dialog:
  the preview goes on, and every sweep sets and verifies its own exposure.
- Error texts are never empty (`describe_error`): an exception without a
  message is described by its type ("the camera did not deliver a frame in
  time (ThorlabsTLCameraTimeoutError)"); the error dialog has a fallback text.
- The sweep **aborts after 3 consecutive failed captures**
  (`AcquisitionSession.SWEEP_MAX_CONSECUTIVE_FAILURES`); a lone failure is
  still skipped as before, and a good frame resets the count. The partial
  folder gets a `SWEEP_ABORTED.txt` (reason, frames saved, frames planned),
  and the user gets one "Sweep aborted" dialog with the number of frames
  saved (new `sweepAborted` signal before `finished`).
- A full disk is treated like a lost camera: 3 failed saves in a row also
  abort the sweep (separate count, same threshold).
- After review: a failed arming no longer lets a pending snapshot spend the
  next attempt without the 1 s pause; a failed exposure's notice clears when
  an exposure is applied successfully (frames at the old exposure do not
  clear it); a given-up snapshot says so ("Preview frame given up ...; the
  live preview is stopped").
- `_wait_on_target`: only GCS error 2 ("unknown command", a controller
  without `qONT`) falls back to the settle-time wait; any other `qONT` error
  is polled again, and 3 in a row fail the move ("Could not confirm that the
  piezo reached its target").
- Dialogs that stay, on purpose: failed hardware connection (nothing works
  until the user acts), failed sweep and failed manual move (the stage may
  not be where the window says), and the end-of-sweep "Incomplete dataset"
  / "Sweep aborted" warnings (the dataset must not be analysed as if whole).
- Simulator: new continuous-stream fault, off by default
  (`faults.stream_stall_after_frames`, `faults.stream_stall_arms`; example in
  `sim/profiles/faults_preview_stall.toml`), because the existing snap faults
  never reach a camera that stays armed.

### Measured on the simulator

Camera unplugged mid-sweep (21 positions, 0.5 s timeout): 3 failed steps and
abort in 1.8 s (was 19 failed steps, 4.3 s). Camera that never delivers
(5 positions): abort after 3 steps, 3.4 s; for the default 501-step sweep
with a 5 s timeout that is about 17 s instead of about 47 min (derived:
3 x 5.6 s). A stalled stream is noticed after the timeout and re-armed; the
preview resumes about 1.3 s after the notice.

### Tests

Diagnostic tests retired from `xfail(strict)` to passing: C4, C4b, C7, plus a
new "never recovers" variant of C7; C7b now also checks the dialog text and
that the piezo error leaves the camera status alone. New unit tests for the
retry and give-up of the preview, the stop/close during the retry pause, the
empty timeout message, the sweep abort and its marker file, the failure
streak reset, the qONT confirmation (C11) and the non-modal status line.

## Continuous live preview with a single camera-owner thread (2026-09-22)

Phase-2 batch 1 (implementation of fixes 1 and 3 of the verified diagnosis,
`_agentes/_trabajo/A4_verificacion.md`). Option 1a chosen by Miguel on
2026-09-21: the preview is a continuous stream and exactly one thread uses
the camera.

### Why

- Every preview frame paid a full `snap()`: pylablib arms the camera (an
  85-frame, ~2.1 GB buffer), sleeps 0.05 s, triggers, waits, sleeps 0.2 s and
  disarms. Measured on the simulator: 0.5 s per frame, 1.5 frames/s at best,
  0.7-2.9 s from a piezo order to the image, and 90 % of manual moves refused
  at the fastest preview interval because a capture was "in flight" (R1-R4,
  R6, C6). The app and Qt cost 10-15 ms per frame: not the problem (R7).
- pylablib has no lock around the SDK: disconnecting, reconnecting, applying
  the exposure or closing the app while a capture was in flight put two
  threads inside the camera object (C1, C1b, C5, C9), and a hung SDK call
  made Qt abort the process at exit ("QThread: Destroyed while thread is
  still running", C2).

### What changed

- New `backend/acquisition/camera_owner.py`: `CameraOwner`, a Python daemon
  thread with a request queue that is the only user of the camera once it is
  connected. It runs the live stream (camera armed ONCE in continuous mode,
  small ring buffer, `wait_for_frame` + `read_newest_image` in a loop, Bayer
  merge on that thread), one-shot snapshots (`snap()`) while the stream is
  off, exposure changes (stream stopped, exposure set, stream re-armed), the
  Z sweep (`run_sweep` unchanged, stream paused before and resumed after)
  and the camera close. Orders record a *wanted* state that the thread
  reconciles, so the last order always wins.
- "Newest frame wins": the camera thread drops each frame into a one-slot
  mailbox in the service and posts a single notification; a slow GUI skips
  intermediate frames instead of queueing them.
- Manual piezo moves never wait for the camera any more; an order arriving
  during a move replaces the queued target (old steps are never replayed).
  Preview frames no longer re-emit the set-point as a position, so the slider
  and Manual Z stop snapping back while the user edits them.
- `AcquisitionSession` gained the stream primitives (`stream_start`,
  `stream_read`, `stream_stop`), `set_exposure`, `close_camera`, and creates
  the owner on demand; `disconnect_all` closes the camera through the owner
  with a deadline (`CAMERA_CLOSE_TIMEOUT_S`, 8 s). If a camera call never
  returns, the thread is abandoned, the handle forgotten and the user told
  ("Camera is not answering ..."); the abandoned thread cannot touch a camera
  reconnected later (`abandon()` + the `_call` guard; `run_sweep` binds its
  camera object once). `connect_camera` no longer arms the camera.
- Service: the sweep, preview and move QThreads are replaced by the owner
  thread plus one QThread per manual move (with the existing graveyard);
  `start_preview()/stop_preview()` and `previewingChanged(bool)`; `start()`
  no longer refuses a sweep during a preview (it queues behind it on the
  same thread); `shutdown()` is bounded and never destroys a running thread.
- Panel: the preview timer, the "Preview interval" spinbox and its Apply
  button are gone (older `app_config.json` values are accepted and ignored);
  "Start sweep" no longer spins the event loop waiting for a capture (no
  re-entrant clicks); an error no longer switches the preview flag off: only
  the camera thread does, through `previewingChanged`.

### Measured on the simulator (default profile, offscreen)

Before: 1.5 fps on screen, 719-756 ms median order-to-image, 18 of 20 moves
refused. After: about 20-21 fps on screen (the simulator's own frame
synthesis is the limit), order-to-image median about 120 ms, 0 moves refused.
Figures and method in `_agentes/_trabajo/B1_nucleo.md`.

### Tests

New `tests/test_camera_owner.py` (single owner, stream, snapshot,
exposure, sweep pause/resume, errors, abandoned camera) and new service tests
(newest frame wins, moves during the stream, latest move wins, exposure and
sweep through the owner, bounded shutdown). Diagnostic tests retired from
`xfail(strict)` to passing: C1, C1b, C2, C5 (both), C6, C7b, C9, plus a new
whole-session single-owner check; the reproduction scripts behind these
tests now live in `tests/diag/`. `FakeCamera`/`FakeSession` extended with the
continuous-acquisition surface. After review: an abandoned camera thread whose
stuck call ends with an exception, or that was in the middle of a sweep, can
no longer touch a camera or piezo reconnected later (`_call` guarded before
and after, `abandon()` cancels the sweep); load-dependent assertions are only
checked with `INTERFEROLAB_TIMING_TESTS` set.

### To confirm in the lab

Arming cost and memory with the 10-frame ring buffer; whether the stream
keeps up at 21.7 fps on the lab PC (the mailbox drops frames if not); the
SDK's behaviour of `set_exposure` on an armed camera is deliberately not
relied on (the stream is stopped around it).

## Hardware simulator (2026-09-21)

**Why.** Lab access is now occasional, and the live preview (worst while moving
the piezo) cannot be diagnosed or fixed without the hardware. `sim/` provides a
simulator faithful enough to measure and fix it offline.

- **The app is not modified.** `sim/fakes/` holds fake `pylablib` and `pipython`
  packages with the exact API surface the app uses; `sim/run_simulated.py` puts
  them first on `sys.path` and starts `main.py`. Without the launcher the app
  imports the real libraries.
- **It can never pass for the real thing**: `[SIMULATION]` window title and a
  fixed red banner, `SIM-…` device serials in the log, output folders prefixed
  `SIM_` with `simulated: true` metadata (pixels are not marked).
  `interferolab.spec` and both `build_release` scripts exclude `sim/` and reject
  a bundle that contains it.
- **Four layers**: API; timing (a step-by-step replica of `pylablib` 1.4.5
  `snap()` and the GCS round trips, every figure in `sim/profiles/default.toml`
  labelled published / measured / derived / estimated); signal (white-light
  fringes over a known surface, BGGR 12-bit mosaic, noise, saturation, ground
  truth saved); injectable faults. Plus a replay mode over a real LP126CU stack.
- **Gate**: `sim/compare_real_vs_sim.py` compares simulated against real
  LP126CU stacks (59/59 checks). Most camera-SDK timings are still estimates.
- **Lab probe**: `sim/lab_timing_probe.py`, packaged as its own executable
  (`sim/lab_timing_probe.spec`), measures the missing timings on the real
  hardware and writes a profile the simulator loads with `--profile`. The piezo
  serial is read from the lab machine's `app_config.json`, never from the repo.
- 48 new tests (367 total).
