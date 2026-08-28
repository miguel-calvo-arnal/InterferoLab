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
