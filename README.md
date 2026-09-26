# InterferoLab

Desktop application (PySide6) for **coherence scanning interferometry (CSI)
microscopy**: it acquires interferogram stacks with a PI piezo stage and a
Thorlabs camera, reconstructs the height map with a C++ backend
(pybind11 + OpenMP + FFTW), and visualizes the results. MVVM architecture:
`views/` → `viewmodels/` → `services/` → `backend/` (acquisition in editable
Python; analysis in compiled C++).

## Repository

Development lives on a self-hosted Forgejo (private); GitHub is a public
mirror. Issues are welcome; pull requests are not merged here — changes land
on the private Forgejo and flow to this mirror.

The piezo SDK (Physik Instrumente E-816) is proprietary and does not travel in
the repository: see `API/README.md` to put it in place after cloning.

## License

[GPL-3.0-or-later](LICENSE). The C++ backend links FFTW (GPL), so the
distributed bundles are GPL-compatible as a whole.

## Getting started

Requirements: Python 3.13 or 3.14 (both verified against the full test suite)
and, to recompile the C++ backend: cmake, g++, FFTW (with threads), OpenCV and
libtiff.

> ⚠️ The backend binary (`.so`/`.pyd`) is tied to the Python version of the
> venv (`cpython-313`, `cpython-314`, ...). If the system Python changes
> version (e.g. after a rolling-release upgrade), the module stops importing
> (`ModuleNotFoundError: analysis_backend`): recreate the venv and recompile
> the backend with the commands below.

> ⚠️ **If the project lives in a cloud-synchronized folder**
>
> The sync client can truncate or drop large binaries — verified with the
> >100 MB files in `dist/` (`llvmlite`, `cv2`).
>
> - **Manually exclude** `dist/`, `build/`, `build-linux/`, `data/` and
>   `output/`: they are regenerable or huge (several GB) and gain nothing from
>   being synced.
> - `.venv/` and `.git/` are safe **only if** your client excludes hidden
>   files (in Nextcloud, a `.*` rule at the end of `sync-exclude.lst`).
>   **Check that it is there**: without it `.git` syncs file by file, and a
>   half-uploaded `git gc` leaves the repository corrupt. That rule is not a
>   default, so a client reinstall can silently remove it.
> - Use git from **one machine only**. On the others, the synced folder holds
>   the files but no `.git`: it is a copy, not a repository.

```bash
# Environment (once; uv works too). Since requirements.txt pins versions,
# there is no need to preinstall numba to help the resolver anymore.
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# C++ backend (once, when touching backend/analysis/src, or after a Python version change)
# (Linux builds in build-linux/; build/ belongs to the Windows VM — see compile_notes.txt)
cmake -S backend/analysis -B backend/analysis/build-linux \
      -DPython3_EXECUTABLE=$PWD/.venv/bin/python -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build backend/analysis/build-linux -j8
cp backend/analysis/build-linux/analysis_backend.cpython-3*.so backend/analysis/

# Run (from the project root: output/ and data/ resolve against the cwd)
.venv/bin/python main.py

# Tests (540, no hardware needed, ~5-10 min)
.venv/bin/python -m pytest

# Full release: compiles the backend, freezes into dist/ and archives into
# releases/.  On Windows: scripts\build_release.ps1  (see -h / the header)
scripts/build_release.sh

# Package only, with the backend already compiled:
.venv/bin/python -m PyInstaller interferolab.spec --noconfirm
```

### Windows (VM sharing the project folder)

The development setup shares the project folder between a Linux host and a
Windows VM (drive `Z:`), so `.venv/` is already taken by the Linux venv
(`bin/` layout, useless on Windows) and the two cannot coexist in the same
directory. The Windows venv goes in **`.venv-win\`** (also dot-prefixed, so a
`.*` sync-exclude rule keeps it out of the cloud):

```powershell
cd Z:\InterferoLab
py -m venv .venv-win
.venv-win\Scripts\pip install -r requirements.txt
```

The full release runs by **double-clicking `scripts\build_release.bat`**
(keeps the window open at the end) or from a console with
`powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1`. The
script looks for the venv in `.venv\` and then `.venv-win\`; any other path
(e.g. on the VM's local disk if the shared drive is slow) is passed with
`-VenvDir <path>`. On Windows PyInstaller packages into `dist\win\` (the root
of `dist/` belongs to the Linux build, whose symlinks cannot be deleted
through vboxsf); the zip still lands in `releases\`. Compilation
prerequisites: `backend/analysis/compile_notes.txt`.

## Usage (acquisition)

- **Start position**: low enough to capture the highest parts of the sample.
  **End position**: high enough to capture the base plane (background). The
  sweep scans Z between the two with the configured step.
- Datasets are saved to `data/<timestamp>/`; analysis writes to
  `output/<name>_M<method>/`.
- Every sweep also writes `positions.csv` in its dataset folder, one row per
  SAVED frame, as the sweep goes: `index`, `filename`, `z_commanded_um`,
  `z_measured_after_move_um` and `timestamp` (local, milliseconds). It is a
  record to check a sweep with (and to see what a partial dataset contains:
  a sweep that stops early leaves the rows of the frames it did save, next to
  `SWEEP_ABORTED.txt`). **The measured z is read from the controller when the
  move ends, before the exposure — not while the frame is being taken — so it
  is not a per-frame z for the reconstruction**, which still takes z from the
  file names (the commanded positions). `nan` means the position could not be
  read. In open loop (not reachable from the window) the two z columns are the
  commanded voltage and the measured voltage instead.
- Noise analysis of a flat reference surface:
  `python scripts/flat_noise_analysis.py output/<dataset>/<dataset>_height.npy`
  (see the script's docstring for how to read its outputs).

## Documentation map

| Document | Contents |
|---|---|
| `README.md` | This file: what it is, getting started, docs map |
| `CHANGELOG.md` | Record of the July 2026 audit and everything since (robustness, 3.5–4.2× performance, visual theme, environment, packaging) |
| `TODO.md` | Living list of pending work (lab validation, feature ideas) |
| `tests/README.md` | Test suite structure and documented conventions |
| `backend/analysis/compile_notes.txt` | Building the C++ backend (Linux and Windows/vcpkg) |
| `scripts/build_release.sh` · `.ps1` · `.bat` | Full release chain (backend → PyInstaller → `releases/`); each header documents requirements and options; the `.bat` is the Windows double-click launcher |
| `scripts/method_accuracy_bench.py` | Synthetic accuracy bench of the four reconstruction methods (known heights, off-grid, with and without noise): prints the error budget of each method and exits non-zero if one degrades. Run it after touching `compute_envelope()` or the height locators |
| `backend/analysis/how_to_add_a_method.txt` | How to add a reconstruction method |
| `API/README.md` | The proprietary piezo SDK and where to place it |
| `install/README.md` | Deployment payloads and the MSVC runtime story |
| `docs/` (outside git, one exception) | Project reports and raw diagnostic data, kept out of the repo. **Versioned exception**: `docs/Quantum efficiency and IR filter/` — the scientific derivation of the Bayer weights (`compute_bayer_weights.py`), cited from `utils/camera_constants.py` and `config.hpp` |

## Status

Code audited and fixed (July 2026; see `CHANGELOG.md`). Since then: the
acquisition was rebuilt around a single camera-owner thread with a streaming
preview, non-blocking error handling, the measured piezo position in the window
and a `positions.csv` per sweep (September 2026); and Methods 3 and 4 of the
reconstruction engine were corrected, λ₀ set to 570 nm and the band sampling made
reproducible (September 2026, commit `66e9d31` — see `CHANGELOG.md`). Published
Method 3 and Method 4 results from before that commit do not reproduce with the
current binary.

Still pending **lab** validation of the real-hardware paths (camera/piezo
connection, exclusions, movement cancellation, exposure verification) — details
in `TODO.md`. The Windows `.pyd` was rebuilt with the current CMake and the
frozen executable verified there in August 2026, so **it predates the September
backend changes and must be rebuilt before the next release**; the same applies
to any `cpython-313` module left next to the current one.
