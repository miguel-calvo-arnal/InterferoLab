# TODO — in order of execution

Updated: 2026-09-22 (original list: 2026-04-21)

---

1. **Debug in the lab to confirm the critical paths work** (STILL PENDING, but
   with a new safety net — see CHANGELOG.md):

   - Preview image very dark / verify camera parameters are truly applied.
     [PARTIALLY COVERED] The sweep now re-reads the exposure with
     `get_exposure` and ABORTS if it differs >5% from the requested value;
     preview errors now reach the UI (they used to be lost silently). Still
     needs confirmation with the real camera.
   - Make sure the piezo moves to position 50 after connecting. [DONE —
     2026-09-22] `AcquisitionPanel._on_connect_finished` now sends a normal
     move to `PARK_POSITION_UM` (50 µm, the centre of the default 45-55 µm
     sweep range) through the same non-blocking path as any manual move; a
     failure is a log warning, not a modal. Still needs confirmation with the
     real piezo (whether 50 was ever the intended park position is Miguel's
     call, not re-litigated here — see `_agentes/_trabajo/A3_uso.md`).
   - Verify the apply buttons of the camera configuration panel. [DONE —
     2026-09-22] "Apply camera settings" now also sends the timeout, not just
     the exposure (it used to be silently dropped until the next sweep
     rebuilt the whole config). Exposure was already applying correctly.
     Real-camera confirmation still pending for the exposure value itself.

2. Review all the code, comment it and make it modular. [DONE — July 2026]
   Full audit + 5 blocks of fixes + a hardware-free test suite.
   See CHANGELOG.md (the audit records themselves are kept outside the repo).

---

## Feature ideas

- Piezo slider synchronized with the real position. [DONE — 2026-07-28]
- Direct monochrome capture ("Mono (superpixel)": weighted Bayer merge + 2x2
  binning, stacks 4x/12x smaller). [DONE — 2026-07-28] Validate in the lab
  together with item 1.
- Slim down the PyInstaller package. [DONE — 2026-07-28: 1.4 GB → 663 MB dist /
  259 MB tar.gz; the rest are real dependencies (llvmlite/opencv/scipy pulled
  in by pylablib at import time)].
- Rebuild the Windows `.pyd` with the updated CMake (needed on the lab
  machine). [DONE — 2026-08-26] Compiled with MSVC + vcpkg; the CMakeLists
  uses `/openmp:experimental` (MSVC's default `/openmp` is OpenMP 2.0 and
  rejects `#pragma omp simd`) and the spec deploys the vcpkg DLLs and the MSVC
  runtime into `_internal/`. Verified: the frozen executable starts and works
  on Windows.
