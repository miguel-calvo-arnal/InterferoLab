# TODO — in order of execution

Updated: 2026-08-27 (original list: 2026-04-21)

---

1. **Debug in the lab to confirm the critical paths work** (STILL PENDING, but
   with a new safety net — see CHANGELOG.md):

   - Preview image very dark / verify camera parameters are truly applied.
     [PARTIALLY COVERED] The sweep now re-reads the exposure with
     `get_exposure` and ABORTS if it differs >5% from the requested value;
     preview errors now reach the UI (they used to be lost silently). Still
     needs confirmation with the real camera.
   - Make sure the piezo moves to position 50 after connecting. [PENDING lab
     time] Movement cancellation now responds in ~0.1 s.
   - Verify the apply buttons of the camera configuration panel. [PENDING lab
     time]

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
