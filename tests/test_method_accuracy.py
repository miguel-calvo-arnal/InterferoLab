"""Accuracy gate for the September 2026 corrections of Methods 3 and 4.

The bench itself lives in ``scripts/method_accuracy_bench.py`` (runnable on its
own, English, documented there); this module runs it once and turns each of its
claims into a test:

* every method stays inside its measured error budget on synthetic
  interferograms of known height (``test_bench_gates``);
* Method 3 no longer returns heights quantised to the axial step
  (``test_m3_resolves_below_the_axial_step``);
* Method 4 no longer collapses when the envelope sits at the middle of the
  scan, where the per-bin phase step is exactly +-pi
  (``test_m4_survives_the_scan_centre``);
* **Methods 1 and 2 return exactly what they returned before the change.**
  The reference maps in ``tests/refdata/pre_m3m4_fix_*.npy`` were produced by
  the backend of commit dee0ae1, i.e. BEFORE the fix, and are compared
  bit-for-bit.  This is the test that would catch a "fix" that quietly moved
  the results of the two methods the report did not touch.

The whole module runs the backend 14 times on 300-frame 32x32 datasets and
takes well under a second.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("analysis_backend")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import method_accuracy_bench as bench  # noqa: E402  (needs the sys.path above)

REFDATA = os.path.join(_ROOT, "tests", "refdata")

# The heights are float32 micrometres; a bit-exact result differs by 0, and
# this tolerance (1e-6 um = 1 pm) only absorbs a last-ulp difference from a
# different compiler or CPU baseline.  Any real change of behaviour is orders
# of magnitude larger: the quantisation floor alone is 5.8 nm.
INVARIANCE_TOL_UM = 1e-6


@pytest.fixture(scope="module")
def bench_run(tmp_path_factory):
    """Run every bench case once; return ``{case: (stats, heightmaps)}``."""
    workdir = str(tmp_path_factory.mktemp("method_bench"))
    cwd = os.getcwd()
    os.chdir(workdir)  # the backend resolves ./output against the cwd
    try:
        return {name: bench.run_case(name, workdir, bench.N_PIXELS_FULL) for name in bench.CASES}
    finally:
        os.chdir(cwd)


def test_bench_gates(bench_run):
    """No method exceeds the error budget recorded in ``bench.GATES``."""
    failures = bench.check_gates({name: stats for name, (stats, _) in bench_run.items()})
    assert not failures, "synthetic accuracy bench failed:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("case", ["clean", "noisy"])
@pytest.mark.parametrize("method", [1, 2])
def test_methods_1_and_2_are_unchanged(bench_run, case, method):
    """M1 and M2 reproduce the pre-fix backend exactly.

    Neither the vector average (M4 only), the parabolic peak refinement (M3
    only, via ``use_peak_locator``) nor ``LAMBDA0_NM`` (read only by M3's
    kernel) is on the code path of Methods 1 and 2.  The reference arrays were
    generated with the backend built from the sources before the change.
    """
    reference = np.load(os.path.join(REFDATA, f"pre_m3m4_fix_{case}_m{method}.npy"))
    current = bench_run[case][1][method]
    assert current.shape == reference.shape
    diff = np.abs(current - reference).max()
    assert diff <= INVARIANCE_TOL_UM, (
        f"method {method} moved on the '{case}' dataset: max |diff| = {diff * 1000:.4f} nm"
    )


def test_m3_resolves_below_the_axial_step(bench_run):
    """Method 3 returns sub-step heights, not multiples of dz.

    The 1024 pixels of the 'clean' case have 1024 different heights spread
    over 1.6 um, i.e. 80 axial steps.  The discrete peak locator could only
    return those ~81 grid values (measured: 81 distinct heights); the
    parabolic refinement returns a distinct value for essentially every pixel.
    """
    heights = bench_run["clean"][1][3].ravel()
    distinct = np.unique(heights).size
    assert distinct > heights.size // 2, (
        f"only {distinct} distinct heights out of {heights.size}: "
        "Method 3 looks quantised to the axial step again"
    )
    # And the residual scatter is far below the dz/sqrt(12) = 5.8 nm floor.
    assert bench_run["clean"][0][3]["rms"] < 2.0


def test_m4_survives_the_scan_centre(bench_run):
    """Method 4 no longer fails where the per-bin phase step is +-pi.

    Every pixel of the 'centre' case has its envelope within half a step of
    index Nz/2.  Averaging the per-bin *angles* mixed the +pi and -pi branches
    there and placed the surface at the bottom of the scan (measured worst
    case: 2.8 um).  The vector average has no branch to mix.
    """
    stats = bench_run["centre"][0][4]
    assert abs(stats["max_abs"]) < 100.0, f"worst-case error {stats['max_abs']:.0f} nm"
    # Sanity: it is now as good as Method 1 on the same data, not merely finite.
    assert stats["rms"] < 2.0 * bench_run["centre"][0][1]["rms"]


def test_m3_bias_on_an_asymmetric_envelope_is_bounded(bench_run):
    """The documented price of the parabolic refinement stays bounded.

    On a split-normal envelope (trailing side 1.6x wider) Method 3 reads
    about -104 nm low.  That offset is NOT the parabola: the discrete locator
    gave -103.60 nm and the parabola gives -103.62 nm on the very same data.
    It is the skew of the 5-point kernel's envelope estimate, which was always
    there.  What the refinement does remove is the 5.8 nm of quantisation on
    top of it.
    """
    stats = bench_run["asym"][0][3]
    assert abs(stats["bias"]) < 120.0
    assert stats["rms"] < 2.0
