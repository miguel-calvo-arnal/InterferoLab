"""The reconstruction must not depend on how much memory the machine has free.

The backend streams the dataset in row-chunks whose size ``auto_row_chunk()``
derives from ``MemAvailable``.  Every per-pixel computation is independent of
that split, but one value was not: Methods 2 and 4 estimate a single bandpass
``{k_avg, dk}`` for the whole image from a sample of Z-traces, and that sample
used to be "256 pixels of the FIRST CHUNK, with stride (R·Nx)/256".  A machine
with different free RAM got a different R, a different set of 256 pixels, a
different mean amplitude spectrum and — through one bin of ``dk`` — a different
band for **every** pixel of the image.  Measured on the real stacks with a
purpose-built binary carrying the old sampling (and independently in report
C3 §5.2):

    data/S1F1, Method 2: {k_avg, dk} = {45, 103} with one chunk and {45, 104}
                         with several; 99.95 % of the pixels changed, max 22.7 nm
    data/S1F5, Method 2: 99.97 % of the pixels changed, max 58.6 nm

The sample is now a fixed ``cfg::BAND_SAMPLE_ROWS`` × ``cfg::BAND_SAMPLE_COLS``
grid over the first rows of the image, so nothing outside the dataset reaches
the result.

What each test is worth
-----------------------
``INTERFEROLAB_ROW_CHUNK`` (honoured by ``auto_row_chunk``) is what lets a test
vary the chunking without touching the machine's memory; it is exactly the
knob ``MemAvailable`` used to turn.  The forced value goes through the same
limits as the RAM estimate (16 ≤ R ≤ 4096, R ≤ Ny), so it can only produce a
chunking the application itself could produce — asking for 8 rows gives 16.
That clamp is not cosmetic: with 8 rows the sampling grid would shrink to
8 × 16 = 128 traces and change the band, which is how the reviewer caught it.

**What the fix does not buy.** A fixed grid gives reproducibility, not
independence from the sample: ``dk = ceil(2·sqrt(var))`` sits within ±0.3 bins
of an integer on both real stacks while the spread between 256-trace samples is
±0.3–0.5 bins, so the grid fixes ``dk`` by convention (on ``data/S1F5`` the
grid's 53 is the minority answer; 62 % of random samples give 54), and one bin
is worth 1.3–1.4 nm in Method 2.  See ``cfg::BAND_SAMPLE_ROWS``.

**Be honest about the synthetic tests.** The failure is a rounding knife-edge
(``dk = ceil(2·sqrt(var))`` crossing an integer), and the synthetic fields
below did NOT trip it even with the old sampling: their mean amplitude spectrum
is too stable.  They are kept because they state the contract and are cheap,
but the test that would really have caught this bug is
``test_real_stack_is_reproducible_across_chunk_sizes``, which needs the real
``data/S1F1`` and is skipped when it is not there.
"""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("analysis_backend")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_STACK = os.path.join(_ROOT, "data", "S1F1")

NZ = 300  # frames of the synthetic scans
DZ_UM = 0.020  # axial step
SIDE = 32  # image side (must exceed 2*10: the pixel-plot margin)
SIGMA_FROM_FWHM = 1.0 / 2.3548200450309493

# One chunk for the whole image, then the floor auto_row_chunk may return.
ROW_CHUNKS = ("32", "16")

# Child process: reconstruct ./dataset with one method, save the map, print the band.
_CHILD = """
import os, sys
import numpy as np
sys.path[:0] = [{root!r}, {backend!r}, {analysis!r}]
import analysis_backend
os.chdir(sys.argv[1])
r = analysis_backend.run_analysis(sys.argv[4], name="probe", method=int(sys.argv[2]),
                                  log_cb=lambda a, b: None)
np.save(sys.argv[3], np.load(r["heightmap"]))
print("BAND", r["k_avg"], r["dk"])
"""


def _write_stress_field(folder: str, drift: bool, seed: int = 7) -> None:
    """Write a scan whose spectrum changes from the top row to the bottom one.

    The variation runs along Y, which is the axis the chunking cuts, so the
    sampled population really does change with the chunk size.
    """
    os.makedirs(folder, exist_ok=True)
    row = np.arange(SIDE)[:, None] / (SIDE - 1) * np.ones((1, SIDE))
    col = np.ones((SIDE, 1)) * np.arange(SIDE)[None, :] / (SIDE - 1)

    heights = 2.2 + 1.6 * col + 0.0073  # off the dz grid, away from the edges
    phases = 2.0 * np.pi * ((np.arange(SIDE * SIDE) * 0.6180339887) % 1.0)
    phases = phases.reshape(SIDE, SIDE)
    lc = 0.6 + 2.4 * row  # coherence length  [um]
    lam = 0.45 + 0.30 * row  # fringe carrier    [um]
    if drift:
        # Method 2 estimates its band on the RAW signal, so it only reacts to
        # a sample change when the low-frequency content varies too.
        background = 700.0 + 900.0 * row
        slope = (-0.25 + 0.5 * row) * background
        amplitude = 250.0 + 350.0 * (1.0 - row)
    else:
        background = np.full_like(row, 1000.0)
        slope = np.zeros_like(row)
        amplitude = np.full_like(row, 400.0)

    rng = np.random.default_rng(seed)
    sigma = lc * SIGMA_FROM_FWHM
    for i in range(NZ):
        z = i * DZ_UM
        delta = z - heights
        env = np.exp(-(delta**2) / (2.0 * sigma**2))
        ramp = (i / (NZ - 1)) - 0.5
        signal = (
            background
            + slope * ramp
            + amplitude * env * np.cos(4.0 * np.pi * delta / lam + phases)
            + rng.normal(0.0, 20.0, size=env.shape)
        )
        img = np.clip(np.round(signal), 0, 4095).astype(np.uint16)
        path = os.path.join(folder, f"piezo_{z:.4f}um_{i:04d}.png")
        if not cv2.imwrite(path, img):
            raise RuntimeError(f"could not write {path}")


@pytest.fixture(scope="module")
def stress_fields(tmp_path_factory):
    """Both stress scans, written once for the whole module."""
    fields = {}
    for name, drift in (("gradient", False), ("drift", True)):
        workdir = tmp_path_factory.mktemp(f"repro_{name}")
        _write_stress_field(str(workdir / "dataset"), drift=drift)
        fields[name] = workdir
    return fields


def _reconstruct(workdir, method: int, row_chunk: str, dataset: str = "dataset"):
    """Run the backend in a child process with a forced row-chunk size.

    Returns ``(height_map, (k_avg, dk))``.
    """
    out = workdir / f"m{method}_R{row_chunk}.npy"
    code = _CHILD.format(
        root=_ROOT,
        backend=os.path.join(_ROOT, "backend"),
        analysis=os.path.join(_ROOT, "backend", "analysis"),
    )
    env = dict(os.environ, INTERFEROLAB_ROW_CHUNK=row_chunk)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(workdir), str(method), str(out), dataset],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"backend failed (R={row_chunk}):\n{proc.stderr}"
    band = next(line.split()[1:] for line in proc.stdout.splitlines() if line.startswith("BAND"))
    return np.load(out), (int(band[0]), int(band[1]))


@pytest.mark.parametrize("field", ["gradient", "drift"])
@pytest.mark.parametrize("method", [1, 2, 3, 4])
def test_height_map_does_not_depend_on_the_row_chunking(stress_fields, field, method):
    """Same dataset, two chunk splittings, one height map — bit for bit."""
    workdir = stress_fields[field]
    reference, band_ref = _reconstruct(workdir, method, ROW_CHUNKS[0])
    for row_chunk in ROW_CHUNKS[1:]:
        current, band = _reconstruct(workdir, method, row_chunk)
        assert band == band_ref, (
            f"method {method} on '{field}': the global bandpass changed with "
            f"INTERFEROLAB_ROW_CHUNK={row_chunk}, {band_ref} -> {band}. The band sample "
            "depends on the chunking, i.e. on the machine's free RAM."
        )
        diff = np.abs(current - reference)
        assert np.array_equal(current, reference), (
            f"method {method} on '{field}' changed with "
            f"INTERFEROLAB_ROW_CHUNK={row_chunk}: {100 * np.mean(diff > 0):.2f} % of the "
            f"pixels, max {diff.max() * 1000:.3f} nm."
        )


def test_methods_1_and_3_have_no_global_band(stress_fields):
    """Only Methods 2 and 4 carry a value shared by the whole image.

    If a future method starts sharing something else across pixels, it has to
    go through the same fixed-grid discipline; this test is the reminder.
    """
    workdir = stress_fields["gradient"]
    for method in (1, 3):
        _, band = _reconstruct(workdir, method, ROW_CHUNKS[0])
        assert band == (-1, -1), f"method {method} reported a global band {band}"
    for method in (2, 4):
        _, band = _reconstruct(workdir, method, ROW_CHUNKS[0])
        assert band[0] > 0 and band[1] > 0, f"method {method} reported no band: {band}"


@pytest.mark.skipif(
    not os.path.isdir(_REAL_STACK),
    reason="needs the real stack data/S1F1 (not in the repository)",
)
@pytest.mark.parametrize("method", [2, 4])
def test_real_stack_is_reproducible_across_chunk_sizes(tmp_path, method):
    """The test that would have caught the bug: a real 1216x1936 stack.

    With the pre-September-2026 sampling this failed for Method 2 with
    {45, 103} against {45, 104} and 99.95 % of the pixels different (max
    22.7 nm).  R = 233 is what ~4 GB of free RAM produced on this stack.
    """
    reference, band_ref = _reconstruct(tmp_path, method, "1216", dataset=_REAL_STACK)
    current, band = _reconstruct(tmp_path, method, "233", dataset=_REAL_STACK)
    assert band == band_ref, f"method {method}: band {band_ref} -> {band} with 233-row chunks"
    diff = np.abs(current - reference)
    assert np.array_equal(current, reference), (
        f"method {method} on data/S1F1 changed with 233-row chunks: "
        f"{100 * np.mean(diff > 0):.2f} % of the pixels, max {diff.max() * 1000:.3f} nm"
    )


@pytest.mark.parametrize("method", [2, 4])
def test_a_chunk_below_the_floor_is_clamped_instead_of_shrinking_the_grid(stress_fields, method):
    """Asking for fewer rows than the floor must not shrink the band sample.

    ``rows_s = min(R, BAND_SAMPLE_ROWS)``, so a chunk of 8 rows would sample
    8 x 16 = 128 traces instead of 256 and give a different band — which is
    what the reviewer found: the one knob meant to PROVE chunk-independence
    was the one way to break it.  ``auto_row_chunk`` now pushes any forced
    value through the same limits as its own estimate, so 8, 4 and 1 all come
    back as 16 and reconstruct identically.
    """
    workdir = stress_fields["drift"]
    reference, band_ref = _reconstruct(workdir, method, "16")
    for below_floor in ("8", "1"):
        current, band = _reconstruct(workdir, method, below_floor)
        assert band == band_ref, (
            f"method {method}: INTERFEROLAB_ROW_CHUNK={below_floor} gave band {band} "
            f"instead of {band_ref}; the forced value is skipping the 16-row floor and "
            "shrinking the sampling grid."
        )
        assert np.array_equal(current, reference), (
            f"method {method}: INTERFEROLAB_ROW_CHUNK={below_floor} changed the height map"
        )


def test_the_band_sample_grid_fits_in_the_smallest_possible_chunk():
    """The sampling grid must never need more rows than a chunk is sure to hold.

    ``auto_row_chunk`` clamps every answer — the RAM estimate, the fallback
    constant and the environment override — to at least 16 rows (and to Ny),
    so a grid of at most 16 rows is always fully present in the first chunk.
    If someone raises ``cfg::BAND_SAMPLE_ROWS`` above that floor, or removes
    the single clamping helper, the sample silently becomes chunk-dependent
    again on low-memory machines.
    """
    with open(
        os.path.join(_ROOT, "backend", "analysis", "include", "config.hpp"), encoding="utf-8"
    ) as fh:
        rows = int(fh.read().split("BAND_SAMPLE_ROWS = ")[1].split(";")[0])
    with open(
        os.path.join(_ROOT, "backend", "analysis", "src", "utils.cpp"), encoding="utf-8"
    ) as fh:
        utils = fh.read()
    assert "if (R < 16)" in utils, "auto_row_chunk no longer has its 16-row floor"
    assert utils.count("clamp_row_chunk(") >= 4, (
        "not every path of auto_row_chunk goes through the clamp: the definition plus the "
        "override, the two fallbacks and the RAM estimate"
    )
    assert rows <= 16, f"BAND_SAMPLE_ROWS = {rows} exceeds the guaranteed chunk floor of 16"
