"""Synthetic accuracy bench for the four CSI reconstruction methods.

Why this exists
---------------
The physical-verification report of 23-Sep-2026 found two estimator defects
that the plateau tests of ``tests/test_backend_reconstruction.py`` cannot see,
because those tests only assert a 2*dz tolerance on the median of a plateau:

* **Method 4** averaged the *angles* of the per-bin cross products
  ``arg(X[k+1]*conj(X[k]))``.  Near the middle of the scan the per-bin step is
  exactly +-pi, so the average mixes the two branches and collapses to ~0; and
  the band half-width ``dk/2`` is inflated by the noise floor, so hundreds of
  signal-free bins enter the mean.  Measured error: 0.45-2.3 um.
* **Method 3** returned the scan position of the discrete envelope maximum, so
  its heights were quantised to the step: an irreducible dz/sqrt(12) = 5.8 nm
  at dz = 20 nm.

Both were fixed in the C++ backend (vector average for M4, parabolic peak
refinement for M3).  This bench measures the error of every method on
synthetic interferograms of *known* height and fails if it degrades, so the
fix cannot be silently undone.  It is the gate of that change.

What is synthesised
-------------------
For a pixel of height ``h`` the frame at piezo position ``z`` carries

    I(z) = DC + AMP * G(z - h) * cos(4*pi*(z - h)/lambda + phi0) + noise

with a Gaussian coherence envelope ``G`` of coherence length ``lc`` (FWHM).
The bench uses the parameters of the reference report: Nz = 300 frames,
dz = 20 nm, lambda = 566 nm, lc = 1.5 um, DC = 1000 DN, AMP = 400 DN,
12-bit quantisation, and noise sigma = 20 DN (5 % of the fringe amplitude)
in the noisy cases.  Heights are deliberately placed OFF the dz grid and each
pixel gets its own fringe phase ``phi0``, because both estimator defects are
phase- and sub-step-dependent.

``dz`` is 20 nm on purpose: it is the value of ``cfg::NOMINAL_DZ_NM``, so the
inter-frame phase step alpha that Method 3 uses is the physically correct one.

The four cases
--------------
=========  ==========================================================
clean      no added noise (only 12-bit quantisation).  Exposes the M3
           quantisation floor and the M4 pi-branch failure.
noisy      sigma = 20 DN.  The realistic regime: M4's inflated band.
centre     sigma = 20 DN and every height within +-0.5 steps of the
           scan centre, i.e. n_peak = Nz/2, the M4 singularity.
asym       no noise, split-normal (skewed) envelope.  The envelope
           maximum is still exactly at ``h`` by construction, so the
           mean error is a pure systematic offset.  Measured: it is
           the skew of the 5-point kernel's own envelope estimate
           (-103.60 nm with the discrete locator, -103.62 nm with the
           parabola), not the price of the refinement.
=========  ==========================================================

Usage
-----
    .venv/bin/python scripts/method_accuracy_bench.py            # full bench
    .venv/bin/python scripts/method_accuracy_bench.py --quick    # fewer pixels
    .venv/bin/python scripts/method_accuracy_bench.py --json out.json

Exit code 0 = every gate passed, 1 = at least one gate failed.  The same
thresholds are asserted from ``tests/test_method_accuracy.py`` so the gate
also runs inside the normal test suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "backend"), os.path.join(_ROOT, "backend", "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------
# Acquisition parameters of the synthetic scan (report of 23-Sep-2026)
# --------------------------------------------------------------------------
NZ = 300  # frames
DZ_UM = 0.020  # axial step  (= cfg::NOMINAL_DZ_NM)
LAMBDA_UM = 0.566  # mean wavelength of the fringe carrier
LC_UM = 1.5  # coherence length (FWHM of the envelope)
DC_DN = 1000.0  # background level
AMP_DN = 400.0  # fringe amplitude
NOISE_DN = 20.0  # read+shot noise of the "noisy" cases (5 % of AMP)
SEED = 20260924

# Heights are kept well inside the scan: Method 2 and Method 4 need about
# 2 um of margin to the scan edges before truncation biases them (report
# D1-07).  The scan spans 0 .. (NZ-1)*DZ = 5.98 um.
H_LOW_UM = 2.2
H_HIGH_UM = 3.8
# Sub-step offset so that no height falls on a multiple of dz.
H_OFFSET_UM = 0.0073

SIGMA_FROM_FWHM = 1.0 / 2.3548200450309493  # sigma = FWHM / (2 sqrt(2 ln 2))

METHODS = (1, 2, 3, 4)

# The backend always writes representative pixel traces with a hardcoded
# margin of 10 px (utils.cpp), so the synthetic images must be larger than
# 20x20.  32x32 = 1024 pixels keeps the bench under a second per method.
_PIXEL_PLOT_MARGIN = 10
N_PIXELS_FULL = 32 * 32
N_PIXELS_QUICK = 24 * 24


# --------------------------------------------------------------------------
# Gates.  rms/bias in NANOMETRES, measured on the heights the backend returns.
# The comment on each line is the value measured on 24-Sep-2026 after the fix
# (and, where it matters, the value the old estimator gave).
# --------------------------------------------------------------------------
GATES: dict[tuple[str, int], dict[str, float]] = {
    # case    method   limit      measured 24-Sep-2026 (old estimator)
    ("clean", 1): {"rms": 4.0},  # 1.30   (1.30 — unchanged)
    ("clean", 2): {"rms": 8.0},  # 0.73   (0.73 — unchanged)
    ("clean", 3): {"rms": 2.0},  # 0.23   (5.78 = dz/sqrt(12), quantisation)
    ("clean", 4): {"rms": 5.0},  # 0.85   (88.5, with 2790 nm outliers)
    ("noisy", 1): {"rms": 25.0},  # 10.3  (10.3 — unchanged)
    ("noisy", 2): {"rms": 45.0},  # 17.3  (17.3 — unchanged)
    ("noisy", 3): {"rms": 30.0},  # 13.6  (14.5)
    ("noisy", 4): {"rms": 60.0},  # 8.9   (808)
    ("centre", 4): {"rms": 60.0},  # 8.6  (2327 — the +-pi singularity)
    # Asymmetric envelope.  "bias" is the mean error and is NOT the price of
    # the parabola: the discrete locator gave -103.60 nm on this case and the
    # parabola gives -103.62 nm.  The offset is the skew of the 5-point kernel
    # itself and was always there; a sweep over skew 1.0-3.0 puts the
    # parabola's own contribution below 0.7 nm (see _agentes/_trabajo/
    # C1_backend.md).  The rms limit is what catches a revert to the discrete
    # locator (5.8 nm) on a skewed envelope.
    ("asym", 3): {"bias": 120.0, "rms": 2.0},  # -103.6 / 0.34
}


def envelope(delta_um: np.ndarray, lc_um: float = LC_UM, skew: float = 1.0) -> np.ndarray:
    """Coherence envelope, maximum exactly at ``delta = 0``.

    ``skew`` = 1 gives a Gaussian of FWHM ``lc_um``.  ``skew`` > 1 widens the
    trailing side only (split normal): the maximum stays at delta = 0 but the
    envelope is no longer symmetric, which is what a broadband halogen source
    with a long red tail produces in practice.
    """
    sigma = lc_um * SIGMA_FROM_FWHM
    sig = np.where(delta_um > 0.0, sigma * skew, sigma)
    return np.exp(-(delta_um**2) / (2.0 * sig**2))


def frame(
    heights_um: np.ndarray,
    phases: np.ndarray,
    z_um: float,
    noise_dn: float,
    rng: np.random.Generator,
    skew: float = 1.0,
) -> np.ndarray:
    """One 12-bit interferogram frame of the synthetic scan."""
    delta = z_um - heights_um
    signal = DC_DN + AMP_DN * envelope(delta, skew=skew) * np.cos(
        4.0 * np.pi * delta / LAMBDA_UM + phases
    )
    if noise_dn > 0.0:
        signal = signal + rng.normal(0.0, noise_dn, size=signal.shape)
    return np.clip(np.round(signal), 0, 4095).astype(np.uint16)


def height_field(n_pixels: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """``(heights_um, phases)`` for ``n_pixels`` pixels, as square 2-D images.

    ``n_pixels`` must be a perfect square (1024 -> 32x32): the
    backend works on images, and a one-row image would exercise an edge case
    of the row-chunking rather than the estimators.

    ``mode`` = "spread": heights fill [H_LOW_UM, H_HIGH_UM] off the dz grid.
    ``mode`` = "centre": every height puts the envelope peak within +-0.5
    steps of index Nz/2 after the backend's Z flip, i.e. exactly on the
    +-pi per-bin step where the old Method 4 collapsed.
    """
    side = int(round(n_pixels**0.5))
    if side * side != n_pixels:
        raise ValueError(f"n_pixels must be a perfect square, got {n_pixels}")
    if side <= 2 * _PIXEL_PLOT_MARGIN:
        raise ValueError(
            f"the image side ({side}) must exceed 2*{_PIXEL_PLOT_MARGIN}: the backend always "
            "saves representative pixel traces with that margin"
        )
    if mode == "spread":
        heights = np.linspace(H_LOW_UM, H_HIGH_UM, n_pixels) + H_OFFSET_UM
    elif mode == "centre":
        z_max = (NZ - 1) * DZ_UM
        h_centre = z_max - (NZ // 2) * DZ_UM  # reported height = Nz/2 * dz
        heights = h_centre + np.linspace(-0.5, 0.5, n_pixels) * DZ_UM
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown height mode {mode!r}")
    # Golden-ratio phase sequence: uniform coverage of [0, 2 pi) and no
    # correlation with the height ordering.
    phases = 2.0 * np.pi * ((np.arange(n_pixels) * 0.6180339887) % 1.0)
    return heights.reshape(side, side), phases.reshape(side, side)


def write_dataset(
    folder: str,
    heights_um: np.ndarray,
    phases: np.ndarray,
    noise_dn: float,
    skew: float = 1.0,
    seed: int = SEED,
) -> None:
    """Write the whole synthetic scan as mono 16-bit PNG frames."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(NZ):
        z = i * DZ_UM
        img = frame(heights_um, phases, z, noise_dn, rng, skew=skew)
        path = os.path.join(folder, f"piezo_{z:.4f}um_{i:04d}.png")
        if not cv2.imwrite(path, img):
            raise RuntimeError(f"could not write {path}")


def expected_height(heights_um: np.ndarray) -> np.ndarray:
    """What the backend reports: it flips the Z axis (p -> max(p) - p)."""
    return (NZ - 1) * DZ_UM - heights_um


def run_method(folder: str, method: int, name: str) -> np.ndarray:
    """Run the C++ backend on ``folder`` and return the height map [um]."""
    import analysis_backend

    result = analysis_backend.run_analysis(
        folder, name=name, method=method, log_cb=lambda level, msg: None
    )
    if result["cancelled"]:
        raise RuntimeError(f"analysis cancelled for method {method}")
    return np.load(result["heightmap"])


def error_stats(heightmap: np.ndarray, heights_um: np.ndarray) -> dict[str, float]:
    """Height error statistics in nanometres (bias removed for ``rms``)."""
    err_nm = (heightmap.ravel() - expected_height(heights_um).ravel()) * 1000.0
    return {
        "bias": float(err_nm.mean()),
        "rms": float(err_nm.std()),
        "rms_total": float(np.sqrt((err_nm**2).mean())),
        "max_abs": float(np.abs(err_nm).max()),
    }


# --------------------------------------------------------------------------
# Case definitions
# --------------------------------------------------------------------------
CASES: dict[str, dict] = {
    "clean": {"mode": "spread", "noise": 0.0, "skew": 1.0, "methods": METHODS},
    "noisy": {"mode": "spread", "noise": NOISE_DN, "skew": 1.0, "methods": METHODS},
    "centre": {"mode": "centre", "noise": NOISE_DN, "skew": 1.0, "methods": (1, 4)},
    "asym": {"mode": "spread", "noise": 0.0, "skew": 1.6, "methods": (1, 3)},
}


def run_case(name: str, workdir: str, n_pixels: int) -> tuple[dict[int, dict], np.ndarray]:
    """Build one synthetic dataset, run its methods and return the stats.

    Returns ``(stats_by_method, heightmap_by_method)``; the second value keeps
    the raw maps so callers (the invariance test) can compare them.
    """
    spec = CASES[name]
    heights, phases = height_field(n_pixels, spec["mode"])
    folder = os.path.join(workdir, f"bench_{name}")
    write_dataset(folder, heights, phases, spec["noise"], skew=spec["skew"])

    stats, maps = {}, {}
    for method in spec["methods"]:
        hm = run_method(folder, method, f"{name}_m{method}")
        stats[method] = error_stats(hm, heights)
        maps[method] = hm
    return stats, maps


def check_gates(results: dict[str, dict[int, dict]]) -> list[str]:
    """Return the list of gate failures (empty = pass)."""
    failures = []
    for (case, method), limits in GATES.items():
        if case not in results or method not in results[case]:
            failures.append(f"{case}/M{method}: not measured")
            continue
        st = results[case][method]
        for key, limit in limits.items():
            value = abs(st[key])
            if value > limit:
                failures.append(f"{case}/M{method}: |{key}| = {value:.1f} nm > {limit:.1f} nm")
    return failures


def bench(n_pixels: int = N_PIXELS_FULL, workdir: str | None = None) -> dict[str, dict[int, dict]]:
    """Run every case and return ``{case: {method: stats}}``."""
    owned = workdir is None
    tmp = tempfile.TemporaryDirectory(prefix="ilab_bench_") if owned else None
    workdir = tmp.name if tmp is not None else workdir
    cwd = os.getcwd()
    try:
        os.chdir(workdir)  # run_analysis resolves ./output against the cwd
        return {name: run_case(name, workdir, n_pixels)[0] for name in CASES}
    finally:
        os.chdir(cwd)
        if tmp is not None:
            tmp.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"{N_PIXELS_QUICK} pixels per case instead of {N_PIXELS_FULL}",
    )
    parser.add_argument("--json", metavar="PATH", help="also write the numbers as JSON")
    args = parser.parse_args(argv)

    results = bench(n_pixels=N_PIXELS_QUICK if args.quick else N_PIXELS_FULL)

    print(
        f"Synthetic bench: Nz={NZ}, dz={DZ_UM * 1000:.0f} nm, lambda={LAMBDA_UM * 1000:.0f} nm, "
        f"lc={LC_UM} um, DC={DC_DN:.0f} DN, amp={AMP_DN:.0f} DN"
    )
    print(f"{'case':8s} {'method':>6s} {'bias':>10s} {'rms':>10s} {'max|e|':>10s}   gate")
    for case, per_method in results.items():
        for method, st in per_method.items():
            limits = GATES.get((case, method))
            gate = ", ".join(f"|{k}| <= {v:g}" for k, v in limits.items()) if limits else "-"
            print(
                f"{case:8s} {method:6d} {st['bias']:9.1f}n {st['rms']:9.1f}n "
                f"{st['max_abs']:9.1f}n   {gate}"
            )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json}")

    failures = check_gates(results)
    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: every method is within its gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
