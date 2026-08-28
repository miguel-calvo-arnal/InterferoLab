"""Shared synthetic-dataset generators for the analysis_backend C++ tests.

Physics of the synthetic data
-----------------------------
For every pixel with surface height ``h`` (micrometres) the coherence-scanning
interferogram sampled at piezo position ``z`` is::

    I(z) = DC + A * exp(-(z - h)^2 / (2 sigma^2)) * cos(4 pi (z - h) / lambda)

with ``lambda`` ~ 0.6 um and ``sigma`` ~ 0.5 um.  Intensities are quantised to
12 bits (0..4095) so that a 16-bit PNG and a .bin12 file carry EXACTLY the same
integer samples (the C++ reader uses raw values for both formats).

Height convention of the backend
--------------------------------
The backend parses the piezo position from each filename, then flips the axis
(``p -> max(p) - p``) before locating the envelope.  A surface at height ``h``
is therefore reconstructed as ``z_max - h`` where ``z_max = (Nz - 1) * dz``.
Use :func:`expected_height` to convert a true height into the value the
backend reports.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from utils import bin12

# Default acquisition parameters (small + fast, yet well resolved:
# fringe period lambda/2 = 0.3 um = 6 samples at dz = 0.05 um).
LAMBDA_UM = 0.6
SIGMA_UM = 0.5
DZ_UM = 0.05
NZ = 120
NY = 48
NX = 64

# Two flat plateaus (fractions of the scanned range) used by the
# reconstruction accuracy tests.  Kept away from the scan edges so the
# envelope (after the backend's smoothing) is not clipped.
PLATEAU_LOW_FRAC = 0.38
PLATEAU_HIGH_FRAC = 0.62


def z_positions(nz: int = NZ, dz: float = DZ_UM) -> np.ndarray:
    """Piezo positions of the synthetic scan, in micrometres."""
    return np.arange(nz, dtype=np.float64) * dz


def expected_height(h_um: float | np.ndarray, nz: int = NZ, dz: float = DZ_UM):
    """Value the backend reports for a surface at true height ``h_um``.

    The backend flips the Z axis (p -> max(p) - p), so a surface whose
    envelope peaks at piezo position h is reported as z_max - h.
    """
    return (nz - 1) * dz - h_um


def two_plateau_heights(
    ny: int = NY,
    nx: int = NX,
    nz: int = NZ,
    dz: float = DZ_UM,
    low_frac: float = PLATEAU_LOW_FRAC,
    high_frac: float = PLATEAU_HIGH_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height map with two flat plateaus (left half low, right half high).

    Returns ``(heights_um, low_mask, high_mask)``.  The masks exclude a
    margin around the image border and around the step between plateaus, so
    tests can take robust per-plateau medians.
    """
    z_max = (nz - 1) * dz
    h_low = low_frac * z_max
    h_high = high_frac * z_max

    heights = np.full((ny, nx), h_low, dtype=np.float64)
    heights[:, nx // 2 :] = h_high

    margin = 6
    low_mask = np.zeros((ny, nx), dtype=bool)
    high_mask = np.zeros((ny, nx), dtype=bool)
    low_mask[margin:-margin, margin : nx // 2 - margin] = True
    high_mask[margin:-margin, nx // 2 + margin : -margin] = True
    return heights, low_mask, high_mask


def csi_frame(
    heights_um: np.ndarray,
    z_um: float,
    lambda_um: float = LAMBDA_UM,
    sigma_um: float = SIGMA_UM,
) -> np.ndarray:
    """One synthetic interferogram frame, quantised to 12 bits (uint16).

    Deterministic (no noise), so PNG and bin12 datasets built from it are
    sample-for-sample identical.
    """
    delta = z_um - heights_um
    envelope = np.exp(-(delta**2) / (2.0 * sigma_um**2))
    fringes = np.cos(4.0 * np.pi * delta / lambda_um)
    intensity = 0.5 + 0.4 * envelope * fringes  # in [0.1, 0.9]
    return np.clip(np.round(intensity * 4095.0), 0, 4095).astype(np.uint16)


def frame_filename(z_um: float, index: int, ext: str) -> str:
    """Dataset filename following the acquisition convention."""
    return f"piezo_{z_um:.4f}um_{index:04d}.{ext}"


def write_png_dataset(
    folder: str | os.PathLike,
    heights_um: np.ndarray,
    nz: int = NZ,
    dz: float = DZ_UM,
    lambda_um: float = LAMBDA_UM,
    sigma_um: float = SIGMA_UM,
) -> list[str]:
    """Write a mono 16-bit PNG dataset; returns the written file paths."""
    folder = os.fspath(folder)
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, z in enumerate(z_positions(nz, dz)):
        frame = csi_frame(heights_um, z, lambda_um, sigma_um)
        path = os.path.join(folder, frame_filename(z, i, "png"))
        assert cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def write_bin12_dataset(
    folder: str | os.PathLike,
    heights_um: np.ndarray,
    nz: int = NZ,
    dz: float = DZ_UM,
    lambda_um: float = LAMBDA_UM,
    sigma_um: float = SIGMA_UM,
) -> list[str]:
    """Write a mono .bin12 dataset with the same samples as the PNG variant."""
    folder = os.fspath(folder)
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, z in enumerate(z_positions(nz, dz)):
        frame = csi_frame(heights_um, z, lambda_um, sigma_um)
        path = os.path.join(folder, frame_filename(z, i, "bin12"))
        with open(path, "wb") as fh:
            fh.write(bin12.pack_frame(frame))
        paths.append(path)
    return paths


def write_color_png_dataset(
    folder: str | os.PathLike,
    heights_r_um: np.ndarray,
    heights_b_um: np.ndarray,
    nz: int = NZ,
    dz: float = DZ_UM,
) -> list[str]:
    """Color dataset whose R channel images a surface at ``heights_r_um`` and
    whose B channel images a different surface at ``heights_b_um`` (G flat).

    Channel weights then select which surface the reconstruction sees:
    [1, 0, 0] must recover heights_r_um and [0, 0, 1] heights_b_um.
    """
    folder = os.fspath(folder)
    os.makedirs(folder, exist_ok=True)
    flat_g = np.full_like(heights_r_um, 2048, dtype=np.uint16)
    paths = []
    for i, z in enumerate(z_positions(nz, dz)):
        r = csi_frame(heights_r_um, z)
        b = csi_frame(heights_b_um, z)
        bgr = np.dstack([b, flat_g, r])  # cv2 writes BGR order
        path = os.path.join(folder, frame_filename(z, i, "png"))
        assert cv2.imwrite(path, bgr)
        paths.append(path)
    return paths


def run_backend(dataset_folder, **kwargs):
    """Thin wrapper around analysis_backend.run_analysis with quiet logging."""
    import analysis_backend

    kwargs.setdefault("log_cb", lambda level, msg: None)
    return analysis_backend.run_analysis(os.fspath(dataset_folder), **kwargs)
