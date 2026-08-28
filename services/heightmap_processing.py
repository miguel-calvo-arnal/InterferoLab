# services/heightmap_processing.py
"""Heightmap post-processing helpers (pure NumPy, no Qt)."""

from __future__ import annotations

import numpy as np


def fit_and_subtract_plane(arr: np.ndarray) -> np.ndarray:
    """
    Fit a linear plane z = a·x + b·y + c to *arr* (2-D float array).

    Only pixels whose height falls in the interquartile range [Q25, Q75] are
    used for the fit — this makes the plane robust against tall features (steps,
    dust particles) that would otherwise skew the tilt estimate.

    After subtraction the corrected map is shifted so that the geometric centre
    of the image has height 0 µm (negative values are valid for pixels below
    the reference plane).

    The original array is never modified.
    """
    H, W = arr.shape
    yy, xx = np.mgrid[0:H, 0:W]

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise ValueError("Heightmap contains no finite values; cannot fit a plane.")

    finite_vals = arr[finite_mask]

    # Restrict the fit to the central height range (IQR) to exclude tall features
    q25, q75 = np.percentile(finite_vals, [25, 75])
    fit_mask = finite_mask & (arr >= q25) & (arr <= q75)
    if not fit_mask.any():
        fit_mask = finite_mask

    x_v = xx[fit_mask].astype(np.float64)
    y_v = yy[fit_mask].astype(np.float64)
    z_v = arr[fit_mask].astype(np.float64)

    # Least-squares fit: z = a·x + b·y + c
    A = np.column_stack([x_v, y_v, np.ones(len(x_v))])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_v, rcond=None)
    a, b, c = coeffs

    plane = (a * xx + b * yy + c).astype(np.float32)
    corrected = arr - plane

    # Shift so the centre pixel of the image is at 0 µm
    cy, cx = H // 2, W // 2
    if np.isfinite(corrected[cy, cx]):
        offset = float(corrected[cy, cx])
    else:
        finite_corr = corrected[np.isfinite(corrected)]
        offset = float(finite_corr.mean()) if finite_corr.size > 0 else 0.0
    return (corrected - offset).astype(np.float32)
