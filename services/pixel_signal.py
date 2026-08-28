# services/pixel_signal.py
"""
Pixel interferogram helpers for the results view (pure NumPy, no Qt).

These functions mirror the corresponding C++ estimators in backend/analysis so
the display stays consistent with the reconstruction, and provide fallbacks
for datasets produced by older backend builds.
"""

from __future__ import annotations

import json
import os

import numpy as np

# Default fallback — matches cfg::BASELINE_SIGMA in config.hpp.
# Never read directly; always call baseline_sigma_for(npy_path) so that
# the value stays in sync with what the C++ backend actually used.
BASELINE_SIGMA_DEFAULT = 15.0


def baseline_sigma_for(npy_path: str) -> float:
    """
    Read baseline_sigma from the metadata.json that lives alongside the pixel
    .npy files in the same output/<dataset>/ folder.

    Falls back to BASELINE_SIGMA_DEFAULT if the file is missing or malformed,
    so the display is never broken even for datasets produced by older builds.
    """
    try:
        meta_path = os.path.join(os.path.dirname(npy_path), "metadata.json")
        with open(meta_path, encoding="utf-8") as fh:
            return float(json.load(fh).get("baseline_sigma", BASELINE_SIGMA_DEFAULT))
    except Exception:
        return BASELINE_SIGMA_DEFAULT


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    1-D Gaussian smoothing with reflect padding.
    Mirrors gaussian_apply() / gaussian_filter_1d() from the C++ backend.
    """
    r = int(np.ceil(3.0 * sigma))
    t = np.arange(-r, r + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * t**2 / sigma**2)
    kernel /= kernel.sum()
    padded = np.pad(x.astype(np.float64), r, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def refined_centroid(pos: np.ndarray, env: np.ndarray) -> float | None:
    """
    Envelope-weighted centroid of *pos*, refined once inside a ±2σ window.

    Python replica of the C++ centroid fallback used when a pixel .npy file
    does not carry the backend height estimate (column 3).
    Returns None when the envelope has no positive weight.
    """
    w = env.clip(min=0.0)
    total_w = w.sum()
    if total_w <= 0:
        return None
    z_c = float((w * pos).sum() / total_w)
    var = float((w * (pos - z_c) ** 2).sum() / total_w)
    dz = 2.0 * float(np.sqrt(max(0.0, var)))
    if dz > 0:
        mask = (pos >= z_c - dz) & (pos <= z_c + dz)
        w2 = w[mask]
        total_w2 = w2.sum()
        if total_w2 > 0:
            z_c = float((w2 * pos[mask]).sum() / total_w2)
    return z_c
