"""Tests for services/heightmap_processing.py: robust plane fit/subtraction."""

from __future__ import annotations

import numpy as np
import pytest

from services.heightmap_processing import fit_and_subtract_plane


def _plane(h: int, w: int, a: float, b: float, c: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (a * xx + b * yy + c).astype(np.float64)


def test_pure_plane_gives_zero_residual():
    """Fitting a pure tilted plane leaves a residual of ~0 everywhere."""
    arr = _plane(40, 50, a=0.02, b=-0.015, c=3.0)
    out = fit_and_subtract_plane(arr)
    assert np.abs(out).max() < 1e-4


def test_step_survives_and_tilt_removed():
    """A tall step survives plane removal while the background tilt vanishes."""
    arr = _plane(40, 50, a=0.02, b=0.03, c=1.0)
    step_mask = np.zeros(arr.shape, dtype=bool)
    step_mask[:8, :10] = True  # ~10% of pixels, far from the centre pixel
    arr[step_mask] += 10.0

    out = fit_and_subtract_plane(arr)

    background = out[~step_mask]
    # Background is flat (tilt removed) and step height is preserved.
    assert background.std() < 1e-3
    step_height = out[step_mask].mean() - background.mean()
    assert step_height == pytest.approx(10.0, abs=1e-3)


def test_centre_pixel_shifted_to_zero():
    """The geometric centre of the corrected map sits at exactly 0."""
    arr = _plane(30, 31, a=0.05, b=0.01, c=-2.0) + 0.7
    out = fit_and_subtract_plane(arr)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    assert out[cy, cx] == pytest.approx(0.0, abs=1e-6)


def test_input_array_is_not_mutated():
    """The original heightmap array is left untouched."""
    arr = _plane(20, 22, a=0.1, b=0.2, c=5.0)
    arr[3:5, 3:5] += 4.0
    before = arr.copy()
    fit_and_subtract_plane(arr)
    np.testing.assert_array_equal(arr, before)


def test_nan_pixels_are_ignored_for_fit_and_propagate():
    """NaN pixels do not break the fit and stay NaN in the output."""
    arr = _plane(30, 30, a=0.02, b=0.01, c=1.0)
    arr[0:3, 0:3] = np.nan
    out = fit_and_subtract_plane(arr)
    assert np.isnan(out[0:3, 0:3]).all()
    finite = out[np.isfinite(out)]
    assert finite.size == arr.size - 9
    assert np.abs(finite).max() < 1e-4


def test_all_nan_raises_value_error():
    """A heightmap with no finite values raises ValueError."""
    arr = np.full((10, 10), np.nan)
    with pytest.raises(ValueError, match="no finite values"):
        fit_and_subtract_plane(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_is_float32_for_both_input_dtypes(dtype):
    """float32 and float64 inputs both produce a float32 corrected map."""
    arr = _plane(16, 18, a=0.01, b=0.02, c=0.5).astype(dtype)
    out = fit_and_subtract_plane(arr)
    assert out.dtype == np.float32
    assert np.abs(out).max() < 1e-3


def test_constant_map_yields_zero():
    """A perfectly flat map is corrected to all zeros."""
    arr = np.full((12, 14), 7.25, dtype=np.float64)
    out = fit_and_subtract_plane(arr)
    assert np.abs(out).max() < 1e-5


def test_nan_centre_falls_back_to_mean_offset():
    """With a NaN centre pixel, the offset falls back to the finite mean."""
    arr = _plane(21, 21, a=0.03, b=0.04, c=2.0)
    arr[10, 10] = np.nan  # centre pixel
    out = fit_and_subtract_plane(arr)
    finite = out[np.isfinite(out)]
    # Residual is ~0 everywhere, so the mean-offset shift keeps it ~0.
    assert abs(finite.mean()) < 1e-5
    assert np.isnan(out[10, 10])
