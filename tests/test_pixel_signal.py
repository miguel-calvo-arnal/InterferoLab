"""Tests for services/pixel_signal.py: smoothing, centroid, baseline sigma."""

from __future__ import annotations

import json

import numpy as np
import pytest

from services.pixel_signal import (
    BASELINE_SIGMA_DEFAULT,
    baseline_sigma_for,
    gaussian_smooth,
    refined_centroid,
)


# ---------------------------------------------------------------------------
# gaussian_smooth
# ---------------------------------------------------------------------------
def test_smooth_constant_stays_constant():
    """Smoothing a constant signal returns the same constant (kernel sums to 1)."""
    x = np.full(64, 3.75)
    out = gaussian_smooth(x, sigma=4.0)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, 3.75, rtol=0, atol=1e-12)


def test_smooth_impulse_is_symmetric():
    """An impulse away from the edges yields a symmetric response."""
    n, centre = 51, 25
    x = np.zeros(n)
    x[centre] = 1.0
    out = gaussian_smooth(x, sigma=2.0)
    for k in range(1, 12):
        assert out[centre - k] == pytest.approx(out[centre + k], abs=1e-15)
    assert out.argmax() == centre


def test_smooth_impulse_conserves_sum():
    """The normalised kernel conserves the total signal energy of an impulse."""
    x = np.zeros(101)
    x[50] = 1.0
    out = gaussian_smooth(x, sigma=3.0)
    assert out.sum() == pytest.approx(1.0, abs=1e-12)


def test_smooth_tiny_sigma_is_identity():
    """A very small sigma degenerates to (numerically) the identity."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    out = gaussian_smooth(x, sigma=0.05)
    np.testing.assert_allclose(out, x, atol=1e-12)


def test_smooth_output_length_matches_input():
    """Reflect padding + 'valid' convolution preserves the signal length."""
    for n in (5, 17, 200):
        assert gaussian_smooth(np.arange(n, dtype=float), sigma=2.5).shape == (n,)


def test_smooth_reduces_noise_variance():
    """Smoothing white noise reduces its variance substantially."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    out = gaussian_smooth(x, sigma=5.0)
    assert out.std() < 0.5 * x.std()


# ---------------------------------------------------------------------------
# refined_centroid
# ---------------------------------------------------------------------------
def test_centroid_recovers_gaussian_centre():
    """A synthetic Gaussian envelope centred at z0 is located within one step."""
    pos = np.linspace(0.0, 10.0, 201)  # step = 0.05
    z0 = 4.63
    env = np.exp(-0.5 * ((pos - z0) / 1.2) ** 2)
    z_est = refined_centroid(pos, env)
    assert z_est is not None
    assert abs(z_est - z0) < 0.05


def test_centroid_off_grid_centre():
    """The refined centroid resolves sub-step positions of the envelope peak."""
    pos = np.linspace(-5.0, 5.0, 101)  # step = 0.1
    z0 = 1.234
    env = np.exp(-0.5 * ((pos - z0) / 0.8) ** 2)
    z_est = refined_centroid(pos, env)
    assert z_est == pytest.approx(z0, abs=0.1)


def test_centroid_flat_zero_envelope_returns_none():
    """An all-zero envelope has no positive weight -> None."""
    pos = np.linspace(0.0, 1.0, 50)
    assert refined_centroid(pos, np.zeros(50)) is None


def test_centroid_negative_envelope_returns_none():
    """A strictly negative envelope is clipped to zero weight -> None."""
    pos = np.linspace(0.0, 1.0, 50)
    assert refined_centroid(pos, np.full(50, -3.0)) is None


def test_centroid_ignores_negative_lobes():
    """Negative envelope lobes are clipped and do not bias the estimate."""
    pos = np.linspace(0.0, 10.0, 201)
    z0 = 6.0
    env = np.exp(-0.5 * ((pos - z0) / 0.9) ** 2)
    env[pos < 2.0] = -5.0  # large negative lobe far from the peak
    z_est = refined_centroid(pos, env)
    assert z_est == pytest.approx(z0, abs=0.05)


# ---------------------------------------------------------------------------
# baseline_sigma_for
# ---------------------------------------------------------------------------
def test_baseline_sigma_read_from_metadata(tmp_path):
    """baseline_sigma is read from metadata.json next to the .npy file."""
    (tmp_path / "metadata.json").write_text(json.dumps({"baseline_sigma": 22.5}))
    npy_path = str(tmp_path / "ds_pixel_y1_x2.npy")
    assert baseline_sigma_for(npy_path) == 22.5


def test_baseline_sigma_missing_file_uses_default(tmp_path):
    """A missing metadata.json falls back to BASELINE_SIGMA_DEFAULT."""
    npy_path = str(tmp_path / "ds_pixel_y1_x2.npy")
    assert baseline_sigma_for(npy_path) == BASELINE_SIGMA_DEFAULT


def test_baseline_sigma_missing_key_uses_default(tmp_path):
    """metadata.json without the key falls back to the default."""
    (tmp_path / "metadata.json").write_text(json.dumps({"other": 1}))
    assert baseline_sigma_for(str(tmp_path / "p.npy")) == BASELINE_SIGMA_DEFAULT


def test_baseline_sigma_malformed_json_uses_default(tmp_path):
    """Malformed JSON falls back to the default instead of raising."""
    (tmp_path / "metadata.json").write_text("{not valid json")
    assert baseline_sigma_for(str(tmp_path / "p.npy")) == BASELINE_SIGMA_DEFAULT


def test_baseline_sigma_non_numeric_value_uses_default(tmp_path):
    """A non-numeric baseline_sigma value falls back to the default."""
    (tmp_path / "metadata.json").write_text(json.dumps({"baseline_sigma": "abc"}))
    assert baseline_sigma_for(str(tmp_path / "p.npy")) == BASELINE_SIGMA_DEFAULT
