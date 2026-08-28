"""Tests for utils/camera_constants.py — Bayer/grayscale calibration weights."""

from __future__ import annotations

from utils import camera_constants as cc


def test_gray_green_is_sum_of_bayer_greens():
    """GRAY_W_G equals BAYER_W_G1 + BAYER_W_G2 exactly (float-identical)."""
    assert cc.GRAY_W_G == cc.BAYER_W_G1 + cc.BAYER_W_G2


def test_gray_matches_per_site_red_and_blue():
    """GRAY_W_R/GRAY_W_B are float-identical to their per-photosite weights."""
    assert cc.GRAY_W_R == cc.BAYER_W_R
    assert cc.GRAY_W_B == cc.BAYER_W_B


def test_gray_weights_sum_to_one():
    """Grayscale weights sum to 1 within 1e-12 (the literals happen to sum to exactly 1.0 in binary64, but only normalization to ~1 is guaranteed by the calibration)."""
    assert abs(cc.GRAY_W_R + cc.GRAY_W_G + cc.GRAY_W_B - 1.0) < 1e-12


def test_bayer_weights_sum_to_one():
    """The four per-photosite RGGB weights sum to 1 within 1e-12."""
    total = cc.BAYER_W_R + cc.BAYER_W_G1 + cc.BAYER_W_G2 + cc.BAYER_W_B
    assert abs(total - 1.0) < 1e-12


def test_exact_calibration_values_regression():
    """Regression guard against divergence from the C++ side: these exact values are duplicated in backend/analysis/include/config.hpp (GRAY_WEIGHT_*); if this test fails, the weights were recalibrated and config.hpp must be updated (and the backend recompiled) — or vice versa."""
    assert cc.GRAY_W_R == 0.257090
    assert cc.GRAY_W_G == 0.622544
    assert cc.GRAY_W_B == 0.120366
    assert cc.BAYER_W_G1 == cc.BAYER_W_G2 == 0.311272


def test_all_weights_are_positive_fractions():
    """Every weight is a float strictly between 0 and 1."""
    weights = [
        cc.BAYER_W_R,
        cc.BAYER_W_G1,
        cc.BAYER_W_G2,
        cc.BAYER_W_B,
        cc.GRAY_W_R,
        cc.GRAY_W_G,
        cc.GRAY_W_B,
    ]
    assert all(isinstance(w, float) and 0.0 < w < 1.0 for w in weights)
