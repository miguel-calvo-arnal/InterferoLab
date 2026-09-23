"""Tests for utils/camera_constants.py — Bayer/grayscale calibration weights."""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# bayer_sites (mosaic phase -> photosite offsets)
# ---------------------------------------------------------------------------
def test_bayer_sites_default_is_rggb():
    """The default phase reproduces the historical RGGB slicing."""
    assert cc.bayer_sites() == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert cc.bayer_sites("red") == cc.bayer_sites()


def test_bayer_sites_blue_phase_is_bggr():
    """Phase "blue" (this camera's LP126CU) puts red at the opposite corner."""
    red, g1, g2, blue = cc.bayer_sites("blue")
    assert red == (1, 1)
    assert blue == (0, 0)
    assert {g1, g2} == {(1, 0), (0, 1)}


def test_bayer_sites_green_phases_are_column_and_row_shifts():
    """The two green-first phases shift red by one column or one row."""
    assert cc.bayer_sites("green_left_or_red")[0] == (0, 1)
    assert cc.bayer_sites("green_left_or_blue")[0] == (1, 0)


def test_bayer_sites_accepts_both_pylablib_spellings():
    """pylablib spells the green phases with both "or" and "of"; both work."""
    assert cc.bayer_sites("green_left_or_red") == cc.bayer_sites("green_left_of_red")
    assert cc.bayer_sites("green_left_or_blue") == cc.bayer_sites("green_left_of_blue")


def test_bayer_sites_all_four_corners_are_distinct():
    """Every phase covers the four corners of the 2x2 block exactly once."""
    for phase in cc.BAYER_RED_SITE:
        assert len(set(cc.bayer_sites(phase))) == 4


def test_bayer_sites_greens_are_diagonal_to_each_other():
    """The two greens always sit on the anti-diagonal of red/blue."""
    for phase in cc.BAYER_RED_SITE:
        red, g1, g2, blue = cc.bayer_sites(phase)
        assert g1 == (red[0], blue[1])
        assert g2 == (blue[0], red[1])


def test_bayer_sites_rejects_unknown_phase():
    """An unrecognised phase name raises instead of silently defaulting."""
    with pytest.raises(ValueError, match="Unknown Bayer mosaic phase"):
        cc.bayer_sites("cyan")
