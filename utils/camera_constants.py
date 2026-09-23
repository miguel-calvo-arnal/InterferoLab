# utils/camera_constants.py
"""
Single source of truth for the Bayer superpixel calibration weights.

Camera: Thorlabs LP126CU (RGGB pattern). The weights are derived from the
camera QE curves and the IR-filter transmission, integrated against a 3200 K
blackbody spectrum (Olympus U-LH100IR, tungsten-halogen, 100 W / 12 V).

Derivation script:
    docs/Quantum efficiency and IR filter/compute_bayer_weights.py

C++ copy to keep in sync:
    backend/analysis/include/config.hpp defines the same values as the
    RGB-to-grayscale defaults for the analysis backend. Recalibrating the
    weights means updating BOTH this module and config.hpp (and recompiling
    the backend).

Two forms are provided:

- Per-site weights (BAYER_W_*): one weight per photosite of the 2x2 RGGB
  block, used when collapsing a raw Bayer frame to a mono superpixel.
- Grayscale weights (GRAY_W_*): for already-demosaiced RGB data, where the
  green channel carries the contribution of both green photosites
  (GRAY_W_G = BAYER_W_G1 + BAYER_W_G2).
"""

# Per-photosite weights of the RGGB 2x2 block
BAYER_W_R = 0.257090
BAYER_W_G1 = 0.311272  # top-right green
BAYER_W_G2 = 0.311272  # bottom-left green
BAYER_W_B = 0.120366

# RGB-to-grayscale weights for demosaiced 3-channel data.
# GRAY_W_G == BAYER_W_G1 + BAYER_W_G2 exactly (asserted below).
GRAY_W_R = 0.257090
GRAY_W_G = 0.622544
GRAY_W_B = 0.120366

assert GRAY_W_R == BAYER_W_R
assert GRAY_W_G == BAYER_W_G1 + BAYER_W_G2
assert GRAY_W_B == BAYER_W_B


# --------------------------------------------------------------------------
# Bayer mosaic phase
# --------------------------------------------------------------------------
# The weights above are per-photosite, so they only mean anything together
# with the mosaic phase: which colour sits at pixel (0, 0) of the raw frame.
# The LP126CU reports "blue" (BGGR), not the RGGB that the code assumed until
# 2026-09-09 -- assuming the wrong phase swaps the red and blue weights and,
# in colour mode, swaps the R and B channels outright.
#
# pylablib exposes the phase as get_color_info().filter_array_phase (Thorlabs
# TL_COLOR_FILTER_ARRAY_PHASE).  Its enum and its own debayering code disagree
# on the spelling ("green_left_or_red" vs "green_left_of_red"), so both are
# accepted here.
#
# Each value is the (row, col) of the RED photosite inside the 2x2 block; blue
# sits diagonally opposite it and the two greens take the other two corners.
BAYER_RED_SITE = {
    "red": (0, 0),  # RGGB
    "blue": (1, 1),  # BGGR
    "green_left_or_red": (0, 1),  # GRBG
    "green_left_of_red": (0, 1),
    "green_left_or_blue": (1, 0),  # GBRG
    "green_left_of_blue": (1, 0),
}

# Phase assumed when the camera reports none (monochrome sensor) or when no
# phase is passed: the historical RGGB behaviour.
DEFAULT_BAYER_PHASE = "red"


def bayer_sites(phase: str = DEFAULT_BAYER_PHASE):
    """
    Return the (row, col) offsets of the four photosites of a 2x2 Bayer block.

    Parameters
    ----------
    phase : mosaic phase as reported by pylablib (see BAYER_RED_SITE).

    Returns
    -------
    (red, green1, green2, blue), each a (row, col) pair with values in {0, 1}.
    green1 is the green sharing a row with red, green2 the other one; since
    BAYER_W_G1 == BAYER_W_G2 the distinction is documentary.
    """
    try:
        r_row, r_col = BAYER_RED_SITE[phase]
    except KeyError:
        raise ValueError(
            f"Unknown Bayer mosaic phase {phase!r}; expected one of {sorted(BAYER_RED_SITE)}"
        ) from None
    return (
        (r_row, r_col),
        (r_row, 1 - r_col),
        (1 - r_row, r_col),
        (1 - r_row, 1 - r_col),
    )
