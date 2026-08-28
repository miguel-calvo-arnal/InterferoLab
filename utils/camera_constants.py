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
