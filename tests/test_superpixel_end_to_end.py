"""End-to-end: sweep in 'Mono (superpixel)' mode -> C++ analysis_backend.

A mocked AcquisitionSession runs a real run_sweep() over synthetic Bayer
interferograms (position-dependent frames, no hardware), saving directly in
mono_superpixel mode; the resulting folders are then fed unmodified to the
C++ analysis backend (method 2), which must produce a finite heightmap that
recovers the known plateau heights.

Frame construction: each raw Bayer frame is a 4x4 upsampling (np.repeat) of
the target interferogram, so the weighted RGGB merge (weights sum to 1) and
the subsequent 2x2 binning reduce it back to the target frame exactly. This
makes the on-disk stack sample-identical to helpers_backend.csi_frame and the
reconstruction tolerances directly comparable to the existing backend tests.
"""

from __future__ import annotations

import os

import helpers_backend as hb
import numpy as np
import pytest

# The C++ backend (.so) may not be compiled: skip cleanly.
pytest.importorskip("analysis_backend")

from helpers_acquisition import FakeCamera, FakePiezo, make_session

# Tolerance for the recovered plateau medians: two axial steps
# (same criterion as tests/test_backend_reconstruction.py).
TOL_UM = 2 * hb.DZ_UM


class _SweepCamera(FakeCamera):
    """Camera whose snap() returns one precomputed frame per sweep step."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        super().__init__(frame=frames[0])
        self.frames = frames

    def snap(self, timeout: float | None = None) -> np.ndarray:
        idx = self.snap_count
        self.snap_count += 1
        return self.frames[idx]


def _sweep_cfg(output_folder: str, fmt: str) -> dict:
    return {
        "start": 0.0,
        "end": (hb.NZ - 1) * hb.DZ_UM,
        "step": hb.DZ_UM,
        "exposure": 0.07,
        "timeout": 1.0,
        "format": fmt,
        "output_folder": output_folder,
        "settle": 0.0,
        "closed_loop": True,
        "axis": "A",
        "color_mode": "mono_superpixel",
    }


@pytest.fixture(scope="module")
def synthetic_bayer_frames():
    """Raw Bayer frames whose superpixel-binned reduction is a known CSI stack."""
    heights, low_mask, high_mask = hb.two_plateau_heights()
    targets = [hb.csi_frame(heights, z) for z in hb.z_positions()]
    raw = [np.repeat(np.repeat(t, 4, axis=0), 4, axis=1) for t in targets]
    return heights, low_mask, high_mask, targets, raw


def _run_sweep(tmp_path, fmt: str, raw_frames: list[np.ndarray]) -> str:
    session = make_session()
    session._camera = _SweepCamera(raw_frames)
    session._piezo = FakePiezo()
    try:
        folder = session.run_sweep(_sweep_cfg(str(tmp_path / f"data_{fmt}"), fmt))
    finally:
        session._camera = None
        session._piezo = None
    return folder


@pytest.mark.parametrize("fmt", ["bin12", "png"])
def test_superpixel_sweep_feeds_analysis_backend(in_tmp_cwd, tmp_path, synthetic_bayer_frames, fmt):
    """A mono_superpixel sweep (bin12 and png) is analyzable by the C++ backend."""
    heights, low_mask, high_mask, targets, raw = synthetic_bayer_frames
    folder = _run_sweep(tmp_path, fmt, raw)

    files = sorted(os.listdir(folder))
    assert len(files) == hb.NZ
    assert all(f.endswith("." + fmt) for f in files)

    # The saved reduction must be sample-identical to the target interferogram
    # (upsample -> weighted merge -> 2x2 binning is an exact round-trip here).
    if fmt == "bin12":
        from utils import bin12

        with open(os.path.join(folder, files[0]), "rb") as fh:
            fh.seek(bin12.HEADER_SIZE)
            first = bin12.unpack_pixels(fh.read(), hb.NY * hb.NX).reshape(hb.NY, hb.NX)
    else:
        import cv2

        first = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(first, targets[0])

    # C++ analysis (method 2) on the acquisition output, unmodified.
    result = hb.run_backend(folder, name=f"sp_{fmt}", method=2)
    assert result["cancelled"] is False

    hm = np.load(result["heightmap"])
    assert hm.shape == heights.shape
    assert np.isfinite(hm).all()

    expected = hb.expected_height(heights)
    for mask in (low_mask, high_mask):
        err = abs(np.median(hm[mask]) - np.median(expected[mask]))
        assert err < TOL_UM, f"plateau median error {err:.4f} um > {TOL_UM} um"


def test_superpixel_bin12_and_png_reconstruct_identically(
    in_tmp_cwd, tmp_path, synthetic_bayer_frames
):
    """bin12 and png superpixel sweeps carry the same samples -> same heightmap."""
    _heights, _low, _high, _targets, raw = synthetic_bayer_frames
    hm_b12 = np.load(
        hb.run_backend(_run_sweep(tmp_path, "bin12", raw), name="eq_b12", method=2)["heightmap"]
    )
    hm_png = np.load(
        hb.run_backend(_run_sweep(tmp_path, "png", raw), name="eq_png", method=2)["heightmap"]
    )
    assert np.array_equal(hm_b12, hm_png)
