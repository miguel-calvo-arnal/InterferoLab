"""Simulator signal layer and profile: BGGR phase, 12-bit range, noise, ground truth.

These tests import only sim.signal_model / sim.hwprofile (no fake drivers),
so they run in-process without shadowing the real pylablib/pipython.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sim.hwprofile import ProfileError, load_profile, toml_figure
from sim.signal_model import SyntheticSource, bayer_channel_pattern, make_surface, superpixel_mean


def _fig(value):
    return {"value": value, "origin": "estimated", "source": "test override"}


def _small_profile(h=64, w=96, **surface):
    over = {
        "camera": {"sensor": {"height": _fig(h), "width": _fig(w)}},
        "surface": {"type": "flat", **surface},
    }
    return load_profile(overrides=over)


# ----------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------
def test_default_profile_every_figure_has_origin_and_source():
    prof = load_profile()
    figs = prof.figures()
    assert len(figs) > 30
    for key, _value, origin, source in figs:
        assert origin in ("published", "measured", "derived", "estimated"), key
        assert source.strip(), key


def test_published_figures_cite_a_link_or_document():
    for key, _v, origin, source in load_profile().figures():
        if origin == "published":
            assert "http" in source or "page" in source or "same" in source, key


def test_profile_rejects_bad_origin(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('[camera.timing]\narm_s = { value = 0.1, origin = "guess", source = "x" }\n')
    with pytest.raises(ProfileError):
        load_profile(str(bad))


def test_measured_profile_merges_over_default(tmp_path):
    lab = tmp_path / "lab.toml"
    lab.write_text(
        "[camera.timing]\narm_s = " + toml_figure(0.42, "measured", "lab, 10 samples") + "\n"
    )
    prof = load_profile(str(lab))
    assert prof["camera.timing.arm_s"] == pytest.approx(0.42)
    assert prof.origin("camera.timing.arm_s") == "measured"
    # untouched figures keep the default
    assert prof["camera.timing.disarm_s"] == load_profile()["camera.timing.disarm_s"]


# ----------------------------------------------------------------------
# Mosaic, range, levels
# ----------------------------------------------------------------------
def test_bggr_pattern_for_blue_phase():
    pat = bayer_channel_pattern("blue")
    # 0=R, 1=G, 2=B: blue at (0,0), red at (1,1)
    assert pat.tolist() == [[2, 1], [1, 0]]


def test_frame_is_bggr_uint16_12bit():
    prof = _small_profile(base_height_um=50.0)
    src = SyntheticSource(prof, 64, 96, "blue")
    f = src.generate(40.0, 0.007)  # 10 um away from focus: background only
    assert f.dtype == np.uint16 and f.shape == (64, 96)
    assert int(f.max()) <= 4095
    b, g1, g2, r = (f[y::2, x::2].astype(float).mean() for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)))
    g = 0.5 * (g1 + g2)
    # raw channel ratios from the QE curves: R/G = 0.826, B/G = 0.387
    assert b < r < g
    assert r / g == pytest.approx(0.826, rel=0.05)
    assert b / g == pytest.approx(0.387, rel=0.07)


def test_saturation_clips_at_4095():
    prof = _small_profile()
    src = SyntheticSource(prof, 64, 96, "blue")
    f = src.generate(40.0, 0.007 * 20)  # 20x the reference exposure
    assert int(f.max()) == 4095
    assert (f == 4095).mean() > 0.5


def test_noise_matches_shot_plus_read():
    prof = _small_profile(128, 128)
    src = SyntheticSource(prof, 128, 128, "blue")
    a = src.generate(40.0, 0.007).astype(float)
    b = src.generate(40.0, 0.007).astype(float)
    g = a[0::2, 1::2]
    mean = g.mean()
    diff_sigma = np.std(a[0::2, 1::2] - b[0::2, 1::2]) / np.sqrt(2)
    gain = prof["camera.sensor.gain_e_per_dn"]
    read = prof["camera.sensor.read_noise_e"] / gain
    expected = np.sqrt(mean / gain + read**2)
    assert diff_sigma == pytest.approx(expected, rel=0.15)


# ----------------------------------------------------------------------
# Fringes and ground truth
# ----------------------------------------------------------------------
def test_envelope_peak_and_period_follow_ground_truth():
    """Each pixel's G envelope peaks where the piezo equals its height."""
    over = {
        "camera": {"sensor": {"height": _fig(16), "width": _fig(16)}},
        "surface": {
            "type": "tilted_plane",
            "base_height_um": 50.0,
            "tilt_x_um": 2.0,
            "tilt_y_um": 0.0,
        },
        "signal": {"illumination_falloff": _fig(0.0)},
    }
    prof = load_profile(overrides=over)
    src = SyntheticSource(prof, 16, 16, "blue")
    zs = np.arange(47.0, 53.0, 0.01)
    stack = np.stack([src.generate(z, 0.007) for z in zs]).astype(float)
    gt = src.ground_truth()["height_um"]
    y, x = 0, 1  # a G site
    for col in (1, 7, 15):
        s = stack[:, y, col] - np.median(stack[:, y, col])
        env = np.convolve(np.abs(s), np.ones(25) / 25, "same")
        z_peak = zs[np.argmax(env)]
        assert z_peak == pytest.approx(gt[y, col], abs=0.08)
    # fringe period of the G channel ~ period_g_um
    s = stack[:, y, x] - stack[:, y, x].mean()
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=1 << 15))
    fr = np.fft.rfftfreq(1 << 15, 0.01)
    spec[fr < 1] = 0
    assert 1 / fr[np.argmax(spec)] == pytest.approx(prof["signal.period_g_um"], rel=0.05)


@pytest.mark.parametrize("kind", ["tilted_plane", "step", "sphere", "flat"])
def test_surfaces_and_superpixel_ground_truth(kind):
    prof = load_profile(overrides={"surface": {"type": kind}})
    h = make_surface(prof, 60, 80)
    assert h.shape == (60, 80) and h.dtype == np.float32
    sp = superpixel_mean(h)
    assert sp.shape == (30, 40)
    assert np.all(np.isfinite(sp))
    if kind == "step":
        assert h[:, -1].mean() - h[:, 0].mean() == pytest.approx(
            prof["surface.step_height_um"], abs=1e-4
        )
    if kind == "sphere":
        assert h.max() > h.min()


def test_full_frame_generation_time_is_reported():
    """The generator must not be the bottleneck: 12 MP well under 1 s (typ. ~30 ms)."""
    prof = load_profile()
    src = SyntheticSource(prof, 3000, 4096, "blue")
    t0 = time.perf_counter()
    for i in range(3):
        f = src.generate(50.0 + 0.02 * i, 0.007)
    per_frame = (time.perf_counter() - t0) / 3
    assert f.shape == (3000, 4096) and f.dtype == np.uint16
    assert per_frame < 0.5, f"frame generation {per_frame * 1000:.0f} ms"
    assert len(src.gen_times_s) == 3
