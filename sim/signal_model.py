"""
Signal layer: known surface -> white-light interferogram -> raw Bayer frame.

Per photosite of colour c (R, G or B, from the mosaic phase):

    mean_DN = L_c * illum(x, y) * (t_exp / t_ref)
              * (1 + V_c * exp(-4 ln2 (d / FWHM_c)^2) * cos(2 pi d / P_c))
    d       = z_stage - h(x, y)          (in focus when the piezo is at h)

then shot + read (+ dark) noise in electrons, black level, 12-bit clip,
uint16 -- the same dtype/shape pylablib returns for a raw snap().

Speed: the fringe profile of every channel is precomputed as ONE lookup
table over the whole piezo travel (d sampled every ``lut_step_um``), and
each pixel keeps an int32 base index, so a frame is ``base + round(z/step)``
-> ``take`` -> a few in-place float32 passes.  Noise comes from a
precomputed standard-normal bank read at a random offset (a view, no copy).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

#: (row, col) of the RED photosite in the 2x2 block for each pylablib phase.
#: Mirrors utils/camera_constants.BAYER_RED_SITE (kept separate on purpose:
#: the simulator must not depend on the app).
RED_SITE = {
    "red": (0, 0),
    "blue": (1, 1),
    "green_left_or_red": (0, 1),
    "green_left_or_blue": (1, 0),
}

CH_R, CH_G, CH_B = 0, 1, 2


def bayer_channel_pattern(phase: str) -> np.ndarray:
    """2x2 array with the channel index (0=R, 1=G, 2=B) of each photosite."""
    ry, rx = RED_SITE[phase]
    pat = np.full((2, 2), CH_G, dtype=np.int8)
    pat[ry, rx] = CH_R
    pat[1 - ry, 1 - rx] = CH_B
    return pat


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
SURFACES = ("tilted_plane", "step", "sphere", "flat")


def make_surface(profile, height: int, width: int) -> np.ndarray:
    """Height map h(y, x) in um (float32, stage coordinates)."""
    kind = str(profile.get("surface.type", "tilted_plane"))
    base = float(profile.get("surface.base_height_um", 50.0))
    y = (np.arange(height, dtype=np.float32) / max(height - 1, 1) - 0.5)[:, None]
    x = (np.arange(width, dtype=np.float32) / max(width - 1, 1) - 0.5)[None, :]
    if kind == "flat":
        h = np.full((height, width), base, dtype=np.float32)
    elif kind == "tilted_plane":
        h = (
            base
            + float(profile.get("surface.tilt_x_um", 0.0)) * x
            + float(profile.get("surface.tilt_y_um", 0.0)) * y
        )
    elif kind == "step":
        step = float(profile.get("surface.step_height_um", 0.5))
        h = base + np.where(x > 0, step, 0.0).astype(np.float32) + 0.0 * y
    elif kind == "sphere":
        R = float(profile.get("surface.sphere_radius_um", 1000.0))
        cap = float(profile.get("surface.sphere_cap_um", 3.0))
        p = float(profile.get("surface.sample_pixel_um", 0.1725))
        yy = (np.arange(height, dtype=np.float64) - (height - 1) / 2)[:, None] * p
        xx = (np.arange(width, dtype=np.float64) - (width - 1) / 2)[None, :] * p
        r2 = xx * xx + yy * yy
        sag = np.sqrt(np.maximum(R * R - r2, 0.0)) - (R - cap)
        h = base + np.maximum(sag, 0.0)
    else:
        raise ValueError(f"unknown surface type {kind!r}; expected one of {SURFACES}")
    return np.ascontiguousarray(h, dtype=np.float32)


def superpixel_mean(a: np.ndarray) -> np.ndarray:
    """2x2 block mean (the resolution of the app's superpixel frames)."""
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w].astype(np.float32)
    return (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * np.float32(0.25)


# ----------------------------------------------------------------------
# Synthetic source
# ----------------------------------------------------------------------
class SyntheticSource:
    """Generates raw BGGR (or other phase) frames for a stage position z."""

    kind = "synthetic"
    #: Elements per generation chunk (~48 rows of 4096): three float32/int32
    #: scratch buffers of a chunk = 2.4 MB, inside one core's L2 cache.
    CHUNK_ELEMENTS = 48 * 4096

    def __init__(self, profile, height: int, width: int, phase: str, rng=None) -> None:
        t0 = time.perf_counter()
        self.profile = profile
        self.shape = (height, width)
        self.phase = phase
        self.rng = (
            rng if rng is not None else np.random.default_rng(int(profile.get("signal.seed", 0)))
        )

        self.bit_max = (1 << int(profile["camera.sensor.bit_depth"])) - 1
        self.gain = float(profile["camera.sensor.gain_e_per_dn"])
        self.read_dn = float(profile["camera.sensor.read_noise_e"]) / self.gain
        self.black = float(profile["camera.sensor.black_level_dn"])
        self.dark_e_s = float(profile["camera.sensor.dark_current_e_per_s"])
        self.ref_exp = float(profile["signal.reference_exposure_s"])
        self.step = float(profile.get("signal.lut_step_um", 0.001))

        # --- ground truth ----------------------------------------------
        self.height_um = make_surface(profile, height, width)

        # --- per-channel fringe LUT over the whole travel ---------------
        zmin = float(profile["piezo.stage.travel_min_um"])
        zmax = float(profile["piezo.stage.travel_max_um"])
        hmin, hmax = float(self.height_um.min()), float(self.height_um.max())
        pad = 1.0
        self._dq_min = int(np.floor((zmin - hmax - pad) / self.step))
        dq_max = int(np.ceil((zmax - hmin + pad) / self.step))
        n = dq_max - self._dq_min + 1
        d = (np.arange(n, dtype=np.float64) + self._dq_min) * self.step
        luts = []
        for c in ("r", "g", "b"):
            V = float(profile[f"signal.visibility_{c}"])
            P = float(profile[f"signal.period_{c}_um"])
            F = float(profile[f"signal.envelope_fwhm_{c}_um"])
            env = np.exp(-4.0 * np.log(2.0) * (d / F) ** 2)
            luts.append((1.0 + V * env * np.cos(2.0 * np.pi * d / P)).astype(np.float32))
        self._lut = np.concatenate(luts)  # [R | G | B]
        self._lut_n = n

        # --- per-pixel base index and level ------------------------------
        pat = bayer_channel_pattern(phase)
        ch = np.tile(pat, (height // 2 + 1, width // 2 + 1))[:height, :width]
        hq = np.rint(self.height_um / self.step).astype(np.int32)
        self._base = (ch.astype(np.int32) * n - self._dq_min) - hq  # idx = base + zq
        del hq

        levels = np.array(
            [
                float(profile["signal.level_g_dn"]) * float(profile["signal.level_ratio_r"]),
                float(profile["signal.level_g_dn"]),
                float(profile["signal.level_g_dn"]) * float(profile["signal.level_ratio_b"]),
            ],
            dtype=np.float32,
        )
        fall = float(profile.get("signal.illumination_falloff", 0.0))
        yy = (np.arange(height, dtype=np.float32) / max(height - 1, 1) - 0.5)[:, None] * 2
        xx = (np.arange(width, dtype=np.float32) / max(width - 1, 1) - 0.5)[None, :] * 2
        illum = 1.0 - fall * (xx * xx + yy * yy) / 2.0
        self._level = (levels[ch] * illum).astype(np.float32)  # DN at the reference exposure
        del ch, illum

        # --- noise bank (standard normal), read at a random offset --------
        self._npix = height * width
        self._bank_margin = 1 << 20
        self._bank = self.rng.standard_normal(self._npix + self._bank_margin, dtype=np.float32)

        # --- scratch buffers (generation is serialised by the camera) -----
        self._idx = np.empty(self.shape, dtype=np.int32)
        self._mean = np.empty(self.shape, dtype=np.float32)
        self._tmp = np.empty(self.shape, dtype=np.float32)
        self._scaled_level = None
        self._scaled_for = None

        # --- row chunks generated in parallel threads ---------------------
        # Every pass is element-wise (numpy releases the GIL), so the frame is
        # generated in chunks of rows small enough for the three scratch
        # buffers of a chunk to stay in the CPU cache (one serial pass over the
        # whole 12 MP frame is memory-bound: ~30 ms).  The result is the SAME
        # frame as a single serial pass.  Needed once the camera stays armed
        # and every frame is generated: the cost must stay below the readout
        # period.  signal.generate_threads: 0 = automatic (up to 8), 1 = serial.
        n_thr = int(profile.get("signal.generate_threads", 0) or 0)
        if n_thr <= 0:
            n_thr = max(1, min(8, os.cpu_count() or 1))
        self.threads = max(1, min(n_thr, height))
        rows_per_chunk = max(1, int(self.CHUNK_ELEMENTS // max(width, 1)))
        self._chunks = [slice(a, min(a + rows_per_chunk, height)) for a in range(0, height, rows_per_chunk)]
        self._pool = (
            ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="sim-gen")
            if self.threads > 1
            else None
        )

        self.build_time_s = time.perf_counter() - t0
        self.gen_times_s: list[float] = []

    # ------------------------------------------------------------------
    def ground_truth(self) -> dict:
        return {
            "height_um": self.height_um,
            "height_superpixel_um": superpixel_mean(self.height_um),
        }

    def describe(self) -> str:
        p = self.profile
        return (
            f"synthetic surface '{p.get('surface.type')}' (base {p.get('surface.base_height_um')} um), "
            f"{self.shape[1]}x{self.shape[0]} raw, phase {self.phase}"
        )

    def _generate_band(self, rows: slice, zq: int, dark_dn: float, off: int, out: np.ndarray) -> None:
        """All the element-wise passes for one band of rows (same maths as serial)."""
        idx = self._idx[rows]
        mean = self._mean[rows]
        tmp = self._tmp[rows]
        np.add(self._base[rows], zq, out=idx)
        np.take(self._lut, idx, out=mean, mode="clip")
        np.multiply(mean, self._scaled_level[rows], out=mean)
        if dark_dn:
            mean += np.float32(dark_dn)
        # sigma^2 [DN^2] = mean/g (shot, Gaussian approx.) + read^2
        np.multiply(mean, np.float32(1.0 / self.gain), out=tmp)
        tmp += np.float32(self.read_dn * self.read_dn)
        np.sqrt(tmp, out=tmp)
        width = self.shape[1]
        start = off + rows.start * width
        tmp *= self._bank[start : start + (rows.stop - rows.start) * width].reshape(tmp.shape)
        mean += tmp
        mean += np.float32(self.black + 0.5)  # +0.5: round, not truncate
        np.clip(mean, 0, self.bit_max, out=mean)
        out[rows] = mean  # float32 -> uint16 (values already in range)

    def generate(self, z_um: float, exposure_s: float) -> np.ndarray:
        """Raw uint16 frame (H, W) for stage position ``z_um``."""
        t0 = time.perf_counter()
        scale = exposure_s / self.ref_exp
        if self._scaled_for != scale:
            self._scaled_level = self._level * np.float32(scale)
            self._scaled_for = scale
        zq = int(round(z_um / self.step))
        dark_dn = self.dark_e_s * exposure_s / self.gain
        off = int(self.rng.integers(0, self._bank_margin))
        out = np.empty(self.shape, dtype=np.uint16)
        if self._pool is None:
            for rows in self._chunks:
                self._generate_band(rows, zq, dark_dn, off, out)
        else:
            futs = [
                self._pool.submit(self._generate_band, rows, zq, dark_dn, off, out)
                for rows in self._chunks
            ]
            for f in futs:
                f.result()
        self.gen_times_s.append(time.perf_counter() - t0)
        return out
