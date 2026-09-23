"""
Replay mode: serve a REAL stack by z, re-mosaicked to a raw Bayer frame.

This is an APPROXIMATION and says so in the log:

* The real stacks on disk are already demosaiced (3 channels).  To look like
  a raw snap() we keep, at each photosite, only the channel that photosite
  would have seen (BGGR for the LP126CU).  The discarded channels were
  interpolated by the original debayer, so the result is smoother and has
  less noise than a true raw frame; pylablib's colour-correction matrix
  (if the stack came from it) is baked in.
* 16-bit TIFF stacks of the LP126CU (4096x3000, 12-bit values) are used at
  their native size and scale.
* 8-bit PNG stacks (data/S1F1, data/S1F5: 1936x1216 RGBA from ANOTHER
  camera and setup) are centre-cropped to the LP126CU aspect ratio,
  resized to 2048x1500 per channel (linear) and scaled 0-255 -> 0-4095.
* The frame for a stage position z is the stack frame with the nearest z
  (stack z + offset).  No noise is added and exposure is ignored.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict

import cv2
import numpy as np

from .signal_model import CH_B, CH_G, CH_R, bayer_channel_pattern

log = logging.getLogger("sim.replay")

_Z_RE = re.compile(r"piezo_([+-]?\d+(?:\.\d+)?)um", re.IGNORECASE)
_EXTS = (".tif", ".tiff", ".png")


def find_dataset(candidates, repo_root: str) -> str | None:
    for c in candidates:
        path = c if os.path.isabs(c) else os.path.join(repo_root, c)
        if os.path.isdir(path):
            return path
    return None


class ReplaySource:
    kind = "replay"

    def __init__(self, profile, height: int, width: int, phase: str, folder: str) -> None:
        self.profile = profile
        self.shape = (height, width)
        self.phase = phase
        self.folder = folder
        self.bit_max = (1 << int(profile["camera.sensor.bit_depth"])) - 1
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in _EXTS]
        items = []
        for f in files:
            m = _Z_RE.search(f)
            if m:
                items.append((float(m.group(1)), f))
        if not items:
            raise RuntimeError(f"replay: no piezo_<z>um.* frames in {folder}")
        items.sort()
        self.z = np.array([z for z, _ in items])
        self.files = [os.path.join(folder, f) for _, f in items]
        off = profile.get("replay.z_offset_um", "auto")
        if off == "auto":
            base = float(profile.get("surface.base_height_um", 50.0))
            inside = self.z[0] <= base <= self.z[-1]
            off = 0.0 if inside else base - 0.5 * (self.z[0] + self.z[-1])
        self.z_offset = float(off)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_n = int(profile.get("replay.cache_frames", 6))
        self._lock = threading.Lock()
        self.gen_times_s: list[float] = []
        self.build_time_s = 0.0
        self._warm: set[int] = set()
        pat = bayer_channel_pattern(phase)
        self._sites = [(y, x, int(pat[y, x])) for y in (0, 1) for x in (0, 1)]
        log.warning(
            "[SIMULATION] REPLAY MODE is an APPROXIMATION: %d demosaiced frames from %s are "
            "re-mosaicked to %s raw; not a true raw readout (see sim/README.md). z offset %+.3f um.",
            len(self.files),
            folder,
            phase,
            self.z_offset,
        )

    def describe(self) -> str:
        return (
            f"replay of {self.folder} ({len(self.files)} frames, z {self.z[0]:.3f}-{self.z[-1]:.3f} um "
            f"+ offset {self.z_offset:+.3f}); APPROXIMATION: re-mosaicked from demosaiced data"
        )

    def ground_truth(self) -> dict:
        return {}

    def _load_rgb(self, path: str) -> np.ndarray:
        """(H/2, W/2, 3) or (H, W, 3) float32 RGB in 12-bit DN."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"replay: cannot read {path}")
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        rgb = img[:, :, 2::-1] if img.shape[2] >= 3 else img  # BGR(A) -> RGB
        H, W = self.shape
        if img.dtype == np.uint8:
            h, w = rgb.shape[:2]
            target_ar = W / H
            cw = min(w, int(round(h * target_ar)))
            ch = min(h, int(round(cw / target_ar)))
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            rgb = rgb[y0 : y0 + ch, x0 : x0 + cw]
            rgb = cv2.resize(
                np.ascontiguousarray(rgb), (W // 2, H // 2), interpolation=cv2.INTER_LINEAR
            )
            return rgb.astype(np.float32) * np.float32(self.bit_max / 255.0)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(np.ascontiguousarray(rgb), (W, H), interpolation=cv2.INTER_LINEAR)
        return rgb.astype(np.float32)

    def _mosaic(self, i: int) -> np.ndarray:
        rgb = self._load_rgb(self.files[i])
        H, W = self.shape
        out = np.empty((H, W), dtype=np.uint16)
        half = rgb.shape[0] == H // 2
        for y, x, c in self._sites:
            chan = {CH_R: 0, CH_G: 1, CH_B: 2}[c]
            src = rgb[:, :, chan] if half else rgb[y::2, x::2, chan]
            out[y::2, x::2] = np.clip(np.rint(src), 0, self.bit_max).astype(np.uint16)
        return out

    def generate(self, z_um: float, exposure_s: float) -> np.ndarray:  # noqa: ARG002
        import time

        t0 = time.perf_counter()
        i = int(np.argmin(np.abs(self.z + self.z_offset - z_um)))
        with self._lock:
            frame = self._cache.get(i)
            if frame is not None:
                self._cache.move_to_end(i)
        if frame is None:
            frame = self._mosaic(i)
            with self._lock:
                self._cache[i] = frame
                while len(self._cache) > self._cache_n:
                    self._cache.popitem(last=False)
        self.gen_times_s.append(time.perf_counter() - t0)
        self._prefetch(i)
        return frame.copy()

    def _prefetch(self, i: int) -> None:
        """Warm the OS page cache for the neighbours of frame i (a sweep walks
        z monotonically): a cold 73 MB TIFF costs ~0.3 s from a hard disk,
        a warm one ~15 ms.  Plain reads in a daemon thread, no decoding."""
        todo = [
            j for j in (i + 1, i + 2, i - 1) if 0 <= j < len(self.files) and j not in self._warm
        ]
        if not todo:
            return
        self._warm.update(todo)

        def warm(paths):
            for path in paths:
                try:
                    with open(path, "rb") as fh:
                        while fh.read(8 << 20):
                            pass
                except OSError:
                    pass

        threading.Thread(
            target=warm,
            args=([self.files[j] for j in todo],),
            daemon=True,
            name="sim-replay-prefetch",
        ).start()
