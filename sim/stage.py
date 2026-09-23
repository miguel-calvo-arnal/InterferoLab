"""
Piezo stage model (P-611.ZS driven by an E-625.SR), shared by the fake
pipython controller (which moves it) and the fake camera (which samples its
position at mid-exposure, so a frame taken during settling is blurred in z
exactly like on the real instrument... to first order: we sample one z).

Closed loop: after MOV the position approaches the target exponentially with
time constant settle_s / 4.6 (within 1 % of the step at settle_s) and the
controller reports on-target (ONT) from settle_s on.  A small Gaussian
position error (position_noise_um) is drawn per move.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np


class PiezoStage:
    def __init__(self, profile, rng: np.random.Generator | None = None) -> None:
        self.travel_min = float(profile["piezo.stage.travel_min_um"])
        self.travel_max = float(profile["piezo.stage.travel_max_um"])
        self.settle_s = float(profile["piezo.timing.settle_s"])
        self.noise_um = float(profile["piezo.stage.position_noise_um"])
        self.um_per_v = float(profile["piezo.stage.open_loop_um_per_v"])
        self.never_on_target = bool(profile.get("faults.piezo_never_on_target", False))
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._lock = threading.Lock()
        self.servo = False
        self._from = 0.0
        self._to = 0.0
        self._t_cmd = time.monotonic() - 10.0
        self.voltage = 0.0

    # -- commands (called by the fake controller) ----------------------
    def move(self, target_um: float) -> None:
        with self._lock:
            now = time.monotonic()
            self._from = self._position_locked(now)
            self._to = float(target_um) + float(self._rng.normal(0.0, self.noise_um))
            self._t_cmd = now

    def set_voltage(self, volts: float) -> None:
        """Open loop: position follows the voltage (no settling model)."""
        with self._lock:
            now = time.monotonic()
            self.voltage = float(volts)
            self._from = self._position_locked(now)
            self._to = min(max(self.voltage * self.um_per_v, self.travel_min), self.travel_max)
            self._t_cmd = now

    # -- queries ---------------------------------------------------------
    def _position_locked(self, t: float) -> float:
        dt = t - self._t_cmd
        if dt <= 0:
            return self._from
        tau = self.settle_s / 4.6 if self.settle_s > 0 else 1e-9
        return self._to + (self._from - self._to) * math.exp(-dt / tau)

    def position(self, t: float | None = None) -> float:
        with self._lock:
            return self._position_locked(time.monotonic() if t is None else t)

    def on_target(self, t: float | None = None) -> bool:
        if self.never_on_target:
            return False
        with self._lock:
            now = time.monotonic() if t is None else t
            return (now - self._t_cmd) >= self.settle_s
