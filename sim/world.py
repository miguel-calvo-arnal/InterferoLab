"""
Shared state of one simulated instrument: profile, piezo stage, image
source and fault injection.  The fake camera and the fake piezo controller
both talk to the same World, so the camera sees where the piezo is.

The launcher calls ``configure()``; if nobody did (e.g. the lab probe in
--dry-run), ``get_world()`` configures the default profile on first use.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

from .hwprofile import Profile, load_profile
from .stage import PiezoStage

log = logging.getLogger("sim")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_world: World | None = None
_world_lock = threading.Lock()


class Faults:
    """Fault injection decisions, reproducible from faults.seed."""

    def __init__(self, profile: Profile) -> None:
        self.p = profile
        self.rng = np.random.default_rng(int(profile.get("faults.seed", 0)))
        self.snaps = 0
        self.gcs_commands = 0
        self._lock = threading.Lock()
        # Continuous-stream stall (faults.stream_stall_after_frames): frames
        # delivered in continuous mode, whether the stall has started, and how
        # many further arms stay stalled (see stream_frame_blocked()).
        self.stream_frames = 0
        self._stall_started = False
        self._stalled = False
        self._stalled_arms_left = 0

    def next_snap_times_out(self) -> bool:
        with self._lock:
            self.snaps += 1
            every = int(self.p.get("faults.snap_timeout_every", 0))
            if every and self.snaps % every == 0:
                return True
            prob = float(self.p.get("faults.snap_timeout_probability", 0.0))
            return bool(prob and self.rng.random() < prob)

    def stream_armed(self) -> None:
        """The camera was armed for a continuous stream (not a snap)."""
        with self._lock:
            if self._stalled:
                if self._stalled_arms_left > 0:
                    self._stalled_arms_left -= 1
                else:
                    self._stalled = False  # this arm delivers again

    def stream_frame_blocked(self) -> bool:
        """True if the next continuous-stream frame must never arrive.

        faults.stream_stall_after_frames = N (0 = never): after N frames of
        continuous streaming the armed camera stops delivering, once.
        faults.stream_stall_arms = K (default 1): the stall lasts for that arm
        and the K-1 following ones; the next arm delivers normally again
        (a large K means the camera never recovers).
        """
        with self._lock:
            if self._stalled:
                return True
            n = int(self.p.get("faults.stream_stall_after_frames", 0))
            if n and not self._stall_started and self.stream_frames >= n:
                self._stall_started = True
                self._stalled = True
                self._stalled_arms_left = max(0, int(self.p.get("faults.stream_stall_arms", 1)) - 1)
                return True
            self.stream_frames += 1
            return False

    def camera_unplugged(self) -> bool:
        n = int(self.p.get("faults.camera_disconnect_after_snaps", 0))
        return bool(n) and self.snaps > n

    def count_gcs_command(self) -> bool:
        """Count one GCS command; True if the link is (now) down."""
        with self._lock:
            self.gcs_commands += 1
            n = int(self.p.get("faults.piezo_disconnect_after_commands", 0))
            return bool(n) and self.gcs_commands > n

    def mov_fails(self) -> bool:
        prob = float(self.p.get("faults.mov_error_probability", 0.0))
        with self._lock:
            return bool(prob and self.rng.random() < prob)


class World:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.rng = np.random.default_rng(int(profile.get("signal.seed", 0)))
        self.stage = PiezoStage(profile, rng=np.random.default_rng(self.rng.integers(1 << 31)))
        self.faults = Faults(profile)
        self._source = None
        self._source_lock = threading.Lock()
        self.events: list[str] = []
        # Every fake device built against this world registers itself here, so
        # tools (perf gate, tests) can reach the camera's frame_log and the
        # controller's command_log without patching anything in the app.
        self.cameras: list = []
        self.piezos: list = []

    @property
    def camera(self):
        """The most recently opened fake camera (or None)."""
        return self.cameras[-1] if self.cameras else None

    @property
    def piezo(self):
        """The most recently created fake GCS controller (or None)."""
        return self.piezos[-1] if self.piezos else None

    @property
    def phase(self) -> str:
        return str(self.profile["camera.sensor.filter_array_phase"])

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return int(self.profile["camera.sensor.height"]), int(self.profile["camera.sensor.width"])

    def source(self):
        """The image source (built on first use: ~0.2 s, 250 MB for 12 MP)."""
        with self._source_lock:
            if self._source is None:
                h, w = self.sensor_shape
                mode = str(self.profile.get("signal.mode", "synthetic"))
                if mode == "replay":
                    from .replay import ReplaySource, find_dataset

                    folder = find_dataset(self.profile.get("replay.datasets", []), REPO_ROOT)
                    if folder is None:
                        raise RuntimeError(
                            "replay mode: none of replay.datasets exists: "
                            f"{self.profile.get('replay.datasets')}"
                        )
                    self._source = ReplaySource(self.profile, h, w, self.phase, folder)
                elif mode == "synthetic":
                    from .signal_model import SyntheticSource

                    self._source = SyntheticSource(
                        self.profile,
                        h,
                        w,
                        self.phase,
                        rng=np.random.default_rng(self.rng.integers(1 << 31)),
                    )
                else:
                    raise ValueError(f"signal.mode must be 'synthetic' or 'replay', not {mode!r}")
                log.info("[SIMULATION] image source: %s", self._source.describe())
            return self._source

    def source_if_built(self):
        return self._source

    def frame_stats(self) -> str:
        src = self._source
        if src is None or not src.gen_times_s:
            return "no frames generated"
        t = np.array(src.gen_times_s) * 1000
        return f"{len(t)} frames, generation median {np.median(t):.1f} ms, max {t.max():.1f} ms"


def configure(profile_path: str | None = None, overrides: dict | None = None) -> World:
    """(Re)build the global World from a profile file plus overrides."""
    global _world
    with _world_lock:
        _world = World(load_profile(profile_path, overrides))
        return _world


def get_world() -> World:
    global _world
    with _world_lock:
        if _world is None:
            _world = World(load_profile())
        return _world
