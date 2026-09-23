"""Tests for the camera owner thread (backend/acquisition/camera_owner.py)
running on a real AcquisitionSession with FakeCamera / FakePiezo.

The design under test: once connected, exactly ONE thread uses the camera
object; the live preview arms the camera once and hands out the newest frame;
exposure changes, sweeps and the close all run on that thread; an unresponsive
camera is abandoned, never touched from another thread.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from helpers_acquisition import (
    FakeCamera,
    FakeCameraTimeoutError,
    FakePiezo,
    LogCollector,
    make_session,
)

from backend.acquisition.acquisition_controller import AcquisitionSession, CancelFlag
from backend.acquisition.camera_owner import OwnerCallbacks, describe_error


def _wait(cond, timeout=3.0) -> bool:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


class Events:
    """Records everything the owner reports (from its thread)."""

    def __init__(self) -> None:
        self.frames: list[tuple[np.ndarray, float]] = []
        self.states: list[bool] = []
        self.errors: list[tuple[str, str]] = []
        self.log = LogCollector()
        self.finished: list[tuple[str, int, int]] = []
        self.notices: list[str] = []
        self.aborted: list[str] = []
        self.callback_threads: set[int] = set()

    def callbacks(self) -> OwnerCallbacks:
        def on_frame(frame, z):
            self.callback_threads.add(threading.get_ident())
            self.frames.append((frame, float(z)))

        return OwnerCallbacks(
            on_frame=on_frame,
            on_preview_state=self.states.append,
            on_error=lambda src, msg: self.errors.append((src, msg)),
            on_log=self.log,
            on_sweep_finished=lambda folder, s, t: self.finished.append((folder, s, t)),
            on_preview_notice=self.notices.append,
            on_sweep_aborted=self.aborted.append,
        )


def _constant_frames(values, h=8, w=8) -> list[np.ndarray]:
    return [np.full((h, w), v, dtype=np.uint16) for v in values]


@pytest.fixture
def owned():
    """Session with a fake camera + piezo attached and an owner thread ready."""
    session = make_session()
    session.CAMERA_CLOSE_TIMEOUT_S = 2.0
    cam = FakeCamera(stream_frames=_constant_frames([100, 200, 300, 400]))
    session._camera = cam
    session._piezo = FakePiezo()
    events = Events()
    session.set_owner_callbacks(events.callbacks())
    yield session, cam, events
    owner = session.camera_owner(create=False)
    if owner is not None:
        owner.abandon()
        if cam.hang_event is not None:
            cam.hang_event.set()
    session._camera = None
    session._piezo = None


# ---------------------------------------------------------------------------
# Single owner
# ---------------------------------------------------------------------------
def test_only_the_owner_thread_touches_the_camera(owned):
    """Stream, snapshot, exposure change and close: one thread id, always the owner's."""
    session, cam, ev = owned
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 5)
    owner.snapshot()
    owner.set_exposure(0.02)
    assert _wait(lambda: cam.exposure == 0.02)
    assert _wait(lambda: cam.start_count == 2)  # re-armed after the exposure change
    assert session.disconnect_all(log_cb=ev.log) is True

    assert cam.call_threads == {owner.ident}
    assert threading.get_ident() not in cam.call_threads
    assert cam.closed and cam.close_thread == owner.ident
    assert not owner.is_alive()
    assert ev.callback_threads == {owner.ident}


def test_stream_arms_once_and_delivers_newest_frames(owned):
    """The camera is armed once; frames arrive in order, never repeated; stop disarms."""
    session, cam, ev = owned
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 8)
    assert cam.start_count == 1 and cam.ring_frames == owner.STREAM_RING_FRAMES
    assert cam.snap_count == 0
    assert ev.states[0] is True
    seq = [int(f[0, 0]) for f, _z in ev.frames]
    # constant frames 100/200/300/400 cycle: the merged value equals the constant
    expected = [[100, 200, 300, 400][i % 4] for i in cam.delivered[: len(seq)]]
    assert seq == expected
    assert cam.delivered == sorted(set(cam.delivered))

    owner.preview_stop()
    assert _wait(lambda: not cam.armed)
    assert _wait(lambda: ev.states[-1] is False)
    assert cam.stop_count == 1
    assert owner.is_preview_busy() is False


def test_frames_skipped_when_reader_is_slow(owned):
    """read_newest_image after a pause returns the latest index, not the backlog."""
    session, cam, ev = owned
    cam.frame_period = 0.0
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 3)
    # The fake advances one frame per wait; the owner reads the newest each time,
    # so indices are strictly increasing and every read is the newest one.
    assert cam.delivered == sorted(cam.delivered)
    assert all(b > a for a, b in zip(cam.delivered, cam.delivered[1:], strict=False))
    owner.preview_stop()


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def test_snapshot_uses_snap_only_when_the_stream_is_off(owned):
    session, cam, ev = owned
    owner = session.camera_owner()
    owner.snapshot()
    assert _wait(lambda: len(ev.frames) == 1)
    assert cam.snap_count == 1 and cam.start_count == 0
    assert owner.is_preview_busy() is False

    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 3)
    n = len(ev.frames)
    owner.snapshot()
    assert _wait(lambda: len(ev.frames) > n)
    assert cam.snap_count == 1  # served by the stream, no arm/disarm
    owner.preview_stop()


def test_snapshot_requests_are_coalesced(owned):
    """Several snapshot requests while one is in flight produce one frame."""
    session, cam, ev = owned
    owner = session.camera_owner()
    gate = threading.Event()
    orig = cam.snap

    def slow_snap(timeout=None):
        gate.wait(2.0)
        return orig(timeout)

    cam.snap = slow_snap
    owner.snapshot()
    owner.snapshot()
    owner.snapshot()
    assert owner.is_preview_busy()
    gate.set()
    assert _wait(lambda: len(ev.frames) == 1)
    time.sleep(0.05)
    assert len(ev.frames) == 1 and cam.snap_count == 1
    assert owner.is_preview_busy() is False


# ---------------------------------------------------------------------------
# Exposure and sweep during the stream
# ---------------------------------------------------------------------------
def test_exposure_change_is_applied_with_the_camera_disarmed(owned):
    session, cam, ev = owned
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 2)
    session.connect_camera(exposure_s=0.02, log_cb=ev.log)  # the app's "Apply" path
    assert _wait(lambda: cam.start_count == 2)
    assert cam.exposure_calls[-1] == (0.02, False)  # set while disarmed
    i = cam.calls.index("set_exposure")
    assert cam.calls[i - 1] == "stop_acquisition" and cam.calls[i + 1] == "start_acquisition"
    assert ev.log.contains("exposure updated", level="info")
    assert owner.is_streaming()
    n = len(ev.frames)
    assert _wait(lambda: len(ev.frames) > n + 2)  # stream resumed
    owner.preview_stop()


def test_sweep_stops_the_stream_and_resumes_it(owned, tmp_path):
    session, cam, ev = owned
    session.MOVE_POLL_S = 0.001
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 2)
    cfg = {
        "start": 0.0,
        "end": 2.0,
        "step": 1.0,
        "exposure": 0.07,
        "timeout": 1.0,
        "format": "tiff",
        "output_folder": str(tmp_path / "data"),
        "settle": 0.0,
        "closed_loop": True,
        "axis": "A",
        "color_mode": "mono",
    }
    owner.run_sweep(cfg, CancelFlag())
    assert _wait(lambda: len(ev.finished) == 1, timeout=10)
    assert ev.finished[0][1:] == (0, 3)
    assert cam.snap_count == 3
    first_snap = cam.calls.index("snap")
    assert "stop_acquisition" in cam.calls[:first_snap]
    assert not cam.armed or cam.start_count == 2
    assert _wait(lambda: cam.start_count == 2)  # resumed after the sweep
    assert cam.calls.index("start_acquisition", first_snap) > cam.calls.index("snap", first_snap)
    assert owner.is_streaming()
    owner.preview_stop()


def test_sweep_error_is_reported_and_stream_resumes(owned, tmp_path):
    session, cam, ev = owned
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 2)
    owner.run_sweep({"start": 0, "end": 1, "step": 0, "exposure": 0.07, "timeout": 1}, CancelFlag())
    assert _wait(lambda: any(src == "sweep" for src, _ in ev.errors))
    assert "STEP must be > 0" in ev.errors[-1][1]
    assert _wait(lambda: cam.start_count == 2)
    owner.preview_stop()


# ---------------------------------------------------------------------------
# Errors and the unresponsive camera
# ---------------------------------------------------------------------------
def test_sweep_aborted_on_camera_failures_is_reported_then_finished(owned, tmp_path):
    """Batch 2 (C4): the owner reports the abort summary, then the (partial) finish."""
    session, cam, ev = owned
    session.MOVE_POLL_S = 0.001
    cam.fail_snap_at = set(range(1, 100))  # one good frame, then the camera is gone
    owner = session.camera_owner()
    cfg = {
        "start": 0.0,
        "end": 9.0,
        "step": 1.0,
        "exposure": 0.07,
        "timeout": 1.0,
        "format": "tiff",
        "output_folder": str(tmp_path / "data"),
        "settle": 0.0,
        "closed_loop": True,
        "axis": "A",
        "color_mode": "mono",
    }
    owner.run_sweep(cfg, CancelFlag())
    assert _wait(lambda: len(ev.finished) == 1, timeout=10)
    assert len(ev.aborted) == 1 and "1 of 10 frame(s) saved" in ev.aborted[0]
    assert ev.finished[0][1:] == (9, 10)
    assert cam.snap_count == 1 + session.SWEEP_MAX_CONSECUTIVE_FAILURES
    assert ev.errors == []


def _fast_retry(owner, delay=0.05):
    owner.PREVIEW_RETRY_DELAY_S = delay
    return owner


def test_stream_error_is_retried_and_the_preview_goes_on(owned):
    """Batch 2: one SDK error while streaming is a notice (with its text) and
    a re-arm, not the end of the preview; the notice clears once frames flow."""
    session, cam, ev = owned
    owner = _fast_retry(session.camera_owner())
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 2)
    cam.wait_error = RuntimeError("device disconnected")
    assert _wait(lambda: ev.notices)
    cam.wait_error = None
    assert "device disconnected" in ev.notices[0] and "Retrying" in ev.notices[0]
    n = len(ev.frames)
    assert _wait(lambda: len(ev.frames) > n + 2)  # streaming again
    assert _wait(lambda: ev.notices[-1] == "")  # notice cleared on recovery
    assert cam.start_count == 2  # re-armed once
    assert False not in ev.states  # the preview was never reported as stopped
    assert ev.errors == []  # nothing for a dialog
    assert ev.log.contains("recovered", level="info")
    owner.preview_stop()


def test_persistent_stream_error_gives_up_after_max_failures(owned):
    session, cam, ev = owned
    owner = _fast_retry(session.camera_owner())
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 2)
    cam.wait_error = RuntimeError("device disconnected")
    assert _wait(lambda: ev.states[-1] is False, timeout=5.0)
    assert cam.start_count == owner.PREVIEW_MAX_FAILURES  # first arm + 2 retries
    retries = [t for t in ev.notices if "Retrying" in t]
    assert len(retries) == owner.PREVIEW_MAX_FAILURES - 1
    final = ev.notices[-1]
    assert f"stopped after {owner.PREVIEW_MAX_FAILURES} failed attempts" in final
    assert "device disconnected" in final
    assert ev.errors == []
    assert owner.is_preview_busy() is False and not cam.armed
    assert ev.log.contains("Live preview stopped", level="error")


def test_no_frames_within_timeout_is_retried_then_given_up(owned):
    session, cam, ev = owned
    cam.no_frames = True
    owner = _fast_retry(session.camera_owner())
    owner.STREAM_POLL_S = 0.02
    owner.preview_start(frame_timeout_s=0.2)
    assert _wait(lambda: ev.states and ev.states[-1] is False, timeout=5.0)
    assert "No frame from the camera for 0.2 s" in ev.notices[0]
    assert ev.states == [True, True, True, False]  # armed three times, then stopped
    assert ev.errors == []
    assert owner.is_preview_busy() is False


def test_arm_failure_is_retried_then_given_up(owned):
    session, cam, ev = owned
    cam.start_error = RuntimeError("arm failed")
    owner = _fast_retry(session.camera_owner())
    owner.preview_start()
    assert _wait(lambda: ev.states == [False], timeout=3.0)
    assert cam.start_count == 0
    assert cam.calls.count("start_acquisition") == owner.PREVIEW_MAX_FAILURES
    assert all("arm failed" in t for t in ev.notices)
    assert owner.is_streaming() is False
    assert ev.errors == []


def test_snapshot_timeout_without_message_is_retried_and_explained(owned):
    """pylablib's timeout has an empty message (C7): the notice must still say
    what happened, and the snapshot is retried by itself."""
    session, cam, ev = owned
    orig = cam.snap
    fails = {"n": 1}

    def flaky_snap(timeout=None):
        if fails["n"]:
            fails["n"] -= 1
            cam.calls.append("snap")
            raise FakeCameraTimeoutError()
        return orig(timeout)

    cam.snap = flaky_snap
    owner = _fast_retry(session.camera_owner())
    owner.snapshot()
    assert _wait(lambda: len(ev.frames) == 1)
    assert ev.notices[0].strip()
    assert "did not deliver a frame in time" in ev.notices[0]
    assert ev.notices[-1] == ""
    assert ev.errors == []


def test_stop_during_the_retry_pause_is_served_at_once(owned):
    """The pause before a retry never delays a stop: no further arm, notice cleared."""
    session, cam, ev = owned
    cam.start_error = RuntimeError("arm failed")
    owner = _fast_retry(session.camera_owner(), delay=30.0)
    owner.preview_start()
    assert _wait(lambda: ev.notices)
    t0 = time.monotonic()
    owner.preview_stop()
    assert _wait(lambda: ev.states == [False], timeout=2.0)
    assert time.monotonic() - t0 < 1.0
    assert ev.notices[-1] == ""
    time.sleep(0.1)
    assert cam.calls.count("start_acquisition") == 1
    assert owner.is_preview_busy() is False


def test_close_during_the_retry_pause_is_served_at_once(owned):
    session, cam, ev = owned
    cam.start_error = RuntimeError("arm failed")
    owner = _fast_retry(session.camera_owner(), delay=30.0)
    owner.preview_start()
    assert _wait(lambda: ev.notices)
    t0 = time.monotonic()
    assert session.disconnect_all(log_cb=ev.log) is True
    assert time.monotonic() - t0 < 1.0
    assert cam.closed and not ev.log.contains("not answering")


def test_failed_arming_does_not_spend_the_snapshot_attempt_without_a_pause(owned):
    """Review H1: arming fails with a snapshot pending (Start preview right
    after connecting): every attempt, stream or snapshot, waits for the retry
    pause.  Lower bounds on the gaps only: load can make them longer, never
    shorter, so the check does not depend on the machine."""
    session, cam, ev = owned
    cam.start_error = RuntimeError("arm failed")
    cam.fail_snap_at = set(range(100))
    stamps: list[float] = []

    def on_notice(text):
        stamps.append(time.monotonic())
        ev.notices.append(text)

    cb = session._owner_callbacks
    cb.on_preview_notice = on_notice
    owner = session.camera_owner()
    assert owner.PREVIEW_RETRY_DELAY_S == 1.0  # the default, not shortened
    owner.snapshot()
    owner.preview_start()
    assert _wait(lambda: ev.states and ev.states[-1] is False, timeout=6.0)
    failures = [t for t in ev.notices if t]
    assert len(failures) == owner.PREVIEW_MAX_FAILURES
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    assert all(g >= owner.PREVIEW_RETRY_DELAY_S for g in gaps[: owner.PREVIEW_MAX_FAILURES - 1])


def test_snapshot_only_give_up_says_the_preview_is_stopped(owned):
    """Review H3: with no stream asked for, the final text speaks of the frame."""
    session, cam, ev = owned
    cam.fail_snap_at = set(range(100))
    owner = _fast_retry(session.camera_owner())
    owner.snapshot()
    assert _wait(lambda: ev.states == [False], timeout=3.0)
    final = ev.notices[-1]
    assert final.startswith("Preview frame given up after 3 failed attempts")
    assert "live preview is stopped" in final and "fake snap failure" in final
    assert cam.snap_count == owner.PREVIEW_MAX_FAILURES


def test_failed_exposure_notice_is_cleared_by_a_successful_one(owned):
    """Review H2: the exposure notice goes away when an exposure is applied;
    frames at the old exposure do not clear it."""
    session, cam, ev = owned
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 2)
    cam.set_exposure_error = RuntimeError("exposure out of range")
    owner.set_exposure(99.0)
    assert _wait(lambda: ev.errors)
    assert ev.errors[0][0] == "exposure" and "exposure out of range" in ev.errors[0][1]
    n = len(ev.frames)
    assert _wait(lambda: len(ev.frames) > n + 3)
    assert "" not in ev.notices  # still on screen while frames flow
    cam.set_exposure_error = None
    owner.set_exposure(0.02)
    assert _wait(lambda: ev.notices and ev.notices[-1] == "")
    assert cam.exposure == 0.02
    owner.preview_stop()


def test_describe_error_is_never_empty():
    assert describe_error(RuntimeError("usb gone")) == "usb gone"
    assert "did not deliver a frame in time" in describe_error(FakeCameraTimeoutError())
    assert "FakeCameraTimeoutError" in describe_error(FakeCameraTimeoutError())
    assert describe_error(ValueError("  ")).startswith("ValueError")


def test_unresponsive_camera_is_abandoned_and_never_touched_again(owned):
    """A hung SDK call: disconnect gives up after the deadline, reports it, and
    the stuck thread does nothing to the camera when it finally returns."""
    session, cam, ev = owned
    session.CAMERA_CLOSE_TIMEOUT_S = 0.3
    cam.hang_event = threading.Event()
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: "wait_for_frame" in cam.calls)

    t0 = time.monotonic()
    log = LogCollector()
    assert session.disconnect_all(log_cb=log) is True
    assert time.monotonic() - t0 < 1.5
    assert log.contains("Camera is not answering", level="warn")
    assert session._camera is None
    assert session.camera_owner(create=False) is None
    assert owner.is_alive()  # still stuck, on purpose left alone

    # A new camera "reconnected" meanwhile must be invisible to the old thread.
    new_cam = FakeCamera()
    session._camera = new_cam
    cam.hang_event.set()
    assert _wait(lambda: not owner.is_alive())
    assert cam.closed is False and cam.stop_count == 0
    assert new_cam.calls == []
    assert len(ev.errors) == 0  # nothing reported after the abandonment
    session._camera = None


@pytest.mark.parametrize("outcome", ["raises", "returns"])
def test_abandoned_thread_never_touches_a_new_camera_when_its_call_ends(owned, outcome):
    """Review H1: the stuck wait_for_frame ends with an exception (or a value)
    after the abandonment; the old thread must not stop/re-arm the NEW camera."""
    session, cam, ev = owned
    session.CAMERA_CLOSE_TIMEOUT_S = 0.3
    cam.hang_event = threading.Event()
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: "wait_for_frame" in cam.calls)
    assert session.disconnect_all(log_cb=ev.log) is True
    assert ev.log.contains("Camera is not answering", level="warn")

    new_cam = FakeCamera()
    session._camera = new_cam
    if outcome == "raises":
        cam.wait_error = RuntimeError("device disconnected")
    cam.hang_event.set()
    assert _wait(lambda: not owner.is_alive())
    assert new_cam.calls == []
    assert cam.stop_count == 0 and not cam.closed
    assert ev.errors == []
    session._camera = None


def test_abandoned_thread_stuck_in_arming_never_retries_on_a_new_camera(owned):
    """Review H1, second case: the hang is inside start_acquisition and ends
    with an error; the 3-attempt retry must not run on the new camera."""
    session, cam, ev = owned
    session.CAMERA_CLOSE_TIMEOUT_S = 0.3
    gate = threading.Event()

    def hanging_start(*a, **k):
        cam.calls.append("start_acquisition")
        gate.wait()
        raise RuntimeError("arm failed after the hang")

    cam.start_acquisition = hanging_start
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: "start_acquisition" in cam.calls)
    assert session.disconnect_all(log_cb=ev.log) is True

    new_cam = FakeCamera()
    session._camera = new_cam
    gate.set()
    assert _wait(lambda: not owner.is_alive())
    assert new_cam.calls == []
    assert ev.errors == []
    session._camera = None


def test_abandoned_sweep_is_cancelled_and_moves_no_piezo(owned, tmp_path):
    """Review H2: a sweep whose snap() hangs; the owner is abandoned; a new
    piezo and camera are connected; when snap() returns (raising) the loop
    must stop: no MOV on the new piezo, no snap on the old camera."""
    session, cam, ev = owned
    session.CAMERA_CLOSE_TIMEOUT_S = 0.3
    session.MOVE_POLL_S = 0.001
    old_piezo = session._piezo
    gate = threading.Event()
    snaps = {"n": 0}

    def hanging_snap(timeout=None):
        snaps["n"] += 1
        if snaps["n"] == 1:
            gate.wait()
            raise RuntimeError("camera error after the hang")
        return cam.frame

    cam.snap = hanging_snap
    cfg = {
        "start": 0.0,
        "end": 5.0,
        "step": 1.0,
        "exposure": 0.07,
        "timeout": 1.0,
        "format": "tiff",
        "output_folder": str(tmp_path / "data"),
        "settle": 0.0,
        "closed_loop": True,
        "axis": "A",
        "color_mode": "mono",
    }
    owner = session.camera_owner()
    flag = CancelFlag()
    owner.run_sweep(cfg, flag)
    assert _wait(lambda: snaps["n"] == 1)
    assert session.disconnect_all(log_cb=ev.log) is True  # abandons the owner
    assert flag.is_cancelled()
    assert len(old_piezo.moves) == 2  # MOV 0.0 of step 0 + the MOV 0 of the disconnect

    new_piezo = FakePiezo()
    session._piezo = new_piezo
    session._camera = FakeCamera()
    gate.set()
    assert _wait(lambda: not owner.is_alive())
    assert new_piezo.moves == []
    assert snaps["n"] == 1
    assert ev.finished == [] and ev.errors == []
    session._piezo = None
    session._camera = None


def test_disconnect_without_owner_closes_inline(owned):
    """No live owner thread: disconnect_all closes the camera in the caller."""
    session, cam, ev = owned
    assert session.camera_owner(create=False) is None
    assert session.disconnect_all() is True
    assert cam.closed and cam.close_thread == threading.get_ident()


def test_close_timeout_covers_a_queued_snapshot(owned):
    """disconnect waits for the snapshot in flight (bounded), then closes."""
    session, cam, ev = owned
    owner = session.camera_owner()
    gate = threading.Event()
    orig = cam.snap

    def slow_snap(timeout=None):
        gate.wait(0.3)
        return orig(timeout)

    cam.snap = slow_snap
    owner.snapshot()
    assert _wait(lambda: "snap" in cam.calls or owner.is_preview_busy())
    assert session.disconnect_all(log_cb=ev.log) is True
    assert cam.closed and cam.close_thread == owner.ident
    assert not ev.log.contains("not answering")


def test_session_capture_preview_still_merges_like_before(owned):
    """The snapshot primitive is the old capture_preview (snap + merge)."""
    session, cam, ev = owned
    frame, z = session.capture_preview(timeout=1.0)
    assert frame.shape == (4, 4) and z == 0.0


def test_stream_read_returns_none_on_timeout_and_raises_when_disarmed():
    session: AcquisitionSession = make_session()
    cam = FakeCamera()
    session._camera = cam
    cam.no_frames = True
    cam.armed = True
    assert session.stream_read(0.01) is None
    cam.no_frames = False
    cam.armed = False
    with pytest.raises(RuntimeError, match="not running"):
        session.stream_read(0.01)
    session._camera = None


def test_disconnect_closes_the_camera_while_the_piezo_parks(owned):
    """Batch 4: the camera close is requested first and runs on the owner
    while the piezo is parked, so the two waits overlap instead of adding up."""
    session, cam, ev = owned
    session.MOVE_POLL_S = 0.01
    piezo = FakePiezo(ont_polls_needed=10**9)
    session._piezo = piezo
    session._closed_loop = True
    seen_closed: list[bool] = []
    piezo.qont_hook = lambda p: seen_closed.append(cam.closed)
    owner = session.camera_owner()
    owner.preview_start()
    assert _wait(lambda: len(ev.frames) >= 2)
    assert session.disconnect_all(log_cb=ev.log, park_timeout_s=1.0) is True
    assert True in seen_closed  # the camera was closed during the piezo wait
    assert cam.close_thread == owner.ident
    assert piezo.closed and not ev.log.contains("not answering")
