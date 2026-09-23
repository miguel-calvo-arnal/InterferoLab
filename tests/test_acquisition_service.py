"""Tests for services/acquisition_service.py with a mocked backend session.

The sweep, the snapshots and the live stream run on the real CameraOwner
thread (backend/acquisition/camera_owner.py) on top of FakeSession's
synchronous primitives; only manual moves still use a QThread.

Note on the move QThread wrappers: dropping the last Python reference to a
finished QThread/worker while the C++ object is being destroyed via
deleteLater from the dying thread can SIGBUS (PySide6 6.11 + Python 3.14).
The service keeps a small graveyard of them; the fixture additionally keeps
each service alive for the session.
"""

from __future__ import annotations

import time

import pytest
from helpers_acquisition import FakeSession
from PySide6.QtCore import QCoreApplication

import services.acquisition_service as acquisition_service_module
from services.acquisition_service import AcquisitionService

# Keep-alive store for services (see module docstring).
_WRAPPER_KEEPALIVE: list[object] = []


@pytest.fixture
def service(qtbot, monkeypatch):
    """AcquisitionService whose AcquisitionSession is replaced by FakeSession."""
    monkeypatch.setattr(acquisition_service_module, "AcquisitionSession", FakeSession)
    svc = AcquisitionService()
    fake = svc._session
    assert isinstance(fake, FakeSession)

    yield svc, fake
    # Always unblock any parked worker, then shut down bounded.
    fake.release_sweep.set()
    fake.release_preview.set()
    fake.release_move.set()
    fake.release_connect.set()
    fake.release_disconnect.set()
    svc.shutdown(timeout_ms=3000)
    _WRAPPER_KEEPALIVE.append(svc)
    qtbot.wait(10)  # drain pending deferred deletions before the next test


def _wait_not_running(qtbot, svc):
    qtbot.waitUntil(lambda: not svc.is_running(), timeout=3000)


# ---------------------------------------------------------------------------
# Signal forwarding
# ---------------------------------------------------------------------------
def test_finished_signal_carries_sweep_summary(service, qtbot):
    """A successful sweep emits finished(output_folder, skipped, total)."""
    svc, fake = service
    svc.apply_config({"start": 0, "end": 1, "step": 1})
    with qtbot.waitSignal(svc.finished, timeout=3000) as blocker:
        assert svc.start() is True
    assert blocker.args == [fake.output_folder, 1, 5]
    _wait_not_running(qtbot, svc)


def test_log_signal_forwarded_from_worker(service, qtbot):
    """log_cb messages from the backend arrive via logReceived."""
    svc, _fake = service
    svc.apply_config({"exposure": 0.07})
    with qtbot.waitSignal(svc.logReceived, timeout=3000) as blocker:
        assert svc.start() is True
    assert blocker.args == ["info", "fake sweep started"]
    _wait_not_running(qtbot, svc)


def test_error_signal_emitted_when_sweep_raises(service, qtbot):
    """An exception inside run_sweep surfaces as the error signal."""
    svc, fake = service
    fake.fail_sweep = RuntimeError("hardware exploded")
    svc.apply_config({"exposure": 0.07})
    with qtbot.waitSignal(svc.error, timeout=3000) as blocker:
        assert svc.start() is True
    assert blocker.args == ["hardware exploded"]
    _wait_not_running(qtbot, svc)


def test_running_changed_true_then_false(service, qtbot):
    """runningChanged goes True on start and False after completion."""
    svc, fake = service
    svc.apply_config({"exposure": 0.07})
    states = []
    svc.runningChanged.connect(states.append)
    fake.block_sweep = True
    with qtbot.waitSignal(svc.runningChanged, timeout=1000):
        assert svc.start() is True
    assert svc.is_running() is True
    fake.release_sweep.set()
    _wait_not_running(qtbot, svc)
    assert states[0] is True
    assert states[-1] is False


def test_progress_signal_forwarded(service, qtbot):
    """progress_cb calls from the backend arrive via progressChanged."""
    svc, _fake = service
    svc.apply_config({"exposure": 0.07})
    with qtbot.waitSignal(svc.progressChanged, timeout=3000) as blocker:
        assert svc.start() is True
    assert blocker.args[0] == pytest.approx(100.0)
    _wait_not_running(qtbot, svc)


# ---------------------------------------------------------------------------
# start() rejections
# ---------------------------------------------------------------------------
def test_start_without_config_rejected(service, qtbot):
    """start() with no configuration is rejected with an error log."""
    svc, _fake = service
    with qtbot.waitSignal(svc.logReceived, timeout=1000) as blocker:
        assert svc.start() is False
    assert blocker.args == ["error", "No configuration set before start()."]


def test_double_start_second_rejected(service, qtbot):
    """A second start() while a sweep is running returns False with a warn."""
    svc, fake = service
    fake.block_sweep = True
    svc.apply_config({"exposure": 0.07})
    assert svc.start() is True
    assert fake.sweep_started.wait(timeout=2.0)

    with qtbot.waitSignal(svc.logReceived, timeout=1000) as blocker:
        assert svc.start() is False
    assert blocker.args == ["warn", "A sweep is already running."]

    fake.release_sweep.set()
    _wait_not_running(qtbot, svc)
    assert fake.sweep_calls == 1


def test_start_accepted_while_snapshot_in_flight_runs_after_it(service, qtbot):
    """A sweep ordered during a snapshot is accepted and runs after it, on the
    same camera thread: the two can never use the camera at once."""
    svc, fake = service
    fake.block_preview = True
    svc.apply_config({"exposure": 0.07})
    assert svc.capture_preview() is True
    assert fake.preview_started.wait(timeout=2.0)
    assert svc.is_previewing() is True

    assert svc.start() is True
    assert svc.is_running() is True
    time.sleep(0.1)
    assert fake.sweep_calls == 0  # still waiting behind the snapshot

    fake.release_preview.set()
    with qtbot.waitSignal(svc.finished, timeout=3000):
        pass
    _wait_not_running(qtbot, svc)
    assert fake.order == ["preview", "sweep"]


def test_start_rejected_while_moving(service, qtbot):
    """The sweep drives the piezo: it is refused while a manual move runs."""
    svc, fake = service
    fake.block_move = True
    svc.apply_config({"exposure": 0.07})
    svc.move_to(10.0)
    qtbot.waitUntil(lambda: fake.move_calls == [10.0], timeout=2000)

    with qtbot.waitSignal(svc.logReceived, timeout=1000) as blocker:
        assert svc.start() is False
    assert "manual move is in progress" in blocker.args[1]
    fake.release_move.set()
    qtbot.waitUntil(lambda: not svc.is_moving(), timeout=3000)


def test_preview_frame_delivered(service, qtbot):
    """capture_preview emits previewFrame(frame, z) from the worker thread."""
    svc, _fake = service
    with qtbot.waitSignal(svc.previewFrame, timeout=3000) as blocker:
        assert svc.capture_preview() is True
    frame, z = blocker.args
    assert frame.shape == (2, 2)
    assert z == 1.5
    qtbot.waitUntil(lambda: not svc.is_previewing(), timeout=3000)


# ---------------------------------------------------------------------------
# apply_config
# ---------------------------------------------------------------------------
def test_apply_config_merges_partial_updates(service):
    """apply_config merges dictionaries and forwards color_mode to the session."""
    svc, fake = service
    svc.apply_config({"start": 0.0, "exposure": 0.07})
    svc.apply_config({"end": 5.0, "color_mode": "color"})
    assert svc._cfg == {"start": 0.0, "exposure": 0.07, "end": 5.0, "color_mode": "color"}
    assert fake.color_mode == "color"


def test_apply_config_rejected_while_running(service, qtbot):
    """apply_config during a sweep is rejected with a warn and does not merge."""
    svc, fake = service
    fake.block_sweep = True
    svc.apply_config({"exposure": 0.07})
    assert svc.start() is True
    assert fake.sweep_started.wait(timeout=2.0)

    with qtbot.waitSignal(svc.logReceived, timeout=1000) as blocker:
        svc.apply_config({"exposure": 0.5})
    assert blocker.args == ["warn", "Configuration change rejected: a sweep is running."]
    assert svc._cfg["exposure"] == 0.07  # unchanged

    fake.release_sweep.set()
    _wait_not_running(qtbot, svc)


# ---------------------------------------------------------------------------
# cancel / shutdown
# ---------------------------------------------------------------------------
def test_cancel_stops_blocked_sweep(service, qtbot):
    """cancel() makes a cooperative worker finish without external release."""
    svc, fake = service
    fake.block_sweep = True
    svc.apply_config({"exposure": 0.07})
    with qtbot.waitSignal(svc.finished, timeout=3000):
        assert svc.start() is True
        assert fake.sweep_started.wait(timeout=2.0)
        svc.cancel()
    _wait_not_running(qtbot, svc)


def test_shutdown_with_live_worker_terminates_quickly(service):
    """shutdown() cancels a live sweep and returns cleanly in bounded time."""
    svc, fake = service
    fake.block_sweep = True
    svc.apply_config({"exposure": 0.07})
    assert svc.start() is True
    assert fake.sweep_started.wait(timeout=2.0)

    t0 = time.monotonic()
    svc.shutdown(timeout_ms=5000)
    elapsed = time.monotonic() - t0

    assert elapsed < 3.0
    assert svc.is_running() is False
    assert fake.camera_owner(create=False) is None  # camera thread closed
    assert fake.camera_closed is True
    assert fake.disconnect_calls >= 1


def test_shutdown_idle_service_is_safe(service):
    """shutdown() with nothing running is a safe no-op that disconnects."""
    svc, fake = service
    svc.shutdown(timeout_ms=1000)
    assert svc.is_running() is False
    assert fake.disconnect_calls >= 1


def test_state_allows_new_sweep_after_completion(service, qtbot):
    """After a sweep finishes, the service accepts a new start()."""
    svc, fake = service
    svc.apply_config({"exposure": 0.07})
    with qtbot.waitSignal(svc.finished, timeout=3000):
        assert svc.start() is True
    _wait_not_running(qtbot, svc)

    with qtbot.waitSignal(svc.finished, timeout=3000):
        assert svc.start() is True
    _wait_not_running(qtbot, svc)
    assert fake.sweep_calls == 2


# ---------------------------------------------------------------------------
# Live preview stream, moves during the stream, newest frame wins
# ---------------------------------------------------------------------------
def test_stream_starts_once_and_stops_on_request(service, qtbot):
    """start_preview arms the camera once and keeps delivering; stop disarms it."""
    svc, fake = service
    frames: list[int] = []
    svc.previewFrame.connect(lambda f, z: frames.append(int(f[0, 0])))
    states: list[bool] = []
    svc.previewingChanged.connect(states.append)

    assert svc.start_preview() is True
    assert svc.is_previewing() is True
    qtbot.waitUntil(lambda: len(frames) >= 10, timeout=3000)
    assert fake.stream_starts == 1
    assert frames == sorted(frames) and len(set(frames)) == len(frames)  # newer, never repeated

    svc.stop_preview()
    qtbot.waitUntil(lambda: states and states[-1] is False, timeout=3000)
    assert svc.is_previewing() is False
    assert fake.stream_stops == 1 and fake.stream_running is False
    assert states[0] is True


def test_newest_frame_wins_when_the_gui_is_busy(service, qtbot):
    """Frames produced while the GUI does not run its loop are dropped except the
    newest: one notification, one previewFrame, carrying the latest sequence."""
    svc, fake = service
    delivered: list[int] = []
    svc.previewFrame.connect(lambda f, z: delivered.append(int(f[0, 0])))
    assert svc.start_preview() is True
    qtbot.waitUntil(lambda: fake.stream_frames >= 3, timeout=3000)
    QCoreApplication.processEvents()
    n_before = len(delivered)

    time.sleep(0.2)  # the camera thread keeps producing; the GUI is "busy"
    produced = fake.stream_frames
    assert produced - (delivered[-1] if delivered else 0) > 5
    QCoreApplication.processEvents()  # one loop turn
    new = delivered[n_before:]
    assert len(new) == 1
    assert new[0] >= produced  # the frame painted is the newest one at that moment


def test_moves_are_never_rejected_by_the_live_preview(service, qtbot):
    """A move during the stream is accepted at once; the stream keeps running."""
    svc, fake = service
    assert svc.start_preview() is True
    qtbot.waitUntil(lambda: fake.stream_frames >= 2, timeout=3000)
    warnings: list[str] = []
    svc.logReceived.connect(lambda lvl, m: warnings.append(m) if lvl == "warn" else None)

    with qtbot.waitSignal(svc.positionChanged, timeout=3000) as blocker:
        svc.move_to(12.5)
    assert blocker.args == [12.5]
    assert warnings == []
    qtbot.waitUntil(lambda: not svc.is_moving(), timeout=3000)
    before = fake.stream_frames
    qtbot.waitUntil(lambda: fake.stream_frames > before + 2, timeout=3000)
    assert fake.stream_running is True


def test_latest_move_order_wins(service, qtbot):
    """Orders arriving during a move replace each other: only the last one runs."""
    svc, fake = service
    fake.block_move = True
    states: list[bool] = []
    svc.movingChanged.connect(states.append)
    svc.move_to(1.0)
    qtbot.waitUntil(lambda: fake.move_calls == [1.0], timeout=2000)
    svc.move_to(2.0)
    svc.move_to(3.0)
    assert svc.is_moving() is True

    fake.release_move.set()
    qtbot.waitUntil(lambda: fake.move_calls == [1.0, 3.0], timeout=3000)
    qtbot.waitUntil(lambda: not svc.is_moving(), timeout=3000)
    assert states[0] is True and states[-1] is False
    assert fake.move_calls == [1.0, 3.0]  # 2.0 was superseded before it could start


def test_exposure_applied_by_the_camera_thread_between_frames(service, qtbot):
    """connect_camera() with the camera connected hands the exposure to the
    camera thread: stream stopped, exposure set, stream re-armed."""
    svc, fake = service
    svc.apply_config({"exposure": 0.007})
    assert svc.start_preview() is True
    qtbot.waitUntil(lambda: fake.stream_frames >= 2, timeout=3000)

    owner = fake.camera_owner()
    with qtbot.waitSignal(
        svc.logReceived, timeout=3000, check_params_cb=lambda _lvl, m: "exposure updated" in m
    ):
        owner.set_exposure(0.02)
    qtbot.waitUntil(lambda: fake.stream_starts == 2, timeout=3000)
    i_stop = fake.order.index("stream_stop")
    i_set = fake.order.index("set_exposure")
    assert fake.order[i_stop : i_set + 2] == ["stream_stop", "set_exposure", "stream_start"]
    assert fake.exposure == 0.02
    assert svc.is_previewing() is True


def test_sweep_pauses_the_stream_and_resumes_it(service, qtbot):
    """A sweep started with the live preview on: stream stopped before it,
    re-armed after it, all in call order on the camera thread."""
    svc, fake = service
    svc.apply_config({"exposure": 0.07})
    assert svc.start_preview() is True
    qtbot.waitUntil(lambda: fake.stream_frames >= 2, timeout=3000)

    with qtbot.waitSignal(svc.finished, timeout=3000):
        assert svc.start() is True
    _wait_not_running(qtbot, svc)
    qtbot.waitUntil(lambda: fake.stream_starts == 2, timeout=3000)
    i_sweep = fake.order.index("sweep")
    assert fake.order[i_sweep - 1] == "stream_stop"
    assert fake.order[i_sweep + 1] == "stream_start"
    assert svc.is_previewing() is True


def _fast_retry(svc, delay=0.05):
    owner = svc._session.camera_owner()
    owner.PREVIEW_RETRY_DELAY_S = delay
    return owner


def test_stream_error_is_a_notice_and_the_preview_recovers(service, qtbot):
    """Batch 2: a camera error inside the stream is a previewNotice with its
    text (never the error signal, which the panel shows as a dialog); the
    stream re-arms and the notice clears."""
    svc, fake = service
    errors: list[str] = []
    notices: list[str] = []
    states: list[bool] = []
    svc.error.connect(errors.append)
    svc.previewNotice.connect(notices.append)
    svc.previewingChanged.connect(states.append)
    _fast_retry(svc)
    assert svc.start_preview() is True
    qtbot.waitUntil(lambda: fake.stream_frames >= 2, timeout=3000)

    fake.stream_error = RuntimeError("usb gone")
    qtbot.waitUntil(lambda: bool(notices), timeout=3000)
    fake.stream_error = None
    assert "usb gone" in notices[0]
    qtbot.waitUntil(lambda: notices[-1] == "", timeout=3000)
    n = fake.stream_frames
    qtbot.waitUntil(lambda: fake.stream_frames > n + 2, timeout=3000)
    assert errors == []
    assert False not in states
    assert svc.is_previewing() is True


def test_persistent_stream_error_stops_the_preview_without_error_signal(service, qtbot):
    svc, fake = service
    errors: list[str] = []
    notices: list[str] = []
    states: list[bool] = []
    svc.error.connect(errors.append)
    svc.previewNotice.connect(notices.append)
    svc.previewingChanged.connect(states.append)
    _fast_retry(svc)
    assert svc.start_preview() is True
    qtbot.waitUntil(lambda: fake.stream_frames >= 2, timeout=3000)
    fake.stream_error = RuntimeError("usb gone")
    qtbot.waitUntil(lambda: states and states[-1] is False, timeout=5000)
    assert "stopped after 3 failed attempts" in notices[-1] and "usb gone" in notices[-1]
    assert errors == []
    assert svc.is_previewing() is False
    assert fake.stream_running is False


def test_snapshot_error_is_a_notice_not_an_error(service, qtbot):
    svc, fake = service
    errors: list[str] = []
    notices: list[str] = []
    svc.error.connect(errors.append)
    svc.previewNotice.connect(notices.append)
    _fast_retry(svc)
    fake.preview_error = RuntimeError("snap failed")
    assert svc.capture_preview() is True
    qtbot.waitUntil(lambda: any("stopped" in t for t in notices), timeout=5000)
    assert fake.order.count("preview") == 3  # retried, then given up
    assert all("snap failed" in t for t in notices)
    assert errors == []


def test_exposure_error_is_a_notice_not_an_error(service, qtbot):
    """A failed exposure change is logged and shown as a notice, no dialog."""
    svc, fake = service
    errors: list[str] = []
    notices: list[str] = []
    logs: list[tuple[str, str]] = []
    svc.error.connect(errors.append)
    svc.previewNotice.connect(notices.append)
    svc.logReceived.connect(lambda lvl, msg: logs.append((lvl, msg)))
    fake.exposure_error = RuntimeError("exposure out of range")
    svc._session.camera_owner().set_exposure(0.02)
    qtbot.waitUntil(lambda: bool(notices), timeout=3000)
    assert "exposure out of range" in notices[0]
    assert ("error", notices[0]) in logs
    assert errors == []


def test_sweep_aborted_is_signalled_before_finished(service, qtbot):
    svc, fake = service
    seen: list[tuple] = []
    svc.sweepAborted.connect(lambda summary: seen.append(("aborted", summary)))
    svc.finished.connect(lambda folder, s, t: seen.append(("finished", folder, s, t)))
    fake.abort_sweep_with = "Sweep aborted after 3 consecutive camera failures."
    svc.apply_config({"start": 0, "end": 1, "step": 1})
    assert svc.start() is True
    qtbot.waitUntil(lambda: len(seen) == 2, timeout=3000)
    assert seen[0] == ("aborted", fake.abort_sweep_with)
    assert seen[1][0] == "finished"
    assert svc.is_running() is False


def test_shutdown_during_stream_closes_camera_from_its_thread(service, qtbot):
    """shutdown() with the stream on: bounded, camera closed by the owner."""
    svc, fake = service
    assert svc.start_preview() is True
    qtbot.waitUntil(lambda: fake.stream_frames >= 2, timeout=3000)
    t0 = time.monotonic()
    svc.shutdown(timeout_ms=3000)
    assert time.monotonic() - t0 < 3.0
    assert fake.camera_closed is True
    assert fake.order[-2:] == ["stream_stop", "close_camera"]
    assert fake.camera_owner(create=False) is None


# ---------------------------------------------------------------------------
# Connect / disconnect off the GUI thread (batch 4: C8, R9)
# ---------------------------------------------------------------------------
def test_connect_runs_off_the_gui_thread_and_blocks_nothing(service, qtbot):
    """connect_hardware returns at once; while it runs, a second connect, a
    disconnect, a sweep, a move and the camera are all refused."""
    svc, fake = service
    fake.block_connect = True
    svc.apply_config({"start": 0, "end": 1, "step": 1})
    t0 = time.monotonic()
    assert svc.connect_hardware("123", "dll") is True
    assert fake.connect_started.wait(2.0)
    assert time.monotonic() - t0 < 2.0  # generous: the fake blocks up to 5 s
    assert svc.connection_busy() == "connecting"
    assert fake.connect_threads == ["hardware-connecting"]

    assert svc.connect_hardware("123", "dll") is False
    assert svc.disconnect_hardware() is False
    assert svc.start() is False
    svc.move_to(5.0)
    assert fake.move_calls == [] and not svc.is_moving()
    assert svc.start_preview() is False and svc.capture_preview() is False
    assert fake.camera_owner(create=False) is None  # nobody touched the camera

    with qtbot.waitSignal(svc.connectFinished, timeout=3000) as blocker:
        fake.release_connect.set()
    assert blocker.args == [True]
    assert svc.connection_busy() == ""
    assert fake.disconnect_calls == 0


def test_failed_connect_releases_on_the_connection_thread(service, qtbot):
    svc, fake = service
    fake.camera_connects = False
    with qtbot.waitSignal(svc.connectFinished, timeout=3000) as blocker:
        assert svc.connect_hardware("123", "dll") is True
    assert blocker.args == [False]
    assert fake.disconnect_log == [("hardware-connecting", None)]


def test_disconnect_runs_off_the_gui_thread(service, qtbot):
    svc, fake = service
    fake.block_disconnect = True
    assert svc.disconnect_hardware() is True
    assert fake.disconnect_started.wait(2.0)
    assert svc.connection_busy() == "disconnecting"
    assert svc.connect_hardware("123", "dll") is False  # not half-way through
    with qtbot.waitSignal(svc.disconnectFinished, timeout=3000):
        fake.release_disconnect.set()
    assert fake.disconnect_log == [("hardware-disconnecting", None)]
    assert svc.connection_busy() == ""


def test_disconnect_refused_during_a_sweep_or_a_move(service, qtbot):
    svc, fake = service
    fake.block_sweep = True
    svc.apply_config({"start": 0, "end": 1, "step": 1})
    assert svc.start() is True
    assert svc.disconnect_hardware() is False
    fake.release_sweep.set()
    _wait_not_running(qtbot, svc)
    fake.block_move = True
    svc.move_to(1.0)
    assert svc.disconnect_hardware() is False
    fake.release_move.set()
    qtbot.waitUntil(lambda: not svc.is_moving(), timeout=3000)
    assert fake.disconnect_calls == 0


def test_shutdown_during_a_disconnect_cuts_it_short_then_parks_briefly(service, qtbot):
    """Closing the app while a (slow) disconnect runs: its piezo wait is
    cancelled, it is joined, and the final disconnect uses the short park."""
    svc, fake = service
    fake.block_disconnect = True
    assert svc.disconnect_hardware() is True
    assert fake.disconnect_started.wait(2.0)
    fake.block_disconnect = False  # the final disconnect of the shutdown is quick
    svc.shutdown(timeout_ms=3000)
    assert fake.disconnect_log == [
        ("hardware-disconnecting", None),
        ("MainThread", svc.CLOSE_PARK_TIMEOUT_S),
    ]


def test_shutdown_leaves_a_stuck_connection_thread_alone(service, qtbot):
    """A connection thread stuck in a driver call: shutdown gives up after its
    deadline, says so, and never releases the hardware from a second thread."""
    svc, fake = service
    svc.CONNECTION_THREAD_WAIT_S = 0.2
    fake.block_disconnect = True
    fake.ignore_cancel = True
    logs: list[tuple[str, str]] = []
    svc.logReceived.connect(lambda lvl, msg: logs.append((lvl, msg)))
    assert svc.disconnect_hardware() is True
    assert fake.disconnect_started.wait(2.0)
    t0 = time.monotonic()
    svc.shutdown(timeout_ms=3000)
    assert time.monotonic() - t0 < 2.0
    assert any("not answering" in m for lvl, m in logs if lvl == "warn")
    assert [t for t, _ in fake.disconnect_log] == ["hardware-disconnecting"]


# ---------------------------------------------------------------------------
# Review corrections (batch 4)
# ---------------------------------------------------------------------------
def test_apply_exposure_is_refused_while_connecting_or_disconnecting(service, qtbot):
    """H1: the service itself refuses it (not only the disabled button)."""
    svc, fake = service
    fake.block_disconnect = True
    assert svc.disconnect_hardware() is True
    assert fake.disconnect_started.wait(2.0)
    assert svc.connect_camera() is False  # "Apply camera settings" from the GUI
    assert fake.camera_connect_calls == []
    with qtbot.waitSignal(svc.disconnectFinished, timeout=3000):
        fake.release_disconnect.set()
    # the connection thread itself still opens the camera while "connecting"
    with qtbot.waitSignal(svc.connectFinished, timeout=3000) as blocker:
        assert svc.connect_hardware("123", "dll") is True
    assert blocker.args == [True]
    assert fake.camera_connect_calls == ["hardware-connecting"]


def test_camera_not_opened_when_the_piezo_failed(service, qtbot):
    """Review H4: a failed piezo ends the attempt before the camera is opened."""
    svc, fake = service
    fake.piezo_connects = False
    with qtbot.waitSignal(svc.connectFinished, timeout=3000) as blocker:
        assert svc.connect_hardware("123", "dll") is True
    assert blocker.args == [False]
    assert fake.camera_connect_calls == []
    assert fake.disconnect_calls == 1  # released anyway (idempotent)


def test_stuck_connection_gives_the_window_back(service, qtbot):
    """Review H3: a connect stuck in a driver: after CONNECTION_STUCK_S the
    window gets connectFinished(False) and "not answering"; no new connection
    while the stuck thread lives; when it finally returns, what it opened is
    released on a new connection thread."""
    svc, fake = service
    svc.CONNECTION_STUCK_S = 0.2
    fake.block_connect = True
    logs: list[tuple[str, str]] = []
    svc.logReceived.connect(lambda lvl, msg: logs.append((lvl, msg)))
    with qtbot.waitSignal(svc.connectFinished, timeout=3000) as blocker:
        assert svc.connect_hardware("123", "dll") is True
    assert blocker.args == [False]
    assert svc.connection_busy() == ""
    assert any("not answering" in m for lvl, m in logs if lvl == "warn")
    assert svc.connect_hardware("123", "dll") is False  # still inside the driver
    assert any("still stuck" in m for _, m in logs)

    fake.release_connect.set()  # the driver call returns at last (ok)
    qtbot.waitUntil(lambda: ("hardware-disconnecting", None) in fake.disconnect_log, timeout=3000)
    qtbot.waitUntil(lambda: svc.connection_busy() == "", timeout=3000)
    assert svc.connect_hardware("123", "dll") is True  # usable again
