"""Tests for services/acquisition_service.py with a mocked backend session.

KNOWN BUG (reported, not fixed here): AcquisitionService drops its last Python
references to the QThread/worker wrappers in _on_thread_finished while the C++
objects are still being destroyed via deleteLater from the dying thread. Under
PySide6 6.11.1 + Python 3.14 this GC/destruction race corrupts memory and
crashes the process with SIGBUS (reproduced with a 20-line standalone script,
independent of pytest). It cannot be marked xfail because it kills the whole
interpreter, so the fixture below keeps every worker/thread wrapper alive for
the session, which reliably avoids the race (verified over 600 cycles).
"""

from __future__ import annotations

import time

import pytest
from helpers_acquisition import FakeSession

import services.acquisition_service as acquisition_service_module
from services.acquisition_service import AcquisitionService

# Keep-alive store for QThread/worker Python wrappers (see module docstring).
_WRAPPER_KEEPALIVE: list[object] = []


@pytest.fixture
def service(qtbot, monkeypatch):
    """AcquisitionService whose AcquisitionSession is replaced by FakeSession."""
    monkeypatch.setattr(acquisition_service_module, "AcquisitionSession", FakeSession)
    svc = AcquisitionService()
    fake = svc._session
    assert isinstance(fake, FakeSession)

    # Wrap the thread-spawning entry points so the wrappers they create are
    # referenced for the whole session (workaround for the SIGBUS bug above).
    orig_start = svc.start
    orig_preview = svc.capture_preview

    def start_keepalive():
        result = orig_start()
        _WRAPPER_KEEPALIVE.append((svc._thread, svc._worker))
        return result

    def preview_keepalive():
        result = orig_preview()
        _WRAPPER_KEEPALIVE.append((svc._preview_thread, svc._preview_worker))
        return result

    monkeypatch.setattr(svc, "start", start_keepalive)
    monkeypatch.setattr(svc, "capture_preview", preview_keepalive)

    yield svc, fake
    # Always unblock any parked worker, then shut down bounded.
    fake.release_sweep.set()
    fake.release_preview.set()
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


def test_start_rejected_while_preview_in_flight(service, qtbot):
    """start() is rejected while a one-shot preview capture is running."""
    svc, fake = service
    fake.block_preview = True
    svc.apply_config({"exposure": 0.07})
    assert svc.capture_preview() is True
    assert fake.preview_started.wait(timeout=2.0)
    assert svc.is_previewing() is True

    with qtbot.waitSignal(svc.logReceived, timeout=1000) as blocker:
        assert svc.start() is False
    assert blocker.args[0] == "warn"
    assert "preview capture is in progress" in blocker.args[1]

    fake.release_preview.set()
    qtbot.waitUntil(lambda: not svc.is_previewing(), timeout=3000)
    assert fake.sweep_calls == 0


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
    assert svc._thread is None and svc._worker is None
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
