"""Diagnostic tests for the concurrency review (phase 2, role A2).

Each test runs one reproduction script from tests/diag/ in a subprocess (the
fakes shadow the real drivers there, exactly like tests/test_sim_smoke.py)
and asserts the behaviour the app SHOULD have.
A test is ``xfail(strict=True)`` while the app does not have that behaviour
yet: the suite stays green, and the day a finding is fixed its test turns
XPASS and fails the run, so the marker (and the finding) get retired on
purpose.

Findings C1...C12 come from the phase-2 concurrency review (2026-09-21).
Retired by B1 (single camera-owner thread + continuous preview, 2026-09-22):
C1, C1b, C2, C5 (both), C6, C7b and C9; plus the B1 single-owner check.
Retired by batch 2 (errors without modal dialogs, sweep abort, 2026-09-22):
C4, C4b and C7 (plus its "never recovers" variant).
Retired by batch 3 (measured piezo position, 2026-09-22): C3.
Retired by batch 4 (connect/disconnect off the GUI thread, 2026-09-22): C8.
Retired by batch 5 (keyboard moves the piezo for real, 2026-09-22): C10.
Whole file: about 3 minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPRO = os.path.join(ROOT, "tests", "diag")

# Load-dependent figures (frame rates, wall-clock bounds) are only asserted on
# demand; the logic they accompany is always asserted.
TIMING = bool(os.environ.get("INTERFEROLAB_TIMING_TESTS"))


def run_repro(script: str, *args: str, tmp_path, timeout: float = 120.0):
    """Run a reproduction script; return (result dict, returncode, stderr)."""
    proc = subprocess.run(
        [sys.executable, os.path.join(REPRO, script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "A2_OUT": str(tmp_path)},
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(lines[-1]), proc.returncode, proc.stderr


def dialogs(res, kind="critical"):
    return [d for d in res["dialogs"] if d[0] == kind]


# ----------------------------------------------------------------------
# C1  camera closed from the GUI while a preview capture is in flight
# ----------------------------------------------------------------------
def test_c1_disconnect_waits_for_the_capture(tmp_path):
    """Fixed by B1: the camera thread finishes the capture, then closes."""
    res, _, _ = run_repro("c1_disconnect_during_preview.py", "0.05", tmp_path=tmp_path)
    assert res["connected"]
    # desired: no capture thread alive once the UI says "disconnected", and
    # no error dialog born from that capture afterwards
    assert not res["previewing_after_disconnect"]
    assert dialogs(res) == []


def test_c1b_reconnect_after_disconnect_has_no_orphan(tmp_path):
    """Fixed by B1: no capture survives the disconnect, so reconnecting right
    away gets a clean session (the snapshot after reconnect is its own, so
    ``previewing_right_after_reconnect`` is legitimately True now)."""
    res, _, _ = run_repro("c1b_reconnect_during_orphan.py", tmp_path=tmp_path)
    assert res["connected"] and res["reconnected"]
    assert not res["previewing_after_disconnect"]
    assert res["errors"] == []
    assert dialogs(res) == []
    assert res["preview_after_error_works"]


# ----------------------------------------------------------------------
# C2  a hung camera call: silent drops, then abort on exit
# ----------------------------------------------------------------------
def test_c2_hung_camera_is_reported_and_does_not_abort(tmp_path):
    """Fixed by B1: the camera thread is a daemon that gets abandoned with a
    message after a deadline; Qt never destroys a running thread."""
    res, code, err = run_repro("c2_hung_camera_exit.py", tmp_path=tmp_path, timeout=180)
    assert res["connected"] and res["previewing"]
    # desired: the user is told the camera is not answering (today the log is
    # silent: previews dropped, move rejected as "preview in progress", only
    # Start says "did not finish in time") ...
    assert "not answering" in res["log"] or "hung" in res["log"].lower()
    # ... and closing the window never aborts the process
    assert "Destroyed while thread" not in err
    assert code == 0


# ----------------------------------------------------------------------
# C3  stale position after a failed move
# ----------------------------------------------------------------------
def test_c3_position_reflects_the_stage_after_failed_move(tmp_path):
    """Fixed by batch 3: the position is read (qPOS) during and after the
    move, also a failed one; the window shows the measured position live."""
    res, _, _ = run_repro("c3_position_lie.py", tmp_path=tmp_path)
    assert res["connected"] and res["move_errors"]
    assert res["stage_after_move_um"] == pytest.approx(60.0, abs=0.1)
    # what the UI shows is where the stage is (or "unknown"), never 0
    assert res["slider_um"] == pytest.approx(res["stage_now_um"], abs=0.1)
    assert res["preview_z"] == pytest.approx(res["stage_now_um"], abs=0.1)
    assert res["measured_label"] == f"Measured Z: {res['stage_now_um']:.3f} µm"
    # followed live while the (1 s, never on target) move lasted: several
    # updates, all of them measured positions near the stage, not the target
    live = res["positions_during_move"]
    assert len(live) >= 3
    assert all(z == pytest.approx(res["stage_now_um"], abs=0.1) for _, z in live)
    assert any(z != 60.0 for _, z in live)


# ----------------------------------------------------------------------
# C4  camera lost mid-sweep
# ----------------------------------------------------------------------
def test_c4_sweep_aborts_when_the_camera_is_gone(tmp_path):
    """Fixed by batch 2: 3 failed captures in a row abort the sweep; the
    partial folder is marked and the user is told how many frames were saved."""
    res, _, _ = run_repro("c4_unplug_midsweep.py", tmp_path=tmp_path, timeout=180)
    assert res["connected"]
    assert res["n_camera_errors"] == 3  # three consecutive failures, then stop
    assert res["sweep_aborted"]
    assert res["finished"][0][1:] == [19, 21]  # 2 of 21 frames acquired
    assert "2 of 21 frame(s) saved" in res["aborted"][0]
    assert "SWEEP_ABORTED.txt" in res["sweep_files"]
    assert len([f for f in res["sweep_files"] if f.startswith("piezo_")]) == 2
    assert [d[:2] for d in res["dialogs"]] == [["warning", "Sweep aborted"]]
    # the snapshot asked for afterwards fails without a dialog, with a notice
    assert res["preview_after"] == []
    assert any("stopped" in t for t in res["preview_after_notices"])


def test_c4b_sweep_aborts_on_repeated_timeouts(tmp_path):
    """Fixed by batch 2: a camera that times out on every snap costs 3 steps,
    not the whole sweep (derived: 501 x 5.6 s = 47 min before)."""
    res, _, _ = run_repro("c4b_timeouts_midsweep.py", tmp_path=tmp_path, timeout=180)
    assert res["connected"]
    assert res["n_camera_errors"] == 3
    assert res["finished"][0][1:] == [5, 5] and len(res["aborted"]) == 1
    # the snapshot after connect timed out: retried, given up, never a dialog
    assert res["connect_preview_error"] == 0
    assert all(t.strip() for t in res["connect_preview_notices"])
    assert [d[:2] for d in res["dialogs"]] == [["warning", "Sweep aborted"]]
    if TIMING:
        # 3 x (0.5 s timeout + 0.25 s pylablib sleeps + arm/disarm + move) ~ 3.4 s
        assert res["t_sweep_s"] < 5.0


# ----------------------------------------------------------------------
# C5  Start sweep re-enters itself through _wait_preview_idle
# ----------------------------------------------------------------------
def test_c5_start_is_not_reentrant(tmp_path):
    """Fixed by B1: _start no longer spins the event loop waiting for the preview."""
    res, _, _ = run_repro("c5_start_reentrant.py", "start", tmp_path=tmp_path)
    assert res["connected"] and res["previewing_before_start"]
    inner = res["inner"]
    # desired: while Start waits for the capture, Start itself is disabled
    assert not inner["start_enabled"]
    assert res["n_could_not_start"] == 0 and res["n_already_running"] == 0
    assert len(res["finished"]) == 1


def test_c5_disconnect_impossible_while_start_waits(tmp_path):
    """Fixed by B1 (same reason as above)."""
    res, _, _ = run_repro("c5_start_reentrant.py", "disconnect", tmp_path=tmp_path)
    assert res["connected"]
    assert not res["inner"]["connect_enabled"]
    assert res["n_sweep_started"] == 0 or res["hardware_connected_after_start"]
    assert dialogs(res) == []


# ----------------------------------------------------------------------
# C6  moves rejected while a capture is in flight; ticks dropped
# ----------------------------------------------------------------------
def test_c6_moves_are_never_rejected_by_the_live_preview(tmp_path):
    """Fixed by B1: the piezo moves without asking the camera; the stream goes on."""
    res, _, _ = run_repro("c6_move_vs_preview.py", "200", "10", tmp_path=tmp_path)
    assert res["connected"]
    assert res["n_rejected"] == 0
    assert res["n_snapback"] == 0
    assert res["attempts"][-1]["stage_after_um"] == pytest.approx(49.0, abs=0.1)


# ----------------------------------------------------------------------
# C7  a single error (camera or piezo) stops the preview with a modal
# ----------------------------------------------------------------------
def test_c7_preview_survives_one_timeout_with_a_message(tmp_path):
    """Fixed by batch 2: a stalled stream is a notice with a text in the window,
    the camera is re-armed by itself and the preview goes on; no dialog."""
    res, _, _ = run_repro("c7_preview_error_modal.py", tmp_path=tmp_path)
    assert res["connected"] and res["notices"]
    # the problem is reported with a text, never as an empty ""
    first = res["notices"][0][1]
    assert first.strip() and "No frame from the camera" in first
    assert res["status_label_during"] == [True, first]  # visible in the window
    # ... the preview keeps running, and the notice is cleared once it recovers
    assert res["ui_after_error"]["preview_active"]
    assert res["frames_after_error"] > res["frames_before_error"]
    assert res["notices"][-1][1] == "" and res["status_label"] == [False, ""]
    assert res["errors"] == [] and res["dialogs"] == []


def test_c7_preview_that_never_recovers_stops_and_says_so(tmp_path):
    """Batch 2: a camera that never delivers again: a few attempts, then the
    preview stops with a visible explanation; still no dialog."""
    res, _, _ = run_repro("c7_preview_error_modal.py", "1000", tmp_path=tmp_path)
    assert res["connected"]
    texts = [t for _, t in res["notices"]]
    assert len([t for t in texts if "Retrying" in t]) == 2
    assert "Live preview stopped after 3 failed attempts" in texts[-1]
    assert res["status_label"] == [True, texts[-1]]
    assert not res["ui_after_error"]["preview_active"]
    assert res["ui_after_error"]["preview"] == [True, "Start preview"]
    assert res["errors"] == [] and res["dialogs"] == []


def test_c7b_piezo_error_leaves_the_preview_running(tmp_path):
    """Fixed by B1: only the camera thread switches the preview off.  Checked
    again by batch 2: the failed move keeps its dialog (the stage is not where
    it was sent, a real hardware problem), with a text, and the preview
    and its status line are untouched."""
    res, _, _ = run_repro("c7b_piezo_error_kills_preview.py", tmp_path=tmp_path)
    assert res["connected"] and res["move_accepted"] and res["move_errors"]
    assert res["ui_after_error"]["preview_active"]
    assert res["frames_after_error"] > 0
    assert all(text.strip() for _, _, text in dialogs(res))
    assert res.get("notices", []) == []


# ----------------------------------------------------------------------
# C8  disconnect / close freeze the GUI for the piezo timeout
# ----------------------------------------------------------------------
def test_c8_disconnect_does_not_freeze_the_gui(tmp_path):
    """Fixed by batch 4: connect and disconnect run on the connection thread.
    Logic always: the event loop ran WHILE disconnecting/connecting, the
    buttons said so and were frozen, a second click was ignored, the piezo
    was still released, and closing the app parks it with a short deadline.
    Wall-clock bounds only with INTERFEROLAB_TIMING_TESTS."""
    res, code, _ = run_repro("c8_disconnect_freeze.py", tmp_path=tmp_path, timeout=120)
    assert res["connected"]
    assert res["timer_fired_while_disconnecting"]
    during = res["ui_during"]
    assert during["connect"] == [False, "Disconnecting…"]
    assert not (during["start"] or during["move"] or during["preview"][0] or during["apply_cam"])
    assert res["second_click_ignored"]
    assert (
        res["ui"]["connect"] == [True, "Connect hardware"] and not res["ui"]["hardware_connected"]
    )
    assert res["piezo_closed"]
    assert res["timer_fired_while_connecting"]
    assert res["ui_while_connecting"]["connect"] == [False, "Connecting…"]
    assert res["reconnected"]
    assert res["closed_piezo_on_close"]
    assert res["dialogs"] == [] and code == 0
    if TIMING:
        assert res["t_disconnect_blocking_s"] < 1.0
        assert res["timer_100ms_fired_after_s"] < 1.0
        assert res["t_connect_blocking_s"] < 0.5
        assert res["t_close_s"] < 4.0  # 2 s park deadline + camera close


# ----------------------------------------------------------------------
# C9  exposure applied while a capture is in flight
# ----------------------------------------------------------------------
def test_c9_exposure_applied_by_the_camera_thread_only(tmp_path):
    """Fixed by B1.  The button stays enabled on purpose: the exposure is now
    applied by the camera-owner thread between frames (stream stopped and
    re-armed around it), so the GUI thread never enters the camera."""
    res, _, _ = run_repro("c9_exposure_during_preview.py", tmp_path=tmp_path)
    assert res["connected"] and res["previewing"]
    assert res["log_says_updated"]
    assert res["camera_exposure_s_now"] == pytest.approx(0.07, rel=1e-3)
    assert res["exposure_visible_in_preview"]
    assert res["preview_still_running"]
    assert dialogs(res) == []
    mon = res["monitor"]
    assert mon["gui_thread_calls"] == 0
    assert mon["n_camera_threads"] == 1 and mon["camera_threads"] == ["camera-owner"]
    assert mon["overlaps"] == []


# ----------------------------------------------------------------------
# C10  keyboard on the piezo slider
# ----------------------------------------------------------------------
def test_c10_keyboard_slider_moves_the_piezo(tmp_path):
    """Fixed by batch 5: the slider's key actions (PageUp/Down, arrows,
    Home/End) reuse the manual-move path, same as a mouse release."""
    res, _, _ = run_repro("c10_slider_keyboard.py", tmp_path=tmp_path)
    assert res["connected"]
    # right after the key presses: still the exact tick count Qt set, no move
    # has landed yet (deferred one event-loop tick -- see _PositionSlider)
    assert res["slider_after_keys_um"] == pytest.approx(5.0, abs=0.01)
    assert res["any_move_logged"] or res["stage_um"] == pytest.approx(5.0, abs=0.1)
    # a second later: the move has landed and the slider now shows the
    # MEASURED position (qPOS, batch 3), which carries the stage's sensor
    # noise -- a wider tolerance than the raw keyboard reading above
    assert res["slider_1s_later_um"] == pytest.approx(5.0, abs=0.1)


# ----------------------------------------------------------------------
# B1  single camera owner through a whole session
# ----------------------------------------------------------------------
def test_b1_single_thread_owns_the_camera(tmp_path):
    """Preview + moves + snapshot + exposure + sweep + disconnect: one thread
    on the camera, never two at once, no move rejected, preview resumed."""
    res, _, _ = run_repro("b1_single_owner.py", tmp_path=tmp_path, timeout=180)
    assert res["connected"]
    mon = res["monitor"]
    assert mon["gui_thread_calls"] == 0
    assert mon["camera_threads"] == ["camera-owner"]
    assert mon["overlaps"] == []
    assert res["moves_rejected"] == 0
    assert res["fps_delivered_during_moves"] > 0  # the stream kept delivering during the moves
    assert len(res["sweep_finished"]) == 1 and res["sweep_finished"][0][1:] == [0, 3]
    assert res["preview_resumed_after_sweep"] and res["preview_active_after_sweep"]
    assert not res["previewing_after_disconnect"]
    assert res["disconnect_answered"]  # the camera thread closed in time: not abandoned
    if TIMING:
        assert res["fps_delivered_during_moves"] >= 10
        assert res["t_disconnect_s"] < 3.0
    assert dialogs(res) == [] and res["errors"] == []


# ----------------------------------------------------------------------
# C12  worker/thread graveyard: no finding, the stress must keep passing
# ----------------------------------------------------------------------
def test_c12_graveyard_survives_back_to_back_captures(tmp_path):
    res, code, err = run_repro("c12_graveyard_stress.py", "300", tmp_path=tmp_path, timeout=180)
    assert code == 0, err[-2000:]
    assert res["completed"] == 300
    assert res["retired_len"] <= 4
