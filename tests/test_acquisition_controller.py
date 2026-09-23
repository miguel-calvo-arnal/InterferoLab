"""Tests for backend/acquisition/acquisition_controller.py with fake hardware."""

from __future__ import annotations

import os
import re
from datetime import datetime

import cv2
import numpy as np
import pytest
from helpers_acquisition import (
    FakeCamera,
    FakePiezo,
    LogCollector,
    make_rggb_frame,
    make_session,
    reference_superpixel,
    reference_superpixel_binned,
)
from pipython import GCSError

from backend.acquisition.acquisition_controller import AcquisitionSession, CancelFlag
from utils import bin12
from utils.camera_constants import BAYER_W_B, BAYER_W_R


@pytest.fixture
def session():
    """AcquisitionSession with its atexit hook removed; hardware detached on exit."""
    s = make_session()
    yield s
    s._camera = None
    s._piezo = None


@pytest.fixture
def fast_poll(monkeypatch):
    """Shrink the piezo polling period so wait loops finish in milliseconds."""
    monkeypatch.setattr(AcquisitionSession, "MOVE_POLL_S", 0.001)


def _base_cfg(tmp_path, **overrides) -> dict:
    cfg = {
        "start": 0.0,
        "end": 4.0,
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
    cfg.update(overrides)
    return cfg


def _attach(session, camera=None, piezo=None):
    session._camera = camera if camera is not None else FakeCamera()
    session._piezo = piezo if piezo is not None else FakePiezo()
    return session._camera, session._piezo


def _output_files(folder: str) -> list[str]:
    """Everything the sweep wrote except positions.csv (batch 6), which is
    written for every sweep and has its own tests."""
    return sorted(f for f in os.listdir(folder) if f != AcquisitionSession.POSITIONS_CSV)


def _positions_rows(folder: str) -> list[dict]:
    import csv

    with open(
        os.path.join(folder, AcquisitionSession.POSITIONS_CSV), newline="", encoding="utf-8"
    ) as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# CancelFlag
# ---------------------------------------------------------------------------
def test_cancel_flag_starts_cleared():
    """A fresh CancelFlag reports not cancelled."""
    assert CancelFlag().is_cancelled() is False


def test_cancel_flag_set_and_idempotent():
    """cancel() latches the flag; calling it again keeps it set."""
    flag = CancelFlag()
    flag.cancel()
    assert flag.is_cancelled() is True
    flag.cancel()
    assert flag.is_cancelled() is True


# ---------------------------------------------------------------------------
# _bayer_to_superpixel (mono, weighted)
# ---------------------------------------------------------------------------
def test_superpixel_hand_computed_block():
    """A single RGGB block matches the hand-computed weighted sum."""
    frame = np.array([[1000, 2000], [1500, 500]], dtype=np.uint16)
    # 0.257090*1000 + 0.311272*2000 + 0.311272*1500 + 0.120366*500 = 1406.725
    out = AcquisitionSession._bayer_to_superpixel(frame)
    assert out.shape == (1, 1)
    assert out.dtype == np.uint16
    assert out[0, 0] == 1407


def test_superpixel_site_mapping():
    """Each Bayer site (R, G1, G2, B) lands in the documented position."""
    frame = np.zeros((2, 2), dtype=np.uint16)
    frame[0, 0] = 4000  # R site only
    out_r = AcquisitionSession._bayer_to_superpixel(frame)[0, 0]
    assert out_r == round(0.257090 * 4000)  # 1028

    frame = np.zeros((2, 2), dtype=np.uint16)
    frame[1, 1] = 4000  # B site only
    out_b = AcquisitionSession._bayer_to_superpixel(frame)[0, 0]
    assert out_b == round(0.120366 * 4000)  # 481


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_superpixel_bit_exact_vs_naive_reference(seed):
    """The optimised merge is bit-exact with the naive float32 formula."""
    frame = make_rggb_frame(64, 80, seed=seed)
    np.testing.assert_array_equal(
        AcquisitionSession._bayer_to_superpixel(frame),
        reference_superpixel(frame),
    )


def test_superpixel_clips_saturated_input_to_4095():
    """Out-of-range uint16 input (>4095) is clipped to the 12-bit ceiling."""
    frame = np.full((4, 4), 65535, dtype=np.uint16)
    out = AcquisitionSession._bayer_to_superpixel(frame)
    assert (out == 4095).all()


def test_superpixel_full_scale_12bit_stays_4095():
    """An all-4095 frame maps to 4095 (weights sum to 1, no overflow)."""
    frame = np.full((6, 6), 4095, dtype=np.uint16)
    out = AcquisitionSession._bayer_to_superpixel(frame)
    assert (out == 4095).all()


def test_superpixel_rejects_odd_dimensions():
    """Odd frame dimensions raise ValueError."""
    with pytest.raises(ValueError, match="even dimensions"):
        AcquisitionSession._bayer_to_superpixel(np.zeros((3, 4), dtype=np.uint16))


def test_superpixel_rejects_non_2d():
    """A 3-D input raises ValueError."""
    with pytest.raises(ValueError, match="2-D Bayer frame"):
        AcquisitionSession._bayer_to_superpixel(np.zeros((4, 4, 3), dtype=np.uint16))


# ---------------------------------------------------------------------------
# _bayer_to_color_superpixel
# ---------------------------------------------------------------------------
def test_color_superpixel_hand_computed_block():
    """One RGGB block collapses to (R, (G1+G2)/2, B)."""
    frame = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    out = AcquisitionSession._bayer_to_color_superpixel(frame)
    assert out.shape == (1, 1, 3)
    assert out.dtype == np.uint16
    assert tuple(out[0, 0]) == (100, 250, 400)


def test_color_superpixel_half_values_round_to_even():
    """Green averages ending in .5 follow numpy round-half-to-even."""
    frame = np.array([[0, 1], [2, 0]], dtype=np.uint16)  # G = 1.5 -> 2
    assert AcquisitionSession._bayer_to_color_superpixel(frame)[0, 0, 1] == 2
    frame = np.array([[0, 0], [1, 0]], dtype=np.uint16)  # G = 0.5 -> 0
    assert AcquisitionSession._bayer_to_color_superpixel(frame)[0, 0, 1] == 0


def test_color_superpixel_shape_and_validation():
    """Output is (H/2, W/2, 3); odd dimensions raise ValueError."""
    out = AcquisitionSession._bayer_to_color_superpixel(make_rggb_frame(10, 12, seed=3))
    assert out.shape == (5, 6, 3)
    with pytest.raises(ValueError, match="even dimensions"):
        AcquisitionSession._bayer_to_color_superpixel(np.zeros((4, 5), dtype=np.uint16))


# ---------------------------------------------------------------------------
# _bayer_to_superpixel_binned (mono_superpixel: weighted merge + 2x2 binning)
# ---------------------------------------------------------------------------
def test_superpixel_binned_hand_computed_4x4_block():
    """A 4x4 Bayer frame collapses to the average of its four mono superpixels."""
    frame = np.array(
        [
            [1000, 2000, 1000, 2000],
            [1500, 500, 1500, 500],
            [0, 0, 4000, 4000],
            [0, 0, 4000, 4000],
        ],
        dtype=np.uint16,
    )
    # Mono superpixels: 1406.725 -> 1407 (twice), 0, 4000 -> mean = 1703.5
    # numpy round-half-to-even: 1703.5 -> 1704
    out = AcquisitionSession._bayer_to_superpixel_binned(frame)
    assert out.shape == (1, 1)
    assert out.dtype == np.uint16
    assert out[0, 0] == 1704


def test_superpixel_binned_quarter_resolution():
    """(H, W) Bayer input becomes (H/4, W/4) mono output."""
    out = AcquisitionSession._bayer_to_superpixel_binned(make_rggb_frame(16, 24, seed=7))
    assert out.shape == (4, 6)
    assert out.dtype == np.uint16


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_superpixel_binned_matches_naive_reference(seed):
    """The production conversion is bit-exact with the naive reference."""
    frame = make_rggb_frame(64, 80, seed=seed)
    np.testing.assert_array_equal(
        AcquisitionSession._bayer_to_superpixel_binned(frame),
        reference_superpixel_binned(frame),
    )


def test_superpixel_binned_crops_odd_intermediate_dims():
    """Odd mono dims (from a 6x10 Bayer frame) are cropped before binning."""
    out = AcquisitionSession._bayer_to_superpixel_binned(make_rggb_frame(6, 10, seed=1))
    assert out.shape == (1, 2)  # mono 3x5 -> cropped 2x4 -> binned 1x2


def test_superpixel_binned_output_width_always_even():
    """An odd binned width is cropped so the frame stays bin12-compatible."""
    out = AcquisitionSession._bayer_to_superpixel_binned(make_rggb_frame(8, 12, seed=2))
    assert out.shape == (2, 2)  # mono 4x6 -> binned 2x3 -> cropped 2x2
    assert out.shape[1] % 2 == 0


def test_superpixel_binned_full_scale_stays_4095():
    """An all-4095 frame maps to 4095 (weights sum to 1; binning is an average)."""
    out = AcquisitionSession._bayer_to_superpixel_binned(np.full((8, 8), 4095, np.uint16))
    assert (out == 4095).all()


def test_superpixel_binned_rejects_odd_bayer_dimensions():
    """Raw-frame validation of the first stage still applies."""
    with pytest.raises(ValueError, match="even dimensions"):
        AcquisitionSession._bayer_to_superpixel_binned(np.zeros((6, 5), dtype=np.uint16))


# ---------------------------------------------------------------------------
# _save_bin12 round-trips
# ---------------------------------------------------------------------------
def test_save_bin12_mono_roundtrip(tmp_path):
    """A mono frame saved as .bin12 unpacks back bit-exactly."""
    frame = (make_rggb_frame(6, 8, seed=5) % 4096).astype(np.uint16)
    path = str(tmp_path / "f.bin12")
    AcquisitionSession._save_bin12(frame, path)

    header = bin12.read_header(path)
    assert header == (8, 6, 12, 1)
    with open(path, "rb") as fh:
        fh.seek(bin12.HEADER_SIZE)
        data = fh.read()
    restored = bin12.unpack_pixels(data, 6 * 8).reshape(6, 8)
    np.testing.assert_array_equal(restored, frame)


def test_save_bin12_color_roundtrip(tmp_path):
    """A BGR frame saved as .bin12 stores R, G, B row-interleaved planes."""
    rng = np.random.default_rng(11)
    bgr = rng.integers(0, 4096, size=(4, 6, 3), dtype=np.uint16)
    path = str(tmp_path / "c.bin12")
    AcquisitionSession._save_bin12(bgr, path)

    header = bin12.read_header(path)
    assert header == (6, 4, 12, 3)
    row_bytes = 6 * 3 // 2
    with open(path, "rb") as fh:
        fh.seek(bin12.HEADER_SIZE)
        data = fh.read()
    channels = {name: np.empty((4, 6), dtype=np.uint16) for name in "RGB"}
    off = 0
    for y in range(4):
        for name in "RGB":
            channels[name][y] = bin12.unpack_pixels(data[off : off + row_bytes], 6)
            off += row_bytes
    np.testing.assert_array_equal(channels["R"], bgr[:, :, 2])
    np.testing.assert_array_equal(channels["G"], bgr[:, :, 1])
    np.testing.assert_array_equal(channels["B"], bgr[:, :, 0])


# ---------------------------------------------------------------------------
# run_sweep validation and behaviour (all with fake hardware)
# ---------------------------------------------------------------------------
def test_run_sweep_requires_connected_hardware(session, tmp_path):
    """run_sweep without camera/piezo raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not connected"):
        session.run_sweep(_base_cfg(tmp_path))


def test_run_sweep_rejects_non_positive_step(session, tmp_path):
    """step <= 0 raises ValueError before touching the hardware loop."""
    _attach(session)
    with pytest.raises(ValueError, match="STEP must be > 0"):
        session.run_sweep(_base_cfg(tmp_path, step=0.0))
    with pytest.raises(ValueError, match="STEP must be > 0"):
        session.run_sweep(_base_cfg(tmp_path, step=-1.0))


def test_run_sweep_aborts_when_set_exposure_fails(session, tmp_path):
    """A set_exposure failure aborts the sweep with RuntimeError."""
    _attach(session, camera=FakeCamera(set_exposure_error=RuntimeError("io error")))
    with pytest.raises(RuntimeError, match="Could not set sweep exposure"):
        session.run_sweep(_base_cfg(tmp_path))


def test_run_sweep_aborts_on_exposure_mismatch(session, tmp_path):
    """A read-back exposure off by >5% aborts the sweep."""
    _attach(session, camera=FakeCamera(exposure_scale=1.2))
    log = LogCollector()
    with pytest.raises(RuntimeError, match="exposure verification failed"):
        session.run_sweep(_base_cfg(tmp_path), log_cb=log)
    assert log.contains("verification failed", level="error")


def test_run_sweep_tolerates_unreadable_exposure(session, tmp_path):
    """If get_exposure raises, the sweep warns but still completes."""
    _attach(session, camera=FakeCamera(get_exposure_error=RuntimeError("no readback")))
    log = LogCollector()
    folder = session.run_sweep(_base_cfg(tmp_path), log_cb=log)
    assert len(_output_files(folder)) == 5
    assert log.contains("Could not read back camera exposure", level="warn")


def test_run_sweep_creates_named_tiff_files(session, tmp_path):
    """A closed-loop sweep writes piezo_<z>um_<idx>.tiff for every position."""
    cam, piezo = _attach(session)
    log = LogCollector()
    folder = session.run_sweep(_base_cfg(tmp_path), log_cb=log)

    files = _output_files(folder)
    assert len(files) == 5
    pattern = re.compile(r"^piezo_-?\d+\.\d{4}um_\d{4}\.tiff$")
    assert all(pattern.match(f) for f in files)
    assert [z for _, z in piezo.moves] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert session.last_sweep_total_frames == 5
    assert session.last_sweep_skipped_frames == 0
    assert log.contains("Sweep finished", level="info")

    # Saved pixel data matches the mono superpixel of the fake frame.
    img = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(img, reference_superpixel(cam.frame))


def test_run_sweep_bin12_files_roundtrip(session, tmp_path):
    """format='bin12' writes .bin12 files that decode to the superpixel frame."""
    cam, _ = _attach(session)
    folder = session.run_sweep(_base_cfg(tmp_path, format="bin12"))
    files = _output_files(folder)
    assert len(files) == 5
    assert all(f.endswith(".bin12") for f in files)

    path = os.path.join(folder, files[0])
    header = bin12.read_header(path)
    assert (header.width, header.height, header.channels) == (4, 4, 1)
    with open(path, "rb") as fh:
        fh.seek(bin12.HEADER_SIZE)
        restored = bin12.unpack_pixels(fh.read(), 16).reshape(4, 4)
    np.testing.assert_array_equal(restored, reference_superpixel(cam.frame))


def test_run_sweep_color_mode_saves_three_channels(session, tmp_path):
    """color_mode='color' saves 3-channel frames."""
    _attach(session)
    folder = session.run_sweep(_base_cfg(tmp_path, color_mode="color"))
    files = _output_files(folder)
    assert len(files) == 5
    img = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_UNCHANGED)
    assert img.shape == (4, 4, 3)


def test_run_sweep_mono_superpixel_saves_quarter_res_mono(session, tmp_path):
    """color_mode='mono_superpixel' saves quarter-resolution mono frames."""
    cam, _ = _attach(session, camera=FakeCamera(frame=make_rggb_frame(16, 16, seed=9)))
    for fmt in ("tiff", "png"):
        folder = session.run_sweep(
            _base_cfg(
                tmp_path,
                color_mode="mono_superpixel",
                format=fmt,
                # Distinct base per format: the timestamped subfolder has 1 s
                # resolution and both sweeps may run within the same second.
                output_folder=str(tmp_path / f"data_{fmt}"),
            )
        )
        files = _output_files(folder)
        assert len(files) == 5
        img = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_UNCHANGED)
        assert img.shape == (4, 4)  # 16x16 raw -> 8x8 mono -> 4x4 binned
        np.testing.assert_array_equal(img, reference_superpixel_binned(cam.frame))


def test_run_sweep_mono_superpixel_bin12_roundtrip(session, tmp_path):
    """bin12 files written in mono_superpixel mode decode to the binned frame."""
    cam, _ = _attach(session, camera=FakeCamera(frame=make_rggb_frame(16, 16, seed=10)))
    folder = session.run_sweep(_base_cfg(tmp_path, color_mode="mono_superpixel", format="bin12"))
    files = _output_files(folder)
    assert len(files) == 5

    path = os.path.join(folder, files[0])
    header = bin12.read_header(path)
    assert (header.width, header.height, header.channels) == (4, 4, 1)
    with open(path, "rb") as fh:
        fh.seek(bin12.HEADER_SIZE)
        restored = bin12.unpack_pixels(fh.read(), 16).reshape(4, 4)
    np.testing.assert_array_equal(restored, reference_superpixel_binned(cam.frame))


def test_run_sweep_mono_superpixel_files_are_4x_smaller(session, tmp_path):
    """The superpixel stack is ~4x smaller than mono and ~12x smaller than color."""
    frame = make_rggb_frame(32, 32, seed=11)

    def first_file_size(color_mode: str) -> int:
        _attach(session, camera=FakeCamera(frame=frame))
        folder = session.run_sweep(
            _base_cfg(
                tmp_path,
                color_mode=color_mode,
                format="bin12",
                output_folder=str(tmp_path / f"data_{color_mode}"),
            )
        )
        path = os.path.join(folder, _output_files(folder)[0])
        return os.path.getsize(path)

    size_mono = first_file_size("mono")
    size_color = first_file_size("color")
    size_sp = first_file_size("mono_superpixel")
    # Payloads: mono 16x16, color 16x16x3, superpixel 8x8 (all 12-bit packed).
    header = bin12.HEADER_SIZE
    assert (size_mono - header) == 4 * (size_sp - header)
    assert (size_color - header) == 12 * (size_sp - header)


def test_run_sweep_mono_superpixel_preview_gets_binned_frame(session, tmp_path):
    """Preview callbacks receive the converted (quarter-res mono) frame."""
    _attach(session, camera=FakeCamera(frame=make_rggb_frame(16, 16, seed=12)))
    previews = []
    session.run_sweep(
        _base_cfg(tmp_path, color_mode="mono_superpixel"),
        preview_cb=lambda frame, z: previews.append(frame),
    )
    assert len(previews) == 5
    assert all(p.shape == (4, 4) for p in previews)


def test_capture_preview_mono_superpixel(session):
    """capture_preview honours the mono_superpixel colour mode."""
    cam = FakeCamera(frame=make_rggb_frame(16, 16, seed=13))
    _attach(session, camera=cam)
    session.color_mode = "mono_superpixel"
    frame, _z = session.capture_preview(timeout=1.0)
    np.testing.assert_array_equal(frame, reference_superpixel_binned(cam.frame))


def test_run_sweep_unknown_format_defaults_to_tiff(session, tmp_path):
    """An unknown format falls back to TIFF with a warning."""
    _attach(session)
    log = LogCollector()
    folder = session.run_sweep(_base_cfg(tmp_path, format="jpeg2000"), log_cb=log)
    assert all(f.endswith(".tiff") for f in _output_files(folder))
    assert log.contains("Unknown format", level="warn")


def test_run_sweep_counts_skipped_frames_and_completes(session, tmp_path):
    """Snap failures are skipped, counted, and reported in the summary."""
    _attach(session, camera=FakeCamera(fail_snap_at=(1, 3)))
    log = LogCollector()
    folder = session.run_sweep(_base_cfg(tmp_path), log_cb=log)

    assert len(_output_files(folder)) == 3  # 5 planned - 2 failed snaps
    assert session.last_sweep_skipped_frames == 2
    assert session.last_sweep_total_frames == 5
    assert log.contains("Incomplete dataset: 2 of 5", level="warn")
    assert log.contains("Sweep finished", level="info")
    assert session.last_sweep_aborted is None  # isolated failures never abort
    assert AcquisitionSession.ABORT_MARKER not in _output_files(folder)


def test_run_sweep_aborts_after_consecutive_capture_failures(session, tmp_path):
    """Batch 2 (C4): three failed captures in a row stop the sweep instead of
    walking every remaining position; the partial folder is marked."""
    # 10 positions; frames 0-1 good, the camera dies from frame 2 on
    _, piezo = _attach(session, camera=FakeCamera(fail_snap_at=tuple(range(2, 100))))
    log = LogCollector()
    folder = session.run_sweep(_base_cfg(tmp_path, end=9.0), log_cb=log)

    assert session.last_sweep_total_frames == 10
    assert session._camera.snap_count == 2 + AcquisitionSession.SWEEP_MAX_CONSECUTIVE_FAILURES
    assert len(piezo.moves) == 5  # no MOV after the third failure
    assert session.last_sweep_skipped_frames == 8  # 3 failed + 5 never attempted
    summary = session.last_sweep_aborted
    assert "3 consecutive camera failures" in summary
    assert "2 of 10 frame(s) saved" in summary and folder in summary
    assert log.contains("Sweep aborted after 3 consecutive", level="error")
    assert not log.contains("Sweep finished")
    assert all(m.strip() for m in log.messages("error"))

    files = _output_files(folder)
    assert AcquisitionSession.ABORT_MARKER in files
    assert len([f for f in files if f.endswith(".tiff")]) == 2
    with open(os.path.join(folder, AcquisitionSession.ABORT_MARKER), encoding="utf-8") as fh:
        marker = fh.read()
    assert "INCOMPLETE DATASET" in marker
    assert "frames_saved = 2" in marker and "frames_planned = 10" in marker


def test_run_sweep_aborts_after_consecutive_save_failures(session, tmp_path, monkeypatch):
    """Review H4: a full disk (every save fails) aborts after 3, like the camera."""
    import backend.acquisition.acquisition_controller as ac

    monkeypatch.setattr(ac.cv2, "imwrite", lambda *a, **k: False)
    _, piezo = _attach(session)
    log = LogCollector()
    folder = session.run_sweep(_base_cfg(tmp_path, end=9.0), log_cb=log)
    assert session._camera.snap_count == AcquisitionSession.SWEEP_MAX_CONSECUTIVE_FAILURES
    assert len(piezo.moves) == 3
    assert "3 consecutive save failures" in session.last_sweep_aborted
    assert "0 of 10 frame(s) saved" in session.last_sweep_aborted
    assert AcquisitionSession.ABORT_MARKER in _output_files(folder)


def test_run_sweep_failure_streak_is_reset_by_a_good_frame(session, tmp_path):
    """Two failures, a good frame, two failures: never three in a row, no abort."""
    _attach(session, camera=FakeCamera(fail_snap_at=(0, 1, 3, 4)))
    folder = session.run_sweep(_base_cfg(tmp_path, end=5.0))
    assert session.last_sweep_aborted is None
    assert session.last_sweep_skipped_frames == 4
    assert len(_output_files(folder)) == 2


def test_run_sweep_timeout_without_message_is_logged_with_text(session, tmp_path):
    """pylablib's snap timeout has an empty message (C7): the log must say what it was."""
    from helpers_acquisition import FakeCameraTimeoutError

    cam = FakeCamera()
    orig = cam.snap

    def snap(timeout=None):
        if cam.snap_count == 1:
            cam.snap_count += 1
            raise FakeCameraTimeoutError()
        return orig(timeout)

    cam.snap = snap
    _attach(session, camera=cam)
    log = LogCollector()
    session.run_sweep(_base_cfg(tmp_path), log_cb=log)
    errs = [m for m in log.messages("error") if "Camera timeout/error" in m]
    assert len(errs) == 1 and "did not deliver a frame in time" in errs[0]


def test_run_sweep_cancel_mid_sweep_stops_cleanly(session, tmp_path):
    """Cancelling mid-sweep stops the loop cleanly with partial output."""
    _attach(session)
    flag = CancelFlag()
    log = LogCollector()

    def progress_cb(percent, z, done, total, elapsed, eta):
        if done == 2:
            flag.cancel()

    folder = session.run_sweep(
        _base_cfg(tmp_path), progress_cb=progress_cb, log_cb=log, cancel_flag=flag
    )
    assert len(_output_files(folder)) == 2
    assert log.contains("Sweep cancelled by user", level="info")
    assert not log.contains("Sweep finished")


def test_run_sweep_open_loop_names_files_in_volts(session, tmp_path):
    """Open-loop sweeps use SVA/qVOL and name files piezo_<v>V_<idx>."""
    _, piezo = _attach(session)
    folder = session.run_sweep(_base_cfg(tmp_path, closed_loop=False))
    files = _output_files(folder)
    assert len(files) == 5
    pattern = re.compile(r"^piezo_-?\d+\.\d{4}V_\d{4}\.tiff$")
    assert all(pattern.match(f) for f in files)
    assert len(piezo.sva_calls) == 5
    assert piezo.moves == []  # MOV never used in open loop


def test_run_sweep_invokes_progress_and_preview(session, tmp_path):
    """Progress and preview callbacks fire once per acquired frame."""
    _attach(session)
    progress, previews = [], []
    session.run_sweep(
        _base_cfg(tmp_path),
        progress_cb=lambda *a: progress.append(a),
        preview_cb=lambda frame, z: previews.append((frame.shape, z)),
    )
    assert len(progress) == 5
    assert progress[-1][0] == pytest.approx(100.0)
    assert len(previews) == 5
    assert previews[0][0] == (4, 4)


# ---------------------------------------------------------------------------
# _wait_on_target
# ---------------------------------------------------------------------------
def test_wait_on_target_waits_k_polls(session, fast_poll):
    """The wait returns True after qONT reports on-target on the k+1-th poll."""
    piezo = FakePiezo(ont_polls_needed=3)
    assert session._wait_on_target(piezo, timeout_s=5.0) is True
    assert piezo._polls == 4


def test_wait_on_target_cancel_aborts_quickly(session, fast_poll):
    """Cancellation mid-wait returns False within one polling period."""
    piezo = FakePiezo(ont_polls_needed=10**9)
    flag = CancelFlag()

    def hook(p):
        if p._polls >= 2:
            flag.cancel()

    piezo.qont_hook = hook
    assert session._wait_on_target(piezo, timeout_s=30.0, cancel_flag=flag) is False
    assert piezo._polls < 10


def test_wait_on_target_timeout_raises(session, fast_poll):
    """Never reaching the target raises TimeoutError at the deadline."""
    piezo = FakePiezo(ont_polls_needed=10**9)
    with pytest.raises(TimeoutError, match="did not reach target"):
        session._wait_on_target(piezo, timeout_s=0.05)


def test_wait_on_target_falls_back_when_qont_unavailable(session, fast_poll):
    """A controller WITHOUT qONT (GCS 2, unknown command) falls back to a
    settle-time wait and returns True."""

    class NoOntPiezo(FakePiezo):
        def qONT(self, axis):  # noqa: N802
            raise GCSError(AcquisitionSession.GCS_UNKNOWN_COMMAND)

    session._settle_time = 0.01
    assert session._wait_on_target(NoOntPiezo(), timeout_s=5.0) is True


@pytest.mark.parametrize("error", [GCSError(-7), RuntimeError("link down")])
def test_wait_on_target_raises_when_arrival_cannot_be_confirmed(session, fast_poll, error):
    """Batch 2 (C11): a qONT that keeps failing is a failed move, not an arrival."""

    class BrokenOntPiezo(FakePiezo):
        def qONT(self, axis):  # noqa: N802
            self._polls += 1
            raise error

    piezo = BrokenOntPiezo()
    with pytest.raises(RuntimeError, match="Could not confirm that the piezo reached"):
        session._wait_on_target(piezo, timeout_s=5.0)
    assert piezo._polls == AcquisitionSession.QONT_MAX_CONSECUTIVE_FAILURES


def test_wait_on_target_tolerates_a_transient_qont_error(session, fast_poll):
    """One transient comms error (e.g. GCS -7) is polled again, not fatal."""

    class FlakyOntPiezo(FakePiezo):
        def qONT(self, axis):  # noqa: N802
            self._polls += 1
            if self._polls == 1:
                raise GCSError(-7)
            return {axis: True}

    piezo = FlakyOntPiezo()
    assert session._wait_on_target(piezo, timeout_s=5.0) is True
    assert piezo._polls == 2


def test_move_to_fails_when_arrival_cannot_be_confirmed(session, fast_poll):
    """The manual-move path surfaces C11 as an error (the UI shows it)."""

    class BrokenOntPiezo(FakePiezo):
        def qONT(self, axis):  # noqa: N802
            raise GCSError(-3)

    _attach(session, piezo=BrokenOntPiezo())
    log = LogCollector()
    with pytest.raises(RuntimeError, match="Could not confirm"):
        session.move_to(10.0, log_cb=log)
    assert log.contains("Closed-loop move to 10.0000 failed", level="error")


# ---------------------------------------------------------------------------
# disconnect_all compartmentalisation
# ---------------------------------------------------------------------------
def test_disconnect_all_camera_failure_still_shuts_down_piezo(session):
    """A camera close() failure never prevents the piezo shutdown sequence."""
    cam = FakeCamera(
        stop_error=RuntimeError("stop failed"),
        close_error=RuntimeError("close failed"),
    )
    cam, piezo = _attach(session, camera=cam)
    session._closed_loop = True
    log = LogCollector()

    assert session.disconnect_all(log_cb=log) is True
    assert cam.closed is True  # close was attempted
    assert piezo.moves == [("A", 0.0)]  # piezo returned to 0 µm
    assert ("A", 0) in piezo.svo_calls  # servo switched off
    assert piezo.closed is True  # connection released
    assert session._camera is None and session._piezo is None
    assert log.contains("Could not close camera", level="warn")
    assert log.contains("Piezo connection closed", level="info")


def test_disconnect_all_piezo_reset_failure_still_closes_connection(session):
    """A failing piezo reset (MOV) still releases the connection."""
    piezo = FakePiezo()
    piezo.mov_error = RuntimeError("axis stuck")
    _attach(session, piezo=piezo)
    session._closed_loop = True
    log = LogCollector()

    assert session.disconnect_all(log_cb=log) is True
    assert piezo.closed is True
    assert log.contains("Could not reset piezo", level="warn")


def test_disconnect_all_is_idempotent(session):
    """Calling disconnect_all twice is safe and a no-op the second time."""
    cam, piezo = _attach(session)
    session.disconnect_all()
    assert session._camera is None and session._piezo is None
    assert session.disconnect_all() is True  # nothing left to release
    assert cam.closed is True and piezo.closed is True


# ---------------------------------------------------------------------------
# Bayer mosaic phase (the LP126CU is BGGR, not RGGB)
# ---------------------------------------------------------------------------
def test_color_superpixel_bggr_swaps_red_and_blue():
    """With phase "blue" the corners are read as B,G,G,R instead of R,G,G,B."""
    frame = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    out = AcquisitionSession._bayer_to_color_superpixel(frame, "blue")
    # (0,0) is blue and (1,1) is red, so the RGB triple is mirrored.
    assert tuple(out[0, 0]) == (400, 250, 100)


def test_color_superpixel_phase_only_permutes_channels():
    """Changing the phase reorders channels; it never invents or drops data."""
    frame = make_rggb_frame(8, 8, seed=11)
    rggb = AcquisitionSession._bayer_to_color_superpixel(frame, "red")
    bggr = AcquisitionSession._bayer_to_color_superpixel(frame, "blue")
    np.testing.assert_array_equal(bggr[:, :, 0], rggb[:, :, 2])
    np.testing.assert_array_equal(bggr[:, :, 2], rggb[:, :, 0])
    np.testing.assert_array_equal(bggr[:, :, 1], rggb[:, :, 1])


def test_mono_superpixel_bggr_applies_red_weight_to_the_real_red():
    """Under BGGR the red weight goes to (1,1); the greens are unaffected."""
    frame = np.array([[1000, 0], [0, 0]], dtype=np.uint16)  # only (0,0) lit
    rggb = AcquisitionSession._bayer_to_superpixel(frame, "red")
    bggr = AcquisitionSession._bayer_to_superpixel(frame, "blue")
    # (0,0) is red under RGGB and blue under BGGR: W_R vs W_B.
    assert rggb[0, 0] == round(1000 * BAYER_W_R)
    assert bggr[0, 0] == round(1000 * BAYER_W_B)


def test_mono_superpixel_default_phase_is_bit_exact_with_rggb():
    """Omitting the phase keeps the historical RGGB result bit-for-bit."""
    frame = make_rggb_frame(12, 10, seed=5)
    np.testing.assert_array_equal(
        AcquisitionSession._bayer_to_superpixel(frame),
        AcquisitionSession._bayer_to_superpixel(frame, "red"),
    )


def test_superpixel_binned_forwards_the_phase():
    """mono_superpixel binning uses the phase instead of always assuming RGGB."""
    frame = make_rggb_frame(8, 8, seed=7)
    assert not np.array_equal(
        AcquisitionSession._bayer_to_superpixel_binned(frame, "red"),
        AcquisitionSession._bayer_to_superpixel_binned(frame, "blue"),
    )


# ---------------------------------------------------------------------------
# Measured position (batch 3: C3, U3, U4)
# ---------------------------------------------------------------------------
def test_move_returns_and_reports_the_measured_position(session, fast_poll):
    """The position is qPOS, not the target (U4); healthy moves read it once."""
    _, piezo = _attach(session)
    piezo.position = 12.3456
    seen: list[float] = []
    assert session.move_to(12.3, position_cb=seen.append) == 12.3456
    assert seen == [12.3456]
    assert session.last_position == 12.3456
    assert piezo.qpos_calls == 1  # no qPOS on the first poll: nothing extra


@pytest.mark.parametrize("polls_needed, expected_qpos", [(0, 1), (1, 1), (3, 3)])
def test_live_position_costs_one_query_per_poll_after_the_first(
    session, fast_poll, polls_needed, expected_qpos
):
    _, piezo = _attach(session, piezo=FakePiezo(ont_polls_needed=polls_needed))
    seen: list[float] = []
    session.move_to(5.0, position_cb=seen.append)
    assert piezo.qpos_calls == expected_qpos
    assert len(seen) == expected_qpos  # every reading reaches the window


def test_failed_move_reports_where_the_stage_is(session, fast_poll):
    """C3: the MOV ran but the arrival never came: the stage position is read
    and reported (live during the wait, and at the end), then the error."""
    _, piezo = _attach(session, piezo=FakePiezo(ont_polls_needed=10**9))
    session.MOVE_TIMEOUT_S = 0.05
    piezo.position = 59.998
    seen: list[float] = []
    with pytest.raises(TimeoutError):
        session.move_to(60.0, position_cb=seen.append)
    assert session.last_position == 59.998
    assert seen and seen[-1] == 59.998 and len(seen) >= 2


def test_cancelled_move_reports_the_measured_position_not_the_target(session, fast_poll):
    _, piezo = _attach(session, piezo=FakePiezo(ont_polls_needed=10**9))
    flag = CancelFlag()
    piezo.qont_hook = lambda p: flag.cancel() if p._polls >= 1 else None
    piezo.position = 31.7
    assert session.move_to(40.0, cancel_flag=flag) == 31.7
    assert session.last_position == 31.7


def test_rejected_mov_reports_the_old_position(session, fast_poll):
    _, piezo = _attach(session)
    session.move_to(10.0)
    piezo.mov_error = GCSError(7)
    seen: list[float] = []
    with pytest.raises(GCSError):
        session.move_to(90.0, position_cb=seen.append)
    assert seen == [10.0] and session.last_position == 10.0


def test_unreadable_position_is_unknown_not_the_target(session, fast_poll):
    _, piezo = _attach(session)
    piezo.position = GCSError(-3)
    log = LogCollector()
    seen: list[float] = []
    z = session.move_to(20.0, log_cb=log, position_cb=seen.append)
    assert np.isnan(z) and np.isnan(seen[-1]) and np.isnan(session.last_position)
    assert log.contains("Could not read the piezo position", level="warn")


def test_sweep_names_files_by_target_and_reports_measured_z(session, tmp_path):
    """Filenames keep the commanded z (the analysis reads it); progress and
    the position indicator get the measured one."""
    _, piezo = _attach(session)
    piezo.position = 1.0123
    progress: list[float] = []
    folder = session.run_sweep(
        _base_cfg(tmp_path, end=1.0),
        progress_cb=lambda pct, z, *a: progress.append(z),
    )
    assert _output_files(folder) == ["piezo_0.0000um_0000.tiff", "piezo_1.0000um_0001.tiff"]
    assert progress == [1.0123, 1.0123]


def test_connect_piezo_reads_the_starting_position(session, monkeypatch):
    import backend.acquisition.acquisition_controller as ac

    class FakeDevice(FakePiezo):
        def __init__(self, devname=None, gcsdll=None):
            super().__init__()
            self.position = 47.25

        def ConnectUSB(self, serialnum):  # noqa: N802
            pass

        def qIDN(self):  # noqa: N802
            return "fake"

        def qERR(self):  # noqa: N802
            return 0

    monkeypatch.setattr(ac, "GCSDevice", FakeDevice)
    assert session.connect_piezo("123", "dll") is True
    assert session.last_position == 47.25


# ---------------------------------------------------------------------------
# disconnect_all (batch 4)
# ---------------------------------------------------------------------------
def test_disconnect_park_timeout_still_switches_the_servo_off(session, fast_poll):
    """A piezo that never reports reaching 0: warned, servo off, link closed."""
    _, piezo = _attach(session, piezo=FakePiezo(ont_polls_needed=10**9))
    session._closed_loop = True
    log = LogCollector()
    assert session.disconnect_all(log_cb=log, park_timeout_s=0.05) is True
    assert piezo.moves[-1] == ("A", 0.0)
    assert piezo.svo_calls == [("A", 0)] and piezo.closed
    assert log.contains("did not report reaching 0 within", level="warn")
    assert not log.contains("Could not reset piezo")


def test_disconnect_cancel_flag_cuts_the_park_wait(session, fast_poll):
    _, piezo = _attach(session, piezo=FakePiezo(ont_polls_needed=10**9))
    session._closed_loop = True
    flag = CancelFlag()
    flag.cancel()
    session.disconnect_all(park_timeout_s=30.0, cancel_flag=flag)
    assert piezo._polls == 0  # the wait ended before its first poll
    assert piezo.svo_calls == [("A", 0)] and piezo.closed


# ---------------------------------------------------------------------------
# positions.csv (batch 6)
# ---------------------------------------------------------------------------
def test_positions_csv_has_one_row_per_saved_frame_in_order(session, tmp_path):
    """Header, one row per saved frame, in sweep order, with both z values."""
    _, piezo = _attach(session)
    piezo.position = 2.0345  # qPOS answers this for every step
    folder = session.run_sweep(_base_cfg(tmp_path, end=2.0))

    rows = _positions_rows(folder)
    assert [r["index"] for r in rows] == ["0", "1", "2"]
    assert [r["filename"] for r in rows] == _output_files(folder)
    assert [r["z_commanded_um"] for r in rows] == ["0.0000", "1.0000", "2.0000"]
    assert [r["z_measured_after_move_um"] for r in rows] == ["2.0345"] * 3
    assert list(rows[0].keys()) == list(AcquisitionSession.POSITIONS_HEADER)
    # timestamps are readable and ordered
    stamps = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    assert stamps == sorted(stamps)
    # the image files are exactly the ones the sweep always produced
    assert _output_files(folder) == [
        "piezo_0.0000um_0000.tiff",
        "piezo_1.0000um_0001.tiff",
        "piezo_2.0000um_0002.tiff",
    ]


def test_positions_csv_says_nan_when_the_position_cannot_be_read(session, tmp_path):
    _, piezo = _attach(session)
    piezo.position = GCSError(-3)  # qPOS fails: the measurement is unknown
    folder = session.run_sweep(_base_cfg(tmp_path, end=1.0))
    rows = _positions_rows(folder)
    assert [r["z_measured_after_move_um"] for r in rows] == ["nan", "nan"]
    assert [r["z_commanded_um"] for r in rows] == ["0.0000", "1.0000"]


def test_positions_csv_skips_frames_that_were_not_saved(session, tmp_path):
    """A failed capture leaves no file and no row (the csv matches the disk)."""
    _attach(session, camera=FakeCamera(fail_snap_at=(1,)))
    folder = session.run_sweep(_base_cfg(tmp_path, end=2.0))
    rows = _positions_rows(folder)
    assert [r["index"] for r in rows] == ["0", "2"]
    assert [r["filename"] for r in rows] == _output_files(folder)


def test_positions_csv_of_an_aborted_sweep_matches_the_marker(session, tmp_path):
    """Batch 2 + 6: the csv keeps the frames that were saved before the abort."""
    _attach(session, camera=FakeCamera(fail_snap_at=tuple(range(2, 100))))
    folder = session.run_sweep(_base_cfg(tmp_path, end=9.0))
    rows = _positions_rows(folder)
    assert [r["index"] for r in rows] == ["0", "1"]
    assert [r["filename"] for r in rows] == [
        f for f in _output_files(folder) if f.startswith("piezo_")
    ]
    assert AcquisitionSession.ABORT_MARKER in _output_files(folder)
    assert "2 of 10 frame(s) saved" in session.last_sweep_aborted
    with open(os.path.join(folder, AcquisitionSession.ABORT_MARKER), encoding="utf-8") as fh:
        assert f"frames_saved = {len(rows)}" in fh.read()


def test_positions_csv_of_a_cancelled_sweep_keeps_what_was_saved(session, tmp_path):
    _attach(session)
    flag = CancelFlag()

    def progress_cb(percent, z, done, total, elapsed, eta):
        if done == 2:
            flag.cancel()

    folder = session.run_sweep(_base_cfg(tmp_path), progress_cb=progress_cb, cancel_flag=flag)
    rows = _positions_rows(folder)
    assert [r["index"] for r in rows] == ["0", "1"]
    assert [r["filename"] for r in rows] == _output_files(folder)


def test_positions_csv_is_written_as_the_sweep_goes(session, tmp_path):
    """Nothing is kept in memory until the end: the row of a frame is on disk
    before the next frame is captured (checked from the camera's snap())."""
    seen: list[int] = []
    cam = FakeCamera()
    orig = cam.snap

    def counting_snap(timeout=None):
        folder = getattr(session, "_test_folder", None)
        if folder is not None:
            seen.append(len(_positions_rows(folder)))
        return orig(timeout)

    cam.snap = counting_snap
    _attach(session, camera=cam)
    real_create = AcquisitionSession.create_output_folder

    def create_output_folder(base_folder, log_cb):
        folder = real_create(base_folder, log_cb)
        session._test_folder = folder
        return folder

    session.create_output_folder = staticmethod(create_output_folder)
    folder = session.run_sweep(_base_cfg(tmp_path, end=3.0))
    # one row already flushed before each capture but the first
    assert seen == [0, 1, 2, 3]
    assert len(_positions_rows(folder)) == 4


def test_positions_csv_failure_never_stops_the_sweep(session, tmp_path, monkeypatch):
    """A folder that cannot take the csv: warned once, the sweep goes on."""
    _attach(session)
    real_open = open

    def failing_open(path, *a, **k):
        if str(path).endswith(AcquisitionSession.POSITIONS_CSV):
            raise OSError("read-only file system")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", failing_open)
    log = LogCollector()
    folder = session.run_sweep(_base_cfg(tmp_path, end=1.0), log_cb=log)
    assert log.contains("Could not write positions.csv", level="warn")
    assert len(_output_files(folder)) == 2
    assert log.contains("Sweep finished", level="info")
