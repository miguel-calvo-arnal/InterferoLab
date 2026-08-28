"""Tests for backend/acquisition/acquisition_controller.py with fake hardware."""

from __future__ import annotations

import os
import re

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

from backend.acquisition.acquisition_controller import AcquisitionSession, CancelFlag
from utils import bin12


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
    return sorted(os.listdir(folder))


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
    """A failing qONT falls back to a settle-time wait and returns True."""

    class NoOntPiezo(FakePiezo):
        def qONT(self, axis):  # noqa: N802
            raise RuntimeError("qONT not supported")

    session._settle_time = 0.01
    assert session._wait_on_target(NoOntPiezo(), timeout_s=5.0) is True


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
