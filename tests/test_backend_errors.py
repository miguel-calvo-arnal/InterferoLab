"""Error handling, cancellation and atomic-output tests for analysis_backend.

Covers the audit-hardened paths: min_frames validation per method, per-file
dimension checks, bin12 header validation, cooperative cancellation (flag
reset included) and the write-to-.partial + atomic-rename output publishing.
"""

from __future__ import annotations

import os
import struct
import threading
import time

import cv2
import helpers_backend as hb
import numpy as np
import pytest

# The C++ backend (.so) may not be compiled: skip cleanly.
analysis_backend = pytest.importorskip("analysis_backend")

from utils import bin12

SMALL_HEIGHTS = np.full((16, 16), 1.0)


def _write_small_png_dataset(folder: str, n_frames: int) -> None:
    """Write ``n_frames`` tiny valid frames with proper piezo filenames."""
    os.makedirs(folder, exist_ok=True)
    for i, z in enumerate(hb.z_positions(n_frames)):
        frame = hb.csi_frame(SMALL_HEIGHTS, z)
        assert cv2.imwrite(os.path.join(folder, hb.frame_filename(z, i, "png")), frame)


def _partial_leftovers() -> list[str]:
    """Names of stale .partial staging folders under ./output (must be none)."""
    if not os.path.isdir("output"):
        return []
    return [d for d in os.listdir("output") if d.endswith(".partial")]


@pytest.mark.parametrize(
    ("method", "n_frames"),
    [(m, n) for m in (1, 2, 3, 4) for n in (1, 2)] + [(3, 4), (4, 3)],
)
def test_too_few_frames_clear_error(in_tmp_cwd, method, n_frames):
    """Undersized datasets fail fast with a clear min_frames RuntimeError.

    Includes the per-method boundaries: method 3 needs >= 5 frames and
    method 4 needs >= 4, so 4 and 3 frames respectively must also fail.
    """
    _write_small_png_dataset("dataset", n_frames)
    with pytest.raises(RuntimeError, match="requires at least"):
        hb.run_backend("dataset", name="few", method=method)
    assert not os.path.exists(os.path.join("output", "few"))
    assert _partial_leftovers() == []


def test_mismatched_dimensions_clear_error(in_tmp_cwd):
    """A frame whose dimensions differ from the first file raises, not crashes."""
    heights, _, _ = hb.two_plateau_heights()
    hb.write_png_dataset("dataset", heights, nz=20)
    # One extra frame with different dimensions, sorted in the middle of the stack.
    odd = hb.csi_frame(heights[:20, :32], 0.5)
    assert cv2.imwrite(os.path.join("dataset", "piezo_0.5001um_0099.png"), odd)

    with pytest.raises(RuntimeError, match="dimensions differ"):
        hb.run_backend("dataset", name="mixdims", method=1)
    assert not os.path.exists(os.path.join("output", "mixdims"))
    assert _partial_leftovers() == []


def test_bin12_corrupt_magic_error(in_tmp_cwd):
    """A bin12 file with a corrupted magic number raises a clear error."""
    heights, _, _ = hb.two_plateau_heights()
    files = hb.write_bin12_dataset("dataset", heights, nz=8)
    with open(files[0], "rb") as fh:
        data = fh.read()
    with open(files[0], "wb") as fh:
        fh.write(b"XXX\x00" + data[4:])

    with pytest.raises(RuntimeError, match="magic"):
        hb.run_backend("dataset", name="badmagic", method=1)
    assert not os.path.exists(os.path.join("output", "badmagic"))


def test_bin12_invalid_channel_count_error(in_tmp_cwd):
    """A hand-crafted bin12 header with channels=2 fails header validation."""
    os.makedirs("dataset", exist_ok=True)
    for i, z in enumerate(hb.z_positions(8)):
        frame = hb.csi_frame(SMALL_HEIGHTS, z)
        body = bin12.pack_frame(frame)[bin12.HEADER_SIZE :]
        bad_header = struct.pack("<4sIIII", bin12.MAGIC, 16, 16, 12, 2)
        with open(os.path.join("dataset", hb.frame_filename(z, i, "bin12")), "wb") as fh:
            fh.write(bad_header + body)

    with pytest.raises(RuntimeError, match="channel count"):
        hb.run_backend("dataset", name="badch", method=1)
    assert not os.path.exists(os.path.join("output", "badch"))


def test_cancellation_and_flag_reset(in_tmp_cwd):
    """cancel_analysis() stops a running analysis quickly and leaves no output.

    The first progress callback (invoked synchronously before the
    reconstruction starts) sleeps 0.4 s, so cancelling ~0.1 s after it fires
    deterministically lands mid-run, before any output is written.
    Afterwards a normal run must succeed (the cancel flag is reset per run).
    """
    heights, _, _ = hb.two_plateau_heights()
    hb.write_png_dataset("dataset", heights)

    started = threading.Event()

    def slow_progress(percent, stage, done, total):
        if not started.is_set():
            started.set()
            time.sleep(0.4)  # hold the run so the cancel below lands mid-run

    result: dict = {}

    def worker():
        result["r"] = hb.run_backend(
            "dataset", name="cancelme", method=1, progress_cb=slow_progress
        )

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(5.0), "analysis never reported progress"
    time.sleep(0.1)
    t_cancel = time.monotonic()
    analysis_backend.cancel_analysis()
    thread.join(3.0)
    elapsed = time.monotonic() - t_cancel

    assert not thread.is_alive(), "run_analysis did not return within 3 s of cancel"
    assert elapsed < 3.0
    r = result["r"]
    assert r["cancelled"] is True
    assert r["output_folder"] == ""
    assert r["heightmap"] == ""
    assert not os.path.exists(os.path.join("output", "cancelme"))
    assert _partial_leftovers() == []

    # The flag is reset on the next call: a normal run now succeeds.
    r2 = hb.run_backend("dataset", name="after_cancel", method=1)
    assert r2["cancelled"] is False
    assert os.path.isfile(r2["heightmap"])
    assert _partial_leftovers() == []


def test_reanalysis_replaces_output_atomically(in_tmp_cwd):
    """Re-running after corrupting the output folder restores a complete result.

    The backend stages into <name>.partial and renames into place, so the
    final folder is always complete: the deleted heightmap reappears, foreign
    junk files are gone, and the data matches the first run exactly.
    """
    heights, _, _ = hb.two_plateau_heights()
    hb.write_png_dataset("dataset", heights)

    r1 = hb.run_backend("dataset", name="exp", method=1)
    hm1 = np.load(r1["heightmap"])

    # Corrupt the published output by hand.
    os.remove(r1["heightmap"])
    junk = os.path.join(r1["output_folder"], "junk_leftover.txt")
    with open(junk, "w", encoding="utf-8") as fh:
        fh.write("garbage")
    # A stale .partial from a hypothetical crashed run must also be cleaned.
    stale = os.path.join("output", "exp.partial")
    os.makedirs(stale, exist_ok=True)

    r2 = hb.run_backend("dataset", name="exp", method=1)
    assert r2["cancelled"] is False
    assert os.path.isfile(r2["heightmap"])
    assert not os.path.exists(junk), "old folder contents must be replaced wholesale"
    assert _partial_leftovers() == []
    assert os.path.isfile(os.path.join(r2["output_folder"], "metadata.json"))
    hm2 = np.load(r2["heightmap"])
    assert np.array_equal(hm1, hm2), "re-analysis of identical data must be reproducible"
