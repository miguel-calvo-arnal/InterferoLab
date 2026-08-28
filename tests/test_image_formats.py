"""Tests for services/image_formats.py: header-only channel detectors."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from services import image_formats
from utils import bin12


def _mono_frame(h: int = 4, w: int = 6) -> np.ndarray:
    return (np.arange(h * w, dtype=np.uint16).reshape(h, w) * 100) % 4096


def _color_frame(h: int = 4, w: int = 6) -> np.ndarray:
    return np.dstack([np.full((h, w), v, dtype=np.uint16) for v in (100, 200, 300)])


def _write_bin12(path, frame) -> None:
    with open(path, "wb") as fh:
        fh.write(bin12.pack_frame(frame))


# ---------------------------------------------------------------------------
# bin12_n_channels
# ---------------------------------------------------------------------------
def test_bin12_mono_channels(tmp_path):
    """A mono .bin12 file written with utils.bin12 reports 1 channel."""
    p = tmp_path / "m.bin12"
    _write_bin12(p, _mono_frame())
    assert image_formats.bin12_n_channels(str(p)) == 1


def test_bin12_color_channels(tmp_path):
    """A 3-channel .bin12 file written with utils.bin12 reports 3 channels."""
    p = tmp_path / "c.bin12"
    _write_bin12(p, _color_frame())
    assert image_formats.bin12_n_channels(str(p)) == 3


def test_bin12_corrupt_returns_minus_one(tmp_path):
    """Garbage bytes with a .bin12 extension yield -1, not an exception."""
    p = tmp_path / "bad.bin12"
    p.write_bytes(b"not a bin12 file at all........")
    assert image_formats.bin12_n_channels(str(p)) == -1


def test_bin12_truncated_header_returns_minus_one(tmp_path):
    """A .bin12 file shorter than the 20-byte header yields -1."""
    p = tmp_path / "short.bin12"
    p.write_bytes(b"B12\x00\x01")
    assert image_formats.bin12_n_channels(str(p)) == -1


# ---------------------------------------------------------------------------
# tiff_n_channels
# ---------------------------------------------------------------------------
def test_tiff_mono16_channels(tmp_path):
    """A cv2-written 16-bit mono TIFF reports SamplesPerPixel == 1."""
    p = tmp_path / "m.tiff"
    assert cv2.imwrite(str(p), _mono_frame())
    assert image_formats.tiff_n_channels(str(p)) == 1


def test_tiff_color_channels(tmp_path):
    """A cv2-written 3-channel TIFF reports SamplesPerPixel == 3."""
    p = tmp_path / "c.tiff"
    assert cv2.imwrite(str(p), _color_frame().astype(np.uint8))
    assert image_formats.tiff_n_channels(str(p)) == 3


def test_tiff_garbage_returns_minus_one(tmp_path):
    """A non-TIFF payload with .tiff extension yields -1."""
    p = tmp_path / "bad.tiff"
    p.write_bytes(b"\x00\x01garbage-not-a-tiff")
    assert image_formats.tiff_n_channels(str(p)) == -1


def test_tiff_empty_file_returns_minus_one(tmp_path):
    """An empty .tiff file yields -1 (header shorter than 8 bytes)."""
    p = tmp_path / "empty.tiff"
    p.write_bytes(b"")
    assert image_formats.tiff_n_channels(str(p)) == -1


def test_tiff_missing_file_returns_minus_one(tmp_path):
    """A nonexistent path yields -1 rather than raising."""
    assert image_formats.tiff_n_channels(str(tmp_path / "nope.tiff")) == -1


# ---------------------------------------------------------------------------
# png_n_channels
# ---------------------------------------------------------------------------
def test_png_mono8_channels(tmp_path):
    """A cv2-written 8-bit grayscale PNG (color type 0) reports 1 channel."""
    p = tmp_path / "m8.png"
    assert cv2.imwrite(str(p), _mono_frame().astype(np.uint8))
    assert image_formats.png_n_channels(str(p)) == 1


def test_png_mono16_channels(tmp_path):
    """A cv2-written 16-bit grayscale PNG still reports 1 channel."""
    p = tmp_path / "m16.png"
    assert cv2.imwrite(str(p), _mono_frame())
    assert image_formats.png_n_channels(str(p)) == 1


def test_png_color_channels(tmp_path):
    """A cv2-written BGR PNG (color type 2) reports 3 channels."""
    p = tmp_path / "c.png"
    assert cv2.imwrite(str(p), _color_frame().astype(np.uint8))
    assert image_formats.png_n_channels(str(p)) == 3


def test_png_garbage_returns_minus_one(tmp_path):
    """A payload without the PNG signature yields -1."""
    p = tmp_path / "bad.png"
    p.write_bytes(b"x" * 40)
    assert image_formats.png_n_channels(str(p)) == -1


def test_png_truncated_returns_minus_one(tmp_path):
    """A valid PNG signature truncated before the IHDR data yields -1."""
    p = tmp_path / "trunc.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)
    assert image_formats.png_n_channels(str(p)) == -1


# ---------------------------------------------------------------------------
# detect_dataset_format
# ---------------------------------------------------------------------------
def test_detect_bin12_mono_folder(tmp_path):
    """A folder of mono .bin12 frames is detected as 'mono'."""
    _write_bin12(tmp_path / "piezo_0.0000um_0000.bin12", _mono_frame())
    assert image_formats.detect_dataset_format(str(tmp_path)) == "mono"


def test_detect_tiff_color_folder(tmp_path):
    """A folder of 3-channel TIFF frames is detected as 'color'."""
    cv2.imwrite(str(tmp_path / "f0.tiff"), _color_frame().astype(np.uint8))
    assert image_formats.detect_dataset_format(str(tmp_path)) == "color"


def test_detect_png_mono_folder(tmp_path):
    """A folder of grayscale PNG frames is detected as 'mono'."""
    cv2.imwrite(str(tmp_path / "f0.png"), _mono_frame().astype(np.uint8))
    assert image_formats.detect_dataset_format(str(tmp_path)) == "mono"


def test_detect_empty_folder_unknown(tmp_path):
    """An empty folder yields 'unknown'."""
    assert image_formats.detect_dataset_format(str(tmp_path)) == "unknown"


def test_detect_missing_folder_unknown(tmp_path):
    """A nonexistent folder yields 'unknown' rather than raising."""
    assert image_formats.detect_dataset_format(str(tmp_path / "missing")) == "unknown"


def test_detect_ignores_unrelated_extensions(tmp_path):
    """Files with non-dataset extensions are ignored -> 'unknown'."""
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "meta.json").write_text("{}")
    assert image_formats.detect_dataset_format(str(tmp_path)) == "unknown"


def test_detect_mixed_folder_uses_first_sorted_file(tmp_path):
    """With mixed formats, detection follows the alphabetically first frame."""
    _write_bin12(tmp_path / "a_first.bin12", _color_frame())  # color
    cv2.imwrite(str(tmp_path / "z_last.png"), _mono_frame().astype(np.uint8))  # mono
    assert image_formats.detect_dataset_format(str(tmp_path)) == "color"


def test_detect_corrupt_first_file_unknown(tmp_path):
    """A corrupt first frame makes the whole folder 'unknown'."""
    (tmp_path / "a.bin12").write_bytes(b"corrupt")
    assert image_formats.detect_dataset_format(str(tmp_path)) == "unknown"


@pytest.mark.parametrize("ext", [".bin12", ".tiff", ".tif", ".png"])
def test_dataset_extensions_recognised(ext):
    """All documented dataset extensions are listed in DATASET_EXTENSIONS."""
    assert ext in image_formats.DATASET_EXTENSIONS


def test_detect_reads_only_header_bytes(tmp_path):
    """Detection succeeds even when pixel data past the header is corrupt."""
    p = tmp_path / "f.bin12"
    good = bin12.pack_frame(_mono_frame())
    p.write_bytes(good[: bin12.HEADER_SIZE] + os.urandom(len(good) - bin12.HEADER_SIZE))
    assert image_formats.detect_dataset_format(str(tmp_path)) == "mono"
