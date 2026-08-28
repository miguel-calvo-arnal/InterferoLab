"""Tests for utils/bin12.py — the packed 12-bit .bin12 image format."""

from __future__ import annotations

import numpy as np
import pytest

from utils.bin12 import (
    HEADER,
    HEADER_SIZE,
    MAGIC,
    Bin12Header,
    pack_frame,
    pack_pixels,
    read_header,
    unpack_pixels,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_header_size_is_20_bytes():
    """The binary header layout is exactly 20 bytes (4s + 4 uint32)."""
    assert HEADER.size == 20
    assert HEADER_SIZE == 20


def test_magic_is_b12_nul():
    """The magic number matches the C++ reference reader ("B12\\0")."""
    assert MAGIC == b"B12\x00"


def test_pack_unpack_pixels_roundtrip_even():
    """pack_pixels/unpack_pixels round-trip an even-length array bit-exactly."""
    pixels = _rng().integers(0, 4096, size=1000, dtype=np.uint16)
    data = pack_pixels(pixels)
    assert len(data) == 1000 // 2 * 3
    out = unpack_pixels(data, 1000)
    np.testing.assert_array_equal(out, pixels)


def test_pack_unpack_pixels_roundtrip_odd_length():
    """Odd-length input is zero-padded on pack and truncated back on unpack."""
    pixels = np.array([1, 4095, 7], dtype=np.uint16)
    data = pack_pixels(pixels)
    assert len(data) == 6  # padded to 4 pixels -> 6 bytes
    np.testing.assert_array_equal(unpack_pixels(data, 3), pixels)
    # The padding pixel is stored as 0.
    np.testing.assert_array_equal(unpack_pixels(data, 4), [1, 4095, 7, 0])


def test_pack_unpack_pixels_boundary_values():
    """The extreme 12-bit values 0 and 4095 survive the byte packing exactly."""
    pixels = np.array([0, 4095, 4095, 0, 2048, 2047], dtype=np.uint16)
    np.testing.assert_array_equal(unpack_pixels(pack_pixels(pixels), 6), pixels)


def test_pack_frame_mono_roundtrip(tmp_path):
    """A mono frame written by pack_frame decodes bit-exactly via unpack_pixels."""
    h, w = 5, 8
    frame = _rng().integers(0, 4096, size=(h, w), dtype=np.uint16)
    blob = pack_frame(frame)
    path = tmp_path / "mono.bin12"
    path.write_bytes(blob)

    hdr = read_header(str(path))
    assert hdr == Bin12Header(width=w, height=h, bit_depth=12, channels=1)
    assert len(blob) == HEADER_SIZE + h * w * 3 // 2

    decoded = unpack_pixels(blob[HEADER_SIZE:], h * w).reshape(h, w)
    np.testing.assert_array_equal(decoded, frame)


def test_pack_frame_3channel_roundtrip(tmp_path):
    """A BGR frame is stored as row-interleaved R,G,B planes and round-trips."""
    h, w = 4, 6
    frame = _rng().integers(0, 4096, size=(h, w, 3), dtype=np.uint16)  # BGR
    blob = pack_frame(frame)
    path = tmp_path / "color.bin12"
    path.write_bytes(blob)

    hdr = read_header(str(path))
    assert hdr == Bin12Header(width=w, height=h, bit_depth=12, channels=3)

    # Each image row stores the packed R, G, B channel rows consecutively.
    decoded = unpack_pixels(blob[HEADER_SIZE:], h * 3 * w).reshape(h, 3, w)
    np.testing.assert_array_equal(decoded[:, 0, :], frame[:, :, 2])  # R
    np.testing.assert_array_equal(decoded[:, 1, :], frame[:, :, 1])  # G
    np.testing.assert_array_equal(decoded[:, 2, :], frame[:, :, 0])  # B


def test_pack_frame_boundary_values():
    """pack_frame preserves the 0 and 4095 extremes of the 12-bit range."""
    frame = np.array([[0, 4095], [4095, 0]], dtype=np.uint16)
    blob = pack_frame(frame)
    np.testing.assert_array_equal(unpack_pixels(blob[HEADER_SIZE:], 4).reshape(2, 2), frame)


def test_pack_frame_rejects_odd_width():
    """pack_frame refuses frames with odd width (2 pixels pack into 3 bytes)."""
    with pytest.raises(ValueError, match="even image width"):
        pack_frame(np.zeros((4, 5), dtype=np.uint16))


def test_pack_frame_rejects_bad_shape():
    """pack_frame rejects arrays that are neither (H, W) nor (H, W, 3)."""
    with pytest.raises(ValueError, match="shape"):
        pack_frame(np.zeros((4, 4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="shape"):
        pack_frame(np.zeros(16, dtype=np.uint16))


def test_read_header_bad_magic(tmp_path):
    """read_header returns None when the magic number does not match."""
    path = tmp_path / "bad.bin12"
    path.write_bytes(HEADER.pack(b"NOPE", 4, 4, 12, 1) + b"\x00" * 24)
    assert read_header(str(path)) is None


def test_read_header_truncated_file(tmp_path):
    """read_header returns None for a file shorter than the 20-byte header."""
    path = tmp_path / "short.bin12"
    path.write_bytes(MAGIC + b"\x00" * 3)
    assert read_header(str(path)) is None


def test_read_header_empty_and_missing(tmp_path):
    """read_header returns None for empty files and nonexistent paths."""
    empty = tmp_path / "empty.bin12"
    empty.write_bytes(b"")
    assert read_header(str(empty)) is None
    assert read_header(str(tmp_path / "does_not_exist.bin12")) is None


def test_pack_pixels_matches_pack_frame_layout():
    """pack_pixels and pack_frame's internal packer produce identical bytes."""
    frame = _rng().integers(0, 4096, size=(3, 10), dtype=np.uint16)
    assert pack_frame(frame)[HEADER_SIZE:] == pack_pixels(frame.ravel())
