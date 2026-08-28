# utils/bin12.py
"""
Single source of truth for the .bin12 packed 12-bit image format (Python side).

File layout
-----------
- 20-byte little-endian header: magic "B12\\0", width, height, bit_depth,
  channels (four uint32).
- Pixel data: rows packed at 12 bits/pixel (2 pixels -> 3 bytes). For
  3-channel files each image row stores the R, G and B channel rows
  consecutively (row-interleaved planar layout).

The C++ reader in backend/analysis/src/reconstruction.cpp implements the same
layout and is the reference decoder; any format change must be mirrored there.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

import numpy as np

MAGIC = b"B12\x00"
HEADER = struct.Struct("<4sIIII")  # magic, width, height, bit_depth, channels
HEADER_SIZE = HEADER.size  # 20 bytes


class Bin12Header(NamedTuple):
    width: int
    height: int
    bit_depth: int
    channels: int


def read_header(path: str) -> Bin12Header | None:
    """
    Read and validate the header of a .bin12 file.

    Returns None if the file cannot be read, is too short, or the magic
    number does not match.
    """
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(HEADER_SIZE)
        if len(hdr) < HEADER_SIZE:
            return None
        magic, width, height, bit_depth, channels = HEADER.unpack(hdr)
        if magic != MAGIC:
            return None
        return Bin12Header(width, height, bit_depth, channels)
    except Exception:
        return None


def pack_pixels(pixels: np.ndarray) -> bytes:
    """
    Pack a flat array of 12-bit values (uint16) into the bin12 byte layout.

    Odd-length inputs are zero-padded to an even pixel count.
    """
    if len(pixels) % 2 != 0:
        pixels = np.append(pixels, 0)
    p0 = pixels[0::2]
    p1 = pixels[1::2]
    packed = np.empty(len(p0) * 3, dtype=np.uint8)
    packed[0::3] = p0 & 0xFF
    packed[1::3] = ((p0 >> 8) & 0x0F) | ((p1 & 0x0F) << 4)
    packed[2::3] = p1 >> 4
    return packed.tobytes()


def unpack_pixels(data: bytes, n_pixels: int) -> np.ndarray:
    """Unpack n_pixels 12-bit values (uint16) from a packed byte buffer."""
    buf = np.frombuffer(data, dtype=np.uint8)
    b0 = buf[0::3].astype(np.uint16)
    b1 = buf[1::3].astype(np.uint16)
    b2 = buf[2::3].astype(np.uint16)
    p0 = b0 | ((b1 & 0x0F) << 8)
    p1 = (b1 >> 4) | (b2 << 4)
    out = np.empty(len(p0) * 2, dtype=np.uint16)
    out[0::2] = p0
    out[1::2] = p1
    return out[:n_pixels]


def pack_frame(frame: np.ndarray) -> bytes:
    """
    Serialise a uint16 frame (12-bit values, [0..4095]) to complete .bin12
    file content (header + packed pixel data).

    Accepts a 2-D mono frame (H, W) or a 3-channel frame (H, W, 3) in BGR
    channel order (cv2 convention); the channels are written to the file in
    R, G, B order.
    """
    if frame.ndim == 2:
        h, w = frame.shape
        channels = 1
        channel_arrays = [frame]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        h, w, _ = frame.shape
        channels = 3
        channel_arrays = [frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]]
    else:
        raise ValueError(f"bin12 supports 2-D mono or H×W×3 frames, got shape {frame.shape}")

    if w % 2 != 0:
        raise ValueError(f"bin12 requires even image width, got {w}")

    def pack_channel(ch: np.ndarray) -> np.ndarray:
        flat = ch.ravel().astype(np.uint16)
        p0, p1 = flat[0::2], flat[1::2]
        b0 = (p0 & 0xFF).astype(np.uint8)
        b1 = ((p0 >> 8) & 0x0F | (p1 & 0x0F) << 4).astype(np.uint8)
        b2 = (p1 >> 4).astype(np.uint8)
        packed = np.empty(len(p0) * 3, dtype=np.uint8)
        packed[0::3] = b0
        packed[1::3] = b1
        packed[2::3] = b2
        return packed.reshape(h, w * 3 // 2)

    packed_rows = [pack_channel(ch) for ch in channel_arrays]
    interleaved = np.stack(packed_rows, axis=1)
    data = interleaved.reshape(-1)

    header = HEADER.pack(MAGIC, w, h, 12, channels)
    return header + data.tobytes()
