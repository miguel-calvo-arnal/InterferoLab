# services/image_formats.py
"""
Lightweight header-only image format detectors (no full image decode needed).

Used by the processing UI to decide whether a dataset folder contains mono or
colour frames by inspecting only the first few bytes of the first image.
"""

from __future__ import annotations

import os
import struct

from utils import bin12

#: Image file extensions recognised as dataset frames.
DATASET_EXTENSIONS = (".bin12", ".tiff", ".tif", ".png")


def bin12_n_channels(path: str) -> int:
    """Return channel count from a .bin12 header, or -1 on failure."""
    header = bin12.read_header(path)
    return header.channels if header is not None else -1


def tiff_n_channels(path: str) -> int:
    """
    Return the SamplesPerPixel tag from a TIFF header, or -1 on failure.
    Reads only the IFD entries — never decodes pixel data.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                return -1
            if raw[:2] == b"II":
                e = "<"
            elif raw[:2] == b"MM":
                e = ">"
            else:
                return -1
            magic = struct.unpack_from(f"{e}H", raw, 2)[0]
            if magic != 42:  # only classic TIFF
                return -1
            ifd_off = struct.unpack_from(f"{e}I", raw, 4)[0]
            fh.seek(ifd_off)
            n_entries = struct.unpack(f"{e}H", fh.read(2))[0]
            for _ in range(min(n_entries, 64)):
                entry = fh.read(12)
                if len(entry) < 12:
                    break
                tag = struct.unpack_from(f"{e}H", entry, 0)[0]
                if tag == 277:  # SamplesPerPixel
                    return struct.unpack_from(f"{e}I", entry, 8)[0]
                if tag > 277:
                    break
        return 1  # tag not found → assume grayscale
    except Exception:
        return -1


def png_n_channels(path: str) -> int:
    """
    Return channel count from a PNG IHDR chunk, or -1 on failure.
    Only reads 26 bytes.
    Color types: 0/4 → mono (1), 2/3/6 → color (3).
    """
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(26)
        if len(hdr) < 26 or hdr[:8] != b"\x89PNG\r\n\x1a\n":
            return -1
        color_type = hdr[25]
        return 1 if color_type in (0, 4) else 3
    except Exception:
        return -1


def detect_dataset_format(folder: str) -> str:
    """
    Inspect the first image in *folder* to decide if it is 'mono' or 'color'.
    Returns 'mono', 'color', or 'unknown' (folder empty / unreadable).
    Reads at most a few dozen bytes — never loads pixel data.
    """
    try:
        files = sorted(
            f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in DATASET_EXTENSIONS
        )
        if not files:
            return "unknown"

        first = os.path.join(folder, files[0])
        ext = os.path.splitext(first)[1].lower()

        if ext == ".bin12":
            n = bin12_n_channels(first)
        elif ext in (".tiff", ".tif"):
            n = tiff_n_channels(first)
        elif ext == ".png":
            n = png_n_channels(first)
        else:
            return "unknown"

        if n < 0:
            return "unknown"
        return "mono" if n == 1 else "color"

    except Exception:
        return "unknown"
