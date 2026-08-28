"""
Convert 3-channel (RGB demosaiced) bin12 stacks to superpixel bin12 using
2×2 spatial binning with optional channel modes.

Modes
-----
mono (default)
    Weighted single-channel output.  Each 2×2 block of the RGB frame is
    combined with the physically-derived Bayer weights and then averaged,
    producing one value per block.  Output: 1-channel bin12, (H/2)×(W/2).

    Weights (Thorlabs LP126CU + Olympus U-LH100IR, 3200 K blackbody):
        W_R = 0.257090
        W_G = 0.622544   (= w_G1 + w_G2 = 2 × 0.311272)
        W_B = 0.120366
    Source: docs/Quantum efficiency and IR filter/compute_bayer_weights.py

color  (--color flag)
    Per-channel 2×2 average binning.  R, G and B channels are binned
    independently and saved as a 3-channel bin12 file, (H/2)×(W/2).
    Use this when one channel may be corrupted and you need to inspect or
    discard individual channels after acquisition.

Usage
-----
    python process_superpixel.py <input_folder>           # mono
    python process_superpixel.py <input_folder> --color   # color

Output folders
--------------
    <input_folder>_superpixel        (mono mode)
    <input_folder>_superpixel_color  (color mode)

For files that are already mono (channels == 1) the file is copied unchanged.
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

# Allow running this file directly (python scripts/process_superpixel.py ...)
# by making the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.bin12 import HEADER, HEADER_SIZE, MAGIC, pack_frame  # noqa: E402
from utils.camera_constants import GRAY_W_B, GRAY_W_G, GRAY_W_R  # noqa: E402

# Grayscale weights for demosaiced RGB data (single source of truth:
# utils/camera_constants.py; see that module for the derivation).
W_R = GRAY_W_R
W_G = GRAY_W_G  # w_G1 + w_G2
W_B = GRAY_W_B


def _unpack_plane(packed: np.ndarray, w: int) -> np.ndarray:
    """
    Vectorised unpack of a whole 12-bit packed channel plane (O-5).

    packed : uint8 array of shape (h, w*3//2) — one packed row per image row.
    Returns a float32 array of shape (h, w).  Bit-exact with the previous
    per-row utils.bin12.unpack_pixels loop (same bit manipulations, applied
    to all rows at once; uint16 -> float32 conversion is exact for 12-bit).
    """
    b0 = packed[:, 0::3].astype(np.uint16)
    b1 = packed[:, 1::3].astype(np.uint16)
    b2 = packed[:, 2::3].astype(np.uint16)
    out = np.empty((packed.shape[0], w), dtype=np.uint16)
    out[:, 0::2] = b0 | ((b1 & 0x0F) << 8)
    out[:, 1::2] = (b1 >> 4) | (b2 << 4)
    return out.astype(np.float32)


def _bin2x2(frame: np.ndarray) -> np.ndarray:
    """2×2 average binning of a float32 frame. Crops to even dims first."""
    h, w = frame.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    m = frame[:h2, :w2]
    return np.clip(
        np.round((m[0::2, 0::2] + m[1::2, 0::2] + m[0::2, 1::2] + m[1::2, 1::2]) * 0.25),
        0,
        4095,
    ).astype(np.uint16)


def procesar_un_archivo(filepath: str, output_folder: str, color: bool) -> bool:
    """Process one bin12 file: 2×2 binned mono or 2×2 binned per-channel color."""
    try:
        filename = os.path.basename(filepath)
        out_path = os.path.join(output_folder, filename)

        with open(filepath, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            raw_data = f.read()

        magic, w, h, bpp, channels = HEADER.unpack(header_bytes)

        if magic != MAGIC:
            print(f"  [SKIP] {filename}: not valid bin12")
            return False

        ch_row_bytes = w * 3 // 2  # bytes per 12-bit channel row
        row_stride = ch_row_bytes * channels

        unpacked_all = np.frombuffer(raw_data, dtype=np.uint8)

        if channels == 1:
            # Already mono — copy unchanged.
            with open(out_path, "wb") as f:
                f.write(header_bytes)
                f.write(raw_data)
            return True

        if channels != 3:
            print(f"  [SKIP] {filename}: {channels} canales (se esperan 1 o 3)")
            return False

        # Unpack whole channel planes at once (O-5, vectorised: the payload is
        # (h, channels, ch_row_bytes) with channels stored in R, G, B order).
        planes = unpacked_all[: h * row_stride].reshape(h, channels, ch_row_bytes)
        R_frame = _unpack_plane(np.ascontiguousarray(planes[:, 0, :]), w)
        G_frame = _unpack_plane(np.ascontiguousarray(planes[:, 1, :]), w)
        B_frame = _unpack_plane(np.ascontiguousarray(planes[:, 2, :]), w)

        if color:
            # --- Per-channel 2×2 binning → 3-channel output ---
            R_out = _bin2x2(R_frame)
            G_out = _bin2x2(G_frame)
            B_out = _bin2x2(B_frame)

            # pack_frame expects BGR channel order (cv2 convention) and writes
            # the channels to the file in R, G, B order — same bytes as the
            # previous per-row pack_pixels loop.
            bgr = np.stack([B_out, G_out, R_out], axis=2)
            with open(out_path, "wb") as f:
                f.write(pack_frame(bgr))
        else:
            # --- Weighted mono + 2×2 binning → 1-channel output ---
            mono_frame = W_R * R_frame + W_G * G_frame + W_B * B_frame
            binned = _bin2x2(mono_frame)

            with open(out_path, "wb") as f:
                f.write(pack_frame(binned))

        return True

    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")
        return False


def procesar_carpeta(input_folder: str, color: bool) -> None:
    suffix = "_superpixel_color" if color else "_superpixel"
    output_folder = f"{input_folder}{suffix}"
    os.makedirs(output_folder, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_folder, "*.bin12")))
    if not files:
        print(f"No .bin12 files found in '{input_folder}'")
        return

    n = len(files)
    mode_str = (
        "color (R, G, B por separado)" if color else f"mono (W_R={W_R}, W_G={W_G}, W_B={W_B})"
    )
    print(f"Processing {n} files from '{input_folder}'")
    print(f"  Modo: {mode_str}")
    print(f"  Output folder: '{output_folder}'")
    print(f"  Cores: {os.cpu_count()}")

    t0 = time.time()

    func = partial(procesar_un_archivo, output_folder=output_folder, color=color)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(func, files))

    ok = sum(results)
    dt = time.time() - t0

    print("\n--- Summary ---")
    print(f"  Success: {ok}/{n} files")
    print(f"  Time: {dt:.2f} s  ({dt / n:.4f} s/file)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RGB bin12 stacks to superpixel with 2x2 binning."
    )
    parser.add_argument(
        "input_folder",
        help="Input folder with .bin12 files (absolute or relative path).",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help=(
            "Save R, G, B channels separately (3 channels) instead of merging them "
            "into mono. Useful when one channel may be corrupted."
        ),
    )
    args = parser.parse_args()

    input_folder = os.path.normpath(args.input_folder)
    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)

    procesar_carpeta(input_folder, color=args.color)


if __name__ == "__main__":
    # The __main__ guard is required for ProcessPoolExecutor on all platforms.
    main()
