"""
Convert all TIFF images in a dataset folder to the packed 3-channel 12-bit
binary format (.bin12) used by InterferoLab.

The .bin12 layout (header + row-interleaved planar 12-bit packing) is defined
in a single place: utils/bin12.py (`pack_frame`). See that module for the
format specification; the C++ reference decoder lives in
backend/analysis/src/reconstruction.cpp.

Usage
-----
    python scripts/convert_to_bin12.py <input_folder> [output_folder] [--workers N]

Defaults: output = <input_folder>_bin12,  workers = all CPU cores.

Notes
-----
- TIFFs are read with cv2 (BGR channel order); `pack_frame` expects BGR and
  writes the channels to disk in R, G, B order.
- Grayscale TIFFs are replicated to three identical channels so the output is
  always a 3-channel .bin12.
- Pixel values are clipped to the 12-bit range [0, 4095]. Width must be even.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Allow running this file directly (python scripts/convert_to_bin12.py ...)
# by making the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.bin12 import pack_frame  # noqa: E402


def convert_one(src: str, dst: str) -> tuple[str, bool]:
    """Read one TIFF, pack as 3-channel bin12, write to dst.

    Must be module-level for ProcessPoolExecutor pickling.
    Returns (basename, success).
    """
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return os.path.basename(src), False

    # Drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    frame = np.clip(img, 0, 4095).astype(np.uint16)
    if frame.ndim == 2:
        # Replicate grayscale to three identical channels (BGR == RGB here).
        frame = np.stack([frame, frame, frame], axis=2)

    try:
        payload = pack_frame(frame)  # expects BGR (cv2 convention)
    except ValueError:
        return os.path.basename(src), False  # odd width or unsupported shape

    with open(dst, "wb") as f:
        f.write(payload)

    return os.path.basename(src), True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TIFF dataset to 3-channel bin12.")
    parser.add_argument("input_folder")
    parser.add_argument("output_folder", nargs="?")
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=os.cpu_count(),
        help="Parallel workers (default: all CPU cores)",
    )
    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/\\")
    output_folder = args.output_folder or input_folder + "_bin12"
    n_workers = max(1, args.workers)

    files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith((".tiff", ".tif")))
    if not files:
        print(f"No TIFF files found in {input_folder}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)
    n = len(files)
    print(f"Found {n} TIFF files  →  {output_folder}  (workers: {n_workers})")

    tasks = [
        (
            os.path.join(input_folder, f),
            os.path.join(output_folder, os.path.splitext(f)[0] + ".bin12"),
        )
        for f in files
    ]

    done = 0
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(convert_one, src, dst): src for src, dst in tasks}
        for future in as_completed(futures):
            fname, ok = future.result()
            if not ok:
                errors.append(fname)
                print(f"  [WARN] Could not convert {fname}")
            done += 1
            if done % 200 == 0 or done == n:
                print(f"  [{done}/{n}]")

    print(f"\nDone. {n - len(errors)}/{n} files converted.  Output: {output_folder}")
    if errors:
        print(f"  {len(errors)} failures: {errors[:5]}{'...' if len(errors) > 5 else ''}")


if __name__ == "__main__":
    # The __main__ guard is required for ProcessPoolExecutor on all platforms.
    main()
