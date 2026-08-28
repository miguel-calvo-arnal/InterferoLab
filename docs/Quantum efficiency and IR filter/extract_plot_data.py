#!/usr/bin/env python3
"""
Extract curve data from LP126CU(/M) plot images and save to CSV.

Outputs
-------
ir_filter_transmission.csv  — wavelength_nm, transmission_pct  (400–1200 nm)
qe_blue.csv                 — wavelength_nm, qe_pct            (400–1000 nm)
qe_red.csv                  — wavelength_nm, qe_pct            (400–1000 nm)
qe_green.csv                — wavelength_nm, qe_pct            (400–1000 nm)

Method
------
1. Auto-detect plot bounding box (dark border lines).
2. Colour-dominance score:  score = channel − mean(other two channels).
   More robust than HSV saturation: works on the light-blue shaded NIR
   background where HSV saturation drops, and naturally rejects a white or
   near-neutral background (score ≈ 0).
3. One point per x-column: argmax of score above a threshold.
4. Rolling-median filter (kernel 41) removes legend line artefacts.
5. Linear extrapolation extends detected data to the full axis limits;
   values are clipped to [0, axis_max].
"""

import os
import numpy as np
from PIL import Image
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_rgb(filename: str) -> np.ndarray:
    img = Image.open(os.path.join(BASE_DIR, filename)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def find_plot_box(img: np.ndarray, dark_thresh: int = 60) -> tuple[int, int, int, int]:
    """Return (x_left, x_right, y_top, y_bottom) of the plot area."""
    h, w = img.shape[:2]
    gray = img.mean(axis=2)
    dark = gray < dark_thresh

    min_run_w = int(0.35 * w)
    min_run_h = int(0.35 * h)

    def max_run(arr: np.ndarray) -> int:
        best = cur = 0
        for v in arr:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    border_rows = [r for r in range(h) if max_run(dark[r, :]) >= min_run_w]
    border_cols = [c for c in range(w) if max_run(dark[:, c]) >= min_run_h]

    if not border_rows or not border_cols:
        raise RuntimeError("Could not detect plot border lines.")

    return border_cols[0], border_cols[-1], border_rows[0], border_rows[-1]


def pixels_to_data(xs: np.ndarray, ys: np.ndarray,
                   x_range: tuple[float, float], y_range: tuple[float, float],
                   box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x_left, x_right, y_top, y_bottom = box
    x_data = x_range[0] + (xs - x_left) / (x_right - x_left) * (x_range[1] - x_range[0])
    y_data = y_range[1] - (ys - y_top)  / (y_bottom - y_top)  * (y_range[1] - y_range[0])
    return x_data, y_data


def extract_curve(img_rgb: np.ndarray,
                  channel_idx: int,              # 0=R, 1=G, 2=B
                  box: tuple[int, int, int, int],
                  x_range: tuple[float, float],
                  y_range: tuple[float, float],
                  threshold: float = 28.0,
                  outlier_thresh: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    """
    For each x-pixel column in the plot box, record the y-pixel with the
    highest colour-dominance score (channel − mean of others) above threshold.

    Then:
      • apply rolling-median outlier removal (removes legend artefacts),
      • linearly interpolate/extrapolate to the full axis x_range,
      • clip y to [0, y_range[1]].

    Returns (x_data, y_data) at 1-unit x spacing over the full x_range.
    """
    x_left, x_right, y_top, y_bottom = box

    # Colour-dominance score over the whole image
    img = img_rgb.astype(np.float32)
    others = [i for i in range(3) if i != channel_idx]
    score = img[:, :, channel_idx] - (img[:, :, others[0]] + img[:, :, others[1]]) / 2.0

    # Crop to plot area
    plot_score = score[y_top:y_bottom + 1, x_left:x_right + 1]
    h_p, w_p = plot_score.shape

    col_xs, col_ys = [], []
    for col in range(w_p):
        col_sc = plot_score[:, col]
        max_s = col_sc.max()
        if max_s >= threshold:
            row = int(col_sc.argmax())
            col_xs.append(col + x_left)
            col_ys.append(row + y_top)

    if len(col_xs) < 5:
        return np.array([]), np.array([])

    xs_px = np.array(col_xs)
    ys_px = np.array(col_ys)

    x_raw, y_raw = pixels_to_data(xs_px, ys_px, x_range, y_range, box)
    y_raw = np.clip(y_raw, 0, y_range[1])

    # ── legend artefact removal ───────────────────────────────────────────────
    # The Thorlabs QE legend box spans ~766–811 nm at QE ≈ 64–75 %.
    # Real QE curves never exceed 63 % for λ > 640 nm, so a hard cut removes
    # the artefact cleanly without touching curve data.
    # A rolling-median pass (kernel 21) catches any remaining smaller spikes.
    if y_range[1] == 80:      # QE plot only — skip for IR filter (y_max=100)
        nir_mask = x_raw > 640
        legend_spike = nir_mask & (y_raw > 63)
        keep_legend = ~legend_spike
    else:
        keep_legend = np.ones(len(y_raw), dtype=bool)

    x_filt = x_raw[keep_legend]
    y_filt = y_raw[keep_legend]

    y_med = medfilt(y_filt, kernel_size=21)
    keep = np.abs(y_filt - y_med) <= outlier_thresh
    x_clean = x_filt[keep]
    y_clean = y_filt[keep]

    if len(x_clean) < 3:
        return np.array([]), np.array([])

    # ── extend to axis boundaries via linear extrapolation ────────────────────
    x_out = np.arange(x_range[0], x_range[1] + 1, 1.0)

    # Interpolate inside detected range
    f_interp = interp1d(x_clean, y_clean, kind='linear',
                        bounds_error=False, fill_value=np.nan)
    y_out = f_interp(x_out)

    # Left boundary extrapolation
    if x_clean[0] > x_range[0]:
        n_fit = min(10, len(x_clean))
        slope, intercept = np.polyfit(x_clean[:n_fit], y_clean[:n_fit], 1)
        x_left_gap = x_out[x_out < x_clean[0]]
        y_extrap = np.polyval([slope, intercept], x_left_gap)
        y_out[x_out < x_clean[0]] = y_extrap

    # Right boundary extrapolation
    if x_clean[-1] < x_range[1]:
        n_fit = min(10, len(x_clean))
        slope, intercept = np.polyfit(x_clean[-n_fit:], y_clean[-n_fit:], 1)
        x_right_gap = x_out[x_out > x_clean[-1]]
        y_extrap = np.polyval([slope, intercept], x_right_gap)
        y_out[x_out > x_clean[-1]] = y_extrap

    y_out = np.clip(y_out, 0, y_range[1])

    valid = ~np.isnan(y_out)
    return x_out[valid], y_out[valid]


def save_csv(x: np.ndarray, y: np.ndarray,
             col_x: str, col_y: str, filename: str) -> None:
    path = os.path.join(BASE_DIR, filename)
    pd.DataFrame({col_x: np.round(x, 2), col_y: np.round(y, 4)}).to_csv(path, index=False)
    print(f"  → saved {len(x):4d} pts  [{x[0]:.0f}–{x[-1]:.0f} nm]  ({filename})")


# ─── IR Filter Transmission (single blue curve) ───────────────────────────────

print("=== IR Filter Transmission ===")
# NOTE: the two input figures ("IR filter transmission.png" and
# "quantum efficiency.png") are Thorlabs datasheet plots (copyrighted) and are
# NOT distributed in the repository — download the LP126CU datasheet from
# thorlabs.com and export them if you need to re-run the digitization. The
# extracted data lives in the CSV files next to this script.
img_ir = load_rgb("IR filter transmission.png")
box_ir = find_plot_box(img_ir)
print(f"  plot box: x=[{box_ir[0]}, {box_ir[1]}]  y=[{box_ir[2]}, {box_ir[3]}]")

x_ir, y_ir = extract_curve(img_ir, channel_idx=2, box=box_ir,
                            x_range=(400, 1200), y_range=(0, 100),
                            threshold=28, outlier_thresh=18)
save_csv(x_ir, y_ir, "wavelength_nm", "transmission_pct", "ir_filter_transmission.csv")


# ─── Quantum Efficiency (Blue, Red, Green) ────────────────────────────────────

print("\n=== Quantum Efficiency ===")
img_qe = load_rgb("quantum efficiency.png")
box_qe = find_plot_box(img_qe)
print(f"  plot box: x=[{box_qe[0]}, {box_qe[1]}]  y=[{box_qe[2]}, {box_qe[3]}]")

# Blue channel.  The NIR background is a light-blue tint (score ≈ 25),
# so threshold=32 keeps the background out while the curve (score ~100+)
# is always detected.
x_b, y_b = extract_curve(img_qe, channel_idx=2, box=box_qe,
                          x_range=(400, 1000), y_range=(0, 80),
                          threshold=32, outlier_thresh=18)
save_csv(x_b, y_b, "wavelength_nm", "qe_pct", "qe_blue.csv")

# Red channel.  White/light-blue background → score ≤ 0, threshold=25 is safe.
x_r, y_r = extract_curve(img_qe, channel_idx=0, box=box_qe,
                          x_range=(400, 1000), y_range=(0, 80),
                          threshold=25, outlier_thresh=18)
save_csv(x_r, y_r, "wavelength_nm", "qe_pct", "qe_red.csv")

# Green channel.  Same background — threshold=25 sufficient.
x_g, y_g = extract_curve(img_qe, channel_idx=1, box=box_qe,
                          x_range=(400, 1000), y_range=(0, 80),
                          threshold=25, outlier_thresh=18)
save_csv(x_g, y_g, "wavelength_nm", "qe_pct", "qe_green.csv")

print("\nDone.")
