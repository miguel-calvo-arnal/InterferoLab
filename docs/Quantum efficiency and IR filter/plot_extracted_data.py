#!/usr/bin/env python3
"""
Load the CSV files produced by extract_plot_data.py and generate verification
plots side-by-side with the original PNG images.

Output: verification_plots.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load(filename: str, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(BASE_DIR, filename))
    return df[x_col].to_numpy(), df[y_col].to_numpy()


# ─── load data ────────────────────────────────────────────────────────────────

wl_ir, tr_ir = load("ir_filter_transmission.csv", "wavelength_nm", "transmission_pct")
wl_b,  qe_b  = load("qe_blue.csv",                "wavelength_nm", "qe_pct")
wl_r,  qe_r  = load("qe_red.csv",                 "wavelength_nm", "qe_pct")
wl_g,  qe_g  = load("qe_green.csv",               "wavelength_nm", "qe_pct")

# ─── load original images ─────────────────────────────────────────────────────

img_ir = mpimg.imread(os.path.join(BASE_DIR, "IR filter transmission.png"))
img_qe = mpimg.imread(os.path.join(BASE_DIR, "quantum efficiency.png"))

# ─── figure layout: 2 rows × 2 cols ──────────────────────────────────────────
#   row 0: original images
#   row 1: reconstructed plots

fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

# ── top-left: original IR filter ──────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(img_ir)
ax0.set_title("Original — LP126CU(/M) IR Filter Transmission", fontsize=11)
ax0.axis("off")

# ── top-right: original QE ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(img_qe)
ax1.set_title("Original — LP126CU(/M) Quantum Efficiency", fontsize=11)
ax1.axis("off")

# ── bottom-left: reconstructed IR filter ──────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(wl_ir, tr_ir, color="royalblue", linewidth=1.5, label="Transmission")
ax2.set_xlim(400, 1200)
ax2.set_ylim(0, 100)
ax2.set_xlabel("Wavelength (nm)", fontsize=11)
ax2.set_ylabel("Transmission (%)", fontsize=11)
ax2.set_title("Reconstructed — IR Filter Transmission", fontsize=11)
ax2.grid(True, linewidth=0.4, color="lightgray")
ax2.legend(framealpha=0.8, fontsize=9)
ax2.tick_params(labelsize=9)

# ── bottom-right: reconstructed QE ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(wl_b, qe_b, color="royalblue",  linewidth=1.5, label="Blue")
ax3.plot(wl_r, qe_r, color="crimson",    linewidth=1.5, label="Red")
ax3.plot(wl_g, qe_g, color="forestgreen", linewidth=1.5, label="Green")
ax3.set_xlim(400, 1000)
ax3.set_ylim(0, 80)
ax3.set_xlabel("Wavelength (nm)", fontsize=11)
ax3.set_ylabel("Quantum Efficiency (%)", fontsize=11)
ax3.set_title("Reconstructed — Quantum Efficiency", fontsize=11)
ax3.grid(True, linewidth=0.4, color="lightgray")
ax3.legend(framealpha=0.8, fontsize=9)
ax3.tick_params(labelsize=9)

# ─── save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "verification_plots.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

plt.show()
