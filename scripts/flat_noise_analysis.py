"""
flat_noise_analysis.py
Axial noise and surface texture from a CSI height map of a flat reference.

Usage:
    python scripts/flat_noise_analysis.py <height_map.npy> [options]

The height map is assumed to be in micrometres (output of InterferoLab).

Pixel selection strategy
------------------------
Gold flakes (and any other tall features) sit significantly above the flat
background.  A two-step approach is used so that tilt in the raw height map
does not bias the selection window:

  1. Fit a plane to ALL pixels and compute per-pixel residuals.
  2. Mask pixels whose residual exceeds +n_sigma * robust_std above the median
     (one-sided: features are always above background, not below it).
     The lower bound keeps a generous symmetric window to catch any bad pixels.
  3. Re-fit the plane on the masked background pixels only.
  4. Compute all statistics on the final residuals.

This makes the selection insensitive to sample tilt, regardless of magnitude.

Note: this plane fit (2-pass, MAD-based background selection) is intentionally
different from the interactive IQR-based tilt correction used in the GUI
(services/heightmap_processing.py) — do not unify them.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import kurtosis, skew

# ── Plane fitting and background selection ───────────────────────────────────


def fit_plane_residuals(H: np.ndarray, mask: np.ndarray):
    """
    Least-squares plane fit z = a·x + b·y + c restricted to the masked pixels.

    Returns (coeffs, plane) where *plane* is evaluated over the full frame.
    """
    rows, cols = H.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    m = mask.ravel()
    A = np.column_stack([xx.ravel()[m], yy.ravel()[m], np.ones(m.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, H.ravel()[m], rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    return coeffs, plane


def select_background(H: np.ndarray, finite_mask: np.ndarray, n_sigma: float):
    """
    Select background pixels (steps 1-2 of the module docstring strategy).

    A preliminary plane removes tilt so that the rejection threshold is based
    on residuals; features (always above background) get a one-sided clip at
    +n_sigma robust std, bad pixels below get a generous 3-sigma clip.

    Returns (mask, rsd_pre, lo_mask_nm, hi_mask_nm) with thresholds in nm.
    """
    _, plane_pre = fit_plane_residuals(H, finite_mask)
    res_pre = np.where(finite_mask, (H - plane_pre) * 1e3, np.nan)  # nm

    # Robust centre and spread (insensitive to the minority of feature pixels)
    res_pre_valid = res_pre[finite_mask]
    med_pre = float(np.median(res_pre_valid))
    mad_pre = float(np.median(np.abs(res_pre_valid - med_pre)))
    rsd_pre = 1.4826 * mad_pre  # consistent σ estimator for Gaussian data

    hi_mask_nm = med_pre + n_sigma * rsd_pre
    lo_mask_nm = med_pre - 3.0 * rsd_pre
    mask = finite_mask & (res_pre >= lo_mask_nm) & (res_pre <= hi_mask_nm)
    return mask, rsd_pre, lo_mask_nm, hi_mask_nm


# ── ISO 25178 surface texture parameters ─────────────────────────────────────


def compute_iso_params(r_nm: np.ndarray):
    """Return (Sq, Sa, Sz, Ssk, Sku) of the residuals in nm."""
    Sq = np.sqrt(np.mean(r_nm**2))
    Sa = np.mean(np.abs(r_nm))
    Sz = r_nm.max() - r_nm.min()
    Ssk = skew(r_nm)
    Sku = kurtosis(r_nm, fisher=False)
    return Sq, Sa, Sz, Ssk, Sku


# ── 2D Power Spectral Density ────────────────────────────────────────────────


def compute_psd(res_map: np.ndarray, mask: np.ndarray) -> dict:
    """
    Hann-windowed 2D PSD of the residual map (feature pixels zeroed) plus its
    radial average and derived indicators.

    Returns a dict with: f_center, psd_rad_nm2, n_bins, noise_floor,
    sq_parseval_nm, bayer_mean_nm2, bayer_snr.
    """
    rows, cols = res_map.shape

    res_psd = res_map.copy()
    res_psd[~mask] = 0.0

    win2d = np.outer(np.hanning(rows), np.hanning(cols))
    win_corr = np.sqrt(np.mean(win2d[mask] ** 2))
    res_w = res_psd * win2d / win_corr

    F = np.fft.fftshift(np.fft.fft2(res_w))
    PSD2D = np.abs(F) ** 2 / (rows * cols)

    fx = np.fft.fftshift(np.fft.fftfreq(cols))
    fy = np.fft.fftshift(np.fft.fftfreq(rows))
    FX, FY = np.meshgrid(fx, fy)
    FR = np.sqrt(FX**2 + FY**2)

    f_max = 0.5
    n_bins = min(256, rows // 2, cols // 2)
    f_bins = np.linspace(0, f_max, n_bins + 1)
    f_center = 0.5 * (f_bins[:-1] + f_bins[1:])
    df = f_bins[1] - f_bins[0]
    psd_rad = np.zeros(n_bins)

    for i in range(n_bins):
        ring = (f_bins[i] <= FR) & (f_bins[i + 1] > FR)
        if ring.sum() > 0:
            psd_rad[i] = PSD2D[ring].mean()

    psd_rad_nm2 = psd_rad * 1e6

    psd_integral = float(np.sum(psd_rad_nm2 * 2 * np.pi * f_center) * df)
    sq_parseval_nm = np.sqrt(max(psd_integral, 0.0))
    noise_floor = float(np.median(psd_rad_nm2[n_bins // 4 :]))

    bayer_r = 0.05
    bayer_mask_psd = (np.abs(np.abs(FX) - 0.5) < bayer_r) & (np.abs(np.abs(FY) - 0.5) < bayer_r)
    bayer_mean_nm2 = float(PSD2D[bayer_mask_psd].mean() * 1e6) if bayer_mask_psd.any() else 0.0
    bayer_snr = bayer_mean_nm2 / noise_floor if noise_floor > 0 else 0.0

    return {
        "f_center": f_center,
        "psd_rad_nm2": psd_rad_nm2,
        "n_bins": n_bins,
        "noise_floor": noise_floor,
        "sq_parseval_nm": sq_parseval_nm,
        "bayer_mean_nm2": bayer_mean_nm2,
        "bayer_snr": bayer_snr,
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def save_figures(
    out_dir: Path,
    res_map: np.ndarray,
    mask: np.ndarray,
    r_nm: np.ndarray,
    N_flat: int,
    coverage: float,
    Sq: float,
    Sa: float,
    Ssk: float,
    Sku: float,
    psd: dict,
    pixel_size: float | None,
) -> None:
    """Save the residual map, height histogram and radial PSD figures."""
    rows, cols = res_map.shape

    # ── Figure 1: Residual height map ────────────────────────────────────────
    res_map_nm = res_map * 1e3
    vmax = 3.0 * Sq

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    extent = (
        [0, cols, rows, 0] if pixel_size is None else [0, cols * pixel_size, rows * pixel_size, 0]
    )
    im = ax1.imshow(
        res_map_nm,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
        extent=extent,
    )
    cb = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("Height residual (nm)", fontsize=10)

    # Overlay feature pixels (gold flakes) in semi-transparent orange.
    # extent must match the underlying image so the overlay aligns correctly.
    feat_overlay = np.zeros((*res_map_nm.shape, 4), dtype=np.float32)
    feat_overlay[~mask, :3] = [1.0, 0.55, 0.0]  # orange
    feat_overlay[~mask, 3] = 0.55
    ax1.imshow(feat_overlay, origin="upper", extent=extent)

    xlabel = "Column (pixels)" if pixel_size is None else "x (µm)"
    ylabel = "Row (pixels)" if pixel_size is None else "y (µm)"
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.set_title(
        f"Residual height map — background only (tilt removed)\n"
        f"$S_q = {Sq:.2f}$ nm,  $S_{{sk}} = {Ssk:.3f}$,  $S_{{ku}} = {Sku:.3f}$\n"
        f"Orange: excluded features ({100 - coverage:.1f}% of pixels)",
        fontsize=9,
    )
    fig1.tight_layout()
    p1 = out_dir / "noise_map.pdf"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"  Figure saved to   : {p1}")

    # ── Figure 2: Height distribution histogram with Gaussian fit ────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    hist_counts, bin_edges, _ = ax2.hist(
        r_nm,
        bins=120,
        density=True,
        color="steelblue",
        alpha=0.75,
        label="Background pixels",
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(_gauss, bin_centers, hist_counts, p0=[r_nm.mean(), Sq])
        mu_fit, sigma_fit = popt
        x_fit = np.linspace(r_nm.min(), r_nm.max(), 500)
        ax2.plot(
            x_fit,
            _gauss(x_fit, mu_fit, sigma_fit),
            "r-",
            lw=1.8,
            label=f"Gaussian fit  ($\\sigma = {abs(sigma_fit):.2f}$ nm)",
        )
    except (RuntimeError, ValueError):
        pass

    ax2.axvline(Sq, color="k", ls="--", lw=1, alpha=0.6)
    ax2.axvline(-Sq, color="k", ls="--", lw=1, alpha=0.6, label=f"$\\pm S_q = {Sq:.2f}$ nm")

    ax2.set_xlabel("Height residual (nm)", fontsize=10)
    ax2.set_ylabel("Probability density (nm⁻¹)", fontsize=10)
    ax2.set_title(
        f"Height distribution — background pixels ({N_flat:,})\n"
        f"$S_q={Sq:.2f}$ nm,  $S_a={Sa:.2f}$ nm,  "
        f"$S_{{\\!sk}}={Ssk:.3f}$,  $S_{{\\!ku}}={Sku:.3f}$",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    p2 = out_dir / "height_histogram.pdf"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"  Figure saved to   : {p2}")

    # ── Figure 3: Radially averaged 1D PSD (log-log) ─────────────────────────
    f_center = psd["f_center"]
    psd_rad_nm2 = psd["psd_rad_nm2"]
    noise_floor = psd["noise_floor"]

    valid = (f_center > 0) & (psd_rad_nm2 > 0)
    f_plot = f_center[valid]
    psd_plot = psd_rad_nm2[valid]

    if pixel_size is not None:
        f_plot_scaled = f_plot / pixel_size
        xlabel_psd = "Spatial frequency (cycles/µm)"
    else:
        f_plot_scaled = f_plot
        xlabel_psd = "Spatial frequency (cycles/pixel)"

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.loglog(f_plot_scaled, psd_plot, color="steelblue", lw=1.2, label="PSD (radial avg.)")
    ax3.axhline(
        noise_floor, color="r", ls="--", lw=1.2, label=f"Noise floor  {noise_floor:.3f} nm²/(c/px)"
    )

    f_nyq = 0.5 / (pixel_size if pixel_size else 1.0)
    ax3.axvline(f_nyq, color="orange", ls=":", lw=1.2, label="Nyquist / Bayer corner")

    ylabel_psd = (
        "Mean PSD amplitude (nm²/(cycles/µm)²)"
        if pixel_size is not None
        else "Mean PSD amplitude (nm²/(cycles/px)²)"
    )
    ax3.set_xlabel(xlabel_psd, fontsize=10)
    ax3.set_ylabel(ylabel_psd, fontsize=10)
    ax3.set_title("Radially averaged 1D power spectral density", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, which="both", ls=":", alpha=0.4)
    fig3.tight_layout()
    p3 = out_dir / "psd_radial.pdf"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    print(f"  Figure saved to   : {p3}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Axial noise from flat CSI height map")
    parser.add_argument("npy", help="Path to .npy height map (µm)")
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=5.0,
        help="Robust-sigma multiplier for background selection (default: 5). "
        "Pixels whose residual (after preliminary tilt removal) exceeds "
        "+n_sigma * robust_std above the median are treated as features "
        "(gold flakes, dust) and excluded.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in µm (for FOV and PSD spatial-frequency axis).",
    )
    parser.add_argument(
        "--method-label",
        default="",
        help="Label for the LaTeX table row (e.g. 'Method 1')",
    )
    parser.add_argument(
        "--output-dir",
        default="resultados",
        help="Root directory for output files (default: resultados). "
        "Results are saved in <output-dir>/<dataset-name>/.",
    )
    args = parser.parse_args()

    # ── Output directory (named after the dataset) ───────────────────────────
    dataset_name = Path(args.npy).parent.name
    out_dir = Path(args.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all text output for the .txt file
    _lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        _lines.append(text)

    # ── Load ─────────────────────────────────────────────────────────────────
    H = np.load(args.npy).astype(np.float64)  # (rows, cols), µm
    rows, cols = H.shape
    N_total = rows * cols

    log(f"\n{'─' * 60}")
    log(f"  File    : {args.npy}")
    log(f"  Dataset : {dataset_name}")
    log(f"  Shape   : {rows} × {cols}  ({N_total:,} pixels)")
    log(f"{'─' * 60}")

    # Identify finite pixels up front; NaN/Inf arise from failed reconstruction
    # (shadow pixels, scan edges) and must be excluded from all operations.
    finite_mask = np.isfinite(H)
    N_invalid = int((~finite_mask).sum())
    if N_invalid > 0:
        log(
            f"\n  WARNING: {N_invalid:,} NaN/Inf pixels detected and excluded "
            f"({100 * N_invalid / N_total:.2f}%)"
        )
        H = np.where(finite_mask, H, np.nan)  # keep NaN for display, not for math

    # ── Steps 1-2: preliminary plane fit + background selection ──────────────
    mask, rsd_pre, lo_mask_nm, hi_mask_nm = select_background(H, finite_mask, args.n_sigma)

    N_flat = int(mask.sum())
    N_feature = N_total - N_flat
    coverage = 100.0 * N_flat / N_total

    log(f"\n  Background selection  (n_sigma = {args.n_sigma})")
    log(f"    Preliminary robust std (MAD-based) : {rsd_pre:.2f} nm")
    log(f"    Rejection window (residuals)        : [{lo_mask_nm:.1f}, {hi_mask_nm:.1f}] nm")
    log(f"    Background pixels : {N_flat:,} / {N_total:,}  ({coverage:.1f} %)")
    log(f"    Feature pixels    : {N_feature:,}  ({100 - coverage:.1f} %)  → excluded")

    # ── Step 3: re-fit plane on background pixels only ───────────────────────
    coeffs, plane = fit_plane_residuals(H, mask)
    a, b, c = coeffs

    res_map = H - plane  # full-frame residual map, µm
    res_flat = res_map[mask].ravel()  # background residuals, µm

    zf = H[mask].ravel()
    ss_res = np.sum(res_flat**2)
    ss_tot = np.sum((zf - zf.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    tilt_x_nmpx = a * 1e3
    tilt_y_nmpx = b * 1e3

    log("\n  Plane fit  (background pixels only)")
    log(f"    Tilt x / y     : {tilt_x_nmpx:.4f} / {tilt_y_nmpx:.4f} nm/pixel")
    log(f"    Intercept c    : {c * 1e3:.2f} nm")
    log(f"    R²             : {r2:.6f}")

    # ── ISO 25178 surface texture parameters ─────────────────────────────────
    r_nm = res_flat * 1e3  # nm
    Sq, Sa, Sz, Ssk, Sku = compute_iso_params(r_nm)

    log("\n  ISO 25178 surface texture  (background pixels, after tilt removal)")
    log(f"    Sq  (RMS)            : {Sq:.3f} nm")
    log(f"    Sa  (mean |z|)       : {Sa:.3f} nm")
    log(f"    Sz  (peak-to-valley) : {Sz:.2f} nm  (sensitive to residual outliers)")
    log(
        f"    Ssk (skewness)       : {Ssk:.4f}   "
        f"{'OK symmetric' if abs(Ssk) < 0.3 else 'WARN asymmetric'}"
    )
    log(
        f"    Sku (kurtosis)       : {Sku:.4f}   "
        f"{'OK Gaussian' if abs(Sku - 3) < 0.5 else ('WARN heavy tails' if Sku > 3 else 'WARN flat dist.')}"
    )

    # ── 2D Power Spectral Density ────────────────────────────────────────────
    psd = compute_psd(res_map, mask)

    freq_unit = "cycles/pixel"
    if args.pixel_size is not None:
        freq_unit = "cycles/µm"

    log("\n  2D Power Spectral Density  (features zeroed, Hann-windowed)")
    log(f"    Frequency unit         : {freq_unit}")
    log(
        f"    PSD noise floor        : {psd['noise_floor']:.3f} nm²/(cycles/px)  "
        f"[= {np.sqrt(psd['noise_floor']):.3f} nm/sqrt(cycles/px)]"
    )
    log(
        f"    Bayer/CFA corner mean  : {psd['bayer_mean_nm2']:.3f} nm²/(cycles/px)  "
        f"(SNR = {psd['bayer_snr']:.2f}x)  "
        f"{'WARN CFA artefact present' if psd['bayer_snr'] > 2 else 'OK no CFA artefact'}"
    )
    log(
        f"    Parseval check  sqrt(int PSD*2pi*f df) : {psd['sq_parseval_nm']:.3f} nm  "
        f"(expect ~0.88×Sq = {0.886 * Sq:.3f} nm for white noise; "
        f"lower → low-freq dominated)"
    )

    # ── FOV ──────────────────────────────────────────────────────────────────
    if args.pixel_size is not None:
        fov_x = cols * args.pixel_size
        fov_y = rows * args.pixel_size
        log(f"\n  Field of view       : {fov_x:.0f} x {fov_y:.0f} µm")
    else:
        log(
            f"\n  Field of view       : {cols} x {rows} pixels  (pass --pixel-size <um> to convert)"
        )

    # ── LaTeX summary ────────────────────────────────────────────────────────
    label = args.method_label if args.method_label else "Method~?"
    log("\n  LaTeX table row:")
    log(
        f"    Axial noise (RMS, flat reference, {label})   & \\SI{{{Sq:.1f}}}{{\\nano\\metre}} \\\\"
    )

    log("\n  LaTeX texture block:")
    log(f"    $S_q = \\SI{{{Sq:.2f}}}{{\\nano\\metre}}$,\\;")
    log(f"    $S_a = \\SI{{{Sa:.2f}}}{{\\nano\\metre}}$,\\;")
    log(f"    $S_{{\\!z}} = \\SI{{{Sz:.1f}}}{{\\nano\\metre}}$,\\;")
    log(f"    $S_{{\\!sk}} = {Ssk:.3f}$,\\;")
    log(f"    $S_{{\\!ku}} = {Sku:.3f}$.")

    log(f"\n{'─' * 60}\n")

    # ── Save .txt ────────────────────────────────────────────────────────────
    txt_path = out_dir / "analysis_results.txt"
    txt_path.write_text("\n".join(_lines), encoding="utf-8")
    print(f"  Results saved to  : {txt_path}")

    # ── Figures ──────────────────────────────────────────────────────────────
    save_figures(
        out_dir,
        res_map,
        mask,
        r_nm,
        N_flat,
        coverage,
        Sq,
        Sa,
        Ssk,
        Sku,
        psd,
        args.pixel_size,
    )

    print(f"\n  Done.  All outputs in '{out_dir}/'")


if __name__ == "__main__":
    main()
