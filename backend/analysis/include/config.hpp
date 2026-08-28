#pragma once
#include <string>

namespace cfg
{
    /**
     * @brief Base directory where subfolders containing image datasets are located.
     *
     * This directory is scanned for TIFF collections used as input to the
     * interferometric reconstruction pipeline.
     *
     * Example: "./data"
     */
    inline const std::string BASE_DIR = "./data";

    /**
     * @brief Output directory for storing results and diagnostic plots.
     *
     * Each dataset generates its own subfolder inside this directory.
     * Example: "./output"
     */
    inline const std::string OUTPUT_FOLDER = "./output";

    /**
     * @brief Gaussian sigma for baseline removal along the Z-axis (samples).
     *
     * Applied BEFORE the Hilbert transform to subtract the slow intensity
     * background (low-frequency envelope of the signal).  Must be larger
     * than the fringe period in samples so the filter only removes DC drift
     * and does not distort the fringe carrier.
     * Rule of thumb: BASELINE_SIGMA > 2 * fringe_period_samples.
     * Fringe period in samples = (lambda_nm / 2) / dz_step_nm.
     * Example: lambda=600 nm, dz=50 nm -> period=6 samples -> sigma >= 12.
     */
    inline constexpr double BASELINE_SIGMA = 15.0;

    /**
     * @brief Gaussian sigma for smoothing the envelope after extraction (samples).
     *
     * Applied AFTER ΔGᵢ² (Method 1) or 2·|IFFT| (Method 2) to suppress
     * shot-noise ripple before the centroid computation.
     * Rule of thumb: ENVELOPE_SIGMA ≥ fringe_period_samples = (λ/2) / dz.
     * For dz=30 nm, λ≈600 nm: period ≈ 10 samples → σ=15 is conservative.
     */
    inline constexpr double ENVELOPE_SIGMA = 15.0;

    /**
     * @brief Centre frequency of the fringe carrier bandpass filter (cycles/sample).
     *
     * Used by HilbertEnvelope::bandpass_inplace(), which is currently NOT
     * called anywhere in the pipeline: this value must be set to match the
     * actual acquisition parameters before wiring the filter in.
     *
     * Formula:  f_center = 2 * dz_step_nm / lambda_nm
     *
     * Known datasets:
     *   dz=20 nm, lambda~600 nm  ->  f_center = 2*20/600 = 0.067
     *   dz=30 nm, lambda~600 nm  ->  f_center = 2*30/600 = 0.100
     *
     * The current value (0.167) is for dz=50 nm / lambda=600 nm and does NOT
     * match the datasets in use.  Update before re-enabling the filter.
     */
    inline constexpr float FRINGE_FREQ_CENTER = 0.167f; // NEEDS CALIBRATION

    /**
     * @brief Full bandwidth of the carrier bandpass filter (cycles/sample).
     *
     * The filter passes the band [f_center - BW/2, f_center + BW/2].
     * Rule: upper edge must be below 2*f_center (second harmonic).
     * For f_center=0.067 (dz=20nm): BW < 0.134, use ~0.05.
     * For f_center=0.100 (dz=30nm): BW < 0.200, use ~0.07.
     */
    inline constexpr float FRINGE_FREQ_BW = 0.12f; // NEEDS CALIBRATION

    /**
     * @brief Fallback row‑chunk size for streaming reconstruction.
     *
     * Only used if auto‑selection based on available RAM is disabled or fails.
     */
    inline constexpr int ROW_CHUNK_SIZE = 32;

    /**
     * @brief Fraction of available RAM allowed to be used for row‑chunk processing.
     *
     * Used by auto_row_chunk() to avoid running out of memory on large datasets.
     * Typical values: 0.5–0.9.
     */
    inline constexpr float AVAILABLE_RAM_RATIO_USED = 0.8f;

    /**
     * @brief RGB‑to‑grayscale weights for colour TIFF images.
     *
     * Derived from the LP126CU QE curves, IR-filter transmission, and a 3200 K
     * blackbody spectrum (Olympus U-LH100IR tungsten-halogen lamp), following
     * the same integral as the Bayer superpixel weights but adapted for
     * demosaiced RGB images: the G channel in a demosaiced frame represents
     * the average of the two green sub-pixels in each 2×2 Bayer block, so its
     * effective weight is w_G1 + w_G2 (double the per-pixel green weight).
     *
     *   W_R = w_R            = 0.257090
     *   W_G = w_G1 + w_G2   = 0.622544  (accounts for 2× green sampling)
     *   W_B = w_B            = 0.120366
     *
     * See docs/Quantum efficiency and IR filter/compute_bayer_weights.py.
     * Standard CIE 601 weights for reference: R=0.299, G=0.587, B=0.114.
     * Adjust if the lamp or sensor changes.
     */
    /**
     * @brief Effective centre wavelength of the detected light [nm].
     *
     * This is the Bayer-weighted spectral centroid of the source as seen by the
     * camera after the IR-cut filter:
     *
     *   λ₀ ≈ Σ_c w_c ⟨λ⟩_c  (Bayer weights × per-channel mean wavelength)
     *
     * For the LP126CU + Olympus U-LH100IR (3200 K) system this evaluates to
     * approximately 550 nm.  Used ONLY by Method 3 (PSI 5-point) to compute
     * the inter-frame phase step α = 4π δz / λ₀.  If the lamp or objective
     * changes, re-run compute_bayer_weights.py and update this value.
     */
    inline constexpr double LAMBDA0_NM = 550.0;

    /**
     * @brief Nominal inter-frame axial step [nm] for Method 3 (PSI 5-point kernel).
     *
     * Used to compute the inter-frame phase step α = 4π δz / λ₀ ≈ 0.457 rad.
     * Must match the step size commanded to the PI P-611.ZS stage.
     * Update here if the acquisition step changes.
     */
    inline constexpr double NOMINAL_DZ_NM = 20.0;

    inline constexpr double GRAY_WEIGHT_R = 0.257090;
    inline constexpr double GRAY_WEIGHT_G = 0.622544;
    inline constexpr double GRAY_WEIGHT_B = 0.120366;

    /**
     * @brief Regular expression for extracting piezo positions from filenames.
     *
     * Matches a floating‑point number followed by optional units such as:
     *   - "um"
     *   - "µm"
     *   - "μm"
     *
     * Example:
     *   Filename: "piezo_26.7400um_0087.tiff"
     *   Extracted value: 26.7400
     *
     * @note The pattern is deliberately permissive and accepts TIFF/PNG.
     */
    inline const char *REGEX_PATTERN =
        R"(.*?([+-]?\d+(?:\.\d+)?)\s*(?:[µμ]?m|um).*\.(?:tiff?|png|bin12)$)";
}