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
     *
     * NOTE: the row-chunk size therefore depends on how much memory the machine
     * happens to have free.  That is fine for every per-pixel computation (they
     * are independent), but it must NEVER reach a value that is shared by all
     * pixels — see BAND_SAMPLE_ROWS below for the one place where it did.
     */
    inline constexpr float AVAILABLE_RAM_RATIO_USED = 0.8f;

    /**
     * @brief Size of the FIXED pixel grid that estimates the global bandpass.
     *
     * Methods 2 and 4 derive one pair {k_avg, dk} for the whole image from a
     * sample of Z-traces, and then apply that same band to every pixel.  The
     * sample must not depend on anything outside the dataset, or the result of
     * the analysis stops being reproducible.
     *
     * It used to: the sample was "256 pixels of the first row-chunk, with a
     * stride of (R·Nx)/256", and R comes from auto_row_chunk(), i.e. from
     * MemAvailable at that instant.  A different amount of free RAM gave a
     * different R, a different set of 256 pixels, a different mean amplitude
     * spectrum and — through one bin of dk — a different band for EVERY pixel.
     * Measured on data/S1F1 by faking /proc/meminfo (report C3 §5.2):
     * 99.95 % of the Method 2 pixels changed, std 1.4 nm, max 22.7 nm.
     * Methods 1 and 3 never read {k_avg, dk} and were unaffected.
     *
     * The sample is now a fixed BAND_SAMPLE_ROWS × BAND_SAMPLE_COLS grid over
     * the first rows of the image, spanning its full width.  16 rows is what
     * auto_row_chunk() is guaranteed to deliver (its floor is 16, and it never
     * returns more than Ny), so the grid is identical for every chunk size,
     * every machine and every run.  16 × 16 = 256 keeps the old sample count,
     * and the mean AMPLITUDE spectrum barely depends on which pixels are
     * chosen: a height change only shifts the phase of the transform, not its
     * modulus, so a band of rows is as good a sample as a scattered one.
     *
     * WHAT THIS DOES NOT BUY  (measured, report C3 §S3 — read before trusting
     * dk to a bin)
     * -----------------------------------------------------------------------
     * The fixed grid buys REPRODUCIBILITY, not independence from the sample.
     * dk = ceil(2·sqrt(var)) and, on both real stacks, 2·sqrt(var) sits within
     * ±0.3 bins of an integer, while its spread over different 256-trace
     * samples is ±0.3–0.5 bins.  So no sample of 256 traces pins dk down: on
     * data/S1F1 the grid gives dk = 103 and 64 % of 200 random samples agree,
     * 36 % give 104; on data/S1F5 the grid gives 53 and the MAJORITY (62 %)
     * of random samples give 54.  The grid fixes the answer by convention, and
     * one bin of dk is worth 1.3–1.4 nm of height dispersion in Method 2.
     * There is also a small structural bias: the grid reads the top 16 rows,
     * whose DC is ~4 % lower on S1F1, which puts its 2·sqrt(var) about 2σ
     * below the mean of random samples.  Making Methods 2 and 4 independent of
     * that convention needs a different dk estimator (a coarser grain, or many
     * more traces), not a different sample; that is a design decision, not a
     * bug, and it is not taken here.
     */
    inline constexpr int BAND_SAMPLE_ROWS = 16;
    inline constexpr int BAND_SAMPLE_COLS = 16;

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
     * For the LP126CU + Olympus U-LH100IR (3200 K) system that integral gives
     * 566 nm.  Used ONLY by Method 3 (PSI 5-point) to compute the inter-frame
     * phase step α = 4π δz / λ₀.  If the lamp or objective changes, re-run
     * compute_bayer_weights.py and update this value.
     *
     * The value used here is 570 nm, NOT the 566 nm of the centroid, because
     * what Method 3 needs is the wavelength of the fringe carrier it actually
     * sees, and the carrier is longer than the spectral centroid:
     *
     *   - measured fringe period on a raw superpixel scan: ≈ 0.287 µm, i.e.
     *     λ_eff = 2 · period ≈ 575 nm (the finite NA of the Mirau objective
     *     stretches the period by (1 + cos θ_max)/2 ≈ +2.4 % at NA ≈ 0.3);
     *   - Bayer-weighted spectral centroid of the detected light: 566 nm
     *     (559–569 nm depending on the definition of "centroid").
     *
     * 570 nm sits between the two, within 1 % of either.  It replaces the old
     * 550 nm, which was 3 % below the centroid and ~4 % below the measured
     * carrier (report D1-18, 23-Sep-2026).
     *
     * Sensitivity: α = 4π·20/570 = 0.4410 rad instead of 0.4570 rad.  The
     * kernel of Method 3 only needs α to balance its two quadrature terms, so
     * a few per cent of error leaves a 1–4 % fringe ripple on the envelope
     * that ENVELOPE_SIGMA removes; the reconstructed heights move by far less
     * than the axial step.  Methods 1, 2 and 4 do not read this constant.
     */
    inline constexpr double LAMBDA0_NM = 570.0;

    /**
     * @brief Nominal inter-frame axial step [nm] for Method 3 (PSI 5-point kernel).
     *
     * Used to compute the inter-frame phase step α = 4π δz / λ₀ ≈ 0.441 rad.
     * Must match the step size commanded to the PI P-611.ZS stage.
     * Update here if the acquisition step changes.
     *
     * Note (measured, 24-Sep-2026): on a dataset acquired at 30 nm this value
     * is wrong by 1.5x, yet the reconstructed heights move by < 0.1 nm.  α
     * only sets the relative scale of the two quadrature terms of the kernel,
     * and the residual fringe ripple it leaves is removed by ENVELOPE_SIGMA
     * before the peak is located.  Keeping it in step is still the right
     * thing to do; it is not a source of height error.
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