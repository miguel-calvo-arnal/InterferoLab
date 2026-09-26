#pragma once

#include <array>
#include <functional>
#include <string>
#include <atomic>
#include "config.hpp"

/**
 * @brief Global cancellation flag shared across the analysis pipeline.
 *
 * This flag is set by the Python/C++ front-end to request cooperative
 * cancellation of the current reconstruction.
 */
extern std::atomic<bool> g_cancel_requested;

/**
 * @brief Set the global cancellation flag.
 *
 * The analysis pipeline checks this flag periodically during
 * long-running operations (TIFF reading, Hilbert computation, etc.).
 *
 * @param value  True to request cancellation, false to reset the flag.
 */
void request_cancel(bool value);

/**
 * @brief Query whether cancellation has been requested.
 *
 * @return True if cancellation is pending.
 */
bool cancel_requested();

// ---------------------------------------------------------------------------
// Callback type aliases
// ---------------------------------------------------------------------------

/**
 * @brief Progress callback for GUI or console visualisation.
 *
 * Typical usage:
 *  - percent ∈ [0,100]
 *  - stage: textual description of the current phase ("load", "reconstruct", "save")
 *  - done/total: work units (e.g. rows processed, chunks completed, etc.)
 */
using ProgressCallback = std::function<void(int percent,
                                            const std::string &stage,
                                            int done,
                                            int total)>;

/**
 * @brief Log callback for textual messages.
 *
 * @param level   For example "info", "warn", "error".
 * @param message The message string to be displayed, printed or stored.
 */
using LogCallback = std::function<void(const std::string &level,
                                       const std::string &message)>;

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

/**
 * @brief Output paths produced by the reconstruction.
 *
 * Contains the locations of the generated height map and any auxiliary files.
 * More fields (pulses/envelopes/metadata) may be added in the future.
 */
struct AnalysisResultPaths
{
    /// Output folder, e.g. "output/2026-03-13_11-21-39"
    std::string output_folder;

    /// Height map stored as .npy, e.g. "output/.../_height.npy"
    std::string heightmap_npy;

    /// Global bandpass actually used by Methods 2 and 4 (-1 for Methods 1 and
    /// 3, which do not have one).  These two integers are shared by every
    /// pixel of the image, so they are the single value of the reconstruction
    /// that is not computed per pixel; reporting them lets a caller check that
    /// two runs of the same dataset really did the same thing.  See
    /// cfg::BAND_SAMPLE_ROWS for why that used to depend on the free RAM.
    int k_avg = -1;
    int dk    = -1;
};

// ---------------------------------------------------------------------------
// API function
// ---------------------------------------------------------------------------

/**
 * @brief Run the entire CSI/white-light reconstruction pipeline.
 *
 * This function:
 *  - reads all images (.tif/.tiff/.bin12/.png) in the selected dataset,
 *  - extracts Z-positions using a regular expression,
 *  - loads metadata into the ImageStack structure (Nx, Ny, Nz),
 *  - performs streaming reconstruction with baseline removal,
 *  - computes the Hilbert envelope to locate the coherence peak,
 *  - and emits progress/log messages through the provided callbacks.
 *
 * The function is designed to support large datasets by processing
 * the TIFF stack row-chunk-by-row-chunk, avoiding excessive RAM usage.
 *
 * @param folder        Directory containing the input TIFF files.
 * @param name          Experiment base-name; used for naming outputs.
 * @param progress_cb   Optional progress callback (may be nullptr).
 * @param log_cb        Optional logging callback (may be nullptr).
 *
 * @return A structure containing the output folder and height-map paths.
 */
/**
 * @brief Reconstruction method selector.
 *
 *  1 — Squared central-difference centroid: pseudo-envelope (ΔGᵢ)² after
 *      Gaussian baseline removal; height from weighted centroid.
 *
 *  2 — FFT Fourier-filter centroid: one-sided bandpass (rectangular window
 *      centred on k_avg ± Δk); envelope 2×|IFFT|; height from centroid.
 *
 *  3 — PSI 5-point kernel (Schwider–Hariharan): DC-insensitive algebraic
 *      envelope from 5 consecutive samples; height from centroid.
 *
 *  4 — Frequency-domain linear fit (group delay): weighted linear regression
 *      of arg[F(k)] vs k in the carrier band; height from slope.
 *
 *  See methods.hpp for the authoritative registry.
 */
/// pixel_plot_grid: 0 = save only the median pixel plot;
///                  N > 0 = save an N×N equispaced grid of pixel plots + median.
///
/// Callbacks are taken by const& on purpose: passing std::function by value
/// would copy the wrapped py::function (incref/decref) inside a section that
/// runs WITHOUT the GIL — a violation of the CPython C-API (B-5).
AnalysisResultPaths run_analysis_for_folder(
    const std::string &folder,
    const std::string &name,
    const ProgressCallback &progress_cb = nullptr,
    const LogCallback &log_cb = nullptr,
    const std::array<double, 3> &channel_weights = {cfg::GRAY_WEIGHT_R,
                                                     cfg::GRAY_WEIGHT_G,
                                                     cfg::GRAY_WEIGHT_B},
    int method = 1,
    int pixel_plot_grid = 0);