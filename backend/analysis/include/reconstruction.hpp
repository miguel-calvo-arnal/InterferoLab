#pragma once
#include "image_stack.hpp"
#include "config.hpp"
#include <array>
#include <functional>
#include <regex>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief Reconstruct a height map from a 3‑D interferometric image stack.
 *
 * This function implements the main step of the CSI / white‑light
 * interferometry reconstruction pipeline.
 *
 * Expected input:
 *  - A 3‑D stack with dimensions (Nz, Ny, Nx), where Nz is the number of
 *    acquired intensity images at different piezo displacements.
 *  - A regular expression pattern used to extract physical Z‑positions
 *    from TIFF filenames (the actual reading occurs inside the .cpp).
 *  - An ImageStack structure for returning metadata (Nx, Ny, Nz).
 *
 * Reconstruction workflow:
 *   1. For each pixel (y, x), extract the Z‑trace:
 *          stack(z, y, x)  for z = 0 .. Nz‑1.
 *   2. Apply baseline removal + Hilbert transform + smoothing to obtain
 *      the analytic signal envelope.
 *   3. Locate the envelope peak (the coherence maximum).
 *   4. Convert the peak index to a physical coordinate using
 *      the stored Z‑positions.
 *   5. Write the resulting heights into a cv::Mat (Ny × Nx, CV_32F).
 *
 * @param files            List of TIFF file paths, already sorted.
 * @param pattern          Regex used to extract numerical Z‑positions.
 * @param stack_info       Output metadata (Nx, Ny, Nz and Z‑ordering).
 * @param progress_callback Optional progress callback, taking a float ∈ [0,1].
 *
 * @return cv::Mat (Ny × Nx, CV_32F) height map in physical units (e.g. microns).
 *
 * @warning This computation is expensive; the implementation in
 *          reconstruction.cpp uses streaming row‑chunks and OpenMP.
 *
 * @note It is assumed that the number of planes Nz matches the number of
 *       extracted Z‑positions; the implementation should validate or throw.
 */
/// out_k_avg and out_dk receive the globally-estimated FFT bandpass parameters
/// for methods with needs_global_params (currently 2 and 4). They ensure that
/// the pixel-plot helpers apply the same filter as the main reconstruction
/// (not a per-pixel one). Both are set to -1 for the other methods or when
/// estimation fails.
cv::Mat reconstruct_height_map_rowchunks(
    const std::vector<std::string> &files,
    const std::regex &pattern,
    ImageStack &stack_info,
    std::function<void(float)> progress_callback, /*=nullptr*/
    const std::array<double, 3> &channel_weights = {cfg::GRAY_WEIGHT_R,
                                                     cfg::GRAY_WEIGHT_G,
                                                     cfg::GRAY_WEIGHT_B},
    int method = 1,
    int *out_k_avg = nullptr,
    int *out_dk    = nullptr); ///< method ID — see methods.hpp for the full list

/**
 * @brief Save a diagnostic per‑pixel Z‑trace and its reconstruction using
 *        streaming access to the TIFF stack.
 *
 * This helper extracts the full Z‑signal at pixel (py, px), applies the
 * same Hilbert‑based processing as the main reconstruction, and writes
 * diagnostic plots or TIFFs (implementation‑defined) for inspection.
 *
 * @param files      List of TIFF filenames.
 * @param pattern    Regex used for Z identification.
 * @param stack_info Metadata containing Nx, Ny, Nz and Z‑ordering.
 * @param py         Row index of the pixel to save.
 * @param px         Column index of the pixel to save.
 * @param base_name  Output basename for diagnostic files.
 */
/// k_avg / dk: when method == 2 and both are >= 0, the fixed-params FFT bandpass
/// overload is used so the pixel plot envelope is identical to the one computed
/// during the main reconstruction. Pass -1/-1 (defaults) to fall back to the
/// per-pixel estimate (only acceptable for standalone diagnostic calls).
void save_pixel_plot_streaming_libtiff(
    const std::vector<std::string> &files,
    const std::regex &pattern,
    const ImageStack &stack_info,
    int py, int px,
    const std::string &base_name,
    const std::array<double, 3> &channel_weights = {cfg::GRAY_WEIGHT_R,
                                                     cfg::GRAY_WEIGHT_G,
                                                     cfg::GRAY_WEIGHT_B},
    int method = 1,
    int k_avg  = -1,
    int dk     = -1); ///< method ID — see methods.hpp for the full list

/**
 * @brief Save multiple pixel plots in a SINGLE pass through the Z-stack.
 *
 * Dramatically faster than calling save_pixel_plot_streaming_libtiff() N times:
 *  - For PNG: reduces cv::imread calls from N×Nz to Nz (N× speedup).
 *  - For TIFF/bin12: reduces file opens from N×Nz to Nz.
 *  - Envelope processing runs in parallel (OpenMP).
 *
 * @param pixels   List of (row, col) = (py, px) pairs to save.
 * @param files    Sorted list of image file paths.
 * @param pattern  Regex for extracting Z-positions from filenames.
 * @param stack_info  Image stack metadata (Nx, Ny, Nz, z_order).
 * @param base_name   Output basename used for file naming.
 * @param out_dir     Optional output directory override.  Empty (default) =
 *                    cfg::OUTPUT_FOLDER/base_name.  Used by the analysis API
 *                    to stage results in a temporary folder that is renamed
 *                    atomically once everything has been written.
 */
void save_pixel_plots_batch_streaming(
    const std::vector<std::pair<int,int>> &pixels,
    const std::vector<std::string> &files,
    const std::regex &pattern,
    const ImageStack &stack_info,
    const std::string &base_name,
    const std::array<double, 3> &channel_weights = {cfg::GRAY_WEIGHT_R,
                                                     cfg::GRAY_WEIGHT_G,
                                                     cfg::GRAY_WEIGHT_B},
    int method = 1,
    int k_avg  = -1,
    int dk     = -1,
    const std::string &out_dir = "");