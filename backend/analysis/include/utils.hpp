#pragma once
#include "image_stack.hpp"
#include "config.hpp"
#include <opencv2/core.hpp>
#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <regex>
#include <cstdint>

/**
 * @brief Detect the amount of free RAM available on the system (cross‑platform).
 *
 * Returns the number of bytes of memory that the operating system reports as
 * available. Implementations exist for Linux (/proc/meminfo), Windows
 * (GlobalMemoryStatusEx), and macOS (sysctl).
 *
 * @return Free RAM in bytes.
 */
std::uint64_t get_free_ram_bytes();

/**
 * @brief Automatically choose an optimal row‑chunk size based on free RAM and
 *        the dimensions of the dataset.
 *
 * The function estimates the memory cost of a row‑chunk of shape (R × Nx × Nz)
 * and selects the maximum value of R that fits within a user‑defined fraction
 * of available RAM.
 *
 * @param Nx           Number of columns in each TIFF image.
 * @param Ny           Number of rows in each TIFF image.
 * @param Nz           Number of slices (TIFF files).
 * @param safety_ratio Fraction of available RAM allowed to be used (0–1).
 *
 * @return Optimal row‑chunk size R.
 */
int auto_row_chunk(int Nx, int Ny, int Nz, double safety_ratio = cfg::AVAILABLE_RAM_RATIO_USED);

/**
 * @brief Wait for ENTER only on Windows consoles.
 *
 * Useful for keeping the window open when compiled as a standalone exe.
 *
 * @note No effect on non‑Windows platforms.
 */
inline void wait_for_enter_windows_only()
{
#ifdef _WIN32
    std::cout << "Press ENTER to exit...";
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
#endif
}

/**
 * @brief Display the subdirectories of a base directory and allow the user
 *        to select one from the console.
 *
 * @param base_dir Root directory (e.g. cfg::BASE_DIR).
 * @return A pair { full_path_to_selected_folder, folder_name }.
 *
 * @throws std::runtime_error If the directory does not exist, contains no
 *         subdirectories, or invalid user input is provided.
 *
 * @note Used for selecting the input TIFF dataset.
 */
std::pair<std::string, std::string> select_folder(const std::string &base_dir);

/**
 * @brief List all image files (.tif/.tiff/.bin12/.png, case‑insensitive) in a
 *        directory and return them sorted lexicographically.
 *
 * @param folder Directory path containing the image dataset.
 *
 * @return Vector of full file paths.
 *
 * @throws std::runtime_error If the directory does not exist, is not a directory,
 *         or contains no supported image files.
 *
 * @note Lexicographic ordering defines the Z‑axis order in reconstruction.
 */
std::vector<std::string> load_files(const std::string &folder);

/**
 * @brief Save the reconstructed height map as a float32 .npy file inside the
 *        dataset’s output directory.
 *
 * @param height_map CV_32F matrix (Ny × Nx) containing physical heights.
 * @param base_name  Dataset name used as directory and filename prefix.
 *
 * @throws std::runtime_error If the output directory or file cannot be written.
 *
 * @details Output structure:
 *      output/<base_name>/<base_name>_height.npy
 *
 * @param out_dir Optional output directory override.  Empty (default) =
 *      cfg::OUTPUT_FOLDER/base_name.  Used to stage results in a temporary
 *      folder that the caller renames atomically once complete.
 */
void save_results(const cv::Mat &height_map, const std::string &base_name,
                  const std::string &out_dir = "");

/**
 * @brief Select a small set of representative pixels (median height, high‑flat,
 *        low‑flat) and save their Z‑profiles using streaming access to TIFF data.
 *
 * @param height_map  CV_32F (Ny × Nx) map of physical heights.
 * @param files       TIFF file list.
 * @param pattern     Regex used for extracting Z‑positions.
 * @param stack_info  Metadata (Nx, Ny, Nz, z‑order).
 * @param base_name   Basename used for saving output diagnostics.
 * @param margin      Exclude a border around the image (default: 5 pixels).
 *
 * @note Uses the same Hilbert‑based Z‑profile reconstruction as the main pipeline.
 * @note Intended for diagnostic visualisation.
 */
/// k_avg / dk: globally-estimated FFT bandpass parameters from the main
/// reconstruction (method == 2 only). Forwarded to save_pixel_plot so that
/// the displayed envelopes use the same filter as the height map.
/// Pass -1/-1 (defaults) to use the per-pixel estimate (legacy behaviour).
///
/// grid_size: controls which pixel plots are saved in addition to the median:
///   0  — save only the median pixel plot (default).
///   N>0 — save an N×N evenly-spaced grid of pixel plots (+ the median).
///
/// out_dir: optional output directory override. Empty (default) =
///   cfg::OUTPUT_FOLDER/base_name. Used to stage results in a temporary folder
///   that the caller renames atomically once complete.
void save_representative_pixels_streaming(
    const cv::Mat &height_map,
    const std::vector<std::string> &files,
    const std::regex &pattern,
    const ImageStack &stack_info,
    const std::string &base_name,
    int margin = 5,
    const std::array<double, 3> &channel_weights = {cfg::GRAY_WEIGHT_R,
                                                     cfg::GRAY_WEIGHT_G,
                                                     cfg::GRAY_WEIGHT_B},
    int method = 1,
    int k_avg = -1,
    int dk = -1,
    int grid_size = 0, ///< 0 = median only; N = N×N equispaced grid + median
    const std::string &out_dir = "");