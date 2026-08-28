#include "utils.hpp"
#include "config.hpp"
#include "npy.hpp"
#include "reconstruction.hpp"

#include <array>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <vector>
#include <numeric>
#include <cstdint>
#include <unordered_set>

namespace fs = std::filesystem;

#if defined(__linux__)
#include <fstream>
#include <string>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

// ------------------------------------------------------------
//   RAM detection (cross-platform)
// ------------------------------------------------------------
std::uint64_t get_free_ram_bytes()
{
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t value;
    std::string unit;

    while (meminfo >> key >> value >> unit)
    {
        if (key == "MemAvailable:")
            return value * 1024ull; // kB -> bytes
    }
    return 0;

#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullAvailPhys;

#elif defined(__APPLE__)
    // Use vm_statistics64 to get free pages (HW_MEMSIZE returns total RAM).
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    host_page_size(host, &page_size);
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_stat),
                          &count) == KERN_SUCCESS)
    {
        return static_cast<std::uint64_t>(vm_stat.free_count) * page_size;
    }
    return 0;

#else
    return 0;
#endif
}

// ------------------------------------------------------------
//   Auto-select chunk size based on RAM + image shape
// ------------------------------------------------------------
int auto_row_chunk(int Nx, int Ny, int Nz, double safety_ratio)
{
    std::uint64_t free_ram = get_free_ram_bytes();

    if (free_ram == 0)
        return cfg::ROW_CHUNK_SIZE; // fallback

    // Only allow a safe fraction of available RAM to be used
    std::uint64_t allowed_ram = std::uint64_t(double(free_ram) * safety_ratio);

    // Memory used per chunk-row: Nz * Nx * float
    std::uint64_t per_row_bytes = std::uint64_t(Nx) *
                                  std::uint64_t(Nz) *
                                  sizeof(float);

    if (per_row_bytes == 0)
        return cfg::ROW_CHUNK_SIZE; // fallback

    // Maximum number of rows fitting into allowed RAM
    int R = int(allowed_ram / per_row_bytes);

    // Reasonable limits
    if (R < 16)
        R = 16;
    if (R > 4096)
        R = 4096;
    if (R > Ny)
        R = Ny;

    return R;
}

// Convert a string to lowercase.
// Useful for case-insensitive file extension comparison (.tiff / .TIFF)
static inline std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [&](unsigned char c)
                   { return std::tolower(c); });
    return s;
}

// -----------------------------------------------------------------------------
// Display all subfolders inside BASE_DIR and allow the user to choose one.
// Returns: full folder path + folder name.
// -----------------------------------------------------------------------------
std::pair<std::string, std::string> select_folder(const std::string &base_dir)
{
    if (!fs::exists(base_dir) || !fs::is_directory(base_dir))
        throw std::runtime_error("Base directory does not exist or is not a directory: " + base_dir);

    std::vector<std::string> subfolders;
    for (auto &p : fs::directory_iterator(base_dir))
        if (p.is_directory())
            subfolders.push_back(p.path().filename().string());

    std::sort(subfolders.begin(), subfolders.end());
    if (subfolders.empty())
        throw std::runtime_error("No subfolders found in: " + base_dir);

    std::cout << "\nAvailable image folders:\n";
    for (size_t i = 0; i < subfolders.size(); ++i)
        std::cout << i << ": " << subfolders[i] << "\n";

    // User interactive input
    int choice;
    while (true)
    {
        std::cout << "\nSelect folder number: ";
        if (!(std::cin >> choice))
            throw std::runtime_error("Invalid input");
        if (choice >= 0 && static_cast<size_t>(choice) < subfolders.size())
            break;
        std::cout << "Invalid selection. Try again.\n";
    }

    const std::string folder =
        (fs::path(base_dir) / subfolders[static_cast<size_t>(choice)]).string();

    return {folder, subfolders[static_cast<size_t>(choice)]};
}

// -----------------------------------------------------------------------------
// Read a folder and return all TIFF files sorted alphabetically.
// -----------------------------------------------------------------------------
std::vector<std::string> load_files(const std::string &folder)
{
    if (!fs::exists(folder) || !fs::is_directory(folder))
        throw std::runtime_error("Folder does not exist or is not a directory: " + folder);

    std::vector<std::string> files;
    for (auto &p : fs::directory_iterator(folder))
    {
        if (!p.is_regular_file())
            continue;
        const auto ext = to_lower(p.path().extension().string());
        if (ext == ".tif" || ext == ".tiff" || ext == ".bin12" || ext == ".png")
            files.push_back(p.path().string());
    }

    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("No image files (.tif/.tiff/.bin12/.png) found in: " + folder);

    std::cout << "Found " << files.size() << " images\n";
    return files;
}

// -----------------------------------------------------------------------------
// Save reconstructed height map in .npy format inside the dataset's output folder.
// -----------------------------------------------------------------------------
void save_results(const cv::Mat &height_map,
                  const std::string &base_name,
                  const std::string &out_dir)
{
    fs::path folder = out_dir.empty() ? fs::path(cfg::OUTPUT_FOLDER) / base_name
                                      : fs::path(out_dir);
    fs::create_directories(folder);

    std::string npy_path =
        (folder / (base_name + "_height.npy")).string();

    save_npy(npy_path,
             height_map.ptr<float>(),
             height_map.rows,
             height_map.cols);

    std::cout << "[OK] Saved height map to: " << npy_path << "\n";
}

// -----------------------------------------------------------------------------
// Save representative pixel traces using streaming (no full stack needed).
// -----------------------------------------------------------------------------
void save_representative_pixels_streaming(
    const cv::Mat &height_map,
    const std::vector<std::string> &files,
    const std::regex &pattern,
    const ImageStack &stack_info,
    const std::string &base_name,
    int margin,
    const std::array<double, 3> &channel_weights,
    int method,
    int k_avg,
    int dk,
    int grid_size,
    const std::string &out_dir)
{
    if (height_map.empty())
        throw std::runtime_error("save_representative_pixels_streaming: empty height_map");
    if (height_map.type() != CV_32F)
        throw std::runtime_error("save_representative_pixels_streaming: height_map must be CV_32F");
    if (stack_info.Nx != height_map.cols || stack_info.Ny != height_map.rows)
        throw std::runtime_error("save_representative_pixels_streaming: stack_info dims != height_map dims");

    const int Ny = height_map.rows;
    const int Nx = height_map.cols;

    if (margin < 0)
        margin = 0;
    if (margin * 2 >= Ny || margin * 2 >= Nx)
        throw std::runtime_error("save_representative_pixels_streaming: margin too large for image size");

    const size_t N = static_cast<size_t>(Ny) * Nx;

    // ------------------------------
    // Flatten height_map into H[]
    // ------------------------------
    std::vector<float> H;
    H.reserve(N);

    for (int y = 0; y < Ny; ++y)
    {
        const float *row = height_map.ptr<float>(y);
        H.insert(H.end(), row, row + Nx);
    }

    // ------------------------------
    // Build list of valid indices
    // ------------------------------
    std::vector<int> valid_idx;
    valid_idx.reserve(N);

    for (int y = margin; y < Ny - margin; ++y)
    {
        int base = y * Nx;
        for (int x = margin; x < Nx - margin; ++x)
        {
            int idx = base + x;

            if (idx < 0 || idx >= (int)H.size())
                continue;

            float v = H[idx];
            if (!std::isfinite(v))
                continue;

            valid_idx.push_back(idx);
        }
    }

    if (valid_idx.empty())
        throw std::runtime_error("save_representative_pixels_streaming: no valid indices after margin exclusion");

    // ------------------------------
    // Compute hmin/hmax range
    // ------------------------------
    float hmin = std::numeric_limits<float>::infinity();
    float hmax = -std::numeric_limits<float>::infinity();

    for (int idx : valid_idx)
    {
        float v = H[idx];
        hmin = std::min(hmin, v);
        hmax = std::max(hmax, v);
    }

    // ---------------------------------------------------------
    // 1. MEDIAN REPRESENTATIVE PIXEL
    // ---------------------------------------------------------
    std::vector<float> Hvalid;
    Hvalid.reserve(valid_idx.size());
    for (int idx : valid_idx)
        Hvalid.push_back(H[idx]);

    size_t mid = Hvalid.size() / 2;
    std::nth_element(Hvalid.begin(), Hvalid.begin() + mid, Hvalid.end());
    float median_val = Hvalid[mid];

    std::vector<std::pair<int,int>> pixel_coords;

    {
        float best_diff = std::numeric_limits<float>::max();
        int best_idx = -1;

        for (int idx : valid_idx)
        {
            float diff = std::fabs(H[idx] - median_val);
            if (diff < best_diff)
            {
                best_diff = diff;
                best_idx = idx;
            }
        }

        if (best_idx >= 0)
            pixel_coords.emplace_back(best_idx / Nx, best_idx % Nx);
    }

    // ---------------------------------------------------------
    // 2. ADDITIONAL PIXELS: N×N equispaced grid (only when grid_size > 0)
    // ---------------------------------------------------------
    if (grid_size > 0)
    {
        const int G = grid_size;
        for (int gi = 0; gi < G; ++gi)
        {
            for (int gj = 0; gj < G; ++gj)
            {
                int y, x;
                if (G == 1)
                {
                    y = margin + (Ny - 1 - 2 * margin) / 2;
                    x = margin + (Nx - 1 - 2 * margin) / 2;
                }
                else
                {
                    y = margin + (int)std::round((double)(Ny - 1 - 2 * margin) * gi / (G - 1));
                    x = margin + (int)std::round((double)(Nx - 1 - 2 * margin) * gj / (G - 1));
                }
                y = std::clamp(y, margin, Ny - 1 - margin);
                x = std::clamp(x, margin, Nx - 1 - margin);

                const int idx = y * Nx + x;
                if (idx >= 0 && idx < (int)H.size() && std::isfinite(H[idx]))
                    pixel_coords.emplace_back(y, x);
            }
        }
    }

    // Deduplicate while preserving insertion order (median pixel stays first).
    // Encode each (row, col) pair as a single int64 key for O(1) lookup.
    {
        std::unordered_set<long long> seen_keys;
        seen_keys.reserve(pixel_coords.size());
        std::vector<std::pair<int,int>> deduped;
        deduped.reserve(pixel_coords.size());
        for (auto &p : pixel_coords)
        {
            const long long key = (static_cast<long long>(p.first) << 32)
                                | static_cast<unsigned int>(p.second);
            if (seen_keys.insert(key).second)
                deduped.push_back(p);
        }
        pixel_coords = std::move(deduped);
    }

    if (!pixel_coords.empty())
        save_pixel_plots_batch_streaming(
            pixel_coords, files, pattern, stack_info, base_name,
            channel_weights, method, k_avg, dk, out_dir);

    (void)hmin;
    (void)hmax;
}