#include "analysis_api.hpp"  // includes config.hpp transitively

#include "utils.hpp"
#include "reconstruction.hpp"
#include "methods.hpp"

#include <array>
#include <cstdio>
#include <ctime>
#include <memory>
#include <regex>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

std::atomic<bool> g_cancel_requested{false};

void request_cancel(bool value)
{
    g_cancel_requested.store(value, std::memory_order_relaxed);
}

bool cancel_requested()
{
    return g_cancel_requested.load(std::memory_order_relaxed);
}

// Helper: obtain the basename of a folder path
static std::string basename_from_folder(const std::string &folder)
{
    fs::path p(folder);
    if (p.has_filename())
    {
        return p.filename().string();
    }
    else
    {
        return p.parent_path().filename().string();
    }
}

AnalysisResultPaths run_analysis_for_folder(
    const std::string &folder,
    const std::string &name_in,
    const ProgressCallback &progress_cb,
    const LogCallback &log_cb,
    const std::array<double, 3> &channel_weights,
    int method,
    int pixel_plot_grid)
{
    AnalysisResultPaths result;

    auto log = [&](const std::string &level, const std::string &msg)
    {
        if (log_cb)
            log_cb(level, msg);
        else
            std::cerr << "[" << level << "] " << msg << '\n';
    };

    log("info", "Starting analysis for folder: " + folder);

    // EARLY CANCEL
    if (cancel_requested())
    {
        log("warn", "Analysis cancelled before start.");
        return AnalysisResultPaths{};
    }

    // 1. Determine experiment basename
    std::string name = name_in.empty() ? basename_from_folder(folder) : name_in;
    log("info", "Experiment name: " + name);

    // 2. Regex
    std::regex pattern(cfg::REGEX_PATTERN);
    ImageStack stack_info;

    // 3. Load files
    log("info", "Loading files...");
    auto files = load_files(folder);
    int total_files = static_cast<int>(files.size());

    if (cancel_requested())
    {
        log("warn", "Analysis cancelled after loading files.");
        return AnalysisResultPaths{};
    }

    // 3b. Validate method id and the minimum number of frames it needs.
    //     Undersized stacks previously reached the per-method stencils and
    //     caused out-of-range reads (B-1/B-2); fail early with a clear message.
    const methods::MethodInfo *method_info = methods::find(method);
    if (!method_info)
    {
        const std::string msg =
            "Unknown reconstruction method id " + std::to_string(method)
            + " (valid ids: see get_reconstruction_methods()).";
        log("error", msg);
        throw std::runtime_error(msg);
    }
    if (total_files < method_info->min_frames)
    {
        const std::string msg =
            "Dataset has only " + std::to_string(total_files)
            + " image(s), but method " + std::to_string(method) + " ("
            + method_info->name + ") requires at least "
            + std::to_string(method_info->min_frames)
            + " frames. Acquire more frames or choose another method.";
        log("error", msg);
        throw std::runtime_error(msg);
    }

    // 4. Progress callback used by reconstruct_height_map_rowchunks
    //    The reconstruction function reports a progress float ∈ [0,1].
    auto progress_reconstruct = [&](float frac)
    {
        if (!progress_cb)
            return;

        frac = std::clamp(frac, 0.0f, 1.0f);
        int percent = static_cast<int>(std::round(frac * 100.0f));
        int done = static_cast<int>(std::round(frac * total_files));

        progress_cb(percent, "reconstruct", done, total_files);
    };

    if (progress_cb)
        progress_cb(0, "load_files", 0, total_files);

    // 5. Reconstruct height map
    log("info", "Reconstructing height map...");

    if (cancel_requested())
    {
        log("warn", "Analysis cancelled before reconstruction.");
        return AnalysisResultPaths{};
    }

    int recon_k_avg = -1, recon_dk = -1;
    auto height_map = reconstruct_height_map_rowchunks(
        files,
        pattern,
        stack_info,
        progress_reconstruct,
        channel_weights,
        method,
        &recon_k_avg,
        &recon_dk);

    if (cancel_requested())
    {
        log("warn", "Analysis cancelled during reconstruction.");
        return AnalysisResultPaths{};
    }

    // 6. Save results
    log("info", "Saving results...");

    if (cancel_requested())
    {
        log("warn", "Analysis cancelled before saving.");
        return AnalysisResultPaths{};
    }

    // All results are staged in a temporary folder and the previous output is
    // only replaced once the new one is complete (write-to-temp + rename).
    // Previously the old folder was removed BEFORE saving, so a failed
    // reanalysis (full disk, crash, cancellation) destroyed the previous
    // result with nothing to replace it (B-12).
    const fs::path final_folder = fs::path(cfg::OUTPUT_FOLDER) / name;
    const fs::path tmp_folder   = fs::path(cfg::OUTPUT_FOLDER) / (name + ".partial");

    {
        std::error_code ec;
        fs::remove_all(tmp_folder, ec); // clean any leftover from a crashed run
    }

    try
    {
        save_results(height_map, name, tmp_folder.string());

        // Write metadata.json so downstream tools know the reconstruction parameters
        {
            fs::path meta_path = tmp_folder / "metadata.json";
            struct FileCloser
            {
                void operator()(FILE *f) const noexcept { if (f) std::fclose(f); }
            };
            std::unique_ptr<FILE, FileCloser> meta(
                std::fopen(meta_path.string().c_str(), "w"));
            if (meta)
            {
                std::time_t now = std::time(nullptr);
                char ts_buf[32];
                struct tm tm_buf{};
#if defined(_WIN32)
                localtime_s(&tm_buf, &now);
#else
                localtime_r(&now, &tm_buf);
#endif
                std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
                std::fprintf(meta.get(),
                    "{\n"
                    "  \"method\": %d,\n"
                    "  \"baseline_sigma\": %g,\n"
                    "  \"envelope_sigma\": %g,\n"
                    "  \"timestamp\": \"%s\"\n"
                    "}\n",
                    method,
                    static_cast<double>(cfg::BASELINE_SIGMA),
                    static_cast<double>(cfg::ENVELOPE_SIGMA),
                    ts_buf);
                log("info", "Metadata saved: " + meta_path.string());
            }
            else
            {
                log("warn", "Could not write metadata file: " + meta_path.string());
            }
        }

        save_representative_pixels_streaming(
            height_map, files, pattern, stack_info, name, 10, channel_weights, method,
            recon_k_avg, recon_dk, pixel_plot_grid, tmp_folder.string());
    }
    catch (...)
    {
        // Saving failed: discard the partial staging folder.  The previous
        // result (if any) is still intact in final_folder.
        std::error_code ec;
        fs::remove_all(tmp_folder, ec);
        throw;
    }

    if (cancel_requested())
    {
        log("warn", "Analysis cancelled while saving results.");
        std::error_code ec;
        fs::remove_all(tmp_folder, ec);
        return AnalysisResultPaths{};
    }

    // 7. Publish the new output via TWO renames instead of remove_all+rename:
    //    if the previous folder is locked (Windows: a viewer, an explorer
    //    window, antivirus), the first rename fails while the old result is
    //    still fully intact — remove_all could die halfway through, leaving
    //    the old output partially destroyed and the new one stranded in the
    //    staging folder (which the next run deletes).
    const fs::path old_folder = fs::path(cfg::OUTPUT_FOLDER) / (name + ".old");
    {
        std::error_code ec;
        fs::remove_all(old_folder, ec); // leftover from a previous failed publish
    }
    if (fs::exists(final_folder))
    {
        try
        {
            fs::rename(final_folder, old_folder);
        }
        catch (const fs::filesystem_error &)
        {
            throw std::runtime_error(
                "Cannot replace the existing output folder '"
                + final_folder.string()
                + "': it appears to be in use. Close any program holding its "
                  "files and re-run the analysis. The previous result is "
                  "untouched.");
        }
    }
    fs::rename(tmp_folder, final_folder);
    {
        std::error_code ec;
        fs::remove_all(old_folder, ec); // best effort; retried at next publish
        if (ec)
            log("warn", "Could not delete previous output backup: "
                        + old_folder.string());
    }

    fs::path height_npy = final_folder / (name + "_height.npy");

    result.output_folder = final_folder.string();
    result.heightmap_npy = height_npy.string();

    log("info", "Analysis finished. Output folder: " + result.output_folder);
    return result;
}