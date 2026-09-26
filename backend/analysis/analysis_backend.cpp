#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "analysis_api.hpp"
#include "config.hpp"
#include "methods.hpp"

#include <array>
#include <fftw3.h>

namespace py = pybind11;

/**
 * Python wrapper for run_analysis_for_folder.
 * - Releases the GIL only during heavy C++ computation.
 * - Reacquires the GIL in every callback.
 *
 * The Python Global Interpreter Lock or GIL, is a mutex (or a lock) that
 * allows only one thread to hold the control of the Python interpreter.
 *
 * It is safer, but here is a bottleneck, so we release it for the heavy
 * functitons to allow multithreading.
 */
static py::dict run_analysis_py(const std::string &dataset_folder,
                                const std::string &name_in,
                                py::object progress,
                                py::object log,
                                py::object channel_weights_obj,
                                int method,
                                int pixel_plot_grid)
{
    // Always reset the cancellation flag
    request_cancel(false);

    // -------------------------------
    // 1. Prepare safe Python callbacks
    // -------------------------------
    ProgressCallback progress_cb = nullptr;
    if (!progress.is_none())
    {
        py::function f = progress.cast<py::function>();
        progress_cb = [f](int percent,
                          const std::string &stage,
                          int done,
                          int total)
        {
            // Reacquire the GIL before calling Python
            py::gil_scoped_acquire gil;
            // A raising progress callback must not tear down the analysis
            // (nor the process): progress is best-effort cosmetics.  Log the
            // error on the Python side and keep going.  (The log callback, in
            // contrast, is allowed to propagate.)
            try
            {
                f(percent, stage, done, total);
            }
            catch (py::error_already_set &e)
            {
                e.discard_as_unraisable("analysis_backend progress callback");
            }
        };
    }

    LogCallback log_cb = nullptr;
    if (!log.is_none())
    {
        py::function f = log.cast<py::function>();
        log_cb = [f](const std::string &level,
                     const std::string &msg)
        {
            // Reacquire the GIL before calling Python
            py::gil_scoped_acquire gil;
            f(level, msg);
        };
    }

    // -------------------------------
    // 2. Parse channel_weights (optional)
    // -------------------------------
    std::array<double, 3> weights = {
        cfg::GRAY_WEIGHT_R, cfg::GRAY_WEIGHT_G, cfg::GRAY_WEIGHT_B};

    if (!channel_weights_obj.is_none())
    {
        py::sequence seq = channel_weights_obj.cast<py::sequence>();
        if (seq.size() != 3)
            throw std::runtime_error("channel_weights must be a sequence of 3 floats");
        for (int i = 0; i < 3; ++i)
            weights[i] = seq[i].cast<double>();
    }

    // -----------------------------------------
    // 3. Run heavy C++ analysis with released GIL
    // -----------------------------------------
    AnalysisResultPaths paths;
    {
        py::gil_scoped_release release; // Release the GIL for the heavy C++ section
        paths = run_analysis_for_folder(dataset_folder, name_in, progress_cb, log_cb, weights, method, pixel_plot_grid);
    }
    // Exiting the block, gil_scoped_release destructor restores the GIL

    // -----------------------------------------
    // 4. Build the result dictionary (requires GIL)
    // -----------------------------------------
    // "cancelled" lets callers distinguish a cancelled run (empty paths)
    // from a successful one. Derived from the authoritative outcome, not by
    // re-reading the global flag: a cancel click landing after the last
    // in-run check would otherwise label a fully published result cancelled.
    py::dict result;
    result["output_folder"] = paths.output_folder;
    result["heightmap"] = paths.heightmap_npy;
    result["cancelled"] = paths.output_folder.empty();
    // Global bandpass of Methods 2 and 4 (-1 for 1 and 3).  It is the only
    // number the whole image shares, so it is what a caller compares to check
    // that two runs of the same dataset were really the same analysis.
    result["k_avg"] = paths.k_avg;
    result["dk"] = paths.dk;
    return result;
}

// ---------------- PYBIND11 MODULE ----------------

PYBIND11_MODULE(analysis_backend, m)
{
    m.doc() = "Bindings for the interferometry reconstruction backend (analysis).";

    // FFTW plan creation/destruction is not thread-safe by default.  Plans are
    // created under our own mutex, but the thread_local plan caches destroy
    // their plans when worker threads die (e.g. OpenMP pool teardown), outside
    // any lock (B-6).  Making the planner thread-safe once at module load
    // closes that race for the whole process.
    fftwf_make_planner_thread_safe();

    m.def(
        "run_analysis",
        &run_analysis_py,
        py::arg("dataset_folder"),
        py::arg("name") = "",
        py::arg("progress_cb") = py::none(),
        py::arg("log_cb") = py::none(),
        py::arg("channel_weights") = py::none(),
        py::arg("method") = 1,
        py::arg("pixel_plot_grid") = 0,
        R"doc(
            Execute the full analysis on the folder 'dataset_folder'.

            channel_weights: optional list of 3 floats [wr, wg, wb] for
            RGB-to-grayscale conversion.  They are normalized internally so only
            the ratios matter (e.g. [1,0,0] = red only, [1,1,0] = red+green
            equal weight).  None uses the defaults from config.hpp.
            Has no effect on mono (single-channel) images.

            method: integer id of the reconstruction algorithm to use (default 1).
              Call get_reconstruction_methods() for the full list of available
              methods with names and descriptions.

            Returns a dict with keys:
              'output_folder' (str), 'heightmap' (str) — empty when cancelled —
              'cancelled' (bool), and 'k_avg' / 'dk' (int): the global bandpass
              Methods 2 and 4 applied to every pixel, or -1 for Methods 1 and 3
              (which have none).  Two runs of the same dataset must report the
              same pair; see cfg::BAND_SAMPLE_ROWS in config.hpp.
        )doc");

    m.def(
        "cancel_analysis",
        [&]
        { request_cancel(true); },
        R"doc(
            Request cancellation of the running analysis.
        )doc");

    m.def(
        "get_reconstruction_methods",
        []() -> py::list
        {
            py::list result;
            for (const auto &m : methods::RECONSTRUCTION_METHODS)
            {
                py::dict d;
                d["id"]          = m.id;
                d["name"]        = m.name;
                d["description"] = m.description;
                result.append(d);
            }
            return result;
        },
        R"doc(
            Return a list of dicts describing all registered reconstruction methods.
            Each dict has keys: 'id' (int), 'name' (str), 'description' (str).
            The UI uses this to populate the method combo box without hardcoding.
        )doc");
}