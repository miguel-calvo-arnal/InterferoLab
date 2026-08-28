#include "config.hpp"
#include "utils.hpp"
#include "analysis_api.hpp"

#include <array>
#include <fftw3.h>
#include <iostream>

int main()
{
    try
    {
        // See analysis_backend.cpp: thread_local FFTW plan caches destroy
        // plans on thread death outside our planning mutex; a thread-safe
        // planner closes that race for the whole process.
        fftwf_make_planner_thread_safe();

        // 1. Select folder as before
        auto [folder, name_from_dialog] = select_folder(cfg::BASE_DIR);

        // 2. Progress callback
        auto progress_cb = [&](int percent,
                               const std::string &stage,
                               int done,
                               int total)
        {
            (void)done;
            (void)total; // not used at the moment
            std::cout << "\r[" << stage << "] "
                      << percent << "%       " << std::flush;
        };

        // 3. Run analysis using the new API (default channel weights from config)
        std::array<double, 3> weights = {
            cfg::GRAY_WEIGHT_R, cfg::GRAY_WEIGHT_G, cfg::GRAY_WEIGHT_B};
        auto result = run_analysis_for_folder(
            folder,
            name_from_dialog,
            progress_cb,
            nullptr, // log_cb -> stderr is used by default
            weights);

        std::cout << "\nDone.\n";
        std::cout << "Heightmap:     " << result.heightmap_npy << "\n";
        std::cout << "Output folder: " << result.output_folder << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << '\n';
        return 1;
    }

    wait_for_enter_windows_only();
    return 0;
}