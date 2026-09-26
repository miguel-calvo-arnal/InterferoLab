#pragma once

/**
 * @file methods.hpp
 * @brief Registry of all available CSI reconstruction methods.
 *
 * This is the SINGLE place to look when adding a new method.
 *
 * ======================================================================
 * HOW TO ADD A NEW METHOD (step-by-step)
 * ======================================================================
 * 1. Pick an unused integer ID (e.g. 3).
 *
 * 2. Add a MethodInfo row to RECONSTRUCTION_METHODS[] below with:
 *      - id                   : the integer you just picked.
 *      - name                 : short label shown in the GUI combo box.
 *      - description          : one-sentence tooltip.
 *      - needs_baseline_removal: set to true if the method needs the
 *          Gaussian DC-subtraction step before envelope extraction
 *          (Method 1 needs it; Method 2 does not because its FFT
 *          bandpass already zeros DC).
 *      - needs_global_params  : set to true if the method requires a
 *          one-time global pre-computation on the first row-chunk (like
 *          the bandpass parameter estimation used by Method 2).
 *
 * 3. In src/reconstruction.cpp, add a case to compute_envelope():
 *        case 3:
 *            // your per-pixel envelope logic here
 *            break;
 *    If needs_global_params == true, also add a case to the
 *    estimate_global_params() block in reconstruct_height_map_rowchunks.
 *
 * 4. Recompile.  The Python UI discovers the new method automatically
 *    via analysis_backend.get_reconstruction_methods() — no Python
 *    changes are required.
 * ======================================================================
 */

namespace methods
{

/**
 * @brief Metadata for one reconstruction method.
 *
 * All fields are plain data (constexpr-friendly).  Behaviour is encoded
 * only through the flags; the actual signal processing lives in
 * src/reconstruction.cpp::compute_envelope().
 */
struct MethodInfo
{
    int         id;
    const char *name;
    const char *description;

    /// True  → subtract Gaussian low-pass baseline from the Z-signal
    ///          BEFORE calling compute_envelope() for this method.
    /// False → the method handles DC itself (e.g. FFT bandpass zeros k=0).
    bool needs_baseline_removal;

    /// True  → call estimate_global_params() once on the first row-chunk
    ///          before the main pixel loop (used by Method 2 for global
    ///          k_avg / dk estimation).
    bool needs_global_params;

    /// True  → report the position of the envelope maximum (peak locator,
    ///          parabolically refined to sub-step precision) instead of the
    ///          weighted centroid.  Use for methods whose envelope is
    ///          asymmetric so the centroid drifts from the peak (e.g. Method 3
    ///          with a broadband halogen source).  See find_envelope_peak()
    ///          in reconstruction.cpp for the bias the parabola introduces on
    ///          an asymmetric envelope.
    bool use_peak_locator;

    /// Minimum number of Z-frames (images) this method needs to produce a
    /// meaningful result without reading outside its stencil:
    ///   M1 central difference needs z-1/z+1  → 3
    ///   M2 FFT bandpass needs N > 2          → 3
    ///   M3 5-point kernel needs z±2          → 5
    ///   M4 group delay needs ≥2 usable bins  → 4
    /// run_analysis_for_folder() validates the dataset against this before
    /// starting the reconstruction.
    int min_frames;
};

// ---------------------------------------------------------------------------
// *** Add your new method here ***
// ---------------------------------------------------------------------------
inline constexpr MethodInfo RECONSTRUCTION_METHODS[] = {
    {
        1,
        "Squared-derivative centroid",
        "Centroid of (dG_i)^2 after Gaussian baseline removal. "
        "Fast and robust for high-SNR acquisitions.",
        /*needs_baseline_removal=*/true,
        /*needs_global_params=*/false,
        /*use_peak_locator=*/false,
        /*min_frames=*/3,
    },
    {
        2,
        "FFT Fourier-filter centroid",
        "Bandpass around the dominant frequency k_avg (estimated once from the "
        "first row-chunk), then 2*|IFFT| envelope and weighted centroid with "
        "2-sigma refinement.",
        /*needs_baseline_removal=*/false,
        /*needs_global_params=*/true,
        /*use_peak_locator=*/false,
        /*min_frames=*/3,
    },
    {
        3,
        "PSI 5-point kernel",
        "5-point PSI correlation kernel (Larkin 1996): "
        "|S_z| = sqrt((2I_{z-1}-2I_{z+1})^2 + (-I_{z-2}+2I_z-I_{z+2})^2), "
        "with the envelope maximum refined to sub-step precision by a "
        "3-point parabola. DC-insensitive; fast; good for high-fringe-density "
        "acquisitions.",
        /*needs_baseline_removal=*/false,
        /*needs_global_params=*/false,
        /*use_peak_locator=*/true,
        /*min_frames=*/5,
    },
    {
        4,
        "Frequency-domain linear fit",
        "Group-delay estimator: arg of the SUM of the cross products "
        "FFT[k+1]*conj(FFT[k]) over the carrier band (vector average). "
        "No phase unwrapping and no singularity at n_peak = Nz/2 "
        "(de Groot & Deck 1995).",
        /*needs_baseline_removal=*/true,
        /*needs_global_params=*/true,
        /*use_peak_locator=*/false,
        /*min_frames=*/4,
    },
};

/// Number of registered methods — computed at compile time.
inline constexpr int N_METHODS =
    static_cast<int>(sizeof(RECONSTRUCTION_METHODS) / sizeof(RECONSTRUCTION_METHODS[0]));

/// Look up a method by integer ID.  Returns nullptr if not found.
inline const MethodInfo *find(int id) noexcept
{
    for (const auto &m : RECONSTRUCTION_METHODS)
        if (m.id == id)
            return &m;
    return nullptr;
}

} // namespace methods
