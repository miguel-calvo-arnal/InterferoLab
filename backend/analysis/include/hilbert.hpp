#pragma once

#include <array>
#include <vector>
#include <fftw3.h>
#include <unordered_map> // Per‑thread cache (FFTW plans by size N)
#include <mutex>         // To synchronise plan creation (FFTW is not thread‑safe during planning)

/**
 * @brief Class for computing the Hilbert envelope of a 1‑D real signal.
 *
 * Implements the classical analytic signal:
 *
 *      x_a = IFFT( FFT(x) * H )
 *
 * where H is the analytic filter (doubling positive frequencies,
 * zeroing negative frequencies, preserving DC and Nyquist).
 *
 * After the inverse FFT, the envelope is obtained via:
 *
 *      env = sqrt( real(x_a)^2 + imag(x_a)^2 )
 *
 * This class is optimised for interferometric work:
 *   - Uses FFTW3 in single precision (fftwf_) for speed.
 *   - Maintains a per‑thread cache of FFTW plans (thread_local),
 *     ensuring safe and efficient execution inside OpenMP regions.
 *   - The creation of FFTW plans is ALWAYS protected by a mutex,
 *     because FFTW is not thread‑safe during plan generation.
 */
class HilbertEnvelope
{
public:
    /**
     * @brief Compute the Hilbert envelope of a real-valued vector.
     *
     * @param signal Input vector (float), real-valued, length N.
     *
     * @return std::vector<float> of length N containing the envelope
     *         (magnitude) of the analytic signal.
     *
     * @note Thread-safe: HilbertEnvelope instances hold no state (the plan
     *       cache is a static thread_local member), so a single instance may
     *       be shared freely between threads — there is no need for
     *       per-thread instances.
     */
    std::vector<float> compute(const std::vector<float> &signal) const;

    // Same computation but writes into caller-owned buffers to avoid per-call
    // heap allocation. buf and out are resized only when their capacity is
    // insufficient (i.e. zero allocations in steady state).
    // ComplexBuf is layout-compatible with fftwf_complex* (float[2]) but avoids
    // std::vector<T[2]> which is ill-formed in C++17/20 (arrays are not value-initializable).
    using ComplexBuf = std::vector<std::array<float, 2>>;

    void compute_into(const std::vector<float> &signal,
                      ComplexBuf               &buf,
                      std::vector<float>       &out) const;

    /**
     * @brief Compute the envelope via the Fourier-filter method (PDF Method 2).
     *
     * Algorithm:
     *   1. Forward FFT of the signal.
     *   2. Zero the DC bin (k=0) and all negative-frequency bins (k > N/2).
     *   3. Find k_avg = argmax |F(k)| in [1, N/2].
     *   4. Estimate Δk = 2σ of |F(k)| over k = 1..N/2 (full positive half-spectrum).
     *      The noise floor inflates Δk to roughly N/4, which makes the bandpass
     *      below wide enough to retain nearly all positive frequencies — equivalent
     *      to the classical Hilbert transform.  A local window was tried but caused
     *      height artefacts on flat surfaces (see implementation comment).
     *   5. Apply a rectangular bandpass: zero bins with |k − k_avg| > Δk.
     *      A Hann-windowed taper is intentionally NOT used: for the one-sided
     *      analytic signal the Hann window introduces phase-dependent bias in
     *      |IFFT|, shifting the centroid differently for flat vs. edge pixels.
     *      The rectangular window's Gibbs ringing (at 2·f₀) is suppressed by
     *      the Gaussian smoothing applied after envelope extraction.
     *   6. Inverse FFT; envelope = 2 × |IFFT| / N.
     *
     * The result is written into `out` (resized as needed).  `buf` is reused
     * as FFT scratch space to avoid per-call heap allocations.
     */
    void compute_fft_bandpass(const std::vector<float> &signal,
                              ComplexBuf               &buf,
                              std::vector<float>       &out) const;

    /**
     * @brief Same as compute_fft_bandpass but uses externally supplied k_avg
     *        and dk instead of computing them from the individual signal.
     *
     * Use this overload when the bandpass parameters have been pre-computed
     * from the amplitude-averaged spectrum of many pixels (see
     * estimate_bandpass_params).  Applying a consistent filter to every pixel
     * eliminates the per-pixel filter variation that causes systematic height
     * artefacts between regions with different local SNR (e.g. the centre vs.
     * the edge of a flat feature).
     */
    void compute_fft_bandpass(const std::vector<float> &signal,
                              ComplexBuf               &buf,
                              std::vector<float>       &out,
                              int k_avg, int dk) const;

    /**
     * @brief Estimate k_avg and Δk from the amplitude spectrum averaged over
     *        a representative set of pixel signals.
     *
     * Averages |FFT(signal)| over all supplied signals, then finds k_avg
     * (argmax of the average amplitude) and dk = 2σ of the amplitude
     * distribution around k_avg.  The averaged spectrum has lower noise than
     * any individual pixel's spectrum, giving a more stable and consistent
     * bandpass for the entire image.
     *
     * @param signals  Representative pixel Z-traces (each of length Nz).
     * @param scratch  Reusable ComplexBuf (avoids per-call allocation).
     * @return {k_avg, dk} to pass to the fixed-params overload.
     */
    static std::pair<int,int> estimate_bandpass_params(
        const std::vector<std::vector<float>> &signals,
        ComplexBuf &scratch);

    /**
     * @brief Bandpass-filter a real signal in place using a Hann-windowed FFT filter.
     *
     * Intended to be called on the baseline-subtracted Z-signal before
     * compute_into().  It suppresses:
     *   - DC / very-low frequencies (already removed by baseline, kept at zero).
     *   - Harmonic distortion (H2, H3) introduced by Bayer demosaicing and
     *     coherent etalon reflections in the optical path.
     *   - High-frequency shot noise above the fringe band.
     *
     * The passband is a Hann window centred at |f| = f_center with full
     * bandwidth bw (cycles/sample).  The filter is symmetric (passes both
     * positive and negative frequencies) so the output remains real-valued.
     *
     * The caller-supplied buf is reused as FFT scratch space: no extra heap
     * allocations occur when buf is already sized to signal.size().
     *
     * @param signal   Z-signal to filter (modified in place).
     * @param buf      Scratch ComplexBuf (reused from the outer pixel loop).
     * @param f_center Centre frequency in cycles/sample (= 2*dz/lambda).
     * @param bw       Full bandwidth in cycles/sample (see cfg::FRINGE_FREQ_BW).
     */
    void bandpass_inplace(std::vector<float> &signal,
                          ComplexBuf         &buf,
                          float               f_center,
                          float               bw) const;

    /**
     * @brief Compute the forward (unnormalized) DFT of \p signal.
     *
     * On return buf[k] = {Re(q(k)), Im(q(k))} for k = 0..N-1.
     * Uses the same thread-local FFTW plan as all other methods.
     */
    void forward_fft(const std::vector<float> &signal, ComplexBuf &buf) const;

private:
    /**
     * @brief Structure holding the forward and inverse FFTW plans for a given size N.
     *
     * Keeping both plans avoids re‑creating FFTW plans for the same size.
     * Plans are stored per‑thread, meaning each OpenMP thread has its own
     * private forward/backward plan set.
     */
    struct PlanSet
    {
        fftwf_plan fwd = nullptr; ///< Forward FFT plan (real -> complex, in‑place)
        fftwf_plan inv = nullptr; ///< Inverse FFT plan (complex -> real, in‑place)

        ~PlanSet()
        {
            if (fwd) { fftwf_destroy_plan(fwd); fwd = nullptr; }
            if (inv) { fftwf_destroy_plan(inv); inv = nullptr; }
        }

        // Non-copyable: FFTW plans must not be duplicated.
        PlanSet() = default;
        PlanSet(const PlanSet &) = delete;
        PlanSet &operator=(const PlanSet &) = delete;
        PlanSet(PlanSet &&o) noexcept : fwd(o.fwd), inv(o.inv) { o.fwd = o.inv = nullptr; }
        PlanSet &operator=(PlanSet &&o) noexcept
        {
            if (this != &o)
            {
                if (fwd) fftwf_destroy_plan(fwd);
                if (inv) fftwf_destroy_plan(inv);
                fwd = o.fwd; inv = o.inv;
                o.fwd = o.inv = nullptr;
            }
            return *this;
        }
    };

    /**
     * @brief Per‑thread cache of FFTW plan sets.
     *
     * thread_local ensures each OpenMP thread keeps its own map<size, PlanSet>,
     * avoiding contention and allowing FFTs to run safely in parallel.
     *
     * Key: signal size N.
     * Value: PlanSet containing forward and inverse plans.
     */
    static thread_local std::unordered_map<int, PlanSet> plan_cache;

    /**
     * @brief Mutex used ONLY during FFTW plan creation.
     *
     * FFTW is not thread‑safe during the planning stage (fftw_plan_*),
     * but plans may be executed concurrently afterwards as long as
     * threads use separate plan sets (ensured by plan_cache).
     */
    static std::mutex plan_mutex;

    /**
     * @brief Ensure FFTW plans exist for a given size N, creating them if necessary.
     *
     * Checks whether the current thread already has a plan for size N.
     * If not, allocates a temporary buffer and calls fftwf_plan_dft_1d
     * inside a mutex‑protected block.
     *
     * @note This is executed at most once per thread per unique size N.
     */
    static void ensure_plans(int N);

    /**
     * @brief Execute the forward FFT using the cached per‑thread plan.
     *
     * @param buf Pointer to FFTW buffer of length N (fftwf_complex).
     */
    static void fft_forward(fftwf_complex *buf, int N);

    /**
     * @brief Execute the inverse FFT using the cached per‑thread plan.
     *
     * @param buf Pointer to FFTW buffer of length N (fftwf_complex).
     */
    static void fft_inverse(fftwf_complex *buf, int N);
};