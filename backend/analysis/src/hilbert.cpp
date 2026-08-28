#include "hilbert.hpp"

#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------------
// thread_local definition
// -------------------------------
thread_local std::unordered_map<int, HilbertEnvelope::PlanSet>
    HilbertEnvelope::plan_cache;

// Global mutex to protect plan creation
// mutual exclusion lock: just one thread at a time in a block code
// ensures that is thread-safe
std::mutex HilbertEnvelope::plan_mutex;

// ---------------------------------------------------------
// Create FFTW plans for this thread if they do not exist
// ---------------------------------------------------------
void HilbertEnvelope::ensure_plans(int N)
{
    auto it = plan_cache.find(N);
    if (it != plan_cache.end())
        return; // Already exists for this thread

    // Temporary planning buffer: std::array<float,2> is layout-compatible with
    // fftwf_complex (float[2]) but avoids std::vector<float[2]> which is ill-formed.
    std::vector<std::array<float, 2>> tmp(static_cast<size_t>(N));
    auto *fftw_tmp = reinterpret_cast<fftwf_complex *>(tmp.data());

    PlanSet ps;

    {
        // FFTW is NOT thread‑safe when creating plans -> protect this section
        std::lock_guard<std::mutex> lock(plan_mutex);

        // FUTURE (audit O-5, deliberately not applied in Bloque C): switching
        // to FFTW_MEASURE and/or the r2c/c2r interface for the real-valued
        // signal could shave some FFT time, but FFTW_MEASURE changes results
        // between runs (plan-dependent arithmetic order breaks bit-for-bit
        // reproducibility) and r2c requires reworking every consumer of the
        // full complex spectrum (bandpass mirroring, group-delay bins).
        // Risk/benefit did not justify it after O-1/O-2/O-4.
        ps.fwd = fftwf_plan_dft_1d(N, fftw_tmp, fftw_tmp, FFTW_FORWARD, FFTW_ESTIMATE);
        ps.inv = fftwf_plan_dft_1d(N, fftw_tmp, fftw_tmp, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    if (!ps.fwd || !ps.inv)
        throw std::runtime_error("Failed to create FFTW plans");

    plan_cache.emplace(N, std::move(ps));
}

// ---------------------------------------------------------
void HilbertEnvelope::fft_forward(fftwf_complex *buffer, int N)
{
    ensure_plans(N);
    fftwf_execute_dft(plan_cache[N].fwd, buffer, buffer);
}

void HilbertEnvelope::fft_inverse(fftwf_complex *buffer, int N)
{
    ensure_plans(N);
    fftwf_execute_dft(plan_cache[N].inv, buffer, buffer);
}

// ---------------------------------------------------------
// Compute Hilbert envelope into caller-supplied buffers
// (zero steady-state heap allocations when N stays constant)
// ---------------------------------------------------------
void HilbertEnvelope::compute_into(const std::vector<float> &signal,
                                   ComplexBuf &buf,
                                   std::vector<float> &out) const
{
    const int N = static_cast<int>(signal.size());
    if (N == 0)
    {
        out.clear();
        return;
    }
    if (N == 1)
    {
        out.assign(1, std::abs(signal[0]));
        return;
    }

    buf.resize(static_cast<size_t>(N));
    out.resize(static_cast<size_t>(N));

    for (int i = 0; i < N; ++i)
    {
        buf[i][0] = signal[i];
        buf[i][1] = 0.0f;
    }

    auto *fbuf = reinterpret_cast<fftwf_complex *>(buf.data());
    fft_forward(fbuf, N);

    // Analytic-signal filter: double positive frequencies, zero negative ones,
    // keep DC (and Nyquist when N is even).
    //   Even N: positive bins are 1..N/2-1, Nyquist is N/2, negatives N/2+1..N-1.
    //   Odd  N: positive bins are 1..(N-1)/2, no Nyquist, negatives (N+1)/2..N-1.
    // (The previous loop bounded by N/2 left one positive bin undoubled and one
    // negative bin non-zeroed for odd N.)
    for (int k = 1; k < (N + 1) / 2; ++k)
    {
        buf[k][0] *= 2.0f;
        buf[k][1] *= 2.0f;
    }
    for (int k = N / 2 + 1; k < N; ++k)
    {
        buf[k][0] = 0.0f;
        buf[k][1] = 0.0f;
    }

    fft_inverse(fbuf, N);

    const float invN = 1.0f / float(N);
    for (int i = 0; i < N; ++i)
    {
        const float re = buf[i][0] * invN;
        const float im = buf[i][1] * invN;
        out[i] = std::hypot(re, im);
    }
}

// ---------------------------------------------------------
// Bandpass-filter a real signal in place.
//
// Algorithm:
//   1. Copy real signal into complex buffer (imaginary = 0).
//   2. Forward FFT.
//   3. Multiply each bin by a Hann weight centred at |f| = f_center
//      with full bandwidth bw.  The weight is applied symmetrically to
//      positive and negative frequencies so the filtered signal is real.
//   4. Inverse FFT and write the real part back to signal[].
//
// The Hann window gives a smooth transition at the passband edges,
// which avoids time-domain ringing (Gibbs phenomenon) that a rectangular
// bandpass would introduce on a short signal like a CSI Z-trace (~100–500
// samples).
// ---------------------------------------------------------
void HilbertEnvelope::bandpass_inplace(std::vector<float> &signal,
                                       ComplexBuf &buf,
                                       float f_center,
                                       float bw) const
{
    const int N = static_cast<int>(signal.size());
    if (N <= 1 || f_center <= 0.0f || bw <= 0.0f)
        return; // nothing to filter

    ensure_plans(N);
    buf.resize(static_cast<size_t>(N));

    // Step 1: load real signal into complex buffer
    for (int i = 0; i < N; ++i)
    {
        buf[i][0] = signal[i];
        buf[i][1] = 0.0f;
    }

    // Step 2: forward FFT
    auto *fb = reinterpret_cast<fftwf_complex *>(buf.data());
    fft_forward(fb, N);

    // Step 3: apply Hann bandpass window
    // For bin k, the corresponding signed frequency is:
    //   f = k/N          for k in [0, N/2]   (positive frequencies)
    //   f = (k-N)/N      for k in [N/2+1, N-1] (negative frequencies)
    // We test distance from |f| = f_center; bins outside bw/2 are zeroed,
    // bins inside get a Hann weight that is 1.0 at the centre and tapers
    // smoothly to 0 at the edges.
    const float half_bw = bw * 0.5f;
    for (int k = 0; k < N; ++k)
    {
        // Signed normalised frequency in [-0.5, 0.5]
        const float f_signed = (k <= N / 2) ? float(k) / float(N)
                                            : float(k - N) / float(N);
        // Distance from the nearest centre (|f| = f_center handles both
        // positive and negative mirror symmetrically)
        const float dist = std::abs(std::abs(f_signed) - f_center);

        float w;
        if (dist >= half_bw)
        {
            w = 0.0f; // outside passband: zero the bin
        }
        else
        {
            // Hann weight: 1.0 at centre, 0.0 at band edges
            w = 0.5f * (1.0f + std::cos(float(M_PI) * dist / half_bw));
        }

        buf[k][0] *= w;
        buf[k][1] *= w;
    }

    // Step 4: inverse FFT and write real part back
    fft_inverse(fb, N);
    const float invN = 1.0f / float(N);
    for (int i = 0; i < N; ++i)
        signal[i] = buf[i][0] * invN;
    // Imaginary residuals (numerical noise) are discarded: the symmetric
    // Hann filter guarantees a real output for a real input.
}

// ---------------------------------------------------------
// Fourier-filter envelope  (PDF Method 2)
// ---------------------------------------------------------
void HilbertEnvelope::compute_fft_bandpass(const std::vector<float> &signal,
                                           ComplexBuf               &buf,
                                           std::vector<float>       &out) const
{
    const int N = static_cast<int>(signal.size());
    if (N == 0) { out.clear(); return; }
    if (N <= 2)  { out.assign(static_cast<size_t>(N), 0.0f); return; }

    ensure_plans(N);
    buf.resize(static_cast<size_t>(N));

    // 1. Load real signal into complex buffer
    for (int i = 0; i < N; ++i) { buf[i][0] = signal[i]; buf[i][1] = 0.0f; }

    // 2. Forward FFT
    auto *fb = reinterpret_cast<fftwf_complex *>(buf.data());
    fft_forward(fb, N);

    // 3. Zero DC and all negative-frequency bins (keep k = 1 .. N/2)
    buf[0][0] = buf[0][1] = 0.0f;
    for (int k = N / 2 + 1; k < N; ++k)
        buf[k][0] = buf[k][1] = 0.0f;
    // k = N/2 (Nyquist) is kept

    // 4. Find k_avg = argmax |F(k)| and accumulate spectral moments for Δk.
    //    Both passes are merged into one loop to avoid iterating twice.
    double sum_A = 0.0, sum_Ak = 0.0, sum_Ak2 = 0.0;
    int k_avg = 1;
    float max_A = -1.0f;

    for (int k = 1; k <= N / 2; ++k)
    {
        const float A = std::hypot(buf[k][0], buf[k][1]);
        if (A > max_A) { max_A = A; k_avg = k; }
        const double Ad = double(A);
        sum_A   += Ad;
        sum_Ak  += Ad * k;
        sum_Ak2 += Ad * k * k;
    }

    // 5. Δk = 2σ of the spectral amplitude distribution, centred on k_avg.
    //    The sum runs over the full positive half-spectrum (k = 1..N/2).
    //    In practice the noise floor keeps Δk large (often ≈ N/4 fallback),
    //    which makes the rectangular window below keep nearly all positive
    //    frequencies — equivalent to the classical Hilbert transform.  A
    //    narrower local window for the variance was tried but produced
    //    height artefacts on flat surfaces (centre of gold flakes appeared
    //    displaced relative to edges) because the resulting narrow bandpass
    //    broadens the coherence envelope asymmetrically between pixels with
    //    different spectral noise floors.
    //    Using the parallel-axis identity:
    //      Var(k − k_avg) = E[k²] − 2·k_avg·E[k] + k_avg²
    int dk = std::max(1, N / 4);   // wide fallback
    if (sum_A > 1e-30)
    {
        const double var = sum_Ak2 / sum_A
                         - 2.0 * double(k_avg) * (sum_Ak / sum_A)
                         + double(k_avg) * double(k_avg);
        dk = std::max(1, static_cast<int>(std::ceil(2.0 * std::sqrt(std::max(0.0, var)))));
    }

    // 6. Rectangular bandpass: zero bins with |k − k_avg| > dk.
    //    A Hann-windowed taper is NOT used here despite eliminating Gibbs ringing
    //    in the spectrum, because for the one-sided analytic signal the Hann
    //    window introduces phase-dependent bias in |IFFT|: the window decomposes
    //    into three frequency-shifted copies of the analytic signal whose phases
    //    vary per-pixel, shifting the centroid differently for flat vs. edge pixels.
    //    The rectangular window's ringing is at 2·f₀ and is suppressed by the
    //    subsequent Gaussian smoothing (σ=15, which covers ≥4 periods of 2·f₀).
    for (int k = 1; k <= N / 2; ++k)
    {
        if (std::abs(k - k_avg) > dk)
            buf[k][0] = buf[k][1] = 0.0f;
    }

    // 7. Inverse FFT; Env(z) = 2 × |IFFT| / N
    fft_inverse(fb, N);
    out.resize(static_cast<size_t>(N));
    const float scale = 2.0f / float(N);
    for (int i = 0; i < N; ++i)
        out[i] = scale * std::hypot(buf[i][0], buf[i][1]);
}

// ---------------------------------------------------------
// Fixed-params overload: apply a pre-computed bandpass to one signal.
// ---------------------------------------------------------
void HilbertEnvelope::compute_fft_bandpass(const std::vector<float> &signal,
                                           ComplexBuf               &buf,
                                           std::vector<float>       &out,
                                           int k_avg, int dk) const
{
    const int N = static_cast<int>(signal.size());
    if (N == 0) { out.clear(); return; }
    if (N <= 2)  { out.assign(static_cast<size_t>(N), 0.0f); return; }

    ensure_plans(N);
    buf.resize(static_cast<size_t>(N));

    for (int i = 0; i < N; ++i) { buf[i][0] = signal[i]; buf[i][1] = 0.0f; }

    auto *fb = reinterpret_cast<fftwf_complex *>(buf.data());
    fft_forward(fb, N);

    buf[0][0] = buf[0][1] = 0.0f;
    for (int k = N / 2 + 1; k < N; ++k)
        buf[k][0] = buf[k][1] = 0.0f;

    for (int k = 1; k <= N / 2; ++k)
    {
        if (std::abs(k - k_avg) > dk)
            buf[k][0] = buf[k][1] = 0.0f;
    }

    fft_inverse(fb, N);
    out.resize(static_cast<size_t>(N));
    const float scale = 2.0f / float(N);
    for (int i = 0; i < N; ++i)
        out[i] = scale * std::hypot(buf[i][0], buf[i][1]);
}

// ---------------------------------------------------------
// Estimate bandpass params from amplitude-averaged spectrum.
// ---------------------------------------------------------
std::pair<int,int> HilbertEnvelope::estimate_bandpass_params(
    const std::vector<std::vector<float>> &signals,
    ComplexBuf &scratch)
{
    if (signals.empty()) return {1, 1};

    const int N = static_cast<int>(signals[0].size());
    if (N <= 2) return {1, 1};

    ensure_plans(N);
    scratch.resize(static_cast<size_t>(N));
    auto *fb = reinterpret_cast<fftwf_complex *>(scratch.data());

    // Accumulate amplitude spectrum; averaging reduces noise relative to
    // any individual pixel's spectrum and gives a stable k_avg and dk.
    std::vector<double> avg_amp(static_cast<size_t>(N / 2 + 1), 0.0);
    int n_valid = 0;

    for (const auto &sig : signals)
    {
        if (static_cast<int>(sig.size()) != N) continue;
        for (int i = 0; i < N; ++i) { scratch[i][0] = sig[i]; scratch[i][1] = 0.0f; }
        fft_forward(fb, N);
        for (int k = 1; k <= N / 2; ++k)
            avg_amp[k] += double(std::hypot(scratch[k][0], scratch[k][1]));
        ++n_valid;
    }

    if (n_valid == 0) return {1, std::max(1, N / 4)};
    for (auto &a : avg_amp) a /= n_valid;

    int k_avg = 1;
    double max_A = -1.0;
    double sum_A = 0.0, sum_Ak = 0.0, sum_Ak2 = 0.0;

    for (int k = 1; k <= N / 2; ++k)
    {
        const double A = avg_amp[k];
        if (A > max_A) { max_A = A; k_avg = k; }
        sum_A   += A;
        sum_Ak  += A * k;
        sum_Ak2 += A * k * k;
    }

    int dk = std::max(1, N / 4);
    if (sum_A > 1e-30)
    {
        const double var = sum_Ak2 / sum_A
                         - 2.0 * double(k_avg) * (sum_Ak / sum_A)
                         + double(k_avg) * double(k_avg);
        dk = std::max(1, static_cast<int>(std::ceil(2.0 * std::sqrt(std::max(0.0, var)))));
    }

    return {k_avg, dk};
}

// ---------------------------------------------------------
// Forward FFT — fills buf with the unnormalized complex DFT
// ---------------------------------------------------------
void HilbertEnvelope::forward_fft(const std::vector<float> &signal,
                                   ComplexBuf               &buf) const
{
    const int N = static_cast<int>(signal.size());
    ensure_plans(N);
    buf.resize(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
    {
        buf[i][0] = signal[i];
        buf[i][1] = 0.0f;
    }
    fft_forward(reinterpret_cast<fftwf_complex *>(buf.data()), N);
}

// ---------------------------------------------------------
// Compute Hilbert envelope (allocating convenience wrapper)
// ---------------------------------------------------------
std::vector<float> HilbertEnvelope::compute(const std::vector<float> &signal) const
{
    ComplexBuf buf;
    std::vector<float> out;
    compute_into(signal, buf, out);
    return out;
}
