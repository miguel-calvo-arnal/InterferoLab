#include "filters.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

std::vector<float> make_gaussian_kernel(double sigma)
{
    if (sigma <= 0.0)
        return {};

    const int radius = std::max(1, static_cast<int>(std::round(3.0 * sigma)));
    const int size   = 2 * radius + 1;

    std::vector<float> kernel(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i)
    {
        const float x = static_cast<float>(i - radius);
        kernel[static_cast<size_t>(i)] =
            std::expf(-(x * x) / (2.0f * static_cast<float>(sigma * sigma)));
    }

    const float sum = std::accumulate(kernel.begin(), kernel.end(), 0.0f);
    if (sum == 0.0f)
        return {};
    for (auto &v : kernel)
        v /= sum;

    return kernel;
}

void gaussian_apply(const std::vector<float> &signal,
                    const std::vector<float> &kernel,
                    std::vector<float> &out)
{
    const size_t N = signal.size();

    if (kernel.empty() || N <= 2)
    {
        out.assign(signal.begin(), signal.end());
        return;
    }

    out.resize(N);

    const int size   = static_cast<int>(kernel.size());
    const int radius = size / 2;

    // Iterative mirror reflection: valid for ANY offset, not only |offset| < N.
    // The single-bounce version was only correct for radius <= N-1; with short
    // stacks (N <= kernel radius, e.g. 20 frames vs radius 45 at sigma=15) it
    // produced negative indices -> out-of-range reads.  Iterating the two
    // bounces converges for every j when N >= 2 (guaranteed by the N <= 2
    // early-out above) and is bit-identical to the old code when radius <= N-1.
    auto reflect = [N](long j) -> long
    {
        const long n = static_cast<long>(N);
        while (j < 0 || j >= n)
        {
            if (j < 0)
                j = -j;
            if (j >= n)
                j = 2 * n - j - 2;
        }
        return j;
    };

    // ------------------------------------------------------------------
    // Vectorisable convolution (O-1).
    //
    // The historical loop called reflect() on every tap, which blocks
    // autovectorisation for ALL outputs.  Instead, build a reflect-padded
    // copy of the signal once:
    //
    //     ext[m] = signal[reflect(m - radius)],   m in [0, N + 2*radius)
    //
    // (the pads use the same iterative reflect as before — valid for any
    // radius, including radius >= N), so every output becomes a plain dot
    // product  out[i] = sum_k ext[i+k] * kernel[k]  over contiguous memory.
    //
    // NOTE on floating-point semantics: `#pragma omp simd reduction(+:acc)`
    // licenses the compiler to reassociate THIS reduction only (OpenMP simd
    // semantics) — no -ffast-math / -fassociative-math is enabled globally,
    // so isfinite()/NaN handling elsewhere stays intact.  The products and
    // their multiset are identical to the sequential k=-radius..radius sum;
    // only the addition order changes, which moves low-order bits (measured
    // max rel. diff ~1e-6 on real datasets; heightmap tolerance documented
    // in the Bloque C verification).
    //
    // The scratch pad is thread_local: gaussian_apply is called from inside
    // OpenMP loops, and this keeps one allocation per thread in steady state.
    // ------------------------------------------------------------------
    const size_t r = static_cast<size_t>(radius);

    thread_local std::vector<float> ext;
    ext.resize(N + 2 * r);

    for (size_t j = 0; j < r; ++j)
        ext[j] = signal[static_cast<size_t>(
            reflect(static_cast<long>(j) - static_cast<long>(r)))];
    std::copy(signal.begin(), signal.end(), ext.begin() + static_cast<long>(r));
    for (size_t j = 0; j < r; ++j)
        ext[r + N + j] = signal[static_cast<size_t>(
            reflect(static_cast<long>(N + j)))];

    const float *ker = kernel.data();
    const float *base = ext.data();
    float *dst = out.data();
    for (size_t i = 0; i < N; ++i)
    {
        const float *s = base + i;
        float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
        for (int k = 0; k < size; ++k)
            acc += s[k] * ker[k];
        dst[i] = acc;
    }
}

std::vector<float> gaussian_filter_1d(const std::vector<float> &signal, double sigma)
{
    const auto kernel = make_gaussian_kernel(sigma);
    std::vector<float> out;
    gaussian_apply(signal, kernel, out);
    return out;
}
