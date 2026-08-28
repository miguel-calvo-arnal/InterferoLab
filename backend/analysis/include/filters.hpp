#pragma once
#include <vector>

// Build a normalized Gaussian kernel for the given sigma.
// Returns an empty vector when sigma <= 0.
std::vector<float> make_gaussian_kernel(double sigma);

// Apply a pre-computed kernel to signal; writes result into out (resized as needed).
// Uses reflect padding identical to gaussian_filter_1d.
// No-op (copies signal into out) if kernel is empty.
void gaussian_apply(const std::vector<float> &signal,
                    const std::vector<float> &kernel,
                    std::vector<float> &out);

// Convenience wrapper: builds kernel and applies it in one call.
// Prefer make_gaussian_kernel + gaussian_apply in hot loops.
std::vector<float> gaussian_filter_1d(const std::vector<float> &signal, double sigma);