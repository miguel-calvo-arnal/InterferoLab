// ============================================================================
//   STREAMING RECONSTRUCTION (NO FULL 3‑D STACK IN MEMORY)
//   ---------------------------------------------------------------
//   - Supports TIFF (libtiff scanlines), packed 12-bit .bin12 and PNG inputs.
//   - Reads only the required scanlines from each file (to control RAM usage).
//   - Sorts the Z‑axis once (z_order) based on physical positions
//      This is just in case, since when the piezo position was not stable,
//      they could get out of order.
//   - Removes the signal baseline with Gaussian filter, for the Hilbert transform
//      to work fine.
//   - Computes Hilbert envelope + Gaussian smoothing of the envelope.
//   - Locates the envelope peak -> physical height.
//   - Allows extracting the raw signal/envelope of a single pixel without
//     loading the full 3‑D stack.
// ============================================================================

#include <tiffio.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <regex>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <thread>
#include <chrono>

#include "image_stack.hpp"
#include "hilbert.hpp"
#include "filters.hpp"
#include "npy.hpp"
#include "config.hpp"
#include "analysis_api.hpp"
#include "utils.hpp"
#include "methods.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

// ============================================================================
// RAII wrappers for C file handles
// ---------------------------------------------------------------------------
// Every FILE*/TIFF* in this translation unit is owned by one of these
// unique_ptrs, so the handle is released on every path — including exceptions
// thrown mid-read and early returns on cancellation.
// ============================================================================
struct FileCloser
{
    void operator()(FILE *f) const noexcept
    {
        if (f)
            std::fclose(f);
    }
};
using FilePtr = std::unique_ptr<FILE, FileCloser>;

struct TiffCloser
{
    void operator()(TIFF *t) const noexcept
    {
        if (t)
            TIFFClose(t);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

// ============================================================================
// 64-bit file positioning (std::fseek takes a 'long', which is 32-bit on
// Windows: files beyond 2 GiB would seek to the wrong offset).
// ============================================================================
static int fseek64(FILE *f, uint64_t offset)
{
#if defined(_WIN32)
    return _fseeki64(f, static_cast<long long>(offset), SEEK_SET);
#else
    return fseeko(f, static_cast<off_t>(offset), SEEK_SET);
#endif
}

static int64_t file_size_64(FILE *f)
{
#if defined(_WIN32)
    if (_fseeki64(f, 0, SEEK_END) != 0)
        return -1;
    return _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0)
        return -1;
    return static_cast<int64_t>(ftello(f));
#endif
}

// ============================================================================
// RAII joiner: guarantees a std::thread is never destroyed while joinable.
// If the scope unwinds due to an exception (e.g. a progress callback that
// throws), the destructor requests cancellation so the worker's OpenMP loop
// terminates quickly, then joins.  Without this, destroying a joinable
// std::thread calls std::terminate and kills the whole host process.
// ============================================================================
struct ThreadJoiner
{
    std::thread &t;
    explicit ThreadJoiner(std::thread &thread) : t(thread) {}
    ~ThreadJoiner()
    {
        if (t.joinable())
        {
            request_cancel(true);
            t.join();
        }
    }
    ThreadJoiner(const ThreadJoiner &) = delete;
    ThreadJoiner &operator=(const ThreadJoiner &) = delete;
};

// ============================================================================
// Packed 12-bit binary format helpers  (.bin12)
// ---------------------------------------------------------------------------
// Header layout (20 bytes, little-endian):
//   [0..3]   Magic    "B12\0"
//   [4..7]   Width    (uint32)
//   [8..11]  Height   (uint32)
//   [12..15] Bit depth (uint32, always 12)
//   [16..19] Channels  (uint32: 1 = mono, 3 = RGB)
//
// Data layout — row-interleaved, planar per row:
//   For each row y (0..Height-1):
//     [ ch0 packed: Width*3/2 bytes ]
//     [ ch1 packed: Width*3/2 bytes ]   (only when channels == 3)
//     [ ch2 packed: Width*3/2 bytes ]   (only when channels == 3)
//   Channels are stored in R, G, B order (indices match channel_weights).
//
// Pixel packing (each channel independently, 2 pixels → 3 bytes):
//   byte0 = p0[7:0]
//   byte1 = p0[11:8] | (p1[3:0] << 4)
//   byte2 = p1[11:4]
// Image width must be even.
// ============================================================================
struct Bin12Header
{
    uint32_t width;
    uint32_t height;
    uint32_t bit_depth;
    uint32_t channels; // 1 (mono) or 3 (RGB)
};

static constexpr size_t BIN12_HEADER_SIZE = 20;
static constexpr uint8_t BIN12_MAGIC[4] = {'B', '1', '2', '\0'};

static Bin12Header open_bin12(FILE *f, const std::string &path)
{
    uint8_t hdr[BIN12_HEADER_SIZE];
    if (std::fread(hdr, 1, BIN12_HEADER_SIZE, f) != BIN12_HEADER_SIZE)
        throw std::runtime_error("Truncated bin12 header: " + path);
    if (std::memcmp(hdr, BIN12_MAGIC, 4) != 0)
        throw std::runtime_error("Invalid bin12 magic in: " + path);
    Bin12Header h;
    std::memcpy(&h.width, hdr + 4, 4);
    std::memcpy(&h.height, hdr + 8, 4);
    std::memcpy(&h.bit_depth, hdr + 12, 4);
    std::memcpy(&h.channels, hdr + 16, 4);

    // ---- Header sanity checks (a corrupted header must never drive reads) --
    if (h.bit_depth != 12)
        throw std::runtime_error("bin12: unsupported bit depth "
                                 + std::to_string(h.bit_depth) + " in: " + path);
    if (h.channels != 1 && h.channels != 3)
        throw std::runtime_error("bin12: invalid channel count "
                                 + std::to_string(h.channels)
                                 + " (must be 1 or 3) in: " + path);
    if (h.width == 0 || h.height == 0)
        throw std::runtime_error("bin12: zero image dimensions in: " + path);
    if (h.width % 2 != 0)
        throw std::runtime_error("bin12: image width must be even in: " + path);

    // ---- File size must hold the full payload the header promises ----------
    const uint64_t expected = BIN12_HEADER_SIZE
        + uint64_t(h.width) * 3 / 2 * uint64_t(h.channels) * uint64_t(h.height);
    const int64_t actual = file_size_64(f);
    if (actual < 0 || uint64_t(actual) < expected)
        throw std::runtime_error(
            "bin12: file truncated (" + std::to_string(actual) + " bytes, expected "
            + std::to_string(expected) + "): " + path);
    if (fseek64(f, BIN12_HEADER_SIZE) != 0)
        throw std::runtime_error("bin12: seek failed after header: " + path);

    return h;
}

// Unpack one row of packed 12-bit data into float.  Nx must be even.
static void unpack_bin12_row(const uint8_t *packed, int Nx, float *dst)
{
    for (int x = 0; x < Nx; x += 2)
    {
        int k = (x / 2) * 3;
        uint16_t p0 = uint16_t(packed[k]) | (uint16_t(packed[k + 1] & 0x0F) << 8);
        uint16_t p1 = (uint16_t(packed[k + 1]) >> 4) | (uint16_t(packed[k + 2]) << 4);
        dst[x] = float(p0);
        dst[x + 1] = float(p1);
    }
}

// Decode one packed bin12 row (all channels) into Nx floats, applying the
// channel weights for RGB files.  Pure CPU work — no file I/O — so it can be
// called from parallel readers.  The RGB scratch buffer is thread_local
// (one allocation per thread in steady state).
static void decode_bin12_row(
    const uint8_t *packed, const Bin12Header &hdr, int Nx, float *dst,
    const std::array<double, 3> &channel_weights)
{
    if (hdr.channels == 1)
    {
        unpack_bin12_row(packed, Nx, dst);
        return;
    }

    // 3-channel RGB: ch0=R, ch1=G, ch2=B (matches channel_weights indices)
    const size_t ch_row_bytes = size_t(Nx) * 3 / 2;
    const double raw_sum = channel_weights[0] + channel_weights[1] + channel_weights[2];
    const double WR = raw_sum > 0.0 ? channel_weights[0] / raw_sum : 1.0 / 3.0;
    const double WG = raw_sum > 0.0 ? channel_weights[1] / raw_sum : 1.0 / 3.0;
    const double WB = raw_sum > 0.0 ? channel_weights[2] / raw_sum : 1.0 / 3.0;
    const double W[3] = {WR, WG, WB};

    std::fill(dst, dst + Nx, 0.0f);
    thread_local std::vector<float> ch_buf;
    ch_buf.resize(static_cast<size_t>(Nx));
    for (int c = 0; c < 3; ++c)
    {
        unpack_bin12_row(packed + size_t(c) * ch_row_bytes, Nx, ch_buf.data());
        for (int x = 0; x < Nx; ++x)
            dst[x] += float(W[c]) * ch_buf[x];
    }
}

static void read_one_scanline_from_bin12(
    FILE *f, const Bin12Header &hdr, uint32_t row, int Nx, float *dst,
    const std::array<double, 3> &channel_weights)
{
    if (Nx % 2 != 0)
        throw std::runtime_error("bin12: image width must be even");

    const size_t ch_row_bytes = size_t(Nx) * 3 / 2;
    const size_t row_stride = ch_row_bytes * size_t(hdr.channels);
    const size_t row_start = BIN12_HEADER_SIZE + size_t(row) * row_stride;

    // Hoisted out of the per-row hot path (O-4): thread_local keeps a single
    // allocation per thread instead of one heap round-trip per row.
    thread_local std::vector<uint8_t> buf;
    buf.resize(row_stride);
    if (fseek64(f, row_start) != 0)
        throw std::runtime_error("bin12 fseek failed for row " + std::to_string(row));
    if (std::fread(buf.data(), 1, row_stride, f) != row_stride)
        throw std::runtime_error("bin12 fread failed for row " + std::to_string(row));

    decode_bin12_row(buf.data(), hdr, Nx, dst, channel_weights);
}

// ============================================================================
// read_rows_block_from_bin12()  (O-4)
// ---------------------------------------------------------------------------
// Reads `nrows` CONSECUTIVE image rows [row0, row0+nrows) with a single
// fseek+fread (bin12 rows are stored consecutively), then decodes them into
// dst0 + r*dst_stride.  Replaces the former per-row fseek+fread+alloc loop
// of the chunk reader (1.9x measured on the read path in the audit).
// ============================================================================
static void read_rows_block_from_bin12(
    FILE *f, const Bin12Header &hdr, uint32_t row0, int nrows, int Nx,
    float *dst0, size_t dst_stride,
    const std::array<double, 3> &channel_weights)
{
    if (Nx % 2 != 0)
        throw std::runtime_error("bin12: image width must be even");

    const size_t ch_row_bytes = size_t(Nx) * 3 / 2;
    const size_t row_stride = ch_row_bytes * size_t(hdr.channels);
    const uint64_t block_start = BIN12_HEADER_SIZE + uint64_t(row0) * row_stride;
    const size_t block_bytes = row_stride * size_t(nrows);

    thread_local std::vector<uint8_t> block;
    block.resize(block_bytes);

    if (fseek64(f, block_start) != 0)
        throw std::runtime_error(
            "bin12 fseek failed for row block at " + std::to_string(row0));
    if (std::fread(block.data(), 1, block_bytes, f) != block_bytes)
        throw std::runtime_error(
            "bin12 fread failed for row block at " + std::to_string(row0));

    for (int r = 0; r < nrows; ++r)
    {
        if (cancel_requested())
            return;
        decode_bin12_row(block.data() + size_t(r) * row_stride, hdr, Nx,
                         dst0 + size_t(r) * dst_stride, channel_weights);
    }
}

// Returns true if 'path' has a .bin12 extension (case-insensitive).
static bool is_bin12_file(const std::string &path)
{
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });
    return ext == ".bin12";
}

// Returns true if 'path' has a .png extension (case-insensitive).
static bool is_png_file(const std::string &path)
{
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });
    return ext == ".png";
}

// ============================================================================
// read_png_row_to_float()
// ---------------------------------------------------------------------------
// Reads one row from a cv::Mat loaded with cv::IMREAD_UNCHANGED|ANYDEPTH.
// - Mono images (channels=1): copy directly to float.
// - Color images (channels>=3): OpenCV stores as BGR.
//   channel_weights = [R, G, B], so the BGR index mapping is:
//     WR (weights[0]) → BGR[2],  WG (weights[1]) → BGR[1],  WB (weights[2]) → BGR[0]
// - Supports CV_8U, CV_16U, CV_32F sample depths.
// ============================================================================
static void read_png_row_to_float(
    const cv::Mat &img, int row, int Nx, float *dst,
    const std::array<double, 3> &channel_weights)
{
    const int ch = img.channels();
    const int depth = img.depth();

    // Normalize weights (same logic as TIFF reader)
    const double raw_sum = channel_weights[0] + channel_weights[1] + channel_weights[2];
    // OpenCV BGR order: index 0=B, 1=G, 2=R
    const double WR = raw_sum > 0.0 ? channel_weights[0] / raw_sum : 1.0 / 3.0;
    const double WG = raw_sum > 0.0 ? channel_weights[1] / raw_sum : 1.0 / 3.0;
    const double WB = raw_sum > 0.0 ? channel_weights[2] / raw_sum : 1.0 / 3.0;

    const uint8_t *row_ptr = img.ptr(row);

    if (ch == 1)
    {
        // ---- Grayscale ----
        if (depth == CV_8U)
        {
            for (int x = 0; x < Nx; ++x)
                dst[x] = float(row_ptr[x]);
        }
        else if (depth == CV_16U)
        {
            const uint16_t *u16 = reinterpret_cast<const uint16_t *>(row_ptr);
            for (int x = 0; x < Nx; ++x)
                dst[x] = float(u16[x]);
        }
        else if (depth == CV_32F)
        {
            std::memcpy(dst, row_ptr, size_t(Nx) * sizeof(float));
        }
        else
        {
            throw std::runtime_error("read_png_row_to_float: unsupported grayscale depth");
        }
    }
    else if (ch >= 3)
    {
        // ---- BGR color ----
        if (depth == CV_8U)
        {
            for (int x = 0; x < Nx; ++x)
            {
                const uint8_t *p = row_ptr + x * ch;
                dst[x] = float(WB * p[0] + WG * p[1] + WR * p[2]);
            }
        }
        else if (depth == CV_16U)
        {
            const uint16_t *u16 = reinterpret_cast<const uint16_t *>(row_ptr);
            for (int x = 0; x < Nx; ++x)
            {
                const uint16_t *p = u16 + x * ch;
                dst[x] = float(WB * p[0] + WG * p[1] + WR * p[2]);
            }
        }
        else if (depth == CV_32F)
        {
            const float *f32 = reinterpret_cast<const float *>(row_ptr);
            for (int x = 0; x < Nx; ++x)
            {
                const float *p = f32 + x * ch;
                dst[x] = float(WB * p[0] + WG * p[1] + WR * p[2]);
            }
        }
        else
        {
            throw std::runtime_error("read_png_row_to_float: unsupported color depth");
        }
    }
    else
    {
        throw std::runtime_error(
            "read_png_row_to_float: unsupported channel count " + std::to_string(ch));
    }
}

// ============================================================================
// read_one_scanline_to_float()
// ---------------------------------------------------------------------------
// Reads a full TIFF scanline using libtiff and converts it to Nx floats.
// - Uses TIFFScanlineSize() -> real row size (avoids overflows or incorrect
//   handling of TIFFs with padding or multiple channels).
// - Supports TIFFs with: 8/16/32‑bit integer and 32‑bit float.
// - Supports photometric types: MINISBLACK (greyscale) and RGB.
// - spp == 1 -> direct grayscale copy.
// - spp >= 3 -> convert to grayscale using the provided channel_weights.
// - PLANARCONFIG_CONTIG: interleaved channels.
// - PLANARCONFIG_SEPARATE: separate R,G,B planes.
// - No normalisation is performed (same as convertTo(CV_32F)).
// ============================================================================
static inline void read_one_scanline_to_float(
    TIFF *tif, uint32_t row, int Nx, float *dst,
    const std::array<double, 3> &channel_weights)
{
    // Required TIFF tags
    uint16_t bps = 0, spp = 1, sample_format = SAMPLEFORMAT_UINT;
    uint16_t planar = PLANARCONFIG_CONTIG;
    uint16_t photometric = PHOTOMETRIC_MINISBLACK;

    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);
    TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);

    const bool separate = (planar == PLANARCONFIG_SEPARATE);

    // RGB -> grayscale weights: normalize so they sum to 1
    const double raw_sum = channel_weights[0] + channel_weights[1] + channel_weights[2];
    const double WR = raw_sum > 0.0 ? channel_weights[0] / raw_sum : 1.0 / 3.0;
    const double WG = raw_sum > 0.0 ? channel_weights[1] / raw_sum : 1.0 / 3.0;
    const double WB = raw_sum > 0.0 ? channel_weights[2] / raw_sum : 1.0 / 3.0;

    // ---------------------------
    // CASE 1: GRAYSCALE SPP = 1
    // ---------------------------
    if (photometric == PHOTOMETRIC_MINISBLACK && spp == 1)
    {
        const tmsize_t scanline_bytes = TIFFScanlineSize(tif);
        const std::size_t nbytes = static_cast<std::size_t>(scanline_bytes);
        // thread_local: one allocation per thread instead of one per row (O-4)
        thread_local std::vector<uint8_t> rowbuf;
        rowbuf.resize(nbytes);

        if (TIFFReadScanline(tif, rowbuf.data(), row, 0) != 1)
            throw std::runtime_error("TIFFReadScanline (gray) failed");

        const uint8_t *p = rowbuf.data();

        // Different bit depths
        if (sample_format == SAMPLEFORMAT_IEEEFP && bps == 32)
        {
            const float *f = reinterpret_cast<const float *>(p);
            std::memcpy(dst, f, Nx * sizeof(float));
            return;
        }
        if (bps == 8)
        {
            const uint8_t *u8 = p;
            for (int x = 0; x < Nx; ++x)
                dst[x] = float(u8[x]);
            return;
        }
        if (bps == 16)
        {
            const uint16_t *u16 = reinterpret_cast<const uint16_t *>(p);
            for (int x = 0; x < Nx; ++x)
                dst[x] = float(u16[x]);
            return;
        }
        if (bps == 32 && sample_format != SAMPLEFORMAT_IEEEFP)
        {
            const uint32_t *u32 = reinterpret_cast<const uint32_t *>(p);
            for (int x = 0; x < Nx; ++x)
                dst[x] = float(u32[x]);
            return;
        }

        throw std::runtime_error("Unsupported grayscale TIFF format");
    }

    // ---------------------------
    // CASE 2: RGB, spp >= 3
    // ---------------------------
    if (photometric == PHOTOMETRIC_RGB && spp >= 3)
    {
        // =============== SEPARATE PLANES ===============
        if (separate)
        {
            const tmsize_t scanline_bytes = TIFFScanlineSize(tif);
            const std::size_t nbytes = static_cast<std::size_t>(scanline_bytes);

            // thread_local: one allocation per thread instead of three per row
            thread_local std::vector<uint8_t> rowR, rowG, rowB;
            rowR.resize(nbytes);
            rowG.resize(nbytes);
            rowB.resize(nbytes);

            if (TIFFReadScanline(tif, rowR.data(), row, 0) != 1 ||
                TIFFReadScanline(tif, rowG.data(), row, 1) != 1 ||
                TIFFReadScanline(tif, rowB.data(), row, 2) != 1)
                throw std::runtime_error("TIFFReadScanline (separate RGB) failed");

            if (sample_format == SAMPLEFORMAT_IEEEFP && bps == 32)
            {
                const float *R = reinterpret_cast<const float *>(rowR.data());
                const float *G = reinterpret_cast<const float *>(rowG.data());
                const float *B = reinterpret_cast<const float *>(rowB.data());
                for (int x = 0; x < Nx; ++x)
                    dst[x] = float(WR * R[x] + WG * G[x] + WB * B[x]);
                return;
            }

            if (bps == 8)
            {
                const uint8_t *R = rowR.data(), *G = rowG.data(), *B = rowB.data();
                for (int x = 0; x < Nx; ++x)
                    dst[x] = float(WR * R[x] + WG * G[x] + WB * B[x]);
                return;
            }

            if (bps == 16)
            {
                const uint16_t *R = reinterpret_cast<const uint16_t *>(rowR.data());
                const uint16_t *G = reinterpret_cast<const uint16_t *>(rowG.data());
                const uint16_t *B = reinterpret_cast<const uint16_t *>(rowB.data());
                for (int x = 0; x < Nx; ++x)
                    dst[x] = float(WR * R[x] + WG * G[x] + WB * B[x]);
                return;
            }

            if (bps == 32 && sample_format != SAMPLEFORMAT_IEEEFP)
            {
                const uint32_t *R = reinterpret_cast<const uint32_t *>(rowR.data());
                const uint32_t *G = reinterpret_cast<const uint32_t *>(rowG.data());
                const uint32_t *B = reinterpret_cast<const uint32_t *>(rowB.data());
                for (int x = 0; x < Nx; ++x)
                    dst[x] = float(WR * R[x] + WG * G[x] + WB * B[x]);
                return;
            }

            throw std::runtime_error("Unsupported separate RGB TIFF format");
        }

        // =============== CONTIG PLANES ===============
        const tmsize_t scanline_bytes = TIFFScanlineSize(tif);
        const std::size_t nbytes = static_cast<std::size_t>(scanline_bytes);
        // thread_local: one allocation per thread instead of one per row (O-4)
        thread_local std::vector<uint8_t> rowbuf;
        rowbuf.resize(nbytes);

        if (TIFFReadScanline(tif, rowbuf.data(), row, 0) != 1)
            throw std::runtime_error("TIFFReadScanline (contig RGB) failed");

        const uint8_t *p = rowbuf.data();

        if (sample_format == SAMPLEFORMAT_IEEEFP && bps == 32)
        {
            const float *f = reinterpret_cast<const float *>(p);
            for (int x = 0; x < Nx; ++x)
            {
                const float R = f[x * spp + 0];
                const float G = f[x * spp + 1];
                const float B = f[x * spp + 2];
                dst[x] = float(WR * R + WG * G + WB * B);
            }
            return;
        }

        if (bps == 8)
        {
            const uint8_t *u8 = p;
            for (int x = 0; x < Nx; ++x)
            {
                const float R = float(u8[x * spp + 0]);
                const float G = float(u8[x * spp + 1]);
                const float B = float(u8[x * spp + 2]);
                dst[x] = float(WR * R + WG * G + WB * B);
            }
            return;
        }

        if (bps == 16)
        {
            const uint16_t *u16 = reinterpret_cast<const uint16_t *>(p);
            for (int x = 0; x < Nx; ++x)
            {
                const float R = float(u16[x * spp + 0]);
                const float G = float(u16[x * spp + 1]);
                const float B = float(u16[x * spp + 2]);
                dst[x] = float(WR * R + WG * G + WB * B);
            }
            return;
        }

        if (bps == 32 && sample_format != SAMPLEFORMAT_IEEEFP)
        {
            const uint32_t *u32 = reinterpret_cast<const uint32_t *>(p);
            for (int x = 0; x < Nx; ++x)
            {
                const float R = float(u32[x * spp + 0]);
                const float G = float(u32[x * spp + 1]);
                const float B = float(u32[x * spp + 2]);
                dst[x] = float(WR * R + WG * G + WB * B);
            }
            return;
        }

        throw std::runtime_error("Unsupported (contig) RGB TIFF format");
    }

    // Other photometric types (YCbCr, CMYK, etc.) are not supported
    throw std::runtime_error("Unsupported photometric; only GRAY or RGB supported");
}

// ============================================================================
// idx3(): linear index inside Z‑major buffer
// ============================================================================
static inline size_t idx3(int z, int r, int x, int R, int Nx)
{
    return (size_t(z) * R + size_t(r)) * Nx + size_t(x);
}

static std::vector<float> parse_positions_from_filenames(
    const std::vector<std::string> &files,
    const std::regex &pattern)
{
    const int Nz = files.size();
    std::vector<float> positions(Nz);

    for (int i = 0; i < Nz; ++i)
    {
        if (cancel_requested())
            return {}; // cancellation -> empty result

        // Match against the FILENAME only, never the full path: the pattern
        // begins with a lazy .*?, so a number in any parent directory name
        // (e.g. data/scan_20um/) would win the search and silently give
        // every frame the same position.
        const std::string fname =
            std::filesystem::path(files[i]).filename().string();

        std::smatch m;
        if (!std::regex_search(fname, m, pattern) || m.size() < 2)
            throw std::runtime_error("Regex failed on: " + files[i]);

        std::stringstream ss(m[1].str()); // Wrap captured substring
        ss.imbue(std::locale::classic()); // Use C-locale for stable float parsing
        float val = 0.f;                  // Output variable
        ss >> val;                        // Convert substring -> float
        positions[i] = val;               // Store parsed physical Z-position
    }

    // Sanity check: identical positions across the whole stack mean the
    // pattern captured something other than the per-frame Z. Without this,
    // the max-p inversion zeroes every position and the reconstruction
    // produces a silently flat height map instead of a loud error.
    if (Nz > 1)
    {
        bool all_same = true;
        for (int i = 1; i < Nz; ++i)
            if (positions[i] != positions[0]) { all_same = false; break; }
        if (all_same)
            throw std::runtime_error(
                "All " + std::to_string(Nz) + " parsed Z positions are identical ("
                + std::to_string(positions[0])
                + "): the filename pattern did not capture the per-frame position");
    }

    return positions;
}

// ============================================================================
// centroid_with_refinement()
// ---------------------------------------------------------------------------
// Computes the weighted centroid (first moment) of a 1-D weight array over
// the corresponding physical positions.  A single refinement pass restricts
// the window to [Z_avg − ΔZ, Z_avg + ΔZ] where ΔZ = 2σ (second moment),
// matching the iterative scheme described in the PDF exercise for both
// Method 1 and Method 2.
// ============================================================================
[[nodiscard]] static float centroid_with_refinement(
    const std::vector<float> &weights,
    const std::vector<float> &positions)
{
    const int N = static_cast<int>(weights.size());

    // -- Single pass: accumulate centroid and second moment simultaneously ---
    double sum_w = 0.0, sum_wz = 0.0, sum_wz2 = 0.0;
    for (int i = 0; i < N; ++i)
    {
        const double w = double(weights[i]);
        if (w <= 0.0)
            continue;
        const double z = double(positions[i]);
        sum_w += w;
        sum_wz += w * z;
        sum_wz2 += w * z * z;
    }
    if (sum_w < 1e-30)
        return positions[N / 2]; // fallback: no significant signal

    const float z_avg = float(sum_wz / sum_w);

    // -- Second moment → ΔZ = 2σ --------------------------------------------
    float dz = 0.0f;
    {
        const double var = sum_wz2 / sum_w - double(z_avg) * double(z_avg);
        if (var > 0.0)
            dz = 2.0f * float(std::sqrt(var));
    }
    if (dz <= 0.0f)
        return z_avg; // degenerate: point-like signal, no refinement needed

    // -- Refined pass: restrict to [z_avg − ΔZ, z_avg + ΔZ] ----------------
    const float z_lo = z_avg - dz;
    const float z_hi = z_avg + dz;
    sum_w = 0.0;
    sum_wz = 0.0;
    for (int i = 0; i < N; ++i)
    {
        if (positions[i] < z_lo || positions[i] > z_hi)
            continue;
        const double w = double(weights[i]);
        if (w <= 0.0)
            continue;
        sum_w += w;
        sum_wz += w * double(positions[i]);
    }
    if (sum_w < 1e-30)
        return z_avg; // nothing in refined range — keep first estimate
    return float(sum_wz / sum_w);
}

// ============================================================================
// find_envelope_peak()
// ---------------------------------------------------------------------------
// Returns the physical position of the maximum of `weights`, refined to
// sub-step precision by fitting a parabola through the maximum and its two
// neighbours.  Used by methods where the coherence envelope is asymmetric,
// making the weighted centroid drift away from the true peak (e.g. Method 3
// with a broadband halogen source whose spectral asymmetry skews the tail).
//
// WHY THE PARABOLA (September 2026)
// ---------------------------------
// Returning positions[idx_max] quantises every height to the scan grid, which
// adds an irreducible uniform error of ±Δz/2, i.e. an rms floor of
// Δz/√12 = 5.8 nm at Δz = 20 nm (8.7 nm at 30 nm).  That floor was large
// enough to dominate the published S_q of Method 3 (8.1 nm on the 2026b
// dataset contains 5.8 nm of pure quantisation) and to produce the "multiple
// peaks" in its height histogram, which are simply the Δz levels.  Measured on
// the synthetic bench (scripts/method_accuracy_bench.py), dispersion of the
// height error: 5.78 nm with the discrete locator, 0.23 nm with the parabola
// (0.34 nm on the skewed envelope of the "asym" case).
//
// BIAS ON AN ASYMMETRIC ENVELOPE  (known, accepted, documented)
// -------------------------------------------------------------
// A parabola is symmetric, so fitting it to three samples of a SKEWED envelope
// pulls the vertex towards the wider side.  The real coherence envelope IS
// skewed: the red channel of the halogen source has a much longer coherence
// length than green and blue, which leaves a tail on one side.  The
// refinement therefore trades a random error (quantisation) for a partly
// systematic one, and Miguel accepted the change knowing that.
//
// How large it is, measured (split-normal envelope, trailing side k times
// wider; scripts/method_accuracy_bench.py "asym" case and the sweep in
// _agentes/_trabajo/C1_backend.md):
//
//   k          1.0    1.2    1.4    1.6    2.0    2.5    3.0
//   discrete  -0.7   +39.3  +73.3 +103.6 +154.3 +204.6 +245.8  nm
//   parabola  -0.1   +39.5  +73.6 +103.6 +154.0 +204.5 +245.5  nm
//
// So the tens-of-nanometre offset is NOT the price of the refinement: the
// discrete locator already returned it.  It is produced by the ENVELOPE
// SMOOTHING (cfg::ENVELOPE_SIGMA = 15 samples) acting on a skewed envelope —
// a symmetric Gaussian filter drags the maximum of an asymmetric curve
// towards its wide side.  Measured separately (report C3 §3): the maximum of
// the kernel's own, unsmoothed envelope is only −16 / −35 / −43 nm off for
// k = 1.6 / 2.5 / 3.5, and after the σ_e = 15 smoothing it becomes
// −100 / −200 / −276 nm, which is what the backend reports.  The bias is
// therefore shared with Method 1 (same smoothing) and is reduced by lowering
// σ_e, not by changing the peak locator.  What the parabola ADDS on top is
// below 0.5 nm for k from 1 to 5, with and without noise, while it removes
// the 5.8 nm of quantisation.  And an offset common to every pixel cancels in
// the quantity this instrument actually measures, which is height DIFFERENCES
// within one field; it does not cancel in an absolute height, which is
// uncalibrated anyway, nor between materials whose spectra differ.
//
// The refinement is skipped when the maximum sits on either end of the scan,
// or when the three samples are not concave (which a flat or noise-dominated
// envelope can produce): in those cases the discrete position is returned.
// ============================================================================
[[nodiscard]] static float find_envelope_peak(
    const std::vector<float> &weights,
    const std::vector<float> &positions)
{
    const int N = static_cast<int>(weights.size());
    if (N == 0) return 0.0f;
    int idx_max = 0;
    float max_val = -1.0f;
    for (int i = 0; i < N; ++i)
    {
        if (weights[i] > max_val)
        {
            max_val = weights[i];
            idx_max = i;
        }
    }
    // Flat/absent envelope (e.g. saturated or signal-free pixel): fall back to
    // the centre of the scan range, matching centroid_with_refinement().  The
    // previous behaviour returned positions[0] (the bottom of the Z-range),
    // which skewed the statistics of the representative pixels (B-15).
    if (max_val <= 0.0f)
        return positions[static_cast<size_t>(N / 2)];

    const float z_peak = positions[static_cast<size_t>(idx_max)];
    if (idx_max == 0 || idx_max == N - 1)
        return z_peak; // no neighbour on one side: nothing to interpolate

    // Parabola through (−1, y0), (0, y1), (+1, y2) in SAMPLE units:
    //   vertex offset  δ = ½ (y0 − y2) / (y0 − 2y1 + y2)
    // The denominator is the (negative) second difference; it is < 0 for a
    // genuine maximum.  A non-negative value means the three samples are not
    // concave (plateau, or two equal maxima) and δ would be meaningless.
    const double y0  = double(weights[static_cast<size_t>(idx_max - 1)]);
    const double y1  = double(weights[static_cast<size_t>(idx_max)]);
    const double y2  = double(weights[static_cast<size_t>(idx_max + 1)]);
    const double den = y0 - 2.0 * y1 + y2;
    if (den >= 0.0)
        return z_peak;

    double delta = 0.5 * (y0 - y2) / den;
    // |δ| ≤ ½ by construction for a true interior maximum; clamp against
    // round-off on a nearly degenerate parabola so the height can never leave
    // the half-step around the sample it came from.
    delta = std::clamp(delta, -0.5, 0.5);

    // Convert the offset from samples to physical units with the LOCAL step.
    // positions_sorted is ascending; the half-difference of the neighbours is
    // the step even if the scan is not perfectly uniform.
    const double step = 0.5 * (double(positions[static_cast<size_t>(idx_max + 1)])
                             - double(positions[static_cast<size_t>(idx_max - 1)]));
    return float(double(z_peak) + delta * step);
}

// ============================================================================
// compute_envelope()
// ---------------------------------------------------------------------------
// Single function that encodes per-method envelope logic.
// `signal` must already have the baseline removed when
// method_info.needs_baseline_removal is true.
// `k_avg` / `dk`: global bandpass params for Methods 2/4 (-1 = auto-estimate).
// `hilb` is stateless (plan caches are static thread_local) and may be shared
// between threads; `fbuf` is per-call scratch, so pass a thread_local buffer
// in parallel contexts to avoid repeated heap allocation.
// ============================================================================
static void compute_envelope(
    const methods::MethodInfo &method_info,
    const std::vector<float>  &signal,
    HilbertEnvelope           &hilb,
    HilbertEnvelope::ComplexBuf &fbuf,
    std::vector<float>        &env,
    int   k_avg  = -1,
    int   dk     = -1,
    float alpha  = -1.0f)   // inter-frame phase step [rad] for M3; -1 = assume π/2
{
    const int Nz = static_cast<int>(signal.size());
    switch (method_info.id)
    {
    case 1:
        env.assign(static_cast<size_t>(Nz), 0.0f);
        for (int z = 1; z < Nz - 1; ++z)
        {
            const float dg = 0.5f * (signal[z + 1] - signal[z - 1]);
            env[z] = dg * dg;
        }
        break;
    case 2:
        if (k_avg >= 0 && dk >= 0)
            hilb.compute_fft_bandpass(signal, fbuf, env, k_avg, dk);
        else
            hilb.compute_fft_bandpass(signal, fbuf, env);
        break;

    case 3:
    {
        // Generalised 5-point quadrature kernel valid for any inter-frame
        // phase step α = 4π δz / λ₀.
        //
        // For signal I_n = A + B·Γ_n·cos(φ + n·α) the two quadrature
        // components (treating Γ as locally constant over 5 samples) are:
        //
        //   Re = -(I[z+1] - I[z-1]) / (2 sin α)  =  B·Γ·sin φ
        //   Im = (-I[z-2] + 2·I[z] - I[z+2]) / (4 sin²α)  =  B·Γ·cos φ
        //   E  = sqrt(Re² + Im²)  =  B·Γ   (phase-independent)
        //
        // At α = π/2 (Schwider-Hariharan) this reduces to the standard kernel
        // (up to a 1/4 scale factor that cancels in the centroid).
        //
        // With δz = 20 nm and λ₀ = 570 nm (cfg::LAMBDA0_NM): α ≈ 0.441 rad
        // (≈ 25°).
        // The un-normalised S-H kernel at this α gives E ∝ Γ·sin(α)·
        // sqrt(sin²φ + sin²α·cos²φ), which has residual fringe ripple
        // because sin²α ≠ 1.  The normalised version eliminates this ripple.
        const float a  = (alpha > 0.0f) ? alpha
                                        : float(M_PI / 2.0);
        const float sa  = std::sin(a);
        const float sa2 = sa * sa;
        // Guard against degenerate phase step (α ≈ 0 or π)
        if (sa2 < 1e-6f)
        {
            env.assign(static_cast<size_t>(Nz), 0.0f);
            break;
        }
        const float c1 = 1.0f / (2.0f * sa);
        const float c2 = 1.0f / (4.0f * sa2);

        env.assign(static_cast<size_t>(Nz), 0.0f);
        for (int z = 2; z < Nz - 2; ++z)
        {
            const float re = -(signal[z + 1] - signal[z - 1]) * c1;
            const float im = (-signal[z - 2] + 2.0f * signal[z] - signal[z + 2]) * c2;
            env[z] = std::sqrt(re * re + im * im);
        }
        break;
    }

    case 4:
    {
        // Frequency-domain group-delay estimator (de Groot & Deck 1995).
        //
        // The envelope position n_peak appears as a linear phase ramp across
        // the spectrum, so each adjacent pair of bins carries the same step
        //
        //   step ≈ -2π n_peak / Nz  =  arg( FFT[k+1] · conj(FFT[k]) )
        //
        // VECTOR AVERAGE (September 2026).  The estimate is the ARGUMENT OF
        // THE SUM of the cross products,
        //
        //   step = arg( Σ_k FFT[k+1] · conj(FFT[k]) )
        //
        // and not, as before, the amplitude-weighted mean of the individual
        // arg() values.  Averaging angles is wrong for two measured reasons
        // (report D1-12, 23-Sep-2026):
        //
        //   1. SINGULARITY AT n_peak = Nz/2.  There the true step is exactly
        //      ±π, so atan2 returns +π for some bins and -π for others.  The
        //      arithmetic mean of the two branches is ≈ 0, which places the
        //      surface at the bottom of the scan: an error of Nz·Δz/2 (2.7 µm
        //      on a 300-frame, 20 nm scan).  A sum of complex numbers has no
        //      branch to choose: the two ±π contributions add coherently and
        //      the argument of the total is π, the correct answer.
        //   2. NOISE.  The band half-width dk/2 comes from the second moment
        //      of the whole amplitude spectrum, which the noise floor inflates
        //      (on a real stack, 315 bins around a peak only ~12 bins wide).
        //      Averaging angles gives every one of those empty bins a vote
        //      whose weight is |FFT[k]| — small, but its angle is uniformly
        //      random.  In the vector sum each bin contributes a vector of
        //      length |FFT[k+1]|·|FFT[k]|, so the noise bins contribute
        //      quadratically less and cancel against each other instead.
        //
        // Measured on the synthetic bench (scripts/method_accuracy_bench.py),
        // height error dispersion: 0.85 nm clean / 8.9 nm with 5 % noise /
        // 8.6 nm at n_peak = Nz/2, against 88.5 / 807.6 / 2327.2 nm for the
        // mean of angles.  (The 0.85 nm floor of the clean case is not the
        // estimator: peak_f is turned into a two-sample pseudo-envelope which
        // is then smoothed and centroided, and the truncation of the ±2σ
        // centroid window moves the result by up to ~1.7 nm.  Report C3 §2.)
        // The cost is identical (the same loop, without the atan2 per bin).
        //
        // Correction for n_peak > Nz/2: the true step then lies in
        // (-2π, -π), atan2 returns step + 2π > 0, yielding peak_f < 0;
        // adding Nz restores the correct index.
        //
        // Rectangular window (no tapering): maximises frequency resolution,
        // keeping the coherence peak confined to its natural bin width (≈1-3
        // bins).  The |FFT[k+1]|·|FFT[k]| weighting implicit in the vector sum
        // already down-weights leakage-polluted low-energy bins, providing the
        // same leakage immunity as Hann without broadening the main lobe.
        hilb.forward_fft(signal, fbuf);

        const int half_bw = (dk > 0)
            ? std::min(Nz / 4, std::max(2, dk / 2))
            : std::min(Nz / 4, 4);
        const int k_lo = (k_avg > 0) ? std::max(1,              k_avg - half_bw) : 1;
        const int k_hi = (k_avg > 0) ? std::min(Nz / 2 - 1, k_avg + half_bw)
                                      : std::min(Nz / 2 - 1, half_bw);

        // Accumulate the cross products themselves (double precision: the
        // products of two float spectra of a 12-bit, Nz-sample signal reach
        // ~1e13 and hundreds of them are summed).
        double sum_re = 0.0, sum_im = 0.0;
        for (int k = k_lo; k < k_hi; ++k)
        {
            const double a_re = double(fbuf[k][0]),     a_im = double(fbuf[k][1]);
            const double b_re = double(fbuf[k + 1][0]), b_im = double(fbuf[k + 1][1]);
            sum_re += b_re * a_re + b_im * a_im;   // Re( X[k+1] · conj(X[k]) )
            sum_im += b_im * a_re - b_re * a_im;   // Im( X[k+1] · conj(X[k]) )
        }

        // Fallback for signal-free/saturated pixels (all in-band bins empty,
        // so the resultant vector vanishes): place the pseudo-envelope at the
        // CENTRE of the scan so the final height is positions[Nz/2],
        // consistent with the fallbacks of the centroid and peak locators
        // (B-15).  The previous value (peak_f = 0) pinned such pixels to the
        // bottom of the Z-range.
        float peak_f = 0.5f * float(Nz);
        if (std::abs(sum_re) > 1e-30 || std::abs(sum_im) > 1e-30)
        {
            const float m = float(std::atan2(sum_im, sum_re));
            peak_f = -m * float(Nz) / float(2.0 * M_PI);
            if (peak_f < 0.0f)
                peak_f += float(Nz);
        }

        const float clamped = std::clamp(peak_f, 0.0f, float(Nz - 1));
        const int   lo      = std::clamp(int(std::floor(clamped)), 0, Nz - 2);
        const float frac    = clamped - float(lo);

        env.assign(static_cast<size_t>(Nz), 0.0f);
        env[static_cast<size_t>(lo)]     = 1.0f - frac;
        env[static_cast<size_t>(lo + 1)] = frac;
        break;
    }

    default:
        throw std::runtime_error(
            "compute_envelope: unsupported method id " + std::to_string(method_info.id));
    }
}

// ============================================================================
// Streaming reconstruction with libtiff
// ============================================================================
cv::Mat reconstruct_height_map_rowchunks(
    const std::vector<std::string> &files,
    const std::regex &pattern,
    ImageStack &stack_info,
    std::function<void(float)> progress_callback,
    const std::array<double, 3> &channel_weights,
    int method,
    int *out_k_avg,
    int *out_dk)
{
    if (files.empty())
        throw std::runtime_error("No TIFF files provided");

    const methods::MethodInfo *method_info = methods::find(method);
    if (!method_info)
        throw std::runtime_error(
            "reconstruct_height_map_rowchunks: unknown method id " + std::to_string(method));

    // ------------------------------------------------
    // 1. Image dimensions: open first file and check
    // ------------------------------------------------
    uint32_t w = 0, h = 0;
    const bool use_bin12 = is_bin12_file(files[0]);
    const bool use_png = !use_bin12 && is_png_file(files[0]);
    {
        if (use_bin12)
        {
            FilePtr f(std::fopen(files[0].c_str(), "rb"));
            if (!f)
                throw std::runtime_error("Failed to open first bin12 file: " + files[0]);
            auto hdr = open_bin12(f.get(), files[0]);
            w = hdr.width;
            h = hdr.height;
        }
        else if (use_png)
        {
            cv::Mat tmp = cv::imread(files[0], cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH);
            if (tmp.empty())
                throw std::runtime_error("Failed to read PNG for dimensions: " + files[0]);
            w = uint32_t(tmp.cols);
            h = uint32_t(tmp.rows);
        }
        else
        {
            TiffPtr tif0(TIFFOpen(files[0].c_str(), "r"));
            if (!tif0)
                throw std::runtime_error("Failed to open first TIFF: " + files[0]);
            if (TIFFGetField(tif0.get(), TIFFTAG_IMAGEWIDTH, &w) != 1 ||
                TIFFGetField(tif0.get(), TIFFTAG_IMAGELENGTH, &h) != 1)
                throw std::runtime_error("TIFF is missing width/length tags: " + files[0]);
        }
        if (w == 0 || h == 0)
            throw std::runtime_error("First image has zero dimensions: " + files[0]);
    }

    const int Nx = int(w), Ny = int(h), Nz = int(files.size());

    // Metadata
    stack_info.Nx = Nx;
    stack_info.Ny = Ny;
    stack_info.Nz = Nz;
    // We don't use the data member, so we can clear it
    // It was used when everything was loaded to RAM, but
    // now it is impossible (big datasets)
    stack_info.data.clear();
    stack_info.data.shrink_to_fit();

    // ------------------------------------------------
    // 2. Extract positions
    // ------------------------------------------------
    auto positions = parse_positions_from_filenames(files, pattern);
    if (positions.empty()) // cancelled during parsing
        return cv::Mat();

    const float maxp = *std::max_element(positions.begin(), positions.end());
    for (float &p : positions)
        p = maxp - p;

    // ------------------------------------------------
    // 3. Sort Z order
    // ------------------------------------------------
    std::vector<int> z_order(Nz);
    std::iota(z_order.begin(), z_order.end(), 0);

    std::sort(z_order.begin(), z_order.end(),
              [&](int a, int b)
              { return positions[a] < positions[b]; });

    std::vector<float> positions_sorted(Nz);
    for (int i = 0; i < Nz; ++i)
        positions_sorted[i] = positions[z_order[i]];

    stack_info.z_order = z_order;

    // Inter-frame phase step for Method 3 (PSI 5-point generalised kernel).
    // α = 4π·δz/λ₀, where δz is the mean step derived from the sorted positions.
    const float alpha_m3 = 4.0f * float(M_PI)
        * float(cfg::NOMINAL_DZ_NM) / float(cfg::LAMBDA0_NM);

    // ------------------------------------------------
    // 4. Auto‑tune row‑chunk size
    // ------------------------------------------------
    // We try to use as much RAM as possible, to minimise
    // the ammount of reading from disk.
    int row_chunk = auto_row_chunk(Nx, Ny, Nz, cfg::AVAILABLE_RAM_RATIO_USED);

    cv::Mat height(Ny, Nx, CV_32F);
    HilbertEnvelope hilb;

    // ------------------------------------------------------------
    // 5. AUTO‑TUNED PROGRESS WEIGHTS (read vs compute)
    // ------------------------------------------------------------
    // To know how much the reading and computing should contribute to
    // the total progress. This depends on how much time they require per step
    // A time-consuming step accounts for more progress.

    // Sample realistic read time (Open + read + Close)
    double sample_read_time = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        const int z_samples = std::min(8, Nz); // number of files to sample
        for (int i = 0; i < z_samples; ++i)
        {
            std::vector<float> tmp(Nx);
            // Validate each sampled file's dimensions against the first file,
            // exactly like the main chunk reader below: the row readers size
            // their buffers from the ACTUAL file but are asked for Nx pixels,
            // so a narrower file here would be a heap over-read (UB) instead
            // of the clean dimension-mismatch error the main loop throws.
            if (use_bin12)
            {
                FilePtr f(std::fopen(files[i].c_str(), "rb"));
                if (f)
                {
                    auto b12hdr = open_bin12(f.get(), files[i]);
                    if (b12hdr.width != w || b12hdr.height != h)
                        throw std::runtime_error(
                            "Image dimensions differ from first file ("
                            + std::to_string(b12hdr.width) + "x" + std::to_string(b12hdr.height)
                            + " vs " + std::to_string(w) + "x" + std::to_string(h) + "): " + files[i]);
                    for (int rr = 0; rr < row_chunk; ++rr)
                    {
                        uint32_t row = std::min<uint32_t>(rr, h - 1);
                        read_one_scanline_from_bin12(f.get(), b12hdr, row, Nx, tmp.data(), channel_weights);
                    }
                }
            }
            else if (use_png)
            {
                cv::Mat img = cv::imread(files[i], cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH);
                if (!img.empty())
                {
                    if (uint32_t(img.cols) != w || uint32_t(img.rows) != h)
                        throw std::runtime_error(
                            "Image dimensions differ from first file ("
                            + std::to_string(img.cols) + "x" + std::to_string(img.rows)
                            + " vs " + std::to_string(w) + "x" + std::to_string(h) + "): " + files[i]);
                    for (int rr = 0; rr < row_chunk; ++rr)
                    {
                        int row = std::min(rr, img.rows - 1);
                        read_png_row_to_float(img, row, Nx, tmp.data(), channel_weights);
                    }
                }
            }
            else
            {
                TiffPtr tif(TIFFOpen(files[i].c_str(), "r"));
                if (tif)
                {
                    uint32_t fw = 0, fh = 0;
                    if (TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &fw) != 1 ||
                        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &fh) != 1 ||
                        fw != w || fh != h)
                        throw std::runtime_error(
                            "Image dimensions differ from first file ("
                            + std::to_string(fw) + "x" + std::to_string(fh)
                            + " vs " + std::to_string(w) + "x" + std::to_string(h) + "): " + files[i]);
                    for (int rr = 0; rr < row_chunk; ++rr)
                    {
                        uint32_t row = std::min<uint32_t>(rr, h - 1);
                        read_one_scanline_to_float(tif.get(), row, Nx, tmp.data(), channel_weights);
                    }
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        float correction_factor = float(Nz) / z_samples; // compensate for sampling
        sample_read_time = std::chrono::duration<double>(t1 - t0).count() * correction_factor;
    }

    // Sample compute time on a realistic number of pixels using the selected method
    double sample_compute_time = 0.0;
    {
        const int px_samples = std::min(64, Nx);
        auto t0 = std::chrono::high_resolution_clock::now();

        {
            HilbertEnvelope hilb_test;
            HilbertEnvelope::ComplexBuf fbuf;
            for (int i = 0; i < px_samples; ++i)
            {
                std::vector<float> sig(Nz, 0.0f);
                std::vector<float> env;
                // The baseline filter now runs fused inside the compute phase
                // (O-2), so include its cost in the compute-time sample.
                if (method_info->needs_baseline_removal)
                    sig = gaussian_filter_1d(sig, cfg::BASELINE_SIGMA);
                compute_envelope(*method_info, sig, hilb_test, fbuf, env,
                                 /*k_avg=*/-1, /*dk=*/-1, alpha_m3);
                env = gaussian_filter_1d(env, cfg::ENVELOPE_SIGMA);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double t_per_px = std::chrono::duration<double>(t1 - t0).count() / px_samples;
        int n_threads = std::max(1, omp_get_max_threads());
        sample_compute_time = t_per_px * double(Nx) * double(row_chunk) / n_threads;
    }

    double total_t = sample_read_time + sample_compute_time;

    double W_read = sample_read_time / total_t;
    double W_comp = sample_compute_time / total_t;

    // Clamp to avoid extreme values
    if (W_read < 0.10)
        W_read = 0.10;
    if (W_comp < 0.10)
        W_comp = 0.10;

    // Normalise so W_read + W_comp == 1.0 exactly.
    // Independent clamping can make their sum exceed 1.0, causing progress to
    // overshoot at the end of each compute phase and then fall back at the
    // start of the next chunk's read phase (visible as backwards jumps in UI).
    {
        double sum = W_read + W_comp;
        W_read /= sum;
        W_comp /= sum;
    }

    // ------------------------------------------------------------
    // 6. CHUNK PROCESSING
    // ------------------------------------------------------------
    // baseline_kernel: Gaussian low-pass applied to the raw Z-signal when
    // method_info->needs_baseline_removal == true (Methods 1 and 4).
    // Subtracts the slowly-varying DC background before envelope extraction.
    // Sigma must be larger than the fringe period to avoid attenuating the carrier.
    // envelope_kernel: Gaussian smoothing applied to the raw envelope after both methods
    // to suppress shot-noise ripple before the centroid computation.
    const std::vector<float> baseline_kernel = make_gaussian_kernel(cfg::BASELINE_SIGMA);
    const std::vector<float> envelope_kernel = make_gaussian_kernel(cfg::ENVELOPE_SIGMA);

    const int UPDATE_Z = 16; // update GUI every 16 TIFFs

    // Bandpass parameters estimated once from the averaged spectrum of a sample
    // of pixels (Method 2 only). Using a single shared filter for all pixels
    // eliminates the per-pixel filter variation that causes systematic height
    // artifacts between regions with different local SNR (e.g. center vs. edge
    // of flat features). Estimated from the first chunk; -1 = not yet computed.
    int global_k_avg = -1;
    int global_dk = -1;

    for (int y0 = 0; y0 < Ny; y0 += row_chunk)
    {
        if (cancel_requested())
            return cv::Mat();

        const int R = std::min(row_chunk, Ny - y0); // To account for the last chunk (possibly smaller)
        std::vector<float> chunk(size_t(Nz) * R * Nx);

        // ------------------------------------------------
        // A. READ images — parallelised across the Nz files (O-4)
        //    Safe because each iteration owns its file handle (libtiff: one
        //    TIFF* per file/thread is safe; cv::imread is thread-safe; bin12
        //    uses a private FILE*) and writes to a disjoint z-slice of the
        //    chunk.  Exceptions are captured and rethrown after the region
        //    (an exception escaping an OpenMP region calls std::terminate).
        //    Progress is reported only from the master thread — the same
        //    thread that called this function, preserving the callback
        //    threading contract — using a monotonic atomic counter, so the
        //    reported fraction never moves backwards.
        // ------------------------------------------------
        {
            std::exception_ptr read_error = nullptr;
            std::mutex read_error_mutex;
            std::atomic<int> files_done{0};
            int last_reported = 0; // master thread only

#pragma omp parallel for schedule(dynamic)
            for (int new_z = 0; new_z < Nz; ++new_z)
            {
                try
                {
                    if (cancel_requested())
                        continue;

                    const int old_z = z_order[new_z];
                    const std::string &fpath = files[old_z];

                    if (use_bin12)
                    {
                        FilePtr f(std::fopen(fpath.c_str(), "rb"));
                        if (!f)
                            throw std::runtime_error("bin12 open failed: " + fpath);
                        auto b12hdr = open_bin12(f.get(), fpath);
                        if (b12hdr.width != w || b12hdr.height != h)
                            throw std::runtime_error(
                                "Image dimensions differ from first file ("
                                + std::to_string(b12hdr.width) + "x" + std::to_string(b12hdr.height)
                                + " vs " + std::to_string(w) + "x" + std::to_string(h) + "): " + fpath);
                        // Single fseek+fread for the R consecutive chunk rows.
                        read_rows_block_from_bin12(
                            f.get(), b12hdr, uint32_t(y0), R, Nx,
                            &chunk[idx3(new_z, 0, 0, R, Nx)], size_t(Nx),
                            channel_weights);
                    }
                    else if (use_png)
                    {
                        cv::Mat img = cv::imread(fpath, cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH);
                        if (img.empty())
                            throw std::runtime_error("PNG read failed: " + fpath);
                        if (uint32_t(img.cols) != w || uint32_t(img.rows) != h)
                            throw std::runtime_error(
                                "Image dimensions differ from first file ("
                                + std::to_string(img.cols) + "x" + std::to_string(img.rows)
                                + " vs " + std::to_string(w) + "x" + std::to_string(h) + "): " + fpath);
                        for (int r = 0; r < R; ++r)
                        {
                            if (cancel_requested())
                                break;
                            float *dst = &chunk[idx3(new_z, r, 0, R, Nx)];
                            read_png_row_to_float(img, y0 + r, Nx, dst, channel_weights);
                        }
                    }
                    else
                    {
                        TiffPtr tif(TIFFOpen(fpath.c_str(), "r"));
                        if (!tif)
                            throw std::runtime_error("TIFF open failed: " + fpath);
                        uint32_t fw = 0, fh = 0;
                        if (TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &fw) != 1 ||
                            TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &fh) != 1 ||
                            fw != w || fh != h)
                            throw std::runtime_error(
                                "Image dimensions differ from first file ("
                                + std::to_string(fw) + "x" + std::to_string(fh)
                                + " vs " + std::to_string(w) + "x" + std::to_string(h) + "): " + fpath);
                        for (int r = 0; r < R; ++r)
                        {
                            if (cancel_requested())
                                break;
                            float *dst = &chunk[idx3(new_z, r, 0, R, Nx)];
                            read_one_scanline_to_float(tif.get(), uint32_t(y0 + r), Nx, dst, channel_weights);
                        }
                    }

                    files_done.fetch_add(1, std::memory_order_relaxed);

                    // Update progress every few files (master thread only; a
                    // throwing Python callback is captured like a read error
                    // and rethrown after the region).
                    if (progress_callback && omp_get_thread_num() == 0)
                    {
                        const int now = files_done.load(std::memory_order_relaxed);
                        if (now - last_reported >= UPDATE_Z)
                        {
                            last_reported = now;
                            float frac_chunk = float(R) / float(Ny);
                            float frac_z = float(now) / float(Nz);
                            float frac = W_read * frac_z * frac_chunk + (float(y0) / float(Ny));
                            progress_callback(frac);
                        }
                    }
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(read_error_mutex);
                    if (!read_error)
                        read_error = std::current_exception();
                    request_cancel(true); // make remaining iterations bail out fast
                }
            }
            if (read_error)
                std::rethrow_exception(read_error);
            if (cancel_requested())
                return cv::Mat();
        }

        // ------------------------------------------------------------
        // A2. ESTIMATE GLOBAL BANDPASS (Methods 2 and 4, first chunk only)
        // Sample pixel Z-traces from the just-read chunk, average their
        // amplitude spectra, and derive a single {k_avg, dk} that is applied
        // to every pixel for the rest of the reconstruction.
        //
        // THE SAMPLE IS A FIXED GRID (September 2026).  It used to be "256
        // pixels of the first chunk, stride (R·Nx)/256", and R is decided by
        // auto_row_chunk() from the free RAM of the moment.  Because this one
        // pair {k_avg, dk} is then shared by EVERY pixel, a machine with a
        // different amount of free memory reconstructed a different height
        // map: measured on data/S1F1 by faking /proc/meminfo, one bin of dk
        // moved 99.95 % of the Method 2 heights (std 1.4 nm, max 22.7 nm).
        // The grid below spans BAND_SAMPLE_ROWS rows — the floor that
        // auto_row_chunk() always delivers — by BAND_SAMPLE_COLS columns
        // across the full width, so it does not depend on R, on the free RAM,
        // or on how the image happens to be split into chunks.  See the
        // comment on cfg::BAND_SAMPLE_ROWS for why a band of rows samples the
        // amplitude spectrum just as well as a scattered set.
        // ------------------------------------------------------------
        if (method_info->needs_global_params && global_k_avg < 0)
        {
            // Rows guaranteed to be in this first chunk whatever the RAM
            // (auto_row_chunk never returns fewer than 16 rows, nor more
            // than Ny; R is further clamped to the rows left in the image).
            const int rows_s = std::min(R, cfg::BAND_SAMPLE_ROWS);
            const int cols_s = std::min(Nx, cfg::BAND_SAMPLE_COLS);
            const int n_sample = rows_s * cols_s;

            std::vector<std::vector<float>> sample_signals;
            sample_signals.reserve(static_cast<size_t>(n_sample));
            std::vector<float> sig(static_cast<size_t>(Nz));
            std::vector<float> sample_baseline;
            for (int iy = 0; iy < rows_s; ++iy)
            {
                for (int ix = 0; ix < cols_s; ++ix)
                {
                    // Cell centres of the grid: never the very first/last
                    // column, and evenly spread whatever Nx is.
                    const int r = iy;
                    const int x = ((2 * ix + 1) * Nx) / (2 * cols_s);
                    for (int z = 0; z < Nz; ++z)
                        sig[static_cast<size_t>(z)] = chunk[idx3(z, r, x, R, Nx)];

                    // Methods that reconstruct from the baseline-subtracted signal
                    // (needs_baseline_removal, e.g. Method 4) must estimate the
                    // bandpass on that same signal.  Estimating on the raw signal
                    // let a strong DC drift anchor the argmax at k=1..3, shifting
                    // the band for the whole image (B-14).  Method 2 keeps the raw
                    // estimate by design: its bandpass zeros DC itself.
                    if (method_info->needs_baseline_removal)
                    {
                        gaussian_apply(sig, baseline_kernel, sample_baseline);
                        for (int z = 0; z < Nz; ++z)
                            sig[static_cast<size_t>(z)] -= sample_baseline[static_cast<size_t>(z)];
                    }
                    sample_signals.push_back(sig);
                }
            }
            HilbertEnvelope::ComplexBuf tmp_buf;
            std::tie(global_k_avg, global_dk) =
                HilbertEnvelope::estimate_bandpass_params(sample_signals, tmp_buf);
        }

        // ------------------------------------------------
        // B+C. BASELINE + ENVELOPE + SMOOTH + CENTROID — parallel compute
        //     + serial progress observer (threaded)
        //    The per-method envelope logic is dispatched inside
        //    compute_envelope() (see methods.hpp for the registry).
        //
        //    O-2: the former separate phase B gathered every pixel's Z-trace,
        //    filtered it and scattered it back into the chunk — and for
        //    methods WITHOUT baseline removal (2/3) the gather ran for
        //    nothing.  The baseline subtraction is now fused into this loop
        //    on the freshly gathered tl_sig (same float32 arithmetic, so the
        //    result is bit-identical), eliminating one full gather+scatter
        //    pass over the chunk for methods 1/4 and the whole pass for 2/3.
        // ------------------------------------------------

        std::atomic<int> rows_done{0};
        std::atomic<bool> worker_done{false};
        std::exception_ptr worker_error = nullptr;
        std::mutex worker_error_mutex;

        // ----------------------------
        // THREAD: OpenMP compute
        // thread_local buffers give zero heap allocations per pixel WITHIN a
        // chunk. NOTE: because this std::thread is created per chunk, its
        // OpenMP team (and every thread_local: tl_* buffers, FFTW plan cache)
        // is torn down and rebuilt at each chunk boundary — plans are
        // re-created per chunk, serialized by the planner mutex. Correct
        // (fftwf_make_planner_thread_safe covers the teardown) but a known
        // inefficiency; running the OpenMP region on the calling thread and
        // moving the progress observer here instead would keep one pool and
        // one plan set for the whole reconstruction.
        // Exceptions are captured per iteration (an exception escaping an
        // OpenMP region calls std::terminate) and rethrown after join().
        // ----------------------------
        std::thread worker([&]
                           {
#pragma omp parallel for schedule(static)
            for (int r = 0; r < R; ++r)
            {
                try
                {
                    if (cancel_requested())
                        continue;

                    thread_local std::vector<float>          tl_sig;
                    thread_local std::vector<float>          tl_base;
                    thread_local HilbertEnvelope::ComplexBuf tl_fft;
                    thread_local std::vector<float>          tl_env;
                    thread_local std::vector<float>          tl_smooth;

                    tl_sig.resize(static_cast<size_t>(Nz));

                    for (int x = 0; x < Nx; ++x)
                    {
                        if (cancel_requested())
                            continue;

                        for (int z = 0; z < Nz; ++z)
                            tl_sig[z] = chunk[idx3(z, r, x, R, Nx)];

                        // Fused baseline subtraction (formerly phase B).
                        if (method_info->needs_baseline_removal)
                        {
                            gaussian_apply(tl_sig, baseline_kernel, tl_base);
                            for (int z = 0; z < Nz; ++z)
                                tl_sig[z] -= tl_base[z];
                        }

                        compute_envelope(*method_info, tl_sig, hilb, tl_fft, tl_env,
                                         global_k_avg, global_dk, alpha_m3);

                        gaussian_apply(tl_env, envelope_kernel, tl_smooth);

                        height.at<float>(y0 + r, x) =
                            method_info->use_peak_locator
                                ? find_envelope_peak(tl_smooth, positions_sorted)
                                : centroid_with_refinement(tl_smooth, positions_sorted);
                    }

                    rows_done.fetch_add(1, std::memory_order_relaxed);
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(worker_error_mutex);
                    if (!worker_error)
                        worker_error = std::current_exception();
                    request_cancel(true); // make remaining iterations bail out fast
                }
            }

            worker_done.store(true); });

        // RAII: if the observer below throws (e.g. a progress callback raising
        // from Python), the joiner cancels the worker and joins it during
        // unwinding.  Destroying a joinable std::thread would otherwise call
        // std::terminate and kill the whole host process (B-4).
        ThreadJoiner joiner(worker);

        // ----------------------------
        // MAIN THREAD: progress observer
        // ----------------------------
        if (progress_callback)
        {
            int last = 0;
            while (!worker_done.load(std::memory_order_relaxed))
            {
                if (cancel_requested())
                    break;

                // Poll every 5 ms: responsive without burning CPU.
                std::this_thread::sleep_for(std::chrono::milliseconds(5));

                int now = rows_done.load(std::memory_order_relaxed);
                if (now != last)
                {
                    last = now;
                    float frac_chunk = float(R) / float(Ny);
                    float frac_r = float(now) / float(R);
                    float frac = (W_read + W_comp * frac_r) * frac_chunk + (float(y0) / float(Ny));
                    progress_callback(std::min(frac, 1.0f));
                }
            }
        }
        worker.join(); // Wait for all OpenMP threads to honour the cancel flag

        if (worker_error)
            std::rethrow_exception(worker_error);

        // Honour the contract "cancelled => empty Mat": if the cancel arrived
        // while the last chunk was computing, the loop above exits normally
        // but 'height' contains unfinished rows.  Do not return them.
        if (cancel_requested())
            return cv::Mat();
    }

    if (out_k_avg)
        *out_k_avg = global_k_avg;
    if (out_dk)
        *out_dk = global_dk;

    return height;
}

// ============================================================================
// SAVE PIXEL PLOT (streaming, no full stack needed)
// ============================================================================
void save_pixel_plot_streaming_libtiff(
    const std::vector<std::string> &files,
    const std::regex &pattern,
    const ImageStack &stack_info,
    int py, int px,
    const std::string &base_name,
    const std::array<double, 3> &channel_weights,
    int method,
    int k_avg,
    int dk)
{
    const int Nx = stack_info.Nx;
    const int Ny = stack_info.Ny;
    const int Nz = stack_info.Nz;

    if (px < 0 || px >= Nx || py < 0 || py >= Ny)
        throw std::runtime_error("Pixel coordinates out of bounds");

    if (stack_info.z_order.empty())
        throw std::runtime_error("z_order is empty - run reconstruction first");

    // 1. Reconstruct positions_sorted for the pixel (ASC order as in reconstruction)
    auto positions = parse_positions_from_filenames(files, pattern);
    if (positions.empty()) // cancelled during parsing
        return;

    const float maxp = *std::max_element(positions.begin(), positions.end());
    for (float &p : positions)
        p = maxp - p;

    std::vector<float> positions_sorted(Nz);
    for (int z = 0; z < Nz; ++z)
        positions_sorted[z] = positions[stack_info.z_order[z]];

    // 2. Build pixel signal (streaming row 'py', in sorted Z order)
    const bool use_bin12_px = is_bin12_file(files[0]);
    const bool use_png_px = !use_bin12_px && is_png_file(files[0]);
    std::vector<float> signal(Nz);
    std::vector<float> rowbuf(Nx);

    for (int new_z = 0; new_z < Nz; ++new_z)
    {
        const int old_z = stack_info.z_order[new_z];
        const std::string &fpath = files[size_t(old_z)];

        if (use_bin12_px)
        {
            FilePtr f(std::fopen(fpath.c_str(), "rb"));
            if (!f)
                throw std::runtime_error("bin12 open failed: " + fpath);
            auto b12hdr = open_bin12(f.get(), fpath);
            if (int(b12hdr.width) != Nx || int(b12hdr.height) != Ny)
                throw std::runtime_error("bin12 dimensions differ from stack: " + fpath);
            read_one_scanline_from_bin12(f.get(), b12hdr, static_cast<uint32_t>(py), Nx, rowbuf.data(), channel_weights);
        }
        else if (use_png_px)
        {
            cv::Mat img = cv::imread(fpath, cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH);
            if (img.empty())
                throw std::runtime_error("PNG read failed: " + fpath);
            if (img.cols != Nx || img.rows != Ny)
                throw std::runtime_error("PNG dimensions differ from stack: " + fpath);
            read_png_row_to_float(img, py, Nx, rowbuf.data(), channel_weights);
        }
        else
        {
            TiffPtr tif(TIFFOpen(fpath.c_str(), "r"));
            if (!tif)
                throw std::runtime_error("TIFF open failed: " + fpath);
            uint32_t fw = 0, fh = 0;
            if (TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &fw) != 1 ||
                TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &fh) != 1 ||
                int(fw) != Nx || int(fh) != Ny)
                throw std::runtime_error("TIFF dimensions differ from stack: " + fpath);
            read_one_scanline_to_float(tif.get(), static_cast<uint32_t>(py), Nx, rowbuf.data(), channel_weights);
        }

        signal[new_z] = rowbuf[px];
    }

    const methods::MethodInfo *method_info = methods::find(method);
    if (!method_info)
        throw std::runtime_error(
            "save_pixel_plot_streaming_libtiff: unknown method id " + std::to_string(method));

    const float alpha_m3 = 4.0f * float(M_PI)
        * float(cfg::NOMINAL_DZ_NM) / float(cfg::LAMBDA0_NM);

    // 3. Baseline removal (when needed by the method)
    if (method_info->needs_baseline_removal)
    {
        std::vector<float> baseline(signal.begin(), signal.end());
        baseline = gaussian_filter_1d(baseline, cfg::BASELINE_SIGMA);
        for (int z = 0; z < Nz; ++z)
            signal[z] -= baseline[z];
    }

    // 4. Envelope extraction
    HilbertEnvelope hilb;
    HilbertEnvelope::ComplexBuf fbuf;
    std::vector<float> env;
    compute_envelope(*method_info, signal, hilb, fbuf, env, k_avg, dk, alpha_m3);
    // Gaussian smoothing — same as main reconstruction; saved envelope must match
    // what the height estimator actually operated on so the Python plots are coherent.
    env = gaussian_filter_1d(env, cfg::ENVELOPE_SIGMA);

    // Height estimate using the same logic as the main reconstruction loop.
    const float h_est = method_info->use_peak_locator
        ? find_envelope_peak(env, positions_sorted)
        : centroid_with_refinement(env, positions_sorted);

    // 5. Save result
    //      Compact version: ONE .npy with shape (Nz, 4)
    //      columns = [positions_sorted, signal, envelope, h_est (repeated)]
    fs::path folder = fs::path(cfg::OUTPUT_FOLDER) / base_name;
    fs::create_directories(folder);

    // Name containing pixel coordinates (easy to parse later)
    //   <base>_pixel_y<py>_x<px>.npy
    const std::string stem = base_name + "_pixel_y" + std::to_string(py) + "_x" + std::to_string(px);
    const std::string f_combined = (folder / (stem + ".npy")).string();

    // Pack (Nz × 4) into contiguous memory: col0=pos, col1=signal, col2=envelope, col3=h_est
    std::vector<float> combined(static_cast<size_t>(Nz) * 4);
    for (int i = 0; i < Nz; ++i)
    {
        combined[size_t(i) * 4 + 0] = positions_sorted[size_t(i)];
        combined[size_t(i) * 4 + 1] = signal[size_t(i)];
        combined[size_t(i) * 4 + 2] = env[size_t(i)];
        combined[size_t(i) * 4 + 3] = h_est;
    }

    // Save (Nz × 4 matrix)
    save_npy(f_combined, combined.data(), static_cast<size_t>(Nz), /*cols=*/4);

    std::cout << "[OK] Pixel plot (y=" << py << ", x=" << px << ") saved -> " << f_combined << "\n";
}

// ============================================================================
// save_pixel_plots_batch_streaming()
// ---------------------------------------------------------------------------
// Saves multiple pixel plots with a SINGLE pass through the Z-stack.
//
// Complexity:
//   Old (N separate calls): N * Nz file-opens  +  N * Nz image-loads (PNG)
//   New (batch):                  Nz file-opens  +      Nz image-loads (PNG)
//
// After the single I/O pass, envelope processing runs in parallel (OpenMP).
// ============================================================================
void save_pixel_plots_batch_streaming(
    const std::vector<std::pair<int,int>> &pixels,
    const std::vector<std::string> &files,
    const std::regex &pattern,
    const ImageStack &stack_info,
    const std::string &base_name,
    const std::array<double, 3> &channel_weights,
    int method,
    int k_avg,
    int dk,
    const std::string &out_dir)
{
    const int N  = static_cast<int>(pixels.size());
    if (N == 0) return;

    const int Nx = stack_info.Nx;
    const int Ny = stack_info.Ny;
    const int Nz = stack_info.Nz;

    if (stack_info.z_order.empty())
        throw std::runtime_error("save_pixel_plots_batch_streaming: z_order empty");

    // Defence in depth: every coordinate must be inside the stack.
    for (const auto &p : pixels)
    {
        if (p.first < 0 || p.first >= Ny || p.second < 0 || p.second >= Nx)
            throw std::runtime_error(
                "save_pixel_plots_batch_streaming: pixel (y=" + std::to_string(p.first)
                + ", x=" + std::to_string(p.second) + ") out of bounds ("
                + std::to_string(Ny) + "x" + std::to_string(Nx) + ")");
    }

    // ------------------------------------------------------------------
    // 1. Sorted Z-positions (identical logic to save_pixel_plot_streaming_libtiff)
    // ------------------------------------------------------------------
    auto positions = parse_positions_from_filenames(files, pattern);
    if (positions.empty()) // cancelled during parsing
        return;
    const float maxp = *std::max_element(positions.begin(), positions.end());
    for (float &p : positions) p = maxp - p;

    std::vector<float> positions_sorted(Nz);
    for (int z = 0; z < Nz; ++z)
        positions_sorted[z] = positions[stack_info.z_order[z]];

    // ------------------------------------------------------------------
    // 2. Allocate signal matrix  signals[pixel_index][z_index]
    // ------------------------------------------------------------------
    std::vector<std::vector<float>> signals(N, std::vector<float>(Nz, 0.0f));

    // Build row -> [(pixel_index, x_column)] map so each scanline is read once.
    // std::map keeps rows in ascending order — optimal for TIFF sequential access.
    std::map<int, std::vector<std::pair<int,int>>> row_map;
    for (int i = 0; i < N; ++i)
        row_map[pixels[i].first].emplace_back(i, pixels[i].second);

    const bool use_bin12 = is_bin12_file(files[0]);
    const bool use_png   = !use_bin12 && is_png_file(files[0]);

    // ------------------------------------------------------------------
    // 3. Single pass through the Z-stack — parallelised across files (O-4).
    //    Each file is opened exactly ONCE; all needed pixels are extracted.
    //    Safe: each iteration owns its file handle, uses a thread_local row
    //    buffer and writes signals[pi][new_z] for its own new_z only.
    //    Cancellation check per file: with PNG each iteration decodes a full
    //    image, so an uncancellable batch could take minutes (B-16).
    //    Exceptions must not escape the OpenMP region (B-9).
    // ------------------------------------------------------------------
    {
        std::exception_ptr read_error = nullptr;
        std::mutex read_error_mutex;

#pragma omp parallel for schedule(dynamic)
        for (int new_z = 0; new_z < Nz; ++new_z)
        {
            try
            {
                if (cancel_requested())
                    continue;

                thread_local std::vector<float> rowbuf;
                rowbuf.resize(static_cast<size_t>(Nx));

                const int old_z = stack_info.z_order[new_z];
                const std::string &fpath = files[static_cast<size_t>(old_z)];

                if (use_png)
                {
                    // Load full image once; then scatter into all pixel signals.
                    cv::Mat img = cv::imread(fpath, cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH);
                    if (img.empty())
                        throw std::runtime_error("PNG read failed: " + fpath);
                    if (img.cols != Nx || img.rows != Ny)
                        throw std::runtime_error("PNG dimensions differ from stack: " + fpath);

                    for (auto &[row, pxlist] : row_map)
                    {
                        read_png_row_to_float(img, row, Nx, rowbuf.data(), channel_weights);
                        for (auto &[pi, px_col] : pxlist)
                            signals[pi][new_z] = rowbuf[px_col];
                    }
                }
                else if (use_bin12)
                {
                    FilePtr f(std::fopen(fpath.c_str(), "rb"));
                    if (!f)
                        throw std::runtime_error("bin12 open failed: " + fpath);
                    auto b12hdr = open_bin12(f.get(), fpath);
                    if (int(b12hdr.width) != Nx || int(b12hdr.height) != Ny)
                        throw std::runtime_error("bin12 dimensions differ from stack: " + fpath);

                    for (auto &[row, pxlist] : row_map)
                    {
                        read_one_scanline_from_bin12(f.get(), b12hdr, static_cast<uint32_t>(row),
                                                    Nx, rowbuf.data(), channel_weights);
                        for (auto &[pi, px_col] : pxlist)
                            signals[pi][new_z] = rowbuf[px_col];
                    }
                }
                else // TIFF
                {
                    TiffPtr tif(TIFFOpen(fpath.c_str(), "r"));
                    if (!tif)
                        throw std::runtime_error("TIFF open failed: " + fpath);
                    uint32_t fw = 0, fh = 0;
                    if (TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &fw) != 1 ||
                        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &fh) != 1 ||
                        int(fw) != Nx || int(fh) != Ny)
                        throw std::runtime_error("TIFF dimensions differ from stack: " + fpath);

                    for (auto &[row, pxlist] : row_map)
                    {
                        read_one_scanline_to_float(tif.get(), static_cast<uint32_t>(row),
                                                  Nx, rowbuf.data(), channel_weights);
                        for (auto &[pi, px_col] : pxlist)
                            signals[pi][new_z] = rowbuf[px_col];
                    }
                }
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(read_error_mutex);
                if (!read_error)
                    read_error = std::current_exception();
                request_cancel(true); // make remaining iterations bail out fast
            }
        }
        if (read_error)
            std::rethrow_exception(read_error);
        if (cancel_requested())
            return;
    }

    // ------------------------------------------------------------------
    // 4. Create output folder once (before the parallel section)
    //    out_dir overrides the default so the caller can stage results in a
    //    temporary folder and atomically rename it afterwards (B-12).
    // ------------------------------------------------------------------
    fs::path folder = out_dir.empty() ? fs::path(cfg::OUTPUT_FOLDER) / base_name
                                      : fs::path(out_dir);
    fs::create_directories(folder);

    const methods::MethodInfo *method_info = methods::find(method);
    if (!method_info)
        throw std::runtime_error(
            "save_pixel_plots_batch_streaming: unknown method id " + std::to_string(method));

    const float alpha_m3 = 4.0f * float(M_PI)
        * float(cfg::NOMINAL_DZ_NM) / float(cfg::LAMBDA0_NM);

    // ------------------------------------------------------------------
    // 5. Baseline + envelope + save — parallel across pixels
    //    Each thread works on its own signals[i], so no races.
    //    HilbertEnvelope uses thread-local FFTW plans (see hilbert.hpp).
    // ------------------------------------------------------------------
    // Exceptions (e.g. save_npy on a full disk) must not escape the OpenMP
    // region: capture the first one and rethrow after the region (B-9).
    std::exception_ptr batch_error = nullptr;
    std::mutex batch_error_mutex;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i)
    {
        try
        {
            if (cancel_requested())
                continue;

            const int py     = pixels[i].first;
            const int px_col = pixels[i].second;
            auto &sig        = signals[i];

            // Baseline removal (when needed by the method)
            if (method_info->needs_baseline_removal)
            {
                auto baseline = gaussian_filter_1d(sig, cfg::BASELINE_SIGMA);
                for (int z = 0; z < Nz; ++z)
                    sig[z] -= baseline[z];
            }

            // Envelope — thread_local avoids re-creating FFTW plans each iteration
            thread_local HilbertEnvelope tl_hilb;
            thread_local HilbertEnvelope::ComplexBuf tl_fbuf;
            std::vector<float> env;
            compute_envelope(*method_info, sig, tl_hilb, tl_fbuf, env, k_avg, dk, alpha_m3);
            env = gaussian_filter_1d(env, cfg::ENVELOPE_SIGMA);

            const float h_est = method_info->use_peak_locator
                ? find_envelope_peak(env, positions_sorted)
                : centroid_with_refinement(env, positions_sorted);

            // Pack (Nz × 4): [position, signal, envelope, h_est (repeated)]
            std::vector<float> combined(static_cast<size_t>(Nz) * 4);
            for (int z = 0; z < Nz; ++z)
            {
                combined[size_t(z) * 4 + 0] = positions_sorted[z];
                combined[size_t(z) * 4 + 1] = sig[z];
                combined[size_t(z) * 4 + 2] = env[z];
                combined[size_t(z) * 4 + 3] = h_est;
            }

            const std::string stem = base_name + "_pixel_y" + std::to_string(py)
                                               + "_x"       + std::to_string(px_col);
            const std::string fout = (folder / (stem + ".npy")).string();
            save_npy(fout, combined.data(), static_cast<size_t>(Nz), 4);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(batch_error_mutex);
            if (!batch_error)
                batch_error = std::current_exception();
            request_cancel(true); // make remaining iterations bail out fast
        }
    }
    if (batch_error)
        std::rethrow_exception(batch_error);
}
