#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

/**
 * @brief Save a 2‑D matrix (rows × cols) of float32 values in NumPy '.npy' format.
 *
 * The function writes a NumPy v1.0 compliant '.npy' file.
 *
 * Format specification (NPY v1.0):
 *   - Magic string: "\x93NUMPY"
 *   - Version: (1, 0)
 *   - Header: Python dictionary in ASCII with fields:
 *         'descr': '<f4'         // float32, little‑endian
 *         'fortran_order': False // C‑order (row‑major)
 *         'shape': (rows, cols)  // tuple describing the matrix shape
 *
 *   - The header is padded with spaces so that:
 *
 *         (10 + header_size) % 16 == 15
 *
 *     and is terminated by a newline (required by the format).
 *
 *   - Afterwards, the raw binary data is written in row‑major order,
 *     as contiguous float32 values.
 *
 * @param fname Path to the output file (e.g., '"output/xx_height.npy"').
 * @param data  Pointer to contiguous row‑major float32 data.
 * @param rows  Number of matrix rows.
 * @param cols  Number of matrix columns.
 *
 * @throws std::runtime_error If the output file cannot be opened for writing,
 *         or if any write fails (e.g. disk full) — verified after flushing.
 *
 * @note The resulting file is fully compatible with NumPy’s 'numpy.load()'.
 * @note It assumes a little‑endian architecture (standard for x86/x64 systems).
 * @note No nullptr check is performed if rows*cols > 0.
 */
inline void save_npy(const std::string &fname,
                     const float *data,
                     size_t rows,
                     size_t cols)
{
    // Open the file in binary mode; throw if not possible.
    std::ofstream f(fname, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open file for writing: " + fname);

    // Python‑style header (ASCII), required by .npy v1.0.
    // '<f4'            => little‑endian float32
    // fortran_order=0  => C‑order (row‑major)
    // shape=(rows,cols)
    std::string header =
        "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
        std::to_string(rows) + ", " + std::to_string(cols) + "), }";

    // Pad the header with spaces until (10 + header.size()) % 16 == 15.
    // (10 bytes = magic string 6 bytes + version 2 bytes + header length 2 bytes)
    while ((10 + header.size()) % 16 != 15)
        header.push_back(' ');

    // Header must end with a newline.
    header.push_back('\n');

    // Magic string: \x93 + "NUMPY"
    f.write("\x93NUMPY", 6);

    // Write version 1.0 (major=1, minor=0)
    f.put(0x01);
    f.put(0x00);

    // Write the header length as 2‑byte little‑endian.
    uint16_t hlen = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<char *>(&hlen), 2);

    // Write the full ASCII header.
    f.write(header.c_str(), static_cast<std::streamsize>(header.size()));

    // Write the contiguous row‑major float32 data.
    f.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(sizeof(float) * rows * cols));

    // Verify that every write actually reached the file.  Without this check a
    // full disk produces a silently truncated .npy that is reported as success.
    f.close(); // flushes; any pending error sets failbit/badbit
    if (f.fail())
        throw std::runtime_error("Failed to write file (disk full?): " + fname);
}