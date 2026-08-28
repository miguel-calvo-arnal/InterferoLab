#pragma once
#include <vector>

/**
 * @brief Container for a 3‑D image volume accessible as stack(z, y, x).
 *
 * This structure stores a stack of Nz images (each of size Ny × Nx) in a
 * single contiguous row‑major vector:
 *
 *      data[ z * (Ny*Nx) + y * Nx + x ]
 *
 * It also supports Z‑axis reordering without moving data, via the
 * 'z_order' vector. If 'z_order' is empty, accesses are direct. If it
 * contains a permutation, then:
 *
 *      stack(z, y, x)  ->  data[ z_order[z] * (Ny*Nx) + y*Nx + x ]
 *
 * This allows Z‑axis sorting (e.g. temporal or positional reordering)
 * without copying gigabytes of data.
 */
struct ImageStack
{
    int Nx = 0; ///< Number of columns (X‑axis)
    int Ny = 0; ///< Number of rows    (Y‑axis)
    int Nz = 0; ///< Number of slices  (Z‑axis)

    /// Linear data buffer of size Nx * Ny * Nz.
    std::vector<float> data;

    /**
     * @brief Optional mapping that reorders the Z‑axis.
     *
     * If empty -> direct indexing.
     * If it contains a permutation (e.g. {5,0,1,3,2,...}), then:
     *
     *      stack(z, y, x) ⇒ data[ z_order[z] * (Ny*Nx) + ... ]
     *
     * This enables Z‑axis reordering without relocating large buffers.
     */
    std::vector<int> z_order;

    /**
     * @brief Mutable voxel access at (z,y,x).
     *
     * Applies z_order if present.
     *
     * @note No bounds checking is performed for efficiency.
     */
    inline float &operator()(int z, int y, int x)
    {
        const int real_z =
            (z_order.empty() ? z : z_order[z]);

        return data[static_cast<size_t>(real_z) * Ny * Nx +
                    static_cast<size_t>(y) * Nx +
                    static_cast<size_t>(x)];
    }

    /**
     * @brief Const voxel access at (z,y,x).
     *
     * Same as the mutable version but read‑only.
     */
    inline const float &operator()(int z, int y, int x) const
    {
        const int real_z =
            (z_order.empty() ? z : z_order[z]);

        return data[static_cast<size_t>(real_z) * Ny * Nx +
                    static_cast<size_t>(y) * Nx +
                    static_cast<size_t>(x)];
    }
};