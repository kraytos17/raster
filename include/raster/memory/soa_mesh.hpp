#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "raster/math/vec.hpp"
#include "raster/memory/framebuffer.hpp"

namespace raster {
    template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;

    // Geometry streams are separate, contiguous, 64-byte-aligned arrays.
    // indices contains unified SoA vertex indices in triangle triplets.
    struct SoAMesh {
        AlignedVector<Vec3f> positions;
        AlignedVector<Vec3f> normals;
        AlignedVector<Vec2f> uvs;
        AlignedVector<std::uint32_t> indices;

        [[nodiscard]]
        std::size_t triangle_count() const noexcept {
            return indices.size() / 3;
        }

        [[nodiscard]]
        bool has_consistent_vertex_streams() const noexcept {
            return positions.size()
                == normals.size()
                && positions.size()
                == uvs.size()
                && indices.size()
                % 3
                == 0;
        }
    };
} // namespace raster
