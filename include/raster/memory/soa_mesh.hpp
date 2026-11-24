#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
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

    // Concatenate mesh parts into one draw, preserving part order so callers
    // can build one chunk per part from the recorded triangle offsets.
    // Null parts are skipped; an empty input yields an empty mesh.
    [[nodiscard]]
    inline SoAMesh combine_meshes(std::span<const SoAMesh* const> parts) {
        SoAMesh combined{};
        std::size_t vertex_total = 0U;
        std::size_t index_total = 0U;
        for (const auto* part : parts) {
            if (part == nullptr) {
                continue;
            }

            vertex_total += part->positions.size();
            index_total += part->indices.size();
        }

        combined.positions.reserve(vertex_total);
        combined.normals.reserve(vertex_total);
        combined.uvs.reserve(vertex_total);
        combined.indices.reserve(index_total);
        for (const auto* part : parts) {
            if (part == nullptr) {
                continue;
            }

            const auto base = static_cast<std::uint32_t>(combined.positions.size());
            combined.positions
                .insert(combined.positions.end(), part->positions.begin(), part->positions.end());
            combined.normals
                .insert(combined.normals.end(), part->normals.begin(), part->normals.end());
            combined.uvs.insert(combined.uvs.end(), part->uvs.begin(), part->uvs.end());
            for (const auto index : part->indices) {
                combined.indices.push_back(
                    static_cast<std::uint32_t>(static_cast<std::size_t>(index) + base)
                );
            }
        }
        return combined;
    }
} // namespace raster
