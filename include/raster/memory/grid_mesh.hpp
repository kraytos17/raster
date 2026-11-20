#pragma once

#include <cstddef>
#include <cstdint>

#include "raster/math/vec.hpp"
#include "raster/memory/soa_mesh.hpp"

namespace raster {
    // Procedural grid-floor stress mesh: columns x rows quads (two triangles
    // each) on the y == 0 plane. Row 0 is far (-Z), higher rows are nearer.
    // Winding matches the bundled floor assets: (near-left, near-right,
    // far-right) + (near-left, far-right, far-left), counter-clockwise as
    // seen in NDC, front-facing with a +Y geometric normal.
    [[nodiscard]]
    inline SoAMesh make_grid_mesh(
        std::size_t columns,
        std::size_t rows,
        float half_extent,
        float near_z,
        float far_z,
        float uv_repeat
    ) {
        SoAMesh mesh{};
        if (columns == 0U || rows == 0U) {
            return mesh;
        }

        mesh.positions.reserve((columns + 1U) * (rows + 1U));
        mesh.normals.reserve((columns + 1U) * (rows + 1U));
        mesh.uvs.reserve((columns + 1U) * (rows + 1U));
        mesh.indices.reserve(columns * rows * 6U);
        for (std::size_t row = 0; row <= rows; ++row) {
            const auto v = static_cast<float>(row) / static_cast<float>(rows);
            for (std::size_t column = 0; column <= columns; ++column) {
                const auto u = static_cast<float>(column) / static_cast<float>(columns);
                mesh.positions.push_back(
                    Vec3f{
                        .x = (u * 2.0F - 1.0F) * half_extent,
                        .y = 0.0F,
                        .z = far_z + (near_z - far_z) * v,
                    }
                );

                mesh.normals.push_back(Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F});
                mesh.uvs.push_back(Vec2f{.x = u * uv_repeat, .y = v * uv_repeat});
            }
        }

        const auto stride = columns + 1U;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const auto top_left = static_cast<std::uint32_t>(row * stride + column);
                const auto top_right = top_left + 1U;
                const auto bottom_left = static_cast<std::uint32_t>((row + 1U) * stride + column);
                const auto bottom_right = bottom_left + 1U;
                mesh.indices.insert(
                    mesh.indices.end(),
                    {bottom_left, bottom_right, top_right, bottom_left, top_right, top_left}
                );
            }
        }
        return mesh;
    }
} // namespace raster
