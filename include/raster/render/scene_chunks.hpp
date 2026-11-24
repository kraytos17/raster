#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "raster/math/mat.hpp"
#include "raster/memory/soa_mesh.hpp"

namespace raster::render {
    // A contiguous triangle range with a conservative object-space sphere.
    // The sphere is built from every referenced vertex; it is safe for coarse
    // rejection but is not intended as a tight meshlet bound.
    struct SceneChunk {
        std::size_t first_triangle{};
        std::size_t triangle_count{};
        Vec3f bounds_center{};
        float bounds_radius{};

        [[nodiscard]]
        float distance_to(const Vec3f& point) const noexcept {
            return std::max(length(point - bounds_center) - bounds_radius, 0.0F);
        }
    };

    [[nodiscard]]
    inline bool valid_scene_chunks(
        std::span<const SceneChunk> chunks, std::size_t triangle_count
    ) noexcept {
        if (chunks.empty()) {
            return true;
        }

        std::size_t expected_first = 0U;
        for (const auto& chunk : chunks) {
            if (chunk.first_triangle
                != expected_first
                || chunk.triangle_count
                == 0U
                || chunk.first_triangle
                > triangle_count
                || chunk.triangle_count
                > triangle_count
                - chunk.first_triangle
                || !std::isfinite(chunk.bounds_center.x)
                || !std::isfinite(chunk.bounds_center.y)
                || !std::isfinite(chunk.bounds_center.z)
                || !std::isfinite(chunk.bounds_radius)
                || chunk.bounds_radius
                < 0.0F) {
                return false;
            }
            expected_first += chunk.triangle_count;
        }
        return expected_first == triangle_count;
    }

    // Conservative sphere-vs-canonical-clip-plane test. Each clip inequality
    // is linear in object coordinates, so the transformed sphere's maximum
    // plane distance is center_distance + radius * |plane.xyz|. Non-finite
    // inputs and invalid bounds conservatively keep the chunk.
    [[nodiscard]]
    inline bool chunk_outside_clip_frustum(
        const Mat4f& model_view_projection, const SceneChunk& chunk, float guard_band = 1.0F
    ) noexcept {
        if (!std::isfinite(guard_band)
            || guard_band
            < 1.0F
            || !std::isfinite(chunk.bounds_radius)
            || chunk.bounds_radius
            < 0.0F) {
            return false;
        }

        const Vec4f center_clip = model_view_projection
            * Vec4f{
                .x = chunk.bounds_center.x,
                .y = chunk.bounds_center.y,
                .z = chunk.bounds_center.z,
                .w = 1.0F,
            };
        if (!std::isfinite(center_clip.x)
            || !std::isfinite(center_clip.y)
            || !std::isfinite(center_clip.z)
            || !std::isfinite(center_clip.w)) {
            return false;
        }

        const std::array<std::array<float, 4>, 6> plane_values{
            {
                {
                    guard_band * center_clip.w + center_clip.x,
                    guard_band * model_view_projection[3U, 0U] + model_view_projection[0U, 0U],
                    guard_band * model_view_projection[3U, 1U] + model_view_projection[0U, 1U],
                    guard_band * model_view_projection[3U, 2U] + model_view_projection[0U, 2U],
                },
                {
                    guard_band * center_clip.w - center_clip.x,
                    guard_band * model_view_projection[3U, 0U] - model_view_projection[0U, 0U],
                    guard_band * model_view_projection[3U, 1U] - model_view_projection[0U, 1U],
                    guard_band * model_view_projection[3U, 2U] - model_view_projection[0U, 2U],
                },
                {
                    guard_band * center_clip.w + center_clip.y,
                    guard_band * model_view_projection[3U, 0U] + model_view_projection[1U, 0U],
                    guard_band * model_view_projection[3U, 1U] + model_view_projection[1U, 1U],
                    guard_band * model_view_projection[3U, 2U] + model_view_projection[1U, 2U],
                },
                {
                    guard_band * center_clip.w - center_clip.y,
                    guard_band * model_view_projection[3U, 0U] - model_view_projection[1U, 0U],
                    guard_band * model_view_projection[3U, 1U] - model_view_projection[1U, 1U],
                    guard_band * model_view_projection[3U, 2U] - model_view_projection[1U, 2U],
                },
                {
                    center_clip.z,
                    model_view_projection[2U, 0U],
                    model_view_projection[2U, 1U],
                    model_view_projection[2U, 2U],
                },
                {
                    center_clip.w - center_clip.z,
                    model_view_projection[3U, 0U] - model_view_projection[2U, 0U],
                    model_view_projection[3U, 1U] - model_view_projection[2U, 1U],
                    model_view_projection[3U, 2U] - model_view_projection[2U, 2U],
                },
            },
        };
        for (const auto& plane : plane_values) {
            const auto variation = chunk.bounds_radius
                * std::sqrt(plane[1] * plane[1] + plane[2] * plane[2] + plane[3] * plane[3]);
            if (!std::isfinite(variation) || !std::isfinite(plane[0])) {
                return false;
            }
            if (plane[0] + variation < 0.0F) {
                return true;
            }
        }
        return false;
    }

    // Conservative AABB-centered sphere over one contiguous triangle
    // range. An empty or out-of-range request yields a zero-triangle chunk,
    // which valid_scene_chunks rejects. Call during scene setup, not per frame.
    [[nodiscard]]
    inline SceneChunk make_chunk_bounds(
        const SoAMesh& mesh, std::size_t first_triangle, std::size_t triangle_count
    ) noexcept {
        if (triangle_count
            == 0U
            || first_triangle
            >= mesh.triangle_count()
            || triangle_count
            > mesh.triangle_count()
            - first_triangle) {
            return SceneChunk{.first_triangle = first_triangle, .triangle_count = 0U};
        }

        Vec3f minimum{
            .x = std::numeric_limits<float>::infinity(),
            .y = std::numeric_limits<float>::infinity(),
            .z = std::numeric_limits<float>::infinity(),
        };
        Vec3f maximum{
            .x = -std::numeric_limits<float>::infinity(),
            .y = -std::numeric_limits<float>::infinity(),
            .z = -std::numeric_limits<float>::infinity(),
        };
        for (std::size_t triangle = first_triangle; triangle < first_triangle + triangle_count;
             ++triangle) {
            for (std::size_t corner = 0U; corner < 3U; ++corner) {
                const auto index = static_cast<std::size_t>(mesh.indices[triangle * 3U + corner]);
                const auto& point = mesh.positions[index];
                minimum.x = std::min(minimum.x, point.x);
                minimum.y = std::min(minimum.y, point.y);
                minimum.z = std::min(minimum.z, point.z);
                maximum.x = std::max(maximum.x, point.x);
                maximum.y = std::max(maximum.y, point.y);
                maximum.z = std::max(maximum.z, point.z);
            }
        }

        const auto center = Vec3f{
            .x = (minimum.x + maximum.x) * 0.5F,
            .y = (minimum.y + maximum.y) * 0.5F,
            .z = (minimum.z + maximum.z) * 0.5F,
        };

        float radius_squared = 0.0F;
        for (std::size_t triangle = first_triangle; triangle < first_triangle + triangle_count;
             ++triangle) {
            for (std::size_t corner = 0U; corner < 3U; ++corner) {
                const auto index = static_cast<std::size_t>(mesh.indices[triangle * 3U + corner]);
                radius_squared =
                    std::max(radius_squared, length_squared(mesh.positions[index] - center));
            }
        }
        return SceneChunk{
            .first_triangle = first_triangle,
            .triangle_count = triangle_count,
            .bounds_center = center,
            .bounds_radius = std::sqrt(radius_squared),
        };
    }

    // Partition a mesh into stable consecutive triangle ranges and precompute
    // an AABB-centered sphere for each range. Returns no chunks for an empty
    // mesh or zero requested chunk size. Call during scene setup, not per frame.
    [[nodiscard]]
    inline std::vector<SceneChunk> make_scene_chunks(
        const SoAMesh& mesh, std::size_t triangles_per_chunk
    ) {
        std::vector<SceneChunk> chunks;
        if (triangles_per_chunk == 0U || mesh.triangle_count() == 0U) {
            return chunks;
        }

        chunks.reserve(
            mesh.triangle_count()
            / triangles_per_chunk
            + static_cast<std::size_t>(mesh.triangle_count() % triangles_per_chunk != 0U)
        );
        for (std::size_t first = 0U; first < mesh.triangle_count();) {
            const auto count = std::min(triangles_per_chunk, mesh.triangle_count() - first);
            chunks.push_back(make_chunk_bounds(mesh, first, count));
            first += count;
        }
        return chunks;
    }
} // namespace raster::render
