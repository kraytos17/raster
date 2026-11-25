#pragma once

// Depth-only light pass: prepares the mesh in light space and rasterizes
// every triangle into the shadow depth buffer, reusing the rasterizer's
// depth test/write. The no-op fragment proves the point: depth happens
// before shading.

#include <chrono>
#include <cstddef>
#include <mdspan>
#include <print>
#include <SDL3/SDL.h>
#include <vector>

#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/render/detail/scene_common.hpp"
#include "raster/render/detail/scene_mesh.hpp"
#include "raster/render/detail/scene_shader.hpp"
#include "raster/render/scene_renderer.hpp"

namespace raster::render::detail {
    [[nodiscard]]
    inline bool render_shadow_map(
        ShadowMap& shadow,
        DepthShader& light_shader,
        const SoAMesh& mesh,
        std::vector<PreparedTriangle<DepthShader>>& scratch,
        SceneRenderDiagnostics* diagnostics,
        std::span<const SceneChunk> scene_chunks,
        const Vec3f& lod_reference
    ) noexcept {
        static const bool profile =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PROFILE") != nullptr;
        const auto timestamp_begin = steady_clock::now();
        if (!prepare_mesh(
                light_shader,
                mesh,
                ShadowMap::k_size,
                ShadowMap::k_size,
                false,
                0U,
                scene_chunks.empty() ? mesh.triangle_count() : 0U,
                0U,
                scene_chunks.size(),
                scratch,
                diagnostics,
                scene_chunks,
                lod_reference
            )) {
            return false;
        }

        shadow.clear();
        const auto timestamp_raster = steady_clock::now();
        auto depth_view = std::mdspan<float, Extents2D>{
            shadow.depth.data(),
            ShadowMap::k_size,
            ShadowMap::k_size,
        };

        const auto record_nothing = [](std::size_t, std::size_t, const QuadBarycentrics&) noexcept {
        };
        for (const auto& triangle : scratch) {
            rasterize_triangle_quads(
                triangle.setup,
                ShadowMap::k_size,
                ShadowMap::k_size,
                depth_view,
                record_nothing
            );
        }
        if (profile) {
            using Milliseconds = std::chrono::duration<double, std::milli>;
            try {
                std::println(
                    stderr,
                    "raster: profile shadow {:.2f} ms tris {} (prepare {:.2f} raster {:.2f})",
                    Milliseconds(steady_clock::now() - timestamp_begin).count(),
                    scratch.size(),
                    Milliseconds(timestamp_raster - timestamp_begin).count(),
                    Milliseconds(steady_clock::now() - timestamp_raster).count()
                );
            } catch (...) {
                try {
                    std::println(stderr, "raster: profile output failed");
                } catch (...) { //NOLINT
                    // Abandon profiling output rather than terminate.
                }
            }
        }
        return true;
    }
} // namespace raster::render::detail
