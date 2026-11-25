#pragma once

// Tiled submission: parallel chunk/triangle preparation with an ordered
// merge, CSR binning, then per-tile rasterization. Chunks (not triangles)
// are the parallel work units when a chunk list is present.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <print>
#include <SDL3/SDL.h>
#include <span>
#include <vector>

#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/pipeline/tile_binner.hpp"
#include "raster/pipeline/tile_executor.hpp"
#include "raster/render/detail/scene_common.hpp"
#include "raster/render/detail/scene_mesh.hpp"
#include "raster/render/scene_renderer.hpp"

namespace raster::render::detail {
    template <typename Layout> struct TileRenderState {
        explicit TileRenderState(std::size_t concurrency) :
            executor{concurrency} {}

        void reserve(
            std::size_t width,
            std::size_t height,
            std::size_t mesh_triangle_count,
            std::size_t worker_count
        ) {
            const auto columns =
                width / raster_tile_size + static_cast<std::size_t>(width % raster_tile_size != 0U);
            const auto rows = height
                / raster_tile_size
                + static_cast<std::size_t>(height % raster_tile_size != 0U);
            const auto tile_count = columns * rows;

            prepared.reserve(mesh_triangle_count * 8U);
            setups.reserve(mesh_triangle_count * 8U);
            shadow_prepared.reserve(mesh_triangle_count * 8U);
            shadow_setups.reserve(mesh_triangle_count * 8U);
            bins.reserve(tile_count, tile_count * 8U);
            thread_prepared.resize(worker_count);
            const auto chunk_reserve =
                (mesh_triangle_count + worker_count - 1U) / worker_count * 8U + 1U;
            for (auto& chunk : thread_prepared) {
                chunk.clear();
                chunk.reserve(chunk_reserve);
            }
        }

        TileExecutor executor;
        TileBins bins{};
        std::vector<PreparedTriangle<FloorShader<Layout>>> prepared{};
        std::vector<ScreenTriangle> setups{};
        std::vector<std::vector<PreparedTriangle<FloorShader<Layout>>>> thread_prepared{};
        ShadowMap shadow_map{};
        std::vector<PreparedTriangle<DepthShader>> shadow_prepared{};
        std::vector<ScreenTriangle> shadow_setups{};
        std::vector<std::vector<PreparedTriangle<DepthShader>>> thread_shadow_prepared{};
    };

    // Merges per-worker prepared triangles into one ordered array plus a
    // matching setup array: prefix offsets keep every worker's range in
    // global input order.
    template <typename Shader> inline void merge_worker_prepared(
        std::vector<PreparedTriangle<Shader>>& merged,
        std::vector<ScreenTriangle>& merged_setups,
        const std::vector<std::vector<PreparedTriangle<Shader>>>& per_worker,
        TileExecutor& executor
    ) noexcept {
        std::size_t merged_total = 0U;
        for (const auto& chunk : per_worker) {
            merged_total += chunk.size();
        }
        // Both arrays are resized together: setups is shared between the
        // main and shadow merges with different counts, so checking only one
        // side leaves the other short on the next frame.
        if (merged.size() != merged_total || merged_setups.size() != merged_total) {
            merged.resize(merged_total);
            merged_setups.resize(merged_total);
        }

        executor.execute(per_worker.size(), [&](std::size_t worker) noexcept {
            std::size_t offset = 0U;
            for (std::size_t prior = 0U; prior < worker; ++prior) {
                offset += per_worker[prior].size();
            }

            const auto& chunk = per_worker[worker];
            for (std::size_t index = 0U; index < chunk.size(); ++index) {
                merged[offset + index] = chunk[index];
                merged_setups[offset + index] = chunk[index].setup;
            }
        });
    }

    template <typename Layout> [[nodiscard]]
    bool draw_tiled_mesh(
        Framebuffer& framebuffer,
        FloorShader<Layout>& shader,
        const SoAMesh& mesh,
        bool guard_band_test,
        TileRenderState<Layout>& tile_state,
        bool allow_allocation,
        SceneRenderDiagnostics* diagnostics,
        std::span<const SceneChunk> scene_chunks,
        const Vec3f& lod_reference
    ) noexcept {
        const auto width = framebuffer.width;
        const auto height = framebuffer.height;
        const auto work_count = scene_chunks.empty() ? mesh.triangle_count() : scene_chunks.size();
        const auto worker_count = tile_state.executor.concurrency();
        if (allow_allocation) {
            tile_state.reserve(width, height, mesh.triangle_count(), worker_count);
        }

        std::atomic_bool prepared_ok{true};
        const bool profile =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PROFILE") != nullptr;
        if (profile) {
            profile_covered_lanes.store(0, std::memory_order_relaxed);
            profile_visible_lanes.store(0, std::memory_order_relaxed);
        }

        const auto timestamp_prepare_begin = steady_clock::now();
        tile_state.executor.execute(worker_count, [&](std::size_t worker) noexcept {
            auto& chunk = tile_state.thread_prepared[worker];
            chunk.clear();
            const auto [begin, end] = partition_range(work_count, worker_count, worker);
            if (!prepare_mesh(
                    shader,
                    mesh,
                    width,
                    height,
                    guard_band_test,
                    begin,
                    end,
                    scene_chunks.empty() ? 0U : begin,
                    scene_chunks.empty() ? 0U : end,
                    chunk,
                    diagnostics,
                    scene_chunks,
                    lod_reference
                )) {
                prepared_ok.store(false, std::memory_order_relaxed);
            }
        });
        if (!prepared_ok.load(std::memory_order_relaxed)) {
            return false;
        }

        const auto timestamp_merge_begin = steady_clock::now();
        auto& prepared = tile_state.prepared;
        auto& setups = tile_state.setups;
        merge_worker_prepared(prepared, setups, tile_state.thread_prepared, tile_state.executor);

        const auto timestamp_bin_begin = steady_clock::now();
        bin_triangles(
            tile_state.setups,
            width,
            height,
            tile_state.bins,
            tile_state.executor,
            worker_count
        );

        const auto timestamp_raster_begin = steady_clock::now();
        tile_state.executor.execute(tile_state.bins.tile_count(), [&](std::size_t tile) noexcept {
            const auto tile_x = tile % tile_state.bins.tile_columns();
            const auto tile_y = tile / tile_state.bins.tile_columns();
            const RasterRegion region{
                .min_x = tile_x * raster_tile_size,
                .min_y = tile_y * raster_tile_size,
                .max_x = std::min((tile_x + 1U) * raster_tile_size, width),
                .max_y = std::min((tile_y + 1U) * raster_tile_size, height),
            };
            for (const auto triangle_index : tile_state.bins.triangle_indices(tile)) {
                rasterize_prepared_triangle(
                    framebuffer,
                    shader,
                    tile_state.prepared[triangle_index],
                    region
                );
            }
        });
        if (profile) {
            using Milliseconds = std::chrono::duration<double, std::milli>;
            const auto covered = profile_covered_lanes.load(std::memory_order_relaxed);
            const auto visible = profile_visible_lanes.load(std::memory_order_relaxed);
            try {
                std::println(
                    stderr,
                    "raster: profile prepare {:.2f} ms merge {:.2f} ms bin {:.2f} ms raster {:.2f} ms",
                    Milliseconds(timestamp_merge_begin - timestamp_prepare_begin).count(),
                    Milliseconds(timestamp_bin_begin - timestamp_merge_begin).count(),
                    Milliseconds(timestamp_raster_begin - timestamp_bin_begin).count(),
                    Milliseconds(std::chrono::steady_clock::now() - timestamp_raster_begin).count()
                );
                std::println(
                    stderr,
                    "raster: profile overdraw covered {} visible {} depth-kill {:.1f}%",
                    covered,
                    visible,
                    covered == 0U ? 0.0
                                  : 100.0
                            * static_cast<double>(covered - visible)
                            / static_cast<double>(covered)
                );
            } catch (...) {
                try {
                    std::println(stderr, "raster: profile output failed");
                } catch (...) { //NOLINT
                }
            }
        }
        return true;
    }

    // Depth-only shadow pass over tiles: parallel prepare into per-worker
    // scratch, ordered merge, parallel bin, then one worker per shadow-map
    // tile rasterizing depth with a no-op fragment.
    template <typename Layout> [[nodiscard]]
    bool render_shadow_map_tiled(
        ShadowMap& shadow,
        DepthShader& light_shader,
        const SoAMesh& mesh,
        TileRenderState<Layout>& tile_state,
        std::span<const SceneChunk> scene_chunks,
        const Vec3f& lod_reference
    ) noexcept {
        static const bool profile =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PROFILE") != nullptr;
        const auto timestamp_begin = steady_clock::now();
        const auto worker_count = tile_state.executor.concurrency();
        // Size worker scratch here rather than in reserve(): the shadow pass
        // runs before draw_tiled_mesh, so this is the first touch on a fresh
        // TileRenderState.
        if (tile_state.thread_shadow_prepared.size() != worker_count) {
            tile_state.thread_shadow_prepared.resize(worker_count);
        }

        const auto chunk_reserve =
            (mesh.triangle_count() + worker_count - 1U) / worker_count * 8U + 1U;
        for (auto& chunk : tile_state.thread_shadow_prepared) {
            chunk.clear();
            chunk.reserve(chunk_reserve);
        }

        const auto work_count = scene_chunks.empty() ? mesh.triangle_count() : scene_chunks.size();

        std::atomic_bool prepared_ok{true};
        tile_state.executor.execute(worker_count, [&](std::size_t worker) noexcept {
            auto& chunk = tile_state.thread_shadow_prepared[worker];
            chunk.clear();
            const auto [begin, end] = partition_range(work_count, worker_count, worker);
            if (!prepare_mesh(
                    light_shader,
                    mesh,
                    ShadowMap::k_size,
                    ShadowMap::k_size,
                    false,
                    begin,
                    end,
                    scene_chunks.empty() ? 0U : begin,
                    scene_chunks.empty() ? 0U : end,
                    chunk,
                    nullptr,
                    scene_chunks,
                    lod_reference
                )) {
                prepared_ok.store(false, std::memory_order_relaxed);
            }
        });
        if (!prepared_ok.load(std::memory_order_relaxed)) {
            return false;
        }

        const auto timestamp_merge_begin = steady_clock::now();
        merge_worker_prepared(
            tile_state.shadow_prepared,
            tile_state.shadow_setups,
            tile_state.thread_shadow_prepared,
            tile_state.executor
        );

        const auto timestamp_bin_begin = steady_clock::now();
        bin_triangles(
            tile_state.shadow_setups,
            ShadowMap::k_size,
            ShadowMap::k_size,
            tile_state.bins,
            tile_state.executor,
            worker_count
        );

        const auto timestamp_raster_begin = steady_clock::now();
        shadow.clear();
        auto depth_view = std::mdspan<float, Extents2D>{
            shadow.depth.data(),
            ShadowMap::k_size,
            ShadowMap::k_size,
        };

        const auto record_nothing = [](std::size_t, std::size_t, const QuadBarycentrics&) noexcept {
        };

        tile_state.executor.execute(tile_state.bins.tile_count(), [&](std::size_t tile) noexcept {
            const auto tile_x = tile % tile_state.bins.tile_columns();
            const auto tile_y = tile / tile_state.bins.tile_columns();
            const RasterRegion region{
                .min_x = tile_x * raster_tile_size,
                .min_y = tile_y * raster_tile_size,
                .max_x = std::min((tile_x + 1U) * raster_tile_size, ShadowMap::k_size),
                .max_y = std::min((tile_y + 1U) * raster_tile_size, ShadowMap::k_size),
            };
            for (const auto triangle_index : tile_state.bins.triangle_indices(tile)) {
                rasterize_triangle_quads(
                    tile_state.shadow_prepared[triangle_index].setup,
                    ShadowMap::k_size,
                    ShadowMap::k_size,
                    depth_view,
                    record_nothing,
                    region
                );
            }
        });
        if (profile) {
            using Milliseconds = std::chrono::duration<double, std::milli>;
            try {
                std::println(
                    stderr,
                    "raster: profile shadow-tiled {:.2f} ms tris {} "
                    "(prepare {:.2f} merge {:.2f} bin {:.2f} raster {:.2f})",
                    Milliseconds(std::chrono::steady_clock::now() - timestamp_begin).count(),
                    tile_state.shadow_prepared.size(),
                    Milliseconds(timestamp_merge_begin - timestamp_begin).count(),
                    Milliseconds(timestamp_bin_begin - timestamp_merge_begin).count(),
                    Milliseconds(timestamp_raster_begin - timestamp_bin_begin).count(),
                    Milliseconds(std::chrono::steady_clock::now() - timestamp_raster_begin).count()
                );
            } catch (...) {
                try {
                    std::println(stderr, "raster: profile output failed");
                } catch (...) { //NOLINT
                }
            }
        }
        return true;
    }
} // namespace raster::render::detail
