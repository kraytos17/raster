#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>

#include "raster/math/vec.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/pipeline/pbr.hpp"
#include "raster/render/scene_chunks.hpp"

namespace raster::render {
    struct SceneRenderDiagnostics {
        std::atomic_size_t source_triangles{};
        std::atomic_size_t frustum_rejected_triangles{};
        std::atomic_size_t source_chunks{};
        std::atomic_size_t frustum_rejected_chunks{};
        std::atomic_size_t clipped_triangles{};
        std::atomic_size_t prepared_triangles{};
        std::atomic_size_t rasterized_triangle_submissions{};

        void reset() noexcept {
            source_triangles.store(0U, std::memory_order_relaxed);
            frustum_rejected_triangles.store(0U, std::memory_order_relaxed);
            source_chunks.store(0U, std::memory_order_relaxed);
            frustum_rejected_chunks.store(0U, std::memory_order_relaxed);
            clipped_triangles.store(0U, std::memory_order_relaxed);
            prepared_triangles.store(0U, std::memory_order_relaxed);
            rasterized_triangle_submissions.store(0U, std::memory_order_relaxed);
        }

        // Plain per-call accumulator: one atomic RMW per triangle stalls
        // every worker on the same cache lines (the tiled path hammers seven
        // shared counters from every thread), so hot loops count locally and
        // merge once per call.
        struct Local {
            std::size_t source_triangles{};
            std::size_t frustum_rejected_triangles{};
            std::size_t source_chunks{};
            std::size_t frustum_rejected_chunks{};
            std::size_t clipped_triangles{};
            std::size_t prepared_triangles{};
            std::size_t rasterized_triangle_submissions{};

            void merge_into(SceneRenderDiagnostics* target) const noexcept {
                if (target == nullptr) {
                    return;
                }

                target->source_triangles.fetch_add(source_triangles, std::memory_order_relaxed);
                target->frustum_rejected_triangles.fetch_add(
                    frustum_rejected_triangles,
                    std::memory_order_relaxed
                );

                target->source_chunks.fetch_add(source_chunks, std::memory_order_relaxed);
                target->frustum_rejected_chunks.fetch_add(
                    frustum_rejected_chunks,
                    std::memory_order_relaxed
                );

                target->clipped_triangles.fetch_add(clipped_triangles, std::memory_order_relaxed);
                target->prepared_triangles.fetch_add(prepared_triangles, std::memory_order_relaxed);
                target->rasterized_triangle_submissions.fetch_add(
                    rasterized_triangle_submissions,
                    std::memory_order_relaxed
                );
            }
        };
    };

    // Diffuse preserves the legacy byte-oriented path. Pbr requires an Srgb
    // base-color texture; material holds linear-light factors. An optional
    // metallic-roughness texture carries linear data (G = roughness,
    // B = metallic, R/A ignored); absent channels default to 1. One
    // descriptor applies per draw: the mesh pipeline has no per-face
    // material groups, so per-triangle materials are out of scope.
    enum class SceneShading {
        Diffuse,
        Pbr,
    };

    struct FreeLookCamera {
        Vec3f position{};
        float yaw{};
        float pitch{};
    };

    // Keep perspective zoom responsive: farther camera positions move faster,
    // with bounds to avoid an abrupt speed spike close to or far from the mesh.
    [[nodiscard]]
    inline constexpr float free_look_movement_speed(const Vec3f& camera_position) noexcept {
        return std::clamp(length(camera_position) * 0.8F, 2.5F, 20.0F);
    }

    class TileRenderer {
    public:

        explicit TileRenderer(std::size_t concurrency);
        TileRenderer(const TileRenderer&) = delete;
        TileRenderer& operator=(const TileRenderer&) = delete;
        TileRenderer(TileRenderer&&) noexcept;
        TileRenderer& operator=(TileRenderer&&) noexcept;
        ~TileRenderer();

        [[nodiscard]]
        bool render_scene(
            Framebuffer& framebuffer,
            const SoAMesh& mesh,
            std::uint64_t elapsed_ns,
            bool clip_stress_test,
            bool free_look_test,
            bool guard_band_test,
            bool solid_color_debug,
            const Texture2D<LayoutTiled4x4>& texture,
            const FreeLookCamera& camera,
            SceneShading shading = SceneShading::Diffuse,
            const pbr::Material& material = pbr::Material(),
            const Texture2D<LayoutTiled4x4>* metallic_roughness = nullptr,
            SceneRenderDiagnostics* diagnostics = nullptr,
            std::span<const SceneChunk> chunks = {},
            const Vec3f* lod_reference = nullptr
        ) noexcept;

    private:

        struct Impl;
        std::unique_ptr<Impl> impl_{};
    };

    // Render the current procedural checker-floor scene or the deterministic
    // camera sweep used by the clipping stress gate. Returns false if clipping
    // rejects malformed geometry.
    [[nodiscard]]
    bool render_scene(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        bool clip_stress_test,
        bool free_look_test,
        bool guard_band_test,
        bool solid_color_debug,
        const Texture2D<LayoutTiled4x4>& texture,
        const FreeLookCamera& camera,
        SceneShading shading = SceneShading::Diffuse,
        const pbr::Material& material = pbr::Material(),
        const Texture2D<LayoutTiled4x4>* metallic_roughness = nullptr,
        SceneRenderDiagnostics* diagnostics = nullptr,
        std::span<const SceneChunk> chunks = {},
        const Vec3f* lod_reference = nullptr
    ) noexcept;

    [[nodiscard]]
    bool render_scene_row_major(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        const Texture2D<std::layout_right>& texture,
        const FreeLookCamera& camera
    ) noexcept;

    [[nodiscard]]
    bool render_scene_morton(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        const Texture2D<LayoutMorton>& texture,
        const FreeLookCamera& camera
    ) noexcept;

    [[nodiscard]]
    bool render_textured_model(
        Framebuffer& framebuffer, const SoAMesh& mesh, const Texture2D<std::layout_right>& texture
    ) noexcept;

    [[nodiscard]]
    bool render_textured_model(
        Framebuffer& framebuffer, const SoAMesh& mesh, const Texture2D<LayoutTiled4x4>& texture
    ) noexcept;

    [[nodiscard]]
    bool render_textured_model(
        Framebuffer& framebuffer, const SoAMesh& mesh, const Texture2D<LayoutMorton>& texture
    ) noexcept;
} // namespace raster::render
