#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>

#include "raster/math/vec.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/pipeline/pbr.hpp"

namespace raster::render {
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
            const Texture2D<LayoutTiled4x4>* metallic_roughness = nullptr
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
        const Texture2D<LayoutTiled4x4>* metallic_roughness = nullptr
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
