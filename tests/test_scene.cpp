#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mdspan>
#include <numbers>
#include <optional>
#include <ranges>
#include <SDL3/SDL.h>
#include <tuple>
#include <vector>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/memory/grid_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/pipeline/clipper.hpp"
#include "raster/pipeline/pbr.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/shader.hpp"
#include "raster/platform/allocation_probe.hpp"
#include "raster/render/scene_renderer.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    raster::SoAMesh make_two_sided_triangle(raster::Vec3f normal);
    raster::SoAMesh make_shadow_scene();

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("tiled scene renderer matches serial color and depth", "[scene][tile]") {
        constexpr std::size_t width = 96U;
        constexpr std::size_t height = 80U;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage image{};
        image.width = 8U;
        image.height = 8U;
        image.pixels.resize(64U);
        for (std::size_t y = 0; y < image.height; ++y) {
            for (std::size_t x = 0; x < image.width; ++x) {
                image.pixels[y * image.width + x] =
                    ((x + y) & 1U) == 0U ? 0xFFFFAA33U : 0xFF3377DDU;
            }
        }
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.1F, .y = -0.2F, .z = 3.5F},
            .yaw = 0.07F,
            .pitch = 0.03F,
        };
        raster::Framebuffer serial{width, height};
        raster::Framebuffer tiled{width, height};
        raster::render::TileRenderer renderer{4U};
        REQUIRE(
            renderer.render_scene(tiled, *mesh, 0U, false, true, false, false, texture, camera)
        );
        REQUIRE(
            raster::render::
                render_scene(serial, *mesh, 0U, false, true, false, false, texture, camera)
        );
        REQUIRE(tiled.color_storage == serial.color_storage);
        REQUIRE(tiled.depth_storage == serial.depth_storage);
        std::size_t allocation_count = 0U;
        bool repeated_rendered = false;
        {
            const raster::platform::ScopedAllocationProbe probe{true};
            repeated_rendered =
                renderer.render_scene(tiled, *mesh, 0U, false, true, false, false, texture, camera);
            allocation_count = probe.allocations();
        }
        REQUIRE(repeated_rendered);
        REQUIRE(tiled.color_storage == serial.color_storage);
        REQUIRE(tiled.depth_storage == serial.depth_storage);
        REQUIRE(allocation_count == 0U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("fully offscreen mesh writes no color or depth lanes", "[scene][m3][frustum]") {
        constexpr std::size_t width = 80U;
        constexpr std::size_t height = 64U;
        raster::SoAMesh offscreen{};
        offscreen.positions = {
            raster::Vec3f{.x = 1000.0F, .y = 0.0F, .z = -2.0F},
            raster::Vec3f{.x = 1001.0F, .y = 0.0F, .z = -2.0F},
            raster::Vec3f{.x = 1000.0F, .y = 1.0F, .z = -2.0F},
        };
        offscreen.normals = {
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
        };
        offscreen.uvs = {
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 1.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 1.0F},
        };
        offscreen.indices = {0U, 1U, 2U};
        REQUIRE(offscreen.has_consistent_vertex_streams());

        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFFFFFFFFU};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.5F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        constexpr std::uint32_t clear = 0xFF101018U;
        raster::Framebuffer serial{width, height};
        raster::Framebuffer framebuffer{width, height};
        raster::render::TileRenderer renderer{3U};
        raster::render::SceneRenderDiagnostics serial_diagnostics{};
        raster::render::SceneRenderDiagnostics tiled_diagnostics{};
        REQUIRE(
            raster::render::render_scene(
                serial,
                offscreen,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera,
                raster::render::SceneShading::Diffuse,
                raster::pbr::Material(),
                nullptr,
                &serial_diagnostics
            )
        );
        REQUIRE(renderer.render_scene(
            framebuffer,
            offscreen,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Diffuse,
            raster::pbr::Material(),
            nullptr,
            &tiled_diagnostics
        ));
        const auto verify_early_reject = [](const raster::render::SceneRenderDiagnostics& stats) {
            REQUIRE(stats.source_triangles == 1U);
            REQUIRE(stats.frustum_rejected_triangles == 1U);
            REQUIRE(stats.clipped_triangles == 0U);
            REQUIRE(stats.prepared_triangles == 0U);
            REQUIRE(stats.rasterized_triangle_submissions == 0U);
        };
        verify_early_reject(serial_diagnostics);
        verify_early_reject(tiled_diagnostics);

        const auto color = std::as_const(framebuffer).color_view();
        const auto depth = std::as_const(framebuffer).depth_view();
        const auto serial_color = std::as_const(serial).color_view();
        const auto serial_depth = std::as_const(serial).depth_view();
        std::size_t non_clear_pixels = 0U;
        std::size_t written_depths = 0U;
        std::size_t serial_non_clear_pixels = 0U;
        std::size_t serial_written_depths = 0U;
        for (std::size_t y = 0U; y < height; ++y) {
            for (std::size_t x = 0U; x < width; ++x) {
                non_clear_pixels += color[y, x] != clear ? 1U : 0U;
                written_depths += depth[y, x] != 0.0F ? 1U : 0U;
                serial_non_clear_pixels += serial_color[y, x] != clear ? 1U : 0U;
                serial_written_depths += serial_depth[y, x] != 0.0F ? 1U : 0U;
            }
        }
        REQUIRE(non_clear_pixels == 0U);
        REQUIRE(written_depths == 0U);
        REQUIRE(serial_non_clear_pixels == 0U);
        REQUIRE(serial_written_depths == 0U);
        REQUIRE(framebuffer.color_storage == serial.color_storage);
        REQUIRE(framebuffer.depth_storage == serial.depth_storage);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("frustum cull retains triangle crossing the view boundary", "[scene][m3][frustum]") {
        constexpr std::size_t width = 80U;
        constexpr std::size_t height = 64U;
        raster::SoAMesh boundary{};
        boundary.positions = {
            raster::Vec3f{.x = 3.0F, .y = -2.0F, .z = -2.0F},
            raster::Vec3f{.x = 6.0F, .y = -2.0F, .z = -2.0F},
            raster::Vec3f{.x = 3.0F, .y = 2.0F, .z = -2.0F},
        };
        const raster::Vec3f normal{.x = 0.0F, .y = 0.0F, .z = 1.0F};
        boundary.normals = {normal, normal, normal};
        boundary.uvs = {
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 1.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 1.0F},
        };
        boundary.indices = {0U, 1U, 2U};

        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFFFFFFFFU};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.5F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        raster::Framebuffer serial{width, height};
        raster::Framebuffer tiled{width, height};
        raster::render::TileRenderer renderer{3U};
        raster::render::SceneRenderDiagnostics serial_diagnostics{};
        raster::render::SceneRenderDiagnostics tiled_diagnostics{};
        REQUIRE(
            raster::render::render_scene(
                serial,
                boundary,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera,
                raster::render::SceneShading::Diffuse,
                raster::pbr::Material(),
                nullptr,
                &serial_diagnostics
            )
        );
        REQUIRE(renderer.render_scene(
            tiled,
            boundary,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Diffuse,
            raster::pbr::Material(),
            nullptr,
            &tiled_diagnostics
        ));

        REQUIRE(serial_diagnostics.source_triangles == 1U);
        REQUIRE(serial_diagnostics.frustum_rejected_triangles == 0U);
        REQUIRE(serial_diagnostics.clipped_triangles == 1U);
        REQUIRE(serial_diagnostics.prepared_triangles > 0U);
        REQUIRE(serial_diagnostics.rasterized_triangle_submissions > 0U);
        REQUIRE(tiled_diagnostics.frustum_rejected_triangles == 0U);
        REQUIRE(tiled_diagnostics.clipped_triangles == 1U);
        REQUIRE(tiled_diagnostics.prepared_triangles > 0U);
        REQUIRE(tiled_diagnostics.rasterized_triangle_submissions > 0U);
        REQUIRE(serial.color_storage == tiled.color_storage);
        REQUIRE(serial.depth_storage == tiled.depth_storage);

        const auto colors = std::as_const(serial).color_view();
        const auto depths = std::as_const(serial).depth_view();
        std::size_t visible_pixels = 0U;
        for (std::size_t y = 0U; y < height; ++y) {
            for (std::size_t x = 0U; x < width; ++x) {
                visible_pixels += depths[y, x] > 0.0F && colors[y, x] != 0xFF101018U ? 1U : 0U;
            }
        }
        REQUIRE(visible_pixels > 20U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "chunked submission matches whole-mesh render and culls distant chunk",
        "[scene][m3][chunks]"
    ) {
        constexpr std::size_t width = 96U;
        constexpr std::size_t height = 80U;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());
        REQUIRE(mesh->triangle_count() > 32U);

        const auto chunks = raster::render::make_scene_chunks(*mesh, 32U);
        REQUIRE(chunks.size() > 1U);
        REQUIRE(raster::render::valid_scene_chunks(chunks, mesh->triangle_count()));
        for (const auto& chunk : chunks) {
            REQUIRE(chunk.triangle_count <= 32U);
            REQUIRE(chunk.bounds_radius >= 0.0F);
            REQUIRE(std::isfinite(chunk.bounds_radius));
            REQUIRE(chunk.distance_to(chunk.bounds_center) == 0.0F);
        }
        const auto nearest_chunk_distance =
            std::ranges::min(chunks | std::views::transform([&](const auto& chunk) {
            return chunk.distance_to(raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.5F});
        }));
        REQUIRE(nearest_chunk_distance >= 0.0F);

        const raster::Vec3f view_origin{.x = 0.0F, .y = 0.0F, .z = 3.5F};
        const raster::render::FreeLookCamera camera{
            .position = view_origin,
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFFB08050U};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{
            image,
            raster::TextureColorSpace::Srgb
        };
        raster::Framebuffer whole{width, height};
        raster::Framebuffer unchunked_tiled{width, height};
        raster::Framebuffer chunked{width, height};
        raster::render::TileRenderer renderer{4U};
        raster::render::SceneRenderDiagnostics whole_stats{};
        raster::render::SceneRenderDiagnostics unchunked_stats{};
        raster::render::SceneRenderDiagnostics chunk_stats{};
        REQUIRE(
            raster::render::render_scene(
                whole,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera,
                raster::render::SceneShading::Pbr,
                raster::pbr::Material{},
                nullptr,
                &whole_stats,
                {},
                &view_origin
            )
        );
        REQUIRE(renderer.render_scene(
            unchunked_tiled,
            *mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Pbr,
            raster::pbr::Material{},
            nullptr,
            &unchunked_stats,
            {},
            &view_origin
        ));
        REQUIRE(renderer.render_scene(
            chunked,
            *mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Pbr,
            raster::pbr::Material{},
            nullptr,
            &chunk_stats,
            chunks,
            &view_origin
        ));
        REQUIRE(chunked.color_storage == whole.color_storage);
        REQUIRE(chunked.depth_storage == whole.depth_storage);
        REQUIRE(unchunked_tiled.color_storage == whole.color_storage);
        REQUIRE(unchunked_tiled.depth_storage == whole.depth_storage);
        REQUIRE(chunk_stats.source_chunks == chunks.size());
        REQUIRE(chunk_stats.frustum_rejected_chunks == 0U);
        REQUIRE(chunk_stats.prepared_triangles == unchunked_stats.prepared_triangles);
        REQUIRE(
            chunk_stats.rasterized_triangle_submissions
            == unchunked_stats.rasterized_triangle_submissions
        );

        auto distant_mesh = *mesh;
        constexpr raster::Vec3f displacement{.x = 10'000.0F, .y = 0.0F, .z = 0.0F};
        for (auto& position : distant_mesh.positions) {
            position = position + displacement;
        }
        const auto distant_chunks = raster::render::make_scene_chunks(distant_mesh, 32U);
        raster::Framebuffer distant_frame{width, height};
        raster::render::SceneRenderDiagnostics distant_stats{};
        REQUIRE(renderer.render_scene(
            distant_frame,
            distant_mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Pbr,
            raster::pbr::Material(),
            nullptr,
            &distant_stats,
            distant_chunks,
            &view_origin
        ));
        REQUIRE(distant_stats.source_chunks == distant_chunks.size());
        REQUIRE(distant_stats.frustum_rejected_chunks == distant_chunks.size());
        REQUIRE(distant_stats.source_triangles == 0U);
        REQUIRE(distant_stats.clipped_triangles == 0U);
        REQUIRE(distant_stats.prepared_triangles == 0U);
        REQUIRE(distant_stats.rasterized_triangle_submissions == 0U);
        for (const auto pixel : distant_frame.color_storage) {
            REQUIRE(pixel == 0xFF101018U);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "PBR scene mode is selectable, deterministic, and material-sensitive", "[scene][pbr]"
    ) {
        constexpr std::size_t width = 96U;
        constexpr std::size_t height = 80U;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFFB08050U};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{
            image,
            raster::TextureColorSpace::Srgb
        };
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.1F, .y = -0.2F, .z = 3.5F},
            .yaw = 0.07F,
            .pitch = 0.03F,
        };
        const raster::pbr::Material dielectric{
            .base_color = raster::Vec3f{.x = 1.0F, .y = 1.0F, .z = 1.0F},
            .metallic = 0.0F,
            .perceptual_roughness = 0.55F,
        };
        const raster::pbr::Material copper{
            .base_color = raster::Vec3f{.x = 0.95F, .y = 0.48F, .z = 0.22F},
            .metallic = 1.0F,
            .perceptual_roughness = 0.3F,
        };

        raster::Framebuffer diffuse{width, height};
        raster::Framebuffer serial{width, height};
        raster::Framebuffer tiled{width, height};
        raster::render::TileRenderer renderer{4U};
        const auto render_pbr = [&](raster::Framebuffer& target,
                                    const raster::pbr::Material& material) noexcept {
            return raster::render::render_scene(
                target,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera,
                raster::render::SceneShading::Pbr,
                material
            );
        };
        REQUIRE(
            raster::render::
                render_scene(diffuse, *mesh, 0U, false, true, false, false, texture, camera)
        );
        REQUIRE(render_pbr(serial, dielectric));
        REQUIRE(renderer.render_scene(
            tiled,
            *mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Pbr,
            dielectric
        ));
        REQUIRE(tiled.color_storage == serial.color_storage);
        REQUIRE(tiled.depth_storage == serial.depth_storage);
        REQUIRE(serial.color_storage != diffuse.color_storage);
        std::size_t pbr_allocations = 0U;
        bool pbr_rendered_after_warmup = false;
        {
            const raster::platform::ScopedAllocationProbe probe{true};
            pbr_rendered_after_warmup = renderer.render_scene(
                tiled,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera,
                raster::render::SceneShading::Pbr,
                dielectric
            );
            pbr_allocations = probe.allocations();
        }
        REQUIRE(pbr_rendered_after_warmup);
        REQUIRE(tiled.color_storage == serial.color_storage);
        REQUIRE(pbr_allocations == 0U);

        REQUIRE(render_pbr(serial, copper));
        REQUIRE(serial.color_storage != diffuse.color_storage);
        REQUIRE(renderer.render_scene(
            tiled,
            *mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Pbr,
            copper
        ));
        REQUIRE(tiled.color_storage == serial.color_storage);

        raster::Texture2D<raster::LayoutTiled4x4> encoded_texture{image};
        REQUIRE_FALSE(
            raster::render::render_scene(
                tiled,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                encoded_texture,
                camera,
                raster::render::SceneShading::Pbr,
                copper
            )
        );
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("PBR metallic-roughness texture modulates material per texel", "[scene][pbr]") {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage base_image{};
        base_image.width = 1U;
        base_image.height = 1U;
        base_image.pixels = {0xFFB08050U};
        const raster::Texture2D<raster::LayoutTiled4x4> base_texture{
            base_image,
            raster::TextureColorSpace::Srgb
        };
        // G = roughness, B = metallic; R/A ignored.
        raster::TgaImage rough_dielectric_image{};
        rough_dielectric_image.width = 1U;
        rough_dielectric_image.height = 1U;
        rough_dielectric_image.pixels = {0xFF00FF00U}; // G=255 rough, B=0 metal
        const raster::Texture2D<raster::LayoutTiled4x4> rough_dielectric{rough_dielectric_image};
        raster::TgaImage smooth_metal_image{};
        smooth_metal_image.width = 1U;
        smooth_metal_image.height = 1U;
        smooth_metal_image.pixels = {0xFF0000FFU}; // G=0 rough, B=255 metal
        const raster::Texture2D<raster::LayoutTiled4x4> smooth_metal{smooth_metal_image};

        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.1F, .y = -0.2F, .z = 3.5F},
            .yaw = 0.07F,
            .pitch = 0.03F,
        };
        constexpr raster::pbr::Material factors{
            .base_color = raster::Vec3f{.x = 1.0F, .y = 1.0F, .z = 1.0F},
            .metallic = 1.0F,
            .perceptual_roughness = 1.0F,
        };
        constexpr std::size_t width = 96U;
        constexpr std::size_t height = 80U;
        raster::Framebuffer serial_rough{width, height};
        raster::Framebuffer serial_metal{width, height};
        raster::Framebuffer tiled_metal{width, height};
        raster::Framebuffer reference_metal{width, height};
        REQUIRE(
            raster::render::render_scene(
                serial_rough,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                base_texture,
                camera,
                raster::render::SceneShading::Pbr,
                factors,
                &rough_dielectric
            )
        );
        REQUIRE(
            raster::render::render_scene(
                serial_metal,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                base_texture,
                camera,
                raster::render::SceneShading::Pbr,
                factors,
                &smooth_metal
            )
        );
        REQUIRE(serial_rough.color_storage != serial_metal.color_storage);
        auto equivalent_factors = factors;
        equivalent_factors.perceptual_roughness = 0.0F;
        REQUIRE(
            raster::render::render_scene(
                reference_metal,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                base_texture,
                camera,
                raster::render::SceneShading::Pbr,
                equivalent_factors
            )
        );
        REQUIRE(reference_metal.color_storage == serial_metal.color_storage);
        REQUIRE(reference_metal.depth_storage == serial_metal.depth_storage);

        raster::render::TileRenderer renderer{4U};
        REQUIRE(renderer.render_scene(
            tiled_metal,
            *mesh,
            0U,
            false,
            true,
            false,
            false,
            base_texture,
            camera,
            raster::render::SceneShading::Pbr,
            factors,
            &smooth_metal
        ));
        REQUIRE(tiled_metal.color_storage == serial_metal.color_storage);
        REQUIRE(tiled_metal.depth_storage == serial_metal.depth_storage);
        std::size_t mr_allocations = 0U;
        bool mr_warmed = false;
        {
            const raster::platform::ScopedAllocationProbe probe{true};
            mr_warmed = renderer.render_scene(
                tiled_metal,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                base_texture,
                camera,
                raster::render::SceneShading::Pbr,
                factors,
                &smooth_metal
            );
            mr_allocations = probe.allocations();
        }
        REQUIRE(mr_warmed);
        REQUIRE(mr_allocations == 0U);

        // Data textures must stay linear: an sRGB mr texture is rejected.
        const raster::Texture2D<raster::LayoutTiled4x4> srgb_mr{
            smooth_metal_image,
            raster::TextureColorSpace::Srgb
        };
        REQUIRE_FALSE(
            raster::render::render_scene(
                tiled_metal,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                base_texture,
                camera,
                raster::render::SceneShading::Pbr,
                factors,
                &srgb_mr
            )
        );
        const raster::Texture2D<raster::LayoutTiled4x4> empty_mr{};
        REQUIRE_FALSE(
            raster::render::render_scene(
                tiled_metal,
                *mesh,
                0U,
                false,
                true,
                false,
                false,
                base_texture,
                camera,
                raster::render::SceneShading::Pbr,
                factors,
                &empty_mr
            )
        );
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("PBR preserves fill under shadows and sRGB-encodes output", "[scene][pbr]") {
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFF808080U};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{
            image,
            raster::TextureColorSpace::Srgb
        };
        constexpr std::size_t width = 64U;
        constexpr std::size_t height = 64U;
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 3.0F, .z = 4.0F},
            .yaw = 0.0F,
            .pitch = -std::atan2(3.0F, 4.0F),
        };
        const raster::pbr::Material white_dielectric{
            .base_color = raster::Vec3f{.x = 1.0F, .y = 1.0F, .z = 1.0F},
            .metallic = 0.0F,
            .perceptual_roughness = 1.0F,
        };
        const auto face_on =
            make_two_sided_triangle(raster::Vec3f{.x = 0.0F, .y = 0.6F, .z = 0.8F});
        raster::Framebuffer face_frame{width, height};
        REQUIRE(SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_NO_SHADOW", "1", true));
        REQUIRE(
            raster::render::render_scene(
                face_frame,
                face_on,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera,
                raster::render::SceneShading::Pbr,
                white_dielectric
            )
        );
        REQUIRE(SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_NO_SHADOW"));
        const auto face_color = std::as_const(face_frame).color_view();
        REQUIRE(face_color[height / 2U, width / 2U] == 0xFF646464U);

        const auto shadow_scene = make_shadow_scene();
        raster::Framebuffer with_shadows{width, height};
        raster::Framebuffer without_shadows{width, height};
        REQUIRE(
            raster::render::render_scene(
                with_shadows,
                shadow_scene,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera,
                raster::render::SceneShading::Pbr,
                white_dielectric
            )
        );
        REQUIRE(SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_NO_SHADOW", "1", true));
        const bool rendered_without_shadows = raster::render::render_scene(
            without_shadows,
            shadow_scene,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            raster::render::SceneShading::Pbr,
            white_dielectric
        );
        REQUIRE(SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_NO_SHADOW"));
        REQUIRE(rendered_without_shadows);
        const auto colors_with_shadows = std::as_const(with_shadows).color_view();
        const auto colors_without_shadows = std::as_const(without_shadows).color_view();
        const auto depth = std::as_const(with_shadows).depth_view();
        std::size_t changed_shadow_pixels = 0U;
        for (std::size_t y = 0U; y < height; ++y) {
            for (std::size_t x = 0U; x < width; ++x) {
                if (depth[y, x]
                    > 0.0F
                    && colors_with_shadows[y, x]
                    != colors_without_shadows[y, x]) {
                    ++changed_shadow_pixels;
                    REQUIRE(colors_with_shadows[y, x] < colors_without_shadows[y, x]);
                }
            }
        }
        REQUIRE(changed_shadow_pixels > 20U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "warmed renderer stays allocation-free after heap churn", "[scene][allocation][aging]"
    ) {
        constexpr std::size_t width = 96U;
        constexpr std::size_t height = 80U;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage image{};
        image.width = 4U;
        image.height = 4U;
        image.pixels.resize(16U);
        for (std::size_t index = 0; index < image.pixels.size(); ++index) {
            image.pixels[index] = (index & 1U) == 0U ? 0xFFE08030U : 0xFF3050C0U;
        }
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.1F, .y = -0.2F, .z = 3.5F},
            .yaw = 0.07F,
            .pitch = 0.03F,
        };
        raster::Framebuffer framebuffer{width, height};
        raster::render::TileRenderer renderer{4U};
        const auto render = [&] noexcept {
            return renderer
                .render_scene(framebuffer, *mesh, 0U, false, true, false, false, texture, camera);
        };

        REQUIRE(render());
        const auto expected_color = framebuffer.color_storage;
        const auto expected_depth = framebuffer.depth_storage;

        // Calibrate the probe with one deliberate allocation before trusting
        // its zero-allocation result around the render below.
        std::size_t detected_allocations = 0U;
        {
            const raster::platform::ScopedAllocationProbe probe{true};
            std::vector<std::uint64_t> deliberate(4096U, 0xA55AA55AU);
            REQUIRE(deliberate.front() == 0xA55AA55AU);
            detected_allocations = probe.allocations();
        }
        REQUIRE(detected_allocations > 0U);

        // Bounded, deterministic mixed-size churn while every render resource
        // remains alive. This tests the actual renderer's invariant; timing is
        // deliberately left to the heap-aging benchmark, never asserted here.
        constexpr std::array<std::size_t, 5> block_sizes{32U, 64U, 256U, 1024U, 4096U};
        std::vector<std::vector<std::byte>> ballast;
        ballast.reserve(512U);
        for (std::size_t index = 0; index < 512U; ++index) {
            ballast.emplace_back(
                block_sizes[index % block_sizes.size()],
                static_cast<std::byte>(index & 0xFFU)
            );
        }
        for (std::size_t index = 0; index < ballast.size(); index += 2U) {
            std::vector<std::byte>{}.swap(ballast[index]);
        }
        for (std::size_t index = 0; index < 256U; ++index) {
            ballast.emplace_back(
                block_sizes[(index * 3U + 1U) % block_sizes.size()],
                static_cast<std::byte>((index + 37U) & 0xFFU)
            );
        }

        std::size_t render_allocations = 0U;
        bool rendered = false;
        {
            const raster::platform::ScopedAllocationProbe probe{true};
            rendered = render();
            render_allocations = probe.allocations();
        }
        REQUIRE(rendered);
        REQUIRE(render_allocations == 0U);
        REQUIRE(framebuffer.color_storage == expected_color);
        REQUIRE(framebuffer.depth_storage == expected_depth);
        REQUIRE_FALSE(ballast.empty());
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("procedural grid mesh is watertight and faces up", "[scene][stress]") {
        const auto mesh = raster::make_grid_mesh(24U, 24U, 4.0F, -0.5F, -20.0F, 8.0F);
        REQUIRE(mesh.triangle_count() == 24U * 24U * 2U);
        REQUIRE(mesh.has_consistent_vertex_streams());
        REQUIRE(mesh.positions.size() == 25U * 25U);
        for (const auto index : mesh.indices) {
            REQUIRE(index < mesh.positions.size());
        }
        for (std::size_t triangle = 0; triangle < mesh.triangle_count(); ++triangle) {
            const auto ia = static_cast<std::size_t>(mesh.indices[triangle * 3U]);
            const auto ib = static_cast<std::size_t>(mesh.indices[triangle * 3U + 1U]);
            const auto ic = static_cast<std::size_t>(mesh.indices[triangle * 3U + 2U]);
            const auto face_normal = raster::cross(
                mesh.positions[ib] - mesh.positions[ia],
                mesh.positions[ic] - mesh.positions[ia]
            );
            REQUIRE(raster::length_squared(face_normal) > 0.0F);
            REQUIRE(
                raster::dot(face_normal, raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}) > 0.0F
            );
        }

        raster::TgaImage one_texel{};
        one_texel.width = 1U;
        one_texel.height = 1U;
        one_texel.pixels.push_back(0xFF336699U);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{one_texel};
        constexpr std::size_t width = 320U;
        constexpr std::size_t height = 240U;
        raster::Framebuffer framebuffer{width, height};
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 3.0F, .z = 2.0F},
            .yaw = 0.0F,
            .pitch = -0.9F,
        };
        REQUIRE(
            raster::render::
                render_scene(framebuffer, mesh, 0U, false, true, false, true, texture, camera)
        );
        std::size_t covered = 0U;
        const auto depths = std::as_const(framebuffer).depth_view();
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                covered += depths[y, x] > 0.0F ? 1U : 0U;
            }
        }
        REQUIRE(covered > width * height / 10U);

        // Parallel prepare must reproduce the serial framebuffer exactly,
        // including color and depth, on a mesh large enough to span chunks.
        raster::Framebuffer threaded{width, height};
        raster::render::TileRenderer renderer{4U};
        REQUIRE(
            renderer.render_scene(threaded, mesh, 0U, false, true, false, true, texture, camera)
        );
        REQUIRE(threaded.color_storage == framebuffer.color_storage);
        REQUIRE(threaded.depth_storage == framebuffer.depth_storage);

        std::size_t allocation_count = 0U;
        bool repeated_rendered = false;
        {
            const raster::platform::ScopedAllocationProbe probe{true};
            repeated_rendered =
                renderer
                    .render_scene(threaded, mesh, 0U, false, true, false, true, texture, camera);
            allocation_count = probe.allocations();
        }
        REQUIRE(repeated_rendered);
        REQUIRE(allocation_count == 0U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("sphere renders the near hemisphere", "[scene][sphere][winding]") {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage one_texel{};
        one_texel.width = 1U;
        one_texel.height = 1U;
        one_texel.pixels.push_back(0xFF336699U);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{one_texel};
        constexpr std::size_t width = 80U;
        constexpr std::size_t height = 60U;
        raster::Framebuffer framebuffer{width, height};
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        REQUIRE(
            raster::render::
                render_scene(framebuffer, *mesh, 0U, false, true, false, true, texture, camera)
        );
        // Reverse-Z: near plane maps to 1, far plane to 0. The sphere center
        // must show the near wall (depth ~0.005 at this camera), not the far
        // wall (~0.001). This pins front-face culling against mirroring.
        const auto depth = std::as_const(framebuffer).depth_view();
        REQUIRE(depth[height / 2U, width / 2U] > 0.004F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "free-look forward speed scales with distance and remains bounded", "[scene][camera]"
    ) {
        const auto near_speed = raster::render::free_look_movement_speed(
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F}
        );
        const auto initial_speed = raster::render::free_look_movement_speed(
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F}
        );
        const auto far_speed = raster::render::free_look_movement_speed(
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 20.0F}
        );
        const auto extreme_speed = raster::render::free_look_movement_speed(
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1000.0F}
        );

        REQUIRE(near_speed == 2.5F);
        REQUIRE(initial_speed == 2.5F);
        REQUIRE(far_speed > initial_speed);
        REQUIRE(far_speed < 20.0F);
        REQUIRE(extreme_speed == 20.0F);
    }

    struct CheckerFloorShader final : raster::ShaderBase {
        struct Varyings {
            raster::Vec4f clip_pos{};
            raster::Vec2f uv{};

            [[nodiscard]]
            constexpr auto attributes(this auto&& self) noexcept {
                return std::tie(self.uv);
            }
        };

        raster::Mat4f model_view_projection{};

        [[nodiscard]]
        constexpr Varyings vertex(raster::Vec3f position, raster::Vec2f uv) const noexcept {
            return Varyings{
                .clip_pos = model_view_projection
                    * raster::Vec4f{
                        .x = position.x,
                        .y = position.y,
                        .z = position.z,
                        .w = 1.0F,
                    },
                .uv = uv,
            };
        }

        [[nodiscard]]
        raster::Vec4f fragment(const Varyings& varying) const noexcept {
            const auto cell = static_cast<int>(std::floor(varying.uv.x))
                + static_cast<int>(std::floor(varying.uv.y));
            const auto parity = ((cell % 2) + 2) % 2;
            const auto shade = parity == 0 ? 0.85F : 0.35F;
            return raster::Vec4f{.x = shade, .y = shade, .z = shade, .w = 1.0F};
        }
    };

    static_assert(raster::ShaderProgram<CheckerFloorShader>);

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("reverse-Z checker floor renders continuously from near to far", "[scene][m3]") {
        constexpr std::size_t width = 160;
        constexpr std::size_t height = 120;
        constexpr std::size_t x_cells = 1;
        constexpr std::size_t z_cells = 1;
        constexpr float half_width = 4.0F;
        constexpr float near_distance = 0.1F;
        constexpr float far_distance = 1000.0F;
        constexpr float uv_scale = 0.5F;

        const raster::Vec3f eye{.x = 0.0F, .y = 0.1F, .z = 0.0F};
        const auto view = raster::look_at4f(
            eye,
            raster::Vec3f{.x = eye.x, .y = eye.y - 0.0001F, .z = eye.z - 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto projection = raster::perspective_reverse_z4f(
            std::numbers::pi_v<float> / 3.0F,
            static_cast<float>(width) / static_cast<float>(height),
            0.1F,
            1000.0F
        );
        CheckerFloorShader shader{};
        shader.model_view_projection = projection * view;

        std::vector<std::uint32_t> colors(width * height, 0U);
        std::vector<float> depths(width * height, 0.0F);
        std::vector<std::uint8_t> coverage(width * height, 0U);
        auto color_view =
            std::mdspan<std::uint32_t, raster::Extents2D>{colors.data(), height, width};
        auto depth_view = std::mdspan<float, raster::Extents2D>{depths.data(), height, width};
        auto coverage_view =
            std::mdspan<std::uint8_t, raster::Extents2D>{coverage.data(), height, width};

        const auto position = [](std::size_t x_index, std::size_t z_index) noexcept {
            const auto x_fraction = static_cast<float>(x_index) / static_cast<float>(x_cells);
            const auto z_fraction = static_cast<float>(z_index) / static_cast<float>(z_cells);
            return raster::Vec3f{
                .x = -half_width + 2.0F * half_width * x_fraction,
                .y = 0.0F,
                .z = -near_distance + (-far_distance + near_distance) * z_fraction,
            };
        };
        const auto make_vertex = [&shader](raster::Vec3f point) noexcept {
            return shader.vertex(
                point,
                raster::Vec2f{.x = point.x * uv_scale, .y = -point.z * uv_scale}
            );
        };

        std::size_t submitted_triangles = 0;
        for (const auto z_index : std::views::iota(std::size_t{0}, z_cells)) {
            for (const auto x_index : std::views::iota(std::size_t{0}, x_cells)) {
                const auto near_left = make_vertex(position(x_index, z_index));
                const auto near_right = make_vertex(position(x_index + 1, z_index));
                const auto far_right = make_vertex(position(x_index + 1, z_index + 1));
                const auto far_left = make_vertex(position(x_index, z_index + 1));
                const std::array<std::array<CheckerFloorShader::Varyings, 3>, 2> source_triangles{
                    std::array{near_left, near_right, far_right},
                    std::array{near_left, far_right, far_left},
                };

                for (const auto& source : source_triangles) {
                    const auto clipped =
                        raster::clip_triangle(shader, source[0], source[1], source[2]);
                    REQUIRE(clipped.has_value());
                    for (const auto& clipped_triangle : clipped->active_triangles()) {
                        std::array<raster::ScreenPoint, 3> screen{};
                        std::array<float, 3> inverse_w{};
                        std::array<float, 3> reverse_z{};
                        for (const auto vertex_index :
                             std::views::iota(std::size_t{0}, std::size_t{3})) {
                            const auto& vertex = clipped_triangle.vertices[vertex_index];
                            REQUIRE(std::isfinite(vertex.clip_pos.w));
                            REQUIRE(vertex.clip_pos.w > 0.0F);
                            inverse_w[vertex_index] = 1.0F / vertex.clip_pos.w;
                            reverse_z[vertex_index] = vertex.clip_pos.z * inverse_w[vertex_index];
                            screen[vertex_index] = raster::viewport_to_screen(
                                vertex.clip_pos.x * inverse_w[vertex_index],
                                vertex.clip_pos.y * inverse_w[vertex_index],
                                width,
                                height
                            );
                        }

                        const auto setup = raster::make_screen_triangle(
                            screen[0],
                            screen[1],
                            screen[2],
                            inverse_w,
                            reverse_z
                        );
                        if (!setup) {
                            continue;
                        }
                        ++submitted_triangles;
                        raster::rasterize_triangle(
                            *setup,
                            width,
                            height,
                            depth_view,
                            [&](std::size_t x,
                                std::size_t y,
                                const raster::PixelBarycentrics& barycentrics) noexcept {
                            ++coverage_view[y, x];
                            const auto attributes = shader.interpolate(
                                clipped_triangle.vertices[0],
                                clipped_triangle.vertices[1],
                                clipped_triangle.vertices[2],
                                barycentrics
                            );
                            const auto shaded = shader.fragment(
                                CheckerFloorShader::Varyings{
                                    .clip_pos = clipped_triangle.vertices[0].clip_pos,
                                    .uv = std::get<0>(attributes),
                                }
                            );
                            color_view[y, x] = shaded.x > 0.5F ? 0xFFCCCCCCU : 0xFF555555U;
                        }
                        );
                    }
                }
            }
        }

        REQUIRE(submitted_triangles > 0);
        std::size_t covered_pixels = 0;
        std::size_t light_checker_pixels = 0;
        std::size_t dark_checker_pixels = 0;
        float minimum_depth = 1.0F;
        float maximum_depth = 0.0F;
        for (std::size_t y = 0U; y < height; ++y) {
            for (std::size_t x = 0U; x < width; ++x) {
                REQUIRE(coverage_view[y, x] <= 1U); // no shared-edge double writes / z-fighting
                if (coverage_view[y, x] == 0U) {
                    REQUIRE(depth_view[y, x] == 0.0F);
                    continue;
                }

                ++covered_pixels;
                REQUIRE(std::isfinite(depth_view[y, x]));
                REQUIRE(depth_view[y, x] >= 0.0F);
                REQUIRE(depth_view[y, x] <= 1.0F);
                minimum_depth = std::min(minimum_depth, depth_view[y, x]);
                maximum_depth = std::max(maximum_depth, depth_view[y, x]);
                light_checker_pixels += color_view[y, x] == 0xFFCCCCCCU ? 1U : 0U;
                dark_checker_pixels += color_view[y, x] == 0xFF555555U ? 1U : 0U;
            }
        }

        REQUIRE(covered_pixels > width * height / 5U);
        REQUIRE(light_checker_pixels > 0U);
        REQUIRE(dark_checker_pixels > 0U);
        REQUIRE(maximum_depth > 0.5F); // near end of the floor is visible
        REQUIRE(minimum_depth < 0.01F); // far end approaches reverse-Z zero
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "high-poly camera-through traversal stays projectable after clipping", "[scene][m4]"
    ) {
        constexpr std::size_t width = 640;
        constexpr std::size_t height = 480;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());
        REQUIRE(mesh->triangle_count() == 64U * 32U * 2U - 64U * 2U);

        const auto eye = raster::Vec3f{};
        const auto view = raster::look_at4f(
            eye,
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = -1.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto projection = raster::perspective_reverse_z4f(
            std::numbers::pi_v<float> / 2.0F,
            static_cast<float>(width) / static_cast<float>(height),
            0.1F,
            10.0F
        );
        CheckerFloorShader shader{};
        shader.model_view_projection = projection * view;

        std::size_t source_triangles = 0;
        std::size_t w_crossing_triangles = 0;
        std::size_t extreme_edge_triangles = 0;
        std::size_t clipped_output_triangles = 0;
        std::size_t guard_band_output_triangles = 0;
        std::size_t guard_band_saved_triangles = 0;
        std::size_t rasterized_fragments = 0;
        std::vector<std::uint8_t> equator_coverage(width, 0U);
        for (const auto triangle_index : std::views::iota(std::size_t{0}, mesh->triangle_count())) {
            const auto index_base = triangle_index * 3U;
            const auto index_a = static_cast<std::size_t>(mesh->indices[index_base]);
            const auto index_b = static_cast<std::size_t>(mesh->indices[index_base + 1U]);
            const auto index_c = static_cast<std::size_t>(mesh->indices[index_base + 2U]);
            ++source_triangles;
            const auto vertex_a = shader.vertex(mesh->positions[index_a], mesh->uvs[index_a]);
            const auto vertex_b = shader.vertex(mesh->positions[index_b], mesh->uvs[index_b]);
            const auto vertex_c = shader.vertex(mesh->positions[index_c], mesh->uvs[index_c]);
            const std::array source{vertex_a, vertex_b, vertex_c};
            bool has_behind = false;
            bool has_in_front = false;
            bool reaches_extreme_edge = false;
            for (const auto& vertex : source) {
                has_behind = has_behind || vertex.clip_pos.w <= 0.0F;
                has_in_front = has_in_front || vertex.clip_pos.w > 0.0F;
                if (vertex.clip_pos.w > 0.0F) {
                    const auto ndc_x = vertex.clip_pos.x / vertex.clip_pos.w;
                    const auto ndc_y = vertex.clip_pos.y / vertex.clip_pos.w;
                    reaches_extreme_edge = reaches_extreme_edge
                        || std::fabs(ndc_x)
                        >= 1.0F
                        || std::fabs(ndc_y)
                        >= 1.0F;
                }
            }
            w_crossing_triangles += has_behind && has_in_front ? 1U : 0U;
            extreme_edge_triangles += reaches_extreme_edge ? 1U : 0U;

            const auto clipped = raster::clip_triangle(shader, vertex_a, vertex_b, vertex_c);
            REQUIRE(clipped.has_value());
            const auto guard_clipped =
                raster::clip_triangle_guard_band(shader, vertex_a, vertex_b, vertex_c, 2.0F);
            REQUIRE(guard_clipped.has_value());
            guard_band_output_triangles += guard_clipped->size;
            if (clipped->size > guard_clipped->size) {
                guard_band_saved_triangles += clipped->size - guard_clipped->size;
            }
            for (const auto& guard_triangle : guard_clipped->active_triangles()) {
                for (const auto& guard_vertex : guard_triangle.vertices) {
                    REQUIRE(guard_vertex.clip_pos.w > 0.0F);
                    const auto ndc_x = guard_vertex.clip_pos.x / guard_vertex.clip_pos.w;
                    const auto ndc_y = guard_vertex.clip_pos.y / guard_vertex.clip_pos.w;
                    REQUIRE(std::fabs(ndc_x) <= 2.00001F);
                    REQUIRE(std::fabs(ndc_y) <= 2.00001F);
                    static_cast<void>(
                        raster::to_fixed(raster::viewport_to_screen(ndc_x, ndc_y, width, height).x)
                    );
                }
            }
            for (const auto& triangle : clipped->active_triangles()) {
                ++clipped_output_triangles;
                std::array<raster::ScreenPoint, 3> screen{};
                std::array<float, 3> inverse_w{};
                std::array<float, 3> depth{};
                for (const auto vertex_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                    const auto& vertex = triangle.vertices[vertex_index];
                    REQUIRE(std::isfinite(vertex.clip_pos.x));
                    REQUIRE(std::isfinite(vertex.clip_pos.y));
                    REQUIRE(std::isfinite(vertex.clip_pos.z));
                    REQUIRE(std::isfinite(vertex.clip_pos.w));
                    REQUIRE(vertex.clip_pos.w > 0.0F);
                    inverse_w[vertex_index] = 1.0F / vertex.clip_pos.w;
                    depth[vertex_index] = vertex.clip_pos.z * inverse_w[vertex_index];
                    const auto ndc_x = vertex.clip_pos.x * inverse_w[vertex_index];
                    const auto ndc_y = vertex.clip_pos.y * inverse_w[vertex_index];
                    REQUIRE(std::fabs(ndc_x) <= 1.00001F);
                    REQUIRE(std::fabs(ndc_y) <= 1.00001F);
                    REQUIRE(depth[vertex_index] >= -1e-5F);
                    REQUIRE(depth[vertex_index] <= 1.00001F);
                    screen[vertex_index] = raster::viewport_to_screen(ndc_x, ndc_y, width, height);
                }
                // Screen setup performs the floating-point to fixed-point
                // conversion exercised by the high-poly traversal.
                const auto setup =
                    raster::make_screen_triangle(screen[0], screen[1], screen[2], inverse_w, depth);
                if (!setup) {
                    continue;
                }
                raster::rasterize_triangle(
                    *setup,
                    width,
                    height,
                    [&](std::size_t x,
                        std::size_t y,
                        const raster::PixelBarycentrics& barycentrics) noexcept {
                    REQUIRE(x < width);
                    REQUIRE(y < height);
                    REQUIRE(std::isfinite(barycentrics.l0));
                    REQUIRE(std::isfinite(barycentrics.l1));
                    REQUIRE(std::isfinite(barycentrics.l2));
                    REQUIRE(
                        std::fabs(barycentrics.l0 + barycentrics.l1 + barycentrics.l2 - 1.0F)
                        < 1e-5F
                    );
                    if (y == height / 2U) {
                        equator_coverage[x] = 1U;
                    }
                    ++rasterized_fragments;
                }
                );
            }
        }

        REQUIRE(source_triangles == mesh->triangle_count());
        REQUIRE(w_crossing_triangles > 0U); // camera lies inside the sphere
        REQUIRE(extreme_edge_triangles > 0U);
        REQUIRE(clipped_output_triangles > 0U);
        REQUIRE(guard_band_output_triangles > 0U);
        REQUIRE(guard_band_saved_triangles > 0U);
        // From inside a closed outward-wound mesh every triangle faces away,
        // so correct backface culling rasterizes nothing; the clipping and
        // projectability assertions above are this traversal's coverage.
        REQUIRE(rasterized_fragments == 0U);
        REQUIRE(std::ranges::all_of(equator_coverage, [](std::uint8_t value) {
            return value == 0U;
        }));
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "partially covered quad keeps finite helper-lane UV derivatives", "[scene][m5][mipmap]"
    ) {
        constexpr std::size_t width = 8;
        constexpr std::size_t height = 8;
        const auto triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 3.0F, .y = 1.0F},
            raster::ScreenPoint{.x = 3.0F, .y = 7.0F},
            raster::ScreenPoint{.x = 7.0F, .y = 1.0F},
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.5F, 0.5F, 0.5F}
        );
        REQUIRE(triangle.has_value());

        raster::TgaImage checker_image{};
        checker_image.width = 128U;
        checker_image.height = 128U;
        checker_image.pixels.resize(checker_image.width * checker_image.height);
        for (const auto y : std::views::iota(std::size_t{0}, checker_image.height)) {
            for (const auto x : std::views::iota(std::size_t{0}, checker_image.width)) {
                const auto cell = (x / 8U) + (y / 8U);
                checker_image.pixels[y * checker_image.width + x] =
                    (cell & 1U) == 0U ? 0xFFFFFFFFU : 0xFF000000U;
            }
        }

        const raster::Texture2D<raster::LayoutTiled4x4> checker{checker_image};
        std::vector<float> depth_storage(width * height, 0.0F);
        auto depth_view =
            std::mdspan<float, raster::Extents2D>{depth_storage.data(), height, width};
        depth_view[3U, 4U] = 1.0F; // occlude one covered lane at the silhouette

        bool saw_partial_quad = false;
        bool saw_partially_occluded_quad = false;
        float minimum_partial_lod = std::numeric_limits<float>::infinity();
        float maximum_partial_lod = 0.0F;
        std::size_t tested_visible_lanes = 0;
        std::size_t bright_pixels = 0;
        std::size_t dark_pixels = 0;
        std::vector<std::uint32_t> rendered_pixels(width * height, 0U);
        auto rendered_view =
            std::mdspan<std::uint32_t, raster::Extents2D>{rendered_pixels.data(), height, width};
        raster::rasterize_triangle_quads(
            *triangle,
            width,
            height,
            std::optional{depth_view},
            [&](std::size_t quad_x,
                std::size_t quad_y,
                const raster::QuadBarycentrics& quad) noexcept {
            if (quad.coverage_mask == 0U || quad.coverage_mask == 0xFU) {
                return;
            }
            saw_partial_quad = true;
            if (quad_x == 4U && quad_y == 2U) {
                saw_partially_occluded_quad =
                    (quad.coverage_mask & 0x4U) != 0U && (quad.visible_mask & 0x4U) == 0U;
            }
            std::array<raster::Vec2f, 4> helper_uv{};
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                const auto& bary = quad.lanes[lane];
                helper_uv[lane] = raster::Vec2f{
                    .x = bary.a2 * 0.25F,
                    .y = bary.a1 * 0.375F,
                };
                REQUIRE(std::isfinite(helper_uv[lane].x));
                REQUIRE(std::isfinite(helper_uv[lane].y));
            }
            const auto derivatives = raster::quad_derivatives(helper_uv);
            const auto lod = raster::mip_lod(derivatives, checker.width(), checker.height());
            // The test triangle maps screen x/y linearly to u/v with slopes
            // du/dx=dv/dy=1/16. The expected LOD is therefore log2(8)=3.
            REQUIRE(std::fabs(derivatives.duv_dx.x - (1.0F / 16.0F)) < 1e-5F);
            REQUIRE(std::fabs(derivatives.duv_dx.y) < 1e-5F);
            REQUIRE(std::fabs(derivatives.duv_dy.x) < 1e-5F);
            REQUIRE(std::fabs(derivatives.duv_dy.y - (1.0F / 16.0F)) < 1e-5F);
            REQUIRE(std::fabs(lod - 3.0F) < 1e-5F);
            REQUIRE(std::isfinite(derivatives.duv_dx.x));
            REQUIRE(std::isfinite(derivatives.duv_dx.y));
            REQUIRE(std::isfinite(derivatives.duv_dy.x));
            REQUIRE(std::isfinite(derivatives.duv_dy.y));
            REQUIRE(std::isfinite(lod));
            minimum_partial_lod = std::min(minimum_partial_lod, lod);
            maximum_partial_lod = std::max(maximum_partial_lod, lod);
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                if ((quad.visible_mask & (1U << lane)) == 0U) {
                    continue;
                }
                const auto filtered =
                    checker.sample_trilinear(helper_uv[lane].x, helper_uv[lane].y, lod);
                // Independent analytic reference: the expected derivative gives
                // LOD 3, and the texture sampler evaluates that explicit LOD.
                const auto reference =
                    checker.sample_trilinear(helper_uv[lane].x, helper_uv[lane].y, 3.0F);
                REQUIRE(filtered == reference);
                const auto x = quad_x + (lane & 1U);
                const auto y = quad_y + (lane >> 1U);
                rendered_view[y, x] = filtered;
                bright_pixels += filtered == 0xFFFFFFFFU ? 1U : 0U;
                dark_pixels += filtered == 0xFF000000U ? 1U : 0U;
                ++tested_visible_lanes;
            }
            INFO("partial quad at " << quad_x << ',' << quad_y);
        }
        );
        REQUIRE(saw_partial_quad);
        REQUIRE(saw_partially_occluded_quad);
        REQUIRE(tested_visible_lanes > 0U);
        REQUIRE(maximum_partial_lod - minimum_partial_lod < 1e-5F);
        REQUIRE(std::fabs(minimum_partial_lod - 3.0F) < 0.1F);
        REQUIRE(bright_pixels > 0U);
        REQUIRE(dark_pixels > 0U);
        for (std::size_t y = 0U; y < height; ++y) {
            for (std::size_t x = 0U; x < width; ++x) {
                const auto pixel = rendered_view[y, x];
                const bool is_valid_output =
                    pixel == 0U || pixel == 0xFFFFFFFFU || pixel == 0xFF000000U;
                REQUIRE(is_valid_output);
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "pole-facing sphere solid-color diagnostic has complete coverage", "[scene][sphere][poles]"
    ) {
        constexpr std::size_t width = 240U;
        constexpr std::size_t height = 180U;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage one_texel{};
        one_texel.width = 1U;
        one_texel.height = 1U;
        one_texel.pixels.push_back(0xFF336699U);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{one_texel};
        const auto camera = [](float pitch) noexcept {
            return raster::render::FreeLookCamera{
                .position = raster::Vec3f{.x = 0.0F, .y = pitch > 0.0F ? -3.0F : 3.0F, .z = 0.0F},
                .yaw = 0.0F,
                .pitch = pitch,
            };
        };

        for (const auto pitch : {1.35F, -1.35F}) {
            raster::Framebuffer framebuffer{width, height};
            const auto rendered = raster::render::render_scene(
                framebuffer,
                *mesh,
                0U,
                false,
                true,
                false,
                true,
                texture,
                camera(pitch)
            );
            REQUIRE(rendered);

            std::size_t sphere_pixels = 0;
            const auto colors = std::as_const(framebuffer).color_view();
            const auto depths = std::as_const(framebuffer).depth_view();
            for (const auto y : std::views::iota(std::size_t{0}, height)) {
                for (const auto x : std::views::iota(std::size_t{0}, width)) {
                    const auto color = colors[y, x];
                    const auto depth = depths[y, x];
                    REQUIRE(std::isfinite(depth));
                    if (depth > 0.0F) {
                        ++sphere_pixels;
                        REQUIRE(color == 0xFFB0B0B0U);
                    } else {
                        REQUIRE(color == 0xFF101018U);
                    }
                }
            }
            REQUIRE(sphere_pixels > width * height / 10U);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("sphere pole remains covered as projected cap shrinks", "[scene][sphere][coverage]") {
        constexpr std::size_t width = 320U;
        constexpr std::size_t height = 240U;
        constexpr std::size_t center_x = width / 2U;
        constexpr std::size_t center_y = height / 2U;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage one_texel{};
        one_texel.width = 1U;
        one_texel.height = 1U;
        one_texel.pixels.push_back(0xFF336699U);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{one_texel};
        constexpr std::array<float, 4> pole_camera_heights{2.25F, 3.0F, 5.0F, 10.0F};

        for (const auto pole_height : pole_camera_heights) {
            raster::Framebuffer framebuffer{width, height};
            const raster::render::FreeLookCamera camera{
                .position = raster::Vec3f{.x = 0.0F, .y = pole_height, .z = 0.0F},
                .yaw = 0.0F,
                .pitch = -std::numbers::pi_v<float> / 2.0F,
            };
            REQUIRE(
                raster::render::
                    render_scene(framebuffer, *mesh, 0U, false, true, false, true, texture, camera)
            );
            const auto color = std::as_const(framebuffer).color_view();
            const auto depth = std::as_const(framebuffer).depth_view();
            INFO("north-pole camera height " << pole_height);
            REQUIRE(depth[center_y, center_x] > 0.0F);
            REQUIRE(color[center_y, center_x] == 0xFFB0B0B0U);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("sphere camera-axis ray remains covered as it recedes", "[scene][sphere][coverage]") {
        constexpr std::size_t width = 320U;
        constexpr std::size_t height = 240U;
        constexpr std::size_t center_x = width / 2U;
        constexpr std::size_t center_y = height / 2U;
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        raster::TgaImage one_texel{};
        one_texel.width = 1U;
        one_texel.height = 1U;
        one_texel.pixels.push_back(0xFF336699U);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{one_texel};
        constexpr std::array<float, 5> camera_distances{3.0F, 4.0F, 5.0F, 8.0F, 12.0F};

        for (const auto distance : camera_distances) {
            raster::Framebuffer framebuffer{width, height};
            const raster::render::FreeLookCamera camera{
                .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = distance},
                .yaw = 0.0F,
                .pitch = 0.0F,
            };
            REQUIRE(
                raster::render::
                    render_scene(framebuffer, *mesh, 0U, false, true, false, true, texture, camera)
            );
            const auto color = std::as_const(framebuffer).color_view();
            const auto depth = std::as_const(framebuffer).depth_view();
            INFO("camera +Z distance " << distance);
            REQUIRE(depth[center_y, center_x] > 0.0F);
            REQUIRE(color[center_y, center_x] == 0xFFB0B0B0U);
        }
    }

    raster::SoAMesh make_two_sided_triangle(raster::Vec3f normal) {
        raster::SoAMesh mesh{};
        mesh.positions = {
            raster::Vec3f{.x = -1.0F, .y = -1.0F, .z = 0.0F},
            raster::Vec3f{.x = 1.0F, .y = -1.0F, .z = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F},
        };
        mesh.normals = {normal, normal, normal};
        mesh.uvs = {
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
        };
        mesh.indices = {0U, 1U, 2U, 0U, 2U, 1U};
        return mesh;
    }

    void check_lit_triangle(
        const raster::SoAMesh& mesh, std::uint32_t texel, std::uint32_t expected
    ) {
        // 1-texel texture makes UVs and filtering irrelevant: every covered
        // pixel must be exactly `expected`, everything else the clear color.
        // Rendered both serial and tiled to lock their parity under lighting.
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels.push_back(texel);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        constexpr std::size_t width = 64U;
        constexpr std::size_t height = 64U;
        constexpr std::uint32_t clear = 0xFF101018U;
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };

        const auto check_frame = [&](const raster::Framebuffer& framebuffer) {
            const auto colors = std::as_const(framebuffer).color_view();
            std::size_t lit = 0U;
            for (std::size_t y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < width; ++x) {
                    if (colors[y, x] == expected) {
                        ++lit;
                    } else {
                        REQUIRE(colors[y, x] == clear);
                    }
                }
            }
            REQUIRE(lit > 20U);
        };

        raster::Framebuffer serial{width, height};
        REQUIRE(
            raster::render::
                render_scene(serial, mesh, 0U, false, true, false, false, texture, camera)
        );
        check_frame(serial);

        raster::Framebuffer tiled{width, height};
        raster::render::TileRenderer renderer{4U};
        REQUIRE(renderer.render_scene(tiled, mesh, 0U, false, true, false, false, texture, camera));
        check_frame(tiled);
        REQUIRE(tiled.color_storage == serial.color_storage);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("backfacing-lit triangle renders ambient only", "[scene][material]") {
        // Sun is hardcoded as direction normalize((0, 3, 4)) (3-4-5 triangle,
        // so the normalized components are exactly (0, 0.6, 0.8)), white at
        // 0.6 with ambient 0.4. A (0, 0, -1) normal gives dot < 0 exactly, so
        // the diffuse term is exactly 0 and every channel is texel * 0.4:
        // 200 * 0.4 = 80.0, float error ~1e-5 against a 0.5 boundary.
        check_lit_triangle(
            make_two_sided_triangle(raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = -1.0F}),
            0xFFC8C8C8U,
            0xFF505050U
        );
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("face-on triangle reproduces texels exactly", "[scene][material]") {
        // Normal (0, 0.6, 0.8) matches the sun direction bit-for-bit, so
        // diffuse is 1 within 1e-7 and K = 0.4 + 0.6 * d is 1 within 1e-7.
        // Mid-range bytes stay 3e-5 from any rounding boundary: output must
        // equal the input texels exactly. Passes before lighting exists too,
        // which validates the harness (winding, coverage); the ambient test
        // above is the one that fails until lighting lands. With shadows
        // enabled it additionally pins self-shadow correctness: the bias must
        // cover interpolation error or this goes dark.
        check_lit_triangle(
            make_two_sided_triangle(raster::Vec3f{.x = 0.0F, .y = 0.6F, .z = 0.8F}),
            0xFF336699U,
            0xFF336699U
        );
    }

    raster::SoAMesh make_shadow_scene() {
        // Receiver: face-on to the sun, so unoccluded it reproduces texels
        // (M1 face-on test). Occluder: square perpendicular to the sun through
        // C = receiver_center + sun * 5, half-size 3 — between the sun-side
        // light and the receiver along every ray through the receiver
        // (shadow depth 0.667 vs receiver 0.5, far outside any bias). The
        // occluder sits behind the main camera, entirely outside its frustum,
        // so the receiver pixel shows the receiver, not the occluder. Its
        // normals face the light, so any main-view visibility would fail
        // loudly instead of hiding behind ambient shading.
        constexpr raster::Vec3f sun{.x = 0.0F, .y = 0.6F, .z = 0.8F};
        constexpr raster::Vec3f center{.x = 0.0F, .y = 3.0F, .z = 4.0F};
        const auto side =
            raster::normalize(raster::cross(sun, raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}));
        const auto up = raster::normalize(raster::cross(sun, side));
        constexpr float half = 3.0F;
        const std::array<raster::Vec3f, 4> corners{
            raster::Vec3f{
                .x = center.x - half * side.x - half * up.x,
                .y = center.y - half * side.y - half * up.y,
                .z = center.z - half * side.z - half * up.z,
            },
            raster::Vec3f{
                .x = center.x + half * side.x - half * up.x,
                .y = center.y + half * side.y - half * up.y,
                .z = center.z + half * side.z - half * up.z,
            },
            raster::Vec3f{
                .x = center.x + half * side.x + half * up.x,
                .y = center.y + half * side.y + half * up.y,
                .z = center.z + half * side.z + half * up.z,
            },
            raster::Vec3f{
                .x = center.x - half * side.x + half * up.x,
                .y = center.y - half * side.y + half * up.y,
                .z = center.z - half * side.z + half * up.z,
            },
        };
        const raster::Vec3f facing{.x = 0.0F, .y = 0.6F, .z = 0.8F};
        raster::SoAMesh mesh{};
        mesh.positions = {
            raster::Vec3f{.x = -1.0F, .y = -1.0F, .z = 0.0F},
            raster::Vec3f{.x = 1.0F, .y = -1.0F, .z = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F},
            corners[0],
            corners[1],
            corners[2],
            corners[3],
        };
        mesh.normals = {facing, facing, facing, facing, facing, facing, facing};
        mesh.uvs = {
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
        };
        mesh.indices = {
            0U,
            1U,
            2U,
            0U,
            2U,
            1U,
            3U,
            4U,
            5U,
            3U,
            5U,
            6U,
            3U,
            5U,
            4U,
            3U,
            6U,
            5U,
        };
        return mesh;
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("occluder darkens receiver to ambient", "[scene][shadow]") {
        // Same texel as the face-on test (0xFF336699), so unoccluded the pixel
        // would be the texel bytes. The occluder sits 0.1667 of reverse-Z
        // depth in front of the receiver along the sun ray — far outside any
        // bias — so every covered receiver lane must be ambient-only:
        // 51 * 0.4 = 20.4 -> 20, 102 * 0.4 = 40.8 -> 41,
        // 153 * 0.4 = 61.2 -> 61. Pre-shadow code renders the lit texel, so
        // this fails until the shadow path lands.
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels.push_back(0xFF336699U);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        constexpr std::size_t width = 64U;
        constexpr std::size_t height = 64U;
        constexpr std::uint32_t clear = 0xFF101018U;
        constexpr std::uint32_t expected = 0xFF14293DU;
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        const auto mesh = make_shadow_scene();
        REQUIRE(mesh.has_consistent_vertex_streams());

        const auto check_frame = [&](const raster::Framebuffer& framebuffer) {
            const auto colors = std::as_const(framebuffer).color_view();
            std::size_t dark = 0U;
            for (std::size_t y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < width; ++x) {
                    if (colors[y, x] == expected) {
                        ++dark;
                    } else {
                        REQUIRE(colors[y, x] == clear);
                    }
                }
            }
            REQUIRE(dark > 20U);
        };

        raster::Framebuffer serial{width, height};
        REQUIRE(
            raster::render::
                render_scene(serial, mesh, 0U, false, true, false, false, texture, camera)
        );
        check_frame(serial);

        raster::Framebuffer tiled{width, height};
        raster::render::TileRenderer renderer{4U};
        REQUIRE(renderer.render_scene(tiled, mesh, 0U, false, true, false, false, texture, camera));
        check_frame(tiled);
        REQUIRE(tiled.color_storage == serial.color_storage);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("textured sphere shows lit pixels under the sun", "[scene][shadow]") {
        // White texel: ambient-only is 0xFF666666 (255 * 0.4 = 102), while a
        // face-on pixel reaches ~224. Anything above 150 must be sunlit, so a
        // healthy sun + shadow pipeline shows many such pixels on a sphere
        // facing the light. Fails if the shadow path darkens everything.
        const auto sphere = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(sphere.has_value());

        raster::TgaImage white{};
        white.width = 1U;
        white.height = 1U;
        white.pixels.push_back(0xFFFFFFFFU);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{white};
        constexpr std::size_t width = 64U;
        constexpr std::size_t height = 64U;
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 5.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };

        raster::Framebuffer framebuffer{width, height};
        raster::render::TileRenderer renderer{4U};
        REQUIRE(
            renderer
                .render_scene(framebuffer, *sphere, 0U, false, true, false, false, texture, camera)
        );

        const auto colors = std::as_const(framebuffer).color_view();
        std::size_t lit = 0U;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto red = static_cast<int>((colors[y, x] >> 16U) & 0xFFU);
                if (red > 150) {
                    ++lit;
                }
            }
        }
        REQUIRE(lit > width * height / 20U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("shadow rendering is deterministic", "[scene][shadow]") {
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels.push_back(0xFF336699U);
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        const auto mesh = make_shadow_scene();
        raster::Framebuffer first{64U, 64U};
        raster::Framebuffer second{64U, 64U};
        REQUIRE(
            raster::render::
                render_scene(first, mesh, 0U, false, true, false, false, texture, camera)
        );
        REQUIRE(
            raster::render::
                render_scene(second, mesh, 0U, false, true, false, false, texture, camera)
        );
        REQUIRE(first.color_storage == second.color_storage);
        REQUIRE(first.depth_storage == second.depth_storage);
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
