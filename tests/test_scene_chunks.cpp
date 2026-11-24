#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <ranges>
#include <vector>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/math/vec.hpp"
#include "raster/memory/grid_mesh.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/render/scene_chunks.hpp"
#include "raster/render/scene_renderer.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    raster::SoAMesh translated_mesh(const raster::SoAMesh& mesh, raster::Vec3f offset) {
        auto moved = mesh;
        for (auto& position : moved.positions) {
            position = position + offset;
        }
        return moved;
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("chunk bounds cover exactly their triangle range", "[scene][chunks]") {
        raster::SoAMesh mesh{};
        mesh.positions = {
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 0.0F},
            raster::Vec3f{.x = 2.0F, .y = 0.0F, .z = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 2.0F, .z = 0.0F},
            raster::Vec3f{.x = 10.0F, .y = 10.0F, .z = 10.0F},
            raster::Vec3f{.x = 12.0F, .y = 10.0F, .z = 10.0F},
            raster::Vec3f{.x = 10.0F, .y = 12.0F, .z = 10.0F},
        };
        mesh.normals = {
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
        };
        mesh.uvs = {
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 1.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 1.0F},
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 1.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 1.0F},
        };
        mesh.indices = {0U, 1U, 2U, 3U, 4U, 5U};
        REQUIRE(mesh.has_consistent_vertex_streams());

        const auto first = raster::render::make_chunk_bounds(mesh, 0U, 1U);
        REQUIRE(first.first_triangle == 0U);
        REQUIRE(first.triangle_count == 1U);
        REQUIRE(first.bounds_center.x == 1.0F);
        REQUIRE(first.bounds_center.y == 1.0F);
        REQUIRE(first.bounds_center.z == 0.0F);
        REQUIRE(first.bounds_radius == std::sqrt(2.0F));
        REQUIRE(first.distance_to(first.bounds_center) == 0.0F);

        const auto invalid = raster::render::make_chunk_bounds(mesh, 1U, 2U);
        REQUIRE(invalid.triangle_count == 0U);
        const auto empty = raster::render::make_chunk_bounds(mesh, 0U, 0U);
        REQUIRE(empty.triangle_count == 0U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("combining meshes concatenates streams with offset indices", "[scene][chunks]") {
        raster::SoAMesh left{};
        left.positions = {
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 0.0F},
            raster::Vec3f{.x = 1.0F, .y = 0.0F, .z = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F},
        };
        left.normals = {
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
        };
        left.uvs = {
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 1.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 1.0F},
        };
        left.indices = {0U, 1U, 2U};

        raster::SoAMesh right =
            translated_mesh(left, raster::Vec3f{.x = 5.0F, .y = 0.0F, .z = 0.0F});
        const std::array<const raster::SoAMesh*, 2> halves{&left, &right};
        const auto combined = raster::combine_meshes(halves);
        REQUIRE(combined.has_consistent_vertex_streams());
        REQUIRE(combined.triangle_count() == 2U);
        REQUIRE(combined.positions.size() == 6U);
        REQUIRE(combined.indices[3U] == 3U);
        REQUIRE(combined.indices[4U] == 4U);
        REQUIRE(combined.indices[5U] == 5U);
        REQUIRE(combined.positions[3U].x == 5.0F);

        const std::array<const raster::SoAMesh*, 0> no_parts{};
        const auto none = raster::combine_meshes(no_parts);
        REQUIRE(none.triangle_count() == 0U);
        REQUIRE(none.has_consistent_vertex_streams());
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("bundled prop meshes load with consistent streams", "[asset][prop]") {
        constexpr std::array<const char*, 3> props{
            "models/prop_crate.obj",
            "models/prop_pillar.obj",
            "models/prop_pyramid.obj",
        };
        constexpr std::array<std::size_t, 3> triangles{12U, 32U, 6U};
        for (std::size_t index = 0U; index < props.size(); ++index) {
            const auto mesh =
                raster::load_obj(std::filesystem::path{RASTER_ASSET_DIR} / props[index]);
            REQUIRE(mesh.has_value());
            REQUIRE(mesh->has_consistent_vertex_streams());
            REQUIRE(mesh->triangle_count() == triangles[index]);
            for (const auto vertex : mesh->indices) {
                REQUIRE(static_cast<std::size_t>(vertex) < mesh->positions.size());
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("chunked prop scene matches whole-mesh render", "[scene][chunks][prop]") {
        const auto crate =
            raster::load_obj(std::filesystem::path{RASTER_ASSET_DIR} / "models/prop_crate.obj");
        const auto pillar =
            raster::load_obj(std::filesystem::path{RASTER_ASSET_DIR} / "models/prop_pillar.obj");
        const auto pyramid =
            raster::load_obj(std::filesystem::path{RASTER_ASSET_DIR} / "models/prop_pyramid.obj");
        REQUIRE(crate.has_value());
        REQUIRE(pillar.has_value());
        REQUIRE(pyramid.has_value());

        const auto floor = raster::make_grid_mesh(8U, 8U, 4.0F, -0.5F, -20.0F, 8.0F);
        const auto placed_crate =
            translated_mesh(*crate, raster::Vec3f{.x = -2.0F, .y = 0.5F, .z = -5.0F});
        const auto placed_pillar =
            translated_mesh(*pillar, raster::Vec3f{.x = 2.0F, .y = 0.0F, .z = -6.0F});
        const auto placed_pyramid =
            translated_mesh(*pyramid, raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = -3.5F});
        const std::array<const raster::SoAMesh*, 4>
            parts{&floor, &placed_crate, &placed_pillar, &placed_pyramid};
        const auto combined = raster::combine_meshes(parts);
        REQUIRE(combined.has_consistent_vertex_streams());

        std::vector<raster::render::SceneChunk> chunks;
        std::size_t first = 0U;
        for (const auto* part : parts) {
            chunks.push_back(
                raster::render::make_chunk_bounds(combined, first, part->triangle_count())
            );
            first += part->triangle_count();
        }
        REQUIRE(raster::render::valid_scene_chunks(chunks, combined.triangle_count()));
        REQUIRE(chunks.size() == 4U);

        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 2.0F, .z = 3.5F},
            .yaw = 0.0F,
            .pitch = -0.35F,
        };
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFFFFFFFFU};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        constexpr std::size_t width = 96U;
        constexpr std::size_t height = 80U;
        raster::Framebuffer whole{width, height};
        raster::Framebuffer chunked{width, height};
        raster::render::TileRenderer renderer{4U};
        raster::render::SceneRenderDiagnostics whole_stats{};
        raster::render::SceneRenderDiagnostics chunk_stats{};
        const raster::Vec3f lod_origin{.x = 0.0F, .y = 2.0F, .z = 3.5F};
        REQUIRE(
            raster::render::render_scene(
                whole,
                combined,
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
                &whole_stats,
                {},
                &lod_origin
            )
        );
        REQUIRE(renderer.render_scene(
            chunked,
            combined,
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
            &chunk_stats,
            chunks,
            &lod_origin
        ));
        REQUIRE(chunked.color_storage == whole.color_storage);
        REQUIRE(chunked.depth_storage == whole.depth_storage);
        REQUIRE(chunk_stats.source_chunks == 4U);
        REQUIRE(chunk_stats.frustum_rejected_chunks == 0U);
        std::size_t lit_pixels = 0U;
        for (const auto pixel : whole.color_storage) {
            lit_pixels += pixel != 0xFF101018U ? 1U : 0U;
        }
        REQUIRE(lit_pixels > width * height / 10U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("distant chunk culls exactly its prop", "[scene][m3][chunks]") {
        const auto crate =
            raster::load_obj(std::filesystem::path{RASTER_ASSET_DIR} / "models/prop_crate.obj");
        const auto pillar =
            raster::load_obj(std::filesystem::path{RASTER_ASSET_DIR} / "models/prop_pillar.obj");
        REQUIRE(crate.has_value());
        REQUIRE(pillar.has_value());

        const auto floor = raster::make_grid_mesh(8U, 8U, 4.0F, -0.5F, -20.0F, 8.0F);
        const auto near_crate =
            translated_mesh(*crate, raster::Vec3f{.x = -2.0F, .y = 0.5F, .z = -5.0F});
        const auto far_pillar =
            translated_mesh(*pillar, raster::Vec3f{.x = 500.0F, .y = 0.0F, .z = -6.0F});
        // Trimmed scene omits the pillar entirely: culling the distant chunk
        // must render exactly like a scene that never contained it.
        const std::array<const raster::SoAMesh*, 2> trimmed_parts{&floor, &near_crate};
        const std::array<const raster::SoAMesh*, 3> far_parts{&floor, &near_crate, &far_pillar};
        const auto trimmed_scene = raster::combine_meshes(trimmed_parts);
        const auto far_scene = raster::combine_meshes(far_parts);

        const auto far_chunks = [&] {
            std::vector<raster::render::SceneChunk> built;
            std::size_t first = 0U;
            for (const auto* part : far_parts) {
                built.push_back(
                    raster::render::make_chunk_bounds(far_scene, first, part->triangle_count())
                );
                first += part->triangle_count();
            }
            return built;
        }();
        REQUIRE(raster::render::valid_scene_chunks(far_chunks, far_scene.triangle_count()));

        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 2.0F, .z = 3.5F},
            .yaw = 0.0F,
            .pitch = -0.35F,
        };
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFFFFFFFFU};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        constexpr std::size_t width = 96U;
        constexpr std::size_t height = 80U;
        raster::Framebuffer reference{width, height};
        raster::Framebuffer culled{width, height};
        raster::render::TileRenderer renderer{4U};
        raster::render::SceneRenderDiagnostics cull_stats{};
        const raster::Vec3f lod_origin{.x = 0.0F, .y = 2.0F, .z = 3.5F};
        REQUIRE(
            raster::render::render_scene(
                reference,
                trimmed_scene,
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
                nullptr,
                {},
                &lod_origin
            )
        );
        REQUIRE(renderer.render_scene(
            culled,
            far_scene,
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
            &cull_stats,
            far_chunks,
            &lod_origin
        ));
        REQUIRE(culled.color_storage == reference.color_storage);
        REQUIRE(culled.depth_storage == reference.depth_storage);
        REQUIRE(cull_stats.source_chunks == 3U);
        REQUIRE(cull_stats.frustum_rejected_chunks == 1U);
        REQUIRE(cull_stats.rasterized_triangle_submissions > 0U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("tiled grid matches row-major grid triangle for triangle", "[scene][chunks]") {
        const auto row_major = raster::make_grid_mesh(9U, 7U, 4.0F, -0.5F, -20.0F, 2.0F);
        const auto tiled = raster::make_grid_mesh_tiled(9U, 7U, 4.0F, -0.5F, -20.0F, 2.0F, 4U);
        REQUIRE(tiled.has_consistent_vertex_streams());
        REQUIRE(tiled.triangle_count() == row_major.triangle_count());
        REQUIRE(tiled.positions == row_major.positions);
        REQUIRE(tiled.normals == row_major.normals);
        REQUIRE(tiled.uvs == row_major.uvs);

        const auto key = [](const raster::SoAMesh& mesh, std::size_t triangle) {
            std::array<std::size_t, 3> corners{
                static_cast<std::size_t>(mesh.indices[triangle * 3U]),
                static_cast<std::size_t>(mesh.indices[triangle * 3U + 1U]),
                static_cast<std::size_t>(mesh.indices[triangle * 3U + 2U]),
            };
            std::ranges::sort(corners);
            return corners;
        };
        std::vector<std::array<std::size_t, 3>> row_tris;
        std::vector<std::array<std::size_t, 3>> tiled_tris;
        for (std::size_t triangle = 0U; triangle < row_major.triangle_count(); ++triangle) {
            row_tris.push_back(key(row_major, triangle));
            tiled_tris.push_back(key(tiled, triangle));
        }
        std::ranges::sort(row_tris);
        std::ranges::sort(tiled_tris);
        REQUIRE(tiled_tris == row_tris);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("behind-camera grid culls every chunk", "[scene][m3][chunks]") {
        const auto mesh = raster::make_grid_mesh_tiled(24U, 24U, 30.0F, -0.5F, -90.0F, 8.0F, 8U);
        const auto chunks = raster::render::make_scene_chunks(mesh, 128U);
        REQUIRE(!chunks.empty());
        REQUIRE(raster::render::valid_scene_chunks(chunks, mesh.triangle_count()));

        const raster::render::FreeLookCamera behind{
            .position = raster::Vec3f{.x = 0.0F, .y = 6.0F, .z = 6.0F},
            .yaw = std::numbers::pi_v<float>,
            .pitch = -0.19F,
        };
        raster::TgaImage image{};
        image.width = 1U;
        image.height = 1U;
        image.pixels = {0xFFFFFFFFU};
        const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
        raster::Framebuffer frame{64U, 48U};
        raster::render::SceneRenderDiagnostics stats{};
        const raster::Vec3f lod_origin{.x = 0.0F, .y = 6.0F, .z = 6.0F};
        REQUIRE(
            raster::render::render_scene(
                frame,
                mesh,
                0U,
                false,
                true,
                false,
                false,
                texture,
                behind,
                raster::render::SceneShading::Diffuse,
                raster::pbr::Material(),
                nullptr,
                &stats,
                chunks,
                &lod_origin
            )
        );
        REQUIRE(stats.source_chunks == chunks.size());
        REQUIRE(stats.frustum_rejected_chunks == chunks.size());
        REQUIRE(stats.prepared_triangles == 0U);
        const auto colors = std::as_const(frame).color_view();
        const auto depths = std::as_const(frame).depth_view();
        for (std::size_t y = 0U; y < 48U; ++y) {
            for (std::size_t x = 0U; x < 64U; ++x) {
                REQUIRE(colors[y, x] == 0xFF101018U);
                REQUIRE(depths[y, x] == 0.0F);
            }
        }
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
