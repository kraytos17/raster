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
#include <span>
#include <tuple>
#include <vector>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/pipeline/clipper.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/shader.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
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
                    std::array{near_left, far_right, near_right},
                    std::array{near_left, far_left, far_right},
                };

                for (const auto& source : source_triangles) {
                    const auto clipped =
                        raster::clip_triangle(shader, source[0], source[1], source[2]);
                    REQUIRE(clipped.has_value());
                    for (const auto& clipped_triangle :
                         std::span{clipped->triangles.data(), clipped->size}) {
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
                            const auto pixel_index = y * width + x;
                            ++coverage[pixel_index];
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
        for (const auto index : std::views::iota(std::size_t{0}, width * height)) {
            REQUIRE(coverage[index] <= 1U); // no shared-edge double writes / z-fighting
            if (coverage[index] == 0U) {
                REQUIRE(depths[index] == 0.0F);
                continue;
            }

            ++covered_pixels;
            REQUIRE(std::isfinite(depths[index]));
            REQUIRE(depths[index] >= 0.0F);
            REQUIRE(depths[index] <= 1.0F);
            minimum_depth = std::min(minimum_depth, depths[index]);
            maximum_depth = std::max(maximum_depth, depths[index]);
            light_checker_pixels += colors[index] == 0xFFCCCCCCU ? 1U : 0U;
            dark_checker_pixels += colors[index] == 0xFF555555U ? 1U : 0U;
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
        REQUIRE(mesh->triangle_count() >= 1000U);

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
            for (const auto& guard_triangle :
                 std::span{guard_clipped->triangles.data(), guard_clipped->size}) {
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
            for (const auto& triangle : std::span{clipped->triangles.data(), clipped->size}) {
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
        REQUIRE(rasterized_fragments > 0U);

        const auto first_covered = std::ranges::find(equator_coverage, std::uint8_t{1});
        REQUIRE(first_covered != equator_coverage.end());
        const auto last_covered = std::ranges::find_last(equator_coverage, std::uint8_t{1});
        REQUIRE(last_covered.begin() != last_covered.end());
        for (auto pixel = first_covered; pixel != last_covered.begin(); ++pixel) {
            REQUIRE(*pixel == 1U);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "partially covered quad keeps finite helper-lane UV derivatives", "[scene][m5][mipmap]"
    ) {
        constexpr std::size_t width = 8;
        constexpr std::size_t height = 8;
        const auto triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 3.0F, .y = 1.0F},
            raster::ScreenPoint{.x = 7.0F, .y = 1.0F},
            raster::ScreenPoint{.x = 3.0F, .y = 7.0F},
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.5F, 0.5F, 0.5F}
        );
        REQUIRE(triangle.has_value());

        raster::TgaImage checker_image{};
        checker_image.width = 64U;
        checker_image.height = 64U;
        checker_image.pixels.resize(checker_image.width * checker_image.height);
        for (const auto y : std::views::iota(std::size_t{0}, checker_image.height)) {
            for (const auto x : std::views::iota(std::size_t{0}, checker_image.width)) {
                checker_image.pixels[y * checker_image.width + x] =
                    ((x + y) & 1U) == 0U ? 0xFFFFFFFFU : 0xFF000000U;
            }
        }
        const raster::Texture2D<raster::LayoutTiled4x4> checker{checker_image};
        std::vector<float> depth_storage(width * height, 0.0F);
        depth_storage[3U * width + 4U] = 1.0F; // occlude one covered lane at the silhouette
        auto depth_view =
            std::mdspan<float, raster::Extents2D>{depth_storage.data(), height, width};

        bool saw_partial_quad = false;
        bool saw_partially_occluded_quad = false;
        float minimum_partial_lod = std::numeric_limits<float>::infinity();
        float maximum_partial_lod = 0.0F;
        std::size_t tested_visible_lanes = 0;
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
                    .x = bary.a1 * 8.0F,
                    .y = bary.a2 * 8.0F,
                };
                REQUIRE(std::isfinite(helper_uv[lane].x));
                REQUIRE(std::isfinite(helper_uv[lane].y));
            }
            const auto derivatives = raster::quad_derivatives(helper_uv);
            const auto lod = raster::mip_lod(derivatives, checker.width(), checker.height());
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
                const auto expected = checker.sample_trilinear(
                    helper_uv[lane].x,
                    helper_uv[lane].y,
                    raster::mip_lod(derivatives, checker.width(), checker.height())
                );
                REQUIRE(filtered == expected);
                REQUIRE(filtered == 0xFF7F7F7FU);
                ++tested_visible_lanes;
            }
            INFO("partial quad at " << quad_x << ',' << quad_y);
        }
        );
        REQUIRE(saw_partial_quad);
        REQUIRE(saw_partially_occluded_quad);
        REQUIRE(tested_visible_lanes > 0U);
        REQUIRE(maximum_partial_lod - minimum_partial_lod < 1e-5F);
        REQUIRE(std::fabs(minimum_partial_lod - 7.0F) < 0.1F);
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
