#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <mdspan>
#include <numbers>
#include <vector>

#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/shader.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {

    struct UvTestShader final : raster::ShaderBase {
        struct Varyings {
            raster::Vec4f clip_pos{};
            raster::Vec2f uv{};

            [[nodiscard]]
            constexpr auto attributes(this const Varyings& self) noexcept {
                return std::tie(self.uv);
            }
        };

        [[nodiscard]]
        constexpr raster::Vec4f fragment(const Varyings& varying) const noexcept {
            return raster::Vec4f{.x = varying.uv.x, .y = varying.uv.y, .z = 0.0F, .w = 1.0F};
        }
    };

    static_assert(raster::ShaderProgram<UvTestShader>);

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "vector arithmetic is constexpr and vector norms use runtime std math",
        "[math][vec]"
    ) {
        STATIC_REQUIRE(sizeof(raster::Vec4f) == 16);
        STATIC_REQUIRE(alignof(raster::Vec4f) == 16);

        constexpr raster::Vec3f x_axis{.x = 1.0F, .y = 0.0F, .z = 0.0F};
        constexpr raster::Vec3f y_axis{.x = 0.0F, .y = 1.0F, .z = 0.0F};
        constexpr auto z_axis = raster::cross(x_axis, y_axis);
        STATIC_REQUIRE(z_axis.x == 0.0F);
        STATIC_REQUIRE(z_axis.y == 0.0F);
        STATIC_REQUIRE(z_axis.z == 1.0F);

        constexpr raster::Vec3f triple{.x = 3.0F, .y = 4.0F, .z = 0.0F};
        REQUIRE(raster::length(triple) == std::sqrt(25.0F));
        const auto unit = raster::normalize(triple);
        REQUIRE(unit.x == 3.0F / std::sqrt(25.0F));
        REQUIRE(unit.y == 4.0F / std::sqrt(25.0F));
        REQUIRE(unit.z == 0.0F);
        const auto zero_normalized = raster::normalize(raster::Vec3f{});
        REQUIRE(zero_normalized.x == 0.0F);
        REQUIRE(zero_normalized.y == 0.0F);
        REQUIRE(zero_normalized.z == 0.0F);

        // Convex combination with exact binary-fraction weights.
        constexpr auto blended = raster::lerp3(1.0F, 2.0F, 3.0F, 0.25F, 0.5F, 0.25F);
        STATIC_REQUIRE(blended == 2.0F);

        constexpr raster::Vec4f four{.x = 1.0F, .y = 2.0F, .z = 3.0F, .w = 4.0F};
        constexpr auto xyz = four.xyz();
        STATIC_REQUIRE(xyz.x == 1.0F);
        STATIC_REQUIRE(xyz.y == 2.0F);
        STATIC_REQUIRE(xyz.z == 3.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("matrix primitives", "[math][mat]") {
        constexpr auto identity = raster::identity4f();
        constexpr raster::Vec4f point{.x = 1.0F, .y = 2.0F, .z = 3.0F, .w = 1.0F};
        constexpr auto unchanged = identity * point;
        STATIC_REQUIRE(unchanged.x == 1.0F);
        STATIC_REQUIRE(unchanged.y == 2.0F);
        STATIC_REQUIRE(unchanged.z == 3.0F);
        STATIC_REQUIRE(unchanged.w == 1.0F);

        // C++23 multidimensional subscript: element (row, column). The
        // translation occupies the last column under the column-vector
        // convention.
        constexpr auto translated =
            raster::translation4f(raster::Vec3f{.x = 1.0F, .y = 2.0F, .z = 3.0F});
        STATIC_REQUIRE(translated[0, 3] == 1.0F);
        STATIC_REQUIRE(translated[1, 3] == 2.0F);
        STATIC_REQUIRE(translated[2, 3] == 3.0F);
        constexpr auto moved = translated * point;
        STATIC_REQUIRE(moved.x == 2.0F);
        STATIC_REQUIRE(moved.y == 4.0F);
        STATIC_REQUIRE(moved.z == 6.0F);

        const auto unrotated = raster::rotation_y4f(0.0F);
        REQUIRE(unrotated[0, 0] == 1.0F);
        REQUIRE(unrotated[1, 1] == 1.0F);
        REQUIRE(unrotated[2, 2] == 1.0F);

        // Composition of translations is exact in binary floats.
        constexpr auto combined =
            raster::translation4f(raster::Vec3f{.x = 1.0F, .y = 0.0F, .z = 0.0F})
            * raster::translation4f(raster::Vec3f{.x = 2.0F, .y = 0.0F, .z = 0.0F});
        constexpr auto composed_point = combined * point;
        STATIC_REQUIRE(composed_point.x == 4.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("look_at builds the right-handed view basis", "[math][mat]") {
        const auto view = raster::look_at4f(
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 5.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );

        // World origin sits 5 units in front: view z must be -5 (RH contract).
        const auto origin_view = view * raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = 0.0F, .w = 1.0F};
        REQUIRE(origin_view.x == 0.0F);
        REQUIRE(origin_view.y == 0.0F);
        REQUIRE(origin_view.z == -5.0F);
        REQUIRE(origin_view.w == 1.0F);

        // +X world stays +X view; +Y world stays +Y view.
        const auto x_view = view * raster::Vec4f{.x = 5.0F, .y = 0.0F, .z = 0.0F, .w = 1.0F};
        REQUIRE(x_view.x == 5.0F);
        REQUIRE(x_view.z == -5.0F);
        const auto y_view = view * raster::Vec4f{.x = 0.0F, .y = 5.0F, .z = 0.0F, .w = 1.0F};
        REQUIRE(y_view.y == 5.0F);
        REQUIRE(y_view.z == -5.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "reverse-z focal projection maps near to 1 and far to 0 at compile time",
        "[math][mat]"
    ) {
        // n=1, f=2 makes every depth-row entry an exact binary float, so the
        // boundary mapping is exact under round-to-nearest.
        constexpr auto projection =
            raster::perspective_reverse_z_from_focal4f(1.0F, 1.0F, 1.0F, 2.0F);
        constexpr auto clip_near =
            projection * raster::Vec4f{.x = 3.0F, .y = -2.0F, .z = -1.0F, .w = 1.0F};
        STATIC_REQUIRE(clip_near.w == 1.0F);
        STATIC_REQUIRE(clip_near.z / clip_near.w == 1.0F);
        constexpr auto clip_far =
            projection * raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -2.0F, .w = 1.0F};
        STATIC_REQUIRE(clip_far.w == 2.0F);
        STATIC_REQUIRE(clip_far.z == 0.0F);
        STATIC_REQUIRE(clip_far.z / clip_far.w == 0.0F);

        constexpr auto mid_clip =
            projection * raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -1.5F, .w = 1.0F};
        STATIC_REQUIRE(mid_clip.z / mid_clip.w > 0.0F);
        STATIC_REQUIRE(mid_clip.z / mid_clip.w < 1.0F);

        // Second exact configuration (n=2, f=4).
        constexpr auto projection2 =
            raster::perspective_reverse_z_from_focal4f(1.0F, 1.0F, 2.0F, 4.0F);
        constexpr auto clip2_near =
            projection2 * raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -2.0F, .w = 1.0F};
        STATIC_REQUIRE(clip2_near.z / clip2_near.w == 1.0F);
        constexpr auto clip2_far =
            projection2 * raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -4.0F, .w = 1.0F};
        STATIC_REQUIRE(clip2_far.z == 0.0F);

        // Realistic near/far stays within float rounding of the boundaries.
        constexpr auto projection_real =
            raster::perspective_reverse_z_from_focal4f(1.0F, 1.0F, 0.1F, 1000.0F);
        constexpr auto real_near =
            projection_real * raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -0.1F, .w = 1.0F};
        constexpr auto real_near_z = real_near.z / real_near.w;
        STATIC_REQUIRE(real_near_z > 0.999999F);
        STATIC_REQUIRE(real_near_z < 1.000001F);
        constexpr auto real_far =
            projection_real * raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -1000.0F, .w = 1.0F};
        STATIC_REQUIRE(real_far.z == 0.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("reverse-z projection rejects invalid parameters", "[math][mat]") {
        const auto invalid = raster::perspective_reverse_z4f(
            std::numbers::pi_v<float> / 2.0F,
            1.0F,
            0.0F, // zero near plane is invalid
            1000.0F
        );
        REQUIRE(invalid[2, 2] == 1.0F);
        REQUIRE(invalid[2, 3] == 0.0F);
        REQUIRE(invalid[3, 3] == 1.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("runtime math wrappers agree with standard math", "[math][runtime]") {
        const auto angle = 0.37F;
        REQUIRE(std::fabs(raster::rotation_y4f(angle)[0, 0] - std::cos(angle)) <= 1e-7F);
        const auto runtime_projection = raster::perspective_reverse_z4f(1.1F, 1.25F, 0.1F, 1000.0F);
        const auto runtime_focal = 1.0F / std::tan(0.55F);
        REQUIRE(std::fabs(runtime_projection[0, 0] - runtime_focal / 1.25F) <= 1e-7F);

        const auto vector = raster::Vec3f{.x = 3.0F, .y = 4.0F, .z = 0.0F};
        REQUIRE(raster::length(vector) == std::sqrt(25.0F));
        REQUIRE(raster::normalize(vector).x == 3.0F / std::sqrt(25.0F));
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("oblique checker sample uses perspective-correct uv", "[math][perspective]") {
        constexpr std::size_t target_x = 79;
        constexpr std::size_t target_y = 126;
        constexpr std::size_t width = 320;
        constexpr std::size_t height = 240;
        const raster::Vec3f eye{.x = 0.0F, .y = 0.1F, .z = 0.0F};
        const auto view = raster::look_at4f(
            eye,
            raster::Vec3f{.x = 0.0F, .y = 0.0999F, .z = -1.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto projection = raster::perspective_reverse_z4f(
            std::numbers::pi_v<float> / 2.0F,
            static_cast<float>(width) / static_cast<float>(height),
            0.1F,
            1000.0F
        );
        const auto mvp = projection * view;

        // Non-coplanar clip w values make affine screen interpolation visibly
        // differ from the analytic perspective-correct checker UV.
        const std::array<raster::Vec3f, 3> world{
            raster::Vec3f{.x = -3.0F, .y = 0.0F, .z = -1.0F},
            raster::Vec3f{.x = 5.0F, .y = 0.0F, .z = -4.0F},
            raster::Vec3f{.x = -2.0F, .y = 0.0F, .z = -7.0F},
        };
        const std::array<raster::Vec2f, 3> uv{
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 8.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 8.0F},
        };
        std::array<raster::Vec4f, 3> clip{};
        std::array<UvTestShader::Varyings, 3> varying{};
        std::array<float, 3> inv_w{};
        std::array<float, 3> z_ndc{};
        std::array<raster::ScreenPoint, 3> screen{};
        for (const auto vertex : std::views::iota(std::size_t{0}, std::size_t{3})) {
            clip[vertex] = mvp
                * raster::Vec4f{
                    .x = world[vertex].x,
                    .y = world[vertex].y,
                    .z = world[vertex].z,
                    .w = 1.0F,
                };
            varying[vertex] = UvTestShader::Varyings{
                .clip_pos = clip[vertex],
                .uv = uv[vertex],
            };
            REQUIRE(clip[vertex].w > 0.0F);
            inv_w[vertex] = 1.0F / clip[vertex].w;
            z_ndc[vertex] = clip[vertex].z * inv_w[vertex];
            screen[vertex] = raster::viewport_to_screen(
                clip[vertex].x * inv_w[vertex],
                clip[vertex].y * inv_w[vertex],
                width,
                height
            );
        }

        // The projected vertices have negative screen-space winding in this
        // order; swap B/C to produce a front-facing triangle and keep all
        // associated varying/depth data in the same reordered convention.
        const std::array<std::size_t, 3> front_order{0, 2, 1};
        const std::array<float, 3> front_inv_w{
            inv_w[front_order[0]],
            inv_w[front_order[1]],
            inv_w[front_order[2]],
        };
        const std::array<float, 3> front_z_ndc{
            z_ndc[front_order[0]],
            z_ndc[front_order[1]],
            z_ndc[front_order[2]],
        };
        const auto triangle = raster::make_screen_triangle(
            screen[front_order[0]],
            screen[front_order[1]],
            screen[front_order[2]],
            front_inv_w,
            front_z_ndc
        );
        const std::array<UvTestShader::Varyings, 3> front_varyings{
            varying[front_order[0]],
            varying[front_order[1]],
            varying[front_order[2]],
        };
        REQUIRE(triangle.has_value());
        std::vector<std::uint32_t> colors(width * height, 0U);
        std::vector<float> depths(width * height, 0.0F);
        auto color_view = std::mdspan<std::uint32_t, raster::Extents2D>{
            colors.data(),
            height,
            width,
        };
        auto depth_view = std::mdspan<float, raster::Extents2D>{depths.data(), height, width};
        bool sample_drawn = false;
        float sampled_u = 0.0F;
        float sampled_v = 0.0F;
        bool interpolation_matches_reference = true;
        raster::PixelBarycentrics sample_barycentrics{};
        constexpr UvTestShader shader{};

        raster::rasterize_triangle(
            *triangle,
            width,
            height,
            depth_view,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            // ShaderBase's tuple interpolation is the path under test.
            const auto interpolated =
                shader.interpolate(front_varyings[0], front_varyings[1], front_varyings[2], bary);
            const auto corrected_uv = std::get<0>(interpolated);

            // Independent rational reference: interpolate uv_i / w_i and
            // divide by the interpolated reciprocal w.
            const auto numerator_u = raster::lerp3(
                uv[front_order[0]].x * front_inv_w[0],
                uv[front_order[1]].x * front_inv_w[1],
                uv[front_order[2]].x * front_inv_w[2],
                bary.l0,
                bary.l1,
                bary.l2
            );
            const auto numerator_v = raster::lerp3(
                uv[front_order[0]].y * front_inv_w[0],
                uv[front_order[1]].y * front_inv_w[1],
                uv[front_order[2]].y * front_inv_w[2],
                bary.l0,
                bary.l1,
                bary.l2
            );
            const float reciprocal_w =
                bary.l0 * front_inv_w[0] + bary.l1 * front_inv_w[1] + bary.l2 * front_inv_w[2];
            interpolation_matches_reference = interpolation_matches_reference
                && std::fabs(corrected_uv.x - numerator_u / reciprocal_w)
                <= 1e-6F
                && std::fabs(corrected_uv.y - numerator_v / reciprocal_w)
                <= 1e-6F;

            const auto shaded = shader.fragment(
                UvTestShader::Varyings{
                    .clip_pos = front_varyings[0].clip_pos,
                    .uv = corrected_uv,
                }
            );
            const auto checker_cell =
                static_cast<int>(std::floor(shaded.x)) + static_cast<int>(std::floor(shaded.y));
            const auto checker_shade =
                ((checker_cell % 2) + 2) % 2 == 0 ? 0xFFCCCCCCU : 0xFF555555U;
            color_view[y, x] = checker_shade;
            if (x == target_x && y == target_y) {
                sample_drawn = true;
                sampled_u = corrected_uv.x;
                sampled_v = corrected_uv.y;
                sample_barycentrics = bary;
            }
        }
        );

        REQUIRE(sample_drawn);
        REQUIRE(interpolation_matches_reference);
        REQUIRE(std::isfinite(sampled_u));
        REQUIRE(std::isfinite(sampled_v));
        const auto expected_u_numerator = raster::lerp3(
            uv[front_order[0]].x * front_inv_w[0],
            uv[front_order[1]].x * front_inv_w[1],
            uv[front_order[2]].x * front_inv_w[2],
            sample_barycentrics.l0,
            sample_barycentrics.l1,
            sample_barycentrics.l2
        );
        const auto expected_v_numerator = raster::lerp3(
            uv[front_order[0]].y * front_inv_w[0],
            uv[front_order[1]].y * front_inv_w[1],
            uv[front_order[2]].y * front_inv_w[2],
            sample_barycentrics.l0,
            sample_barycentrics.l1,
            sample_barycentrics.l2
        );
        const float expected_reciprocal_w = sample_barycentrics.l0
            * front_inv_w[0]
            + sample_barycentrics.l1
            * front_inv_w[1]
            + sample_barycentrics.l2
            * front_inv_w[2];
        const float expected_u = expected_u_numerator / expected_reciprocal_w;
        const float expected_v = expected_v_numerator / expected_reciprocal_w;
        const float affine_u = sample_barycentrics.l0
            * uv[front_order[0]].x
            + sample_barycentrics.l1
            * uv[front_order[1]].x
            + sample_barycentrics.l2
            * uv[front_order[2]].x;
        const float affine_v = sample_barycentrics.l0
            * uv[front_order[0]].y
            + sample_barycentrics.l1
            * uv[front_order[1]].y
            + sample_barycentrics.l2
            * uv[front_order[2]].y;
        REQUIRE(std::fabs(sampled_u - expected_u) <= 1e-6F);
        REQUIRE(std::fabs(sampled_v - expected_v) <= 1e-6F);
        REQUIRE(std::fabs(sampled_u - affine_u) > 1e-2F);
        REQUIRE(std::fabs(sampled_v - affine_v) > 1e-2F);
        const auto sample_color = colors[target_y * width + target_x];
        const bool sample_is_checker_color =
            sample_color == 0xFFCCCCCCU || sample_color == 0xFF555555U;
        REQUIRE(sample_is_checker_color);
        REQUIRE(depths[target_y * width + target_x] > 0.0F);
        REQUIRE(depths[target_y * width + target_x] <= 1.0F);
    }

} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
