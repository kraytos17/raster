#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ranges>
#include <tuple>

#include "raster/pipeline/clipper.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    struct ClipTestShader final : raster::ShaderBase {
        struct Varyings {
            raster::Vec4f clip_pos{};
            raster::Vec2f uv{};
            raster::Vec3f color{};

            [[nodiscard]]
            constexpr auto attributes(this auto&& self) noexcept {
                return std::tie(self.uv, self.color);
            }
        };

        [[nodiscard]]
        constexpr raster::Vec4f fragment(const Varyings& varying) const noexcept {
            return raster::Vec4f{
                .x = varying.color.x,
                .y = varying.color.y,
                .z = varying.color.z,
                .w = 1.0F,
            };
        }
    };

    static_assert(raster::ShaderProgram<ClipTestShader>);

    [[nodiscard]]
    constexpr ClipTestShader::Varyings make_vertex(
        raster::Vec4f clip_pos, raster::Vec2f uv, raster::Vec3f color
    ) noexcept {
        return ClipTestShader::Varyings{
            .clip_pos = clip_pos,
            .uv = uv,
            .color = color,
        };
    }

    [[nodiscard]]
    bool inside_all_planes(const raster::Vec4f& p) noexcept {
        return p.w
            > 0.0F
            && p.x
            >= -p.w
            && p.x
            <= p.w
            && p.y
            >= -p.w
            && p.y
            <= p.w
            && p.z
            >= 0.0F
            && p.z
            <= p.w;
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("clipper trivially accepts an in-frustum triangle", "[clipper][accept]") {
        constexpr ClipTestShader shader{};
        constexpr auto a = make_vertex(
            raster::Vec4f{.x = -0.5F, .y = -0.5F, .z = 0.25F, .w = 1.0F},
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec3f{.x = 1.0F, .y = 0.0F, .z = 0.0F}
        );
        constexpr auto b = make_vertex(
            raster::Vec4f{.x = 0.5F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
            raster::Vec2f{.x = 1.0F, .y = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        constexpr auto c = make_vertex(
            raster::Vec4f{.x = 0.0F, .y = 0.5F, .z = 0.75F, .w = 1.0F},
            raster::Vec2f{.x = 0.5F, .y = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F}
        );

        const auto clipped = raster::clip_triangle(shader, a, b, c);
        REQUIRE(clipped.has_value());
        REQUIRE(clipped->size == 1);
        REQUIRE(clipped->active_triangles().size() == 1U);
        REQUIRE(clipped->triangles[0].vertices[0].clip_pos.x == a.clip_pos.x);
        REQUIRE(clipped->triangles[0].vertices[1].uv.x == b.uv.x);
        REQUIRE(clipped->triangles[0].vertices[2].color.z == c.color.z);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "clipper trivially rejects a triangle outside each canonical plane", "[clipper][reject]"
    ) {
        constexpr ClipTestShader shader{};
        constexpr std::array<raster::Vec4f, raster::clip_plane_count> outside_points{
            raster::Vec4f{.x = -2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F}, // left
            raster::Vec4f{.x = 2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F}, // right
            raster::Vec4f{.x = 0.0F, .y = -2.0F, .z = 0.5F, .w = 1.0F}, // bottom
            raster::Vec4f{.x = 0.0F, .y = 2.0F, .z = 0.5F, .w = 1.0F}, // top
            raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -0.5F, .w = 1.0F}, // far (reverse-Z)
            raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = 1.5F, .w = 1.0F}, // near (reverse-Z)
        };

        for (const auto plane_index : std::views::iota(std::size_t{0}, raster::clip_plane_count)) {
            const auto outside = outside_points[plane_index];
            const auto a = make_vertex(
                outside,
                raster::Vec2f{.x = 0.0F, .y = 0.0F},
                raster::Vec3f{.x = 1.0F, .y = 0.0F, .z = 0.0F}
            );
            const auto b = make_vertex(
                outside,
                raster::Vec2f{.x = 1.0F, .y = 0.0F},
                raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
            );
            const auto c = make_vertex(
                outside,
                raster::Vec2f{.x = 0.0F, .y = 1.0F},
                raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F}
            );
            const auto clipped = raster::clip_triangle(shader, a, b, c);
            REQUIRE(clipped.has_value());
            INFO("plane index " << plane_index);
            REQUIRE(clipped->size == 0);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "conservative clip-frustum reject keeps straddling and invalid triangles", "[clipper][cull]"
    ) {
        constexpr std::array<raster::Vec4f, raster::clip_plane_count> outside_points{
            raster::Vec4f{.x = -2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 0.0F, .y = -2.0F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 0.0F, .y = 2.0F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -0.5F, .w = 1.0F},
            raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = 1.5F, .w = 1.0F},
        };
        for (const auto& outside : outside_points) {
            REQUIRE(raster::triangle_outside_clip_frustum({outside, outside, outside}));
        }

        constexpr std::array<raster::Vec4f, 3> straddling_right{
            raster::Vec4f{.x = 0.5F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 1.5F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 0.5F, .y = 0.5F, .z = 0.5F, .w = 1.0F},
        };
        REQUIRE_FALSE(raster::triangle_outside_clip_frustum(straddling_right));
        REQUIRE_FALSE(raster::triangle_outside_clip_frustum(straddling_right, 2.0F));

        constexpr std::array<raster::Vec4f, 3> guard_band_only{
            raster::Vec4f{.x = 1.5F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 1.75F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{.x = 1.5F, .y = 0.5F, .z = 0.5F, .w = 1.0F},
        };
        REQUIRE(raster::triangle_outside_clip_frustum(guard_band_only));
        REQUIRE_FALSE(raster::triangle_outside_clip_frustum(guard_band_only, 2.0F));

        constexpr std::array<raster::Vec4f, 3> non_finite{
            raster::Vec4f{.x = 2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
            raster::Vec4f{
                .x = std::numeric_limits<float>::infinity(),
                .y = 0.0F,
                .z = 0.5F,
                .w = 1.0F,
            },
            raster::Vec4f{.x = 2.0F, .y = 1.0F, .z = 0.5F, .w = 1.0F},
        };
        REQUIRE_FALSE(raster::triangle_outside_clip_frustum(non_finite));
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "clipper intersects a left plane and interpolates varyings", "[clipper][intersection]"
    ) {
        constexpr ClipTestShader shader{};
        const auto outside = make_vertex(
            raster::Vec4f{.x = -2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 0.0F}
        );
        const auto inside_a = make_vertex(
            raster::Vec4f{.x = 0.0F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
            raster::Vec2f{.x = 2.0F, .y = 4.0F},
            raster::Vec3f{.x = 1.0F, .y = 2.0F, .z = 3.0F}
        );
        const auto inside_b = make_vertex(
            raster::Vec4f{.x = 0.5F, .y = 0.5F, .z = 0.75F, .w = 1.0F},
            raster::Vec2f{.x = 4.0F, .y = 8.0F},
            raster::Vec3f{.x = 3.0F, .y = 4.0F, .z = 5.0F}
        );

        const auto clipped = raster::clip_triangle(shader, outside, inside_a, inside_b);
        REQUIRE(clipped.has_value());
        REQUIRE(clipped->size == 2); // triangle clips to a quad => fan of two triangles
        REQUIRE(clipped->active_triangles().size() == 2U);
        for (const auto& triangle : clipped->active_triangles()) {
            for (const auto vertex_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                const auto& vertex = triangle.vertices[vertex_index];
                REQUIRE(inside_all_planes(vertex.clip_pos));
                REQUIRE(vertex.clip_pos.w > 0.0F);
            }
        }

        // Both outside->inside crossings hit x=-w at t=0.5; custom varyings
        // must use that same homogeneous edge parameter.
        bool found_expected_intersection = false;
        for (const auto& triangle : clipped->active_triangles()) {
            for (const auto vertex_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                const auto& vertex = triangle.vertices[vertex_index];
                if (std::fabs(vertex.clip_pos.x + vertex.clip_pos.w)
                    < 1e-6F
                    && std::fabs(vertex.uv.x - 1.0F)
                    < 1e-6F
                    && std::fabs(vertex.uv.y - 2.0F)
                    < 1e-6F
                    && std::fabs(vertex.color.x - 0.5F)
                    < 1e-6F) {
                    found_expected_intersection = true;
                }
            }
        }
        REQUIRE(found_expected_intersection);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("clipper intersects every canonical plane", "[clipper][intersection]") {
        constexpr ClipTestShader shader{};
        constexpr raster::Vec2f uv{};
        constexpr raster::Vec3f color{};

        struct PlaneCrossing {
            raster::Vec4f outside;
            raster::Vec4f inside_a;
            raster::Vec4f inside_b;
            raster::Vec4f expected_boundary;
        };

        const std::array<PlaneCrossing, raster::clip_plane_count> cases{
            PlaneCrossing{
                .outside = {.x = -2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
                .inside_a = {.x = 0.0F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
                .inside_b = {.x = 0.5F, .y = 0.5F, .z = 0.75F, .w = 1.0F},
                .expected_boundary = {.x = -1.0F, .y = -0.25F, .z = 0.5F, .w = 1.0F},
            },
            PlaneCrossing{
                .outside = {.x = 2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
                .inside_a = {.x = 0.0F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
                .inside_b = {.x = -0.5F, .y = 0.5F, .z = 0.75F, .w = 1.0F},
                .expected_boundary = {.x = 1.0F, .y = -0.25F, .z = 0.5F, .w = 1.0F},
            },
            PlaneCrossing{
                .outside = {.x = 0.0F, .y = -2.0F, .z = 0.5F, .w = 1.0F},
                .inside_a = {.x = -0.5F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
                .inside_b = {.x = 0.5F, .y = 0.5F, .z = 0.75F, .w = 1.0F},
                .expected_boundary = {.x = -0.25F, .y = -1.0F, .z = 0.5F, .w = 1.0F},
            },
            PlaneCrossing{
                .outside = {.x = 0.0F, .y = 2.0F, .z = 0.5F, .w = 1.0F},
                .inside_a = {.x = -0.5F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
                .inside_b = {.x = 0.5F, .y = -0.5F, .z = 0.75F, .w = 1.0F},
                .expected_boundary = {.x = -0.25F, .y = 1.0F, .z = 0.5F, .w = 1.0F},
            },
            PlaneCrossing{
                .outside = {.x = 0.0F, .y = 0.0F, .z = -0.5F, .w = 1.0F},
                .inside_a = {.x = -0.5F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
                .inside_b = {.x = 0.5F, .y = 0.5F, .z = 0.75F, .w = 1.0F},
                .expected_boundary = {.x = -0.25F, .y = -0.25F, .z = 0.0F, .w = 1.0F},
            },
            PlaneCrossing{
                .outside = {.x = 0.0F, .y = 0.0F, .z = 1.5F, .w = 1.0F},
                .inside_a = {.x = -0.5F, .y = -0.5F, .z = 0.5F, .w = 1.0F},
                .inside_b = {.x = 0.5F, .y = 0.5F, .z = 0.75F, .w = 1.0F},
                .expected_boundary = {.x = -0.25F, .y = -0.25F, .z = 1.0F, .w = 1.0F},
            },
        };

        for (const auto plane_index : std::views::iota(std::size_t{0}, raster::clip_plane_count)) {
            const auto& crossing = cases[plane_index];
            const auto outside = make_vertex(crossing.outside, uv, color);
            const auto inside_a = make_vertex(crossing.inside_a, uv, color);
            const auto inside_b = make_vertex(crossing.inside_b, uv, color);
            const auto clipped = raster::clip_triangle(shader, outside, inside_a, inside_b);
            REQUIRE(clipped.has_value());
            REQUIRE(clipped->size == 2);
            REQUIRE(clipped->active_triangles().size() == 2U);

            bool found_expected = false;
            for (const auto& triangle : clipped->active_triangles()) {
                for (const auto vertex_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                    const auto& point = triangle.vertices[vertex_index].clip_pos;
                    REQUIRE(inside_all_planes(point));
                    REQUIRE(point.w > 0.0F);
                    if (std::fabs(point.x - crossing.expected_boundary.x)
                        < 1e-6F
                        && std::fabs(point.y - crossing.expected_boundary.y)
                        < 1e-6F
                        && std::fabs(point.z - crossing.expected_boundary.z)
                        < 1e-6F
                        && std::fabs(point.w - crossing.expected_boundary.w)
                        < 1e-6F) {
                        found_expected = true;
                    }
                }
            }
            INFO("plane index " << plane_index);
            REQUIRE(found_expected);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("clipper handles a vertex at w zero without dividing", "[clipper][near-plane]") {
        constexpr ClipTestShader shader{};
        const auto behind = make_vertex(
            raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = -0.25F, .w = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 0.0F}
        );
        const auto visible_a = make_vertex(
            raster::Vec4f{.x = -0.25F, .y = -0.25F, .z = 0.5F, .w = 1.0F},
            raster::Vec2f{.x = 1.0F, .y = 0.0F},
            raster::Vec3f{.x = 1.0F, .y = 0.0F, .z = 0.0F}
        );
        const auto visible_b = make_vertex(
            raster::Vec4f{.x = 0.25F, .y = 0.25F, .z = 0.5F, .w = 1.0F},
            raster::Vec2f{.x = 0.0F, .y = 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto clipped = raster::clip_triangle(shader, behind, visible_a, visible_b);
        REQUIRE(clipped.has_value());
        REQUIRE(clipped->size >= 1);
        REQUIRE(clipped->active_triangles().size() == clipped->size);
        for (const auto& triangle : clipped->active_triangles()) {
            for (const auto vertex_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                const auto& vertex = triangle.vertices[vertex_index];
                REQUIRE(inside_all_planes(vertex.clip_pos));
                REQUIRE(vertex.clip_pos.w > 0.0F);
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("clipper reports non-finite input and rejects common outcodes", "[clipper][errors]") {
        constexpr ClipTestShader shader{};
        constexpr raster::Vec2f empty_uv{};
        constexpr raster::Vec3f empty_value{};
        const auto visible_b = make_vertex(
            raster::Vec4f{.x = 0.5F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
            empty_uv,
            empty_value
        );
        const auto visible_c = make_vertex(
            raster::Vec4f{.x = 0.0F, .y = 0.5F, .z = 0.5F, .w = 1.0F},
            empty_uv,
            empty_value
        );

        const auto non_finite = make_vertex(
            raster::Vec4f{
                .x = std::numeric_limits<float>::infinity(),
                .y = 0.0F,
                .z = 0.0F,
                .w = 1.0F,
            },
            empty_uv,
            empty_value
        );
        const auto rejected = raster::clip_triangle(shader, non_finite, visible_b, visible_c);
        REQUIRE_FALSE(rejected.has_value());
        REQUIRE(rejected.error() == raster::ClipError::NonFiniteVertex);

        const auto outside_a = make_vertex(
            raster::Vec4f{.x = 2.0F, .y = 0.0F, .z = 0.5F, .w = 1.0F},
            empty_uv,
            empty_value
        );
        const auto outside_b = make_vertex(
            raster::Vec4f{.x = 3.0F, .y = 0.25F, .z = 0.5F, .w = 1.0F},
            empty_uv,
            empty_value
        );
        const auto outside_c = make_vertex(
            raster::Vec4f{.x = 4.0F, .y = -0.25F, .z = 0.5F, .w = 1.0F},
            empty_uv,
            empty_value
        );
        const auto trivial_reject = raster::clip_triangle(shader, outside_a, outside_b, outside_c);
        REQUIRE(trivial_reject.has_value());
        REQUIRE(trivial_reject->size == 0);

        // The sole point satisfying all six non-strict inequalities with
        // w=0 is the homogeneous origin, which is not projectable.
        const auto zero_w_origin = make_vertex(
            raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = 0.0F, .w = 0.0F},
            empty_uv,
            empty_value
        );
        const auto non_projectable =
            raster::clip_triangle(shader, zero_w_origin, zero_w_origin, zero_w_origin);
        REQUIRE_FALSE(non_projectable.has_value());
        REQUIRE(non_projectable.error() == raster::ClipError::InvalidSurvivorW);
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
