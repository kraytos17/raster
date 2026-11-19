#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <ranges>
#include <vector>

#include "raster/math/barycentric.hpp"
#include "raster/pipeline/rasterizer.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("fragment callback contract is enforced at compile time", "[rasterizer][concept]") {
        const auto valid_fragment =
            [](std::size_t, std::size_t, const raster::PixelBarycentrics&) noexcept {
        };
        STATIC_REQUIRE(raster::FragmentCallback<decltype(valid_fragment)>);

        // Returning a value is fine; the concept leaves the return type free.
        const auto returning_fragment =
            [](std::size_t,
               std::size_t,
               const raster::PixelBarycentrics&) noexcept -> std::uint32_t {
            return 0U;
        };
        STATIC_REQUIRE(raster::FragmentCallback<decltype(returning_fragment)>);

        // Standard implicit argument conversions are accepted by design.
        const auto converting_fragment = [](int, long long, raster::PixelBarycentrics) noexcept {
        };
        STATIC_REQUIRE(raster::FragmentCallback<decltype(converting_fragment)>);

        // A throwing fragment would terminate the noexcept rasterizer per
        // pixel, so the concept rejects it statically.
        const auto throwing_fragment =
            [](std::size_t, std::size_t, const raster::PixelBarycentrics&) {
        };
        STATIC_REQUIRE_FALSE(raster::FragmentCallback<decltype(throwing_fragment)>);

        // Wrong arity cannot be invoked with pixel coordinates and weights.
        const auto wrong_arity_fragment = [](std::size_t, std::size_t) noexcept {
        };
        STATIC_REQUIRE_FALSE(raster::FragmentCallback<decltype(wrong_arity_fragment)>);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("fixed-point conversion and pixel centers", "[barycentric][fixed]") {
        REQUIRE(raster::to_fixed(0.0F) == 0);
        REQUIRE(raster::to_fixed(0.5F) == 8);
        REQUIRE(raster::to_fixed(-0.5F) == -8);
        REQUIRE(raster::to_fixed(1.5F) == 24);
        REQUIRE(raster::pixel_center(0) == 8);
        REQUIRE(raster::pixel_center(-1) == -8);
        REQUIRE(raster::pixel_center(7) == 120);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("edge function, deltas and barycentrics are exact", "[barycentric][edge]") {
        // Triangle A(1,1) B(9,1) C(1,5) in pixels; all values hand-derived.
        constexpr std::int32_t ax = 16;
        constexpr std::int32_t ay = 16;
        constexpr std::int32_t bx = 144;
        constexpr std::int32_t by = 16;
        constexpr std::int32_t cx = 16;
        constexpr std::int32_t cy = 80;

        constexpr std::int64_t area = raster::edge_function(ax, ay, bx, by, cx, cy);
        REQUIRE(area == 8192);

        // Center of pixel (2, 2).
        constexpr std::int64_t px = 40;
        constexpr std::int64_t py = 40;
        const auto e01 = raster::edge_function(ax, ay, bx, by, px, py);
        const auto e12 = raster::edge_function(bx, by, cx, cy, px, py);
        const auto e20 = raster::edge_function(cx, cy, ax, ay, px, py);
        REQUIRE(e01 == 3072);
        REQUIRE(e12 == 3584);
        REQUIRE(e20 == 1536);
        REQUIRE(e01 + e12 + e20 == area);

        // One-pixel steps change each edge by exactly the analytic deltas.
        const auto step_x_01 = raster::edge_step_x(ay, by);
        const auto step_y_01 = raster::edge_step_y(ax, bx);
        REQUIRE(step_x_01 == 0);
        REQUIRE(step_y_01 == 2048);
        REQUIRE(raster::edge_function(ax, ay, bx, by, px + 16, py) == e01 + step_x_01);
        REQUIRE(raster::edge_function(ax, ay, bx, by, px, py + 16) == e01 + step_y_01);

        const auto step_x_12 = raster::edge_step_x(by, cy);
        REQUIRE(step_x_12 == -1024);
        REQUIRE(raster::edge_function(bx, by, cx, cy, px + 16, py) == e12 + step_x_12);

        // Barycentric weights reproduce the sample position (2.5, 2.5).
        const float inv_area = 1.0F / static_cast<float>(area);
        const auto l0 = static_cast<float>(e12) * inv_area;
        const auto l1 = static_cast<float>(e20) * inv_area;
        const auto l2 = 1.0F - l0 - l1;
        REQUIRE(l0 == 0.4375F);
        REQUIRE(l1 == 0.1875F);
        REQUIRE(l2 == 0.375F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("top-left rule classifies top and left edges", "[barycentric][fill]") {
        // Positive-shoelace (front-facing) edges only.
        REQUIRE(raster::is_top_or_left_edge(0, 0, 16, 0)); // horizontal, interior below
        REQUIRE_FALSE(raster::is_top_or_left_edge(16, 0, 0, 0)); // horizontal, interior above
        REQUIRE_FALSE(raster::is_top_or_left_edge(0, 0, 16, 16)); // down-right diagonal
        REQUIRE(raster::is_top_or_left_edge(16, 16, 0, 0)); // up-left diagonal
        REQUIRE_FALSE(raster::is_top_or_left_edge(0, 0, 0, 16)); // down vertical
        REQUIRE(raster::is_top_or_left_edge(0, 16, 0, 0)); // up vertical
        REQUIRE(raster::edge_bias(0, 0, 16, 0) == 0);
        REQUIRE(raster::edge_bias(0, 0, 16, 16) == -1);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("backfacing and degenerate triangles are culled", "[rasterizer][setup]") {
        const auto backfacing = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 0.0F, .y = 0.0F},
            raster::ScreenPoint{.x = 0.0F, .y = 4.0F},
            raster::ScreenPoint{.x = 4.0F, .y = 0.0F}
        );
        REQUIRE_FALSE(backfacing.has_value());
        REQUIRE(backfacing.error() == raster::SetupError::Backfacing);

        const auto degenerate = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 0.0F, .y = 0.0F},
            raster::ScreenPoint{.x = 2.0F, .y = 0.0F},
            raster::ScreenPoint{.x = 4.0F, .y = 0.0F}
        );
        REQUIRE_FALSE(degenerate.has_value());
        REQUIRE(degenerate.error() == raster::SetupError::Degenerate);
    }

    namespace {
        struct CoverageCounters {
            std::vector<std::uint32_t> counts;
            std::vector<std::uint32_t> owner;
            bool barycentrics_valid = true;
            std::size_t covered_pixels = 0;

            explicit CoverageCounters(std::size_t width, std::size_t height) :
                counts(width * height, 0),
                owner(width * height, 0) {}

            void record(
                std::size_t x,
                std::size_t y,
                std::size_t width,
                std::uint32_t triangle_id,
                const raster::PixelBarycentrics& barycentrics
            ) noexcept {
                const auto index = y * width + x;
                ++counts[index];
                owner[index] = triangle_id;
                if (barycentrics.l0 < 0.0F || barycentrics.l1 < 0.0F || barycentrics.l2 < 0.0F) {
                    barycentrics_valid = false;
                }
                if (barycentrics.l0 + barycentrics.l1 + barycentrics.l2 != 1.0F) {
                    barycentrics_valid = false;
                }
                ++covered_pixels;
            }
        };
    } // namespace

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("100 edge-sharing triangles tile 512x512 watertight", "[rasterizer][watertight]") {
        constexpr std::size_t size = 512;
        constexpr std::size_t columns = 10;
        constexpr std::size_t rows = 5;

        // Shared grid lines are computed once so adjacent squares reuse the
        // identical rounded 28.4 vertex coordinates on their shared edges.
        std::vector<float> grid_x(columns + 1);
        std::vector<float> grid_y(rows + 1);
        for (const auto column : std::views::iota(std::size_t{0}, columns + 1)) {
            grid_x[column] = static_cast<float>(column * size) / static_cast<float>(columns);
        }
        for (const auto row : std::views::iota(std::size_t{0}, rows + 1)) {
            grid_y[row] = static_cast<float>(row * size) / static_cast<float>(rows);
        }

        CoverageCounters counters{size, size};
        for (const auto row : std::views::iota(std::size_t{0}, rows)) {
            for (const auto column : std::views::iota(std::size_t{0}, columns)) {
                const raster::ScreenPoint top_left{.x = grid_x[column], .y = grid_y[row]};
                const raster::ScreenPoint top_right{.x = grid_x[column + 1], .y = grid_y[row]};
                const raster::ScreenPoint bottom_right{
                    .x = grid_x[column + 1],
                    .y = grid_y[row + 1],
                };
                const raster::ScreenPoint bottom_left{.x = grid_x[column], .y = grid_y[row + 1]};

                const auto upper = raster::make_screen_triangle(top_left, top_right, bottom_right);
                const auto lower =
                    raster::make_screen_triangle(top_left, bottom_right, bottom_left);
                REQUIRE(upper.has_value());
                REQUIRE(lower.has_value());

                const auto fragment = [&](std::size_t x,
                                          std::size_t y,
                                          const raster::PixelBarycentrics& barycentrics) noexcept {
                    counters.record(x, y, size, 1, barycentrics);
                };
                raster::rasterize_triangle(*upper, size, size, fragment);
                raster::rasterize_triangle(*lower, size, size, fragment);
            }
        }

        REQUIRE(counters.barycentrics_valid);
        std::size_t mismatched = 0;
        std::size_t first_mismatch = 0;
        for (const auto index : std::views::iota(std::size_t{0}, counters.counts.size())) {
            if (counters.counts[index] != 1) {
                if (mismatched == 0) {
                    first_mismatch = index;
                }
                ++mismatched;
            }
        }
        REQUIRE(counters.covered_pixels == size * size);
        INFO("first mismatching pixel index: " << first_mismatch);
        REQUIRE(mismatched == 0);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("vertical seam center column is drawn exactly once", "[rasterizer][watertight]") {
        // Two triangles sharing the vertical edge x = 0.5 px, which passes
        // exactly through the centers of pixel column x = 0. The right-hand
        // triangle traverses that edge upward (a left edge), so the top-left
        // rule assigns the seam column to it.
        constexpr std::size_t size = 8;
        CoverageCounters counters{size, size};

        const auto left = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 0.0F, .y = 0.0F},
            raster::ScreenPoint{.x = 0.5F, .y = 0.0F},
            raster::ScreenPoint{.x = 0.5F, .y = 4.0F}
        );
        const auto right = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 0.5F, .y = 0.0F},
            raster::ScreenPoint{.x = 4.0F, .y = 2.0F},
            raster::ScreenPoint{.x = 0.5F, .y = 4.0F}
        );
        REQUIRE(left.has_value());
        REQUIRE(right.has_value());

        raster::rasterize_triangle(
            *left,
            size,
            size,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            counters.record(x, y, size, 1, bary);
        }
        );
        raster::rasterize_triangle(
            *right,
            size,
            size,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            counters.record(x, y, size, 2, bary);
        }
        );

        REQUIRE(counters.barycentrics_valid);
        for (const auto index : std::views::iota(std::size_t{0}, counters.counts.size())) {
            REQUIRE(counters.counts[index] <= 1);
        }
        for (const auto y : std::views::iota(std::size_t{0}, std::size_t{4})) {
            REQUIRE(counters.counts[y * size + 0] == 1);
            REQUIRE(counters.owner[y * size + 0] == 2);
        }
        for (const auto y : std::views::iota(std::size_t{4}, size)) {
            REQUIRE(counters.counts[y * size + 0] == 0);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("horizontal seam center row is drawn exactly once", "[rasterizer][watertight]") {
        // Two triangles sharing the horizontal edge y = 0.5 px, which passes
        // exactly through the centers of pixel row y = 0. The lower triangle
        // traverses that edge rightward (a top edge), so it owns the seam row.
        constexpr std::size_t size = 8;
        CoverageCounters counters{size, size};

        const auto upper = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 0.0F, .y = 0.5F},
            raster::ScreenPoint{.x = 2.0F, .y = -2.5F},
            raster::ScreenPoint{.x = 4.0F, .y = 0.5F}
        );
        const auto lower = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 0.0F, .y = 0.5F},
            raster::ScreenPoint{.x = 4.0F, .y = 0.5F},
            raster::ScreenPoint{.x = 2.0F, .y = 3.5F}
        );
        REQUIRE(upper.has_value());
        REQUIRE(lower.has_value());

        raster::rasterize_triangle(
            *upper,
            size,
            size,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            counters.record(x, y, size, 1, bary);
        }
        );
        raster::rasterize_triangle(
            *lower,
            size,
            size,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            counters.record(x, y, size, 2, bary);
        }
        );

        REQUIRE(counters.barycentrics_valid);
        for (const auto index : std::views::iota(std::size_t{0}, counters.counts.size())) {
            REQUIRE(counters.counts[index] <= 1);
        }
        for (const auto x : std::views::iota(std::size_t{0}, std::size_t{4})) {
            REQUIRE(counters.counts[x] == 1);
            REQUIRE(counters.owner[x] == 2);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("triangle wider than 4096 pixels rasterizes without overflow", "[rasterizer][wide]") {
        // Spans 16384 px horizontally; its 28.8 fixed-point area (~3.4e10)
        // exceeds the 32-bit range, which is why all edge math is 64-bit.
        constexpr std::size_t size = 256;
        CoverageCounters counters{size, size};

        const auto wide = raster::make_screen_triangle(
            raster::ScreenPoint{.x = -8192.0F, .y = 100.0F},
            raster::ScreenPoint{.x = 8192.0F, .y = 100.0F},
            raster::ScreenPoint{.x = 0.0F, .y = 8192.0F}
        );
        REQUIRE(wide.has_value());

        raster::rasterize_triangle(
            *wide,
            size,
            size,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            counters.record(x, y, size, 1, bary);
        }
        );

        REQUIRE(counters.barycentrics_valid);
        for (const auto y : std::views::iota(std::size_t{0}, size)) {
            const auto row_begin = static_cast<std::size_t>(y * size);
            const std::uint32_t expected = y >= 100 ? 1U : 0U;
            const auto row_total = counters.counts[row_begin]
                + counters.counts[row_begin + 127]
                + counters.counts[row_begin + 255];
            INFO("row " << y << " sampled total " << row_total << " expected " << expected * 3U);
            REQUIRE(row_total == expected * 3U);
        }
        REQUIRE(counters.counts[100 * size + 0] == 1);
        REQUIRE(counters.counts[255 * size + 255] == 1);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "perspective weights default to screen weights when inv_w is one",
        "[rasterizer][perspective]"
    ) {
        constexpr std::size_t size = 32;
        CoverageCounters counters{size, size};
        bool weights_identical = true;

        const auto triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 10.25F, .y = 10.25F},
            raster::ScreenPoint{.x = 30.25F, .y = 10.25F},
            raster::ScreenPoint{.x = 20.25F, .y = 30.25F},
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.25F, 0.5F, 0.75F}
        );
        REQUIRE(triangle.has_value());
        raster::rasterize_triangle(
            *triangle,
            size,
            size,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            counters.record(x, y, size, 1, bary);
            weights_identical =
                weights_identical && bary.a0 == bary.l0 && bary.a1 == bary.l1 && bary.a2 == bary.l2;
        }
        );

        REQUIRE(counters.barycentrics_valid);
        REQUIRE(counters.covered_pixels > 0);
        REQUIRE(weights_identical);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("perspective weights match the independent derivation", "[rasterizer][perspective]") {
        // The rasterizer computes alpha_i = l_i * inv_w_i * (1 / sum(l_j *
        // inv_w_j)); the geometrically independent form is
        // alpha_i = (l_i * inv_w_i) / sum(l_j * inv_w_j). They must agree.
        constexpr std::size_t size = 32;
        CoverageCounters counters{size, size};
        bool weights_match = true;
        bool weights_sum_to_one = true;

        const auto triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 10.25F, .y = 10.25F},
            raster::ScreenPoint{.x = 30.25F, .y = 10.25F},
            raster::ScreenPoint{.x = 20.25F, .y = 30.25F},
            std::array<float, 3>{1.0F, 0.5F, 2.0F},
            std::array<float, 3>{0.25F, 0.75F, 0.5F}
        );
        REQUIRE(triangle.has_value());
        raster::rasterize_triangle(
            *triangle,
            size,
            size,
            [&](std::size_t x, std::size_t y, const raster::PixelBarycentrics& bary) noexcept {
            counters.record(x, y, size, 1, bary);
            const float w_recip = bary.l0 * 1.0F + bary.l1 * 0.5F + bary.l2 * 2.0F;
            const auto independent_a0 = (bary.l0 * 1.0F) / w_recip;
            const auto independent_a1 = (bary.l1 * 0.5F) / w_recip;
            const auto independent_a2 = (bary.l2 * 2.0F) / w_recip;
            weights_match = weights_match
                && std::fabs(bary.a0 - independent_a0)
                <= 1e-6F
                && std::fabs(bary.a1 - independent_a1)
                <= 1e-6F
                && std::fabs(bary.a2 - independent_a2)
                <= 1e-6F;
            weights_sum_to_one =
                weights_sum_to_one && std::fabs(bary.a0 + bary.a1 + bary.a2 - 1.0F) <= 1e-5F;
        }
        );

        REQUIRE(counters.barycentrics_valid);
        REQUIRE(counters.covered_pixels > 0);
        REQUIRE(weights_match);
        REQUIRE(weights_sum_to_one);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("depth-enabled rasterizer skips fully occluded pixels", "[rasterizer][depth]") {
        constexpr std::size_t size = 32;
        std::vector<float> depth_storage(size * size, 0.0F);
        auto depth = std::mdspan<float, raster::Extents2D>{depth_storage.data(), size, size};
        std::uint32_t far_shaded = 0;
        std::uint32_t near_shaded = 0;
        std::uint32_t occluded_resubmission_shaded = 0;

        const auto far_triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 4.25F, .y = 4.25F},
            raster::ScreenPoint{.x = 20.25F, .y = 4.25F},
            raster::ScreenPoint{.x = 12.25F, .y = 20.25F},
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.25F, 0.25F, 0.25F}
        );
        const auto near_triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 8.25F, .y = 8.25F},
            raster::ScreenPoint{.x = 24.25F, .y = 8.25F},
            raster::ScreenPoint{.x = 16.25F, .y = 24.25F},
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.75F, 0.75F, 0.75F}
        );
        REQUIRE(far_triangle.has_value());
        REQUIRE(near_triangle.has_value());

        raster::rasterize_triangle(
            *far_triangle,
            size,
            size,
            depth,
            [&](std::size_t, std::size_t, const raster::PixelBarycentrics&) noexcept {
            ++far_shaded;
        }
        );
        const auto far_shaded_first = far_shaded;
        REQUIRE(far_shaded_first > 0U);
        REQUIRE(std::ranges::all_of(depth_storage, [](float value) noexcept {
            return value >= 0.0F && value <= 0.25F;
        }));

        raster::rasterize_triangle(
            *near_triangle,
            size,
            size,
            depth,
            [&](std::size_t, std::size_t, const raster::PixelBarycentrics&) noexcept {
            ++near_shaded;
        }
        );
        REQUIRE(near_shaded > 0U);
        REQUIRE(std::ranges::all_of(depth_storage, [](float value) noexcept {
            return value >= 0.0F && value <= 0.75F;
        }));

        // Equal far depth fails strict `z_pixel > depth`; this exact rerender
        // must be rejected at the depth test and never invoke the callback.
        raster::rasterize_triangle(
            *far_triangle,
            size,
            size,
            depth,
            [&](std::size_t, std::size_t, const raster::PixelBarycentrics&) noexcept {
            ++occluded_resubmission_shaded;
        }
        );
        REQUIRE(occluded_resubmission_shaded == 0U);
        REQUIRE(far_shaded == far_shaded_first);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("reverse-z early depth keeps the nearest surface", "[rasterizer][depth]") {
        constexpr std::size_t size = 32;
        std::vector<float> depth_storage(size * size, 0.0F);
        std::vector<std::uint32_t> color_storage(size * size, 0U);
        std::vector<std::uint32_t> far_first_count(size * size, 0U);
        std::vector<std::uint32_t> near_count(size * size, 0U);
        std::vector<std::uint32_t> far_second_count(size * size, 0U);
        auto depth = std::mdspan<float, raster::Extents2D>{depth_storage.data(), size, size};

        const auto far_triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 4.25F, .y = 4.25F},
            raster::ScreenPoint{.x = 20.25F, .y = 4.25F},
            raster::ScreenPoint{.x = 12.25F, .y = 20.25F},
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.25F, 0.25F, 0.25F}
        );
        const auto near_triangle = raster::make_screen_triangle(
            raster::ScreenPoint{.x = 8.25F, .y = 8.25F},
            raster::ScreenPoint{.x = 24.25F, .y = 8.25F},
            raster::ScreenPoint{.x = 16.25F, .y = 24.25F},
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.75F, 0.75F, 0.75F}
        );
        REQUIRE(far_triangle.has_value());
        REQUIRE(near_triangle.has_value());

        const auto color_fragment = [&](std::uint32_t id) noexcept {
            return
                [&, id](std::size_t x, std::size_t y, const raster::PixelBarycentrics&) noexcept {
                color_storage[y * size + x] = id;
            };
        };
        raster::rasterize_triangle(*far_triangle, size, size, depth, color_fragment(1U));
        for (const auto index : std::views::iota(std::size_t{0}, color_storage.size())) {
            if (color_storage[index] == 1U) {
                far_first_count[index] = 1U;
            }
        }
        raster::rasterize_triangle(*near_triangle, size, size, depth, color_fragment(2U));
        for (const auto index : std::views::iota(std::size_t{0}, color_storage.size())) {
            if (color_storage[index] == 2U) {
                near_count[index] = 1U;
            }
        }
        // Re-drawing the far triangle must be rejected everywhere by the
        // strictly-nearer reverse-Z test without invoking the fragment even
        // once — the quad-level early-Z path.
        raster::rasterize_triangle(*far_triangle, size, size, depth, color_fragment(3U));
        for (const auto index : std::views::iota(std::size_t{0}, color_storage.size())) {
            if (color_storage[index] == 3U) {
                far_second_count[index] = 1U;
            }
        }

        bool overlap_exists = false;
        bool near_depths_correct = true;
        bool far_depths_correct = true;
        for (const auto index : std::views::iota(std::size_t{0}, color_storage.size())) {
            overlap_exists =
                overlap_exists || (far_first_count[index] == 1U && near_count[index] == 1U);
            if (near_count[index] == 1U) {
                near_depths_correct = near_depths_correct
                    && depth_storage[index]
                    > 0.749999F
                    && depth_storage[index]
                    < 0.750001F;
            } else if (far_first_count[index] == 1U) {
                far_depths_correct = far_depths_correct
                    && depth_storage[index]
                    > 0.249999F
                    && depth_storage[index]
                    < 0.250001F;
            }
        }
        REQUIRE(overlap_exists);
        REQUIRE(near_depths_correct);
        REQUIRE(far_depths_correct);
        REQUIRE(std::ranges::all_of(far_second_count, [](std::uint32_t count) noexcept {
            return count == 0U;
        }));
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
