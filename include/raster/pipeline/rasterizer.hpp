#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <ranges>
#include <type_traits>

#include "raster/math/barycentric.hpp"

namespace raster {

    // Why a triangle was rejected during screen-space setup.
    enum class SetupError {
        Backfacing,
        Degenerate,
    };

    // A screen-space point in pixel coordinates.
    struct ScreenPoint {
        float x{};
        float y{};
    };

    // Per-pixel interpolation weights: l0/l1/l2 belong to vertices A/B/C and
    // are computed from the *unbiased* edge values so the top-left fill bias
    // never distorts shading.
    struct PixelBarycentrics {
        float l0{};
        float l1{};
        float l2{};
    };

    // Contract for the per-pixel fragment callback handed to
    // rasterize_triangle: invocable as an lvalue (the rasterizer always calls
    // its local copy) with the covered pixel coordinates and the unbiased
    // barycentric weights, and noexcept — rasterize_triangle is noexcept per
    // the zero-exceptions mandate, so a throwing fragment would otherwise call
    // std::terminate() per pixel. The return type is deliberately free, and
    // standard implicit argument conversions remain allowed (a fragment taking
    // narrower integer types or PixelBarycentrics by value still satisfies
    // this concept).
    template <typename Fragment>
    concept FragmentCallback =
        std::invocable<Fragment&, std::size_t, std::size_t, const PixelBarycentrics&>
        && std::
            is_nothrow_invocable_v<Fragment&, std::size_t, std::size_t, const PixelBarycentrics&>;

    // Pre-evaluated half-space setup for one front-facing triangle. All edge
    // accumulators are 28.8 fixed-point 64-bit values; stepping a full pixel
    // costs exactly one integer addition per edge.
    struct ScreenTriangle {
        std::array<std::int64_t, 3> row_start{}; // edge values at (min_x, min_y) center
        std::array<std::int64_t, 3> step_x{}; // delta for one pixel right
        std::array<std::int64_t, 3> step_y{}; // delta for one pixel down
        std::array<std::int64_t, 3> bias{}; // top-left fill bias (0 or -1)
        float inv_area{};
        std::int32_t min_x{}; // bounding box, inclusive, pixels
        std::int32_t min_y{};
        std::int32_t max_x{}; // bounding box, exclusive, pixels
        std::int32_t max_y{};
    };

    namespace detail {

        // Floor division that stays correct for negative dividends (a plain
        // shift or truncated division rounds toward zero).
        [[nodiscard]]
        constexpr std::int32_t floor_div(std::int32_t value, std::int32_t divisor) noexcept {
            const std::int32_t quotient = value / divisor;
            const std::int32_t remainder = value % divisor;
            if (remainder != 0 && (remainder < 0) != (divisor < 0)) {
                return quotient - 1;
            }
            return quotient;
        }

        // Floor-division of a 28.4 sub-pixel coordinate to its containing
        // pixel index: (X - center_offset) / 16 for any sign of X.
        [[nodiscard]]
        constexpr std::int32_t pixel_floor(std::int32_t fixed_x) noexcept {
            return floor_div(fixed_x - subpixel_center_offset, subpixel_scale);
        }

        [[nodiscard]]
        constexpr bool covers_pixel(
            const std::array<std::int64_t, 3>& edge,
            const std::array<std::int64_t, 3>& bias
        ) noexcept {
            return (edge[0] + bias[0]) >= 0 && (edge[1] + bias[1]) >= 0 && (edge[2] + bias[2]) >= 0;
        }
    } // namespace detail

    // Builds the screen-space setup for triangle A -> B -> C. Returns
    // SetupError::Backfacing for negative twice-signed area (plan: cull when
    // area_fx <= 0) and Degenerate for zero area.
    [[nodiscard]]
    inline std::expected<ScreenTriangle, SetupError> make_screen_triangle(
        ScreenPoint a,
        ScreenPoint b,
        ScreenPoint c
    ) noexcept {
        const std::int32_t ax = to_fixed(a.x);
        const std::int32_t ay = to_fixed(a.y);
        const std::int32_t bx = to_fixed(b.x);
        const std::int32_t by = to_fixed(b.y);
        const std::int32_t cx = to_fixed(c.x);
        const std::int32_t cy = to_fixed(c.y);
        const std::int64_t area = edge_function(ax, ay, bx, by, cx, cy);
        if (area == 0) {
            return std::unexpected{SetupError::Degenerate};
        }
        if (area < 0) {
            return std::unexpected{SetupError::Backfacing};
        }

        ScreenTriangle triangle{};
        triangle.inv_area = 1.0F / static_cast<float>(area);
        triangle.bias = {
            edge_bias(ax, ay, bx, by),
            edge_bias(bx, by, cx, cy),
            edge_bias(cx, cy, ax, ay),
        };
        triangle.step_x = {
            edge_step_x(ay, by),
            edge_step_x(by, cy),
            edge_step_x(cy, ay),
        };
        triangle.step_y = {
            edge_step_y(ax, bx),
            edge_step_y(bx, cx),
            edge_step_y(cx, ax),
        };

        const std::int32_t min_fixed_x = std::min({ax, bx, cx});
        const std::int32_t min_fixed_y = std::min({ay, by, cy});
        const std::int32_t max_fixed_x = std::max({ax, bx, cx});
        const std::int32_t max_fixed_y = std::max({ay, by, cy});

        triangle.min_x = detail::pixel_floor(min_fixed_x);
        triangle.min_y = detail::pixel_floor(min_fixed_y);
        triangle.max_x = detail::pixel_floor(max_fixed_x) + 1;
        triangle.max_y = detail::pixel_floor(max_fixed_y) + 1;

        const std::int64_t origin_x = pixel_center(triangle.min_x);
        const std::int64_t origin_y = pixel_center(triangle.min_y);
        triangle.row_start = {
            edge_function(ax, ay, bx, by, origin_x, origin_y),
            edge_function(bx, by, cx, cy, origin_x, origin_y),
            edge_function(cx, cy, ax, ay, origin_x, origin_y),
        };
        return triangle;
    }

    // Rasterizes one front-facing triangle over the clamped intersection of
    // its bounding box and the viewport, walking aligned 2x2 pixel quads.
    // Coverage uses the biased edge values; the fragment callback receives
    // unbiased barycentrics.
    template <FragmentCallback Fragment>
    void rasterize_triangle(
        const ScreenTriangle& triangle,
        std::size_t viewport_width,
        std::size_t viewport_height,
        Fragment&& fragment
    ) noexcept {
        const auto viewport_max_x = static_cast<std::int32_t>(viewport_width);
        const auto viewport_max_y = static_cast<std::int32_t>(viewport_height);
        const std::int32_t x_begin = std::max(triangle.min_x, static_cast<std::int32_t>(0));
        const std::int32_t y_begin = std::max(triangle.min_y, static_cast<std::int32_t>(0));
        const std::int32_t x_end = std::min(triangle.max_x, viewport_max_x);
        const std::int32_t y_end = std::min(triangle.max_y, viewport_max_y);
        if (x_begin >= x_end || y_begin >= y_end) {
            return;
        }

        Fragment fragment_callback = std::forward<Fragment>(fragment);
        // Advance the row-start accumulators from the unclamped box origin to
        // the clamped one without re-evaluating the edge functions.
        const std::int64_t column_offset = static_cast<std::int64_t>(x_begin - triangle.min_x);
        const std::int64_t row_offset = static_cast<std::int64_t>(y_begin - triangle.min_y);
        std::array<std::int64_t, 3> row_accumulator{};
        for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
            row_accumulator[edge_index] = triangle.row_start[edge_index]
                + triangle.step_x[edge_index]
                * column_offset
                + triangle.step_y[edge_index]
                * row_offset;
        }
        for (std::int32_t quad_y = y_begin; quad_y < y_end; quad_y += 2) {
            const std::int32_t quad_height = std::min(static_cast<std::int32_t>(2), y_end - quad_y);
            std::array<std::int64_t, 3> quad_accumulator = row_accumulator;
            for (std::int32_t quad_x = x_begin; quad_x < x_end; quad_x += 2) {
                const std::int32_t quad_width =
                    std::min(static_cast<std::int32_t>(2), x_end - quad_x);
                std::uint32_t coverage_mask = 0;
                std::array<std::array<std::int64_t, 3>, 4> corner_edges{};
                corner_edges[0] = quad_accumulator;
                for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                    corner_edges[1][edge_index] =
                        quad_accumulator[edge_index] + triangle.step_x[edge_index];
                    corner_edges[2][edge_index] =
                        quad_accumulator[edge_index] + triangle.step_y[edge_index];
                    corner_edges[3][edge_index] =
                        corner_edges[1][edge_index] + triangle.step_y[edge_index];
                }
                if (detail::covers_pixel(corner_edges[0], triangle.bias)) {
                    coverage_mask |= 0x1U;
                }
                if (quad_width == 2 && detail::covers_pixel(corner_edges[1], triangle.bias)) {
                    coverage_mask |= 0x2U;
                }
                if (quad_height == 2 && detail::covers_pixel(corner_edges[2], triangle.bias)) {
                    coverage_mask |= 0x4U;
                }
                if (quad_width
                    == 2
                    && quad_height
                    == 2
                    && detail::covers_pixel(corner_edges[3], triangle.bias)) {
                    coverage_mask |= 0x8U;
                }
                if (coverage_mask != 0) {
                    const auto base_x = static_cast<std::size_t>(quad_x);
                    const auto base_y = static_cast<std::size_t>(quad_y);
                    for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                        if ((coverage_mask & (1U << corner)) == 0) {
                            continue;
                        }

                        const auto corner_x = base_x + static_cast<std::size_t>(corner & 1U);
                        const auto corner_y = base_y + static_cast<std::size_t>(corner >> 1U);
                        // Edge 0 is A->B and carries the weight of vertex C,
                        // edge 1 (B->C) carries vertex A, edge 2 (C->A) carries
                        // vertex B. The third weight is a single subtraction from
                        // the summed pair, which makes l0 + l1 + l2 == 1.0F exactly
                        // under round-to-nearest (Sterbenz-exact for s >= 0.5, < half-ulp error below)
                        // and keeps every weight non-negative because the
                        // float pair-sum cannot exceed 1 for interior pixels.
                        const auto l0 =
                            static_cast<float>(corner_edges[corner][1]) * triangle.inv_area;
                        const auto l1 =
                            static_cast<float>(corner_edges[corner][2]) * triangle.inv_area;
                        const auto l0_plus_l1 = l0 + l1;
                        const PixelBarycentrics barycentrics{
                            .l0 = l0,
                            .l1 = l1,
                            .l2 = 1.0F - l0_plus_l1,
                        };
                        fragment_callback(corner_x, corner_y, barycentrics);
                    }
                }
                for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                    quad_accumulator[edge_index] += 2 * triangle.step_x[edge_index];
                }
            }
            for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                row_accumulator[edge_index] += 2 * triangle.step_y[edge_index];
            }
        }
    }
} // namespace raster
