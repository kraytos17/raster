#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>

#include "raster/math/barycentric.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/pipeline/simd_quad.hpp"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

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

    // Per-pixel interpolation weights. l0/l1/l2 are the screen-space
    // barycentrics computed from the *unbiased* edge values (they sum to
    // 1.0F exactly and drive screen-linear quantities such as reverse-Z
    // depth). a0/a1/a2 are the perspective-correct weights built from the
    // per-vertex cached inv_w; when all three inv_w values equal 1 they are
    // bitwise identical to the l values.
    struct PixelBarycentrics {
        float l0{};
        float l1{};
        float l2{};
        float a0{};
        float a1{};
        float a2{};
    };

    struct QuadBarycentrics {
        std::array<PixelBarycentrics, 4> lanes{};
        std::uint32_t coverage_mask{};
        std::uint32_t visible_mask{};
    };

    struct RasterRegion {
        std::size_t min_x{};
        std::size_t min_y{};
        std::size_t max_x{static_cast<std::size_t>(-1)};
        std::size_t max_y{static_cast<std::size_t>(-1)};
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
    template <typename Fragment> concept FragmentCallback = requires(
        Fragment& fragment, std::size_t x, std::size_t y, const PixelBarycentrics& barycentrics
    ) {
        { std::invoke(fragment, x, y, barycentrics) } noexcept;
    };

    template <typename Fragment> concept QuadFragmentCallback = requires(
        Fragment& fragment, std::size_t x, std::size_t y, const QuadBarycentrics& barycentrics
    ) {
        { std::invoke(fragment, x, y, barycentrics) } noexcept;
    };

    // Pre-evaluated half-space setup for one front-facing triangle. All edge
    // accumulators are 28.8 fixed-point 64-bit values; stepping a full pixel
    // costs exactly one integer addition per edge. inv_w holds the cached
    // 1/w_c per vertex and z_ndc the reverse-Z NDC depth per vertex;
    // defaults of 1/0 reproduce the plain screen-space behavior.
    struct ScreenTriangle {
        std::array<std::int64_t, 3> row_start{}; // edge values at (min_x, min_y) center
        std::array<std::int64_t, 3> step_x{}; // delta for one pixel right
        std::array<std::int64_t, 3> step_y{}; // delta for one pixel down
        std::array<std::int64_t, 3> bias{}; // top-left fill bias (0 or +1)
        std::array<float, 3> inv_w{1.0F, 1.0F, 1.0F};
        std::array<float, 3> z_ndc{};
        float inv_area{};
        std::int32_t min_x{}; // bounding box, inclusive, pixels
        std::int32_t min_y{};
        std::int32_t max_x{}; // bounding box, exclusive, pixels
        std::int32_t max_y{};
    };

    namespace detail {
        // Converts three screen points to 28.4 fixed coordinates in one pass;
        // out is {ax, ay, bx, by, cx, cy}. The packed path multiplies by the
        // subpixel scale and rounds with the MXCSR mode, exactly matching six
        // to_fixed calls lane for lane (out-of-range and NaN inputs convert
        // to INT32_MIN on both paths, and the scale is an exact power of two).
        inline void to_fixed_triangle(
            const std::array<ScreenPoint, 3>& points, std::array<std::int32_t, 6>& out
        ) noexcept {
#if defined(__x86_64__)
            const __m128 scale = _mm_set1_ps(static_cast<float>(subpixel_scale));
            const __m128 xy01 = _mm_mul_ps(
                // NOLINTNEXTLINE
                _mm_loadu_ps(reinterpret_cast<const float*>(points.data())),
                scale
            );
            const __m128 xy12 = _mm_mul_ps(
                // NOLINTNEXTLINE
                _mm_loadu_ps(reinterpret_cast<const float*>(&points[1])),
                scale
            );
            // The stores overlap on bx/by with identical values: lanes 2-3 of
            // the first register and lanes 0-1 of the second convert the same
            // scaled inputs.
            _mm_storeu_si128(
                // NOLINTNEXTLINE
                reinterpret_cast<__m128i*>(out.data()),
                _mm_cvtps_epi32(xy01)
            );
            _mm_storeu_si128(
                // NOLINTNEXTLINE
                reinterpret_cast<__m128i*>(&out[2]),
                _mm_cvtps_epi32(xy12)
            );
#else
            out = {
                to_fixed(points[0].x),
                to_fixed(points[0].y),
                to_fixed(points[1].x),
                to_fixed(points[1].y),
                to_fixed(points[2].x),
                to_fixed(points[2].y),
            };
#endif
        }

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

        // A pixel center is covered when all biased edge values are on the
        // interior side. Interior edge values are <= 0 for front-facing
        // triangles (negative twice-signed area), so the test is <= 0.
        [[nodiscard]]
        constexpr bool covers_pixel(
            const std::array<std::int64_t, 3>& edge, const std::array<std::int64_t, 3>& bias
        ) noexcept {
            return (edge[0] + bias[0]) <= 0 && (edge[1] + bias[1]) <= 0 && (edge[2] + bias[2]) <= 0;
        }

        // Conservative trivial-coverage test for a half-open pixel region
        // [x_begin, x_end) x [y_begin, y_end): true only if every pixel
        // center in the region passes the biased edge tests. Linearity of
        // the edge functions means the four extreme centers decide it.
        [[nodiscard]]
        inline bool region_is_trivially_covered(
            const ScreenTriangle& triangle,
            std::int32_t x_begin,
            std::int32_t y_begin,
            std::int32_t x_end,
            std::int32_t y_end
        ) noexcept {
            if (x_begin >= x_end || y_begin >= y_end) {
                return false;
            }

            const std::array<std::int64_t, 4> corner_dx{
                static_cast<std::int64_t>(x_begin - triangle.min_x),
                static_cast<std::int64_t>(x_end - 1 - triangle.min_x),
                static_cast<std::int64_t>(x_begin - triangle.min_x),
                static_cast<std::int64_t>(x_end - 1 - triangle.min_x),
            };
            const std::array<std::int64_t, 4> corner_dy{
                static_cast<std::int64_t>(y_begin - triangle.min_y),
                static_cast<std::int64_t>(y_begin - triangle.min_y),
                static_cast<std::int64_t>(y_end - 1 - triangle.min_y),
                static_cast<std::int64_t>(y_end - 1 - triangle.min_y),
            };
            for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                    const auto value = triangle.row_start[edge_index]
                        + triangle.step_x[edge_index]
                        * corner_dx[corner]
                        + triangle.step_y[edge_index]
                        * corner_dy[corner];
                    if (value + triangle.bias[edge_index] > 0) {
                        return false;
                    }
                }
            }
            return true;
        }
    } // namespace detail

    // Builds the screen-space setup for triangle A -> B -> C from post-viewport
    // pixel coordinates plus per-vertex cached inv_w (1/w_c) and reverse-Z NDC
    // depth. Front-facing triangles are counter-clockwise as seen in NDC
    // (y-up), which is negative shoelace area in y-down pixels. Returns
    // SetupError::Backfacing for positive twice-signed area and Degenerate
    // for zero area.
    [[nodiscard]]
    inline std::expected<ScreenTriangle, SetupError> make_screen_triangle(
        const std::array<ScreenPoint, 3>& points,
        const std::array<float, 3>& inv_w,
        const std::array<float, 3>& z_ndc
    ) noexcept {
        std::array<std::int32_t, 6> fixed{};
        detail::to_fixed_triangle(points, fixed);

        const std::int32_t ax = fixed[0];
        const std::int32_t ay = fixed[1];
        const std::int32_t bx = fixed[2];
        const std::int32_t by = fixed[3];
        const std::int32_t cx = fixed[4];
        const std::int32_t cy = fixed[5];
        const std::int64_t area = edge_function(ax, ay, bx, by, cx, cy);
        if (area == 0) {
            return std::unexpected{SetupError::Degenerate};
        }
        if (area > 0) {
            return std::unexpected{SetupError::Backfacing};
        }

        ScreenTriangle triangle{};
        triangle.inv_w = inv_w;
        triangle.z_ndc = z_ndc;
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

    // Screen-space triangle from three points: packs into the array entry.
    [[nodiscard]]
    inline std::expected<ScreenTriangle, SetupError> make_screen_triangle(
        ScreenPoint a,
        ScreenPoint b,
        ScreenPoint c,
        const std::array<float, 3>& inv_w,
        const std::array<float, 3>& z_ndc
    ) noexcept {
        return make_screen_triangle({a, b, c}, inv_w, z_ndc);
    }

    // Plain screen-space triangle (screen-linear depth of 0, inv_w of 1).
    [[nodiscard]]
    inline std::expected<ScreenTriangle, SetupError> make_screen_triangle(
        ScreenPoint a, ScreenPoint b, ScreenPoint c
    ) noexcept {
        return make_screen_triangle(
            a,
            b,
            c,
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{0.0F, 0.0F, 0.0F}
        );
    }

    // Maps NDC (-1..1, +Y up) to top-down pixel coordinates.
    [[nodiscard]]
    inline ScreenPoint viewport_to_screen(
        float ndc_x, float ndc_y, std::size_t viewport_width, std::size_t viewport_height
    ) noexcept {
        const auto width = static_cast<float>(viewport_width);
        const auto height = static_cast<float>(viewport_height);
        return ScreenPoint{.x = (ndc_x + 1.0F) * 0.5F * width, .y = (1.0F - ndc_y) * 0.5F * height};
    }

    // Rasterizes one front-facing triangle over the clamped intersection of
    // its bounding box and the viewport, walking aligned 2x2 pixel quads.
    // Coverage uses the biased edge values; the fragment callback receives
    // unbiased screen-space weights plus perspective-correct weights.
    template <FragmentCallback Fragment> void rasterize_triangle(
        const ScreenTriangle& triangle,
        std::size_t viewport_width,
        std::size_t viewport_height,
        Fragment&& fragment
    ) noexcept {
        rasterize_triangle_quads(
            triangle,
            viewport_width,
            viewport_height,
            std::nullopt,
            [fragment_callback = std::forward<Fragment>(fragment)](
                std::size_t quad_x,
                std::size_t quad_y,
                const QuadBarycentrics& quad
            ) mutable noexcept {
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                if ((quad.visible_mask & (1U << lane)) == 0U) {
                    continue;
                }
                fragment_callback(quad_x + (lane & 1U), quad_y + (lane >> 1U), quad.lanes[lane]);
            }
        }
        );
    }

    // Depth-enabled variant: passes a reverse-Z depth view.
    // Screen-linear reverse-Z depth is interpolated per covered corner, tested
    // against the depth buffer (nearer = larger) to build a quad-level
    // visibility mask — a fully occluded quad skips perspective recovery and
    // fragment shading entirely — and passing pixels write depth before
    // shading.
    template <FragmentCallback Fragment> void rasterize_triangle(
        const ScreenTriangle& triangle,
        std::size_t viewport_width,
        std::size_t viewport_height,
        std::optional<std::mdspan<float, Extents2D>> depth_view,
        Fragment&& fragment,
        const RasterRegion& region = {}
    ) noexcept {
        rasterize_triangle_quads(
            triangle,
            viewport_width,
            viewport_height,
            depth_view,
            [fragment_callback = std::forward<Fragment>(fragment)](
                std::size_t quad_x,
                std::size_t quad_y,
                const QuadBarycentrics& quad
            ) mutable noexcept {
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                if ((quad.visible_mask & (1U << lane)) == 0U) {
                    continue;
                }
                fragment_callback(quad_x + (lane & 1U), quad_y + (lane >> 1U), quad.lanes[lane]);
            }
        },
            region
        );
    }

    template <QuadFragmentCallback Fragment> void rasterize_triangle_quads(
        const ScreenTriangle& triangle,
        std::size_t viewport_width,
        std::size_t viewport_height,
        Fragment&& fragment
    ) noexcept {
        rasterize_triangle_quads(
            triangle,
            viewport_width,
            viewport_height,
            std::nullopt,
            std::forward<Fragment>(fragment),
            {}
        );
    }

    template <QuadFragmentCallback Fragment> void rasterize_triangle_quads(
        const ScreenTriangle& triangle,
        std::size_t viewport_width,
        std::size_t viewport_height,
        std::optional<std::mdspan<float, Extents2D>> depth_view,
        Fragment&& fragment,
        const RasterRegion& region = {}
    ) noexcept {
        const auto viewport_max_x = static_cast<std::int32_t>(viewport_width);
        const auto viewport_max_y = static_cast<std::int32_t>(viewport_height);
        const std::int32_t x_begin = std::max({
            triangle.min_x,
            static_cast<std::int32_t>(0),
            static_cast<std::int32_t>(std::min(region.min_x, viewport_width)),
        });
        const std::int32_t y_begin = std::max({
            triangle.min_y,
            static_cast<std::int32_t>(0),
            static_cast<std::int32_t>(std::min(region.min_y, viewport_height)),
        });
        const std::int32_t x_end = std::min({
            triangle.max_x,
            viewport_max_x,
            static_cast<std::int32_t>(std::min(region.max_x, viewport_width)),
        });
        const std::int32_t y_end = std::min({
            triangle.max_y,
            viewport_max_y,
            static_cast<std::int32_t>(std::min(region.max_y, viewport_height)),
        });
        if (x_begin >= x_end || y_begin >= y_end) {
            return;
        }

        Fragment fragment_callback = std::forward<Fragment>(fragment);
        // Trivially covered regions skip per-quad biased edge tests: every
        // pixel center in [x_begin, x_end) x [y_begin, y_end) is covered, so
        // full quads take coverage 0xF and edge quads mask to region bounds.
        // Corner edge values are still evaluated for barycentrics/depth.
        const bool trivially_covered =
            detail::region_is_trivially_covered(triangle, x_begin, y_begin, x_end, y_end);
        // Align quads to the framebuffer's even pixel grid so a triangle can be
        // split across tile boundaries without changing helper-lane derivatives.
        const auto quad_x_begin = detail::floor_div(x_begin, 2) * 2;
        const auto quad_y_begin = detail::floor_div(y_begin, 2) * 2;
        // Advance the row-start accumulators from the unclamped box origin to
        // the first globally aligned quad without re-evaluating edge functions.
        const std::int64_t column_offset = static_cast<std::int64_t>(quad_x_begin - triangle.min_x);
        const std::int64_t row_offset = static_cast<std::int64_t>(quad_y_begin - triangle.min_y);
        std::array<std::int64_t, 3> row_accumulator{};
        for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
            row_accumulator[edge_index] = triangle.row_start[edge_index]
                + triangle.step_x[edge_index]
                * column_offset
                + triangle.step_y[edge_index]
                * row_offset;
        }
        for (std::int32_t quad_y = quad_y_begin; quad_y < y_end; quad_y += 2) {
            const std::int32_t quad_height = std::min(static_cast<std::int32_t>(2), y_end - quad_y);
            std::array<std::int64_t, 3> quad_accumulator = row_accumulator;
            // Shades one covered quad: reverse-Z depth test, perspective
            // recovery of the helper-lane barycentrics, and the fragment
            // callback. Shared by the scalar path below and the wide
            // quad-pair path, so both produce identical pixels.
            const auto shade_covered_quad =
                [&](std::int32_t quad_origin_x,
                    const std::array<std::array<float, 3>, 4>& corner_weights,
                    std::uint32_t coverage_mask) noexcept {
                std::uint32_t visible_mask = coverage_mask;
                if (depth_view.has_value()) {
                    // Quad-level reverse-Z early depth rejection.
                    for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                        if ((coverage_mask & (1U << corner)) == 0) {
                            continue;
                        }

                        const auto corner_x = static_cast<std::size_t>(quad_origin_x)
                            + static_cast<std::size_t>(corner & 1U);
                        const auto corner_y = static_cast<std::size_t>(quad_y)
                            + static_cast<std::size_t>(corner >> 1U);
                        const float z_pixel = corner_weights[corner][0]
                            * triangle.z_ndc[0]
                            + corner_weights[corner][1]
                            * triangle.z_ndc[1]
                            + corner_weights[corner][2]
                            * triangle.z_ndc[2];
                        if (z_pixel > (*depth_view)[corner_y, corner_x]) {
                            (*depth_view)[corner_y, corner_x] = z_pixel;
                        } else {
                            visible_mask &= static_cast<std::uint32_t>(~(1U << corner));
                        }
                    }
                    if (visible_mask == 0) {
                        return;
                    }
                }

                QuadBarycentrics quad_barycentrics{};
                quad_barycentrics.coverage_mask = coverage_mask;
                quad_barycentrics.visible_mask = visible_mask;
                const auto fallback_lane = static_cast<std::size_t>(std::countr_zero(visible_mask));
                std::array<bool, 4> valid_lane{};
                for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    const auto l0 = corner_weights[corner][0];
                    const auto l1 = corner_weights[corner][1];
                    const auto l2 = corner_weights[corner][2];
                    // Perspective correction from the cached per-vertex inv_w
                    // w_pixel = 1 / sum(l_i * inv_w_i),
                    // then alpha_i = l_i * inv_w_i * w_pixel. With all inv_w
                    // equal to 1 the pair-sum is exactly 1.0F, w_pixel is
                    // exactly 1.0F, and alpha_i is bitwise identical to l_i.
                    const float w_recip =
                        l0 * triangle.inv_w[0] + l1 * triangle.inv_w[1] + l2 * triangle.inv_w[2];
                    if (std::isfinite(w_recip) && w_recip > std::numeric_limits<float>::min()) {
                        const float w_pixel = 1.0F / w_recip;
                        quad_barycentrics.lanes[corner] = PixelBarycentrics{
                            .l0 = l0,
                            .l1 = l1,
                            .l2 = l2,
                            .a0 = l0 * triangle.inv_w[0] * w_pixel,
                            .a1 = l1 * triangle.inv_w[1] * w_pixel,
                            .a2 = l2 * triangle.inv_w[2] * w_pixel,
                        };
                        valid_lane[corner] = std::isfinite(w_pixel);
                    }
                }
                for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    if (!valid_lane[corner]) {
                        quad_barycentrics.lanes[corner] = quad_barycentrics.lanes[fallback_lane];
                    }
                }
                fragment_callback(
                    static_cast<std::size_t>(quad_origin_x),
                    static_cast<std::size_t>(quad_y),
                    quad_barycentrics
                );
            };

            const bool use_wide_pairs = simd::cpu_has_avx2();
            for (std::int32_t quad_x = quad_x_begin; quad_x < x_end;) {
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
                if (trivially_covered ? quad_x >= x_begin
                                      : detail::covers_pixel(corner_edges[0], triangle.bias)) {
                    coverage_mask |= 0x1U;
                }
                if (quad_width
                    == 2
                    && (trivially_covered ? quad_x + 1 >= x_begin
                                          : detail::covers_pixel(corner_edges[1], triangle.bias))) {
                    coverage_mask |= 0x2U;
                }
                if (quad_height
                    == 2
                    && (trivially_covered ? quad_y >= y_begin
                                          : detail::covers_pixel(corner_edges[2], triangle.bias))) {
                    coverage_mask |= 0x4U;
                }
                if (quad_width
                    == 2
                    && quad_height
                    == 2
                    && (trivially_covered ? quad_x + 1 >= x_begin && quad_y + 1 >= y_begin
                                          : detail::covers_pixel(corner_edges[3], triangle.bias))) {
                    coverage_mask |= 0x8U;
                }
                if (use_wide_pairs
                    && quad_height
                    == 2
                    && quad_width
                    == 2
                    && x_end
                    - quad_x
                    >= 4
                    && quad_x
                    >= x_begin
                    && quad_y
                    >= y_begin) {
                    std::array<std::array<std::int64_t, 3>, 4> corner_edges_b{};
                    for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                        corner_edges_b[0][edge_index] =
                            corner_edges[0][edge_index] + 2 * triangle.step_x[edge_index];
                        corner_edges_b[1][edge_index] =
                            corner_edges_b[0][edge_index] + triangle.step_x[edge_index];
                        corner_edges_b[2][edge_index] =
                            corner_edges_b[0][edge_index] + triangle.step_y[edge_index];
                        corner_edges_b[3][edge_index] =
                            corner_edges_b[1][edge_index] + triangle.step_y[edge_index];
                    }

                    std::uint32_t coverage_mask_b = 0;
                    if (trivially_covered) {
                        coverage_mask_b = 0xFU;
                    } else {
                        for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                            if (detail::covers_pixel(corner_edges_b[corner], triangle.bias)) {
                                coverage_mask_b |= static_cast<std::uint32_t>(1U << corner);
                            }
                        }
                    }

                    if (coverage_mask == 0 && coverage_mask_b == 0) {
                        for (const auto edge_index :
                             std::views::iota(std::size_t{0}, std::size_t{3})) {
                            quad_accumulator[edge_index] += 4 * triangle.step_x[edge_index];
                        }

                        quad_x += 4;
                        continue;
                    }

                    std::array<float, 8> edge_values_1{};
                    std::array<float, 8> edge_values_2{};
                    for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                        edge_values_1[corner] = static_cast<float>(corner_edges[corner][1]);
                        edge_values_2[corner] = static_cast<float>(corner_edges[corner][2]);
                        edge_values_1[corner + 4U] = static_cast<float>(corner_edges_b[corner][1]);
                        edge_values_2[corner + 4U] = static_cast<float>(corner_edges_b[corner][2]);
                    }

                    std::array<float, 8> lane_l0{};
                    std::array<float, 8> lane_l1{};
                    std::array<float, 8> lane_l2{};
                    simd::barycentric_weights8(
                        edge_values_1,
                        edge_values_2,
                        triangle.inv_area,
                        lane_l0,
                        lane_l1,
                        lane_l2
                    );

                    std::array<std::array<float, 3>, 4> weights_a{};
                    std::array<std::array<float, 3>, 4> weights_b{};
                    for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                        weights_a[corner] = std::array<float, 3>{
                            lane_l0[corner],
                            lane_l1[corner],
                            lane_l2[corner],
                        };
                        weights_b[corner] = std::array<float, 3>{
                            lane_l0[corner + 4U],
                            lane_l1[corner + 4U],
                            lane_l2[corner + 4U],
                        };
                    }
                    if (coverage_mask != 0U) {
                        shade_covered_quad(quad_x, weights_a, coverage_mask);
                    }
                    if (coverage_mask_b != 0U) {
                        shade_covered_quad(quad_x + 2, weights_b, coverage_mask_b);
                    }
                    for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                        quad_accumulator[edge_index] += 4 * triangle.step_x[edge_index];
                    }

                    quad_x += 4;
                    continue;
                }
                if (coverage_mask == 0) {
                    for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                        quad_accumulator[edge_index] += 2 * triangle.step_x[edge_index];
                    }

                    quad_x += 2;
                    continue;
                }

                std::array<std::array<float, 3>, 4> corner_weights{};
                for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    const auto l0 = static_cast<float>(corner_edges[corner][1]) * triangle.inv_area;
                    const auto l1 = static_cast<float>(corner_edges[corner][2]) * triangle.inv_area;
                    const auto l0_plus_l1 = l0 + l1;
                    corner_weights[corner] = std::array<float, 3>{
                        l0,
                        l1,
                        1.0F - l0_plus_l1,
                    };
                }

                shade_covered_quad(quad_x, corner_weights, coverage_mask);
                for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                    quad_accumulator[edge_index] += 2 * triangle.step_x[edge_index];
                }
                quad_x += 2;
            }
            for (const auto edge_index : std::views::iota(std::size_t{0}, std::size_t{3})) {
                row_accumulator[edge_index] += 2 * triangle.step_y[edge_index];
            }
        }
    }
} // namespace raster
