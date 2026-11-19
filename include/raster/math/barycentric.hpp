#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace raster {
    // The float <-> 28.4 fixed-point hybrid in this header (and the exact
    // barycentric partition in pipeline/rasterizer.hpp) assumes that float is
    // exactly IEEE-754 binary32 with round-to-nearest-even and no excess
    // precision.
    static_assert(
        std::numeric_limits<float>::is_iec559, "raster requires an IEEE-754 binary32 float"
    );
    static_assert(
        std::numeric_limits<float>::digits == 24,
        "raster requires float to have exactly 24 significand bits (binary32)"
    );
    static_assert(
        std::numeric_limits<float>::round_style == std::round_to_nearest,
        "the exact barycentric partition requires round-to-nearest-even"
    );
    static_assert(
        FLT_EVAL_METHOD == 0,
        "excess-precision evaluation (x87) would break exact 28.4 fixed-point math"
    );

    // 28.4 fixed-point screen-space rasterization primitives.
    //
    // Coordinate contract: screen origin is top-left, +X right, +Y down. One
    // pixel spans 16 sub-pixel units (2^4); the sampling center of integer
    // pixel (x, y) is exactly ((x << 4) + 8, (y << 4) + 8) in sub-pixel units.
    //
    // Edge-function sign convention: the plan (§5.1) writes
    //   E_ab(P) = (X_a - X_b)(Y_p - Y_a) - (Y_a - Y_b)(X_p - X_a)
    // but also requires Area_fx = E_01(V_2) > 0 for front-facing triangles,
    // where a CCW-winding (view-space, +Y up) triangle projects to a visually
    // clockwise loop under the +Y-down viewport flip. Those two statements are
    // inconsistent as written: for the clockwise loop the formula above
    // evaluates to -E_standard, so Area_fx = E_01(V_2) < 0. The consistent
    // convention — used here and matching the engine's winding claims — is the
    // standard cross product:
    //   E_ab(P) = (X_b - X_a)(Y_p - Y_a) - (Y_b - Y_a)(X_p - X_a)
    // so that interior points satisfy E >= 0 on every edge and the twice-signed
    // area (scaled by 2^8, see below) is E_01(V_2) > 0 for front faces.

    inline constexpr std::uint32_t subpixel_bits = 4;
    inline constexpr std::int32_t subpixel_scale = static_cast<std::int32_t>(1U << subpixel_bits);
    inline constexpr std::int32_t subpixel_center_offset = subpixel_scale / 2;

    // Edge values and areas live two bits deeper than inputs: (2^4) * (2^4)
    // products make them 28.8 fixed-point quantities, so they are always
    // evaluated in 64 bits to stay overflow-free for triangles far larger than
    // the screen.
    inline constexpr std::int32_t edge_fractional_bits =
        static_cast<std::int32_t>(2U * subpixel_bits);

    // Rounds a floating-point pixel coordinate to the nearest 28.4 sub-pixel
    // unit.
    [[nodiscard]]
    constexpr std::int32_t to_fixed(float pixel_coordinate) noexcept {
        return static_cast<std::int32_t>(
            std::nearbyint(pixel_coordinate * static_cast<float>(subpixel_scale))
        );
    }

    // Sub-pixel center of integer pixel coordinate x (28.4). Multiplication
    // instead of a shift keeps negative pixel indices well defined.
    [[nodiscard]]
    constexpr std::int32_t pixel_center(std::int32_t pixel) noexcept {
        return (pixel * subpixel_scale) + subpixel_center_offset;
    }

    // Standard 2D cross product (B - A) x (P - A) in 28.8 fixed-point:
    // > 0 when P is on the interior side of edge A -> B for a front-facing
    // (positive-shoelace) triangle.
    [[nodiscard]]
    constexpr std::int64_t edge_function(
        std::int32_t ax,
        std::int32_t ay,
        std::int32_t bx,
        std::int32_t by,
        std::int64_t px,
        std::int64_t py
    ) noexcept {
        const std::int64_t edge_dx = static_cast<std::int64_t>(bx) - ax;
        const std::int64_t edge_dy = static_cast<std::int64_t>(by) - ay;
        return edge_dx * (py - ay) - edge_dy * (px - ax);
    }

    // Exact per-pixel edge deltas in 28.8 fixed-point: moving the sample point
    // one full pixel right adds -16 * edge_dy; one full pixel down adds
    // +16 * edge_dx (plan §5.1 deltas, sign-corrected for the standard cross).
    [[nodiscard]]
    constexpr std::int64_t edge_step_x(std::int32_t ay, std::int32_t by) noexcept {
        return -static_cast<std::int64_t>(by - ay) * subpixel_scale;
    }

    [[nodiscard]]
    constexpr std::int64_t edge_step_y(std::int32_t ax, std::int32_t bx) noexcept {
        return static_cast<std::int64_t>(bx - ax) * subpixel_scale;
    }

    // Top-left fill rule (D3D/GL semantics under the +Y-down flip):
    // a pixel whose center lies exactly on an edge is drawn only for
    //   * top edges: exactly horizontal, interior below it (directed +X), or
    //   * left edges: interior to the visually-right side (directed -Y).
    // All other exact-on-edge centers are culled, which makes shared edges
    // watertight: adjacent triangles traverse a shared edge in opposite
    // directions, so exactly one of them qualifies as top/left.
    [[nodiscard]]
    constexpr bool is_top_or_left_edge(
        std::int32_t ax, std::int32_t ay, std::int32_t bx, std::int32_t by
    ) noexcept {
        const std::int32_t edge_dx = bx - ax;
        const std::int32_t edge_dy = by - ay;
        return edge_dy < 0 || (edge_dy == 0 && edge_dx > 0);
    }

    // Coverage bias added to a raw edge value before the >= 0 test: 0 keeps
    // exact-on-edge centers for top/left edges, -1 culls them otherwise.
    [[nodiscard]]
    constexpr std::int64_t edge_bias(
        std::int32_t ax, std::int32_t ay, std::int32_t bx, std::int32_t by
    ) noexcept {
        return is_top_or_left_edge(ax, ay, bx, by) ? 0 : -1;
    }
} // namespace raster
