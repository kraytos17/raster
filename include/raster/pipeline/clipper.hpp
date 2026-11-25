#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>

#include "raster/math/vec.hpp"
#include "raster/pipeline/shader.hpp"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace raster {
    inline constexpr std::size_t clip_plane_count = 6;
    inline constexpr std::size_t clip_polygon_capacity = 12;
    inline constexpr std::size_t clipped_triangle_capacity = clip_polygon_capacity - 2;

    enum class ClipPlane : std::uint8_t {
        Left,
        Right,
        Bottom,
        Top,
        Far,
        Near,
    };

    enum class ClipError {
        NonFiniteVertex,
        PolygonCapacityExceeded,
        InvalidIntersection,
        InvalidSurvivorW,
        InvalidGuardBand,
    };

    template <typename T, std::size_t Capacity> struct FixedPolygon {
        std::array<T, Capacity> vertices{};
        std::size_t size{};

        [[nodiscard]]
        constexpr bool empty() const noexcept {
            return size == 0;
        }

        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept {
            return vertices[index];
        }

        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept {
            return vertices[index];
        }

        [[nodiscard]]
        constexpr std::span<const T> active_vertices() const noexcept {
            return {vertices.data(), size};
        }

        [[nodiscard]]
        constexpr std::span<T> active_vertices() noexcept {
            return {vertices.data(), size};
        }
    };

    template <typename VaryingsT> struct ClippedTriangle {
        std::array<VaryingsT, 3> vertices{};
    };

    template <typename VaryingsT> struct ClippedTriangleFan {
        std::array<ClippedTriangle<VaryingsT>, clipped_triangle_capacity> triangles{};
        std::size_t size{};

        [[nodiscard]]
        constexpr std::span<const ClippedTriangle<VaryingsT>> active_triangles() const noexcept {
            return {triangles.data(), size};
        }

        [[nodiscard]]
        constexpr std::span<ClippedTriangle<VaryingsT>> active_triangles() noexcept {
            return {triangles.data(), size};
        }
    };

    namespace detail {
        [[nodiscard]]
        inline float clip_plane_distance(
            const Vec4f& point, ClipPlane plane, float guard_band = 1.0F
        ) noexcept {
            switch (plane) {
                case ClipPlane::Left:
                    return point.x + guard_band * point.w;
                case ClipPlane::Right:
                    return guard_band * point.w - point.x;
                case ClipPlane::Bottom:
                    return point.y + guard_band * point.w;
                case ClipPlane::Top:
                    return guard_band * point.w - point.y;
                case ClipPlane::Far:
                    return point.z;
                case ClipPlane::Near:
                    return point.w - point.z;
            }
            return -1.0F;
        }

        [[nodiscard]]
        inline std::uint8_t clip_outcode(const Vec4f& point, float guard_band) noexcept {
            const auto scaled_w = guard_band * point.w;
            const auto left_out = point.x + scaled_w < 0.0F;
            const auto right_out = scaled_w - point.x < 0.0F;
            const auto bottom_out = point.y + scaled_w < 0.0F;
            const auto top_out = scaled_w - point.y < 0.0F;
            const auto far_out = point.z < 0.0F;
            const auto near_out = point.w - point.z < 0.0F;
            const auto code = (static_cast<unsigned>(left_out) << 0U)
                | (static_cast<unsigned>(right_out) << 1U)
                | (static_cast<unsigned>(bottom_out) << 2U)
                | (static_cast<unsigned>(top_out) << 3U)
                | (static_cast<unsigned>(far_out) << 4U)
                | (static_cast<unsigned>(near_out) << 5U);
            return static_cast<std::uint8_t>(code);
        }

        inline void clip_triangle_codes(
            const Vec4f& vertex_a,
            const Vec4f& vertex_b,
            const Vec4f& vertex_c,
            float guard_band,
            unsigned& common_out,
            unsigned& combined_out
        ) noexcept {
#if defined(__x86_64__)
            __m128 row0 = _mm_loadu_ps(&vertex_a.x);
            __m128 row1 = _mm_loadu_ps(&vertex_b.x);
            __m128 row2 = _mm_loadu_ps(&vertex_c.x);
            __m128 row3 = row2;
            // 4x4 transpose rows (ax,ay,az,aw)... into columns, so each
            // plane test runs over all three vertices in one vector op.
            // The fourth lane duplicates vertex C and is masked off below.
            const __m128 unpack0 = _mm_unpacklo_ps(row0, row1);
            const __m128 unpack1 = _mm_unpacklo_ps(row2, row3);
            const __m128 unpack2 = _mm_unpackhi_ps(row0, row1);
            const __m128 unpack3 = _mm_unpackhi_ps(row2, row3);
            const __m128 xs = _mm_movelh_ps(unpack0, unpack1);
            const __m128 ys = _mm_movehl_ps(unpack1, unpack0);
            const __m128 zs = _mm_movelh_ps(unpack2, unpack3);
            const __m128 ws = _mm_movehl_ps(unpack3, unpack2);
            const __m128 scaled = _mm_mul_ps(_mm_set1_ps(guard_band), ws);
            const __m128 zero = _mm_setzero_ps();

            unsigned common_bits = (1U << clip_plane_count) - 1U;
            unsigned combined_bits = 0U;
            const auto accumulate = [&](__m128 distance, unsigned bit) noexcept {
                const auto mask =
                    static_cast<unsigned>(_mm_movemask_ps(_mm_cmplt_ps(distance, zero))) & 0x7U;
                combined_bits |= (mask != 0U) ? bit : 0U;
                common_bits &= (mask == 0x7U) ? ~0U : ~bit;
            };

            accumulate(_mm_add_ps(xs, scaled), 1U << 0U);
            accumulate(_mm_sub_ps(scaled, xs), 1U << 1U);
            accumulate(_mm_add_ps(ys, scaled), 1U << 2U);
            accumulate(_mm_sub_ps(scaled, ys), 1U << 3U);
            accumulate(zs, 1U << 4U);
            accumulate(_mm_sub_ps(ws, zs), 1U << 5U);

            common_out = common_bits;
            combined_out = combined_bits;
#else
            const auto code_a = static_cast<unsigned>(clip_outcode(vertex_a, guard_band));
            const auto code_b = static_cast<unsigned>(clip_outcode(vertex_b, guard_band));
            const auto code_c = static_cast<unsigned>(clip_outcode(vertex_c, guard_band));

            common_out = ((1U << clip_plane_count) - 1U) & code_a & code_b & code_c;
            combined_out = code_a | code_b | code_c;
#endif
        }

        [[nodiscard]]
        inline bool finite(const Vec4f& value) noexcept {
            return std::isfinite(value.x)
                && std::isfinite(value.y)
                && std::isfinite(value.z)
                && std::isfinite(value.w);
        }

        [[nodiscard]]
        inline float intersection_t(float distance_a, float distance_b) noexcept {
            return distance_a / (distance_a - distance_b);
        }

        inline void snap_intersection_to_plane(
            Vec4f& point, ClipPlane plane, float guard_band
        ) noexcept {
            switch (plane) {
                case ClipPlane::Left:
                    point.x = -guard_band * point.w;
                    break;
                case ClipPlane::Right:
                    point.x = guard_band * point.w;
                    break;
                case ClipPlane::Bottom:
                    point.y = -guard_band * point.w;
                    break;
                case ClipPlane::Top:
                    point.y = guard_band * point.w;
                    break;
                case ClipPlane::Far:
                    point.z = 0.0F;
                    break;
                case ClipPlane::Near:
                    point.z = point.w;
                    break;
            }
        }

        [[nodiscard]]
        inline bool snap_roundoff_to_interval(float& value, float lower, float upper) noexcept {
            const auto scale = std::max({
                std::fabs(value),
                std::fabs(lower),
                std::fabs(upper),
                std::numeric_limits<float>::min(),
            });

            const auto tolerance = 4.0F * std::numeric_limits<float>::epsilon() * scale;
            if (value < lower) {
                if (lower - value > tolerance) {
                    return false;
                }
                value = lower;
            } else if (value > upper) {
                if (value - upper > tolerance) {
                    return false;
                }
                value = upper;
            }
            return true;
        }
    } // namespace detail

    // A triangle is safely outside the canonical clip volume only when all
    // three vertices lie beyond the same plane.
    [[nodiscard]]
    inline bool triangle_outside_clip_frustum(
        const std::array<Vec4f, 3>& clip_positions, float guard_band = 1.0F
    ) noexcept {
        if (!std::isfinite(guard_band) || guard_band < 1.0F) {
            return false;
        }

        auto common_outcode = (1U << clip_plane_count) - 1U;
        for (const auto& position : clip_positions) {
            if (!detail::finite(position)) {
                return false;
            }
            common_outcode &= static_cast<unsigned>(detail::clip_outcode(position, guard_band));
        }
        return common_outcode != 0U;
    }

    // Six-plane homogeneous Sutherland-Hodgman clipping. Every edge intersection
    // interpolates clip position and all ShaderBase-tied varyings in homogeneous space before
    // any perspective divide. Scratch storage and triangle-fan output are
    // fixed-size stack arrays.
    //
    // The with-outcodes entry shares one outcode computation between the
    // coarse frustum reject and the clip itself.
    template <ShaderProgram Shader> [[nodiscard]]
    std::expected<ClippedTriangleFan<typename Shader::Varyings>, ClipError>
    clip_triangle_impl_with_masks(
        const Shader& shader,
        const typename Shader::Varyings& vertex_a,
        const typename Shader::Varyings& vertex_b,
        const typename Shader::Varyings& vertex_c,
        float guard_band,
        unsigned common_outcode,
        unsigned combined_outcode
    ) noexcept {
        if (!std::isfinite(guard_band) || guard_band < 1.0F) {
            return std::unexpected{ClipError::InvalidGuardBand};
        }

        using VaryingsT = Shader::Varyings;
        FixedPolygon<VaryingsT, clip_polygon_capacity> input{};
        FixedPolygon<VaryingsT, clip_polygon_capacity> output{};

        input.vertices[0] = vertex_a;
        input.vertices[1] = vertex_b;
        input.vertices[2] = vertex_c;
        input.size = 3;

        ClippedTriangleFan<VaryingsT> fan{};
        if (common_outcode != 0U) {
            return fan; // trivial reject
        }
        if (combined_outcode == 0U) {
            if (vertex_a.clip_pos.w
                <= 0.0F
                || vertex_b.clip_pos.w
                <= 0.0F
                || vertex_c.clip_pos.w
                <= 0.0F) {
                return std::unexpected{ClipError::InvalidSurvivorW};
            }

            fan.triangles[0].vertices = {vertex_a, vertex_b, vertex_c};
            fan.size = 1;
            return fan; // trivial accept
        }
        for (std::size_t plane_index = 0; plane_index < clip_plane_count; ++plane_index) {
            const auto plane = static_cast<ClipPlane>(plane_index);
            output.size = 0;
            if (input.empty()) {
                break;
            }

            const auto input_vertices = input.active_vertices();
            const VaryingsT* previous = &input_vertices.back();
            float previous_distance =
                detail::clip_plane_distance(previous->clip_pos, plane, guard_band);

            bool previous_inside = previous_distance >= 0.0F;
            for (std::size_t vertex_index = 0; vertex_index < input.size; ++vertex_index) {
                const VaryingsT& current = input_vertices[vertex_index];
                const float current_distance =
                    detail::clip_plane_distance(current.clip_pos, plane, guard_band);
                const bool current_inside = current_distance >= 0.0F;
                if (current_inside != previous_inside) {
                    if (output.size >= clip_polygon_capacity) {
                        return std::unexpected{ClipError::PolygonCapacityExceeded};
                    }

                    const float t = detail::intersection_t(previous_distance, current_distance);
                    if (!std::isfinite(t) || t < 0.0F || t > 1.0F) {
                        return std::unexpected{ClipError::InvalidIntersection};
                    }

                    auto intersection = shader.interpolate_edge(*previous, current, t);
                    detail::snap_intersection_to_plane(intersection.clip_pos, plane, guard_band);
                    output.vertices[output.size++] = intersection;
                }
                if (current_inside) {
                    if (output.size >= clip_polygon_capacity) {
                        return std::unexpected{ClipError::PolygonCapacityExceeded};
                    }
                    output.vertices[output.size++] = current;
                }

                previous = &current;
                previous_distance = current_distance;
                previous_inside = current_inside;
            }
            input = output;
        }
        // The six canonical inequalities force positive w for any nonzero
        // surviving point. Check explicitly before downstream inv_w = 1/w.
        for (auto& vertex : input.active_vertices()) {
            const auto& point = vertex.clip_pos;
            if (!detail::finite(point) || point.w <= 0.0F) {
                return std::unexpected{ClipError::InvalidSurvivorW};
            }

            auto canonical_point = point;
            if (!detail::snap_roundoff_to_interval(
                    canonical_point.x,
                    -guard_band * point.w,
                    guard_band * point.w
                )
                || !detail::snap_roundoff_to_interval(
                    canonical_point.y,
                    -guard_band * point.w,
                    guard_band * point.w
                )
                || !detail::snap_roundoff_to_interval(canonical_point.z, 0.0F, point.w)) {
                return std::unexpected{ClipError::InvalidSurvivorW};
            }
            vertex.clip_pos = canonical_point;
        }
        if (input.size < 3) {
            return fan;
        }
        if (input.size - 2 > fan.triangles.size()) {
            return std::unexpected{ClipError::PolygonCapacityExceeded};
        }

        const auto iv = input.active_vertices();
        for (std::size_t index = 1; index + 1 < iv.size(); ++index) {
            fan.triangles[fan.size++].vertices = {
                iv[0],
                iv[index],
                iv[index + 1],
            };
        }
        return fan;
    }

    template <ShaderProgram Shader> [[nodiscard]]
    std::expected<ClippedTriangleFan<typename Shader::Varyings>, ClipError>
    clip_triangle_impl_with_outcodes(
        const Shader& shader,
        const typename Shader::Varyings& vertex_a,
        const typename Shader::Varyings& vertex_b,
        const typename Shader::Varyings& vertex_c,
        float guard_band,
        std::uint8_t code_a,
        std::uint8_t code_b,
        std::uint8_t code_c
    ) noexcept {
        const auto outcode_a = static_cast<unsigned>(code_a);
        const auto outcode_b = static_cast<unsigned>(code_b);
        const auto outcode_c = static_cast<unsigned>(code_c);
        const auto all_inside = (1U << clip_plane_count) - 1U;
        return clip_triangle_impl_with_masks(
            shader,
            vertex_a,
            vertex_b,
            vertex_c,
            guard_band,
            all_inside & outcode_a & outcode_b & outcode_c,
            outcode_a | outcode_b | outcode_c
        );
    }

    template <ShaderProgram Shader> [[nodiscard]]
    std::expected<ClippedTriangleFan<typename Shader::Varyings>, ClipError> clip_triangle_impl(
        const Shader& shader,
        const typename Shader::Varyings& vertex_a,
        const typename Shader::Varyings& vertex_b,
        const typename Shader::Varyings& vertex_c,
        float guard_band
    ) noexcept {
        if (!std::isfinite(guard_band) || guard_band < 1.0F) {
            return std::unexpected{ClipError::InvalidGuardBand};
        }
        if (!detail::finite(vertex_a.clip_pos)
            || !detail::finite(vertex_b.clip_pos)
            || !detail::finite(vertex_c.clip_pos)) {
            return std::unexpected{ClipError::NonFiniteVertex};
        }
        return clip_triangle_impl_with_outcodes(
            shader,
            vertex_a,
            vertex_b,
            vertex_c,
            guard_band,
            detail::clip_outcode(vertex_a.clip_pos, guard_band),
            detail::clip_outcode(vertex_b.clip_pos, guard_band),
            detail::clip_outcode(vertex_c.clip_pos, guard_band)
        );
    }

    template <ShaderProgram Shader> [[nodiscard]]
    std::expected<ClippedTriangleFan<typename Shader::Varyings>, ClipError> clip_triangle(
        const Shader& shader,
        const typename Shader::Varyings& vertex_a,
        const typename Shader::Varyings& vertex_b,
        const typename Shader::Varyings& vertex_c
    ) noexcept {
        return clip_triangle_impl(shader, vertex_a, vertex_b, vertex_c, 1.0F);
    }

    // Experimental wider X/Y clipping envelope. Near/far remain canonical;
    // the screen-space rasterizer still clamps coverage to the viewport.
    template <ShaderProgram Shader> [[nodiscard]]
    std::
        expected<ClippedTriangleFan<typename Shader::Varyings>, ClipError> clip_triangle_guard_band(
            const Shader& shader,
            const typename Shader::Varyings& vertex_a,
            const typename Shader::Varyings& vertex_b,
            const typename Shader::Varyings& vertex_c,
            float guard_band
        ) noexcept {
        return clip_triangle_impl(shader, vertex_a, vertex_b, vertex_c, guard_band);
    }
} // namespace raster
