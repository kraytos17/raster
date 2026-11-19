#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>

#include "raster/math/vec.hpp"
#include "raster/pipeline/shader.hpp"

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
    };

    template <typename VaryingsT> struct ClippedTriangle {
        std::array<VaryingsT, 3> vertices{};
    };

    template <typename VaryingsT> struct ClippedTriangleFan {
        std::array<ClippedTriangle<VaryingsT>, clipped_triangle_capacity> triangles{};
        std::size_t size{};
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
            std::uint8_t code = 0;
            for (std::size_t index = 0; index < clip_plane_count; ++index) {
                const auto plane = static_cast<ClipPlane>(index);
                if (clip_plane_distance(point, plane, guard_band) < 0.0F) {
                    code = static_cast<std::uint8_t>(code | (1U << index));
                }
            }
            return code;
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

    // Six-plane homogeneous Sutherland-Hodgman clipping. Every edge intersection
    // interpolates clip position and all ShaderBase-tied varyings in homogeneous space before
    // any perspective divide. Scratch storage and triangle-fan output are
    // fixed-size stack arrays.
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

        using VaryingsT = Shader::Varyings;
        FixedPolygon<VaryingsT, clip_polygon_capacity> input{};
        FixedPolygon<VaryingsT, clip_polygon_capacity> output{};

        input.vertices[0] = vertex_a;
        input.vertices[1] = vertex_b;
        input.vertices[2] = vertex_c;
        input.size = 3;

        std::uint8_t common_outcode = static_cast<std::uint8_t>((1U << clip_plane_count) - 1U);
        std::uint8_t combined_outcode = 0;
        const std::array<const VaryingsT*, 3> input_vertices{&vertex_a, &vertex_b, &vertex_c};
        for (const auto* const vertex : input_vertices) {
            if (!detail::finite(vertex->clip_pos)) {
                return std::unexpected{ClipError::NonFiniteVertex};
            }

            const auto code = detail::clip_outcode(vertex->clip_pos, guard_band);
            common_outcode = static_cast<std::uint8_t>(common_outcode & code);
            combined_outcode = static_cast<std::uint8_t>(combined_outcode | code);
        }

        ClippedTriangleFan<VaryingsT> fan{};
        if (common_outcode != 0) {
            return fan; // trivial reject
        }
        if (combined_outcode == 0) {
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

            const VaryingsT* previous = &input.vertices[input.size - 1];
            float previous_distance =
                detail::clip_plane_distance(previous->clip_pos, plane, guard_band);
            bool previous_inside = previous_distance >= 0.0F;
            for (std::size_t vertex_index = 0; vertex_index < input.size; ++vertex_index) {
                const VaryingsT& current = input.vertices[vertex_index];
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
        for (std::size_t vertex_index = 0; vertex_index < input.size; ++vertex_index) {
            const auto& point = input.vertices[vertex_index].clip_pos;
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
            input.vertices[vertex_index].clip_pos = canonical_point;
        }
        if (input.size < 3) {
            return fan;
        }
        if (input.size - 2 > fan.triangles.size()) {
            return std::unexpected{ClipError::PolygonCapacityExceeded};
        }
        for (std::size_t index = 1; index + 1 < input.size; ++index) {
            fan.triangles[fan.size++].vertices = {
                input.vertices[0],
                input.vertices[index],
                input.vertices[index + 1],
            };
        }
        return fan;
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
