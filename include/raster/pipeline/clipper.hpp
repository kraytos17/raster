#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>

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
        inline float clip_plane_distance(const Vec4f& point, ClipPlane plane) noexcept {
            switch (plane) {
                case ClipPlane::Left:
                    return point.x + point.w;
                case ClipPlane::Right:
                    return point.w - point.x;
                case ClipPlane::Bottom:
                    return point.y + point.w;
                case ClipPlane::Top:
                    return point.w - point.y;
                case ClipPlane::Far:
                    return point.z;
                case ClipPlane::Near:
                    return point.w - point.z;
            }
            return -1.0F;
        }

        [[nodiscard]]
        inline std::uint8_t clip_outcode(const Vec4f& point) noexcept {
            std::uint8_t code = 0;
            for (std::size_t index = 0; index < clip_plane_count; ++index) {
                const auto plane = static_cast<ClipPlane>(index);
                if (clip_plane_distance(point, plane) < 0.0F) {
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
    } // namespace detail

    // Six-plane homogeneous Sutherland-Hodgman clipping. Every edge intersection
    // interpolates clip position and all ShaderBase-tied varyings in homogeneous space before
    // any perspective divide. Scratch storage and triangle-fan output are
    // fixed-size stack arrays.
    template <ShaderProgram Shader> [[nodiscard]]
    std::expected<ClippedTriangleFan<typename Shader::Varyings>, ClipError> clip_triangle(
        const Shader& shader,
        const typename Shader::Varyings& vertex_a,
        const typename Shader::Varyings& vertex_b,
        const typename Shader::Varyings& vertex_c
    ) noexcept {
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

            const auto code = detail::clip_outcode(vertex->clip_pos);
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
            float previous_distance = detail::clip_plane_distance(previous->clip_pos, plane);
            bool previous_inside = previous_distance >= 0.0F;
            for (std::size_t vertex_index = 0; vertex_index < input.size; ++vertex_index) {
                const VaryingsT& current = input.vertices[vertex_index];
                const float current_distance = detail::clip_plane_distance(current.clip_pos, plane);
                const bool current_inside = current_distance >= 0.0F;
                if (current_inside != previous_inside) {
                    if (output.size >= clip_polygon_capacity) {
                        return std::unexpected{ClipError::PolygonCapacityExceeded};
                    }

                    const float t = detail::intersection_t(previous_distance, current_distance);
                    if (!std::isfinite(t) || t < 0.0F || t > 1.0F) {
                        return std::unexpected{ClipError::InvalidIntersection};
                    }
                    output.vertices[output.size++] = shader.interpolate_edge(*previous, current, t);
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
} // namespace raster
