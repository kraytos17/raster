#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <ranges>

#include "raster/math/vec.hpp"

namespace raster {

    // Column-major 4x4 matrix: columns[i] holds column i, so element (row, col)
    // is columns[col][row].
    struct Mat4f {
        std::array<Vec4f, 4> columns{};

        [[nodiscard]]
        constexpr float operator[](std::size_t row, std::size_t column) const noexcept {
            const Vec4f& column_vector = columns[column];
            switch (row) {
                case 0:
                    return column_vector.x;
                case 1:
                    return column_vector.y;
                case 2:
                    return column_vector.z;
                default:
                    return column_vector.w;
            }
        }
    };

    [[nodiscard]]
    constexpr Mat4f identity4f() noexcept {
        return Mat4f{
            .columns = {
                Vec4f{.x = 1.0F, .y = 0.0F, .z = 0.0F, .w = 0.0F},
                Vec4f{.x = 0.0F, .y = 1.0F, .z = 0.0F, .w = 0.0F},
                Vec4f{.x = 0.0F, .y = 0.0F, .z = 1.0F, .w = 0.0F},
                Vec4f{.x = 0.0F, .y = 0.0F, .z = 0.0F, .w = 1.0F},
            },
        };
    }

    [[nodiscard]]
    constexpr Mat4f translation4f(const Vec3f& offset) noexcept {
        Mat4f result = identity4f();
        result.columns[3] = Vec4f{.x = offset.x, .y = offset.y, .z = offset.z, .w = 1.0F};
        return result;
    }

    [[nodiscard]]
    inline Mat4f rotation_y4f(float angle) noexcept {
        const float sine = std::sin(angle);
        const float cosine = std::cos(angle);
        return Mat4f{
            .columns = {
                Vec4f{.x = cosine, .y = 0.0F, .z = -sine, .w = 0.0F},
                Vec4f{.x = 0.0F, .y = 1.0F, .z = 0.0F, .w = 0.0F},
                Vec4f{.x = sine, .y = 0.0F, .z = cosine, .w = 0.0F},
                Vec4f{.x = 0.0F, .y = 0.0F, .z = 0.0F, .w = 1.0F},
            },
        };
    }

    // Right-handed world/view space (+X right, +Y up, -Z forward into the
    // screen): a point in front of the camera gets a negative view z.
    [[nodiscard]]
    inline Mat4f look_at4f(const Vec3f& eye, const Vec3f& target, const Vec3f& up) noexcept {
        const Vec3f forward = normalize(target - eye);
        const Vec3f right = normalize(cross(forward, up));
        const Vec3f camera_up = cross(right, forward);
        return Mat4f{
            .columns = {
                Vec4f{
                    .x = right.x,
                    .y = camera_up.x,
                    .z = -forward.x,
                    .w = 0.0F,
                },
                Vec4f{
                    .x = right.y,
                    .y = camera_up.y,
                    .z = -forward.y,
                    .w = 0.0F,
                },
                Vec4f{
                    .x = right.z,
                    .y = camera_up.z,
                    .z = -forward.z,
                    .w = 0.0F,
                },
                Vec4f{
                    .x = -dot(right, eye),
                    .y = -dot(camera_up, eye),
                    .z = dot(forward, eye),
                    .w = 1.0F,
                },
            },
        };
    }

    // Reverse-Z perspective projection (plan §5.1): the near plane maps to
    // z_ndc = 1.0f and the far plane to z_ndc = 0.0f, aligning the 1/z curve
    // with IEEE-754 exponent density. w_c = -z_view, and
    //   z_c = (n / (f - n)) * z_view + (n * f / (f - n))
    // so that z_c / w_c == 1 at the near plane and 0 at the far plane.
    // Arithmetic-only reverse-Z projection constructor. `focal` is cot(fov_y/2),
    // computed by the runtime FOV wrapper below or supplied directly by a
    // constexpr caller. Invalid parameters return identity instead of throwing.
    [[nodiscard]]
    constexpr Mat4f perspective_reverse_z_from_focal4f(
        float focal,
        float aspect,
        float z_near,
        float z_far
    ) noexcept {
        if (!(z_far > z_near) || !(z_near > 0.0F) || !(aspect > 0.0F) || !(focal > 0.0F)) {
            return identity4f();
        }

        const float depth_scale = z_near / (z_far - z_near);
        const float depth_offset = z_near * z_far / (z_far - z_near);
        return Mat4f{
            .columns = {
                Vec4f{.x = focal / aspect, .y = 0.0F, .z = 0.0F, .w = 0.0F},
                Vec4f{.x = 0.0F, .y = focal, .z = 0.0F, .w = 0.0F},
                Vec4f{.x = 0.0F, .y = 0.0F, .z = depth_scale, .w = -1.0F},
                Vec4f{.x = 0.0F, .y = 0.0F, .z = depth_offset, .w = 0.0F},
            },
        };
    }

    // Runtime field-of-view API uses the platform/libm implementation of tan;
    // the compile-time reverse-Z invariants use the arithmetic-only focal form.
    [[nodiscard]]
    inline Mat4f perspective_reverse_z4f(
        float fov_y,
        float aspect,
        float z_near,
        float z_far
    ) noexcept {
        if (!(fov_y > 0.0F) || !(fov_y < std::numbers::pi_v<float>)) {
            return identity4f();
        }

        const float focal = 1.0F / std::tan(fov_y * 0.5F);
        return perspective_reverse_z_from_focal4f(focal, aspect, z_near, z_far);
    }

    [[nodiscard]]
    constexpr Vec4f operator*(const Mat4f& matrix, const Vec4f& vector) noexcept {
        return matrix.columns[0]
            * vector.x
            + matrix.columns[1]
            * vector.y
            + matrix.columns[2]
            * vector.z
            + matrix.columns[3]
            * vector.w;
    }

    [[nodiscard]]
    constexpr Mat4f operator*(const Mat4f& a, const Mat4f& b) noexcept {
        Mat4f result{};
        for (const auto column : std::views::iota(std::size_t{0}, std::size_t{4})) {
            const Vec4f& b_column = b.columns[column];
            result.columns[column] = a.columns[0]
                * b_column.x
                + a.columns[1]
                * b_column.y
                + a.columns[2]
                * b_column.z
                + a.columns[3]
                * b_column.w;
        }
        return result;
    }
} // namespace raster
