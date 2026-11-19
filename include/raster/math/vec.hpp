#pragma once

#include <cmath>

namespace raster {
    struct Vec2f {
        float x{};
        float y{};
    };

    struct Vec3f {
        float x{};
        float y{};
        float z{};
    };

    struct alignas(16) Vec4f {
        float x{};
        float y{};
        float z{};
        float w{};

        [[nodiscard]]
        constexpr Vec3f xyz() const noexcept {
            return Vec3f{.x = x, .y = y, .z = z};
        }
    };

    [[nodiscard]]
    constexpr float lerp(float a, float b, float t) noexcept {
        return a + (b - a) * t;
    }

    [[nodiscard]]
    constexpr Vec2f lerp(const Vec2f& a, const Vec2f& b, float t) noexcept {
        return Vec2f{.x = lerp(a.x, b.x, t), .y = lerp(a.y, b.y, t)};
    }

    [[nodiscard]]
    constexpr Vec3f lerp(const Vec3f& a, const Vec3f& b, float t) noexcept {
        return Vec3f{
            .x = lerp(a.x, b.x, t),
            .y = lerp(a.y, b.y, t),
            .z = lerp(a.z, b.z, t),
        };
    }

    [[nodiscard]]
    constexpr Vec4f lerp(const Vec4f& a, const Vec4f& b, float t) noexcept {
        return Vec4f{
            .x = lerp(a.x, b.x, t),
            .y = lerp(a.y, b.y, t),
            .z = lerp(a.z, b.z, t),
            .w = lerp(a.w, b.w, t),
        };
    }

    // Component-wise ternary weighted sum a*w0 + b*w1 + c*w2 used by shader
    // attribute interpolation (weights are the per-pixel barycentrics).
    [[nodiscard]]
    constexpr float lerp3(float a, float b, float c, float w0, float w1, float w2) noexcept {
        return a * w0 + b * w1 + c * w2;
    }

    [[nodiscard]]
    constexpr Vec2f lerp3(
        const Vec2f& a, const Vec2f& b, const Vec2f& c, float w0, float w1, float w2
    ) noexcept {
        return Vec2f{
            .x = lerp3(a.x, b.x, c.x, w0, w1, w2),
            .y = lerp3(a.y, b.y, c.y, w0, w1, w2),
        };
    }

    [[nodiscard]]
    constexpr Vec3f lerp3(
        const Vec3f& a, const Vec3f& b, const Vec3f& c, float w0, float w1, float w2
    ) noexcept {
        return Vec3f{
            .x = lerp3(a.x, b.x, c.x, w0, w1, w2),
            .y = lerp3(a.y, b.y, c.y, w0, w1, w2),
            .z = lerp3(a.z, b.z, c.z, w0, w1, w2),
        };
    }

    [[nodiscard]]
    constexpr Vec4f lerp3(
        const Vec4f& a, const Vec4f& b, const Vec4f& c, float w0, float w1, float w2
    ) noexcept {
        return Vec4f{
            .x = lerp3(a.x, b.x, c.x, w0, w1, w2),
            .y = lerp3(a.y, b.y, c.y, w0, w1, w2),
            .z = lerp3(a.z, b.z, c.z, w0, w1, w2),
            .w = lerp3(a.w, b.w, c.w, w0, w1, w2),
        };
    }

    [[nodiscard]]
    constexpr Vec4f operator+(const Vec4f& a, const Vec4f& b) noexcept {
        return Vec4f{.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w};
    }

    [[nodiscard]]
    constexpr Vec4f operator*(const Vec4f& a, float scalar) noexcept {
        return Vec4f{
            .x = a.x * scalar,
            .y = a.y * scalar,
            .z = a.z * scalar,
            .w = a.w * scalar,
        };
    }

    [[nodiscard]]
    constexpr Vec3f operator-(const Vec3f& a, const Vec3f& b) noexcept {
        return Vec3f{.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
    }

    [[nodiscard]]
    constexpr Vec3f operator+(const Vec3f& a, const Vec3f& b) noexcept {
        return Vec3f{.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z};
    }

    [[nodiscard]]
    constexpr Vec3f operator*(const Vec3f& a, float scalar) noexcept {
        return Vec3f{.x = a.x * scalar, .y = a.y * scalar, .z = a.z * scalar};
    }

    [[nodiscard]]
    constexpr Vec2f operator-(const Vec2f& a, const Vec2f& b) noexcept {
        return Vec2f{.x = a.x - b.x, .y = a.y - b.y};
    }

    [[nodiscard]]
    constexpr float dot(const Vec2f& a, const Vec2f& b) noexcept {
        return a.x * b.x + a.y * b.y;
    }

    [[nodiscard]]
    constexpr float dot(const Vec3f& a, const Vec3f& b) noexcept {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    [[nodiscard]]
    constexpr float dot(const Vec4f& a, const Vec4f& b) noexcept {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    [[nodiscard]]
    constexpr Vec3f cross(const Vec3f& a, const Vec3f& b) noexcept {
        return Vec3f{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    [[nodiscard]]
    constexpr float length_squared(const Vec3f& a) noexcept {
        return dot(a, a);
    }

    [[nodiscard]]
    inline float length(const Vec3f& a) noexcept {
        return std::sqrt(length_squared(a));
    }

    // Normalizes to unit length; the zero vector normalizes to the zero vector
    // instead of producing NaNs (callers rely on well-defined output).
    [[nodiscard]]
    inline Vec3f normalize(const Vec3f& a) noexcept {
        const float len = length(a);
        if (len == 0.0F) {
            return Vec3f{};
        }
        return Vec3f{.x = a.x / len, .y = a.y / len, .z = a.z / len};
    }
} // namespace raster
