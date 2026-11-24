#pragma once

// Shared non-shader fundamentals for the scene render pipeline:
// framebuffer packing, the reverse-Z shadow-map store, and the overdraw
// probes. Included by the other scene detail headers; no renderer entry
// points live here.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "raster/memory/texture.hpp"

namespace raster::render::detail {
    using std::chrono::steady_clock;

    struct Rgb {
        float r{};
        float g{};
        float b{};
    };

    [[nodiscard]]
    inline std::uint32_t to_argb(const Rgb& rgb) noexcept {
        const auto red = quantize_channel(std::clamp(rgb.r, 0.0F, 1.0F) * 255.0F);
        const auto green = quantize_channel(std::clamp(rgb.g, 0.0F, 1.0F) * 255.0F);
        const auto blue = quantize_channel(std::clamp(rgb.b, 0.0F, 1.0F) * 255.0F);
        return 0xFF000000U | (red << 16U) | (green << 8U) | blue;
    }

    // Fixed-size reverse-Z shadow map: one float per texel, cleared to far
    // (0.0) each frame, written by the depth-only light pass. Allocated
    // once — never in the per-frame render path.
    struct ShadowMap {
        static constexpr std::size_t k_size = 1024;
        std::vector<float> depth = std::vector<float>(k_size * k_size, 0.0F);

        void clear() noexcept {
            std::ranges::fill(depth, 0.0F);
        }
    };

    inline constexpr float k_shadow_eye_distance = 30.0F;
    inline constexpr float k_shadow_half_extent = 16.0F;
    inline constexpr float k_shadow_near = 15.0F;
    inline constexpr float k_shadow_far = 45.0F;
    inline constexpr float k_shadow_bias = 0.002F;
    inline constexpr float k_shadow_slope = 0.01F;

    // Overdraw probe accumulators.
    inline std::atomic_uint64_t profile_covered_lanes{0};
    inline std::atomic_uint64_t profile_visible_lanes{0};
} // namespace raster::render::detail
