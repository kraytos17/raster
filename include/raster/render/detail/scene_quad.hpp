#pragma once

// Wide quad shading: one trilinear sample per quad for the textured path
// (scalar gathers, wide blend/mix/pack), scalar fragment loop for
// solid-debug and checker fallbacks.

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <SDL3/SDL.h>

#include "raster/memory/texture.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/simd_quad.hpp"
#include "raster/render/detail/scene_shader.hpp"

namespace raster::render::detail {
    // Shared quad store: one wide trilinear sample per quad for the
    // textured path (scalar gathers, wide blend/mix/pack), scalar
    // fragment loop for solid-debug and checker fallbacks.
    template <typename ColorView, typename Layout> void store_shaded_quad(
        ColorView color,
        FloorShader<Layout>& shader,
        const typename FloorShader<Layout>::Varyings& varying_a,
        std::size_t quad_x,
        std::size_t quad_y,
        const QuadBarycentrics& quad,
        const std::array<Vec2f, 4>& quad_uv,
        const std::array<Vec3f, 4>& quad_normal,
        const std::array<Vec3f, 4>& quad_position,
        const std::array<Vec4f, 4>& quad_light,
        float lod,
        float mr_lod
    ) noexcept {
        static const bool profile =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PROFILE") != nullptr;
        if (profile) {
            profile_covered_lanes.fetch_add(
                static_cast<std::uint64_t>(std::popcount(quad.coverage_mask)),
                std::memory_order_relaxed
            );
            profile_visible_lanes.fetch_add(
                static_cast<std::uint64_t>(std::popcount(quad.visible_mask)),
                std::memory_order_relaxed
            );
        }
        if (!shader.uniforms.solid_color_debug
            && shader.uniforms.texture
            != nullptr
            && shader.uniforms.shading
            == SceneShading::Pbr) {
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                if ((quad.visible_mask & (1U << lane)) == 0U) {
                    continue;
                }

                const auto sampled = shader.uniforms.texture->sample_trilinear_linear(
                    quad_uv[lane].x,
                    quad_uv[lane].y,
                    lod
                );

                Vec4f mr_sample{.x = 0.0F, .y = 1.0F, .z = 1.0F, .w = 1.0F};
                if (shader.uniforms.metallic_roughness_texture != nullptr) {
                    mr_sample = shader.uniforms.metallic_roughness_texture->sample_trilinear_linear(
                        quad_uv[lane].x,
                        quad_uv[lane].y,
                        mr_lod
                    );
                }

                const auto resolved = pbr::resolve_textured_material(
                    shader.uniforms.material,
                    Vec3f{.x = sampled.x, .y = sampled.y, .z = sampled.z},
                    Vec3f{.x = mr_sample.x, .y = mr_sample.y, .z = mr_sample.z},
                    shader.uniforms.metallic_roughness_texture != nullptr
                );

                const auto lit = shader.shadow_lit(quad_light[lane], quad_normal[lane]);
                const auto shaded =
                    shader.shade_pbr(resolved, quad_position[lane], quad_normal[lane], lit);
                color[quad_y + (lane >> 1U), quad_x + (lane & 1U)] = pack_channels({
                    linear_to_srgb(shaded.z) * 255.0F,
                    linear_to_srgb(shaded.y) * 255.0F,
                    linear_to_srgb(shaded.x) * 255.0F,
                    255.0F,
                });
            }
            return;
        }
        if (!shader.uniforms.solid_color_debug && shader.uniforms.texture != nullptr) {
            std::array<float, 4> u{};
            std::array<float, 4> v{};
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                u[lane] = quad_uv[lane].x;
                v[lane] = quad_uv[lane].y;
            }

            std::array<std::uint32_t, 4> packed{};
            shader.uniforms.texture->sample_trilinear_quad4(u, v, lod, packed);
            // Unpack, light wide with the same operation order as the
            // scalar shader math, repack. The sample→pack→unpack roundtrip
            // is idempotent on texel bytes, so this matches a
            // single-rounding float path exactly.
            std::array<float, 4> red{};
            std::array<float, 4> green{};
            std::array<float, 4> blue{};
            std::array<float, 4> nx{};
            std::array<float, 4> ny{};
            std::array<float, 4> nz{};
            std::array<float, 4> lane_sun{};
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                red[lane] = static_cast<float>((packed[lane] >> 16U) & 0xFFU) / 255.0F;
                green[lane] = static_cast<float>((packed[lane] >> 8U) & 0xFFU) / 255.0F;
                blue[lane] = static_cast<float>(packed[lane] & 0xFFU) / 255.0F;
                nx[lane] = quad_normal[lane].x;
                ny[lane] = quad_normal[lane].y;
                nz[lane] = quad_normal[lane].z;
                // Shadowed lanes keep ambient only: a zero scale zeroes
                // the diffuse term exactly like the scalar branch.
                const bool lit = (quad.visible_mask & (1U << lane))
                    == 0U
                    || shader.shadow_lit(quad_light[lane], quad_normal[lane]);
                lane_sun[lane] = lit ? shader.uniforms.sun_scale : 0.0F;
            }

            std::array<float, 4> lit_red{};
            std::array<float, 4> lit_green{};
            std::array<float, 4> lit_blue{};
            simd::apply_light4(
                red,
                green,
                blue,
                nx,
                ny,
                nz,
                shader.uniforms.sun_direction.x,
                shader.uniforms.sun_direction.y,
                shader.uniforms.sun_direction.z,
                shader.uniforms.ambient,
                lane_sun,
                lit_red,
                lit_green,
                lit_blue
            );
            simd::quantize_pack4(lit_red, lit_green, lit_blue, packed);
            for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                if ((quad.visible_mask & (1U << lane)) == 0U) {
                    continue;
                }
                color[quad_y + (lane >> 1U), quad_x + (lane & 1U)] = packed[lane];
            }
            return;
        }

        std::array<float, 4> red{};
        std::array<float, 4> green{};
        std::array<float, 4> blue{};
        for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
            if ((quad.visible_mask & (1U << lane)) == 0U) {
                continue;
            }

            const auto shaded = shader.fragment(
                typename FloorShader<Layout>::Varyings{
                    .clip_pos = varying_a.clip_pos,
                    .uv = quad_uv[lane],
                    .normal = quad_normal[lane],
                    .position = quad_position[lane],
                    .light_clip = quad_light[lane],
                    .mip_lod = lod,
                    .metallic_roughness_lod = mr_lod,
                }
            );

            red[lane] = shaded.x;
            green[lane] = shaded.y;
            blue[lane] = shaded.z;
        }

        std::array<std::uint32_t, 4> packed{};
        simd::quantize_pack4(red, green, blue, packed);
        for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
            if ((quad.visible_mask & (1U << lane)) == 0U) {
                continue;
            }
            color[quad_y + (lane >> 1U), quad_x + (lane & 1U)] = packed[lane];
        }
    }
} // namespace raster::render::detail
