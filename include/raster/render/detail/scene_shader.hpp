#pragma once

// Forward metallic-roughness floor shader: lighting, shadows, and the
// scalar fragment reference. The wide quad path in scene_quad.hpp reuses
// the same math; keep the two in lockstep.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/pipeline/pbr.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/shader.hpp"
#include "raster/render/detail/scene_common.hpp"
#include "raster/render/scene_renderer.hpp"

namespace raster::render::detail {
    template <typename Layout> struct FloorShader final : ShaderBase {
        struct Uniforms {
            Mat4f model_view_projection{};
            Mat4f light_view_projection{};
            const Texture2D<Layout>* texture{};
            const Texture2D<Layout>* metallic_roughness_texture{};
            const ShadowMap* shadow_map{};
            bool solid_color_debug{};
            SceneShading shading{SceneShading::Diffuse};
            pbr::Material material{};
            Vec3f eye_position{};
            Vec3f sun_direction{}; // unit vector from the surface toward the sun
            float ambient{};
            float sun_scale{};
            float shadow_bias{};
            float shadow_slope{};
        };

        Uniforms uniforms{};

        struct Varyings {
            Vec4f clip_pos{};
            Vec2f uv{};
            Vec3f normal{};
            Vec3f position{};
            Vec4f light_clip{};
            float mip_lod{};
            float metallic_roughness_lod{};

            [[nodiscard]]
            constexpr auto attributes(this auto&& self) noexcept {
                return std::tie(self.uv, self.normal, self.position, self.light_clip);
            }
        };

        [[nodiscard]]
        constexpr Varyings vertex(Vec3f position, Vec2f uv, Vec3f normal) const noexcept {
            const Vec4f world{
                .x = position.x,
                .y = position.y,
                .z = position.z,
                .w = 1.0F,
            };
            return Varyings{
                .clip_pos = uniforms.model_view_projection * world,
                .uv = uv,
                .normal = normal,
                .position = position,
                .light_clip = uniforms.light_view_projection * world,
            };
        }

        // Ambient + Lambertian diffuse from the uniforms sun. When the
        // shadow map says dark, only ambient survives.
        [[nodiscard]]
        constexpr Vec3f shade_lit(Vec3f albedo, Vec3f normal, bool lit) const noexcept {
            const auto diffuse =
                lit ? std::max(dot(normalize(normal), uniforms.sun_direction), 0.0F) : 0.0F;

            const auto gain = uniforms.ambient + diffuse * uniforms.sun_scale;
            return Vec3f{.x = albedo.x * gain, .y = albedo.y * gain, .z = albedo.z * gain};
        }

        [[nodiscard]]
        Vec3f shade_pbr(
            const pbr::Material& resolved, Vec3f position, Vec3f normal, bool lit
        ) const noexcept {
            const auto n = normalize(normal);
            const auto view = normalize(uniforms.eye_position - position);
            const auto direct = pbr::evaluate_direct(
                resolved,
                n,
                view,
                uniforms.sun_direction,
                Vec3f{
                    .x = uniforms.sun_scale,
                    .y = uniforms.sun_scale,
                    .z = uniforms.sun_scale,
                },
                lit ? 1.0F : 0.0F
            );
            // Deliberately non-physical linear fill remains independent of
            // shadow visibility; it is not part of the direct BRDF.
            const auto fill = uniforms.ambient * (1.0F - pbr::saturate(resolved.metallic));
            return Vec3f{
                .x = direct.x + resolved.base_color.x * fill,
                .y = direct.y + resolved.base_color.y * fill,
                .z = direct.z + resolved.base_color.z * fill,
            };
        }

        // Shadow test for one fragment: project the interpolated light
        // clip position, fetch the nearest occluder depth, compare with a
        // slope-scaled bias. Outside the shadow frustum counts as lit.
        [[nodiscard]]
        bool shadow_lit(Vec4f light_clip, Vec3f normal) const noexcept {
            if (uniforms.shadow_map == nullptr || light_clip.w <= 0.0F) {
                return true;
            }

            const auto inv_w = 1.0F / light_clip.w;
            const auto ndc_x = light_clip.x * inv_w;
            const auto ndc_y = light_clip.y * inv_w;
            const auto ndc_z = light_clip.z * inv_w;
            // Defensive: NaN would make the float-to-size_t casts below
            // undefined; treat as outside the frustum instead.
            if (!std::isfinite(ndc_x) || !std::isfinite(ndc_y) || !std::isfinite(ndc_z)) {
                return true;
            }
            if (ndc_x
                < -1.0F
                || ndc_x
                > 1.0F
                || ndc_y
                < -1.0F
                || ndc_y
                > 1.0F
                || ndc_z
                < 0.0F
                || ndc_z
                > 1.0F) {
                return true;
            }

            // Same mapping the rasterizer used to write the map.
            const auto sample_at =
                viewport_to_screen(ndc_x, ndc_y, ShadowMap::k_size, ShadowMap::k_size);
            if (sample_at.x
                < 0.0F
                || sample_at.y
                < 0.0F
                || sample_at.x
                >= static_cast<float>(ShadowMap::k_size)
                || sample_at.y
                >= static_cast<float>(ShadowMap::k_size)) {
                return true;
            }

            const auto ix = static_cast<std::size_t>(sample_at.x);
            const auto iy = static_cast<std::size_t>(sample_at.y);
            const auto texel = (*uniforms.shadow_map).depth[iy * ShadowMap::k_size + ix];
            const auto slope =
                1.0F - std::max(dot(normalize(normal), uniforms.sun_direction), 0.0F);

            const auto bias = uniforms.shadow_bias + uniforms.shadow_slope * slope;
            return ndc_z + bias >= texel;
        }

        [[nodiscard]]
        constexpr Vec4f fragment(const Varyings& varying) const noexcept {
            const auto use_pbr = uniforms.shading == SceneShading::Pbr;
            const auto texel = [&] -> uint32_t {
                if (uniforms.solid_color_debug) {
                    return 0xFFB0B0B0U;
                }
                if (uniforms.texture == nullptr) {
                    const auto u =
                        static_cast<unsigned int>(static_cast<int>(std::floor(varying.uv.x)));
                    const auto v =
                        static_cast<unsigned int>(static_cast<int>(std::floor(varying.uv.y)));
                    return ((u + v) & 1U) == 0U ? 0xFFCCCCCCU : 0xFF555555U;
                }
                if (!use_pbr) {
                    return uniforms.texture
                        ->sample_trilinear(varying.uv.x, varying.uv.y, varying.mip_lod);
                }
                return 0U;
            }();

            const bool textured = !uniforms.solid_color_debug && uniforms.texture != nullptr;
            const auto sampled = textured && uniforms.shading == SceneShading::Pbr
                ? uniforms.texture
                      ->sample_trilinear_linear(varying.uv.x, varying.uv.y, varying.mip_lod)
                : Vec4f{
                      .x = static_cast<float>((texel >> 16U) & 0xFFU) / 255.0F,
                      .y = static_cast<float>((texel >> 8U) & 0xFFU) / 255.0F,
                      .z = static_cast<float>(texel & 0xFFU) / 255.0F,
                      .w = 1.0F,
                  };

            const Vec3f texture_color{.x = sampled.x, .y = sampled.y, .z = sampled.z};
            Vec4f mr_sample{.x = 0.0F, .y = 1.0F, .z = 1.0F, .w = 1.0F};
            if (textured && use_pbr && uniforms.metallic_roughness_texture != nullptr) {
                mr_sample = uniforms.metallic_roughness_texture->sample_trilinear_linear(
                    varying.uv.x,
                    varying.uv.y,
                    varying.metallic_roughness_lod
                );
            }

            const auto resolved = pbr::resolve_textured_material(
                uniforms.material,
                texture_color,
                Vec3f{.x = mr_sample.x, .y = mr_sample.y, .z = mr_sample.z},
                uniforms.metallic_roughness_texture != nullptr
            );

            // Solid and checker diagnostics stay unlit; only the textured
            // path goes through the sun and the shadow map.
            const bool lit = textured && shadow_lit(varying.light_clip, varying.normal);
            Vec3f shaded = texture_color;
            if (textured) {
                if (use_pbr) {
                    shaded = shade_pbr(resolved, varying.position, varying.normal, lit);
                } else {
                    const Vec3f albedo{
                        .x = texture_color.x * uniforms.material.base_color.x,
                        .y = texture_color.y * uniforms.material.base_color.y,
                        .z = texture_color.z * uniforms.material.base_color.z,
                    };
                    shaded = shade_lit(albedo, varying.normal, lit);
                }
            }

            const bool pbr_output = textured && uniforms.shading == SceneShading::Pbr;
            return Vec4f{
                .x = pbr_output ? linear_to_srgb(shaded.x) : std::clamp(shaded.x, 0.0F, 1.0F),
                .y = pbr_output ? linear_to_srgb(shaded.y) : std::clamp(shaded.y, 0.0F, 1.0F),
                .z = pbr_output ? linear_to_srgb(shaded.z) : std::clamp(shaded.z, 0.0F, 1.0F),
                .w = 1.0F,
            };
        }
    };

    // Depth-only shader for the shadow-map pass: the rasterizer consumes
    // clip positions (screen setup plus reverse-Z depth) and nothing else,
    // so the second model-view multiply and every interpolated attribute of
    // FloorShader are dead weight here. Single-matvec vertex plus empty
    // attributes keeps interpolate_edge to a clip_pos lerp.
    struct DepthShader final : ShaderBase {
        struct Uniforms {
            Mat4f model_view_projection{};
        };

        Uniforms uniforms{};

        struct Varyings {
            Vec4f clip_pos{};

            [[nodiscard]]
            constexpr auto attributes(this auto&&) noexcept {
                return std::tie();
            }
        };

        [[nodiscard]]
        constexpr Varyings vertex(Vec3f position, Vec2f, Vec3f) const noexcept {
            const Vec4f world{
                .x = position.x,
                .y = position.y,
                .z = position.z,
                .w = 1.0F,
            };
            return Varyings{.clip_pos = uniforms.model_view_projection * world};
        }

        // Never invoked: the shadow pass rasterizes depth with a no-op
        // fragment. Present only to satisfy ShaderProgram.
        [[nodiscard]]
        constexpr Vec4f fragment(const Varyings&) const noexcept {
            return Vec4f{};
        }
    };

    static_assert(ShaderProgram<FloorShader<LayoutTiled4x4>>);
    static_assert(ShaderProgram<DepthShader>);
} // namespace raster::render::detail
