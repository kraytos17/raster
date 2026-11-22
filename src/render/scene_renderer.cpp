#include "raster/render/scene_renderer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <numbers>
#include <print>
#include <ranges>
#include <SDL3/SDL.h>
#include <span>
#include <tuple>
#include <vector>

#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/memory/texture.hpp"
#include "raster/pipeline/clipper.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/shader.hpp"
#include "raster/pipeline/simd_quad.hpp"
#include "raster/pipeline/tile_binner.hpp"
#include "raster/pipeline/tile_executor.hpp"

namespace raster::render {
    namespace {
        using std::chrono::steady_clock;

        struct Rgb {
            float r{};
            float g{};
            float b{};
        };

        [[nodiscard]]
        std::uint32_t to_argb(const Rgb& rgb) noexcept {
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

        static_assert(ShaderProgram<FloorShader<LayoutTiled4x4>>);

        // Overdraw probe accumulators.
        inline std::atomic_uint64_t profile_covered_lanes{0};
        inline std::atomic_uint64_t profile_visible_lanes{0};

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
                        mr_sample =
                            shader.uniforms.metallic_roughness_texture
                                ->sample_trilinear_linear(quad_uv[lane].x, quad_uv[lane].y, mr_lod);
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

        void draw_overlay_triangle(
            Framebuffer& framebuffer,
            const ScreenPoint& a,
            const ScreenPoint& b,
            const ScreenPoint& c,
            const Rgb& color_a,
            const Rgb& color_b,
            const Rgb& color_c
        ) noexcept {
            const auto triangle = make_screen_triangle(
                a,
                b,
                c,
                std::array<float, 3>{1.0F, 1.0F, 1.0F},
                std::array<float, 3>{2.0F, 2.0F, 2.0F}
            );
            if (!triangle) {
                return;
            }

            auto color = framebuffer.color_view();
            rasterize_triangle(
                *triangle,
                framebuffer.width,
                framebuffer.height,
                framebuffer.depth_view(),
                [&](std::size_t x, std::size_t y, const PixelBarycentrics& barycentrics) noexcept {
                const Rgb shaded{
                    .r = color_a.r
                        * barycentrics.l0
                        + color_b.r
                        * barycentrics.l1
                        + color_c.r
                        * barycentrics.l2,
                    .g = color_a.g
                        * barycentrics.l0
                        + color_b.g
                        * barycentrics.l1
                        + color_c.g
                        * barycentrics.l2,
                    .b = color_a.b
                        * barycentrics.l0
                        + color_b.b
                        * barycentrics.l1
                        + color_c.b
                        * barycentrics.l2,
                };
                color[y, x] = to_argb(shaded);
            }
            );
        }

        template <typename Layout> void draw_mesh_triangle(
            Framebuffer& framebuffer,
            FloorShader<Layout>& shader,
            const typename FloorShader<Layout>::Varyings& varying_a,
            const typename FloorShader<Layout>::Varyings& varying_b,
            const typename FloorShader<Layout>::Varyings& varying_c
        ) noexcept {
            const std::array<float, 3> inverse_w{
                1.0F / varying_a.clip_pos.w,
                1.0F / varying_b.clip_pos.w,
                1.0F / varying_c.clip_pos.w,
            };
            const std::array<float, 3> reverse_z{
                varying_a.clip_pos.z * inverse_w[0],
                varying_b.clip_pos.z * inverse_w[1],
                varying_c.clip_pos.z * inverse_w[2],
            };

            const auto width = framebuffer.width;
            const auto height = framebuffer.height;
            const std::array<ScreenPoint, 3> screen{
                viewport_to_screen(
                    varying_a.clip_pos.x * inverse_w[0],
                    varying_a.clip_pos.y * inverse_w[0],
                    width,
                    height
                ),
                viewport_to_screen(
                    varying_b.clip_pos.x * inverse_w[1],
                    varying_b.clip_pos.y * inverse_w[1],
                    width,
                    height
                ),
                viewport_to_screen(
                    varying_c.clip_pos.x * inverse_w[2],
                    varying_c.clip_pos.y * inverse_w[2],
                    width,
                    height
                ),
            };

            const auto triangle =
                make_screen_triangle(screen[0], screen[1], screen[2], inverse_w, reverse_z);
            if (!triangle) {
                return;
            }

            auto color = framebuffer.color_view();
            rasterize_triangle_quads(
                *triangle,
                width,
                height,
                framebuffer.depth_view(),
                [&](std::size_t quad_x, std::size_t quad_y, const QuadBarycentrics& quad) noexcept {
                std::array<Vec2f, 4> quad_uv{};
                std::array<Vec3f, 4> quad_normal{};
                std::array<Vec3f, 4> quad_position{};
                std::array<Vec4f, 4> quad_light{};
                for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    const auto& barycentrics = quad.lanes[lane];
                    const auto attributes =
                        shader.interpolate(varying_a, varying_b, varying_c, barycentrics);

                    quad_uv[lane] = std::get<0>(attributes);
                    quad_normal[lane] = std::get<1>(attributes);
                    quad_position[lane] = std::get<2>(attributes);
                    quad_light[lane] = std::get<3>(attributes);
                }

                const auto derivatives = quad_derivatives(quad_uv);
                const auto lod = mip_lod(
                    derivatives,
                    shader.uniforms.texture->width(),
                    shader.uniforms.texture->height()
                );

                const auto mr_lod = shader.uniforms.metallic_roughness_texture != nullptr
                    ? mip_lod(
                          derivatives,
                          shader.uniforms.metallic_roughness_texture->width(),
                          shader.uniforms.metallic_roughness_texture->height()
                      )
                    : lod;
                store_shaded_quad(
                    color,
                    shader,
                    varying_a,
                    quad_x,
                    quad_y,
                    quad,
                    quad_uv,
                    quad_normal,
                    quad_position,
                    quad_light,
                    lod,
                    mr_lod
                );
            }
            );
        }

        template <typename Layout> [[nodiscard]]
        bool draw_mesh(
            Framebuffer& framebuffer,
            FloorShader<Layout>& shader,
            const SoAMesh& mesh,
            bool guard_band_test
        ) noexcept {
            for (const auto triangle_index :
                 std::views::iota(std::size_t{0}, mesh.triangle_count())) {
                const auto index_base = triangle_index * 3U;
                const auto vertex_a = static_cast<std::size_t>(mesh.indices[index_base]);
                const auto vertex_b = static_cast<std::size_t>(mesh.indices[index_base + 1U]);
                const auto vertex_c = static_cast<std::size_t>(mesh.indices[index_base + 2U]);
                const auto varying_a = shader.vertex(
                    mesh.positions[vertex_a],
                    mesh.uvs[vertex_a],
                    mesh.normals[vertex_a]
                );
                const auto varying_b = shader.vertex(
                    mesh.positions[vertex_b],
                    mesh.uvs[vertex_b],
                    mesh.normals[vertex_b]
                );
                const auto varying_c = shader.vertex(
                    mesh.positions[vertex_c],
                    mesh.uvs[vertex_c],
                    mesh.normals[vertex_c]
                );
                const auto clipped = guard_band_test
                    ? clip_triangle_guard_band(shader, varying_a, varying_b, varying_c, 2.0F)
                    : clip_triangle(shader, varying_a, varying_b, varying_c);
                if (!clipped) {
                    return false;
                }
                for (const auto& triangle : std::span{clipped->triangles}.first(clipped->size)) {
                    draw_mesh_triangle(
                        framebuffer,
                        shader,
                        triangle.vertices[0],
                        triangle.vertices[1],
                        triangle.vertices[2]
                    );
                }
            }
            return true;
        }

        template <typename Layout> struct PreparedTriangle {
            ScreenTriangle setup{};
            std::array<typename FloorShader<Layout>::Varyings, 3> vertices{};
        };

        template <typename Layout> [[nodiscard]]
        bool prepare_mesh(
            FloorShader<Layout>& shader,
            const SoAMesh& mesh,
            std::size_t width,
            std::size_t height,
            bool guard_band_test,
            std::size_t begin_triangle,
            std::size_t end_triangle,
            std::vector<PreparedTriangle<Layout>>& prepared
        ) noexcept {
            prepared.clear();
            for (const auto triangle_index :
                 std::views::iota(begin_triangle, std::min(end_triangle, mesh.triangle_count()))) {
                const auto index_base = triangle_index * 3U;
                const auto vertex_a = static_cast<std::size_t>(mesh.indices[index_base]);
                const auto vertex_b = static_cast<std::size_t>(mesh.indices[index_base + 1U]);
                const auto vertex_c = static_cast<std::size_t>(mesh.indices[index_base + 2U]);
                const auto varying_a = shader.vertex(
                    mesh.positions[vertex_a],
                    mesh.uvs[vertex_a],
                    mesh.normals[vertex_a]
                );
                const auto varying_b = shader.vertex(
                    mesh.positions[vertex_b],
                    mesh.uvs[vertex_b],
                    mesh.normals[vertex_b]
                );
                const auto varying_c = shader.vertex(
                    mesh.positions[vertex_c],
                    mesh.uvs[vertex_c],
                    mesh.normals[vertex_c]
                );
                const auto clipped = guard_band_test
                    ? clip_triangle_guard_band(shader, varying_a, varying_b, varying_c, 2.0F)
                    : clip_triangle(shader, varying_a, varying_b, varying_c);
                if (!clipped) {
                    return false;
                }
                for (const auto& clipped_triangle :
                     std::span{clipped->triangles}.first(clipped->size)) {
                    const std::array<typename FloorShader<Layout>::Varyings, 3> vertices{
                        clipped_triangle.vertices[0],
                        clipped_triangle.vertices[1],
                        clipped_triangle.vertices[2],
                    };
                    const std::array<float, 3> inverse_w{
                        1.0F / vertices[0].clip_pos.w,
                        1.0F / vertices[1].clip_pos.w,
                        1.0F / vertices[2].clip_pos.w,
                    };
                    const std::array<float, 3> reverse_z{
                        vertices[0].clip_pos.z * inverse_w[0],
                        vertices[1].clip_pos.z * inverse_w[1],
                        vertices[2].clip_pos.z * inverse_w[2],
                    };
                    const auto setup = make_screen_triangle(
                        viewport_to_screen(
                            vertices[0].clip_pos.x * inverse_w[0],
                            vertices[0].clip_pos.y * inverse_w[0],
                            width,
                            height
                        ),
                        viewport_to_screen(
                            vertices[1].clip_pos.x * inverse_w[1],
                            vertices[1].clip_pos.y * inverse_w[1],
                            width,
                            height
                        ),
                        viewport_to_screen(
                            vertices[2].clip_pos.x * inverse_w[2],
                            vertices[2].clip_pos.y * inverse_w[2],
                            width,
                            height
                        ),
                        inverse_w,
                        reverse_z
                    );
                    if (setup) {
                        prepared.push_back(
                            PreparedTriangle<Layout>{.setup = *setup, .vertices = vertices}
                        );
                    }
                }
            }
            return true;
        }

        template <typename Layout> void rasterize_prepared_triangle(
            Framebuffer& framebuffer,
            FloorShader<Layout>& shader,
            const PreparedTriangle<Layout>& triangle,
            const RasterRegion& region
        ) noexcept {
            const auto color = framebuffer.color_view();
            const auto varying_a = triangle.vertices[0];
            const auto varying_b = triangle.vertices[1];
            const auto varying_c = triangle.vertices[2];
            rasterize_triangle_quads(
                triangle.setup,
                framebuffer.width,
                framebuffer.height,
                framebuffer.depth_view(),
                [&](std::size_t quad_x, std::size_t quad_y, const QuadBarycentrics& quad) noexcept {
                std::array<Vec2f, 4> quad_uv{};
                std::array<Vec3f, 4> quad_normal{};
                std::array<Vec3f, 4> quad_position{};
                std::array<Vec4f, 4> quad_light{};
                for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    const auto attributes =
                        shader.interpolate(varying_a, varying_b, varying_c, quad.lanes[lane]);

                    quad_uv[lane] = std::get<0>(attributes);
                    quad_normal[lane] = std::get<1>(attributes);
                    quad_position[lane] = std::get<2>(attributes);
                    quad_light[lane] = std::get<3>(attributes);
                }

                const auto derivatives = quad_derivatives(quad_uv);
                const auto lod = mip_lod(
                    derivatives,
                    shader.uniforms.texture->width(),
                    shader.uniforms.texture->height()
                );

                const auto mr_lod = shader.uniforms.metallic_roughness_texture != nullptr
                    ? mip_lod(
                          derivatives,
                          shader.uniforms.metallic_roughness_texture->width(),
                          shader.uniforms.metallic_roughness_texture->height()
                      )
                    : lod;
                store_shaded_quad(
                    color,
                    shader,
                    varying_a,
                    quad_x,
                    quad_y,
                    quad,
                    quad_uv,
                    quad_normal,
                    quad_position,
                    quad_light,
                    lod,
                    mr_lod
                );
            },
                region
            );
        }

        template <typename Layout> struct TileRenderState {
            explicit TileRenderState(std::size_t concurrency) :
                executor{concurrency} {}

            void reserve(
                std::size_t width,
                std::size_t height,
                std::size_t mesh_triangle_count,
                std::size_t worker_count
            ) {
                const auto columns = width
                    / raster_tile_size
                    + static_cast<std::size_t>(width % raster_tile_size != 0U);
                const auto rows = height
                    / raster_tile_size
                    + static_cast<std::size_t>(height % raster_tile_size != 0U);
                const auto tile_count = columns * rows;

                prepared.reserve(mesh_triangle_count * 8U);
                setups.reserve(mesh_triangle_count * 8U);
                shadow_prepared.reserve(mesh_triangle_count * 8U);
                bins.reserve(tile_count, tile_count * 8U);
                thread_prepared.resize(worker_count);
                const auto chunk_reserve =
                    (mesh_triangle_count + worker_count - 1U) / worker_count * 8U + 1U;
                for (auto& chunk : thread_prepared) {
                    chunk.clear();
                    chunk.reserve(chunk_reserve);
                }
            }

            TileExecutor executor;
            TileBins bins{};
            std::vector<PreparedTriangle<Layout>> prepared{};
            std::vector<ScreenTriangle> setups{};
            std::vector<std::vector<PreparedTriangle<Layout>>> thread_prepared{};
            ShadowMap shadow_map{};
            std::vector<PreparedTriangle<Layout>> shadow_prepared{};
        };

        // Depth-only light pass: prepares the mesh in light space and
        // rasterizes every triangle into the shadow depth buffer, reusing the
        // rasterizer's depth test/write. The no-op fragment proves the point:
        // depth happens before shading. Runs serially; parallelize only if
        // the perf gate says the shadow pass dominates (it reuses the same
        // prepare_mesh as the main pass, so no new math to verify).
        template <typename Layout> [[nodiscard]]
        bool render_shadow_map(
            ShadowMap& shadow,
            FloorShader<Layout>& light_shader,
            const SoAMesh& mesh,
            std::vector<PreparedTriangle<Layout>>& scratch
        ) noexcept {
            static const bool profile =
                SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PROFILE") != nullptr;
            const auto timestamp_begin = steady_clock::now();
            if (!prepare_mesh(
                    light_shader,
                    mesh,
                    ShadowMap::k_size,
                    ShadowMap::k_size,
                    false,
                    0U,
                    mesh.triangle_count(),
                    scratch
                )) {
                return false;
            }

            shadow.clear();
            auto depth_view = std::mdspan<float, Extents2D>{
                shadow.depth.data(),
                ShadowMap::k_size,
                ShadowMap::k_size,
            };

            const auto record_nothing =
                [](std::size_t, std::size_t, const QuadBarycentrics&) noexcept {
            };
            for (const auto& triangle : scratch) {
                rasterize_triangle_quads(
                    triangle.setup,
                    ShadowMap::k_size,
                    ShadowMap::k_size,
                    depth_view,
                    record_nothing
                );
            }
            if (profile) {
                using Milliseconds = std::chrono::duration<double, std::milli>;
                try {
                    std::println(
                        stderr,
                        "raster: profile shadow {:.2f} ms tris {}",
                        Milliseconds(steady_clock::now() - timestamp_begin).count(),
                        scratch.size()
                    );
                } catch (...) {
                    try {
                        std::println(stderr, "raster: profile output failed");
                    } catch (...) { //NOLINT
                        // Abandon profiling output rather than terminate.
                    }
                }
            }
            return true;
        }

        template <typename Layout> [[nodiscard]]
        bool draw_tiled_mesh(
            Framebuffer& framebuffer,
            FloorShader<Layout>& shader,
            const SoAMesh& mesh,
            bool guard_band_test,
            TileRenderState<Layout>& tile_state,
            bool allow_allocation
        ) noexcept {
            const auto width = framebuffer.width;
            const auto height = framebuffer.height;
            const auto triangle_count = mesh.triangle_count();
            const auto worker_count = tile_state.executor.concurrency();
            if (allow_allocation) {
                tile_state.reserve(width, height, triangle_count, worker_count);
            }

            std::atomic_bool prepared_ok{true};
            const bool profile =
                SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PROFILE") != nullptr;
            if (profile) {
                profile_covered_lanes.store(0, std::memory_order_relaxed);
                profile_visible_lanes.store(0, std::memory_order_relaxed);
            }

            const auto timestamp_prepare_begin = steady_clock::now();
            tile_state.executor.execute(worker_count, [&](std::size_t worker) noexcept {
                auto& chunk = tile_state.thread_prepared[worker];
                chunk.clear();
                const auto [begin, end] = partition_range(triangle_count, worker_count, worker);
                if (!prepare_mesh(
                        shader,
                        mesh,
                        width,
                        height,
                        guard_band_test,
                        begin,
                        end,
                        chunk
                    )) {
                    prepared_ok.store(false, std::memory_order_relaxed);
                }
            });
            if (!prepared_ok.load(std::memory_order_relaxed)) {
                return false;
            }

            const auto timestamp_merge_begin = steady_clock::now();
            auto& prepared = tile_state.prepared;
            auto& setups = tile_state.setups;

            prepared.clear();
            setups.clear();
            setups.reserve(prepared.capacity());
            for (const auto& chunk : tile_state.thread_prepared) {
                for (const auto& triangle : chunk) {
                    prepared.push_back(triangle);
                    setups.push_back(triangle.setup);
                }
            }

            const auto timestamp_bin_begin = steady_clock::now();
            bin_triangles(tile_state.setups, width, height, tile_state.bins);
            const auto timestamp_raster_begin = steady_clock::now();
            tile_state.executor.execute(
                tile_state.bins.tile_count(),
                [&](std::size_t tile) noexcept {
                const auto tile_x = tile % tile_state.bins.tile_columns();
                const auto tile_y = tile / tile_state.bins.tile_columns();
                const RasterRegion region{
                    .min_x = tile_x * raster_tile_size,
                    .min_y = tile_y * raster_tile_size,
                    .max_x = std::min((tile_x + 1U) * raster_tile_size, width),
                    .max_y = std::min((tile_y + 1U) * raster_tile_size, height),
                };
                for (const auto triangle_index : tile_state.bins.triangle_indices(tile)) {
                    rasterize_prepared_triangle(
                        framebuffer,
                        shader,
                        tile_state.prepared[triangle_index],
                        region
                    );
                }
            }
            );
            if (profile) {
                using Milliseconds = std::chrono::duration<double, std::milli>;
                const auto covered = profile_covered_lanes.load(std::memory_order_relaxed);
                const auto visible = profile_visible_lanes.load(std::memory_order_relaxed);
                try {
                    std::println(
                        stderr,
                        "raster: profile prepare {:.2f} ms merge {:.2f} ms bin {:.2f} ms raster {:.2f} ms",
                        Milliseconds(timestamp_merge_begin - timestamp_prepare_begin).count(),
                        Milliseconds(timestamp_bin_begin - timestamp_merge_begin).count(),
                        Milliseconds(timestamp_raster_begin - timestamp_bin_begin).count(),
                        Milliseconds(std::chrono::steady_clock::now() - timestamp_raster_begin)
                            .count()
                    );
                    std::println(
                        stderr,
                        "raster: profile overdraw covered {} visible {} depth-kill {:.1f}%",
                        covered,
                        visible,
                        covered == 0U ? 0.0
                                      : 100.0
                                * static_cast<double>(covered - visible)
                                / static_cast<double>(covered)
                    );
                } catch (...) {
                    try {
                        std::println(stderr, "raster: profile output failed");
                    } catch (...) { //NOLINT
                    }
                }
            }
            return true;
        }
    } // namespace

    struct TileRenderer::Impl : TileRenderState<LayoutTiled4x4> {
        explicit Impl(std::size_t concurrency) :
            TileRenderState<LayoutTiled4x4>{concurrency} {}
    };

    template <typename Layout> [[nodiscard]]
    bool render_scene_impl(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        bool clip_stress_test,
        bool free_look_test,
        bool guard_band_test,
        bool solid_color_debug,
        const Texture2D<Layout>& texture,
        const FreeLookCamera& camera,
        SceneShading shading,
        const pbr::Material& material,
        bool draw_overlay,
        TileRenderState<Layout>* tile_state = nullptr,
        const Texture2D<Layout>* metallic_roughness = nullptr
    ) noexcept {
        const auto width = framebuffer.width;
        const auto height = framebuffer.height;
        if (width < 2U || height < 2U) {
            return true;
        }
        if (shading
            == SceneShading::Pbr
            && (texture.level_count() == 0U || texture.color_space() != TextureColorSpace::Srgb)) {
            return false;
        }
        if (shading
            == SceneShading::Pbr
            && metallic_roughness
            != nullptr
            && (metallic_roughness->level_count()
                == 0U
                || metallic_roughness->color_space()
                != TextureColorSpace::EncodedBytes)) {
            return false;
        }

        simd::stream_fill_u32(framebuffer.color_storage, 0xFF101018U);
        std::ranges::fill(framebuffer.depth_storage, 0.0F);
        const auto seconds = static_cast<float>(static_cast<double>(elapsed_ns) * 1e-9);
        const auto stress_phase = std::fmod(seconds * 0.5F, 2.0F);
        const auto sweep = stress_phase <= 1.0F ? stress_phase : 2.0F - stress_phase;
        const auto eye = [&] -> Vec3f {
            if (clip_stress_test) {
                return {.x = -3.0F + 6.0F * sweep, .y = 0.0F, .z = 0.0F};
            }
            if (free_look_test) {
                return camera.position;
            }
            return {
                .x = 0.05F * std::sin(seconds * 0.3F),
                .y = 0.1F,
                .z = 0.05F * std::cos(seconds * 0.3F),
            };
        }();
        const auto view = [&] {
            if (free_look_test) {
                const auto cosine_pitch = std::cos(camera.pitch);
                const Vec3f direction{
                    .x = std::sin(camera.yaw) * cosine_pitch,
                    .y = std::sin(camera.pitch),
                    .z = -std::cos(camera.yaw) * cosine_pitch,
                };
                const auto camera_up = std::fabs(direction.y) > 0.99F
                    ? Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F}
                    : Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F};
                return look_at4f(eye, eye + direction, camera_up);
            }
            return look_at4f(
                eye,
                Vec3f{
                    .x = eye.x,
                    .y = clip_stress_test ? eye.y : eye.y - 0.0001F,
                    .z = eye.z - 1.0F,
                },
                Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
            );
        }();
        const auto projection = perspective_reverse_z4f(
            1.2566371F,
            static_cast<float>(width) / static_cast<float>(height),
            clip_stress_test || free_look_test ? 0.005F : 0.1F,
            // Keep the free-look far plane well beyond the sphere: reverse-Z
            // maps the far boundary to exactly zero, equal to the depth clear,
            // so a surface at that boundary would fail the strict `>` test.
            clip_stress_test ? 10.0F : 100.0F
        );

        FloorShader<Layout> shader{};
        shader.uniforms.model_view_projection = projection * view;
        shader.uniforms.texture = &texture;
        shader.uniforms.metallic_roughness_texture = metallic_roughness;
        shader.uniforms.solid_color_debug = solid_color_debug;
        shader.uniforms.shading = shading;
        shader.uniforms.material = material;
        shader.uniforms.eye_position = eye;
        // Direction points from a shaded point toward the directional light.
        shader.uniforms.sun_direction = normalize(Vec3f{.x = 0.0F, .y = 3.0F, .z = 4.0F});
        shader.uniforms.ambient = 0.4F;
        shader.uniforms.sun_scale = 0.6F;
        shader.uniforms.shadow_bias = k_shadow_bias;
        shader.uniforms.shadow_slope = k_shadow_slope;

        const auto light_view = look_at4f(
            Vec3f{
                .x = shader.uniforms.sun_direction.x * k_shadow_eye_distance,
                .y = shader.uniforms.sun_direction.y * k_shadow_eye_distance,
                .z = shader.uniforms.sun_direction.z * k_shadow_eye_distance,
            },
            Vec3f{.x = 0.0F, .y = 0.0F, .z = 0.0F},
            Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto light_projection = ortho_reverse_z4f(
            -k_shadow_half_extent,
            k_shadow_half_extent,
            -k_shadow_half_extent,
            k_shadow_half_extent,
            k_shadow_near,
            k_shadow_far
        );

        const auto light_view_projection = light_projection * light_view;
        shader.uniforms.light_view_projection = light_view_projection;

        FloorShader<Layout> light_shader{};
        light_shader.uniforms.model_view_projection = light_view_projection;
        light_shader.uniforms.light_view_projection = light_view_projection;
        const bool shadows_enabled =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_NO_SHADOW") == nullptr;
        if (tile_state != nullptr) {
            shader.uniforms.shadow_map = shadows_enabled ? &tile_state->shadow_map : nullptr;
            if (shadows_enabled
                && !render_shadow_map(
                    tile_state->shadow_map,
                    light_shader,
                    mesh,
                    tile_state->shadow_prepared
                )) {
                return false;
            }
            if (!draw_tiled_mesh(
                    framebuffer,
                    shader,
                    mesh,
                    guard_band_test,
                    *tile_state,
                    !clip_stress_test || elapsed_ns == 0U
                )) {
                return false;
            }
        } else {
            thread_local ShadowMap serial_shadow{};
            thread_local std::vector<PreparedTriangle<Layout>> serial_shadow_prepared{};
            shader.uniforms.shadow_map = shadows_enabled ? &serial_shadow : nullptr;
            if (shadows_enabled
                && !render_shadow_map(serial_shadow, light_shader, mesh, serial_shadow_prepared)) {
                return false;
            }
            if (!draw_mesh(framebuffer, shader, mesh, guard_band_test)) {
                return false;
            }
        }
        if (!draw_overlay || clip_stress_test || free_look_test) {
            return true;
        }

        const auto theta = static_cast<float>(static_cast<double>(elapsed_ns) * 1.5e-9);
        const auto center_x = static_cast<float>(width) * 0.5F;
        const auto center_y = static_cast<float>(height) * 0.5F;
        const auto radius = 0.35F * static_cast<float>(std::min(width, height));
        const Rgb red{.r = 1.0F, .g = 0.0F, .b = 0.0F};
        const Rgb green{.r = 0.0F, .g = 1.0F, .b = 0.0F};
        const Rgb blue{.r = 0.0F, .g = 0.0F, .b = 1.0F};
        std::array<ScreenPoint, 3> spinner{};
        for (const auto vertex : std::views::iota(std::size_t{0}, std::size_t{3})) {
            const auto angle =
                theta + static_cast<float>(vertex) * (2.0F * std::numbers::pi_v<float> / 3.0F);
            spinner[vertex] = ScreenPoint{
                .x = center_x + radius * std::cos(angle),
                .y = center_y + radius * std::sin(angle),
            };
        }

        // Counter-clockwise as seen in NDC (y-up): the front-facing order.
        draw_overlay_triangle(framebuffer, spinner[0], spinner[2], spinner[1], red, blue, green);
        return true;
    }

    bool render_scene(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        bool clip_stress_test,
        bool free_look_test,
        bool guard_band_test,
        bool solid_color_debug,
        const Texture2D<LayoutTiled4x4>& texture,
        const FreeLookCamera& camera,
        SceneShading shading,
        const pbr::Material& material,
        const Texture2D<LayoutTiled4x4>* metallic_roughness
    ) noexcept {
        return render_scene_impl<LayoutTiled4x4>(
            framebuffer,
            mesh,
            elapsed_ns,
            clip_stress_test,
            free_look_test,
            guard_band_test,
            solid_color_debug,
            texture,
            camera,
            shading,
            material,
            true,
            nullptr,
            metallic_roughness
        );
    }

    TileRenderer::TileRenderer(std::size_t concurrency) :
        impl_{std::make_unique<Impl>(concurrency)} {}

    TileRenderer::TileRenderer(TileRenderer&&) noexcept = default;
    TileRenderer& TileRenderer::operator=(TileRenderer&&) noexcept = default;
    TileRenderer::~TileRenderer() = default;

    bool TileRenderer::render_scene(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        bool clip_stress_test,
        bool free_look_test,
        bool guard_band_test,
        bool solid_color_debug,
        const Texture2D<LayoutTiled4x4>& texture,
        const FreeLookCamera& camera,
        SceneShading shading,
        const pbr::Material& material,
        const Texture2D<LayoutTiled4x4>* metallic_roughness
    ) noexcept {
        return render_scene_impl(
            framebuffer,
            mesh,
            elapsed_ns,
            clip_stress_test,
            free_look_test,
            guard_band_test,
            solid_color_debug,
            texture,
            camera,
            shading,
            material,
            true,
            impl_.get(),
            metallic_roughness
        );
    }

    bool render_textured_model(
        Framebuffer& framebuffer, const SoAMesh& mesh, const Texture2D<std::layout_right>& texture
    ) noexcept {
        constexpr FreeLookCamera camera{
            .position = Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        return render_scene_impl(
            framebuffer,
            mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            SceneShading::Diffuse,
            pbr::Material(),
            false
        );
    }

    bool render_textured_model(
        Framebuffer& framebuffer, const SoAMesh& mesh, const Texture2D<LayoutTiled4x4>& texture
    ) noexcept {
        constexpr FreeLookCamera camera{
            .position = Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        return render_scene_impl(
            framebuffer,
            mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            SceneShading::Diffuse,
            pbr::Material(),
            false
        );
    }

    bool render_textured_model(
        Framebuffer& framebuffer, const SoAMesh& mesh, const Texture2D<LayoutMorton>& texture
    ) noexcept {
        constexpr FreeLookCamera camera{
            .position = Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        return render_scene_impl(
            framebuffer,
            mesh,
            0U,
            false,
            true,
            false,
            false,
            texture,
            camera,
            SceneShading::Diffuse,
            pbr::Material(),
            false
        );
    }
} // namespace raster::render
