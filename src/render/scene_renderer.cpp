#include "raster/render/scene_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <ranges>
#include <SDL3/SDL.h>
#include <span>
#include <vector>

#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/memory/texture.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/render/detail/scene_common.hpp"
#include "raster/render/detail/scene_mesh.hpp"
#include "raster/render/detail/scene_shader.hpp"
#include "raster/render/detail/scene_shadow.hpp"
#include "raster/render/detail/scene_tiled.hpp"

namespace raster::render {
    namespace {
        void draw_overlay_triangle(
            Framebuffer& framebuffer,
            const ScreenPoint& a,
            const ScreenPoint& b,
            const ScreenPoint& c,
            const detail::Rgb& color_a,
            const detail::Rgb& color_b,
            const detail::Rgb& color_c
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
                const detail::Rgb shaded{
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
                color[y, x] = detail::to_argb(shaded);
            }
            );
        }
    } // namespace

    struct TileRenderer::Impl : detail::TileRenderState<LayoutTiled4x4> {
        explicit Impl(std::size_t concurrency) :
            detail::TileRenderState<LayoutTiled4x4>{concurrency} {}
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
        detail::TileRenderState<Layout>* tile_state = nullptr,
        const Texture2D<Layout>* metallic_roughness = nullptr,
        SceneRenderDiagnostics* diagnostics = nullptr,
        std::span<const SceneChunk> chunks = {},
        const Vec3f* lod_reference = nullptr
    ) noexcept {
        if (diagnostics != nullptr) {
            diagnostics->reset();
        }

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
        if (!valid_scene_chunks(chunks, mesh.triangle_count())) {
            return false;
        }
        if (lod_reference
            != nullptr
            && (!std::isfinite(lod_reference->x)
                || !std::isfinite(lod_reference->y)
                || !std::isfinite(lod_reference->z))) {
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

        detail::FloorShader<Layout> shader{};
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
        shader.uniforms.shadow_bias = detail::k_shadow_bias;
        shader.uniforms.shadow_slope = detail::k_shadow_slope;

        const auto chunk_lod_reference = lod_reference == nullptr ? eye : *lod_reference;
        const auto light_view = look_at4f(
            Vec3f{
                .x = shader.uniforms.sun_direction.x * detail::k_shadow_eye_distance,
                .y = shader.uniforms.sun_direction.y * detail::k_shadow_eye_distance,
                .z = shader.uniforms.sun_direction.z * detail::k_shadow_eye_distance,
            },
            Vec3f{.x = 0.0F, .y = 0.0F, .z = 0.0F},
            Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto light_projection = ortho_reverse_z4f(
            -detail::k_shadow_half_extent,
            detail::k_shadow_half_extent,
            -detail::k_shadow_half_extent,
            detail::k_shadow_half_extent,
            detail::k_shadow_near,
            detail::k_shadow_far
        );

        const auto light_view_projection = light_projection * light_view;
        shader.uniforms.light_view_projection = light_view_projection;

        detail::FloorShader<Layout> light_shader{};
        light_shader.uniforms.model_view_projection = light_view_projection;
        light_shader.uniforms.light_view_projection = light_view_projection;
        light_shader.uniforms.eye_position = eye;
        const bool shadows_enabled =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_NO_SHADOW") == nullptr;
        if (tile_state != nullptr) {
            shader.uniforms.shadow_map = shadows_enabled ? &tile_state->shadow_map : nullptr;
            // Keep diagnostics scoped to the visible camera pass; the shadow
            // pass uses a different projection and its own preparation path.
            if (shadows_enabled
                && !detail::render_shadow_map(
                    tile_state->shadow_map,
                    light_shader,
                    mesh,
                    tile_state->shadow_prepared,
                    nullptr
                )) {
                return false;
            }
            if (!detail::draw_tiled_mesh(
                    framebuffer,
                    shader,
                    mesh,
                    guard_band_test,
                    *tile_state,
                    !clip_stress_test || elapsed_ns == 0U,
                    diagnostics,
                    chunks,
                    chunk_lod_reference
                )) {
                return false;
            }
        } else {
            thread_local detail::ShadowMap serial_shadow{};
            thread_local std::vector<detail::PreparedTriangle<Layout>> serial_shadow_prepared{};
            shader.uniforms.shadow_map = shadows_enabled ? &serial_shadow : nullptr;
            if (shadows_enabled
                && !detail::render_shadow_map(
                    serial_shadow,
                    light_shader,
                    mesh,
                    serial_shadow_prepared,
                    nullptr
                )) {
                return false;
            }
            if (!detail::draw_mesh(
                    framebuffer,
                    shader,
                    mesh,
                    guard_band_test,
                    diagnostics,
                    chunks,
                    chunk_lod_reference
                )) {
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
        const detail::Rgb red{.r = 1.0F, .g = 0.0F, .b = 0.0F};
        const detail::Rgb green{.r = 0.0F, .g = 1.0F, .b = 0.0F};
        const detail::Rgb blue{.r = 0.0F, .g = 0.0F, .b = 1.0F};

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
        const Texture2D<LayoutTiled4x4>* metallic_roughness,
        SceneRenderDiagnostics* diagnostics,
        std::span<const SceneChunk> chunks,
        const Vec3f* lod_reference
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
            metallic_roughness,
            diagnostics,
            chunks,
            lod_reference
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
        const Texture2D<LayoutTiled4x4>* metallic_roughness,
        SceneRenderDiagnostics* diagnostics,
        std::span<const SceneChunk> chunks,
        const Vec3f* lod_reference
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
            metallic_roughness,
            diagnostics,
            chunks,
            lod_reference
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
