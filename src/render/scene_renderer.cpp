#include "raster/render/scene_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <ranges>
#include <span>
#include <tuple>

#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/pipeline/clipper.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/shader.hpp"

namespace raster::render {
    namespace {
        struct Rgb {
            float r{};
            float g{};
            float b{};
        };

        [[nodiscard]]
        std::uint32_t to_argb(const Rgb& rgb) noexcept {
            const auto clamp_channel = [](float channel) noexcept {
                return std::clamp(channel, 0.0F, 1.0F);
            };

            const auto red = static_cast<std::uint32_t>(std::lround(clamp_channel(rgb.r) * 255.0F));
            const auto green =
                static_cast<std::uint32_t>(std::lround(clamp_channel(rgb.g) * 255.0F));
            const auto blue =
                static_cast<std::uint32_t>(std::lround(clamp_channel(rgb.b) * 255.0F));
            return 0xFF000000U | (red << 16U) | (green << 8U) | blue;
        }

        struct FloorShader final : ShaderBase {
            struct Uniforms {
                Mat4f model_view_projection{};
            };

            Uniforms uniforms{};

            struct Varyings {
                Vec4f clip_pos{};
                Vec2f uv{};

                [[nodiscard]]
                constexpr auto attributes(this auto&& self) noexcept {
                    return std::tie(self.uv);
                }
            };

            [[nodiscard]]
            constexpr Varyings vertex(Vec3f position, Vec2f uv) const noexcept {
                return Varyings{
                    .clip_pos = uniforms.model_view_projection
                        * Vec4f{
                            .x = position.x,
                            .y = position.y,
                            .z = position.z,
                            .w = 1.0F,
                        },
                    .uv = uv,
                };
            }

            [[nodiscard]]
            constexpr Vec4f fragment(const Varyings& varying) const noexcept {
                const auto cell = static_cast<int>(std::floor(varying.uv.x))
                    + static_cast<int>(std::floor(varying.uv.y));
                const auto parity = ((cell % 2) + 2) % 2;
                const auto shade = parity == 0 ? 0.85F : 0.35F;
                return Vec4f{.x = shade, .y = shade, .z = shade, .w = 1.0F};
            }
        };

        static_assert(ShaderProgram<FloorShader>);

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

        void draw_mesh_triangle(
            Framebuffer& framebuffer,
            FloorShader& shader,
            const FloorShader::Varyings& varying_a,
            const FloorShader::Varyings& varying_b,
            const FloorShader::Varyings& varying_c
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
            rasterize_triangle(
                *triangle,
                width,
                height,
                framebuffer.depth_view(),
                [&](std::size_t x, std::size_t y, const PixelBarycentrics& barycentrics) noexcept {
                const auto attributes =
                    shader.interpolate(varying_a, varying_b, varying_c, barycentrics);
                const auto shaded = shader.fragment(
                    FloorShader::Varyings{
                        .clip_pos = varying_a.clip_pos,
                        .uv = std::get<0>(attributes),
                    }
                );
                color[y, x] = to_argb(Rgb{.r = shaded.x, .g = shaded.y, .b = shaded.z});
            }
            );
        }

        [[nodiscard]]
        bool draw_mesh(
            Framebuffer& framebuffer, FloorShader& shader, const SoAMesh& mesh
        ) noexcept {
            for (const auto triangle_index :
                 std::views::iota(std::size_t{0}, mesh.triangle_count())) {
                const auto index_base = triangle_index * 3U;
                const auto vertex_a = static_cast<std::size_t>(mesh.indices[index_base]);
                const auto vertex_b = static_cast<std::size_t>(mesh.indices[index_base + 1U]);
                const auto vertex_c = static_cast<std::size_t>(mesh.indices[index_base + 2U]);
                const auto varying_a = shader.vertex(mesh.positions[vertex_a], mesh.uvs[vertex_a]);
                const auto varying_b = shader.vertex(mesh.positions[vertex_b], mesh.uvs[vertex_b]);
                const auto varying_c = shader.vertex(mesh.positions[vertex_c], mesh.uvs[vertex_c]);
                const auto clipped = clip_triangle(shader, varying_a, varying_b, varying_c);
                if (!clipped) {
                    return false;
                }
                for (const auto& triangle : std::span{clipped->triangles.data(), clipped->size}) {
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
    } // namespace

    bool render_scene(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        bool clip_stress_test
    ) noexcept {
        const auto width = framebuffer.width;
        const auto height = framebuffer.height;
        if (width < 2U || height < 2U) {
            return true;
        }

        std::ranges::fill(framebuffer.color_storage, 0xFF101018U);
        std::ranges::fill(framebuffer.depth_storage, 0.0F);
        const auto seconds = static_cast<float>(static_cast<double>(elapsed_ns) * 1e-9);
        const auto stress_phase = std::fmod(seconds * 0.5F, 2.0F);
        const auto sweep = stress_phase <= 1.0F ? stress_phase : 2.0F - stress_phase;
        const auto eye = clip_stress_test ? Vec3f{.x = -3.0F + 6.0F * sweep, .y = 0.0F, .z = 0.0F}
                                          : Vec3f{
                                                .x = 0.05F * std::sin(seconds * 0.3F),
                                                .y = 0.1F,
                                                .z = 0.05F * std::cos(seconds * 0.3F),
                                            };
        const auto view = look_at4f(
            eye,
            Vec3f{
                .x = eye.x,
                .y = clip_stress_test ? eye.y : eye.y - 0.0001F,
                .z = eye.z - 1.0F,
            },
            Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto projection = perspective_reverse_z4f(
            1.2566371F,
            static_cast<float>(width) / static_cast<float>(height),
            clip_stress_test ? 0.05F : 0.1F,
            clip_stress_test ? 10.0F : 1000.0F
        );

        FloorShader shader{};
        shader.uniforms.model_view_projection = projection * view;
        if (!draw_mesh(framebuffer, shader, mesh)) {
            return false;
        }
        if (clip_stress_test) {
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

        draw_overlay_triangle(framebuffer, spinner[0], spinner[1], spinner[2], red, green, blue);
        return true;
    }
} // namespace raster::render
