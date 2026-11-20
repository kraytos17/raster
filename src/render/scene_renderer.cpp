#include "raster/render/scene_renderer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <ranges>
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

        template <typename Layout> struct FloorShader final : ShaderBase {
            struct Uniforms {
                Mat4f model_view_projection{};
                const Texture2D<Layout>* texture{};
                bool solid_color_debug{};
            };

            Uniforms uniforms{};

            struct Varyings {
                Vec4f clip_pos{};
                Vec2f uv{};
                float mip_lod{};

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
                const auto texel = uniforms.solid_color_debug ? 0xFFB0B0B0U
                    : uniforms.texture == nullptr
                    ? ((static_cast<int>(std::floor(varying.uv.x))
                        + static_cast<int>(std::floor(varying.uv.y)))
                               % 2
                               == 0
                           ? 0xFFCCCCCCU
                           : 0xFF555555U)
                    : uniforms.texture
                          ->sample_trilinear(varying.uv.x, varying.uv.y, varying.mip_lod);
                return Vec4f{
                    .x = static_cast<float>((texel >> 16U) & 0xFFU) / 255.0F,
                    .y = static_cast<float>((texel >> 8U) & 0xFFU) / 255.0F,
                    .z = static_cast<float>(texel & 0xFFU) / 255.0F,
                    .w = 1.0F,
                };
            }
        };

        static_assert(ShaderProgram<FloorShader<LayoutTiled4x4>>);

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
            float lod
        ) noexcept {
            if (!shader.uniforms.solid_color_debug && shader.uniforms.texture != nullptr) {
                std::array<float, 4> u{};
                std::array<float, 4> v{};
                for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    u[lane] = quad_uv[lane].x;
                    v[lane] = quad_uv[lane].y;
                }

                std::array<std::uint32_t, 4> packed{};
                shader.uniforms.texture
                    ->sample_trilinear_quad4(u.data(), v.data(), lod, packed.data());
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
                        .mip_lod = lod,
                    }
                );

                red[lane] = shaded.x;
                green[lane] = shaded.y;
                blue[lane] = shaded.z;
            }

            std::array<std::uint32_t, 4> packed{};
            simd::quantize_pack4(red.data(), green.data(), blue.data(), packed.data());
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
                for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    const auto& barycentrics = quad.lanes[lane];
                    const auto attributes =
                        shader.interpolate(varying_a, varying_b, varying_c, barycentrics);
                    quad_uv[lane] = std::get<0>(attributes);
                }

                const auto derivatives = quad_derivatives(quad_uv);
                const auto lod = mip_lod(
                    derivatives,
                    shader.uniforms.texture->width(),
                    shader.uniforms.texture->height()
                );
                store_shaded_quad(color, shader, varying_a, quad_x, quad_y, quad, quad_uv, lod);
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
                const auto varying_a = shader.vertex(mesh.positions[vertex_a], mesh.uvs[vertex_a]);
                const auto varying_b = shader.vertex(mesh.positions[vertex_b], mesh.uvs[vertex_b]);
                const auto varying_c = shader.vertex(mesh.positions[vertex_c], mesh.uvs[vertex_c]);
                const auto clipped = guard_band_test
                    ? clip_triangle_guard_band(shader, varying_a, varying_b, varying_c, 2.0F)
                    : clip_triangle(shader, varying_a, varying_b, varying_c);
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
                const auto varying_a = shader.vertex(mesh.positions[vertex_a], mesh.uvs[vertex_a]);
                const auto varying_b = shader.vertex(mesh.positions[vertex_b], mesh.uvs[vertex_b]);
                const auto varying_c = shader.vertex(mesh.positions[vertex_c], mesh.uvs[vertex_c]);
                const auto clipped = guard_band_test
                    ? clip_triangle_guard_band(shader, varying_a, varying_b, varying_c, 2.0F)
                    : clip_triangle(shader, varying_a, varying_b, varying_c);
                if (!clipped) {
                    return false;
                }
                for (const auto& clipped_triangle :
                     std::span{clipped->triangles.data(), clipped->size}) {
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
                for (const auto lane : std::views::iota(std::size_t{0}, std::size_t{4})) {
                    const auto attributes =
                        shader.interpolate(varying_a, varying_b, varying_c, quad.lanes[lane]);
                    quad_uv[lane] = std::get<0>(attributes);
                }

                const auto derivatives = quad_derivatives(quad_uv);
                const auto lod = mip_lod(
                    derivatives,
                    shader.uniforms.texture->width(),
                    shader.uniforms.texture->height()
                );
                store_shaded_quad(color, shader, varying_a, quad_x, quad_y, quad, quad_uv, lod);
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
        };

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

            bin_triangles(tile_state.setups, width, height, tile_state.bins);
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
        bool draw_overlay,
        TileRenderState<Layout>* tile_state = nullptr
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
        shader.uniforms.solid_color_debug = solid_color_debug;
        if (tile_state != nullptr) {
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
        const FreeLookCamera& camera
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
            true
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
        const FreeLookCamera& camera
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
            true,
            impl_.get()
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
            false
        );
    }
} // namespace raster::render
