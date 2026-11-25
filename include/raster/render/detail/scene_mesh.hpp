#pragma once

// Mesh submission: per-triangle vertex fetch, conservative frustum reject,
// clip, and screen setup shared by the serial and tiled prepare paths.
// The quarantine rule is simple: uncertainty keeps the triangle, and only
// the shared-outcode test skips clip work.

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/pipeline/clipper.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/render/detail/scene_quad.hpp"
#include "raster/render/detail/scene_shader.hpp"
#include "raster/render/scene_chunks.hpp"
#include "raster/render/scene_renderer.hpp"

namespace raster::render::detail {
    // Shared vertex-fetch, frustum-reject, and clip prologue for the serial
    // and tiled prepare paths.
    template <typename Shader> [[nodiscard]]
    inline bool clip_source_triangle(
        Shader& shader,
        const SoAMesh& mesh,
        std::size_t triangle_index,
        bool guard_band_test,
        SceneRenderDiagnostics* diagnostics,
        ClippedTriangleFan<typename Shader::Varyings>& fan_out
    ) noexcept {
        const auto index_base = triangle_index * 3U;
        const auto vertex_a = static_cast<std::size_t>(mesh.indices[index_base]);
        const auto vertex_b = static_cast<std::size_t>(mesh.indices[index_base + 1U]);
        const auto vertex_c = static_cast<std::size_t>(mesh.indices[index_base + 2U]);
        const auto varying_a =
            shader.vertex(mesh.positions[vertex_a], mesh.uvs[vertex_a], mesh.normals[vertex_a]);
        const auto varying_b =
            shader.vertex(mesh.positions[vertex_b], mesh.uvs[vertex_b], mesh.normals[vertex_b]);
        const auto varying_c =
            shader.vertex(mesh.positions[vertex_c], mesh.uvs[vertex_c], mesh.normals[vertex_c]);
        if (diagnostics != nullptr) {
            diagnostics->source_triangles.fetch_add(1U, std::memory_order_relaxed);
        }
        if (!raster::detail::finite(varying_a.clip_pos)
            || !raster::detail::finite(varying_b.clip_pos)
            || !raster::detail::finite(varying_c.clip_pos)) {
            const auto clipped = guard_band_test
                ? clip_triangle_guard_band(shader, varying_a, varying_b, varying_c, 2.0F)
                : clip_triangle(shader, varying_a, varying_b, varying_c);
            if (!clipped) {
                return false;
            }
            if (diagnostics != nullptr) {
                diagnostics->clipped_triangles.fetch_add(1U, std::memory_order_relaxed);
            }

            fan_out = std::move(*clipped);
            return true;
        }

        const auto guard = guard_band_test ? 2.0F : 1.0F;
        unsigned common = 0U;
        unsigned combined = 0U;
        raster::detail::clip_triangle_codes(
            varying_a.clip_pos,
            varying_b.clip_pos,
            varying_c.clip_pos,
            guard,
            common,
            combined
        );
        if (common != 0U) {
            if (diagnostics != nullptr) {
                diagnostics->frustum_rejected_triangles.fetch_add(1U, std::memory_order_relaxed);
            }
            return true;
        }

        const auto clipped = clip_triangle_impl_with_masks(
            shader,
            varying_a,
            varying_b,
            varying_c,
            guard,
            common,
            combined
        );
        if (!clipped) {
            return false;
        }
        if (diagnostics != nullptr) {
            diagnostics->clipped_triangles.fetch_add(1U, std::memory_order_relaxed);
        }

        fan_out = std::move(*clipped);
        return true;
    }

    // Per-quad mip selection shared by the serial and tiled rasterize paths.
    // The metallic-roughness texture gets its own LOD from its own size.
    template <typename Layout> inline void compute_quad_lods(
        const FloorShader<Layout>& shader,
        const QuadDerivatives& derivatives,
        float& lod,
        float& mr_lod
    ) noexcept {
        lod = mip_lod(
            derivatives,
            shader.uniforms.texture->width(),
            shader.uniforms.texture->height()
        );
        mr_lod = shader.uniforms.metallic_roughness_texture != nullptr
            ? mip_lod(
                  derivatives,
                  shader.uniforms.metallic_roughness_texture->width(),
                  shader.uniforms.metallic_roughness_texture->height()
              )
            : lod;
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

        const auto triangle = make_screen_triangle(screen, inverse_w, reverse_z);
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
            float lod = 0.0F;
            float mr_lod = 0.0F;
            compute_quad_lods(shader, derivatives, lod, mr_lod);
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
        bool guard_band_test,
        SceneRenderDiagnostics* diagnostics,
        std::span<const SceneChunk> chunks,
        const Vec3f& lod_reference
    ) noexcept {
        const auto draw_triangle = [&](std::size_t triangle_index) noexcept -> bool {
            ClippedTriangleFan<typename FloorShader<Layout>::Varyings> fan{};
            if (!clip_source_triangle(
                    shader,
                    mesh,
                    triangle_index,
                    guard_band_test,
                    diagnostics,
                    fan
                )) {
                return false;
            }
            for (const auto& triangle : fan.active_triangles()) {
                if (diagnostics != nullptr) {
                    diagnostics->prepared_triangles.fetch_add(1U, std::memory_order_relaxed);
                }

                draw_mesh_triangle(
                    framebuffer,
                    shader,
                    triangle.vertices[0],
                    triangle.vertices[1],
                    triangle.vertices[2]
                );
                if (diagnostics != nullptr) {
                    diagnostics->rasterized_triangle_submissions.fetch_add(
                        1U,
                        std::memory_order_relaxed
                    );
                }
            }
            return true;
        };
        if (chunks.empty()) {
            for (const auto triangle_index :
                 std::views::iota(std::size_t{0}, mesh.triangle_count())) {
                if (!draw_triangle(triangle_index)) {
                    return false;
                }
            }
            return true;
        }

        const auto guard_band = guard_band_test ? 2.0F : 1.0F;
        for (const auto& chunk : chunks) {
            if (diagnostics != nullptr) {
                diagnostics->source_chunks.fetch_add(1U, std::memory_order_relaxed);
            }
            if (chunk_outside_clip_frustum(
                    shader.uniforms.model_view_projection,
                    chunk,
                    guard_band
                )) {
                if (diagnostics != nullptr) {
                    diagnostics->frustum_rejected_chunks.fetch_add(1U, std::memory_order_relaxed);
                }
                continue;
            }

            const auto chunk_lod_distance = chunk.distance_to(lod_reference);
            static_cast<void>(chunk_lod_distance); // future authored-LOD selection metric
            const auto end =
                std::min(chunk.first_triangle + chunk.triangle_count, mesh.triangle_count());
            for (const auto triangle_index : std::views::iota(chunk.first_triangle, end)) {
                if (!draw_triangle(triangle_index)) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename Shader> struct PreparedTriangle {
        ScreenTriangle setup{};
        std::array<typename Shader::Varyings, 3> vertices{};
    };

    template <typename Shader> [[nodiscard]]
    bool prepare_mesh(
        Shader& shader,
        const SoAMesh& mesh,
        std::size_t width,
        std::size_t height,
        bool guard_band_test,
        std::size_t begin_work,
        std::size_t end_work,
        std::size_t begin_chunk,
        std::size_t end_chunk,
        std::vector<PreparedTriangle<Shader>>& prepared,
        SceneRenderDiagnostics* diagnostics,
        std::span<const SceneChunk> chunks,
        const Vec3f& lod_reference
    ) noexcept {
        prepared.clear();
        const auto prepare_triangle = [&](std::size_t triangle_index) noexcept -> bool {
            ClippedTriangleFan<typename Shader::Varyings> fan{};
            if (!clip_source_triangle(
                    shader,
                    mesh,
                    triangle_index,
                    guard_band_test,
                    diagnostics,
                    fan
                )) {
                return false;
            }
            for (const auto& clipped_triangle : fan.active_triangles()) {
                const std::array<typename Shader::Varyings, 3> vertices{
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
                const std::array<ScreenPoint, 3> screen{
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
                };

                const auto setup = make_screen_triangle(screen, inverse_w, reverse_z);
                if (setup) {
                    if (diagnostics != nullptr) {
                        diagnostics->prepared_triangles.fetch_add(1U, std::memory_order_relaxed);
                        diagnostics->rasterized_triangle_submissions.fetch_add(
                            1U,
                            std::memory_order_relaxed
                        );
                    }
                    prepared.push_back(
                        PreparedTriangle<Shader>{.setup = *setup, .vertices = vertices}
                    );
                }
            }
            return true;
        };
        if (chunks.empty()) {
            for (const auto triangle_index :
                 std::views::iota(begin_work, std::min(end_work, mesh.triangle_count()))) {
                if (!prepare_triangle(triangle_index)) {
                    return false;
                }
            }
            return true;
        }
        for (const auto chunk_index :
             std::views::iota(begin_chunk, std::min(end_chunk, chunks.size()))) {
            const auto& chunk = chunks[chunk_index];
            if (diagnostics != nullptr) {
                diagnostics->source_chunks.fetch_add(1U, std::memory_order_relaxed);
            }
            if (chunk_outside_clip_frustum(
                    shader.uniforms.model_view_projection,
                    chunk,
                    guard_band_test ? 2.0F : 1.0F
                )) {
                if (diagnostics != nullptr) {
                    diagnostics->frustum_rejected_chunks.fetch_add(1U, std::memory_order_relaxed);
                }
                continue;
            }

            const auto chunk_lod_distance = chunk.distance_to(lod_reference);
            static_cast<void>(chunk_lod_distance); // future authored-LOD selection metric
            const auto end =
                std::min(chunk.first_triangle + chunk.triangle_count, mesh.triangle_count());
            for (const auto triangle_index : std::views::iota(chunk.first_triangle, end)) {
                if (!prepare_triangle(triangle_index)) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename Layout> void rasterize_prepared_triangle(
        Framebuffer& framebuffer,
        FloorShader<Layout>& shader,
        const PreparedTriangle<FloorShader<Layout>>& triangle,
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
            float lod = 0.0F;
            float mr_lod = 0.0F;
            compute_quad_lods(shader, derivatives, lod, mr_lod);
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
} // namespace raster::render::detail
