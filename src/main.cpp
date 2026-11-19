#define SDL_MAIN_USE_CALLBACKS
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numbers>
#include <print>
#include <ranges>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <tuple>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/math/mat.hpp"
#include "raster/math/vec.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/pipeline/clipper.hpp"
#include "raster/pipeline/rasterizer.hpp"
#include "raster/pipeline/shader.hpp"
#include "raster/platform/sdl3_presenter.hpp"

namespace {
    constexpr std::size_t initial_width = 960;
    constexpr std::size_t initial_height = 540;

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
        const auto green = static_cast<std::uint32_t>(std::lround(clamp_channel(rgb.g) * 255.0F));
        const auto blue = static_cast<std::uint32_t>(std::lround(clamp_channel(rgb.b) * 255.0F));
        return 0xFF000000U | (red << 16U) | (green << 8U) | blue;
    }

    void draw_shaded_triangle(
        raster::Framebuffer& framebuffer,
        const raster::ScreenPoint& a,
        const raster::ScreenPoint& b,
        const raster::ScreenPoint& c,
        const Rgb& color_a,
        const Rgb& color_b,
        const Rgb& color_c,
        const std::array<float, 3>& inv_w = std::array<float, 3>{1.0F, 1.0F, 1.0F},
        const std::array<float, 3>& z_ndc = std::array<float, 3>{0.0F, 0.0F, 0.0F}
    ) noexcept {
        auto triangle = raster::make_screen_triangle(a, b, c, inv_w, z_ndc);
        if (!triangle.has_value()) {
            return;
        }

        auto color = framebuffer.color_view();
        raster::rasterize_triangle(
            *triangle,
            framebuffer.width,
            framebuffer.height,
            framebuffer.depth_view(),
            [&](std::size_t x,
                std::size_t y,
                const raster::PixelBarycentrics& barycentrics) noexcept {
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

    struct FloorShader final : raster::ShaderBase {
        struct Uniforms {
            raster::Mat4f model_view_proj{};
        };

        Uniforms uniforms{};

        struct Varyings {
            raster::Vec4f clip_pos{};
            raster::Vec2f uv{};

            [[nodiscard]]
            constexpr auto attributes(this auto&& self) noexcept {
                return std::tie(self.uv);
            }
        };

        [[nodiscard]]
        constexpr Varyings vertex(raster::Vec3f position, raster::Vec2f uv) const noexcept {
            return Varyings{
                .clip_pos = uniforms.model_view_proj
                    * raster::Vec4f{
                        .x = position.x,
                        .y = position.y,
                        .z = position.z,
                        .w = 1.0F,
                    },
                .uv = uv,
            };
        }

        [[nodiscard]]
        constexpr raster::Vec4f fragment(const Varyings& varying) const noexcept {
            const auto cell = static_cast<int>(std::floor(varying.uv.x))
                + static_cast<int>(std::floor(varying.uv.y));
            const auto parity = ((cell % 2) + 2) % 2;
            const auto shade = parity == 0 ? 0.85F : 0.35F;
            return raster::Vec4f{.x = shade, .y = shade, .z = shade, .w = 1.0F};
        }
    };

    static_assert(raster::ShaderProgram<FloorShader>);

    void draw_floor_triangle(
        raster::Framebuffer& framebuffer,
        FloorShader& shader,
        const FloorShader::Varyings& varying_a,
        const FloorShader::Varyings& varying_b,
        const FloorShader::Varyings& varying_c
    ) noexcept {
        const std::array<float, 3> inv_w{
            1.0F / varying_a.clip_pos.w,
            1.0F / varying_b.clip_pos.w,
            1.0F / varying_c.clip_pos.w,
        };
        const std::array<float, 3> z_ndc{
            varying_a.clip_pos.z * inv_w[0],
            varying_b.clip_pos.z * inv_w[1],
            varying_c.clip_pos.z * inv_w[2],
        };

        const auto viewport_width = framebuffer.width;
        const auto viewport_height = framebuffer.height;
        const std::array<raster::ScreenPoint, 3> screen{
            raster::viewport_to_screen(
                varying_a.clip_pos.x * inv_w[0],
                varying_a.clip_pos.y * inv_w[0],
                viewport_width,
                viewport_height
            ),
            raster::viewport_to_screen(
                varying_b.clip_pos.x * inv_w[1],
                varying_b.clip_pos.y * inv_w[1],
                viewport_width,
                viewport_height
            ),
            raster::viewport_to_screen(
                varying_c.clip_pos.x * inv_w[2],
                varying_c.clip_pos.y * inv_w[2],
                viewport_width,
                viewport_height
            ),
        };

        auto triangle = raster::make_screen_triangle(screen[0], screen[1], screen[2], inv_w, z_ndc);
        if (!triangle.has_value()) {
            return;
        }

        auto color = framebuffer.color_view();
        raster::rasterize_triangle(
            *triangle,
            viewport_width,
            viewport_height,
            framebuffer.depth_view(),
            [&](std::size_t x,
                std::size_t y,
                const raster::PixelBarycentrics& barycentrics) noexcept {
            const auto attributes =
                shader.interpolate(varying_a, varying_b, varying_c, barycentrics);
            const FloorShader::Varyings interpolated{
                .clip_pos = varying_a.clip_pos,
                .uv = std::get<0>(attributes),
            };

            const auto shaded = shader.fragment(interpolated);
            color[y, x] = to_argb(Rgb{.r = shaded.x, .g = shaded.y, .b = shaded.z});
        }
        );
    }

    [[nodiscard]]
    SDL_AppResult draw_mesh(
        raster::Framebuffer& framebuffer, FloorShader& shader, const raster::SoAMesh& mesh
    ) noexcept {
        for (const auto triangle_index : std::views::iota(std::size_t{0}, mesh.triangle_count())) {
            const auto index_base = triangle_index * 3;
            const auto vertex_a = static_cast<std::size_t>(mesh.indices[index_base]);
            const auto vertex_b = static_cast<std::size_t>(mesh.indices[index_base + 1]);
            const auto vertex_c = static_cast<std::size_t>(mesh.indices[index_base + 2]);
            const auto varying_a = shader.vertex(mesh.positions[vertex_a], mesh.uvs[vertex_a]);
            const auto varying_b = shader.vertex(mesh.positions[vertex_b], mesh.uvs[vertex_b]);
            const auto varying_c = shader.vertex(mesh.positions[vertex_c], mesh.uvs[vertex_c]);

            auto clipped = raster::clip_triangle(shader, varying_a, varying_b, varying_c);
            if (!clipped) {
                // Malformed/non-finite geometry is reported to the caller;
                // the noexcept frame path never performs formatted I/O.
                return SDL_APP_FAILURE;
            }

            for (const auto& subtriangle : std::span{clipped->triangles.data(), clipped->size}) {
                draw_floor_triangle(
                    framebuffer,
                    shader,
                    subtriangle.vertices[0],
                    subtriangle.vertices[1],
                    subtriangle.vertices[2]
                );
            }
        }
        return SDL_APP_CONTINUE;
    }

    SDL_AppResult render_scene(
        raster::Framebuffer& framebuffer, const raster::SoAMesh& mesh, std::uint64_t elapsed_ns
    ) noexcept {
        const auto width = framebuffer.width;
        const auto height = framebuffer.height;
        if (width < 2 || height < 2) {
            return SDL_APP_CONTINUE;
        }

        std::ranges::fill(framebuffer.color_storage, 0xFF101018U);
        std::ranges::fill(framebuffer.depth_storage, 0.0F);
        const auto seconds = static_cast<float>(static_cast<double>(elapsed_ns) * 1e-9);
        const auto eye = raster::Vec3f{
            .x = 0.05F * std::sin(seconds * 0.3F),
            .y = 0.1F,
            .z = 0.05F * std::cos(seconds * 0.3F),
        };
        const auto view = raster::look_at4f(
            eye,
            // Only a slight downward pitch lets the floor remain visible at
            // z=1000 while the near rows occupy the lower part of the view.
            raster::Vec3f{.x = eye.x, .y = eye.y - 0.0001F, .z = eye.z - 1.0F},
            raster::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F}
        );
        const auto projection = raster::perspective_reverse_z4f(
            1.2566371F, // 72 degrees
            static_cast<float>(width) / static_cast<float>(height),
            0.1F,
            1000.0F
        );

        FloorShader shader{};
        shader.uniforms.model_view_proj = projection * view;
        if (auto result = draw_mesh(framebuffer, shader, mesh); result != SDL_APP_CONTINUE) {
            return result;
        }

        // One spinning RGB overlay on top to keep the scene animated. Its
        // reverse-Z sentinel of 2.0 lies beyond the [0, 1] depth range on
        // purpose: it is nearer than any floor pixel, so it always wins.
        const auto theta = static_cast<float>(static_cast<double>(elapsed_ns) * 1.5e-9);
        const auto center_x = static_cast<float>(width) * 0.5F;
        const auto center_y = static_cast<float>(height) * 0.5F;
        const auto radius = 0.35F * static_cast<float>(std::min(width, height));
        const Rgb red{.r = 1.0F, .g = 0.0F, .b = 0.0F};
        const Rgb green{.r = 0.0F, .g = 1.0F, .b = 0.0F};
        const Rgb blue{.r = 0.0F, .g = 0.0F, .b = 1.0F};

        std::array<raster::ScreenPoint, 3> spinner{};
        for (const auto vertex : std::views::iota(std::size_t{0}, std::size_t{3})) {
            const auto angle =
                theta + static_cast<float>(vertex) * (2.0F * std::numbers::pi_v<float> / 3.0F);
            spinner[vertex] = raster::ScreenPoint{
                .x = center_x + radius * std::cos(angle),
                .y = center_y + radius * std::sin(angle),
            };
        }
        draw_shaded_triangle(
            framebuffer,
            spinner[0],
            spinner[1],
            spinner[2],
            red,
            green,
            blue,
            std::array<float, 3>{1.0F, 1.0F, 1.0F},
            std::array<float, 3>{2.0F, 2.0F, 2.0F}
        );
        return SDL_APP_CONTINUE;
    }

    struct AppState {
        raster::Framebuffer framebuffer{initial_width, initial_height};
        raster::SoAMesh mesh{};
        raster::TgaImage checker_texture{};
        raster::WindowPresenter presenter;
        std::uint64_t frame_count{};
        std::uint64_t report_start_ns{};
        std::uint64_t animation_start_ns{};
        std::uint32_t stress_step{};
        bool auto_resize_test{};
    };
} // namespace

// NOLINTNEXTLINE(readability-identifier-naming)
SDL_AppResult SDLCALL SDL_AppInit(void** appstate, int, char*[]) {
    try {
        auto state = std::make_unique<AppState>();
        if (auto result = state->presenter.initialize(
                static_cast<int>(initial_width),
                static_cast<int>(initial_height),
                state->framebuffer
            );
            !result) {
            std::println(
                stderr,
                "raster: SDL initialization failed ({})",
                static_cast<int>(result.error())
            );
            return SDL_APP_FAILURE;
        }

        const auto now = SDL_GetTicksNS();
        const auto asset_directory = std::filesystem::path{RASTER_ASSET_DIR};
        auto mesh = raster::load_obj(asset_directory / "models/raster_milestone.obj");
        if (!mesh) {
            std::println(stderr, "raster: OBJ load failed ({})", static_cast<int>(mesh.error()));
            return SDL_APP_FAILURE;
        }
        state->mesh = std::move(*mesh);

        auto checker =
            raster::load_tga(std::filesystem::path{RASTER_ASSET_DIR} / "textures/checker.tga");
        if (!checker) {
            std::println(stderr, "raster: TGA load failed ({})", static_cast<int>(checker.error()));
            return SDL_APP_FAILURE;
        }
        state->checker_texture = std::move(*checker);

        state->report_start_ns = now;
        state->animation_start_ns = now;
        state->auto_resize_test =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_AUTO_RESIZE_TEST") != nullptr;
        *appstate = state.release();
        return SDL_APP_CONTINUE;
    } catch (...) {
        std::println(stderr, "raster: failed to allocate application state");
        return SDL_APP_FAILURE;
    }
}

// NOLINTNEXTLINE(readability-identifier-naming)
SDL_AppResult SDLCALL SDL_AppEvent(void* appstate, SDL_Event* event) {
    auto& state = *static_cast<AppState*>(appstate);
    if (event->type == SDL_EVENT_QUIT) {
        return SDL_APP_SUCCESS;
    }
    if (event->type
        == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED
        || event->type
        == SDL_EVENT_WINDOW_RESTORED) {
        state.presenter.set_minimized(false);
        if (auto result = state.presenter.resize_to_window_pixels(state.framebuffer); !result) {
            std::println(
                stderr,
                "raster: failed to resize framebuffer ({})",
                static_cast<int>(result.error())
            );
            return SDL_APP_FAILURE;
        }
    }
    if (event->type == SDL_EVENT_WINDOW_MINIMIZED) {
        state.presenter.set_minimized(true);
    }
    return SDL_APP_CONTINUE;
}

// NOLINTNEXTLINE(readability-identifier-naming)
SDL_AppResult SDLCALL SDL_AppIterate(void* appstate) {
    auto& state = *static_cast<AppState*>(appstate);
    if (state.auto_resize_test) {
        switch (state.stress_step) {
            case 0:
                std::println("raster: stress: resize to 640x480");
                SDL_SetWindowSize(state.presenter.window(), 640, 480);
                break;
            case 1:
                std::println("raster: stress: resize to 1280x720");
                SDL_SetWindowSize(state.presenter.window(), 1280, 720);
                break;
            case 2:
                std::println("raster: stress: resize to 320x200");
                SDL_SetWindowSize(state.presenter.window(), 320, 200);
                break;
            case 3:
                std::println("raster: stress: minimize");
                SDL_MinimizeWindow(state.presenter.window());
                break;
            case 4:
                std::println("raster: stress: restore");
                SDL_RestoreWindow(state.presenter.window());
                break;
            case 5:
                std::println("raster: stress: resize to 1024x768");
                SDL_SetWindowSize(state.presenter.window(), 1024, 768);
                break;
            case 6:
                if (auto result = state.presenter.resize_to_window_pixels(state.framebuffer);
                    !result) {
                    std::println(
                        stderr,
                        "raster: stress: final resize failed ({})",
                        static_cast<int>(result.error())
                    );
                    return SDL_APP_FAILURE;
                }

                std::println(
                    "raster: stress complete — framebuffer {}x{}",
                    state.framebuffer.width,
                    state.framebuffer.height
                );
                if (state.framebuffer.width != 1024U || state.framebuffer.height != 768U) {
                    std::println(stderr, "raster: stress: framebuffer did not reach 1024x768");
                    return SDL_APP_FAILURE;
                }
                return SDL_APP_SUCCESS;
            default:
                break;
        }
        ++state.stress_step;
    }
    if (state.presenter.is_minimized()) {
        return SDL_APP_CONTINUE;
    }

    const auto now = SDL_GetTicksNS();
    const auto render_result =
        render_scene(state.framebuffer, state.mesh, now - state.animation_start_ns);
    if (render_result != SDL_APP_CONTINUE) {
        return render_result;
    }
    if (auto result = state.presenter.present(state.framebuffer); !result) {
        std::println(stderr, "raster: presentation failed ({})", static_cast<int>(result.error()));
        return SDL_APP_FAILURE;
    }

    ++state.frame_count;
    const auto elapsed_ns = now - state.report_start_ns;
    if (elapsed_ns >= 1'000'000'000U) {
        const auto fps = static_cast<double>(state.frame_count)
            * 1'000'000'000.0
            / static_cast<double>(elapsed_ns);
        std::println("raster: {:.1f} FPS", fps);
        state.frame_count = 0;
        state.report_start_ns = now;
    }
    return SDL_APP_CONTINUE;
}

// NOLINTNEXTLINE(readability-identifier-naming)
void SDLCALL SDL_AppQuit(void* appstate, SDL_AppResult) {
    delete static_cast<AppState*>(appstate);
}
