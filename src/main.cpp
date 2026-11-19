#define SDL_MAIN_USE_CALLBACKS
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <print>
#include <ranges>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "raster/memory/framebuffer.hpp"
#include "raster/pipeline/rasterizer.hpp"
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
        const Rgb& color_c
    ) noexcept {
        auto triangle = raster::make_screen_triangle(a, b, c);
        if (!triangle.has_value()) {
            return;
        }

        auto color = framebuffer.color_view();
        raster::rasterize_triangle(
            *triangle,
            framebuffer.width,
            framebuffer.height,
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

    void render_scene(raster::Framebuffer& framebuffer, std::uint64_t elapsed_ns) noexcept {
        const auto width = framebuffer.width;
        const auto height = framebuffer.height;
        if (width < 2 || height < 2) {
            return;
        }

        std::ranges::fill(framebuffer.color_storage, 0xFF101018U);
        std::array<float, 11> grid_x{};
        std::array<float, 6> grid_y{};
        for (const auto column : std::views::iota(std::size_t{0}, std::size_t{11})) {
            grid_x[column] = static_cast<float>(column) * static_cast<float>(width) / 10.0F;
        }
        for (const auto row : std::views::iota(std::size_t{0}, std::size_t{6})) {
            grid_y[row] = static_cast<float>(row) * static_cast<float>(height) / 5.0F;
        }
        const auto vertex_color = [](std::size_t column, std::size_t row) noexcept {
            const auto u = static_cast<float>(column) / 10.0F;
            const auto v = static_cast<float>(row) / 5.0F;
            return Rgb{.r = u, .g = v, .b = 0.5F};
        };

        for (const auto row : std::views::iota(std::size_t{0}, std::size_t{5})) {
            for (const auto column : std::views::iota(std::size_t{0}, std::size_t{10})) {
                const raster::ScreenPoint top_left{.x = grid_x[column], .y = grid_y[row]};
                const raster::ScreenPoint top_right{.x = grid_x[column + 1], .y = grid_y[row]};
                const raster::ScreenPoint bottom_right{
                    .x = grid_x[column + 1],
                    .y = grid_y[row + 1],
                };

                const raster::ScreenPoint bottom_left{.x = grid_x[column], .y = grid_y[row + 1]};
                const Rgb color_top_left = vertex_color(column, row);
                const Rgb color_top_right = vertex_color(column + 1, row);
                const Rgb color_bottom_right = vertex_color(column + 1, row + 1);
                const Rgb color_bottom_left = vertex_color(column, row + 1);
                draw_shaded_triangle(
                    framebuffer,
                    top_left,
                    top_right,
                    bottom_right,
                    color_top_left,
                    color_top_right,
                    color_bottom_right
                );
                draw_shaded_triangle(
                    framebuffer,
                    top_left,
                    bottom_right,
                    bottom_left,
                    color_top_left,
                    color_bottom_right,
                    color_bottom_left
                );
            }
        }

        // One spinning RGB triangle on top to keep the scene animated.
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
        draw_shaded_triangle(framebuffer, spinner[0], spinner[1], spinner[2], red, green, blue);
    }

    struct AppState {
        raster::Framebuffer framebuffer{initial_width, initial_height};
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
    render_scene(state.framebuffer, now - state.animation_start_ns);
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
