#define SDL_MAIN_USE_CALLBACKS
#include <cstddef>
#include <cstdint>
#include <memory>
#include <print>
#include <ranges>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "raster/memory/framebuffer.hpp"
#include "raster/platform/sdl3_presenter.hpp"

namespace {

    constexpr std::size_t initial_width = 960;
    constexpr std::size_t initial_height = 540;

    struct AppState {
        raster::Framebuffer framebuffer{initial_width, initial_height};
        raster::WindowPresenter presenter;
        std::uint64_t frame_count{};
        std::uint64_t report_start_ns{};
        std::uint64_t animation_start_ns{};
        std::uint32_t stress_step{};
        bool auto_resize_test{};
    };

    void fill_pattern(raster::Framebuffer& framebuffer, std::uint64_t elapsed_ns) noexcept {
        const auto width = framebuffer.width;
        const auto height = framebuffer.height;
        if (width < 2 || height < 2) {
            return;
        }

        auto color = framebuffer.color_view();
        const auto shift = static_cast<std::size_t>((elapsed_ns / 10'000'000U) % 255U);
        const auto x_range = std::views::iota(std::size_t{0}, width);
        const auto y_range = std::views::iota(std::size_t{0}, height);
        for (const auto [y, x] : std::views::cartesian_product(y_range, x_range)) {
            const auto red = static_cast<std::uint32_t>((x * 255U / (width - 1U) + shift) % 256U);
            const auto green = static_cast<std::uint32_t>(y * 255U / (height - 1U));
            const auto blue = static_cast<std::uint32_t>(
                ((x + y) * 255U / (width + height - 2U) + shift * 2U) % 256U
            );
            color[y, x] = 0xFF000000U | (red << 16U) | (green << 8U) | blue;
        }
    }
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
    fill_pattern(state.framebuffer, now - state.animation_start_ns);
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
