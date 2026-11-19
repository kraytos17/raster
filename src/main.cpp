#define SDL_MAIN_USE_CALLBACKS
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <print>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/platform/allocation_probe.hpp"
#include "raster/platform/sdl3_presenter.hpp"
#include "raster/render/scene_renderer.hpp"

namespace {
    constexpr std::size_t initial_width = 960;
    constexpr std::size_t initial_height = 540;
    constexpr std::uint32_t clip_stress_frame_count = 120;

    struct AppState {
        raster::Framebuffer framebuffer{initial_width, initial_height};
        raster::SoAMesh mesh{};
        raster::TgaImage checker_texture{};
        raster::WindowPresenter presenter;
        std::uint64_t frame_count{};
        std::uint64_t report_start_ns{};
        std::uint64_t animation_start_ns{};
        std::uint32_t resize_step{};
        std::uint32_t clip_stress_frame{};
        bool auto_resize_test{};
        bool clip_stress_test{};
    };

    [[nodiscard]]
    bool load_assets(AppState& state) {
        const auto assets = std::filesystem::path{RASTER_ASSET_DIR};
        state.clip_stress_test =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_CLIP_STRESS_TEST") != nullptr;

        const auto mesh_path = state.clip_stress_test ? assets / "models/clipping_sphere.obj"
                                                      : assets / "models/raster_milestone.obj";
        auto mesh = raster::load_obj(mesh_path);
        if (!mesh) {
            std::println(stderr, "raster: OBJ load failed ({})", static_cast<int>(mesh.error()));
            return false;
        }

        state.mesh = std::move(*mesh);
        if (!state.clip_stress_test) {
            auto texture = raster::load_tga(assets / "textures/checker.tga");
            if (!texture) {
                std::println(
                    stderr,
                    "raster: TGA load failed ({})",
                    static_cast<int>(texture.error())
                );
                return false;
            }
            state.checker_texture = std::move(*texture);
        }
        return true;
    }

    [[nodiscard]]
    SDL_AppResult run_resize_test(AppState& state) {
        switch (state.resize_step) {
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
                        "raster: final resize failed ({})",
                        static_cast<int>(result.error())
                    );
                    return SDL_APP_FAILURE;
                }
                if (state.framebuffer.width != 1024U || state.framebuffer.height != 768U) {
                    std::println(stderr, "raster: stress: framebuffer did not reach 1024x768");
                    return SDL_APP_FAILURE;
                }

                std::println(
                    "raster: resize stress passed — framebuffer {}x{}",
                    state.framebuffer.width,
                    state.framebuffer.height
                );
                return SDL_APP_SUCCESS;
            default:
                break;
        }
        ++state.resize_step;
        return SDL_APP_CONTINUE;
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
        if (!load_assets(*state)) {
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
        std::println(stderr, "raster: failed to initialize application state");
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
                "raster: framebuffer resize failed ({})",
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
        const auto resize_result = run_resize_test(state);
        if (resize_result != SDL_APP_CONTINUE) {
            return resize_result;
        }
    }
    if (state.presenter.is_minimized()) {
        return SDL_APP_CONTINUE;
    }

    const auto now = SDL_GetTicksNS();
    const auto elapsed_ns = state.clip_stress_test
        ? static_cast<std::uint64_t>(state.clip_stress_frame) * 50'000'000U
        : now - state.animation_start_ns;
    bool rendered = false;
    std::size_t allocation_count = 0;
    {
        const raster::platform::ScopedAllocationProbe allocation_probe{state.clip_stress_test};
        rendered = raster::render::render_scene(
            state.framebuffer,
            state.mesh,
            elapsed_ns,
            state.clip_stress_test
        );
        allocation_count = allocation_probe.allocations();
    }

    if (!rendered) {
        std::println(stderr, "raster: scene rendering failed");
        return SDL_APP_FAILURE;
    }
    if (state.clip_stress_test && allocation_count != 0U) {
        std::println(stderr, "raster: render allocation trap: {} allocations", allocation_count);
        return SDL_APP_FAILURE;
    }
    if (auto result = state.presenter.present(state.framebuffer); !result) {
        std::println(stderr, "raster: presentation failed ({})", static_cast<int>(result.error()));
        return SDL_APP_FAILURE;
    }
    if (state.clip_stress_test && ++state.clip_stress_frame >= clip_stress_frame_count) {
        std::println(
            "raster: clipping stress passed — {} triangles, {} frames, zero render allocations",
            state.mesh.triangle_count(),
            state.clip_stress_frame
        );
        return SDL_APP_SUCCESS;
    }

    ++state.frame_count;
    const auto report_elapsed_ns = now - state.report_start_ns;
    if (report_elapsed_ns >= 1'000'000'000U) {
        const auto fps = static_cast<double>(state.frame_count)
            * 1'000'000'000.0
            / static_cast<double>(report_elapsed_ns);
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
