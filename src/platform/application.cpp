#include "raster/platform/application.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <print>
#include <SDL3/SDL.h>
#include <thread>

#include "raster/io/obj_loader.hpp"
#include "raster/io/pbr_material_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/pipeline/tile_executor.hpp"
#include "raster/platform/allocation_probe.hpp"
#include "raster/platform/sdl3_presenter.hpp"
#include "raster/render/scene_renderer.hpp"

namespace raster::platform {
    namespace {
        constexpr std::size_t initial_width = 960;
        constexpr std::size_t initial_height = 540;
        constexpr std::size_t minimum_free_look_resolution_divisor = 3;
        constexpr std::size_t maximum_free_look_resolution_divisor = 6;
        constexpr std::uint64_t free_look_frame_budget_ns = 16'666'667U;
        constexpr std::uint64_t free_look_headroom_budget_ns = 12'500'000U;
        constexpr std::size_t free_look_timing_sample_capacity = 240U;
        constexpr std::uint32_t clip_stress_frame_count = 120;

        // Explicit worker override from RASTER_WORKERS; 0 means unset or
        // unparsable (full match required, so "4x" doesn't silently become 4).
        [[nodiscard]]
        std::size_t requested_tile_workers() noexcept {
            const auto* value = SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_WORKERS");
            if (value == nullptr) {
                return 0U;
            }

            const auto length = std::strlen(value);
            std::size_t parsed = 0U;
            const auto [end, code] = std::from_chars(value, value + length, parsed);
            if (code != std::errc{} || end != value + length) {
                return 0U;
            }
            return parsed;
        }

        struct AppState {
            raster::Framebuffer framebuffer{initial_width, initial_height};
            std::unique_ptr<raster::Framebuffer> free_look_framebuffer{};
            raster::SoAMesh mesh{};
            raster::TgaImage checker_texture{};
            raster::Texture2D<raster::LayoutTiled4x4> texture{};
            raster::pbr::Material pbr_factors{};
            raster::Texture2D<raster::LayoutTiled4x4> pbr_base_color_texture{};
            raster::Texture2D<raster::LayoutTiled4x4> pbr_metallic_roughness_texture{};
            bool pbr_material_loaded{};
            bool pbr_has_metallic_roughness_texture{};
            raster::render::TileRenderer tile_renderer{raster::resolve_tile_workers(
                static_cast<std::size_t>(std::thread::hardware_concurrency()),
                requested_tile_workers()
            )};

            bool tile_renderer_warmed{};
            raster::WindowPresenter presenter;
            std::uint64_t frame_count{};
            std::uint64_t report_start_ns{};
            std::uint64_t animation_start_ns{};
            std::uint64_t previous_frame_ns{};
            double free_look_frame_time_ema_ns{};
            std::array<std::uint64_t, free_look_timing_sample_capacity> free_look_frame_times_ns{};
            std::size_t free_look_frame_time_sample_count{};
            std::size_t free_look_frame_time_sample_cursor{};
            std::uint32_t resize_step{};
            std::uint32_t clip_stress_frame{};
            std::uint32_t free_look_over_budget_frames{};
            std::uint64_t free_look_headroom_since_ns{};
            std::uint64_t free_look_last_scale_change_ns{};
            std::size_t free_look_resolution_divisor{3U};
            bool auto_resize_test{};
            bool clip_stress_test{};
            bool free_look_test{};
            bool guard_band_test{};
            bool solid_color_debug{};
            raster::render::FreeLookCamera camera{
                .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
                .yaw = 0.0F,
                .pitch = 0.0F,
            };
        };

        [[nodiscard]]
        bool resize_free_look_target(AppState& state) noexcept;

        [[nodiscard]]
        bool load_assets(AppState& state) {
            const auto assets = std::filesystem::path{RASTER_ASSET_DIR};
            state.clip_stress_test =
                SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_CLIP_STRESS_TEST")
                != nullptr;
            state.free_look_test =
                SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_FREE_LOOK_TEST")
                != nullptr;
            state.guard_band_test =
                SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_GUARD_BAND_TEST")
                != nullptr;

            const auto use_sphere =
                state.clip_stress_test || state.free_look_test || state.guard_band_test;
            const auto mesh_path = use_sphere ? assets / "models/clipping_sphere.obj"
                                              : assets / "models/raster_milestone.obj";
            auto mesh = raster::load_obj(mesh_path);
            if (!mesh) {
                std::println(
                    stderr,
                    "raster: OBJ load failed ({})",
                    static_cast<int>(mesh.error())
                );
                return false;
            }

            state.mesh = std::move(*mesh);
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
            state.texture = raster::Texture2D<raster::LayoutTiled4x4>{state.checker_texture};
            if (const auto* const descriptor =
                    SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PBR_MATERIAL");
                descriptor != nullptr) {
                const std::filesystem::path descriptor_path{descriptor};
                const auto resolved_descriptor =
                    descriptor_path.is_absolute() ? descriptor_path : assets / descriptor_path;

                auto pbr_material = raster::load_pbr_material(resolved_descriptor);
                if (!pbr_material) {
                    std::println(
                        stderr,
                        "raster: PBR material load failed ({})",
                        static_cast<int>(pbr_material.error())
                    );
                    return false;
                }
                if (!pbr_material->base_color_texture.has_value()) {
                    std::println(stderr, "raster: PBR material requires base_color_texture");
                    return false;
                }

                state.pbr_factors = pbr_material->factors;
                state.pbr_base_color_texture = std::move(pbr_material->base_color_texture.value());
                if (pbr_material->metallic_roughness_texture.has_value()) {
                    state.pbr_metallic_roughness_texture =
                        std::move(pbr_material->metallic_roughness_texture.value());
                    state.pbr_has_metallic_roughness_texture = true;
                }
                state.pbr_material_loaded = true;
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
                    if (!resize_free_look_target(state)) {
                        std::println(stderr, "raster: free-look render target resize failed");
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

        [[nodiscard]]
        bool resize_free_look_target(AppState& state) noexcept {
            if (!state.free_look_test || !state.free_look_framebuffer) {
                return true;
            }

            const auto width = std::max(
                state.framebuffer.width / state.free_look_resolution_divisor,
                std::size_t{1}
            );
            const auto height = std::max(
                state.framebuffer.height / state.free_look_resolution_divisor,
                std::size_t{1}
            );
            if (state.free_look_framebuffer->width
                == width
                && state.free_look_framebuffer->height
                == height) {
                return true;
            }
            try {
                state.free_look_framebuffer->resize(width, height);
                return true;
            } catch (...) {
                return false;
            }
        }
    } // namespace

    // NOLINTNEXTLINE(readability-identifier-naming)
    SDL_AppResult SDLCALL application_init(void** appstate, int, char*[]) {
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
            if (SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_NO_VSYNC") != nullptr) {
                // Immediate presentation for true frame-rate measurement.
                // Non-fatal: tearing or dummy-driver refusal is acceptable.
                if (auto result = state->presenter.set_vsync(false); !result) {
                    std::println(
                        stderr,
                        "raster: uncapped presentation unavailable, keeping vsync"
                    );
                }
            }
            if (state->free_look_test) {
                state->free_look_framebuffer = std::make_unique<raster::Framebuffer>(
                    std::max(
                        state->framebuffer.width / state->free_look_resolution_divisor,
                        std::size_t{1}
                    ),
                    std::max(
                        state->framebuffer.height / state->free_look_resolution_divisor,
                        std::size_t{1}
                    )
                );
            }

            const auto now = SDL_GetTicksNS();
            state->report_start_ns = now;
            state->animation_start_ns = now;
            state->previous_frame_ns = now;
            state->auto_resize_test =
                SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_AUTO_RESIZE_TEST")
                != nullptr;
            if (state->free_look_test
                && !SDL_SetWindowRelativeMouseMode(state->presenter.window(), true)) {
                std::println(stderr, "raster: failed to enable relative mouse mode");
                return SDL_APP_FAILURE;
            }
            *appstate = state.release();
            return SDL_APP_CONTINUE;
        } catch (...) {
            std::println(stderr, "raster: failed to initialize application state");
            return SDL_APP_FAILURE;
        }
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    SDL_AppResult SDLCALL application_event(void* appstate, SDL_Event* event) {
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
            if (!resize_free_look_target(state)) {
                std::println(stderr, "raster: free-look render target resize failed");
                return SDL_APP_FAILURE;
            }
        }
        if (event->type == SDL_EVENT_WINDOW_MINIMIZED) {
            state.presenter.set_minimized(true);
        }
        if (state.free_look_test) {
            if (event->type == SDL_EVENT_MOUSE_MOTION) {
                constexpr float mouse_sensitivity = 0.0025F;
                state.camera.yaw += event->motion.xrel * mouse_sensitivity;
                state.camera.pitch = std::clamp(
                    state.camera.pitch - event->motion.yrel * mouse_sensitivity,
                    -1.5F,
                    1.5F
                );
            }
            if (event->type == SDL_EVENT_KEY_DOWN && event->key.repeat == 0) {
                if (event->key.key == SDLK_ESCAPE) {
                    return SDL_APP_SUCCESS;
                }
                if (event->key.key == SDLK_F2) {
                    state.guard_band_test = !state.guard_band_test;
                    std::println(
                        "raster: {} clipping",
                        state.guard_band_test ? "guard-band" : "six-plane"
                    );
                }
                if (event->key.key == SDLK_F3) {
                    state.solid_color_debug = !state.solid_color_debug;
                    std::println(
                        "raster: {} shading",
                        state.solid_color_debug ? "solid-color pole diagnostic" : "textured"
                    );
                }
            }
        }
        return SDL_APP_CONTINUE;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    SDL_AppResult SDLCALL application_iterate(void* appstate) {
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

        const auto frame_start_ns = SDL_GetTicksNS();
        const auto now = frame_start_ns;
        if (state.free_look_test) {
            constexpr float vertical_speed = 2.0F;
            const auto frame_dt = std::clamp(
                static_cast<float>(static_cast<double>(now - state.previous_frame_ns) * 1e-9),
                0.0F,
                0.25F
            );

            state.previous_frame_ns = now;
            const auto* const keys = SDL_GetKeyboardState(nullptr);
            const raster::Vec3f forward{
                .x = std::sin(state.camera.yaw),
                .y = 0.0F,
                .z = -std::cos(state.camera.yaw),
            };
            const raster::Vec3f right{
                .x = std::cos(state.camera.yaw),
                .y = 0.0F,
                .z = std::sin(state.camera.yaw),
            };

            const auto move_speed = raster::render::free_look_movement_speed(state.camera.position);
            if (keys[SDL_SCANCODE_W]) {
                state.camera.position = state.camera.position + forward * (move_speed * frame_dt);
            }
            if (keys[SDL_SCANCODE_S]) {
                state.camera.position = state.camera.position - forward * (move_speed * frame_dt);
            }
            if (keys[SDL_SCANCODE_D]) {
                state.camera.position = state.camera.position + right * (move_speed * frame_dt);
            }
            if (keys[SDL_SCANCODE_A]) {
                state.camera.position = state.camera.position - right * (move_speed * frame_dt);
            }
            if (keys[SDL_SCANCODE_E]) {
                state.camera.position.y += vertical_speed * frame_dt;
            }
            if (keys[SDL_SCANCODE_Q]) {
                state.camera.position.y -= vertical_speed * frame_dt;
            }
        }

        const auto elapsed_ns = state.clip_stress_test
            ? static_cast<std::uint64_t>(state.clip_stress_frame) * 50'000'000U
            : now - state.animation_start_ns;

        const auto upscale_nearest = [](const raster::Framebuffer& source,
                                        raster::Framebuffer& destination) noexcept {
            const auto source_color = source.color_view();
            auto destination_color = destination.color_view();
            for (std::size_t y = 0; y < destination.height; ++y) {
                const auto source_y = y * source.height / destination.height;
                for (std::size_t x = 0; x < destination.width; ++x) {
                    const auto source_x = x * source.width / destination.width;
                    destination_color[y, x] = source_color[source_y, source_x];
                }
            }
        };

        auto& render_target = state.free_look_test && state.free_look_framebuffer
            ? *state.free_look_framebuffer
            : state.framebuffer;

        const auto use_pbr = state.pbr_material_loaded;
        const auto* base_color_texture = use_pbr ? &state.pbr_base_color_texture : &state.texture;
        const auto* metallic_roughness = use_pbr && state.pbr_has_metallic_roughness_texture
            ? &state.pbr_metallic_roughness_texture
            : nullptr;

        const auto check_render_allocations = state.clip_stress_test && state.tile_renderer_warmed;
        bool rendered = false;
        std::size_t allocation_count = 0;
        {
            const raster::platform::ScopedAllocationProbe allocation_probe{
                check_render_allocations
            };

            rendered = state.tile_renderer.render_scene(
                render_target,
                state.mesh,
                elapsed_ns,
                state.clip_stress_test,
                state.free_look_test,
                state.guard_band_test,
                state.solid_color_debug,
                *base_color_texture,
                state.camera,
                use_pbr ? raster::render::SceneShading::Pbr : raster::render::SceneShading::Diffuse,
                state.pbr_factors,
                metallic_roughness
            );
            if (rendered && state.free_look_test && state.free_look_framebuffer) {
                upscale_nearest(*state.free_look_framebuffer, state.framebuffer);
            }
            allocation_count = allocation_probe.allocations();
        }

        if (state.clip_stress_test && !state.tile_renderer_warmed) {
            state.tile_renderer_warmed = true;
            return SDL_APP_CONTINUE;
        }

        if (!rendered) {
            std::println(stderr, "raster: scene rendering failed");
            return SDL_APP_FAILURE;
        }
        if (check_render_allocations && allocation_count != 0U) {
            std::println(
                stderr,
                "raster: render allocation trap: {} allocations",
                allocation_count
            );
            return SDL_APP_FAILURE;
        }

        static const bool profile =
            SDL_GetEnvironmentVariable(SDL_GetEnvironment(), "RASTER_PROFILE") != nullptr;
        const auto present_begin_ns = SDL_GetTicksNS();
        if (auto result = state.presenter.present(state.framebuffer); !result) {
            std::println(
                stderr,
                "raster: presentation failed ({})",
                static_cast<int>(result.error())
            );
            return SDL_APP_FAILURE;
        }

        const auto presented_ns = SDL_GetTicksNS();
        if (profile) {
            std::println(
                stderr,
                "raster: profile frame render {:.2f} ms present {:.2f} ms",
                static_cast<double>(present_begin_ns - frame_start_ns) / 1.0e6,
                static_cast<double>(presented_ns - present_begin_ns) / 1.0e6
            );
        }
        if (state.free_look_test) {
            const auto frame_time_ns = presented_ns - frame_start_ns;
            state.free_look_frame_times_ns[state.free_look_frame_time_sample_cursor] =
                frame_time_ns;
            state.free_look_frame_time_sample_cursor =
                (state.free_look_frame_time_sample_cursor + 1U) % free_look_timing_sample_capacity;
            state.free_look_frame_time_sample_count = std::min(
                state.free_look_frame_time_sample_count + 1U,
                free_look_timing_sample_capacity
            );

            const auto frame_time = static_cast<double>(frame_time_ns);
            state.free_look_frame_time_ema_ns = state.free_look_frame_time_ema_ns == 0.0
                ? frame_time
                : state.free_look_frame_time_ema_ns * 0.95 + frame_time * 0.05;
            if (frame_time_ns > free_look_frame_budget_ns) {
                ++state.free_look_over_budget_frames;
                state.free_look_headroom_since_ns = 0;
            } else {
                state.free_look_over_budget_frames = state.free_look_over_budget_frames > 0U
                    ? state.free_look_over_budget_frames - 1U
                    : 0U;
            }

            if (frame_time_ns < free_look_headroom_budget_ns) {
                if (state.free_look_headroom_since_ns == 0U) {
                    state.free_look_headroom_since_ns = frame_start_ns;
                }
            } else {
                state.free_look_headroom_since_ns = 0U;
            }

            const bool lower_resolution = state.free_look_over_budget_frames
                >= 4U
                && state.free_look_resolution_divisor
                < maximum_free_look_resolution_divisor;
            const bool sustained_headroom = state.free_look_headroom_since_ns
                != 0U
                && frame_start_ns
                - state.free_look_headroom_since_ns
                >= 5'000'000'000U;
            const bool scale_cooldown_elapsed = state.free_look_last_scale_change_ns
                == 0U
                || frame_start_ns
                - state.free_look_last_scale_change_ns
                >= 3'000'000'000U;
            const bool raise_resolution = sustained_headroom
                && state.free_look_resolution_divisor
                > minimum_free_look_resolution_divisor
                && state.free_look_frame_time_ema_ns
                < static_cast<double>(free_look_headroom_budget_ns);
            bool resolution_changed = false;
            if (scale_cooldown_elapsed && lower_resolution) {
                ++state.free_look_resolution_divisor;
                resolution_changed = true;
            } else if (scale_cooldown_elapsed && raise_resolution) {
                --state.free_look_resolution_divisor;
                resolution_changed = true;
            }

            if (resolution_changed) {
                if (!resize_free_look_target(state)) {
                    std::println(stderr, "raster: adaptive free-look resolution change failed");
                    return SDL_APP_FAILURE;
                }

                state.free_look_over_budget_frames = 0;
                state.free_look_headroom_since_ns = 0U;
                state.free_look_last_scale_change_ns = frame_start_ns;
                state.free_look_frame_time_ema_ns = 0.0;
                std::println(
                    "raster: free-look render scale 1/{} after sustained budget condition",
                    state.free_look_resolution_divisor
                );
            }
            if (frame_time_ns < free_look_frame_budget_ns) {
                SDL_DelayPrecise(free_look_frame_budget_ns - frame_time_ns);
            }
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
        const auto report_now_ns = state.free_look_test ? SDL_GetTicksNS() : now;
        const auto report_elapsed_ns = report_now_ns - state.report_start_ns;
        if (report_elapsed_ns >= 1'000'000'000U) {
            const auto fps = static_cast<double>(state.frame_count)
                * 1'000'000'000.0
                / static_cast<double>(report_elapsed_ns);
            if (state.free_look_test && state.free_look_frame_time_sample_count > 0U) {
                auto frame_times = state.free_look_frame_times_ns;
                std::ranges::sort(
                    frame_times.begin(),
                    frame_times.begin()
                        + static_cast<std::ptrdiff_t>(state.free_look_frame_time_sample_count)
                );

                const auto percentile_index = [&state](std::size_t percentile) noexcept {
                    const auto rank =
                        (state.free_look_frame_time_sample_count * percentile + 99U) / 100U;
                    return rank == 0U ? std::size_t{0} : rank - 1U;
                };
                std::println(
                    "raster: {:.1f} FPS; frame p95 {:.2f} ms, p99 {:.2f} ms; target {:.2f} ms; scale 1/{}",
                    fps,
                    static_cast<double>(frame_times[percentile_index(95U)]) / 1.0e6,
                    static_cast<double>(frame_times[percentile_index(99U)]) / 1.0e6,
                    static_cast<double>(free_look_frame_budget_ns) / 1.0e6,
                    state.free_look_resolution_divisor
                );

                state.free_look_frame_time_sample_count = 0U;
                state.free_look_frame_time_sample_cursor = 0U;
            } else {
                std::println("raster: {:.1f} FPS", fps);
            }

            state.frame_count = 0;
            state.report_start_ns = report_now_ns;
        }
        return SDL_APP_CONTINUE;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void SDLCALL application_quit(void* appstate, SDL_AppResult) {
        delete static_cast<AppState*>(appstate);
    }
} // namespace raster::platform
