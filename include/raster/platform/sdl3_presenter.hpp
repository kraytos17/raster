#pragma once

#include <expected>
#include <memory>
#include <SDL3/SDL.h>

#include "raster/memory/framebuffer.hpp"

namespace raster {
    enum class PresenterError {
        VideoInitialization,
        WindowCreation,
        PixelSizeQuery,
        InvalidPixelSize,
        FramebufferResize,
        TextureCreation,
        Presentation,
    };

    class SdlVideoGuard {
    public:

        SdlVideoGuard() noexcept = default;

        ~SdlVideoGuard() {
            if (initialized_) {
                SDL_Quit();
            }
        }

        SdlVideoGuard(const SdlVideoGuard&) = delete;
        SdlVideoGuard& operator=(const SdlVideoGuard&) = delete;
        SdlVideoGuard(SdlVideoGuard&&) = delete;
        SdlVideoGuard& operator=(SdlVideoGuard&&) = delete;

        [[nodiscard]]
        bool initialize() noexcept {
            initialized_ = SDL_Init(SDL_INIT_VIDEO);
            return initialized_;
        }

    private:

        bool initialized_{};
    };

    struct SDLWindowDeleter {
        void operator()(SDL_Window* window) const noexcept {
            SDL_DestroyWindow(window);
        }
    };

    struct SDLRendererDeleter {
        void operator()(SDL_Renderer* renderer) const noexcept {
            SDL_DestroyRenderer(renderer);
        }
    };

    struct SDLTextureDeleter {
        void operator()(SDL_Texture* texture) const noexcept {
            SDL_DestroyTexture(texture);
        }
    };

    using SDLWindowPtr = std::unique_ptr<SDL_Window, SDLWindowDeleter>;
    using SDLRendererPtr = std::unique_ptr<SDL_Renderer, SDLRendererDeleter>;
    using SDLTexturePtr = std::unique_ptr<SDL_Texture, SDLTextureDeleter>;

    class WindowPresenter {
    public:

        WindowPresenter() noexcept = default;

        [[nodiscard]]
        std::expected<void, PresenterError> initialize(
            int width, int height, Framebuffer& framebuffer
        ) noexcept;

        [[nodiscard]]
        std::expected<void, PresenterError> resize_to_window_pixels(
            Framebuffer& framebuffer
        ) noexcept;

        [[nodiscard]]
        std::expected<void, PresenterError> present(const Framebuffer& framebuffer) const noexcept;

        // Selects vertical-sync presentation (true) or immediate uncapped
        // presentation (false).
        [[nodiscard]]
        std::expected<void, PresenterError> set_vsync(bool enabled) noexcept;

        void set_minimized(bool minimized) noexcept {
            minimized_ = minimized;
        }

        [[nodiscard]]
        bool is_minimized() const noexcept {
            return minimized_;
        }

        [[nodiscard]]
        SDL_Window* window() const noexcept {
            return window_.get();
        }

    private:

        SdlVideoGuard video_;
        SDLWindowPtr window_;
        SDLRendererPtr renderer_;
        SDLTexturePtr texture_;
        bool minimized_{};

        [[nodiscard]]
        std::expected<void, PresenterError> recreate_texture(
            std::size_t width, std::size_t height
        ) noexcept;
    };
} // namespace raster
