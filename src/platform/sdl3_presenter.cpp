#include "raster/platform/sdl3_presenter.hpp"

#include <limits>

namespace raster {
    std::expected<void, PresenterError> WindowPresenter::initialize(
        int width, int height, Framebuffer& framebuffer
    ) noexcept {
        if (width <= 0 || height <= 0) {
            return std::unexpected{PresenterError::InvalidPixelSize};
        }
        if (!video_.initialize()) {
            return std::unexpected{PresenterError::VideoInitialization};
        }

        SDL_Window* raw_window = nullptr;
        SDL_Renderer* raw_renderer = nullptr;
        if (!SDL_CreateWindowAndRenderer(
                "raster",
                width,
                height,
                SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY,
                &raw_window,
                &raw_renderer
            )) {
            return std::unexpected{PresenterError::WindowCreation};
        }

        window_.reset(raw_window);
        renderer_.reset(raw_renderer);
        return resize_to_window_pixels(framebuffer);
    }

    std::expected<void, PresenterError> WindowPresenter::resize_to_window_pixels(
        Framebuffer& framebuffer
    ) noexcept {
        if (!window_) {
            return std::unexpected{PresenterError::WindowCreation};
        }

        int pixel_width = 0;
        int pixel_height = 0;
        if (!SDL_GetWindowSizeInPixels(window_.get(), &pixel_width, &pixel_height)) {
            return std::unexpected{PresenterError::PixelSizeQuery};
        }
        if (pixel_width <= 0 || pixel_height <= 0) {
            minimized_ = true;
            return {};
        }

        minimized_ = false;
        const auto new_width = static_cast<std::size_t>(pixel_width);
        const auto new_height = static_cast<std::size_t>(pixel_height);
        if (framebuffer.width != new_width || framebuffer.height != new_height) {
            if (new_width
                > static_cast<std::size_t>(std::numeric_limits<int>::max())
                / sizeof(std::uint32_t)) {
                return std::unexpected{PresenterError::InvalidPixelSize};
            }

            SDLTexturePtr new_texture{SDL_CreateTexture(
                renderer_.get(),
                SDL_PIXELFORMAT_ARGB8888,
                SDL_TEXTUREACCESS_STREAMING,
                pixel_width,
                pixel_height
            )};
            if (!new_texture || !SDL_SetTextureBlendMode(new_texture.get(), SDL_BLENDMODE_NONE)) {
                return std::unexpected{PresenterError::TextureCreation};
            }

            try {
                framebuffer.resize(new_width, new_height);
            } catch (...) {
                return std::unexpected{PresenterError::FramebufferResize};
            }
            texture_ = std::move(new_texture);
        } else if (!texture_) {
            return recreate_texture(new_width, new_height);
        }
        return {};
    }

    std::expected<void, PresenterError> WindowPresenter::present(
        const Framebuffer& framebuffer
    ) const noexcept {
        if (minimized_) {
            return {};
        }
        if (!renderer_ || !texture_) {
            return std::unexpected{PresenterError::Presentation};
        }

        const auto pitch = static_cast<int>(framebuffer.width * sizeof(std::uint32_t));
        const auto* const pixels = framebuffer.color_storage.data();
        if (!SDL_UpdateTexture(texture_.get(), nullptr, pixels, pitch)) {
            return std::unexpected{PresenterError::Presentation};
        }
        if (!SDL_RenderClear(renderer_.get())) {
            return std::unexpected{PresenterError::Presentation};
        }
        if (!SDL_RenderTexture(renderer_.get(), texture_.get(), nullptr, nullptr)) {
            return std::unexpected{PresenterError::Presentation};
        }
        if (!SDL_RenderPresent(renderer_.get())) {
            return std::unexpected{PresenterError::Presentation};
        }
        return {};
    }

    std::expected<void, PresenterError> WindowPresenter::recreate_texture(
        std::size_t width, std::size_t height
    ) noexcept {
        if (!renderer_
            || width
            == 0
            || height
            == 0
            || width
            > static_cast<std::size_t>(std::numeric_limits<int>::max())
            || height
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return std::unexpected{PresenterError::InvalidPixelSize};
        }

        SDLTexturePtr new_texture{SDL_CreateTexture(
            renderer_.get(),
            SDL_PIXELFORMAT_ARGB8888,
            SDL_TEXTUREACCESS_STREAMING,
            static_cast<int>(width),
            static_cast<int>(height)
        )};
        if (!new_texture) {
            return std::unexpected{PresenterError::TextureCreation};
        }
        if (!SDL_SetTextureBlendMode(new_texture.get(), SDL_BLENDMODE_NONE)) {
            return std::unexpected{PresenterError::TextureCreation};
        }

        texture_ = std::move(new_texture);
        return {};
    }
} // namespace raster
