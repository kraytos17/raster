#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <memory>
#include <new>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace raster {
    inline constexpr std::size_t cache_line_size = 64;

    template <typename T> class AlignedAllocator {
    public:

        using value_type = T;
        using is_always_equal = std::true_type;

        constexpr AlignedAllocator() noexcept = default;

        template <typename U>
        explicit constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

        [[nodiscard]]
        T* allocate(std::size_t count) {
            if (count > max_size()) {
                throw std::bad_array_new_length{};
            }
            return static_cast<T*>(
                ::operator new(count * sizeof(T), std::align_val_t{cache_line_size})
            );
        }

        void deallocate(T* ptr, std::size_t) noexcept {
            ::operator delete(ptr, std::align_val_t{cache_line_size});
        }

        [[nodiscard]]
        constexpr std::size_t max_size() const noexcept {
            return static_cast<std::size_t>(-1) / sizeof(T);
        }

        template <typename U> struct rebind {
            using other = AlignedAllocator<U>;
        };

        template <typename U> friend constexpr bool operator==(
            const AlignedAllocator&, const AlignedAllocator<U>&
        ) noexcept {
            return true;
        }
    };

    using ColorStorage = std::vector<std::uint32_t, AlignedAllocator<std::uint32_t>>;
    using DepthStorage = std::vector<float, AlignedAllocator<float>>;
    using Extents2D = std::dextents<std::size_t, 2>;

    struct Framebuffer {
        std::size_t width{};
        std::size_t height{};
        ColorStorage color_storage;
        DepthStorage depth_storage;

        Framebuffer(std::size_t framebuffer_width, std::size_t framebuffer_height) :
            width(framebuffer_width),
            height(framebuffer_height),
            color_storage(pixel_count(framebuffer_width, framebuffer_height), 0xFF000000U),
            depth_storage(pixel_count(framebuffer_width, framebuffer_height), 0.0F) {
            if (width == 0 || height == 0) {
                throw std::invalid_argument("framebuffer dimensions must be non-zero");
            }
        }

        template <typename Self> [[nodiscard]]
        auto color_view(this Self& self) noexcept {
            using Element = std::conditional_t<
                std::is_const_v<std::remove_reference_t<Self>>,
                const std::uint32_t,
                std::uint32_t
            >;
            auto* data = std::assume_aligned<cache_line_size>(self.color_storage.data());
            return std::mdspan<Element, Extents2D>{data, self.height, self.width};
        }

        template <typename Self> [[nodiscard]]
        auto depth_view(this Self& self) noexcept {
            using Element = std::
                conditional_t<std::is_const_v<std::remove_reference_t<Self>>, const float, float>;
            auto* data = std::assume_aligned<cache_line_size>(self.depth_storage.data());
            return std::mdspan<Element, Extents2D>{data, self.height, self.width};
        }

        void resize(std::size_t new_width, std::size_t new_height) {
            const auto count = pixel_count(new_width, new_height);
            if (new_width == 0 || new_height == 0) {
                return;
            }

            ColorStorage new_color(count, 0xFF000000U);
            DepthStorage new_depth(count, 0.0F);
            color_storage.swap(new_color);
            depth_storage.swap(new_depth);
            width = new_width;
            height = new_height;
        }

        void clear_tile(
            std::size_t tile_x,
            std::size_t tile_y,
            std::uint32_t clear_color = 0xFF181820U,
            float clear_depth = 0.0F
        ) noexcept {
            constexpr std::size_t tile_size = 16;
            if (tile_x > width / tile_size || tile_y > height / tile_size) {
                return;
            }

            const auto start_x = tile_x * tile_size;
            const auto start_y = tile_y * tile_size;
            if (start_x >= width || start_y >= height) {
                return;
            }

            const auto end_x = std::min(start_x + tile_size, width);
            const auto end_y = std::min(start_y + tile_size, height);
            auto color = color_view();
            auto depth = depth_view();
            for (const auto y : std::views::iota(start_y, end_y)) {
                for (const auto x : std::views::iota(start_x, end_x)) {
                    color[y, x] = clear_color;
                    depth[y, x] = clear_depth;
                }
            }
        }

    private:

        [[nodiscard]]
        static std::size_t pixel_count(std::size_t width, std::size_t height) {
            if (width != 0 && height > static_cast<std::size_t>(-1) / width) {
                throw std::length_error("framebuffer dimensions overflow storage size");
            }
            return width * height;
        }
    };
} // namespace raster
