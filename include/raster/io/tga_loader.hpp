#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <limits>
#include <span>

#include "raster/io/obj_loader.hpp"
#include "raster/memory/framebuffer.hpp"

namespace raster {
    struct TgaImage {
        std::size_t width{};
        std::size_t height{};
        ColorStorage pixels{}; // native 0xAARRGGBB, 64B-aligned

        [[nodiscard]]
        auto view() noexcept {
            auto* const data = std::assume_aligned<cache_line_size>(pixels.data());
            return std::mdspan<std::uint32_t, Extents2D>{data, height, width};
        }

        [[nodiscard]]
        auto view() const noexcept {
            const auto* const data = std::assume_aligned<cache_line_size>(pixels.data());
            return std::mdspan<const std::uint32_t, Extents2D>{data, height, width};
        }
    };

    namespace detail {
        [[nodiscard]]
        inline std::uint8_t byte_value(std::byte value) noexcept {
            return std::to_integer<std::uint8_t>(value);
        }

        [[nodiscard]]
        inline std::expected<std::uint16_t, AssetError> read_u16_le(
            std::span<const std::byte> bytes, std::size_t offset
        ) noexcept {
            if (offset >= bytes.size() || bytes.size() - offset < sizeof(std::uint16_t)) {
                return std::unexpected{AssetError::MalformedImage};
            }

            std::array<std::byte, sizeof(std::uint16_t)> raw{
                bytes[offset],
                bytes[offset + 1],
            };
            auto value = std::bit_cast<std::uint16_t>(raw);
            if constexpr (std::endian::native == std::endian::big) {
                value = std::byteswap(value);
            }
            return value;
        }
    } // namespace detail

    // Loads uncompressed true-color TGA types 2/10? (type 2 only) with 24- or
    // 32-bit pixels, converting BGR(A) bytes to native 0xAARRGGBB. The image
    // descriptor's horizontal/vertical origin bits are normalized to top-left
    // row-major storage.
    [[nodiscard]]
    inline std::expected<TgaImage, AssetError> parse_tga_buffer(
        std::span<const std::byte> bytes
    ) noexcept {
        constexpr std::size_t header_size = 18;
        if (bytes.size() < header_size) {
            return std::unexpected{AssetError::MalformedImage};
        }
        try {
            const auto id_length = detail::byte_value(bytes[0]);
            const auto color_map_type = detail::byte_value(bytes[1]);
            const auto image_type = detail::byte_value(bytes[2]);
            const auto width_result = detail::read_u16_le(bytes, 12);
            const auto height_result = detail::read_u16_le(bytes, 14);
            const auto bits_per_pixel = detail::byte_value(bytes[16]);
            const auto descriptor = detail::byte_value(bytes[17]);
            if (!width_result || !height_result) {
                return std::unexpected{AssetError::MalformedImage};
            }

            const auto width = static_cast<std::size_t>(*width_result);
            const auto height = static_cast<std::size_t>(*height_result);
            if (color_map_type
                != 0
                || image_type
                != 2
                || width
                == 0
                || height
                == 0
                || (bits_per_pixel != 24 && bits_per_pixel != 32)) {
                return std::unexpected{AssetError::UnsupportedFormat};
            }

            const auto bytes_per_pixel = static_cast<std::size_t>(bits_per_pixel / 8U);
            if (height > std::numeric_limits<std::size_t>::max() / width) {
                return std::unexpected{AssetError::MalformedImage};
            }

            const auto pixel_count = width * height;
            if (pixel_count > std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
                return std::unexpected{AssetError::MalformedImage};
            }

            const auto payload_offset = header_size + static_cast<std::size_t>(id_length);
            const auto payload_size = pixel_count * bytes_per_pixel;
            if (payload_offset > bytes.size() || bytes.size() - payload_offset < payload_size) {
                return std::unexpected{AssetError::MalformedImage};
            }

            TgaImage image{};
            image.width = width;
            image.height = height;
            image.pixels.resize(pixel_count);
            const bool top_origin = (descriptor & 0x20U) != 0;
            const bool right_origin = (descriptor & 0x10U) != 0;
            for (std::size_t source_y = 0; source_y < height; ++source_y) {
                for (std::size_t source_x = 0; source_x < width; ++source_x) {
                    const auto source_index = source_y * width + source_x;
                    const auto source_offset = payload_offset + source_index * bytes_per_pixel;
                    const auto blue = detail::byte_value(bytes[source_offset]);
                    const auto green = detail::byte_value(bytes[source_offset + 1]);
                    const auto red = detail::byte_value(bytes[source_offset + 2]);
                    const auto alpha = bytes_per_pixel == 4
                        ? detail::byte_value(bytes[source_offset + 3])
                        : std::uint8_t{0xFFU};
                    const auto target_x = right_origin ? width - 1 - source_x : source_x;
                    const auto target_y = top_origin ? source_y : height - 1 - source_y;
                    image.pixels[target_y * width + target_x] =
                        (static_cast<std::uint32_t>(alpha) << 24U)
                        | (static_cast<std::uint32_t>(red) << 16U)
                        | (static_cast<std::uint32_t>(green) << 8U)
                        | static_cast<std::uint32_t>(blue);
                }
            }
            return image;
        } catch (...) {
            return std::unexpected{AssetError::AllocationFailed};
        }
    }

    [[nodiscard]]
    inline std::expected<TgaImage, AssetError> load_tga(
        const std::filesystem::path& path
    ) noexcept {
        auto contents = read_file_raw(path);
        if (!contents) {
            return std::unexpected{contents.error()};
        }
        return parse_tga_buffer(std::as_bytes(std::span{contents->data(), contents->size()}));
    }
} // namespace raster
