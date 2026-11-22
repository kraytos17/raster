#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <expected>
#include <filesystem>
#include <optional>
#include <string_view>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/pipeline/pbr.hpp"

namespace raster {
    struct LoadedPbrMaterial {
        pbr::Material factors{};
        std::optional<Texture2D<LayoutTiled4x4>> base_color_texture{};
        std::optional<Texture2D<LayoutTiled4x4>> metallic_roughness_texture{};
    };

    namespace material_detail {
        [[nodiscard]]
        inline std::string_view trim_ascii(std::string_view text) noexcept {
            while (!text.empty()
                   && (text.front()
                       == ' '
                       || text.front()
                       == '\t'
                       || text.front()
                       == '\r'
                       || text.front()
                       == '\n')) {
                text.remove_prefix(1U);
            }
            while (!text.empty()
                   && (text.back()
                       == ' '
                       || text.back()
                       == '\t'
                       || text.back()
                       == '\r'
                       || text.back()
                       == '\n')) {
                text.remove_suffix(1U);
            }
            return text;
        }

        [[nodiscard]]
        inline std::expected<std::filesystem::path, AssetError> resolve_texture_path(
            const std::filesystem::path& base_directory, std::string_view relative
        ) noexcept {
            try {
                const std::filesystem::path relative_path{relative};
                if (relative.empty()
                    || relative_path.is_absolute()
                    || relative_path.has_root_name()) {
                    return std::unexpected{AssetError::InvalidMaterialPath};
                }
                return (base_directory / relative_path).lexically_normal();
            } catch (...) {
                return std::unexpected{AssetError::InvalidMaterialPath};
            }
        }

        [[nodiscard]]
        inline std::expected<float, AssetError> parse_factor(std::string_view token) noexcept {
            float value{};
            const auto [end, error] = std::from_chars(
                token.data(),
                token.data() + token.size(),
                value,
                std::chars_format::general
            );
            if (error
                != std::errc{}
                || end
                != token.data()
                + token.size()
                || !std::isfinite(value)) {
                return std::unexpected{AssetError::MalformedMaterial};
            }
            return value;
        }
    } // namespace material_detail

    // Format is project-owned (not OBJ MTL or glTF): `base_color r g b`,
    // `metallic m`, `roughness r`, `base_color_texture path`, and
    // `metallic_roughness_texture path`. Texture paths are relative to the
    // descriptor. Per-face materials are not represented; this descriptor
    // applies to one renderer draw.
    [[nodiscard]]
    inline std::expected<LoadedPbrMaterial, AssetError> load_pbr_material_buffer(
        std::string_view source, const std::filesystem::path& base_directory
    ) noexcept {
        try {
            LoadedPbrMaterial material{};
            std::size_t offset = 0U;
            while (offset < source.size()) {
                const auto newline = source.find('\n', offset);
                const auto end = newline == std::string_view::npos ? source.size() : newline;
                auto line = material_detail::trim_ascii(source.substr(offset, end - offset));
                offset = newline == std::string_view::npos ? source.size() : newline + 1U;
                if (line.empty() || line.front() == '#') {
                    continue;
                }
                if (const auto comment = line.find('#'); comment != std::string_view::npos) {
                    line = material_detail::trim_ascii(line.substr(0U, comment));
                }
                if (line.empty()) {
                    continue;
                }

                const auto separator = line.find_first_of(" \t");
                if (separator == std::string_view::npos) {
                    return std::unexpected{AssetError::MalformedMaterial};
                }

                const auto key = line.substr(0U, separator);
                auto payload = material_detail::trim_ascii(line.substr(separator + 1U));
                if (key == "base_color") {
                    std::array<float, 3> channels{};
                    for (auto& channel : channels) {
                        if (payload.empty()) {
                            return std::unexpected{AssetError::MalformedMaterial};
                        }

                        const auto token_end = payload.find_first_of(" \t");
                        const auto token = token_end == std::string_view::npos
                            ? payload
                            : payload.substr(0U, token_end);
                        const auto parsed = material_detail::parse_factor(token);
                        if (!parsed) {
                            return std::unexpected{parsed.error()};
                        }

                        channel = pbr::saturate(*parsed);
                        payload = token_end == std::string_view::npos
                            ? std::string_view{}
                            : material_detail::trim_ascii(payload.substr(token_end + 1U));
                    }
                    if (!payload.empty()) {
                        return std::unexpected{AssetError::MalformedMaterial};
                    }
                    material.factors.base_color =
                        Vec3f{.x = channels[0], .y = channels[1], .z = channels[2]};
                } else if (key == "metallic" || key == "roughness") {
                    if (payload.empty() || payload.find_first_of(" \t") != std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedMaterial};
                    }

                    const auto parsed = material_detail::parse_factor(payload);
                    if (!parsed) {
                        return std::unexpected{parsed.error()};
                    }
                    if (key == "metallic") {
                        material.factors.metallic = pbr::saturate(*parsed);
                    } else {
                        material.factors.perceptual_roughness = pbr::saturate(*parsed);
                    }
                } else if (key == "base_color_texture" || key == "metallic_roughness_texture") {
                    if (payload.empty() || payload.find_first_of(" \t") != std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedMaterial};
                    }

                    const auto path =
                        material_detail::resolve_texture_path(base_directory, payload);
                    if (!path) {
                        return std::unexpected{path.error()};
                    }

                    auto image = load_tga(*path);
                    if (!image) {
                        return std::unexpected{image.error()};
                    }
                    if (key == "base_color_texture") {
                        material.base_color_texture.emplace(*image, TextureColorSpace::Srgb);
                    } else {
                        material.metallic_roughness_texture.emplace(
                            *image,
                            TextureColorSpace::EncodedBytes
                        );
                    }
                } else {
                    return std::unexpected{AssetError::MalformedMaterial};
                }
            }
            if (source.empty()) {
                return std::unexpected{AssetError::MalformedMaterial};
            }
            return material;
        } catch (...) {
            return std::unexpected{AssetError::AllocationFailed};
        }
    }

    [[nodiscard]]
    inline std::expected<LoadedPbrMaterial, AssetError> load_pbr_material(
        const std::filesystem::path& descriptor_path
    ) noexcept {
        try {
            auto contents = read_file_raw(descriptor_path);
            if (!contents) {
                return std::unexpected{contents.error()};
            }

            const auto parent = [&] {
                auto p = descriptor_path.parent_path();
                if (p.empty()) {
                    return std::filesystem::current_path();
                }
                return p;
            }();
            return load_pbr_material_buffer(*contents, parent);
        } catch (...) {
            return std::unexpected{AssetError::FileReadFailed};
        }
    }
} // namespace raster
