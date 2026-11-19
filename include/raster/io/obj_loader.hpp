#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <flat_map>
#include <fstream>
#include <generator>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "raster/math/vec.hpp"
#include "raster/memory/soa_mesh.hpp"

namespace raster {
    enum class AssetError {
        FileNotFound,
        FileReadFailed,
        MalformedVertex,
        MalformedFace,
        MalformedImage,
        UnsupportedFaceIndex,
        UnsupportedFormat,
        IndexOutOfBounds,
        EmptyMesh,
        AllocationFailed,
    };

    struct ObjVertexKey {
        std::uint32_t position{};
        std::uint32_t uv{};
        std::uint32_t normal{};

        friend auto operator<=>(const ObjVertexKey&, const ObjVertexKey&) = default;
    };

    [[nodiscard]]
    inline std::expected<std::string, AssetError> read_file_raw(
        const std::filesystem::path& path
    ) noexcept {
        try {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input) {
                return std::unexpected{AssetError::FileNotFound};
            }

            const auto end_position = input.tellg();
            if (end_position < 0) {
                return std::unexpected{AssetError::FileReadFailed};
            }

            const auto length = static_cast<std::uintmax_t>(end_position);
            if (length > static_cast<std::uintmax_t>(std::string{}.max_size())) {
                return std::unexpected{AssetError::FileReadFailed};
            }

            std::string contents(static_cast<std::size_t>(length), '\0');
            input.seekg(0, std::ios::beg);
            if (!contents.empty()) {
                input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
                if (!input || input.gcount() != static_cast<std::streamsize>(contents.size())) {
                    return std::unexpected{AssetError::FileReadFailed};
                }
            }
            return contents;
        } catch (...) {
            return std::unexpected{AssetError::AllocationFailed};
        }
    }

    namespace detail {
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
                text.remove_prefix(1);
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
                text.remove_suffix(1);
            }
            return text;
        }

        [[nodiscard]]
        inline std::expected<float, AssetError> parse_float(std::string_view token) noexcept {
            float value{};
            const auto [end, error] = std::from_chars(
                token.data(),
                token.data() + token.size(),
                value,
                std::chars_format::general
            );
            if (error != std::errc{} || end != token.data() + token.size()) {
                return std::unexpected{AssetError::MalformedVertex};
            }
            return value;
        }

        [[nodiscard]]
        inline std::expected<std::int32_t, AssetError> parse_integer(
            std::string_view token
        ) noexcept {
            std::int32_t value{};
            const auto [end, error] =
                std::from_chars(token.data(), token.data() + token.size(), value);
            if (error != std::errc{} || end != token.data() + token.size() || value == 0) {
                return std::unexpected{AssetError::MalformedFace};
            }
            return value;
        }

        [[nodiscard]]
        inline std::expected<std::uint32_t, AssetError> resolve_obj_index(
            std::int32_t obj_index, std::size_t element_count
        ) noexcept {
            const auto count = static_cast<std::int64_t>(element_count);
            const auto index = static_cast<std::int64_t>(obj_index);
            const auto resolved = index > 0 ? index - 1 : count + index;
            if (resolved
                < 0
                || resolved
                >= count
                || std::cmp_greater(resolved, std::numeric_limits<std::uint32_t>::max())) {
                return std::unexpected{AssetError::IndexOutOfBounds};
            }
            return static_cast<std::uint32_t>(resolved);
        }

        struct RawFaceIndex {
            std::int32_t position{};
            std::int32_t uv{};
            std::int32_t normal{};
        };

        [[nodiscard]]
        inline std::expected<RawFaceIndex, AssetError> parse_face_index(
            std::string_view token
        ) noexcept {
            RawFaceIndex result{};
            const auto first_slash = token.find('/');
            if (first_slash == std::string_view::npos) {
                // The engine contract uses unified v/vt/vn face tuples. Bare
                // position-only faces are rejected rather than silently
                // fabricating normals or UVs.
                return std::unexpected{AssetError::UnsupportedFaceIndex};
            }

            const auto second_slash = token.find('/', first_slash + 1);
            if (second_slash
                == std::string_view::npos
                || second_slash
                == first_slash
                + 1
                || second_slash
                + 1
                >= token.size()
                || token.find('/', second_slash + 1)
                != std::string_view::npos) {
                return std::unexpected{AssetError::UnsupportedFaceIndex};
            }

            const auto position = parse_integer(token.substr(0, first_slash));
            const auto uv =
                parse_integer(token.substr(first_slash + 1, second_slash - first_slash - 1));
            const auto normal = parse_integer(token.substr(second_slash + 1));
            if (!position || !uv || !normal) {
                return std::unexpected{AssetError::MalformedFace};
            }

            result.position = *position;
            result.uv = *uv;
            result.normal = *normal;
            return result;
        }
    } // namespace detail

    [[nodiscard]]
    inline std::generator<std::string_view> obj_lines(std::string_view source) {
        std::size_t offset = 0;
        while (offset < source.size()) {
            const auto newline = source.find('\n', offset);
            const auto end = newline == std::string_view::npos ? source.size() : newline;
            auto line = detail::trim_ascii(source.substr(offset, end - offset));
            if (!line.empty() && line.front() != '#') {
                if (const auto comment = line.find('#'); comment != std::string_view::npos) {
                    line = detail::trim_ascii(line.substr(0, comment));
                }
                if (!line.empty()) {
                    co_yield line;
                }
            }
            offset = newline == std::string_view::npos ? source.size() : newline + 1;
        }
    }

    // Parses v/vt/vn triangle and polygon faces, resolving positive and
    // negative Wavefront indices and deduplicating unified tuples in a flat_map.
    // Polygon faces are triangulated as a fan (v0, vi, vi+1).
    [[nodiscard]]
    inline std::expected<SoAMesh, AssetError> parse_obj_buffer(std::string_view content) noexcept {
        try {
            std::vector<Vec3f> source_positions;
            std::vector<Vec3f> source_normals;
            std::vector<Vec2f> source_uvs;
            SoAMesh mesh;
            std::flat_map<ObjVertexKey, std::uint32_t> unified_vertices;

            const auto get_unified_index =
                [&](const detail::RawFaceIndex& raw) -> std::expected<std::uint32_t, AssetError> {
                const auto position =
                    detail::resolve_obj_index(raw.position, source_positions.size());
                const auto uv = detail::resolve_obj_index(raw.uv, source_uvs.size());
                const auto normal = detail::resolve_obj_index(raw.normal, source_normals.size());
                if (!position || !uv || !normal) {
                    return std::unexpected{AssetError::IndexOutOfBounds};
                }

                const ObjVertexKey key{.position = *position, .uv = *uv, .normal = *normal};
                if (const auto found = unified_vertices.find(key);
                    found != unified_vertices.end()) {
                    return found->second;
                }
                if (std::cmp_greater_equal(
                        mesh.positions.size(),
                        std::numeric_limits<std::uint32_t>::max()
                    )) {
                    return std::unexpected{AssetError::AllocationFailed};
                }

                const auto unified_index = static_cast<std::uint32_t>(mesh.positions.size());
                unified_vertices.emplace(key, unified_index);
                mesh.positions.push_back(source_positions[*position]);
                mesh.uvs.push_back(source_uvs[*uv]);
                mesh.normals.push_back(source_normals[*normal]);
                return unified_index;
            };

            for (std::string_view line : obj_lines(content)) {
                const auto separator = line.find_first_of(" \t");
                const auto keyword =
                    separator == std::string_view::npos ? line : line.substr(0, separator);
                auto payload = separator == std::string_view::npos
                    ? std::string_view{}
                    : detail::trim_ascii(line.substr(separator + 1));
                if (keyword == "v") {
                    const auto x_end = payload.find_first_of(" \t");
                    if (x_end == std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }

                    const auto y_start = payload.find_first_not_of(" \t", x_end);
                    if (y_start == std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }

                    const auto y_end = payload.find_first_of(" \t", y_start);
                    if (y_end == std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }

                    const auto z_start = payload.find_first_not_of(" \t", y_end);
                    if (z_start == std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }

                    const auto z_end = payload.find_first_of(" \t", z_start);
                    const auto x = detail::parse_float(payload.substr(0, x_end));
                    const auto y = detail::parse_float(payload.substr(y_start, y_end - y_start));
                    const auto z = detail::parse_float(payload.substr(
                        z_start,
                        z_end == std::string_view::npos ? payload.size() - z_start : z_end - z_start
                    ));
                    if (!x || !y || !z) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }
                    source_positions.push_back(Vec3f{.x = *x, .y = *y, .z = *z});
                } else if (keyword == "vt") {
                    const auto u_end = payload.find_first_of(" \t");
                    if (u_end == std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }

                    const auto v_start = payload.find_first_not_of(" \t", u_end);
                    if (v_start == std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }

                    const auto v_end = payload.find_first_of(" \t", v_start);
                    const auto u = detail::parse_float(payload.substr(0, u_end));
                    const auto v = detail::parse_float(payload.substr(
                        v_start,
                        v_end == std::string_view::npos ? payload.size() - v_start : v_end - v_start
                    ));
                    if (!u || !v) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }
                    source_uvs.push_back(Vec2f{.x = *u, .y = *v});
                } else if (keyword == "vn") {
                    const auto x_end = payload.find_first_of(" \t");
                    const auto y_start = x_end == std::string_view::npos
                        ? std::string_view::npos
                        : payload.find_first_not_of(" \t", x_end);
                    const auto y_end = y_start == std::string_view::npos
                        ? std::string_view::npos
                        : payload.find_first_of(" \t", y_start);
                    const auto z_start = y_end == std::string_view::npos
                        ? std::string_view::npos
                        : payload.find_first_not_of(" \t", y_end);
                    if (x_end
                        == std::string_view::npos
                        || y_start
                        == std::string_view::npos
                        || y_end
                        == std::string_view::npos
                        || z_start
                        == std::string_view::npos) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }

                    const auto z_end = payload.find_first_of(" \t", z_start);
                    const auto x = detail::parse_float(payload.substr(0, x_end));
                    const auto y = detail::parse_float(payload.substr(y_start, y_end - y_start));
                    const auto z = detail::parse_float(payload.substr(
                        z_start,
                        z_end == std::string_view::npos ? payload.size() - z_start : z_end - z_start
                    ));
                    if (!x || !y || !z) {
                        return std::unexpected{AssetError::MalformedVertex};
                    }
                    source_normals.push_back(Vec3f{.x = *x, .y = *y, .z = *z});
                } else if (keyword == "f") {
                    std::vector<std::uint32_t> face_indices;
                    while (!payload.empty()) {
                        const auto token_end = payload.find_first_of(" \t");
                        const auto token = payload.substr(0, token_end);
                        const auto raw_index = detail::parse_face_index(token);
                        if (!raw_index) {
                            return std::unexpected{raw_index.error()};
                        }

                        const auto unified_index = get_unified_index(*raw_index);
                        if (!unified_index) {
                            return std::unexpected{unified_index.error()};
                        }

                        face_indices.push_back(*unified_index);
                        if (token_end == std::string_view::npos) {
                            break;
                        }
                        payload = detail::trim_ascii(payload.substr(token_end + 1));
                    }
                    if (face_indices.size() < 3) {
                        return std::unexpected{AssetError::MalformedFace};
                    }
                    for (std::size_t index = 1; index + 1 < face_indices.size(); ++index) {
                        mesh.indices.push_back(face_indices[0]);
                        mesh.indices.push_back(face_indices[index]);
                        mesh.indices.push_back(face_indices[index + 1]);
                    }
                }
            }
            if (mesh.indices.empty()) {
                return std::unexpected{AssetError::EmptyMesh};
            }
            return mesh;
        } catch (...) {
            return std::unexpected{AssetError::AllocationFailed};
        }
    }

    [[nodiscard]]
    inline std::expected<SoAMesh, AssetError> load_obj(const std::filesystem::path& path) noexcept {
        auto contents = read_file_raw(path);
        if (!contents) {
            return std::unexpected{contents.error()};
        }
        return parse_obj_buffer(*contents);
    }
} // namespace raster
