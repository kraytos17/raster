#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <ranges>
#include <span>
#include <string_view>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/math/vec.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("OBJ line generator skips blank and comment lines", "[asset][obj][generator]") {
        constexpr std::string_view source =
            "\n  # comment\n v 1 2 3 # trailing comment\n\tvt 0.25 0.75\r\n";
        auto lines = raster::obj_lines(source);
        auto iterator = lines.begin();
        REQUIRE(iterator != lines.end());
        REQUIRE(*iterator == "v 1 2 3");
        ++iterator;
        REQUIRE(iterator != lines.end());
        REQUIRE(*iterator == "vt 0.25 0.75");
        ++iterator;
        REQUIRE(iterator == lines.end());
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("OBJ loader parses SoA streams and triangulates polygon faces", "[asset][obj]") {
        constexpr std::string_view obj = R"(
# quad with a repeated tuple and relative face indices
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1
f -4/-4/-1 -3/-3/-1 -2/-2/-1
)";

        const auto mesh = raster::parse_obj_buffer(obj);
        REQUIRE(mesh.has_value());
        REQUIRE(mesh->has_consistent_vertex_streams());
        REQUIRE(mesh->positions.size() == 4);
        REQUIRE(mesh->uvs.size() == 4);
        REQUIRE(mesh->normals.size() == 4);
        REQUIRE(mesh->triangle_count() == 3);
        constexpr std::array<std::uint32_t, 9> expected_indices{0, 1, 2, 0, 2, 3, 0, 1, 2};
        REQUIRE(mesh->indices.size() == expected_indices.size());
        for (const auto index : std::views::iota(std::size_t{0}, expected_indices.size())) {
            REQUIRE(mesh->indices[index] == expected_indices[index]);
        }
        REQUIRE(mesh->positions[2].x == 1.0F);
        REQUIRE(mesh->positions[2].y == 1.0F);
        REQUIRE(mesh->uvs[3].x == 0.0F);
        REQUIRE(mesh->normals[1].z == 1.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "bundled Milestone 4 OBJ fixture loads and passes all tuple indices", "[asset][obj]"
    ) {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/raster_milestone.obj"
        );
        REQUIRE(mesh.has_value());
        REQUIRE(mesh->triangle_count() == 384U);
        REQUIRE(mesh->has_consistent_vertex_streams());
        REQUIRE(mesh->positions.size() == 225U);
        REQUIRE(mesh->uvs.size() == 225U);
        REQUIRE(mesh->normals.size() == mesh->positions.size());
        for (const auto index : mesh->indices) {
            REQUIRE(index < mesh->positions.size());
            REQUIRE(index < mesh->uvs.size());
            REQUIRE(index < mesh->normals.size());
        }
        for (std::size_t triangle = 0; triangle < mesh->triangle_count(); ++triangle) {
            const auto ia = static_cast<std::size_t>(mesh->indices[triangle * 3U]);
            const auto ib = static_cast<std::size_t>(mesh->indices[triangle * 3U + 1U]);
            const auto ic = static_cast<std::size_t>(mesh->indices[triangle * 3U + 2U]);
            const auto face_normal = raster::cross(
                mesh->positions[ib] - mesh->positions[ia],
                mesh->positions[ic] - mesh->positions[ia]
            );
            REQUIRE(raster::dot(face_normal, mesh->normals[0]) > 0.0F);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "two-triangle floor fixture is loadable for long-triangle tests", "[asset][obj][floor]"
    ) {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/raster_milestone_2tri.obj"
        );
        REQUIRE(mesh.has_value());
        REQUIRE(mesh->triangle_count() == 2U);
        REQUIRE(mesh->positions.size() == 4U);
        REQUIRE(mesh->uvs.size() == 4U);
        REQUIRE(mesh->normals.size() == mesh->positions.size());
        for (std::size_t triangle = 0; triangle < mesh->triangle_count(); ++triangle) {
            const auto ia = static_cast<std::size_t>(mesh->indices[triangle * 3U]);
            const auto ib = static_cast<std::size_t>(mesh->indices[triangle * 3U + 1U]);
            const auto ic = static_cast<std::size_t>(mesh->indices[triangle * 3U + 2U]);
            const auto face_normal = raster::cross(
                mesh->positions[ib] - mesh->positions[ia],
                mesh->positions[ic] - mesh->positions[ia]
            );
            REQUIRE(raster::length_squared(face_normal) > 0.0F);
            REQUIRE(raster::dot(face_normal, mesh->normals[0]) > 0.0F);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "replacement checker TGA preserves diagnostic colors and markers", "[asset][tga][texture]"
    ) {
        const auto image =
            raster::load_tga(std::filesystem::path{RASTER_ASSET_DIR} / "textures/checker.tga");
        REQUIRE(image.has_value());
        REQUIRE(image->width == 512U);
        REQUIRE(image->height == 512U);
        REQUIRE(image->pixels.size() == 512U * 512U);

        bool saw_background = false;
        bool saw_red_marker = false;
        bool saw_green_channel_variation = false;
        for (const auto pixel : image->pixels) {
            const auto red = (pixel >> 16U) & 0xFFU;
            const auto green = (pixel >> 8U) & 0xFFU;
            const auto blue = pixel & 0xFFU;
            saw_background = saw_background || pixel == 0xFF14141CU;
            saw_red_marker = saw_red_marker || (red > 200U && red > green * 2U && red > blue * 2U);
            saw_green_channel_variation = saw_green_channel_variation || green > 120U;
        }
        REQUIRE(saw_background);
        REQUIRE(saw_red_marker);
        REQUIRE(saw_green_channel_variation);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("bundled clipping sphere fixture loads as a high-poly mesh", "[asset][obj][m4]") {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());
        REQUIRE(mesh->triangle_count() == 64U * (2U * 31U));
        REQUIRE(mesh->has_consistent_vertex_streams());
        REQUIRE(mesh->positions.size() > 500U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "clipping sphere has exact poles and nondegenerate cap triangles", "[asset][obj][sphere]"
    ) {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());

        constexpr std::size_t sectors = 64U;
        constexpr std::size_t stacks = 32U;
        const auto south_pole = mesh->positions[0];
        REQUIRE(std::hypot(south_pole.x, south_pole.z) < 1e-6F);
        REQUIRE(south_pole.y == -2.0F);

        // The first 32 triangles are the south cap and the final 32 are the
        // north cap: exactly one nondegenerate triangle per longitude sector.
        REQUIRE(mesh->triangle_count() == sectors * stacks * 2U - sectors * 2U);
        for (const auto face : std::views::iota(std::size_t{0}, sectors)) {
            const auto a = static_cast<std::size_t>(mesh->indices[face * 3U]);
            const auto b = static_cast<std::size_t>(mesh->indices[face * 3U + 1U]);
            const auto c = static_cast<std::size_t>(mesh->indices[face * 3U + 2U]);
            const auto ab = mesh->positions[b] - mesh->positions[a];
            const auto ac = mesh->positions[c] - mesh->positions[a];
            REQUIRE(raster::length_squared(raster::cross(ab, ac)) > 1e-8F);
            REQUIRE(mesh->uvs[a].x >= 0.0F);
            REQUIRE(mesh->uvs[a].x <= 1.0F);
            const auto min_u = std::min({mesh->uvs[a].x, mesh->uvs[b].x, mesh->uvs[c].x});
            const auto max_u = std::max({mesh->uvs[a].x, mesh->uvs[b].x, mesh->uvs[c].x});
            REQUIRE(max_u - min_u <= 1.0F / static_cast<float>(sectors) + 1e-6F);
        }

        const auto first_north_cap_face = sectors * (2U * (stacks - 2U)) + sectors;
        std::size_t north_pole_hits = 0;
        for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{3})) {
            const auto index =
                static_cast<std::size_t>(mesh->indices[first_north_cap_face * 3U + corner]);
            const auto& point = mesh->positions[index];
            if (point.y > 1.99999F && std::hypot(point.x, point.z) < 1e-6F) {
                ++north_pole_hits;
            }
        }
        REQUIRE(north_pole_hits == 1U);

        const auto first_regular_face = sectors;
        const auto a = static_cast<std::size_t>(mesh->indices[first_regular_face * 3U]);
        const auto b = static_cast<std::size_t>(mesh->indices[first_regular_face * 3U + 1U]);
        const auto c = static_cast<std::size_t>(mesh->indices[first_regular_face * 3U + 2U]);
        const auto ab = mesh->positions[b] - mesh->positions[a];
        const auto ac = mesh->positions[c] - mesh->positions[a];
        const auto face_normal = raster::cross(ab, ac);
        const auto face_center = mesh->positions[a] + mesh->positions[b] + mesh->positions[c];
        REQUIRE(raster::dot(face_normal, face_center) > 0.0F); // outward winding
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("clipping sphere longitude seam has continuous face UVs", "[asset][obj][uv-seam]") {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());
        for (std::size_t face = 0; face < mesh->triangle_count(); ++face) {
            float minimum_u = 1.0F;
            float maximum_u = 0.0F;
            for (const auto corner : std::views::iota(std::size_t{0}, std::size_t{3})) {
                const auto index = static_cast<std::size_t>(mesh->indices[face * 3U + corner]);
                minimum_u = std::min(minimum_u, mesh->uvs[index].x);
                maximum_u = std::max(maximum_u, mesh->uvs[index].x);
            }
            REQUIRE(maximum_u - minimum_u <= 1.0F / 64.0F + 1e-6F);
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("OBJ loader reports malformed and unsupported data", "[asset][obj][errors]") {
        const auto malformed_vertex = raster::parse_obj_buffer("v 0 nope 0\n");
        REQUIRE_FALSE(malformed_vertex.has_value());
        REQUIRE(malformed_vertex.error() == raster::AssetError::MalformedVertex);

        constexpr std::string_view missing_normal = R"(
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
)";
        const auto unsupported = raster::parse_obj_buffer(missing_normal);
        REQUIRE_FALSE(unsupported.has_value());
        REQUIRE(unsupported.error() == raster::AssetError::UnsupportedFaceIndex);

        constexpr std::string_view out_of_bounds = R"(
v 0 0 0
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 1/1/1
)";
        const auto out_of_range = raster::parse_obj_buffer(out_of_bounds);
        REQUIRE_FALSE(out_of_range.has_value());
        REQUIRE(out_of_range.error() == raster::AssetError::IndexOutOfBounds);

        const auto empty = raster::parse_obj_buffer("# no geometry\n");
        REQUIRE_FALSE(empty.has_value());
        REQUIRE(empty.error() == raster::AssetError::EmptyMesh);
    }

    [[nodiscard]]
    std::array<std::byte, 18 + 2 * 2 * 3> make_tga_2x2(
        std::uint8_t descriptor, const std::array<std::array<std::uint8_t, 3>, 4>& bgr
    ) {
        std::array<std::byte, 18 + 2 * 2 * 3> bytes{};
        bytes[2] = std::byte{2}; // uncompressed true-color
        bytes[12] = std::byte{2}; // width = 2, little-endian
        bytes[14] = std::byte{2}; // height = 2, little-endian
        bytes[16] = std::byte{24};
        bytes[17] = static_cast<std::byte>(descriptor);
        for (std::size_t pixel = 0; pixel < bgr.size(); ++pixel) {
            const auto offset = 18 + pixel * 3;
            bytes[offset] = static_cast<std::byte>(bgr[pixel][0]);
            bytes[offset + 1] = static_cast<std::byte>(bgr[pixel][1]);
            bytes[offset + 2] = static_cast<std::byte>(bgr[pixel][2]);
        }
        return bytes;
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("TGA loader converts BGR and normalizes bottom-left origin", "[asset][tga]") {
        // Source rows in bottom-left-origin order: red, green, blue, white.
        constexpr std::array<std::array<std::uint8_t, 3>, 4> bgr{
            std::array<std::uint8_t, 3>{0, 0, 255},
            std::array<std::uint8_t, 3>{0, 255, 0},
            std::array<std::uint8_t, 3>{255, 0, 0},
            std::array<std::uint8_t, 3>{255, 255, 255},
        };
        const auto bytes = make_tga_2x2(0, bgr);
        const auto image = raster::parse_tga_buffer(bytes);
        REQUIRE(image.has_value());
        REQUIRE(image->width == 2);
        REQUIRE(image->height == 2);
        REQUIRE(image->pixels.size() == 4);
        REQUIRE(image->view()[1, 0] == 0xFFFF0000U);
        REQUIRE(image->view()[1, 1] == 0xFF00FF00U);
        REQUIRE(image->view()[0, 0] == 0xFF0000FFU);
        REQUIRE(image->view()[0, 1] == 0xFFFFFFFFU);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): verify aligned allocation.
        REQUIRE(reinterpret_cast<std::uintptr_t>(image->pixels.data()) % 64U == 0U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "TGA loader normalizes top-right origin and rejects malformed payload", "[asset][tga]"
    ) {
        constexpr std::array<std::array<std::uint8_t, 3>, 4> bgr{
            std::array<std::uint8_t, 3>{0, 0, 255},
            std::array<std::uint8_t, 3>{0, 255, 0},
            std::array<std::uint8_t, 3>{255, 0, 0},
            std::array<std::uint8_t, 3>{255, 255, 255},
        };
        const auto bytes = make_tga_2x2(0x30U, bgr); // top + right origin
        const auto image = raster::parse_tga_buffer(bytes);
        REQUIRE(image.has_value());
        REQUIRE(image->view()[0, 0] == 0xFF00FF00U);
        REQUIRE(image->view()[0, 1] == 0xFFFF0000U);
        REQUIRE(image->view()[1, 0] == 0xFFFFFFFFU);
        REQUIRE(image->view()[1, 1] == 0xFF0000FFU);

        const auto truncated =
            raster::parse_tga_buffer(std::span<const std::byte>{bytes}.first(20));
        REQUIRE_FALSE(truncated.has_value());
        REQUIRE(truncated.error() == raster::AssetError::MalformedImage);

        auto unsupported_bytes = bytes;
        unsupported_bytes[2] = std::byte{10}; // RLE true-color is intentionally unsupported.
        const auto unsupported = raster::parse_tga_buffer(unsupported_bytes);
        REQUIRE_FALSE(unsupported.has_value());
        REQUIRE(unsupported.error() == raster::AssetError::UnsupportedFormat);
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
