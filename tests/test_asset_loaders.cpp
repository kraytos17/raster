#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <ranges>
#include <span>
#include <string_view>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"

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
        REQUIRE(mesh->triangle_count() == 2);
        REQUIRE(mesh->has_consistent_vertex_streams());
        REQUIRE(mesh->positions.size() == 4);
        REQUIRE(mesh->uvs.size() == 4);
        REQUIRE(mesh->normals.size() == 4);
        for (const auto index : mesh->indices) {
            REQUIRE(index < mesh->positions.size());
            REQUIRE(index < mesh->uvs.size());
            REQUIRE(index < mesh->normals.size());
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("bundled clipping sphere fixture loads as a high-poly mesh", "[asset][obj][m4]") {
        const auto mesh = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        REQUIRE(mesh.has_value());
        REQUIRE(mesh->triangle_count() == 16U * 32U * 2U);
        REQUIRE(mesh->has_consistent_vertex_streams());
        REQUIRE(mesh->positions.size() > 500U);
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
