#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <ranges>
#include <vector>

#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    template <typename Mapping> concept RequiredTextureMapping =
        requires(const Mapping& mapping, Mapping::index_type row, Mapping::index_type column) {
            { mapping(row, column) } -> std::same_as<typename Mapping::index_type>;
            { mapping.required_span_size() } -> std::same_as<typename Mapping::size_type>;
            { mapping.is_unique() } -> std::same_as<bool>;
            { mapping.is_exhaustive() } -> std::same_as<bool>;
        };

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "Morton and tiled texture layout mappings are unique and in span", "[texture][layout]"
    ) {
        using Extents = std::dextents<std::size_t, 2>;
        using MortonMapping = raster::LayoutMorton::mapping<Extents>;
        using TiledMapping = raster::LayoutTiled4x4::mapping<Extents>;
        STATIC_REQUIRE(RequiredTextureMapping<MortonMapping>);
        STATIC_REQUIRE(RequiredTextureMapping<TiledMapping>);

        const Extents extents{8U, 8U};
        const MortonMapping morton{extents};
        const TiledMapping tiled{extents};
        std::vector<std::uint8_t> morton_seen(morton.required_span_size());
        std::vector<std::uint8_t> tiled_seen(tiled.required_span_size());
        for (const auto y : std::views::iota(std::size_t{0}, std::size_t{8})) {
            for (const auto x : std::views::iota(std::size_t{0}, std::size_t{8})) {
                const auto morton_index = static_cast<std::size_t>(morton(y, x));
                const auto tiled_index = static_cast<std::size_t>(tiled(y, x));
                REQUIRE(morton_index < morton_seen.size());
                REQUIRE(tiled_index < tiled_seen.size());
                REQUIRE(morton_seen[morton_index] == 0U);
                REQUIRE(tiled_seen[tiled_index] == 0U);
                morton_seen[morton_index] = 1U;
                tiled_seen[tiled_index] = 1U;
            }
        }
        REQUIRE(std::ranges::all_of(morton_seen, [](auto value) { return value == 1U; }));
        REQUIRE(std::ranges::all_of(tiled_seen, [](auto value) { return value == 1U; }));
        REQUIRE(morton.is_exhaustive());
        REQUIRE(tiled.is_exhaustive());
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("swizzled layouts are bijective for padded dimensions", "[texture][layout]") {
        using Extents = std::dextents<std::size_t, 2>;
        const Extents extents{5U, 7U};
        const raster::LayoutMorton::mapping<Extents> morton{extents};
        const raster::LayoutTiled4x4::mapping<Extents> tiled{extents};
        std::vector<std::uint8_t> morton_seen(morton.required_span_size());
        std::vector<std::uint8_t> tiled_seen(tiled.required_span_size());
        for (const auto y : std::views::iota(std::size_t{0}, std::size_t{5})) {
            for (const auto x : std::views::iota(std::size_t{0}, std::size_t{7})) {
                const auto mi = static_cast<std::size_t>(morton(y, x));
                const auto ti = static_cast<std::size_t>(tiled(y, x));
                REQUIRE(morton_seen[mi] == 0U);
                REQUIRE(tiled_seen[ti] == 0U);
                morton_seen[mi] = 1U;
                tiled_seen[ti] = 1U;
            }
        }
        REQUIRE(std::ranges::count(morton_seen, std::uint8_t{1}) == 35);
        REQUIRE(std::ranges::count(tiled_seen, std::uint8_t{1}) == 35);
        REQUIRE_FALSE(morton.is_exhaustive());
        REQUIRE_FALSE(tiled.is_exhaustive());
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("texture mip chain reaches 1x1 and nearest samples wrap", "[texture][mipmap]") {
        raster::TgaImage image{};
        image.width = 4;
        image.height = 4;
        image.pixels.resize(16U);
        for (const auto index : std::views::iota(std::size_t{0}, image.pixels.size())) {
            image.pixels[index] = index < 8U ? 0xFF000000U : 0xFFFFFFFFU;
        }
        const raster::Texture2D<std::layout_right> texture{image};
        REQUIRE(texture.level_count() == 3U);
        REQUIRE(texture.width(0) == 4U);
        REQUIRE(texture.height(0) == 4U);
        REQUIRE(texture.width(1) == 2U);
        REQUIRE(texture.height(1) == 2U);
        REQUIRE(texture.width(2) == 1U);
        REQUIRE(texture.height(2) == 1U);
        REQUIRE(texture.sample_nearest(0.1F, 0.1F) == 0xFF000000U);
        REQUIRE(texture.sample_nearest(1.1F, -0.9F) == 0xFF000000U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("2x2 quad UV derivatives and mip LOD are finite", "[texture][derivatives]") {
        const std::array<raster::Vec2f, 4> uv{
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.25F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 0.5F},
            raster::Vec2f{.x = 0.25F, .y = 0.5F},
        };
        const auto derivatives = raster::quad_derivatives(uv);
        REQUIRE(derivatives.duv_dx.x == 0.25F);
        REQUIRE(derivatives.duv_dx.y == 0.0F);
        REQUIRE(derivatives.duv_dy.x == 0.0F);
        REQUIRE(derivatives.duv_dy.y == 0.5F);
        REQUIRE(raster::mip_lod(derivatives, 256, 256) == 7.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE(
        "mipmapped texture samples a coarser level at minification", "[texture][mipmap][render]"
    ) {
        raster::TgaImage image{};
        image.width = 64;
        image.height = 64;
        image.pixels.resize(image.width * image.height);
        for (const auto y : std::views::iota(std::size_t{0}, image.height)) {
            for (const auto x : std::views::iota(std::size_t{0}, image.width)) {
                const auto color = ((x + y) % 2U) == 0U ? 0xFFFFFFFFU : 0xFF000000U;
                image.pixels[y * image.width + x] = color;
            }
        }

        const raster::Texture2D<> texture{image};
        const std::array<raster::Vec2f, 4> minified_uv{
            raster::Vec2f{.x = 0.0F, .y = 0.0F},
            raster::Vec2f{.x = 0.5F, .y = 0.0F},
            raster::Vec2f{.x = 0.0F, .y = 0.5F},
            raster::Vec2f{.x = 0.5F, .y = 0.5F},
        };
        const auto derivatives = raster::quad_derivatives(minified_uv);
        const auto lod = raster::mip_lod(derivatives, texture.width(), texture.height());
        REQUIRE(lod == 5.0F);
        REQUIRE(texture.sample_nearest(0.1F, 0.1F) != 0xFF7F7F7FU);
        REQUIRE(texture.sample_nearest(0.1F, 0.1F, static_cast<std::size_t>(lod)) == 0xFF7F7F7FU);
        REQUIRE(texture.sample_trilinear(0.1F, 0.1F, lod) == 0xFF7F7F7FU);
        REQUIRE(texture.sample_trilinear(0.1F, 0.1F, 0.0F) != texture.sample_nearest(0.1F, 0.1F));
        REQUIRE(
            texture.sample_trilinear(0.1F, 0.1F, 100.0F)
            == texture.sample_nearest(0.1F, 0.1F, texture.level_count() - 1U)
        );
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
