#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>

#include "raster/memory/framebuffer.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions)
    TEST_CASE("framebuffer storage is cache-line aligned", "[framebuffer][memory]") {
        const raster::Framebuffer framebuffer{16, 8};

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): inspect pointer alignment.
        REQUIRE(reinterpret_cast<std::uintptr_t>(framebuffer.color_storage.data()) % 64U == 0U);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): inspect pointer alignment.
        REQUIRE(reinterpret_cast<std::uintptr_t>(framebuffer.depth_storage.data()) % 64U == 0U);
        STATIC_REQUIRE(alignof(raster::AlignedAllocator<std::uint32_t>) <= 64);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions)
    TEST_CASE("framebuffer views preserve constness", "[framebuffer][mdspan]") {
        raster::Framebuffer framebuffer{3, 2};
        auto mutable_color = framebuffer.color_view();
        auto mutable_depth = framebuffer.depth_view();
        const auto& const_framebuffer = framebuffer;
        auto const_color = const_framebuffer.color_view();
        auto const_depth = const_framebuffer.depth_view();

        STATIC_REQUIRE(
            std::is_same_v<typename decltype(mutable_color)::element_type, std::uint32_t>
        );
        STATIC_REQUIRE(std::is_same_v<typename decltype(mutable_depth)::element_type, float>);
        STATIC_REQUIRE(
            std::is_same_v<typename decltype(const_color)::element_type, const std::uint32_t>
        );
        STATIC_REQUIRE(std::is_same_v<typename decltype(const_depth)::element_type, const float>);
        REQUIRE(mutable_color.extent(0) == 2);
        REQUIRE(mutable_color.extent(1) == 3);
        REQUIRE(mutable_depth.extent(0) == 2);
        REQUIRE(mutable_depth.extent(1) == 3);

        mutable_color[1, 2] = 0xFF123456U;
        mutable_depth[1, 2] = 0.25F;
        REQUIRE(const_color[1, 2] == 0xFF123456U);
        REQUIRE(const_depth[1, 2] == 0.25F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions)
    TEST_CASE("framebuffer resize is safe and re-aligns backing storage", "[framebuffer][resize]") {
        raster::Framebuffer framebuffer{3, 2};
        framebuffer.color_view()[1, 2] = 0xFF123456U;
        framebuffer.resize(13, 7);

        REQUIRE(framebuffer.width == 13);
        REQUIRE(framebuffer.height == 7);
        REQUIRE(framebuffer.color_storage.size() == 91);
        REQUIRE(framebuffer.depth_storage.size() == 91);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): verify resize alignment.
        REQUIRE(reinterpret_cast<std::uintptr_t>(framebuffer.color_storage.data()) % 64U == 0U);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): verify resize alignment.
        REQUIRE(reinterpret_cast<std::uintptr_t>(framebuffer.depth_storage.data()) % 64U == 0U);
        REQUIRE(framebuffer.color_view()[6, 12] == 0xFF000000U);
        REQUIRE(framebuffer.depth_view()[6, 12] == 0.0F);

        framebuffer.resize(0, 0);
        REQUIRE(framebuffer.width == 13);
        REQUIRE(framebuffer.height == 7);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions)
    TEST_CASE("clear_tile clears only its framebuffer tile", "[framebuffer][clear]") {
        raster::Framebuffer framebuffer{18, 17};
        framebuffer.clear_tile(1, 1, 0xFFAABBCCU, 0.75F);
        auto color = framebuffer.color_view();
        auto depth = framebuffer.depth_view();

        REQUIRE(color[16, 16] == 0xFFAABBCCU);
        REQUIRE(depth[16, 16] == 0.75F);
        REQUIRE(color[15, 15] == 0xFF000000U);
        REQUIRE(depth[15, 15] == 0.0F);
        REQUIRE(color[0, 0] == 0xFF000000U);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions)
    TEST_CASE("clear_tile ignores tiles outside framebuffer extent", "[framebuffer][clear]") {
        raster::Framebuffer framebuffer{18, 17};
        framebuffer.color_view()[16, 16] = 0xFFAABBCCU;
        framebuffer.depth_view()[16, 16] = 0.5F;

        framebuffer.clear_tile(2, 1, 0xFF010203U, 0.25F);
        REQUIRE(framebuffer.color_view()[16, 16] == 0xFFAABBCCU);
        REQUIRE(framebuffer.depth_view()[16, 16] == 0.5F);
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
