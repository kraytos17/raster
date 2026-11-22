#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>

#include "raster/memory/texture.hpp"
#include "raster/pipeline/simd_quad.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    struct DeterministicRng {
        std::uint32_t state{0x12345678U};

        [[nodiscard]]
        std::uint32_t next() noexcept {
            state = state * 1'664'525U + 1'013'904'223U;
            return state;
        }

        [[nodiscard]]
        float unit() noexcept {
            return static_cast<float>(next() >> 8U) * (1.0F / 16777216.0F);
        }

        [[nodiscard]]
        float range(float low, float high) noexcept {
            return low + (high - low) * unit();
        }
    };

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("wide barycentric weights match scalar exactly", "[simd][weights]") {
        if (!raster::simd::cpu_has_avx2()) {
            SUCCEED("AVX2 unavailable; wide path untestable here");
            return;
        }

        DeterministicRng rng{};
        for (std::size_t trial = 0; trial < 256U; ++trial) {
            std::array<float, 8> edge1{};
            std::array<float, 8> edge2{};
            for (std::size_t lane = 0; lane < 8U; ++lane) {
                edge1[lane] = rng.range(-10000000.0F, 10000000.0F);
                edge2[lane] = rng.range(-10000000.0F, 10000000.0F);
            }
            const float inv_area = (static_cast<float>(rng.next() % 1000U) + 1.0F) * 1e-6F;
            std::array<float, 8> l0{};
            std::array<float, 8> l1{};
            std::array<float, 8> l2{};
            raster::simd::barycentric_weights8(
                edge1.data(),
                edge2.data(),
                inv_area,
                l0.data(),
                l1.data(),
                l2.data()
            );
            for (std::size_t lane = 0; lane < 8U; ++lane) {
                const auto expected_l0 = edge1[lane] * inv_area;
                const auto expected_l1 = edge2[lane] * inv_area;
                REQUIRE(l0[lane] == expected_l0);
                REQUIRE(l1[lane] == expected_l1);
                REQUIRE(l2[lane] == 1.0F - (expected_l0 + expected_l1));
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("wide color packing matches scalar quantization exactly", "[simd][pack]") {
        if (!raster::simd::cpu_has_avx2()) {
            SUCCEED("AVX2 unavailable; wide path untestable here");
            return;
        }

        DeterministicRng rng{};
        for (std::size_t trial = 0; trial < 256U; ++trial) {
            std::array<float, 8> red{};
            std::array<float, 8> green{};
            std::array<float, 8> blue{};
            for (std::size_t lane = 0; lane < 8U; ++lane) {
                // Include out-of-range values to pin clamping behavior.
                red[lane] = rng.range(-0.5F, 1.5F);
                green[lane] = rng.range(-0.5F, 1.5F);
                blue[lane] = rng.range(-0.5F, 1.5F);
            }
            std::array<std::uint32_t, 8> packed{};
            raster::simd::quantize_pack8(red.data(), green.data(), blue.data(), packed.data());
            for (std::size_t lane = 0; lane < 8U; ++lane) {
                const auto clamp = [](float channel) noexcept {
                    return std::clamp(channel, 0.0F, 1.0F);
                };
                const auto expected = 0xFF000000U
                    | (raster::quantize_channel(clamp(red[lane]) * 255.0F) << 16U)
                    | (raster::quantize_channel(clamp(green[lane]) * 255.0F) << 8U)
                    | raster::quantize_channel(clamp(blue[lane]) * 255.0F);
                REQUIRE(packed[lane] == expected);
            }
        }
    }

    // NOTE: quad derivatives and mip LOD stay scalar: LOD is computed once
    // per 2x2 quad, not per pixel, so there is no wide fragment work to
    // vectorize there. The wide path covers per-pixel lanes only.

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("wide 4-lane packing matches scalar to_argb exactly", "[simd][pack4]") {
        DeterministicRng rng{};
        for (std::size_t trial = 0; trial < 256U; ++trial) {
            std::array<float, 4> red{};
            std::array<float, 4> green{};
            std::array<float, 4> blue{};
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                // Include out-of-range values to pin clamping behavior.
                red[lane] = rng.range(-0.5F, 1.5F);
                green[lane] = rng.range(-0.5F, 1.5F);
                blue[lane] = rng.range(-0.5F, 1.5F);
            }
            std::array<std::uint32_t, 4> packed{};
            raster::simd::quantize_pack4(red.data(), green.data(), blue.data(), packed.data());
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                const auto clamp = [](float channel) noexcept {
                    return std::clamp(channel, 0.0F, 1.0F);
                };
                const auto expected = 0xFF000000U
                    | (raster::quantize_channel(clamp(red[lane]) * 255.0F) << 16U)
                    | (raster::quantize_channel(clamp(green[lane]) * 255.0F) << 8U)
                    | raster::quantize_channel(clamp(blue[lane]) * 255.0F);
                REQUIRE(packed[lane] == expected);
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("wide bilinear blend matches scalar exactly", "[simd][bilinear]") {
        DeterministicRng rng{};
        for (std::size_t trial = 0; trial < 256U; ++trial) {
            std::array<float, 4> v00{};
            std::array<float, 4> v10{};
            std::array<float, 4> v01{};
            std::array<float, 4> v11{};
            std::array<float, 4> fx{};
            std::array<float, 4> fy{};
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                v00[lane] = rng.range(0.0F, 255.0F);
                v10[lane] = rng.range(0.0F, 255.0F);
                v01[lane] = rng.range(0.0F, 255.0F);
                v11[lane] = rng.range(0.0F, 255.0F);
                // Pin edge weights 0 and 1 alongside interiors.
                fx[lane] = (trial + lane) % 7U == 0U ? 0.0F
                    : (trial + lane) % 7U == 1U      ? 1.0F
                                                     : rng.unit();
                fy[lane] = (trial + lane) % 5U == 0U ? 0.0F
                    : (trial + lane) % 5U == 1U      ? 1.0F
                                                     : rng.unit();
            }
            std::array<float, 4> blended{};
            raster::simd::bilinear_blend4(
                v00.data(),
                v10.data(),
                v01.data(),
                v11.data(),
                fx.data(),
                fy.data(),
                blended.data()
            );
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                const auto wx0 = 1.0F - fx[lane];
                const auto wy0 = 1.0F - fy[lane];
                const auto expected = (v00[lane] * wx0 + v10[lane] * fx[lane])
                    * wy0
                    + (v01[lane] * wx0 + v11[lane] * fx[lane])
                    * fy[lane];
                REQUIRE(blended[lane] == expected);
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("wide trilinear mix matches scalar exactly", "[simd][mix]") {
        DeterministicRng rng{};
        for (std::size_t trial = 0; trial < 256U; ++trial) {
            std::array<float, 4> low{};
            std::array<float, 4> high{};
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                low[lane] = rng.range(0.0F, 255.0F);
                high[lane] = rng.range(0.0F, 255.0F);
            }
            const float fraction = trial % 3U == 0U ? 0.0F : trial % 3U == 1U ? 1.0F : rng.unit();
            std::array<float, 4> mixed{};
            raster::simd::trilinear_mix4(low.data(), high.data(), fraction, mixed.data());
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                REQUIRE(mixed[lane] == low[lane] + (high[lane] - low[lane]) * fraction);
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("wide channel packing matches pack_channels exactly", "[simd][packch]") {
        DeterministicRng rng{};
        for (std::size_t trial = 0; trial < 512U; ++trial) {
            std::array<float, 4> c0{};
            std::array<float, 4> c1{};
            std::array<float, 4> c2{};
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                // Byte-range blends incl. exact .5 boundaries and near ties.
                const auto pick = [&rng, trial, lane] {
                    switch ((trial + lane) % 6U) {
                        case 0:
                            return 127.5F;
                        case 1:
                            return 0.0F;
                        case 2:
                            return 255.0F;
                        case 3:
                            return 127.49999F;
                        case 4:
                            return 0.49999F;
                        default:
                            return rng.range(0.0F, 255.0F);
                    }
                };
                c0[lane] = pick();
                c1[lane] = pick();
                c2[lane] = pick();
            }
            std::array<std::uint32_t, 4> packed{};
            raster::simd::pack_channels4(c0.data(), c1.data(), c2.data(), packed.data());
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                const std::array<float, 4> channels{c0[lane], c1[lane], c2[lane], 255.0F};
                REQUIRE(packed[lane] == raster::pack_channels(channels));
            }
        }
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("streaming fill matches scalar fill exactly", "[simd][streamfill]") {
        DeterministicRng rng{};
        const std::array<std::size_t, 8> sizes{0U, 1U, 3U, 4U, 5U, 17U, 1000U, 100'000U};
        for (const auto count : sizes) {
            const auto value = rng.next();
            // Offset by one element to force 4-byte-but-not-16-byte alignment.
            std::vector<std::uint32_t> wide(count + 1U, 0xDEADBEEFU);
            std::vector<std::uint32_t> reference(count + 1U, 0xDEADBEEFU);
            raster::simd::stream_fill_u32(wide.data() + 1U, count, value);
            std::ranges::fill(reference.begin() + 1, reference.end(), value);
            REQUIRE(wide == reference);
        }
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
