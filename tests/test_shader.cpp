#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <tuple>

#include "raster/pipeline/shader.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {

    // Minimal two-attribute shader used to verify the ShaderBase mixin and the
    // ShaderProgram concept.
    struct TestShader final : raster::ShaderBase {
        struct Varyings {
            raster::Vec4f clip_pos{};
            raster::Vec2f uv{};
            raster::Vec3f world_position{};

            [[nodiscard]]
            constexpr auto attributes(this const Varyings& self) noexcept {
                return std::tie(self.uv, self.world_position);
            }
        };

        [[nodiscard]]
        constexpr raster::Vec4f vertex(raster::Vec3f position) const noexcept {
            return raster::Vec4f{
                .x = position.x,
                .y = position.y,
                .z = position.z,
                .w = 1.0F,
            };
        }

        [[nodiscard]]
        constexpr raster::Vec4f fragment(const Varyings& varying) const noexcept {
            const auto cell = static_cast<int>(std::floor(varying.uv.x))
                + static_cast<int>(std::floor(varying.uv.y));
            const auto parity = ((cell % 2) + 2) % 2;
            const auto shade = parity == 0 ? 0.85F : 0.35F;
            return raster::Vec4f{.x = shade, .y = shade, .z = shade, .w = 1.0F};
        }
    };

    struct ShaderWithoutFragment {
        struct Varyings {
            raster::Vec4f clip_pos{};

            [[nodiscard]]
            constexpr auto attributes(this const Varyings& self) noexcept {
                return std::tie(self.clip_pos);
            }
        };
    };

    struct ShaderWithWrongClipPosition {
        struct Varyings {
            raster::Vec3f clip_pos{};

            [[nodiscard]]
            constexpr auto attributes(this const Varyings& self) noexcept {
                return std::tie(self.clip_pos);
            }
        };

        [[nodiscard]]
        constexpr raster::Vec4f fragment(const Varyings&) const noexcept {
            return raster::Vec4f{};
        }
    };

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("ShaderProgram concept accepts well-formed shaders", "[shader][concept]") {
        STATIC_REQUIRE(raster::ShaderProgram<TestShader>);
        STATIC_REQUIRE_FALSE(raster::ShaderProgram<ShaderWithoutFragment>);
        STATIC_REQUIRE_FALSE(raster::ShaderProgram<ShaderWithWrongClipPosition>);
        STATIC_REQUIRE_FALSE(raster::ShaderProgram<int>);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("ShaderBase interpolates every tied attribute", "[shader][interpolate]") {
        const TestShader shader{};

        constexpr TestShader::Varyings varying_a{
            .clip_pos = raster::Vec4f{.x = 1.0F, .y = 0.0F, .z = 0.0F, .w = 1.0F},
            .uv = raster::Vec2f{.x = 0.0F, .y = 0.0F},
            .world_position = raster::Vec3f{.x = 1.0F, .y = 1.0F, .z = 1.0F},
        };
        constexpr TestShader::Varyings varying_b{
            .clip_pos = raster::Vec4f{.x = 0.0F, .y = 1.0F, .z = 0.0F, .w = 1.0F},
            .uv = raster::Vec2f{.x = 2.0F, .y = 4.0F},
            .world_position = raster::Vec3f{.x = 2.0F, .y = 2.0F, .z = 2.0F},
        };
        constexpr TestShader::Varyings varying_c{
            .clip_pos = raster::Vec4f{.x = 0.0F, .y = 0.0F, .z = 1.0F, .w = 1.0F},
            .uv = raster::Vec2f{.x = 0.0F, .y = 6.0F},
            .world_position = raster::Vec3f{.x = 3.0F, .y = 3.0F, .z = 3.0F},
        };
        constexpr raster::PixelBarycentrics barycentrics{
            .l0 = 0.25F,
            .l1 = 0.25F,
            .l2 = 0.5F,
            .a0 = 0.25F,
            .a1 = 0.5F,
            .a2 = 0.25F,
        };

        const auto interpolated = shader.interpolate(varying_a, varying_b, varying_c, barycentrics);
        REQUIRE(std::get<0>(interpolated).x == 1.0F); // 2 * 0.5
        REQUIRE(std::get<0>(interpolated).y == 3.5F); // 4 * 0.5 + 6 * 0.25
        REQUIRE(std::get<1>(interpolated).x == 2.0F); // 1*0.25 + 2*0.5 + 3*0.25
        REQUIRE(std::get<1>(interpolated).y == 2.0F);
        REQUIRE(std::get<1>(interpolated).z == 2.0F);

        // The checker fragment stage reads through the interpolated varyings.
        const TestShader::Varyings interpolated_varyings{
            .clip_pos = varying_a.clip_pos,
            .uv = std::get<0>(interpolated),
            .world_position = std::get<1>(interpolated),
        };
        const auto color = shader.fragment(interpolated_varyings);
        REQUIRE(color.w == 1.0F);
        REQUIRE(color.x == 0.85F); // floor(1.0) + floor(3.5) -> even cell
    }

} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
