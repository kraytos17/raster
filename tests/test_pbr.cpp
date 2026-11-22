#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "raster/pipeline/pbr.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc2y-extensions"
#endif

namespace {
    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("GGX NDF is finite, positive, and peaks at the normal", "[pbr][ggx]") {
        constexpr auto alpha = 0.5F;
        const auto at_normal = raster::pbr::ggx_distribution(1.0F, alpha);
        REQUIRE(std::isfinite(at_normal));
        REQUIRE(at_normal == 1.0F / (std::numbers::pi_v<float> * alpha * alpha));
        REQUIRE(raster::pbr::ggx_distribution(0.5F, alpha) < at_normal);
        REQUIRE(raster::pbr::ggx_distribution(0.0F, alpha) > 0.0F);

        constexpr auto alpha_min = raster::pbr::minimum_alpha;
        REQUIRE(raster::pbr::perceptual_roughness_to_alpha(0.0F) == alpha_min);
        REQUIRE(raster::pbr::perceptual_roughness_to_alpha(1.0F) == 1.0F);
        REQUIRE(raster::pbr::perceptual_roughness_to_alpha(2.0F) == 1.0F);
        REQUIRE(raster::pbr::perceptual_roughness_to_alpha(-1.0F) == alpha_min);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("correlated Smith GGX visibility has analytic normal limit", "[pbr][ggx]") {
        constexpr auto alpha = 0.5F;
        const auto expected = 0.5F / (2.0F * std::sqrt(1.0F));
        REQUIRE(raster::pbr::smith_ggx_correlated_visibility(1.0F, 1.0F, alpha) == expected);
        REQUIRE(raster::pbr::smith_ggx_correlated_visibility(0.0F, 1.0F, alpha) == 0.0F);
        REQUIRE(raster::pbr::smith_ggx_correlated_visibility(1.0F, 0.0F, alpha) == 0.0F);

        const auto grazing = raster::pbr::smith_ggx_correlated_visibility(1.0e-5F, 0.8F, alpha);
        REQUIRE(std::isfinite(grazing));
        REQUIRE(grazing >= 0.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("Schlick Fresnel has exact normal and grazing limits", "[pbr][fresnel]") {
        constexpr auto f0 = 0.04F;
        REQUIRE(raster::pbr::fresnel_schlick(f0, 1.0F) == f0);
        REQUIRE(raster::pbr::fresnel_schlick(f0, 0.0F) == 1.0F);
        REQUIRE(raster::pbr::fresnel_schlick(f0, 0.5F) > f0);
        REQUIRE(raster::pbr::fresnel_schlick(f0, -2.0F) == 1.0F);
        REQUIRE(raster::pbr::fresnel_schlick(f0, 2.0F) == f0);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("metallic workflow partitions diffuse and specular energy", "[pbr][material]") {
        constexpr raster::Vec3f base{.x = 0.8F, .y = 0.4F, .z = 0.2F};
        const auto dielectric = raster::pbr::make_metallic_roughness(base, 0.0F, 0.5F);
        REQUIRE(dielectric.f0.x == 0.04F);
        REQUIRE(dielectric.f0.y == 0.04F);
        REQUIRE(dielectric.f0.z == 0.04F);
        REQUIRE(dielectric.diffuse_color.x == base.x);
        REQUIRE(dielectric.diffuse_color.y == base.y);
        REQUIRE(dielectric.diffuse_color.z == base.z);
        REQUIRE(dielectric.alpha == 0.25F);

        const auto metal = raster::pbr::make_metallic_roughness(base, 1.0F, 0.5F);
        REQUIRE(metal.f0.x == base.x);
        REQUIRE(metal.f0.y == base.y);
        REQUIRE(metal.f0.z == base.z);
        REQUIRE(metal.diffuse_color.x == 0.0F);
        REQUIRE(metal.diffuse_color.y == 0.0F);
        REQUIRE(metal.diffuse_color.z == 0.0F);

        const auto half_metal = raster::pbr::make_metallic_roughness(base, 0.5F, 0.5F);
        REQUIRE(half_metal.f0.x == 0.04F * 0.5F + base.x * 0.5F);
        REQUIRE(half_metal.f0.y == 0.04F * 0.5F + base.y * 0.5F);
        REQUIRE(half_metal.f0.z == 0.04F * 0.5F + base.z * 0.5F);
        REQUIRE(half_metal.diffuse_color.x == base.x * 0.5F);
        REQUIRE(half_metal.diffuse_color.y == base.y * 0.5F);
        REQUIRE(half_metal.diffuse_color.z == base.z * 0.5F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("GGX direct BRDF is finite and reciprocal", "[pbr][brdf]") {
        constexpr raster::pbr::Material material{
            .base_color = raster::Vec3f{.x = 0.65F, .y = 0.32F, .z = 0.12F},
            .metallic = 0.35F,
            .perceptual_roughness = 0.42F,
        };
        const raster::Vec3f normal{.x = 0.0F, .y = 0.0F, .z = 1.0F};
        const raster::Vec3f view =
            raster::normalize(raster::Vec3f{.x = 0.2F, .y = 0.1F, .z = 1.0F});
        const raster::Vec3f light =
            raster::normalize(raster::Vec3f{.x = -0.3F, .y = 0.4F, .z = 1.0F});

        const auto forward = raster::pbr::evaluate_brdf(material, normal, view, light);
        const auto reversed = raster::pbr::evaluate_brdf(material, normal, light, view);
        REQUIRE(std::isfinite(forward.x));
        REQUIRE(std::isfinite(forward.y));
        REQUIRE(std::isfinite(forward.z));
        REQUIRE(forward.x == reversed.x);
        REQUIRE(forward.y == reversed.y);
        REQUIRE(forward.z == reversed.z);

        const auto backface = raster::pbr::evaluate_direct(
            material,
            normal,
            view,
            raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = -1.0F}
        );
        REQUIRE(backface.x == 0.0F);
        REQUIRE(backface.y == 0.0F);
        REQUIRE(backface.z == 0.0F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("white dielectric normal-incidence BRDF matches the hand case", "[pbr][brdf]") {
        constexpr raster::pbr::Material material{
            .base_color = raster::Vec3f{.x = 1.0F, .y = 1.0F, .z = 1.0F},
            .metallic = 0.0F,
            .perceptual_roughness = 1.0F,
        };
        constexpr raster::Vec3f normal{.x = 0.0F, .y = 0.0F, .z = 1.0F};
        const auto brdf = raster::pbr::evaluate_brdf(material, normal, normal, normal);
        // Lambert diffuse is (1-F)/pi = 0.96/pi. At alpha=1,
        // D=1/pi and correlated V=1/4, so specular is 0.01/pi.
        const auto expected = 0.97F / raster::pbr::pi;
        REQUIRE(std::fabs(brdf.x - expected) < 1.0e-7F);
        REQUIRE(std::fabs(brdf.y - expected) < 1.0e-7F);
        REQUIRE(std::fabs(brdf.z - expected) < 1.0e-7F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("textured factors combine base color with G/B data channels", "[pbr][material]") {
        constexpr raster::pbr::Material factors{
            .base_color = raster::Vec3f{.x = 1.0F, .y = 0.5F, .z = 0.25F},
            .metallic = 0.5F,
            .perceptual_roughness = 0.8F,
        };
        constexpr raster::Vec3f base_linear{.x = 0.5F, .y = 0.25F, .z = 1.0F};
        // R ignored, G = roughness texel, B = metallic texel.
        constexpr raster::Vec3f mr_linear{.x = 0.0F, .y = 0.5F, .z = 1.0F};
        const auto resolved =
            raster::pbr::resolve_textured_material(factors, base_linear, mr_linear, true);
        REQUIRE(resolved.base_color.x == 0.5F);
        REQUIRE(resolved.base_color.y == 0.125F);
        REQUIRE(resolved.base_color.z == 0.25F);
        REQUIRE(resolved.metallic == 0.5F);
        REQUIRE(resolved.perceptual_roughness == 0.4F);

        const auto no_map =
            raster::pbr::resolve_textured_material(factors, base_linear, mr_linear, false);
        REQUIRE(no_map.metallic == 0.5F);
        REQUIRE(no_map.perceptual_roughness == 0.8F);
        REQUIRE(no_map.base_color.x == 0.5F);
    }

    // NOLINTNEXTLINE(clang-diagnostic-c2y-extensions): Catch2 uses __COUNTER__ for registration.
    TEST_CASE("project material descriptor parses factors and rejects junk", "[pbr][material]") {
        constexpr std::string_view good =
            "# dielectric\nbase_color 1.0 0.5 0.25\nmetallic 0.2\nroughness 0.6\n";
        const auto material = raster::pbr::parse_material_buffer(good);
        REQUIRE(material.has_value());
        REQUIRE(material->base_color.x == 1.0F);
        REQUIRE(material->base_color.y == 0.5F);
        REQUIRE(material->base_color.z == 0.25F);
        REQUIRE(material->metallic == 0.2F);
        REQUIRE(material->perceptual_roughness == 0.6F);
        REQUIRE_FALSE(raster::pbr::parse_material_buffer("metallic nope\n").has_value());
        REQUIRE_FALSE(raster::pbr::parse_material_buffer("").has_value());
        REQUIRE_FALSE(raster::pbr::parse_material_buffer("metallic nan\n").has_value());
        REQUIRE_FALSE(raster::pbr::parse_material_buffer("roughness inf\n").has_value());
        REQUIRE_FALSE(raster::pbr::parse_material_buffer("base_color 0 1 nan\n").has_value());
        REQUIRE_FALSE(raster::pbr::parse_material_buffer("base_color 0 1\n").has_value());
        REQUIRE_FALSE(raster::pbr::parse_material_buffer("unknown 1\n").has_value());
    }
} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
