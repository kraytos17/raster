#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

#include "raster/math/vec.hpp"

namespace raster::pbr {
    inline constexpr float pi = std::numbers::pi_v<float>;
    inline constexpr float minimum_alpha = 0.045F;
    inline constexpr float minimum_cosine = 1.0e-6F;

    struct Material {
        Vec3f base_color{}; // linear-light RGB
        float metallic{};
        float perceptual_roughness{1.0F};
    };

    struct Parameters {
        Vec3f diffuse_color{}; // linear-light, metallic energy removed
        Vec3f f0{}; // linear-light normal-incidence reflectance
        float alpha{}; // remapped/clamped microfacet roughness
    };

    [[nodiscard]]
    inline float saturate(float value) noexcept {
        return std::clamp(value, 0.0F, 1.0F);
    }

    [[nodiscard]]
    inline float perceptual_roughness_to_alpha(float perceptual_roughness) noexcept {
        const auto roughness = saturate(perceptual_roughness);
        return std::max(roughness * roughness, minimum_alpha);
    }

    [[nodiscard]]
    inline Parameters make_metallic_roughness(
        Vec3f base_color, float metallic, float perceptual_roughness
    ) noexcept {
        const auto metal = saturate(metallic);
        const auto dielectric_weight = 1.0F - metal;
        return Parameters{
            .diffuse_color =
                Vec3f{
                    .x = base_color.x * dielectric_weight,
                    .y = base_color.y * dielectric_weight,
                    .z = base_color.z * dielectric_weight,
                },
            .f0 =
                Vec3f{
                    .x = 0.04F * dielectric_weight + base_color.x * metal,
                    .y = 0.04F * dielectric_weight + base_color.y * metal,
                    .z = 0.04F * dielectric_weight + base_color.z * metal,
                },
            .alpha = perceptual_roughness_to_alpha(perceptual_roughness),
        };
    }

    // Isotropic Trowbridge-Reitz / GGX normal distribution function.
    // Inputs are NoH and alpha; NoH is saturated defensively.
    [[nodiscard]]
    inline float ggx_distribution(float no_h, float alpha) noexcept {
        const auto nh = saturate(no_h);
        const auto a = std::max(alpha, minimum_alpha);
        const auto alpha2 = a * a;
        const auto denominator_base = nh * nh * (alpha2 - 1.0F) + 1.0F;
        const auto denominator = pi * denominator_base * denominator_base;
        return alpha2 / denominator;
    }

    // Height-correlated Smith-GGX visibility, i.e. G/(4 NoV NoL), matching
    // Filament's real-time direct-light formulation. Returns 0 at either
    // horizon, where the direct-light cosine makes the contribution vanish.
    [[nodiscard]]
    inline float smith_ggx_correlated_visibility(float no_v, float no_l, float alpha) noexcept {
        const auto nv = saturate(no_v);
        const auto nl = saturate(no_l);
        if (nv <= 0.0F || nl <= 0.0F) {
            return 0.0F;
        }

        const auto a = std::max(alpha, minimum_alpha);
        const auto alpha2 = a * a;
        const auto lambda_v = nl * std::sqrt(nv * nv * (1.0F - alpha2) + alpha2);
        const auto lambda_l = nv * std::sqrt(nl * nl * (1.0F - alpha2) + alpha2);
        const auto denominator = lambda_v + lambda_l;
        return denominator > 0.0F ? 0.5F / denominator : 0.0F;
    }

    [[nodiscard]]
    inline float fresnel_schlick(float f0, float vo_h) noexcept {
        const auto cosine = saturate(vo_h);
        const auto one_minus_cosine = 1.0F - cosine;
        const auto x2 = one_minus_cosine * one_minus_cosine;
        const auto x4 = x2 * x2;
        const auto x5 = x4 * one_minus_cosine;
        return f0 + (1.0F - f0) * x5;
    }

    [[nodiscard]]
    inline Vec3f fresnel_schlick(Vec3f f0, float vo_h) noexcept {
        return Vec3f{
            .x = fresnel_schlick(f0.x, vo_h),
            .y = fresnel_schlick(f0.y, vo_h),
            .z = fresnel_schlick(f0.z, vo_h),
        };
    }

    // Direct Cook-Torrance metallic-roughness BRDF. Returned BRDF is not
    // multiplied by sun radiance or NoL; evaluate_direct applies both.
    [[nodiscard]]
    inline Vec3f evaluate_brdf(
        const Material& material, const Vec3f& normal, const Vec3f& view, const Vec3f& light
    ) noexcept {
        const auto no_l = saturate(dot(normal, light));
        const auto no_v = saturate(dot(normal, view));
        if (no_l <= 0.0F || no_v <= 0.0F) {
            return Vec3f{};
        }

        const auto half_sum = view + light;
        const auto half_length_squared = length_squared(half_sum);
        if (!(half_length_squared > 0.0F) || !std::isfinite(half_length_squared)) {
            return Vec3f{};
        }

        const auto half_vector = normalize(half_sum);
        const auto no_h = saturate(dot(normal, half_vector));
        const auto vo_h = saturate(dot(view, half_vector));
        const auto parameters = make_metallic_roughness(
            material.base_color,
            material.metallic,
            material.perceptual_roughness
        );

        const auto distribution = ggx_distribution(no_h, parameters.alpha);
        const auto visibility = smith_ggx_correlated_visibility(no_v, no_l, parameters.alpha);

        const auto fresnel = fresnel_schlick(parameters.f0, vo_h);
        const auto specular_scale = distribution * visibility;
        const auto diffuse_scale = (1.0F - saturate(material.metallic)) / pi;
        return Vec3f{
            .x = parameters.diffuse_color.x * diffuse_scale + fresnel.x * specular_scale,
            .y = parameters.diffuse_color.y * diffuse_scale + fresnel.y * specular_scale,
            .z = parameters.diffuse_color.z * diffuse_scale + fresnel.z * specular_scale,
        };
    }

    // One directional-light contribution. Shadow visibility is a scalar
    // multiplier on incident radiance, not on the BRDF or material albedo.
    [[nodiscard]]
    inline Vec3f evaluate_direct(
        const Material& material,
        const Vec3f& normal,
        const Vec3f& view,
        const Vec3f& light,
        Vec3f sun_radiance = Vec3f{.x = 1.0F, .y = 1.0F, .z = 1.0F},
        float visibility = 1.0F
    ) noexcept {
        const auto no_l = saturate(dot(normal, light));
        if (no_l <= 0.0F || saturate(visibility) <= 0.0F) {
            return Vec3f{};
        }

        const auto brdf = evaluate_brdf(material, normal, view, light);
        const auto scale = no_l * saturate(visibility);
        return Vec3f{
            .x = brdf.x * sun_radiance.x * scale,
            .y = brdf.y * sun_radiance.y * scale,
            .z = brdf.z * sun_radiance.z * scale,
        };
    }
} // namespace raster::pbr
