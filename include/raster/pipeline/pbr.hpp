#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <expected>
#include <numbers>
#include <string_view>

#include "raster/math/vec.hpp"

namespace raster::pbr {
    inline constexpr float pi = std::numbers::pi_v<float>;
    inline constexpr float minimum_alpha = 0.045F;
    inline constexpr float minimum_cosine = 1.0e-6F;

    struct Material {
        Vec3f base_color{.x = 1.0F, .y = 1.0F, .z = 1.0F}; // linear-light RGB
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

    // Direct Cook-Torrance metallic-roughness BRDF. `light` points from the
    // surface toward the source. Returned BRDF is not multiplied by sun
    // radiance or NoL; evaluate_direct applies both.
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
        const auto diffuse_scale = 1.0F / pi;
        return Vec3f{
            .x = parameters.diffuse_color.x
                * (1.0F - fresnel.x)
                * diffuse_scale
                + fresnel.x
                * specular_scale,
            .y = parameters.diffuse_color.y
                * (1.0F - fresnel.y)
                * diffuse_scale
                + fresnel.y
                * specular_scale,
            .z = parameters.diffuse_color.z
                * (1.0F - fresnel.z)
                * diffuse_scale
                + fresnel.z
                * specular_scale,
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

    // Combine per-draw factors with texture samples. base_linear is the
    // sRGB-decoded base-color texel; mr_linear is the linear mr texel with
    // G = roughness and B = metallic (R/A ignored, glTF convention).
    // Without an mr texture the factors are used directly (absent channels
    // default to 1, so factor * 1 == factor).
    [[nodiscard]]
    inline Material resolve_textured_material(
        const Material& factors, Vec3f base_linear, Vec3f mr_linear, bool has_mr_texture
    ) noexcept {
        const auto metallic = saturate(factors.metallic * (has_mr_texture ? mr_linear.z : 1.0f));
        const auto roughness =
            saturate(factors.perceptual_roughness * (has_mr_texture ? mr_linear.y : 1.0f));
        return Material{
            .base_color =
                Vec3f{
                    .x = saturate(factors.base_color.x * base_linear.x),
                    .y = saturate(factors.base_color.y * base_linear.y),
                    .z = saturate(factors.base_color.z * base_linear.z),
                },
            .metallic = metallic,
            .perceptual_roughness = roughness,
        };
    }

    enum class MaterialParseError {
        Empty,
        Malformed,
    };

    namespace detail {
        [[nodiscard]]
        inline std::string_view trim_material_text(std::string_view text) noexcept {
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
        inline std::expected<float, MaterialParseError> parse_material_float(
            std::string_view token
        ) noexcept {
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
                return std::unexpected{MaterialParseError::Malformed};
            }
            return value;
        }
    } // namespace detail

    // Minimal project-owned material descriptor (not MTL/glTF):
    //   base_color r g b
    //   metallic m
    //   roughness r
    // Blank lines and `#` comments ignored. Missing keys keep Material
    // defaults (white base, metallic 0, roughness 1); values saturate to
    // [0,1]. Per-face/per-group materials are out of scope: one descriptor
    // applies to the whole draw, matching the renderer's per-draw uniforms.
    [[nodiscard]]
    inline std::expected<Material, MaterialParseError> parse_material_buffer(
        std::string_view source
    ) noexcept {
        Material out{};
        bool saw_any = false;
        std::size_t offset = 0;
        while (offset < source.size()) {
            const auto newline = source.find('\n', offset);
            const auto end = newline == std::string_view::npos ? source.size() : newline;
            auto line = detail::trim_material_text(source.substr(offset, end - offset));
            offset = newline == std::string_view::npos ? source.size() : newline + 1;
            if (line.empty() || line.front() == '#') {
                continue;
            }
            if (const auto comment = line.find('#'); comment != std::string_view::npos) {
                line = detail::trim_material_text(line.substr(0, comment));
                if (line.empty()) {
                    continue;
                }
            }

            const auto separator = line.find_first_of(" \t");
            const auto keyword =
                separator == std::string_view::npos ? line : line.substr(0, separator);
            auto payload = separator == std::string_view::npos
                ? std::string_view{}
                : detail::trim_material_text(line.substr(separator + 1));
            if (keyword == "base_color") {
                std::array<float, 3> channels{};
                for (auto& channel : channels) {
                    if (payload.empty()) {
                        return std::unexpected{MaterialParseError::Malformed};
                    }

                    const auto token_end = payload.find_first_of(" \t");
                    const auto token = token_end == std::string_view::npos
                        ? payload
                        : payload.substr(0, token_end);

                    const auto parsed = detail::parse_material_float(token);
                    if (!parsed) {
                        return std::unexpected{parsed.error()};
                    }

                    channel = *parsed;
                    payload = token_end == std::string_view::npos
                        ? std::string_view{}
                        : detail::trim_material_text(payload.substr(token_end + 1));
                }
                if (!payload.empty()) {
                    return std::unexpected{MaterialParseError::Malformed};
                }

                out.base_color = Vec3f{
                    .x = saturate(channels[0]),
                    .y = saturate(channels[1]),
                    .z = saturate(channels[2]),
                };
                saw_any = true;
            } else if (keyword == "metallic" || keyword == "roughness") {
                if (payload.empty() || payload.find_first_of(" \t") != std::string_view::npos) {
                    return std::unexpected{MaterialParseError::Malformed};
                }

                const auto parsed = detail::parse_material_float(payload);
                if (!parsed) {
                    return std::unexpected{parsed.error()};
                }
                if (keyword == "metallic") {
                    out.metallic = saturate(*parsed);
                } else {
                    out.perceptual_roughness = saturate(*parsed);
                }
                saw_any = true;
            } else {
                return std::unexpected{MaterialParseError::Malformed};
            }
        }
        if (!saw_any) {
            return std::unexpected{MaterialParseError::Empty};
        }
        return out;
    }
} // namespace raster::pbr
