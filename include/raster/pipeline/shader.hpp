#pragma once

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "raster/math/vec.hpp"
#include "raster/pipeline/rasterizer.hpp"

namespace raster {
    // Contract for a programmable shader stage. A shader provides a nested
    // Varyings struct holding a Vec4f clip_pos plus an attributes(this auto&&
    // self) method returning a std::tie tuple of every custom interpolated
    // field, and a fragment(const Varyings&) stage returning linear RGBA. The
    // vertex(...) signature is shader-specific (it consumes whatever SoA
    // vertex attributes the scene passes), so it is not constrained here; the
    // scene wires it to its own mesh layout.
    template <typename Shader> concept ShaderProgram =
        requires(const Shader& shader, const Shader::Varyings& varying) {
            typename Shader::Varyings;
            { auto(varying.clip_pos) } -> std::same_as<Vec4f>;
            { varying.attributes() };
            { shader.fragment(varying) } -> std::convertible_to<Vec4f>;
        };

    // Mixin providing automatic barycentric interpolation of every attribute
    // exposed by Varyings::attributes(). Weights used are the
    // perspective-correct alpha values; with all inv_w equal to 1 they are
    // bitwise identical to the screen-space weights.
    struct ShaderBase {
        template <typename VaryingsT> [[nodiscard]]
        constexpr auto interpolate(
            this const auto&,
            const VaryingsT& varying_a,
            const VaryingsT& varying_b,
            const VaryingsT& varying_c,
            const PixelBarycentrics& barycentrics
        ) noexcept {
            const auto attributes_a = varying_a.attributes();
            constexpr auto attribute_count = std::tuple_size_v<decltype(attributes_a)>;
            return [&]<std::size_t... Index>(std::index_sequence<Index...>) {
                using ResultTuple = std::tuple<
                    std::remove_cvref_t<std::tuple_element_t<Index, decltype(attributes_a)>>...
                >;
                return ResultTuple{lerp3(
                    std::get<Index>(attributes_a),
                    std::get<Index>(varying_b.attributes()),
                    std::get<Index>(varying_c.attributes()),
                    barycentrics.a0,
                    barycentrics.a1,
                    barycentrics.a2
                )...};
            }(std::make_index_sequence<attribute_count>{});
        }

        // Homogeneous clipping intersects an edge in clip space before the
        // perspective divide, so every varying attribute is interpolated
        // linearly with the same edge parameter t (not with corrected pixel
        // barycentrics). The shader's attributes() tuple is the single list of
        // custom fields that participates in clipping and later rasterization.
        template <typename VaryingsT> [[nodiscard]]
        constexpr auto interpolate_edge(
            this const auto&, const VaryingsT& varying_a, const VaryingsT& varying_b, float t
        ) noexcept {
            VaryingsT result = varying_a;
            result.clip_pos = raster::lerp(varying_a.clip_pos, varying_b.clip_pos, t);

            const auto attributes_a = varying_a.attributes();
            const auto attributes_b = varying_b.attributes();
            auto result_attributes = result.attributes();
            constexpr auto attribute_count = std::tuple_size_v<decltype(attributes_a)>;
            [&]<std::size_t... Index>(std::index_sequence<Index...>) {
                ((std::get<Index>(result_attributes) = raster::lerp(
                      std::get<Index>(attributes_a),
                      std::get<Index>(attributes_b),
                      t
                  )),
                 ...);
            }(std::make_index_sequence<attribute_count>{});
            return result;
        }
    };
} // namespace raster
