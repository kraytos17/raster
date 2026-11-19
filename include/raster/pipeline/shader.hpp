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
    template <typename Shader>
    concept ShaderProgram = requires(const Shader& shader, const Shader::Varyings& varying) {
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
        template <typename VaryingsT>
        [[nodiscard]]
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
                    std::remove_reference_t<std::tuple_element_t<Index, decltype(attributes_a)>>...
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
    };
} // namespace raster
