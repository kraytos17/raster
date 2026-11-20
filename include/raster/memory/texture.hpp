#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <vector>

#include "raster/io/tga_loader.hpp"
#include "raster/math/vec.hpp"
#include "raster/pipeline/simd_quad.hpp"

namespace raster {
    struct TextureLevel {
        std::size_t width{};
        std::size_t height{};
        std::vector<std::uint32_t> texels{};
    };

    // Round-half-up quantization for non-negative bounded channel values in
    // [0, 255]. Truncating (value + 0.5F) is exact for every float32 input in
    // that domain, matching std::lround without the out-of-line libm call.
    // Single home for the rounding idiom used by texture filtering and
    // color output.
    [[nodiscard]]
    inline std::uint32_t quantize_channel(float value) noexcept {
        // NOLINTNEXTLINE(bugprone-incorrect-roundings): non-negative bounded domain is exact.
        return static_cast<std::uint32_t>(value + 0.5F);
    }

    // Packs four float channels into 0xAARRGGBB with one rounding each.
    [[nodiscard]]
    inline std::uint32_t pack_channels(const std::array<float, 4>& channels) noexcept {
        std::uint32_t result = 0;
        for (std::size_t channel = 0; channel < 4U; ++channel) {
            const auto shift = static_cast<std::uint32_t>(channel * 8U);
            result |= (quantize_channel(channels[channel]) & 0xFFU) << shift;
        }
        return result;
    }

    template <typename Layout = std::layout_right> class Texture2D {
    public:

        using Extents = std::dextents<std::size_t, 2>;
        using View = std::mdspan<const std::uint32_t, Extents, Layout>;

        Texture2D() = default;

        explicit Texture2D(const TgaImage& image) :
            width_{image.width},
            height_{image.height} {
            const typename View::mapping_type mapping{Extents{height_, width_}};
            levels_.push_back(
                TextureLevel{
                    .width = width_,
                    .height = height_,
                    .texels = std::vector<std::uint32_t>(mapping.required_span_size()),
                }
            );
            for (std::size_t y = 0; y < height_; ++y) {
                for (std::size_t x = 0; x < width_; ++x) {
                    levels_.front().texels[mapping(y, x)] = image.pixels[y * width_ + x];
                }
            }
            build_mips();
        }

        [[nodiscard]]
        std::size_t level_count() const noexcept {
            return levels_.size();
        }

        [[nodiscard]]
        std::size_t width(std::size_t level = 0) const noexcept {
            return levels_[std::min(level, levels_.size() - 1U)].width;
        }

        [[nodiscard]]
        std::size_t height(std::size_t level = 0) const noexcept {
            return levels_[std::min(level, levels_.size() - 1U)].height;
        }

        [[nodiscard]]
        const TextureLevel& level(std::size_t index) const noexcept {
            return levels_[index];
        }

        [[nodiscard]]
        std::uint32_t sample_nearest(float u, float v, std::size_t mip_level = 0) const noexcept {
            const auto& mip = levels_[std::min(mip_level, levels_.size() - 1U)];
            const auto wrap = [](float coordinate) noexcept {
                const auto fractional = coordinate - std::floor(coordinate);
                return std::clamp(fractional, 0.0F, 0.99999994F);
            };

            const auto x = static_cast<std::size_t>(wrap(u) * static_cast<float>(mip.width));
            const auto y = static_cast<std::size_t>(wrap(v) * static_cast<float>(mip.height));
            const typename View::mapping_type mapping{Extents{mip.height, mip.width}};
            return mip.texels[mapping(y, x)];
        }

        [[nodiscard]]
        std::uint32_t sample_trilinear(float u, float v, float lod) const noexcept {
            const auto clamped_lod = std::clamp(lod, 0.0F, static_cast<float>(levels_.size() - 1U));
            const auto low_level = static_cast<std::size_t>(std::floor(clamped_lod));
            const auto high_level = std::min(low_level + 1U, levels_.size() - 1U);
            const auto fraction = clamped_lod - static_cast<float>(low_level);
            const auto low = sample_bilinear_channels(u, v, low_level);
            const auto high = sample_bilinear_channels(u, v, high_level);
            std::array<float, 4> mixed{};
            for (std::size_t channel = 0; channel < 4U; ++channel) {
                mixed[channel] = low[channel] + (high[channel] - low[channel]) * fraction;
            }
            return pack_channels(mixed);
        }

        // Four-lane trilinear for one 2x2 quad: scalar gathers per lane (same
        // wrap, floor, and wrap-index operations as the single-pixel path, so
        // the same texels are fetched), then wide bilinear blend, wide level
        // mix, and one wide pack.
        void sample_trilinear_quad4(
            const float* u, const float* v, float lod, std::uint32_t* out
        ) const noexcept {
            const auto clamped_lod = std::clamp(lod, 0.0F, static_cast<float>(levels_.size() - 1U));

            const auto low_level = static_cast<std::size_t>(std::floor(clamped_lod));
            const auto high_level = std::min(low_level + 1U, levels_.size() - 1U);
            const auto fraction = clamped_lod - static_cast<float>(low_level);
            const auto& low_mip = levels_[low_level];
            const auto& high_mip = levels_[high_level];

            float low_v00[3][4]{};
            float low_v10[3][4]{};
            float low_v01[3][4]{};
            float low_v11[3][4]{};
            float low_fx[4]{};
            float low_fy[4]{};
            float high_v00[3][4]{};
            float high_v10[3][4]{};
            float high_v01[3][4]{};
            float high_v11[3][4]{};
            float high_fx[4]{};
            float high_fy[4]{};
            for (std::size_t lane = 0; lane < 4U; ++lane) {
                std::uint32_t low_corners[4]{};
                std::uint32_t high_corners[4]{};
                gather_bilinear_corners(
                    low_mip,
                    u[lane],
                    v[lane],
                    low_corners,
                    low_fx[lane],
                    low_fy[lane]
                );
                gather_bilinear_corners(
                    high_mip,
                    u[lane],
                    v[lane],
                    high_corners,
                    high_fx[lane],
                    high_fy[lane]
                );

                for (std::size_t channel = 0; channel < 3U; ++channel) {
                    const auto shift = static_cast<std::uint32_t>(channel * 8U);
                    low_v00[channel][lane] = static_cast<float>((low_corners[0] >> shift) & 0xFFU);
                    low_v10[channel][lane] = static_cast<float>((low_corners[1] >> shift) & 0xFFU);
                    low_v01[channel][lane] = static_cast<float>((low_corners[2] >> shift) & 0xFFU);
                    low_v11[channel][lane] = static_cast<float>((low_corners[3] >> shift) & 0xFFU);
                    high_v00[channel][lane] =
                        static_cast<float>((high_corners[0] >> shift) & 0xFFU);
                    high_v10[channel][lane] =
                        static_cast<float>((high_corners[1] >> shift) & 0xFFU);
                    high_v01[channel][lane] =
                        static_cast<float>((high_corners[2] >> shift) & 0xFFU);
                    high_v11[channel][lane] =
                        static_cast<float>((high_corners[3] >> shift) & 0xFFU);
                }
            }

            float low_blended[3][4]{};
            float high_blended[3][4]{};
            float mixed[3][4]{};
            for (std::size_t channel = 0; channel < 3U; ++channel) {
                simd::bilinear_blend4(
                    low_v00[channel],
                    low_v10[channel],
                    low_v01[channel],
                    low_v11[channel],
                    low_fx,
                    low_fy,
                    low_blended[channel]
                );
                simd::bilinear_blend4(
                    high_v00[channel],
                    high_v10[channel],
                    high_v01[channel],
                    high_v11[channel],
                    high_fx,
                    high_fy,
                    high_blended[channel]
                );
                simd::trilinear_mix4(
                    low_blended[channel],
                    high_blended[channel],
                    fraction,
                    mixed[channel]
                );
            }
            // mixed[0/1/2] are B/G/R (channel c packs at bits c*8).
            simd::pack_channels4(mixed[0], mixed[1], mixed[2], out);
        }

    private:

        // Scalar corner gather shared by the single-pixel and quad paths:
        // wrap, floor, wrap-index, and texel loads in the same order, so both
        // paths fetch the same four corners and fractions.
        void gather_bilinear_corners(
            const TextureLevel& mip,
            float u,
            float v,
            std::uint32_t* corners,
            float& fraction_x,
            float& fraction_y
        ) const noexcept {
            const auto wrap = [](float coordinate) noexcept {
                return coordinate - std::floor(coordinate);
            };

            const auto texel_x = wrap(u) * static_cast<float>(mip.width) - 0.5F;
            const auto texel_y = wrap(v) * static_cast<float>(mip.height) - 0.5F;
            const auto base_x = static_cast<std::int64_t>(std::floor(texel_x));
            const auto base_y = static_cast<std::int64_t>(std::floor(texel_y));

            fraction_x = texel_x - static_cast<float>(base_x);
            fraction_y = texel_y - static_cast<float>(base_y);
            const typename View::mapping_type mapping{Extents{mip.height, mip.width}};
            const auto wrap_index = [](std::int64_t value, std::size_t extent) noexcept {
                const auto signed_extent = static_cast<std::int64_t>(extent);
                if (value < 0) {
                    return static_cast<std::size_t>(value + signed_extent);
                }
                if (value >= signed_extent) {
                    return static_cast<std::size_t>(value - signed_extent);
                }
                return static_cast<std::size_t>(value);
            };

            corners[0] =
                mip.texels[mapping(wrap_index(base_y, mip.height), wrap_index(base_x, mip.width))];
            corners[1] =
                mip.texels
                    [mapping(wrap_index(base_y, mip.height), wrap_index(base_x + 1, mip.width))];
            corners[2] =
                mip.texels
                    [mapping(wrap_index(base_y + 1, mip.height), wrap_index(base_x, mip.width))];
            corners[3] = mip.texels[mapping(
                wrap_index(base_y + 1, mip.height),
                wrap_index(base_x + 1, mip.width)
            )];
        }

        // Bilinear interpolation in float: the four corner weights are
        // non-negative and sum to 1, so each channel stays in [0, 255].
        [[nodiscard]]
        std::array<float, 4> sample_bilinear_channels(
            float u, float v, std::size_t mip_level
        ) const noexcept {
            const auto& mip = levels_[std::min(mip_level, levels_.size() - 1U)];
            std::uint32_t corners[4]{};
            float fraction_x{};
            float fraction_y{};

            gather_bilinear_corners(mip, u, v, corners, fraction_x, fraction_y);
            const auto c00 = corners[0];
            const auto c10 = corners[1];
            const auto c01 = corners[2];
            const auto c11 = corners[3];

            // Single-pass bilinear: one rounding per channel at pack time.
            const auto wx0 = 1.0F - fraction_x;
            const auto wy0 = 1.0F - fraction_y;
            std::array<float, 4> result{};
            for (std::size_t channel = 0; channel < 4U; ++channel) {
                const auto shift = static_cast<std::uint32_t>(channel * 8U);
                const auto v00 = static_cast<float>((c00 >> shift) & 0xFFU);
                const auto v10 = static_cast<float>((c10 >> shift) & 0xFFU);
                const auto v01 = static_cast<float>((c01 >> shift) & 0xFFU);
                const auto v11 = static_cast<float>((c11 >> shift) & 0xFFU);
                result[channel] = (v00 * wx0 + v10 * fraction_x)
                    * wy0
                    + (v01 * wx0 + v11 * fraction_x)
                    * fraction_y;
            }
            return result;
        }

        void build_mips() {
            while (levels_.back().width > 1U || levels_.back().height > 1U) {
                const auto& source = levels_.back();
                const auto next_width = std::max((source.width + 1U) / 2U, std::size_t{1});
                const auto next_height = std::max((source.height + 1U) / 2U, std::size_t{1});
                const typename View::mapping_type destination_mapping{
                    Extents{next_height, next_width},
                };

                TextureLevel destination{
                    .width = next_width,
                    .height = next_height,
                    .texels = std::vector<std::uint32_t>(destination_mapping.required_span_size()),
                };

                const typename View::mapping_type source_mapping{
                    Extents{source.height, source.width},
                };
                for (std::size_t y = 0; y < next_height; ++y) {
                    for (std::size_t x = 0; x < next_width; ++x) {
                        std::uint32_t sums[4]{};
                        std::size_t sample_count = 0;
                        for (std::size_t dy = 0; dy < 2U; ++dy) {
                            for (std::size_t dx = 0; dx < 2U; ++dx) {
                                const auto source_x = std::min(x * 2U + dx, source.width - 1U);
                                const auto source_y = std::min(y * 2U + dy, source.height - 1U);
                                const auto texel =
                                    source.texels[source_mapping(source_y, source_x)];
                                for (std::size_t channel = 0; channel < 4U; ++channel) {
                                    sums[channel] += (texel >> (channel * 8U)) & 0xFFU;
                                }
                                ++sample_count;
                            }
                        }

                        std::uint32_t result = 0;
                        for (std::size_t channel = 0; channel < 4U; ++channel) {
                            result |=
                                ((sums[channel] / static_cast<std::uint32_t>(sample_count)) & 0xFFU)
                                << (channel * 8U);
                        }
                        destination.texels[destination_mapping(y, x)] = result;
                    }
                }
                levels_.push_back(std::move(destination));
            }
        }

        std::size_t width_{};
        std::size_t height_{};
        std::vector<TextureLevel> levels_{};
    };

    struct QuadDerivatives {
        Vec2f duv_dx{};
        Vec2f duv_dy{};
    };

    [[nodiscard]]
    inline QuadDerivatives quad_derivatives(const std::array<Vec2f, 4>& uv) noexcept {
        return QuadDerivatives{
            .duv_dx =
                Vec2f{
                    .x = ((uv[1].x - uv[0].x) + (uv[3].x - uv[2].x)) * 0.5F,
                    .y = ((uv[1].y - uv[0].y) + (uv[3].y - uv[2].y)) * 0.5F,
                },
            .duv_dy = Vec2f{
                .x = ((uv[2].x - uv[0].x) + (uv[3].x - uv[1].x)) * 0.5F,
                .y = ((uv[2].y - uv[0].y) + (uv[3].y - uv[1].y)) * 0.5F,
            },
        };
    }

    [[nodiscard]]
    inline float mip_lod(
        const QuadDerivatives& derivatives, std::size_t texture_width, std::size_t texture_height
    ) noexcept {
        const auto dx_x = derivatives.duv_dx.x * static_cast<float>(texture_width);
        const auto dx_y = derivatives.duv_dx.y * static_cast<float>(texture_height);
        const auto dy_x = derivatives.duv_dy.x * static_cast<float>(texture_width);
        const auto dy_y = derivatives.duv_dy.y * static_cast<float>(texture_height);
        const auto vector_length = [](float x, float y) noexcept {
            return std::sqrt(x * x + y * y);
        };

        const auto rho = std::max(vector_length(dx_x, dx_y), vector_length(dy_x, dy_y));
        return std::max(std::log2(std::max(rho, 1.0F)), 0.0F);
    }
} // namespace raster
