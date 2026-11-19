#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <vector>

#include "raster/io/tga_loader.hpp"
#include "raster/math/vec.hpp"

namespace raster {
    struct TextureLevel {
        std::size_t width{};
        std::size_t height{};
        std::vector<std::uint32_t> texels{};
    };

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

    private:

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
        const auto rho = std::max(std::hypot(dx_x, dx_y), std::hypot(dy_x, dy_y));
        return std::max(std::log2(std::max(rho, 1.0F)), 0.0F);
    }
} // namespace raster
