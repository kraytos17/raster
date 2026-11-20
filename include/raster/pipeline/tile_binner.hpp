#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "raster/pipeline/rasterizer.hpp"

namespace raster {
    inline constexpr std::size_t raster_tile_size = 16U;

    // Compact CSR-style triangle lists, one stable input-order list per tile.
    // The result is intended to be built once per frame and consumed by a tile
    // scheduler; it does not own or modify the ScreenTriangle records.
    class TileBins {
    public:

        void reserve(std::size_t tile_count, std::size_t triangle_reference_count) {
            offsets_.reserve(tile_count + 1U);
            write_offsets_.reserve(tile_count);
            triangle_indices_.reserve(triangle_reference_count);
        }

        [[nodiscard]]
        std::size_t tile_columns() const noexcept {
            return tile_columns_;
        }

        [[nodiscard]]
        std::size_t tile_rows() const noexcept {
            return tile_rows_;
        }

        [[nodiscard]]
        std::size_t tile_count() const noexcept {
            return tile_columns_ * tile_rows_;
        }

        [[nodiscard]]
        std::span<const std::size_t> triangle_indices(std::size_t tile_index) const noexcept {
            if (tile_index >= tile_count()) {
                return {};
            }
            const auto begin = offsets_[tile_index];
            const auto end = offsets_[tile_index + 1U];
            return std::span<const std::size_t>{triangle_indices_}.subspan(begin, end - begin);
        }

    private:

        friend void bin_triangles(
            std::span<const ScreenTriangle>, std::size_t, std::size_t, TileBins&
        );

        std::size_t tile_columns_{};
        std::size_t tile_rows_{};
        std::vector<std::size_t> offsets_{};
        std::vector<std::size_t> triangle_indices_{};
        std::vector<std::size_t> write_offsets_{};
    };

    // Bin each triangle's conservative half-open screen-space bounding box.
    // Empty/offscreen boxes are ignored. Triangle order in every bin matches
    // input order, preserving deterministic depth behavior within each tile.
    inline void bin_triangles(
        std::span<const ScreenTriangle> triangles,
        std::size_t viewport_width,
        std::size_t viewport_height,
        TileBins& bins
    ) {
        bins.tile_columns_ = viewport_width
            / raster_tile_size
            + static_cast<std::size_t>(viewport_width % raster_tile_size != 0U);
        bins.tile_rows_ = viewport_height
            / raster_tile_size
            + static_cast<std::size_t>(viewport_height % raster_tile_size != 0U);
        if (bins.tile_columns_ == 0U || bins.tile_rows_ == 0U) {
            bins.offsets_.clear();
            bins.triangle_indices_.clear();
            bins.write_offsets_.clear();
            return;
        }

        const auto tile_count = bins.tile_columns_ * bins.tile_rows_;
        bins.offsets_.resize(tile_count + 1U);
        std::ranges::fill(bins.offsets_, 0U);
        const auto clamped_tile_range =
            [](std::int32_t minimum, std::int32_t maximum, std::size_t extent) noexcept {
            const auto lower = std::max<std::int64_t>(minimum, 0);
            const auto upper = std::min<std::int64_t>(maximum, static_cast<std::int64_t>(extent));
            if (lower >= upper) {
                return std::pair<std::size_t, std::size_t>{0U, 0U};
            }
            return std::pair{
                static_cast<std::size_t>(lower) / raster_tile_size,
                (static_cast<std::size_t>(upper) - 1U) / raster_tile_size + 1U,
            };
        };

        for (const auto& triangle : triangles) {
            const auto [first_column, column_end] =
                clamped_tile_range(triangle.min_x, triangle.max_x, viewport_width);
            const auto [first_row, row_end] =
                clamped_tile_range(triangle.min_y, triangle.max_y, viewport_height);
            for (std::size_t row = first_row; row < row_end; ++row) {
                for (std::size_t column = first_column; column < column_end; ++column) {
                    ++bins.offsets_[row * bins.tile_columns_ + column + 1U];
                }
            }
        }
        for (std::size_t tile = 0; tile < tile_count; ++tile) {
            bins.offsets_[tile + 1U] += bins.offsets_[tile];
        }

        bins.triangle_indices_.resize(bins.offsets_.back());
        bins.write_offsets_.resize(tile_count);
        std::ranges::copy(
            bins.offsets_.begin(),
            bins.offsets_.end() - 1U,
            bins.write_offsets_.begin()
        );
        for (std::size_t triangle_index = 0; triangle_index < triangles.size(); ++triangle_index) {
            const auto& triangle = triangles[triangle_index];
            const auto [first_column, column_end] =
                clamped_tile_range(triangle.min_x, triangle.max_x, viewport_width);
            const auto [first_row, row_end] =
                clamped_tile_range(triangle.min_y, triangle.max_y, viewport_height);
            for (std::size_t row = first_row; row < row_end; ++row) {
                for (std::size_t column = first_column; column < column_end; ++column) {
                    const auto tile = row * bins.tile_columns_ + column;
                    bins.triangle_indices_[bins.write_offsets_[tile]++] = triangle_index;
                }
            }
        }
    }
} // namespace raster
