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

    namespace detail {
        // Conservative half-open tile span for one axis: empty/offscreen
        // boxes collapse to {0, 0}. Shared by the serial and parallel
        // binning passes so both agree on every triangle exactly.
        [[nodiscard]]
        inline std::pair<std::size_t, std::size_t> clamped_tile_range(
            std::int32_t minimum, std::int32_t maximum, std::size_t extent
        ) noexcept {
            const auto lower = std::max<std::int64_t>(minimum, 0);
            const auto upper = std::min<std::int64_t>(maximum, static_cast<std::int64_t>(extent));
            if (lower >= upper) {
                return std::pair<std::size_t, std::size_t>{0uz, 0uz};
            }
            return std::pair{
                static_cast<std::size_t>(lower) / raster_tile_size,
                (static_cast<std::size_t>(upper) - 1U) / raster_tile_size + 1U,
            };
        }
    } // namespace detail

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

        template <typename Executor, typename Prepared> friend void bin_triangles(
            const std::vector<std::vector<Prepared>>&,
            std::size_t,
            std::size_t,
            TileBins&,
            Executor&
        );

        std::size_t tile_columns_{};
        std::size_t tile_rows_{};
        std::vector<std::size_t> offsets_{};
        std::vector<std::size_t> triangle_indices_{};
        std::vector<std::size_t> write_offsets_{};
        // Per-worker scratch for the parallel binning pass: worker w owns
        // slice [w * stride, (w + 1) * stride) with stride = tile_count + 1.
        std::vector<std::size_t> worker_cursors_{};
    };

    // Parallel counting-sort binning over per-worker triangle vectors with
    // identical per-tile order to the serial entry: worker w holds a
    // contiguous slice of global input order, so concatenating the slices
    // reproduces it exactly. Stored indices pack the source location as
    // (worker << 32) | local; the rasterizer unpacks them to fetch directly
    // from the worker vectors, which removes the merged-array copies.
    namespace detail {
        [[nodiscard]]
        inline std::size_t pack_triangle_index(std::size_t worker, std::size_t local) noexcept {
            return (worker << 32U) | local;
        }

        [[nodiscard]]
        inline std::size_t unpack_triangle_worker(std::size_t packed) noexcept {
            return packed >> 32U;
        }

        [[nodiscard]]
        inline std::size_t unpack_triangle_local(std::size_t packed) noexcept {
            return packed & 0xFFFFFFFFU;
        }
    } // namespace detail

    template <typename Executor, typename Prepared> void bin_triangles(
        const std::vector<std::vector<Prepared>>& worker_triangles,
        std::size_t viewport_width,
        std::size_t viewport_height,
        TileBins& bins,
        Executor& executor
    ) {
        const auto worker_count = worker_triangles.size();
        const auto grid = [&] noexcept {
            const auto columns = viewport_width
                / raster_tile_size
                + static_cast<std::size_t>(viewport_width % raster_tile_size != 0U);
            const auto rows = viewport_height
                / raster_tile_size
                + static_cast<std::size_t>(viewport_height % raster_tile_size != 0U);
            return std::pair{columns, rows};
        };

        const auto [tile_columns, tile_rows] = grid();
        const auto tile_count = tile_columns * tile_rows;
        if (tile_columns == 0U || tile_rows == 0U) {
            bins.offsets_.clear();
            bins.triangle_indices_.clear();
            bins.write_offsets_.clear();
            bins.worker_cursors_.clear();
            bins.tile_columns_ = tile_columns;
            bins.tile_rows_ = tile_rows;
            return;
        }

        bins.tile_columns_ = tile_columns;
        bins.tile_rows_ = tile_rows;
        if (bins.offsets_.size() != tile_count + 1U) {
            bins.offsets_.resize(tile_count + 1U);
        }

        std::ranges::fill(bins.offsets_, 0U);
        const auto cursor_stride = tile_count + 1U;
        if (bins.worker_cursors_.size() != worker_count * cursor_stride) {
            bins.worker_cursors_.resize(worker_count * cursor_stride);
        }

        executor.execute(worker_count, [&](std::size_t worker) noexcept {
            auto* const local = bins.worker_cursors_.data() + worker * cursor_stride;
            for (std::size_t slot = 0U; slot < cursor_stride; ++slot) {
                local[slot] = 0U;
            }
            for (const auto& triangle : worker_triangles[worker]) {
                const auto [first_column, column_end] = detail::clamped_tile_range(
                    triangle.setup.min_x,
                    triangle.setup.max_x,
                    viewport_width
                );
                const auto [first_row, row_end] = detail::clamped_tile_range(
                    triangle.setup.min_y,
                    triangle.setup.max_y,
                    viewport_height
                );
                for (std::size_t row = first_row; row < row_end; ++row) {
                    for (std::size_t column = first_column; column < column_end; ++column) {
                        ++local[row * tile_columns + column + 1U];
                    }
                }
            }
        });

        for (std::size_t tile = 0U; tile < tile_count; ++tile) {
            std::size_t running = bins.offsets_[tile];
            for (std::size_t worker = 0U; worker < worker_count; ++worker) {
                auto* const local = bins.worker_cursors_.data() + worker * cursor_stride;
                const auto count = local[tile + 1U];
                local[tile] = running;
                running += count;
            }
            bins.offsets_[tile + 1U] = running;
        }

        bins.triangle_indices_.resize(bins.offsets_.back());
        executor.execute(worker_count, [&](std::size_t worker) noexcept {
            auto* const local = bins.worker_cursors_.data() + worker * cursor_stride;
            const auto& chunk = worker_triangles[worker];
            for (std::size_t index = 0U; index < chunk.size(); ++index) {
                const auto [first_column, column_end] = detail::clamped_tile_range(
                    chunk[index].setup.min_x,
                    chunk[index].setup.max_x,
                    viewport_width
                );
                const auto [first_row, row_end] = detail::clamped_tile_range(
                    chunk[index].setup.min_y,
                    chunk[index].setup.max_y,
                    viewport_height
                );
                for (std::size_t row = first_row; row < row_end; ++row) {
                    for (std::size_t column = first_column; column < column_end; ++column) {
                        const auto tile = row * tile_columns + column;
                        bins.triangle_indices_[local[tile]++] =
                            detail::pack_triangle_index(worker, index);
                    }
                }
            }
        });
    }

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
        for (const auto& triangle : triangles) {
            const auto [first_column, column_end] =
                detail::clamped_tile_range(triangle.min_x, triangle.max_x, viewport_width);
            const auto [first_row, row_end] =
                detail::clamped_tile_range(triangle.min_y, triangle.max_y, viewport_height);
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
                detail::clamped_tile_range(triangle.min_x, triangle.max_x, viewport_width);
            const auto [first_row, row_end] =
                detail::clamped_tile_range(triangle.min_y, triangle.max_y, viewport_height);
            for (std::size_t row = first_row; row < row_end; ++row) {
                for (std::size_t column = first_column; column < column_end; ++column) {
                    const auto tile = row * bins.tile_columns_ + column;
                    bins.triangle_indices_[bins.write_offsets_[tile]++] = triangle_index;
                }
            }
        }
    }
} // namespace raster
