#pragma once

#include <algorithm>
#include <bit>

namespace raster {
    struct LayoutTiled4x4;
    struct LayoutMorton;

    namespace detail {
        template <typename Extents> class LayoutTiled4x4Mapping {
        public:

            using extents_type = Extents;
            using layout_type = LayoutTiled4x4;
            using index_type = extents_type::index_type;
            using size_type = extents_type::size_type;
            using rank_type = extents_type::rank_type;

            static constexpr bool is_always_unique() noexcept {
                return true;
            }

            static constexpr bool is_always_exhaustive() noexcept {
                return false;
            }

            static constexpr bool is_always_strided() noexcept {
                return false;
            }

            constexpr LayoutTiled4x4Mapping() noexcept = default;

            constexpr explicit LayoutTiled4x4Mapping(const extents_type& extents) noexcept :
                extents_{extents} {}

            [[nodiscard]]
            constexpr const extents_type& extents() const noexcept {
                return extents_;
            }

            [[nodiscard]]
            constexpr index_type operator()(index_type row, index_type column) const noexcept {
                const auto tiles_x = (extent(1) + 3U) / 4U;
                const auto tile = (row / 4U) * tiles_x + column / 4U;
                const auto in_tile = (row % 4U) * 4U + column % 4U;
                return static_cast<index_type>(tile * 16U + in_tile);
            }

            [[nodiscard]]
            constexpr size_type required_span_size() const noexcept {
                const auto tiles_x = (extent(1) + 3U) / 4U;
                const auto tiles_y = (extent(0) + 3U) / 4U;
                return static_cast<size_type>(tiles_x * tiles_y * 16U);
            }

            [[nodiscard]]
            constexpr bool is_unique() const noexcept {
                return true;
            }

            [[nodiscard]]
            constexpr bool is_exhaustive() const noexcept {
                return extent(0) % 4U == 0U && extent(1) % 4U == 0U;
            }

            [[nodiscard]]
            constexpr bool is_strided() const noexcept {
                return false;
            }

            friend constexpr bool operator==(
                const LayoutTiled4x4Mapping&, const LayoutTiled4x4Mapping&
            ) noexcept = default;

        private:

            [[nodiscard]]
            constexpr size_type extent(rank_type dimension) const noexcept {
                return static_cast<size_type>(extents_.extent(dimension));
            }

            extents_type extents_{};
        };

        template <typename Extents> class LayoutMortonMapping {
        public:

            using extents_type = Extents;
            using layout_type = LayoutMorton;
            using index_type = extents_type::index_type;
            using size_type = extents_type::size_type;
            using rank_type = extents_type::rank_type;

            static constexpr bool is_always_unique() noexcept {
                return true;
            }

            static constexpr bool is_always_exhaustive() noexcept {
                return false;
            }

            static constexpr bool is_always_strided() noexcept {
                return false;
            }

            constexpr LayoutMortonMapping() noexcept = default;

            constexpr explicit LayoutMortonMapping(const extents_type& extents) noexcept :
                extents_{extents} {}

            [[nodiscard]]
            constexpr const extents_type& extents() const noexcept {
                return extents_;
            }

            [[nodiscard]]
            constexpr index_type operator()(index_type row, index_type column) const noexcept {
                size_type morton = 0;
                const auto bits = static_cast<size_type>(std::bit_width(padded_side() - 1U));
                for (size_type bit = 0; bit < bits; ++bit) {
                    morton |= ((static_cast<size_type>(column) >> bit) & 1U) << (2U * bit);
                    morton |= ((static_cast<size_type>(row) >> bit) & 1U) << (2U * bit + 1U);
                }
                return static_cast<index_type>(morton);
            }

            [[nodiscard]]
            constexpr size_type required_span_size() const noexcept {
                const auto side = padded_side();
                return side * side;
            }

            [[nodiscard]]
            constexpr bool is_unique() const noexcept {
                return true;
            }

            [[nodiscard]]
            constexpr bool is_exhaustive() const noexcept {
                return extent(0) == padded_side() && extent(1) == padded_side();
            }

            [[nodiscard]]
            constexpr bool is_strided() const noexcept {
                return false;
            }

            friend constexpr bool operator==(
                const LayoutMortonMapping&, const LayoutMortonMapping&
            ) noexcept = default;

        private:

            [[nodiscard]]
            constexpr size_type extent(rank_type dimension) const noexcept {
                return static_cast<size_type>(extents_.extent(dimension));
            }

            [[nodiscard]]
            constexpr size_type padded_side() const noexcept {
                return std::bit_ceil(std::max(extent(0), extent(1)));
            }

            extents_type extents_{};
        };
    } // namespace detail

    struct LayoutTiled4x4 {
        template <typename Extents> using mapping = detail::LayoutTiled4x4Mapping<Extents>;
    };

    struct LayoutMorton {
        template <typename Extents> using mapping = detail::LayoutMortonMapping<Extents>;
    };
} // namespace raster
