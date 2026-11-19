#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mdspan>
#include <print>
#include <ranges>
#include <SDL3/SDL.h>
#include <vector>

#include "raster/memory/texture_layout.hpp"

namespace {
    using Extents = std::dextents<std::size_t, 2>;
    constexpr std::size_t texture_width = 1024;
    constexpr std::size_t texture_height = 1024;
    constexpr std::size_t quad_grid = texture_width / 2U;
    constexpr std::size_t warmup_passes = 2;
    constexpr std::size_t measured_passes = 7;
    constexpr std::size_t quad_count = quad_grid * quad_grid;

    void report_failure(const char* message) noexcept {
        try {
            std::println(stderr, "texture benchmark failed: {}", message);
        } catch (...) {
            std::terminate();
        }
    }

    struct QuadOrigin {
        std::uint32_t x{};
        std::uint32_t y{};
    };

    [[nodiscard]]
    std::vector<QuadOrigin> make_quad_order(bool randomized) {
        std::vector<QuadOrigin> order;
        order.reserve(quad_count);
        for (std::size_t y = 0; y < quad_grid; ++y) {
            for (std::size_t x = 0; x < quad_grid; ++x) {
                order.push_back(
                    QuadOrigin{
                        .x = static_cast<std::uint32_t>(x * 2U),
                        .y = static_cast<std::uint32_t>(y * 2U),
                    }
                );
            }
        }
        if (randomized) {
            std::uint32_t state = 0x52415354U;
            const auto next_random = [&state] noexcept {
                state ^= state << 13U;
                state ^= state >> 17U;
                state ^= state << 5U;
                return state;
            };
            for (std::size_t index = order.size(); index > 1U; --index) {
                const auto swap_index = static_cast<std::size_t>(next_random()) % index;
                std::swap(order[index - 1U], order[swap_index]);
            }
        }
        return order;
    }

    template <typename Mapping> [[nodiscard]]
    std::uint64_t run_pass(
        const Mapping& mapping,
        const std::vector<QuadOrigin>& order,
        const volatile std::uint32_t* texels,
        std::uint64_t& checksum
    ) noexcept {
        const auto start = SDL_GetTicksNS();
        for (const auto& quad : order) {
            const auto x = static_cast<std::size_t>(quad.x);
            const auto y = static_cast<std::size_t>(quad.y);
            checksum += texels[mapping(y, x)];
            checksum += texels[mapping(y, x + 1U)];
            checksum += texels[mapping(y + 1U, x)];
            checksum += texels[mapping(y + 1U, x + 1U)];
        }
        return SDL_GetTicksNS() - start;
    }

    template <typename Mapping> [[nodiscard]]
    double benchmark_layout(
        const char* name, const Mapping& mapping, const std::vector<QuadOrigin>& order
    ) {
        std::vector<std::uint32_t> storage(mapping.required_span_size());
        for (std::size_t y = 0; y < texture_height; ++y) {
            for (std::size_t x = 0; x < texture_width; ++x) {
                storage[mapping(y, x)] = static_cast<std::uint32_t>((x * 17U) ^ (y * 31U));
            }
        }

        std::uint64_t checksum = 0;
        const volatile std::uint32_t* const texels = storage.data();
        for ([[maybe_unused]]
             const auto pass : std::views::iota(std::size_t{0}, warmup_passes)) {
            static_cast<void>(run_pass(mapping, order, texels, checksum));
        }

        std::vector<std::uint64_t> timings;
        timings.reserve(measured_passes);
        for ([[maybe_unused]]
             const auto pass : std::views::iota(std::size_t{0}, measured_passes)) {
            timings.push_back(run_pass(mapping, order, texels, checksum));
        }

        std::ranges::sort(timings);
        const auto median_ns = timings[timings.size() / 2U];
        const auto ns_per_quad = static_cast<double>(median_ns) / static_cast<double>(quad_count);
        std::println(
            "{}: median {:.3f} ms/pass, {:.3f} ns/quad ({} runs), checksum={}",
            name,
            static_cast<double>(median_ns) / 1.0e6,
            ns_per_quad,
            measured_passes,
            checksum
        );
        return ns_per_quad;
    }
} // namespace

int main() noexcept {
    try {
        const Extents extents{texture_height, texture_width};
        const std::layout_right::mapping<Extents> row_major{extents};
        const raster::LayoutTiled4x4::mapping<Extents> tiled{extents};
        const raster::LayoutMorton::mapping<Extents> morton{extents};
        const auto sequential_order = make_quad_order(false);
        const auto randomized_order = make_quad_order(true);
        std::println(
            "{}x{} texture; {} warmups, {} measured full quad passes (median)",
            texture_width,
            texture_height,
            warmup_passes,
            measured_passes
        );

        const auto row_major_ns =
            benchmark_layout("layout_right sequential", row_major, sequential_order);
        const auto tiled_ns = benchmark_layout("tiled4x4 sequential", tiled, sequential_order);
        const auto morton_ns = benchmark_layout("morton sequential", morton, sequential_order);
        const auto row_major_random_ns =
            benchmark_layout("layout_right shuffled", row_major, randomized_order);

        const auto tiled_random_ns = benchmark_layout("tiled4x4 shuffled", tiled, randomized_order);
        const auto morton_random_ns = benchmark_layout("morton shuffled", morton, randomized_order);
        std::println(
            "sequential relative to layout_right: tiled {:.2f}x, Morton {:.2f}x",
            row_major_ns / tiled_ns,
            row_major_ns / morton_ns
        );
        std::println(
            "shuffled relative to layout_right: tiled {:.2f}x, Morton {:.2f}x",
            row_major_random_ns / tiled_random_ns,
            row_major_random_ns / morton_random_ns
        );
    } catch (const std::exception& error) {
        report_failure(error.what());
        return 1;
    } catch (...) {
        report_failure("unknown exception");
        return 1;
    }
    return 0;
}
