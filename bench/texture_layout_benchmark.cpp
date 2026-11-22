#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <mdspan>
#include <memory>
#include <print>
#include <random>
#include <ranges>
#include <SDL3/SDL.h>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/memory/grid_mesh.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"
#include "raster/render/scene_renderer.hpp"

namespace {
    using Extents = std::dextents<std::size_t, 2>;
    constexpr std::size_t texture_width = 4096;
    constexpr std::size_t texture_height = 4096;
    constexpr std::size_t quad_grid = texture_width / 2U;
    constexpr std::size_t warmup_passes = 1;
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

    template <typename Texture> [[nodiscard]]
    double benchmark_render_scene(
        const char* name,
        const raster::SoAMesh& mesh,
        const Texture& texture,
        std::size_t width,
        std::size_t height
    ) {
        raster::Framebuffer framebuffer{width, height};
        constexpr std::size_t warmups = 1U;
        constexpr std::size_t measured = 3U;
        const auto render_once = [&] noexcept {
            return raster::render::render_textured_model(framebuffer, mesh, texture);
        };
        for ([[maybe_unused]]
             const auto pass : std::views::iota(std::size_t{0}, warmups)) {
            if (!render_once()) {
                throw std::runtime_error("scene benchmark warmup render failed");
            }
        }

        std::array<std::uint64_t, measured> times{};
        for (std::size_t pass = 0; pass < warmups + measured; ++pass) {
            const auto start = SDL_GetTicksNS();
            if (!render_once()) {
                throw std::runtime_error("scene benchmark render failed");
            }

            const auto elapsed = SDL_GetTicksNS() - start;
            if (pass >= warmups) {
                times[pass - warmups] = elapsed;
            }
        }

        std::ranges::sort(times);
        const auto median = times[times.size() / 2U];
        std::println(
            "{}: 4K texture + {} triangles at {}x{}, median {:.3f} ms/frame ({} runs)",
            name,
            mesh.triangle_count(),
            width,
            height,
            static_cast<double>(median) / 1.0e6,
            measured
        );
        return static_cast<double>(median);
    }

    template <typename Texture> [[nodiscard]]
    double benchmark_scene_configuration(
        const char* name,
        const raster::SoAMesh& mesh,
        const Texture& texture,
        std::size_t width,
        std::size_t height
    ) {
        constexpr std::size_t repeats = 1U;
        std::array<double, repeats> times{};
        for (auto& time : times) {
            time = benchmark_render_scene(name, mesh, texture, width, height);
        }

        std::ranges::sort(times);
        return times[times.size() / 2U];
    }

    template <typename Render> [[nodiscard]]
    std::uint64_t benchmark_frame_times(
        Render& render, const char* failure_message, std::size_t warmups, std::size_t measured
    ) {
        for ([[maybe_unused]]
             const auto pass : std::views::iota(std::size_t{0}, warmups)) {
            if (!render()) {
                throw std::runtime_error(failure_message);
            }
        }

        std::vector<std::uint64_t> times(measured);
        for (auto& elapsed : times) {
            const auto start = SDL_GetTicksNS();
            if (!render()) {
                throw std::runtime_error(failure_message);
            }
            elapsed = SDL_GetTicksNS() - start;
        }

        std::ranges::sort(times);
        return times[times.size() / 2U];
    }

    void benchmark_thread_scaling(
        const raster::SoAMesh& mesh,
        const raster::Texture2D<raster::LayoutTiled4x4>& texture,
        std::size_t width,
        std::size_t height,
        const raster::render::FreeLookCamera& camera,
        std::string_view label,
        std::size_t warmups,
        std::size_t measured
    ) {
        constexpr std::array<std::size_t, 4> concurrency_levels{1U, 2U, 4U, 8U};
        raster::Framebuffer serial_framebuffer{width, height};
        const auto render_serial = [&] noexcept {
            return raster::render::render_scene(
                serial_framebuffer,
                mesh,
                0U,
                false,
                true,
                false,
                false,
                texture,
                camera
            );
        };

        const auto serial_median =
            benchmark_frame_times(render_serial, "serial render failed", warmups, measured);
        std::println(
            "serial {}: {} triangles at {}x{}, median {:.3f} ms/frame ({} runs)",
            label,
            mesh.triangle_count(),
            width,
            height,
            static_cast<double>(serial_median) / 1.0e6,
            measured
        );
        for (const auto concurrency : concurrency_levels) {
            raster::render::TileRenderer renderer{concurrency};
            raster::Framebuffer threaded_framebuffer{width, height};
            const auto render_threaded = [&] noexcept {
                return renderer.render_scene(
                    threaded_framebuffer,
                    mesh,
                    0U,
                    false,
                    true,
                    false,
                    false,
                    texture,
                    camera
                );
            };

            const auto threaded_median =
                benchmark_frame_times(render_threaded, "threaded render failed", warmups, measured);
            std::println(
                "tile concurrency {}: median {:.3f} ms/frame, {:.2f}x serial ({} runs)",
                concurrency,
                static_cast<double>(threaded_median) / 1.0e6,
                static_cast<double>(serial_median) / static_cast<double>(threaded_median),
                measured
            );
        }
    }

    void benchmark_scene_threads(
        const raster::SoAMesh& mesh,
        const raster::Texture2D<raster::LayoutTiled4x4>& texture,
        std::size_t width,
        std::size_t height
    ) {
        constexpr raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 0.0F, .z = 3.0F},
            .yaw = 0.0F,
            .pitch = 0.0F,
        };
        benchmark_thread_scaling(
            mesh,
            texture,
            width,
            height,
            camera,
            "tiled-layout baseline",
            2U,
            7U
        );
    }

    void benchmark_scene_stress(
        const raster::Texture2D<raster::LayoutTiled4x4>& texture,
        std::size_t width,
        std::size_t height,
        std::size_t columns
    ) {
        // Heavy-vertex floor viewed at a grazing angle: prepares and bins
        // columns x columns x 2 triangles, exercising prepare/bin scaling
        // plus minified UVs and mip paths in rasterization.
        const auto mesh = raster::make_grid_mesh(columns, columns, 30.0F, -0.5F, -90.0F, 8.0F);
        constexpr raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.0F, .y = 6.0F, .z = 6.0F},
            .yaw = 0.0F,
            .pitch = -0.19F,
        };
        benchmark_thread_scaling(mesh, texture, width, height, camera, "grid stress", 1U, 3U);
    }

    struct CompactHeapNode {
        const CompactHeapNode* next{};
        std::uint64_t payload{};
        std::array<std::uint64_t, 2> padding{};
    };

    struct ScatteredHeapNode {
        const ScatteredHeapNode* next{};
        std::uint64_t payload{};
        std::array<std::uint64_t, 2> padding{};
    };

    static_assert(sizeof(CompactHeapNode) == 32U);
    static_assert(sizeof(ScatteredHeapNode) == 32U);

    struct CompactHeapChain {
        std::vector<CompactHeapNode> nodes;

        CompactHeapChain(std::size_t count, std::uint64_t seed) :
            nodes(count) {
            for (std::size_t index = 0; index < count; ++index) {
                nodes[index].payload =
                    seed + static_cast<std::uint64_t>(index) * 0x9E3779B97F4A7C15ULL;
            }
            relink();
        }

        void relink() noexcept {
            for (std::size_t index = 0; index < nodes.size(); ++index) {
                nodes[index].next = index + 1U < nodes.size() ? &nodes[index + 1U] : nullptr;
            }
        }
    };

    using ScatteredHeapChain = std::vector<std::unique_ptr<ScatteredHeapNode>>;

    [[nodiscard]]
    ScatteredHeapChain make_scattered_heap_chain(std::size_t count, std::uint64_t seed) {
        ScatteredHeapChain storage;
        storage.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            auto node = std::make_unique<ScatteredHeapNode>();
            node->payload = seed + static_cast<std::uint64_t>(index) * 0x9E3779B97F4A7C15ULL;
            storage.push_back(std::move(node));
        }
        for (std::size_t index = 0; index + 1U < storage.size(); ++index) {
            storage[index]->next = storage[index + 1U].get();
        }
        return storage;
    }

    [[nodiscard]]
    std::uint64_t scan_heap_chain(
        const CompactHeapChain& storage, std::size_t repetitions
    ) noexcept {
        if (storage.nodes.empty()) {
            return 0U;
        }

        std::uint64_t checksum = 0U;
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            const CompactHeapNode* node = storage.nodes.data();
            for (std::size_t visited = 0; visited < storage.nodes.size(); ++visited) {
                checksum = checksum * 0x100000001B3ULL + node->payload;
                node = node->next;
            }
        }
        return checksum;
    }

    [[nodiscard]]
    std::uint64_t scan_heap_chain(
        const ScatteredHeapChain& storage, std::size_t repetitions
    ) noexcept {
        if (storage.empty()) {
            return 0U;
        }

        std::uint64_t checksum = 0U;
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            const ScatteredHeapNode* node = storage.front().get();
            for (std::size_t index = 0; index < storage.size(); ++index) {
                checksum = checksum * 0x100000001B3ULL + node->payload;
                node = node->next;
            }
        }
        return checksum;
    }

    template <typename Function> [[nodiscard]]
    std::uint64_t median_timing(Function function, std::size_t warmups, std::size_t samples) {
        for (std::size_t warmup = 0; warmup < warmups; ++warmup) {
            static_cast<void>(function());
        }

        std::vector<std::uint64_t> timings;
        timings.reserve(samples);
        std::uint64_t checksum = 0U;
        for (std::size_t sample = 0; sample < samples; ++sample) {
            const auto begin = SDL_GetTicksNS();
            checksum ^= function();
            timings.push_back(SDL_GetTicksNS() - begin);
        }

        std::ranges::sort(timings);
        const auto median = timings[timings.size() / 2U];
        std::vector<std::uint64_t> deviations;
        deviations.reserve(timings.size());
        for (const auto timing : timings) {
            deviations.push_back(timing > median ? timing - median : median - timing);
        }

        std::ranges::sort(deviations);
        const auto mad = deviations[deviations.size() / 2U];
        std::println(
            "median {:.3f} ms, MAD {:.3f} ms ({} runs), checksum={}",
            static_cast<double>(median) / 1.0e6,
            static_cast<double>(mad) / 1.0e6,
            samples,
            checksum
        );
        return median;
    }

    [[nodiscard]]
    std::vector<std::vector<std::byte>> make_heap_churn(std::size_t seed, std::size_t& live_bytes) {
        constexpr std::array<std::size_t, 5> block_sizes{32U, 64U, 256U, 1024U, 4096U};
        constexpr std::size_t block_count = 2048U;

        std::mt19937 random{static_cast<std::mt19937::result_type>(seed)};
        std::vector<std::vector<std::byte>> ballast;
        ballast.reserve(block_count + block_count / 2U);
        live_bytes = 0U;
        for (std::size_t index = 0; index < block_count; ++index) {
            const auto size = block_sizes[random() % block_sizes.size()];
            ballast.emplace_back(size, static_cast<std::byte>(random() & 0xFFU));
            live_bytes += size;
        }
        for (std::size_t index = 0; index < ballast.size(); index += 2U) {
            live_bytes -= ballast[index].size();
            std::vector<std::byte>{}.swap(ballast[index]);
        }
        for (std::size_t index = 0; index < block_count / 2U; ++index) {
            const auto size = block_sizes[random() % block_sizes.size()];
            ballast.emplace_back(size, static_cast<std::byte>(random() & 0xFFU));
            live_bytes += size;
        }
        return ballast;
    }

    [[nodiscard]]
    bool render_aging_scene(
        raster::render::TileRenderer& renderer,
        raster::Framebuffer& framebuffer,
        const raster::SoAMesh& mesh,
        const raster::Texture2D<raster::LayoutTiled4x4>& texture,
        const raster::render::FreeLookCamera& camera
    ) noexcept {
        return renderer
            .render_scene(framebuffer, mesh, 0U, false, true, false, false, texture, camera);
    }

    struct AgingRenderer {
        raster::Framebuffer framebuffer;
        raster::render::TileRenderer renderer;

        AgingRenderer(std::size_t width, std::size_t height) :
            framebuffer{width, height},
            renderer{4U} {}

        [[nodiscard]]
        bool render(
            const raster::SoAMesh& mesh,
            const raster::Texture2D<raster::LayoutTiled4x4>& texture,
            const raster::render::FreeLookCamera& camera
        ) noexcept {
            return render_aging_scene(renderer, framebuffer, mesh, texture, camera);
        }
    };

    void benchmark_heap_aging(
        const raster::Texture2D<raster::LayoutTiled4x4>& texture,
        std::size_t width,
        std::size_t height,
        std::size_t chain_count,
        std::size_t churn_seed
    ) {
        constexpr std::size_t scan_repetitions = 16U;
        constexpr std::size_t warmups = 2U;
        constexpr std::size_t samples = 9U;
        std::println(
            "heap-aging: nodes={} (32 B), churn seed={}, scan repetitions={}, warmups={}, samples={} (diagnostic; no thresholds)",
            chain_count,
            churn_seed,
            scan_repetitions,
            warmups,
            samples
        );

        const auto mesh_result = raster::load_obj(
            std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
        );
        if (!mesh_result) {
            throw std::runtime_error("could not load bundled clipping sphere OBJ");
        }

        const raster::render::FreeLookCamera camera{
            .position = raster::Vec3f{.x = 0.1F, .y = -0.2F, .z = 3.5F},
            .yaw = 0.07F,
            .pitch = 0.03F,
        };

        // Allocate the synthetic contiguous control before either renderer
        // fixture so it cannot explain a pre/post placement difference.
        auto compact = CompactHeapChain{chain_count, 0x434F4D50414354ULL};
        const auto compact_checksum = scan_heap_chain(compact, scan_repetitions);
        std::println("contiguous chain before churn:");
        static_cast<void>(median_timing([&] {
            return scan_heap_chain(compact, scan_repetitions);
        }, warmups, samples));

        // Fixture A is fully allocated/warmed before churn and remains live.
        auto fixture_before = std::make_unique<AgingRenderer>(width, height);
        const auto render_before = [&]() noexcept {
            return fixture_before->render(*mesh_result, texture, camera);
        };
        if (!render_before()) {
            throw std::runtime_error("heap-aging renderer baseline failed");
        }

        const auto expected_color = fixture_before->framebuffer.color_storage;
        const auto expected_depth = fixture_before->framebuffer.depth_storage;

        std::size_t churn_live_bytes = 0U;
        auto churn = make_heap_churn(churn_seed, churn_live_bytes);

        // Allocate both comparison fixtures back-to-back after churn. A's
        // allocations live from before churn; B's are new in the aged heap.
        auto fixture_after = std::make_unique<AgingRenderer>(width, height);
        const auto render_after = [&] noexcept {
            return fixture_after->render(*mesh_result, texture, camera);
        };

        constexpr std::size_t pair_warmups = warmups;
        constexpr std::size_t pair_samples = samples;
        for (std::size_t warmup = 0; warmup < pair_warmups; ++warmup) {
            if (!render_before() || !render_after()) {
                throw std::runtime_error("paired renderer warmup failed");
            }
        }

        std::array<std::uint64_t, pair_samples> before_times{};
        std::array<std::uint64_t, pair_samples> after_times{};
        const auto time_render = [](const auto& render_fn) {
            const auto begin = SDL_GetTicksNS();
            if (!render_fn()) {
                throw std::runtime_error("paired heap-aging render failed");
            }
            return SDL_GetTicksNS() - begin;
        };
        for (std::size_t sample = 0; sample < pair_samples; ++sample) {
            if ((sample & 1U) == 0U) {
                before_times[sample] = time_render(render_before);
                after_times[sample] = time_render(render_after);
            } else {
                after_times[sample] = time_render(render_after);
                before_times[sample] = time_render(render_before);
            }

            if (fixture_before->framebuffer.color_storage
                != expected_color
                || fixture_before->framebuffer.depth_storage
                != expected_depth
                || fixture_after->framebuffer.color_storage
                != expected_color
                || fixture_after->framebuffer.depth_storage
                != expected_depth) {
                throw std::runtime_error("heap-aging render output changed");
            }
        }

        const auto report_paired =
            [pair_samples](std::string_view label, std::array<std::uint64_t, samples> times) {
            std::ranges::sort(times);
            const auto median = times[pair_samples / 2U];
            std::array<std::uint64_t, samples> deviations{};
            for (std::size_t index = 0; index < pair_samples; ++index) {
                deviations[index] =
                    times[index] > median ? times[index] - median : median - times[index];
            }

            std::ranges::sort(deviations);
            std::println(
                "{}: median {:.3f} ms, MAD {:.3f} ms ({} alternating pairs)",
                label,
                static_cast<double>(median) / 1.0e6,
                static_cast<double>(deviations[pair_samples / 2U]) / 1.0e6,
                pair_samples
            );
            return median;
        };

        const auto renderer_before_ns =
            report_paired("renderer fixture allocated before churn", before_times);
        const auto renderer_after_ns =
            report_paired("renderer fixture allocated after churn", after_times);
        std::println(
            "post-churn/pre-churn fixture ratio: {:.3f} (paired diagnostic only)",
            static_cast<double>(renderer_after_ns) / static_cast<double>(renderer_before_ns)
        );

        // The retained vector's storage cannot be relocated by churn; rescan
        // it as the control for process-wide side-effects and verify checksum.
        std::println("same contiguous chain after churn:");
        static_cast<void>(median_timing([&] {
            return scan_heap_chain(compact, scan_repetitions);
        }, warmups, samples));

        const auto post_churn_compact_checksum = scan_heap_chain(compact, scan_repetitions);
        std::println("retained compact chain after mixed-size churn:");
        static_cast<void>(median_timing([&] {
            return scan_heap_chain(compact, scan_repetitions);
        }, warmups, samples));
        if (post_churn_compact_checksum != compact_checksum) {
            throw std::runtime_error("compact chain checksum changed across scan phases");
        }

        // Per-node allocations expose allocator placement after churn. Keep
        // this separate from the contiguous vector control and renderer timing.
        auto aged = make_scattered_heap_chain(chain_count, 0x414745442D434841ULL);
        std::println("post-churn per-node chain (placement control):");
        static_cast<void>(median_timing([&] {
            return scan_heap_chain(aged, scan_repetitions);
        }, warmups, samples));

        if (!render_after()) {
            throw std::runtime_error("post-churn warmed render failed");
        }
        std::println(
            "churn blocks live={}, bytes={}; allocation invariant is checked by the dedicated Catch2 test",
            churn.size(),
            churn_live_bytes
        );
    }

    [[nodiscard]]
    raster::TgaImage make_checker_4k() {
        raster::TgaImage image{};
        image.width = texture_width;
        image.height = texture_height;
        image.pixels.resize(texture_width * texture_height);
        constexpr std::size_t cell_size = 16U;
        for (std::size_t y = 0; y < texture_height; ++y) {
            for (std::size_t x = 0; x < texture_width; ++x) {
                image.pixels[y * texture_width + x] =
                    (((x / cell_size) + (y / cell_size)) & 1U) == 0U ? 0xFFFFFFFFU : 0xFF101010U;
            }
        }
        return image;
    }
} // namespace

int main(int argc, char* argv[]) noexcept {
    try {
        const Extents extents{texture_height, texture_width};
        const std::layout_right::mapping<Extents> row_major{extents};
        const raster::LayoutTiled4x4::mapping<Extents> tiled{extents};
        const raster::LayoutMorton::mapping<Extents> morton{extents};
        const auto sequential_order = make_quad_order(false);
        const auto randomized_order = make_quad_order(true);
        const auto first_argument = argc > 1 ? std::string_view{argv[1]} : std::string_view{};
        const auto selected_mode = first_argument
                == "scene"
                || first_argument
                == "scene-threads"
                || first_argument
                == "scene-stress"
                || first_argument
                == "heap-aging"
            ? first_argument
            : std::string_view{"fetch"};

        if (selected_mode == "heap-aging") {
            const auto parse_extent = [&](int index, std::size_t fallback) {
                if (argc <= index) {
                    return fallback;
                }

                const auto parsed = std::string_view(argv[index]);
                std::size_t value = 0U;
                for (const auto digit : parsed) {
                    if (digit < '0' || digit > '9') {
                        throw std::runtime_error("heap-aging arguments must be positive integers");
                    }

                    const auto numeric_digit = static_cast<std::size_t>(digit - '0');
                    if (value > (std::numeric_limits<std::size_t>::max() - numeric_digit) / 10U) {
                        throw std::runtime_error("heap-aging integer argument overflows size_t");
                    }
                    value = value * 10U + numeric_digit;
                }
                if (value == 0U) {
                    throw std::runtime_error("heap-aging arguments must be positive integers");
                }
                return value;
            };

            constexpr std::size_t aging_texture_width = 16U;
            constexpr std::size_t aging_texture_height = 16U;

            raster::TgaImage image{};
            image.width = aging_texture_width;
            image.height = aging_texture_height;
            image.pixels.resize(aging_texture_width * aging_texture_height);
            for (std::size_t y = 0; y < aging_texture_height; ++y) {
                for (std::size_t x = 0; x < aging_texture_width; ++x) {
                    image.pixels[y * aging_texture_width + x] =
                        ((x + y) & 1U) == 0U ? 0xFFE08030U : 0xFF3050C0U;
                }
            }

            const raster::Texture2D<raster::LayoutTiled4x4> texture{image};
            const auto width = parse_extent(2, 640U);
            const auto height = parse_extent(3, 480U);
            const auto chain_nodes = parse_extent(4, 65'536U);
            const auto churn_seed = parse_extent(5, 0xA61D1U);
            if (width > 4096U || height > 4096U || chain_nodes > 1'000'000U) {
                throw std::runtime_error(
                    "heap-aging limits: width/height <= 4096, chain_nodes <= 1000000"
                );
            }

            benchmark_heap_aging(texture, width, height, chain_nodes, churn_seed);
            return 0;
        }

        const auto layout_argument_index = selected_mode == "scene" ? 2 : 1;
        const auto order_argument_index = layout_argument_index + 1;
        const auto selected_layout = argc > layout_argument_index
            ? std::string_view{argv[layout_argument_index]}
            : std::string_view{};

        const auto selected_order = argc > order_argument_index
            ? std::string_view{argv[order_argument_index]}
            : std::string_view{};
        if (selected_mode
            == "scene"
            || selected_mode
            == "scene-threads"
            || selected_mode
            == "scene-stress") {
            auto checker_image = make_checker_4k();
            const raster::Texture2D<std::layout_right> row_major_texture{checker_image};
            const raster::Texture2D<raster::LayoutTiled4x4> tiled_texture{checker_image};
            const raster::Texture2D<raster::LayoutMorton> morton_texture{checker_image};
            if (selected_mode == "scene-stress") {
                const auto parse_extent = [&](int index, std::size_t fallback) noexcept {
                    if (argc <= index) {
                        return fallback;
                    }

                    const auto parsed = std::string_view(argv[index]);
                    std::size_t value = 0;
                    for (const auto digit : parsed) {
                        if (digit < '0' || digit > '9') {
                            return fallback;
                        }
                        value = value * 10U + static_cast<std::size_t>(digit - '0');
                    }
                    return value == 0U ? fallback : value;
                };

                benchmark_scene_stress(
                    tiled_texture,
                    parse_extent(3, 1920U),
                    parse_extent(4, 1080U),
                    parse_extent(2, 224U)
                );
                return 0;
            }

            const auto mesh_result = raster::load_obj(
                std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
            );
            if (!mesh_result) {
                throw std::runtime_error("could not load bundled clipping sphere OBJ");
            }

            constexpr std::size_t viewport_width = 640U;
            constexpr std::size_t viewport_height = 480U;
            if (selected_mode == "scene-threads") {
                benchmark_scene_threads(
                    *mesh_result,
                    tiled_texture,
                    viewport_width,
                    viewport_height
                );
                return 0;
            }

            double row_major_time = 0.0;
            double tiled_time = 0.0;
            double morton_time = 0.0;
            if (selected_layout.empty() || selected_layout == "row-major") {
                row_major_time = benchmark_scene_configuration(
                    "layout_right scene",
                    *mesh_result,
                    row_major_texture,
                    viewport_width,
                    viewport_height
                );
            }
            if (selected_layout.empty() || selected_layout == "tiled4x4") {
                tiled_time = benchmark_scene_configuration(
                    "tiled4x4 scene",
                    *mesh_result,
                    tiled_texture,
                    viewport_width,
                    viewport_height
                );
            }
            if (selected_layout.empty() || selected_layout == "morton") {
                morton_time = benchmark_scene_configuration(
                    "Morton scene",
                    *mesh_result,
                    morton_texture,
                    viewport_width,
                    viewport_height
                );
            }
            if (selected_layout.empty()) {
                std::println(
                    "scene relative to layout_right: tiled {:.2f}x, Morton {:.2f}x",
                    row_major_time / tiled_time,
                    row_major_time / morton_time
                );
            }
            return 0;
        }
        if (selected_mode != "fetch") {
            report_failure("mode must be fetch, scene, scene-threads, scene-stress, or heap-aging");
            return 2;
        }
        if ((!selected_order.empty()
             && selected_order
             != "sequential"
             && selected_order
             != "shuffled")
            || (!selected_layout.empty()
                && selected_layout
                != "row-major"
                && selected_layout
                != "tiled4x4"
                && selected_layout
                != "morton")) {
            report_failure(
                "usage: raster_texture_benchmark [scene] [row-major|tiled4x4|morton] [sequential|shuffled]"
            );
            return 2;
        }

        constexpr std::size_t texture_bytes =
            texture_width * texture_height * sizeof(std::uint32_t);
        std::println(
            "mode={}, {}x{} RGBA texture ({} MiB), {} warmups, {} measured passes; SDL_GetTicksNS median",
            selected_mode,
            texture_width,
            texture_height,
            texture_bytes / (std::size_t{1024} * std::size_t{1024}),
            warmup_passes,
            measured_passes
        );

        const bool use_sequential = selected_order.empty() || selected_order == "sequential";
        const bool use_shuffled = selected_order.empty() || selected_order == "shuffled";
        if (selected_layout.empty() || selected_layout == "row-major") {
            if (use_sequential) {
                static_cast<void>(
                    benchmark_layout("layout_right sequential", row_major, sequential_order)
                );
            }
            if (use_shuffled) {
                static_cast<void>(
                    benchmark_layout("layout_right shuffled", row_major, randomized_order)
                );
            }
        }
        if (selected_layout.empty() || selected_layout == "tiled4x4") {
            if (use_sequential) {
                static_cast<void>(benchmark_layout("tiled4x4 sequential", tiled, sequential_order));
            }
            if (use_shuffled) {
                static_cast<void>(benchmark_layout("tiled4x4 shuffled", tiled, randomized_order));
            }
        }
        if (selected_layout.empty() || selected_layout == "morton") {
            if (use_sequential) {
                static_cast<void>(benchmark_layout("morton sequential", morton, sequential_order));
            }
            if (use_shuffled) {
                static_cast<void>(benchmark_layout("morton shuffled", morton, randomized_order));
            }
        }
    } catch (const std::exception& error) {
        report_failure(error.what());
        return 1;
    } catch (...) {
        report_failure("unknown exception");
        return 1;
    }
    return 0;
}
