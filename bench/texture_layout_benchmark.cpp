#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <mdspan>
#include <print>
#include <ranges>
#include <SDL3/SDL.h>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "raster/io/obj_loader.hpp"
#include "raster/io/tga_loader.hpp"
#include "raster/memory/framebuffer.hpp"
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
    std::uint64_t benchmark_frame_times(Render& render, const char* failure_message) {
        constexpr std::size_t thread_warmups = 2U;
        constexpr std::size_t thread_measured_passes = 7U;
        for ([[maybe_unused]]
             const auto pass : std::views::iota(std::size_t{0}, thread_warmups)) {
            if (!render()) {
                throw std::runtime_error(failure_message);
            }
        }

        std::array<std::uint64_t, thread_measured_passes> times{};
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

        const auto serial_median = benchmark_frame_times(render_serial, "serial render failed");
        std::println(
            "serial tiled-layout baseline: {} triangles at {}x{}, median {:.3f} ms/frame (7 runs)",
            mesh.triangle_count(),
            width,
            height,
            static_cast<double>(serial_median) / 1.0e6
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
                benchmark_frame_times(render_threaded, "threaded render failed");
            std::println(
                "tile concurrency {}: median {:.3f} ms/frame, {:.2f}x serial (7 runs)",
                concurrency,
                static_cast<double>(threaded_median) / 1.0e6,
                static_cast<double>(serial_median) / static_cast<double>(threaded_median)
            );
        }
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
        const auto selected_mode = first_argument == "scene" || first_argument == "scene-threads"
            ? first_argument
            : std::string_view{"fetch"};

        const auto layout_argument_index = selected_mode == "scene" ? 2 : 1;
        const auto order_argument_index = layout_argument_index + 1;
        const auto selected_layout = argc > layout_argument_index
            ? std::string_view{argv[layout_argument_index]}
            : std::string_view{};

        const auto selected_order = argc > order_argument_index
            ? std::string_view{argv[order_argument_index]}
            : std::string_view{};
        if (selected_mode == "scene" || selected_mode == "scene-threads") {
            const auto mesh_result = raster::load_obj(
                std::filesystem::path{RASTER_ASSET_DIR} / "models/clipping_sphere.obj"
            );
            if (!mesh_result) {
                throw std::runtime_error("could not load bundled clipping sphere OBJ");
            }

            auto checker_image = make_checker_4k();
            const raster::Texture2D<std::layout_right> row_major_texture{checker_image};
            const raster::Texture2D<raster::LayoutTiled4x4> tiled_texture{checker_image};
            const raster::Texture2D<raster::LayoutMorton> morton_texture{checker_image};
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
            report_failure("mode must be fetch, scene, or scene-threads");
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
