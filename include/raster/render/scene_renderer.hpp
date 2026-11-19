#pragma once

#include <cstdint>

#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"

namespace raster::render {
    // Render the current procedural checker-floor scene or the deterministic
    // camera sweep used by the clipping stress gate. Returns false if clipping
    // rejects malformed geometry.
    [[nodiscard]]
    bool render_scene(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        bool clip_stress_test
    ) noexcept;
} // namespace raster::render
