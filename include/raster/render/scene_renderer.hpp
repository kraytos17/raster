#pragma once

#include <cstdint>

#include "raster/math/vec.hpp"
#include "raster/memory/framebuffer.hpp"
#include "raster/memory/soa_mesh.hpp"
#include "raster/memory/texture.hpp"
#include "raster/memory/texture_layout.hpp"

namespace raster::render {
    struct FreeLookCamera {
        Vec3f position{};
        float yaw{};
        float pitch{};
    };

    // Render the current procedural checker-floor scene or the deterministic
    // camera sweep used by the clipping stress gate. Returns false if clipping
    // rejects malformed geometry.
    [[nodiscard]]
    bool render_scene(
        Framebuffer& framebuffer,
        const SoAMesh& mesh,
        std::uint64_t elapsed_ns,
        bool clip_stress_test,
        bool free_look_test,
        bool guard_band_test,
        const Texture2D<LayoutTiled4x4>& texture,
        const FreeLookCamera& camera
    ) noexcept;
} // namespace raster::render
