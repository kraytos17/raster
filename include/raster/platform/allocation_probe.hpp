#pragma once

#include <cstddef>

namespace raster::platform {
    // Counts C++ global allocation calls from every thread while the probe is
    // active. Intended for diagnostics/tests around parallel render work.
    class ScopedAllocationProbe {
    public:

        explicit ScopedAllocationProbe(bool enabled) noexcept;
        ScopedAllocationProbe(const ScopedAllocationProbe&) = delete;
        ScopedAllocationProbe& operator=(const ScopedAllocationProbe&) = delete;
        ScopedAllocationProbe(ScopedAllocationProbe&&) = delete;
        ScopedAllocationProbe& operator=(ScopedAllocationProbe&&) = delete;
        ~ScopedAllocationProbe();

        [[nodiscard]]
        std::size_t allocations() const noexcept;

    private:

        bool enabled_{};
    };
} // namespace raster::platform
