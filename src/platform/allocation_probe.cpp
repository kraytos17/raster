#include "raster/platform/allocation_probe.hpp"

#include <algorithm>
#include <cstdlib>
#include <new>

namespace {
    thread_local bool allocation_probe_active{};
    thread_local std::size_t allocation_probe_count{};

    [[nodiscard]]
    void* allocate(std::size_t size) {
        if (allocation_probe_active) {
            ++allocation_probe_count;
        }
        if (void* const memory = std::malloc(std::max(size, std::size_t{1}))) {
            return memory;
        }
        throw std::bad_alloc{};
    }

    [[nodiscard]]
    void* allocate_aligned(std::size_t size, std::align_val_t alignment) {
        if (allocation_probe_active) {
            ++allocation_probe_count;
        }
        const auto alignment_size = static_cast<std::size_t>(alignment);
        const auto requested_size = std::max(size, std::size_t{1});
        const auto rounded_size =
            ((requested_size + alignment_size - 1U) / alignment_size) * alignment_size;
        if (void* const memory = std::aligned_alloc(alignment_size, rounded_size)) {
            return memory;
        }
        throw std::bad_alloc{};
    }
} // namespace

void* operator new(std::size_t size) {
    return allocate(size);
}

void* operator new[](std::size_t size) {
    return allocate(size);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocate_aligned(size, alignment);
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return allocate_aligned(size, alignment);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(size);
    } catch (...) {
        return nullptr;
    }
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](size);
    } catch (...) {
        return nullptr;
    }
}

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(size, alignment);
    } catch (...) {
        return nullptr;
    }
}

void* operator new[](
    std::size_t size, std::align_val_t alignment, const std::nothrow_t& tag
) noexcept {
    try {
        return ::operator new[](size, alignment);
    } catch (...) {
        static_cast<void>(tag);
        return nullptr;
    }
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete[](void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::align_val_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::align_val_t) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t, std::align_val_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept {
    std::free(memory);
}

void operator delete(void* memory, const std::nothrow_t&) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, const std::nothrow_t&) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::align_val_t, const std::nothrow_t&) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::align_val_t, const std::nothrow_t&) noexcept {
    std::free(memory);
}

namespace raster::platform {
    ScopedAllocationProbe::ScopedAllocationProbe(bool enabled) noexcept :
        enabled_{enabled} {
        allocation_probe_count = 0;
        allocation_probe_active = enabled_;
    }

    ScopedAllocationProbe::~ScopedAllocationProbe() {
        allocation_probe_active = false;
    }

    std::size_t ScopedAllocationProbe::allocations() const noexcept {
        return allocation_probe_count;
    }
} // namespace raster::platform
