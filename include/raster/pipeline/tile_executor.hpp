#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace raster {
    // Upper bound on tile workers. Oversubscription beyond physical cores
    // collapses throughput on small-core-count hosts, so both the
    // default and any explicit override clamp here.
    inline constexpr std::size_t maximum_tile_concurrency = 8U;

    // Resolve worker count: an explicit nonzero request (RASTER_WORKERS) wins
    // but stays capped; otherwise hardware concurrency clamped into range.
    [[nodiscard]]
    inline std::size_t resolve_tile_workers(
        std::size_t hardware_concurrency, std::size_t requested
    ) noexcept {
        if (requested != 0U) {
            return std::min(requested, maximum_tile_concurrency);
        }
        return std::clamp(hardware_concurrency, std::size_t{1}, maximum_tile_concurrency);
    }

#if defined(__linux__)
    // Best-effort pin of the calling worker thread to one CPU, round-robin.
    inline void pin_worker_to_cpu(std::size_t worker_index) noexcept {
        const auto cpu_count = std::thread::hardware_concurrency();
        if (cpu_count == 0U) {
            return;
        }

        cpu_set_t cpus{};
        CPU_ZERO(&cpus);
        CPU_SET(worker_index % cpu_count, &cpus);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    }
#endif

    // Static contiguous partition of [0, item_count) across worker_count
    // workers: worker w gets [begin, end). Extra workers get empty ranges;
    // concatenating every worker's range in order reproduces the input order,
    // which keeps deterministic per-item output stable under threading.
    [[nodiscard]]
    inline std::pair<std::size_t, std::size_t> partition_range(
        std::size_t item_count, std::size_t worker_count, std::size_t worker_index
    ) noexcept {
        if (worker_count == 0U || worker_index >= worker_count) {
            return {0U, 0U};
        }

        const auto chunk = item_count / worker_count;
        const auto remainder = item_count % worker_count;
        const auto begin = worker_index * chunk + std::min(worker_index, remainder);
        return {begin, begin + chunk + static_cast<std::size_t>(worker_index < remainder)};
    }

    // Persistent tile workers with an atomic work index. worker_count is total
    // concurrency including the calling thread. The caller participates in each
    // dispatch, and the executor retains worker/synchronization storage between
    // frames.
    class TileExecutor {
    public:

        [[nodiscard]]
        std::size_t concurrency() const noexcept {
            return workers_.size() + 1U;
        }

        explicit TileExecutor(std::size_t concurrency) {
            if (concurrency == 0U) {
                concurrency = 1U;
            }

            workers_.reserve(concurrency - 1U);
            for (std::size_t index = 1U; index < concurrency; ++index) {
                workers_.emplace_back([this, index] {
#if defined(__linux__)
                    pin_worker_to_cpu(index);
#endif
                    worker_loop();
                });
            }
        }

        TileExecutor(const TileExecutor&) = delete;
        TileExecutor& operator=(const TileExecutor&) = delete;
        TileExecutor(TileExecutor&&) = delete;
        TileExecutor& operator=(TileExecutor&&) = delete;

        ~TileExecutor() {
            {
                std::lock_guard state_lock{state_mutex_};
                stopping_ = true;
            }

            work_available_.notify_all();
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        template <typename Function>
        void execute(std::size_t tile_count, Function&& function) noexcept {
            auto&& callable = std::forward<Function>(function);
            using FunctionType = std::remove_reference_t<decltype(callable)>;
            static_assert(std::is_nothrow_invocable_v<FunctionType&, std::size_t>);
            execute_erased(tile_count, std::addressof(callable), &invoke_function<FunctionType>);
        }

    private:

        using TileFunction = void (*)(void*, std::size_t) noexcept;

        template <typename FunctionType>
        static void invoke_function(void* context, std::size_t tile) noexcept {
            (*static_cast<FunctionType*>(context))(tile);
        }

        void execute_erased(std::size_t tile_count, void* context, TileFunction function) noexcept {
            assert(function != nullptr);
            std::scoped_lock execute_lock{execute_mutex_};
            if (tile_count == 0U) {
                return;
            }

            {
                std::lock_guard state_lock{state_mutex_};
                tile_count_ = tile_count;
                context_ = context;
                function_ = function;
                next_tile_.store(0U, std::memory_order_relaxed);
                remaining_workers_ = workers_.size();
                ++generation_;
            }

            work_available_.notify_all();
            run_tiles();
            std::unique_lock state_lock{state_mutex_};
            work_complete_.wait(state_lock, [this] { return remaining_workers_ == 0U; });
            context_ = nullptr;
            function_ = nullptr;
        }

        void run_tiles() noexcept {
            for (;;) {
                const auto tile = next_tile_.fetch_add(1U, std::memory_order_relaxed);
                if (tile >= tile_count_) {
                    return;
                }
                function_(context_, tile);
            }
        }

        void worker_loop() noexcept {
            std::size_t observed_generation = 0U;
            for (;;) {
                std::unique_lock state_lock{state_mutex_};
                work_available_.wait(state_lock, [this, &observed_generation] {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) {
                    return;
                }

                observed_generation = generation_;
                state_lock.unlock();
                run_tiles();
                state_lock.lock();
                --remaining_workers_;
                if (remaining_workers_ == 0U) {
                    work_complete_.notify_one();
                }
            }
        }

        std::vector<std::jthread> workers_{};
        std::mutex execute_mutex_{};
        std::mutex state_mutex_{};
        std::condition_variable_any work_available_{};
        std::condition_variable work_complete_{};
        std::atomic_size_t next_tile_{};
        std::size_t tile_count_{};
        std::size_t remaining_workers_{};
        std::size_t generation_{};
        bool stopping_{};
        void* context_{};
        TileFunction function_{};
    };
} // namespace raster
