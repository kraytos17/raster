#pragma once

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

namespace raster {
    // Persistent tile workers with an atomic work index. worker_count is total
    // concurrency including the calling thread. The caller participates in each
    // dispatch, and the executor retains worker/synchronization storage between
    // frames.
    class TileExecutor {
    public:

        explicit TileExecutor(std::size_t concurrency) {
            if (concurrency == 0U) {
                concurrency = 1U;
            }

            workers_.reserve(concurrency - 1U);
            for (std::size_t index = 1U; index < concurrency; ++index) {
                workers_.emplace_back([this] { worker_loop(); });
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
