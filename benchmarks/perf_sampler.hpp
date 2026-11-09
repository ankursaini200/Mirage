#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace mirage::benchmark {

/**
 * @brief Hardware performance counter measurement results.
 */
struct PerfCounterStats {
    bool available = false;
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t cache_references = 0;
    uint64_t cache_misses = 0;
    uint64_t branch_instructions = 0;
    uint64_t branch_misses = 0;
    uint64_t task_clock_ns = 0;

    [[nodiscard]] double ipc() const noexcept {
        return (available && cycles > 0)
                   ? (static_cast<double>(instructions) / static_cast<double>(cycles))
                   : 0.0;
    }

    [[nodiscard]] double cache_miss_rate() const noexcept {
        return (available && cache_references > 0)
                   ? (static_cast<double>(cache_misses) / static_cast<double>(cache_references))
                   : 0.0;
    }

    [[nodiscard]] double branch_miss_rate() const noexcept {
        return (available && branch_instructions > 0)
                   ? (static_cast<double>(branch_misses) / static_cast<double>(branch_instructions))
                   : 0.0;
    }
};

/**
 * @brief RAII Hardware performance counter sampler using Linux perf_event_open.
 */
class HardwarePerfSampler {
public:
    HardwarePerfSampler() {
#if defined(__linux__) && defined(__NR_perf_event_open)
        fd_cycles_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        fd_instructions_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        fd_cache_ref_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
        fd_cache_miss_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
        fd_branches_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
        fd_branch_miss_ = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
        fd_task_clock_ = open_counter(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK);

        available_ = (fd_cycles_ >= 0 && fd_instructions_ >= 0);
#endif
    }

    ~HardwarePerfSampler() {
#if defined(__linux__) && defined(__NR_perf_event_open)
        close_fd(fd_cycles_);
        close_fd(fd_instructions_);
        close_fd(fd_cache_ref_);
        close_fd(fd_cache_miss_);
        close_fd(fd_branches_);
        close_fd(fd_branch_miss_);
        close_fd(fd_task_clock_);
#endif
    }

    HardwarePerfSampler(const HardwarePerfSampler&) = delete;
    HardwarePerfSampler& operator=(const HardwarePerfSampler&) = delete;
    HardwarePerfSampler(HardwarePerfSampler&&) = delete;
    HardwarePerfSampler& operator=(HardwarePerfSampler&&) = delete;

    [[nodiscard]] bool is_available() const noexcept {
        return available_;
    }

    void start() noexcept {
#if defined(__linux__) && defined(__NR_perf_event_open)
        if (!available_) return;
        reset_and_enable(fd_cycles_);
        reset_and_enable(fd_instructions_);
        reset_and_enable(fd_cache_ref_);
        reset_and_enable(fd_cache_miss_);
        reset_and_enable(fd_branches_);
        reset_and_enable(fd_branch_miss_);
        reset_and_enable(fd_task_clock_);
#endif
    }

    [[nodiscard]] PerfCounterStats stop() noexcept {
        PerfCounterStats s{};
#if defined(__linux__) && defined(__NR_perf_event_open)
        if (!available_) return s;
        disable_fd(fd_cycles_);
        disable_fd(fd_instructions_);
        disable_fd(fd_cache_ref_);
        disable_fd(fd_cache_miss_);
        disable_fd(fd_branches_);
        disable_fd(fd_branch_miss_);
        disable_fd(fd_task_clock_);

        s.available = true;
        s.cycles = read_counter(fd_cycles_);
        s.instructions = read_counter(fd_instructions_);
        s.cache_references = read_counter(fd_cache_ref_);
        s.cache_misses = read_counter(fd_cache_miss_);
        s.branch_instructions = read_counter(fd_branches_);
        s.branch_misses = read_counter(fd_branch_miss_);
        s.task_clock_ns = read_counter(fd_task_clock_);
#endif
        return s;
    }

private:
#if defined(__linux__) && defined(__NR_perf_event_open)
    static int open_counter(uint32_t type, uint64_t config) noexcept {
        struct perf_event_attr pe;
        std::memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = type;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        const long fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        return static_cast<int>(fd);
    }

    static void reset_and_enable(int fd) noexcept {
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }

    static void disable_fd(int fd) noexcept {
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        }
    }

    static uint64_t read_counter(int fd) noexcept {
        if (fd < 0) return 0;
        uint64_t val = 0;
        if (read(fd, &val, sizeof(val)) == sizeof(val)) {
            return val;
        }
        return 0;
    }

    static void close_fd(int& fd) noexcept {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    bool available_ = false;
    int fd_cycles_ = -1;
    int fd_instructions_ = -1;
    int fd_cache_ref_ = -1;
    int fd_cache_miss_ = -1;
    int fd_branches_ = -1;
    int fd_branch_miss_ = -1;
    int fd_task_clock_ = -1;
#else
    bool available_ = false;
#endif
};

}  // namespace mirage::benchmark
