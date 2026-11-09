#pragma once

#include <string>
#include <string_view>

namespace mirage::cpu {

/**
 * @brief Enumeration of supported host processor architectures.
 */
enum class Architecture {
    Unknown,
    X86_32,
    X86_64,
    ARM32,
    ARM64,
    RISCV,
    PowerPC
};

/**
 * @brief Comprehensive CPU capability descriptors for runtime dispatch.
 */
struct CpuFeatures {
    Architecture architecture = Architecture::Unknown;
    bool has_sse4_2 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_fma = false;
    bool has_f16c = false;
    bool has_neon = false;

    /**
     * @brief Detects the capabilities of the host processor at runtime.
     * Guaranteed to be thread-safe and evaluated once. Never executes SIMD instructions directly.
     */
    [[nodiscard]] static const CpuFeatures& detect() noexcept;

    /**
     * @brief Returns a human-readable name of the detected CPU architecture.
     */
    [[nodiscard]] std::string_view architecture_name() const noexcept;

    /**
     * @brief Returns human-readable status for AVX2 availability.
     */
    [[nodiscard]] std::string_view avx2_status_string() const noexcept {
        return has_avx2 ? "Supported" : "Unsupported";
    }

    /**
     * @brief Formats detected features into a diagnostic string.
     */
    [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Convenience function to check whether AVX2 is available on the host.
 */
[[nodiscard]] bool is_avx2_supported() noexcept;

/**
 * @brief Convenience function to query the host architecture name.
 */
[[nodiscard]] std::string_view get_host_architecture_name() noexcept;

}  // namespace mirage::cpu

