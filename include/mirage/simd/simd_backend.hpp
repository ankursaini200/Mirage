#pragma once

#include <string>
#include <string_view>

namespace mirage::cpu {
struct CpuFeatures;
}

namespace mirage::simd {

/**
 * @brief Computation backend for image enhancement operations.
 */
enum class ExecutionBackend {
    Auto,    ///< Selects the highest performing verified implementation supported by the host CPU.
    Scalar,  ///< Forces portable scalar reference implementation.
    AVX2     ///< Requests AVX2 SIMD acceleration; throws or reports an error if unsupported.
};

/**
 * @brief Returns the human-readable string representation of an execution backend.
 */
[[nodiscard]] constexpr std::string_view to_string(ExecutionBackend backend) noexcept {
    switch (backend) {
        case ExecutionBackend::Auto:
            return "auto";
        case ExecutionBackend::Scalar:
            return "scalar";
        case ExecutionBackend::AVX2:
            return "avx2";
    }
    return "unknown";
}

/**
 * @brief Parses an execution backend string ("auto", "scalar", "avx2").
 * @throws std::invalid_argument if the string is unrecognized.
 */
[[nodiscard]] ExecutionBackend parse_execution_backend(std::string_view str);

/**
 * @brief Checks whether the specified backend can execute given explicit CPU features.
 */
[[nodiscard]] bool is_backend_supported(ExecutionBackend backend, const cpu::CpuFeatures& features) noexcept;

/**
 * @brief Checks whether the specified backend can execute on the current host CPU.
 */
[[nodiscard]] bool is_backend_supported(ExecutionBackend backend) noexcept;

/**
 * @brief Resolves ExecutionBackend::Auto to the optimal available backend given explicit CPU features.
 */
[[nodiscard]] ExecutionBackend resolve_backend(ExecutionBackend backend, const cpu::CpuFeatures& features) noexcept;

/**
 * @brief Resolves ExecutionBackend::Auto to the optimal available backend (AVX2 if available, else Scalar).
 */
[[nodiscard]] ExecutionBackend resolve_backend(ExecutionBackend backend) noexcept;

}  // namespace mirage::simd

