#include "mirage/simd/simd_backend.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "mirage/cpu/cpu_features.hpp"

namespace mirage::simd {

ExecutionBackend parse_execution_backend(std::string_view str) {
    std::string lower;
    lower.reserve(str.size());
    for (char ch : str) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    if (lower == "auto") {
        return ExecutionBackend::Auto;
    }
    if (lower == "scalar") {
        return ExecutionBackend::Scalar;
    }
    if (lower == "avx2") {
        return ExecutionBackend::AVX2;
    }

    throw std::invalid_argument("Unknown execution backend: '" + std::string(str) +
                                "'. Supported backends: 'auto', 'scalar', 'avx2'.");
}

bool is_backend_supported(ExecutionBackend backend, const cpu::CpuFeatures& features) noexcept {
    switch (backend) {
        case ExecutionBackend::Auto:
        case ExecutionBackend::Scalar:
            return true;
        case ExecutionBackend::AVX2:
            // AVX2 kernels utilize 256-bit AVX2 vectors and FMA fused multiply-add operations.
            // Both hardware features must be present and enabled by the OS.
            return features.has_avx2 && features.has_fma;
    }
    return false;
}

bool is_backend_supported(ExecutionBackend backend) noexcept {
    return is_backend_supported(backend, cpu::CpuFeatures::detect());
}

ExecutionBackend resolve_backend(ExecutionBackend backend, const cpu::CpuFeatures& features) noexcept {
    if (backend == ExecutionBackend::Auto) {
        return is_backend_supported(ExecutionBackend::AVX2, features) ? ExecutionBackend::AVX2 : ExecutionBackend::Scalar;
    }
    return backend;
}

ExecutionBackend resolve_backend(ExecutionBackend backend) noexcept {
    return resolve_backend(backend, cpu::CpuFeatures::detect());
}

}  // namespace mirage::simd

