#include "mirage/cpu/cpu_features.hpp"

#include <cstdint>
#include <sstream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MIRAGE_ARCH_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define MIRAGE_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
#define MIRAGE_ARCH_ARM32 1
#elif defined(__riscv)
#define MIRAGE_ARCH_RISCV 1
#elif defined(__powerpc__) || defined(__ppc__)
#define MIRAGE_ARCH_PPC 1
#endif

namespace mirage::cpu {

namespace {

#if defined(MIRAGE_ARCH_X86)

void cpuid_native(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]) noexcept {
#if defined(_MSC_VER)
    int cpu_info[4];
    __cpuidex(cpu_info, static_cast<int>(leaf), static_cast<int>(subleaf));
    regs[0] = static_cast<uint32_t>(cpu_info[0]);
    regs[1] = static_cast<uint32_t>(cpu_info[1]);
    regs[2] = static_cast<uint32_t>(cpu_info[2]);
    regs[3] = static_cast<uint32_t>(cpu_info[3]);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

uint64_t xgetbv_safe(uint32_t xcr) noexcept {
#if defined(_MSC_VER)
    return _xgetbv(xcr);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax = 0;
    uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#else
    (void)xcr;
    return 0;
#endif
}

#endif  // MIRAGE_ARCH_X86

CpuFeatures detect_internal() noexcept {
    CpuFeatures features{};

#if defined(MIRAGE_ARCH_X86)
#if defined(__x86_64__) || defined(_M_X64)
    features.architecture = Architecture::X86_64;
#else
    features.architecture = Architecture::X86_32;
#endif

    uint32_t regs[4] = {0, 0, 0, 0};

    // Query basic CPUID information and maximum supported leaf
    cpuid_native(0, 0, regs);
    const uint32_t max_basic_leaf = regs[0];

    if (max_basic_leaf >= 1) {
        cpuid_native(1, 0, regs);
        const uint32_t ecx = regs[2];

        // Check SSE4.2 (bit 20)
        features.has_sse4_2 = (ecx & (1u << 20)) != 0;

        // Check OSXSAVE (bit 27) and AVX (bit 28)
        const bool has_osxsave = (ecx & (1u << 27)) != 0;
        const bool cpu_has_avx = (ecx & (1u << 28)) != 0;

        if (has_osxsave && cpu_has_avx) {
            // Verify OS has enabled YMM register state management (XMM: bit 1, YMM: bit 2)
            const uint64_t xcr0 = xgetbv_safe(0);
            if ((xcr0 & 0x6ULL) == 0x6ULL) {
                features.has_avx = true;
                features.has_fma = (ecx & (1u << 12)) != 0;
                features.has_f16c = (ecx & (1u << 29)) != 0;

                // Query extended features for AVX2 (Leaf 7, Subleaf 0)
                if (max_basic_leaf >= 7) {
                    cpuid_native(7, 0, regs);
                    const uint32_t ebx = regs[1];
                    // AVX2 is bit 5 of EBX
                    features.has_avx2 = (ebx & (1u << 5)) != 0;
                }
            }
        }
    }

#elif defined(MIRAGE_ARCH_ARM64)
    features.architecture = Architecture::ARM64;
    features.has_neon = true;  // NEON is baseline in ARM64
#elif defined(MIRAGE_ARCH_ARM32)
    features.architecture = Architecture::ARM32;
#elif defined(MIRAGE_ARCH_RISCV)
    features.architecture = Architecture::RISCV;
#elif defined(MIRAGE_ARCH_PPC)
    features.architecture = Architecture::PowerPC;
#else
    features.architecture = Architecture::Unknown;
#endif

    return features;
}

}  // namespace

const CpuFeatures& CpuFeatures::detect() noexcept {
    static const CpuFeatures instance = detect_internal();
    return instance;
}

std::string_view CpuFeatures::architecture_name() const noexcept {
    switch (architecture) {
        case Architecture::X86_64:
            return "x86_64";
        case Architecture::X86_32:
            return "x86_32";
        case Architecture::ARM64:
            return "ARM64";
        case Architecture::ARM32:
            return "ARM32";
        case Architecture::RISCV:
            return "RISC-V";
        case Architecture::PowerPC:
            return "PowerPC";
        case Architecture::Unknown:
        default:
            return "Unknown";
    }
}

std::string CpuFeatures::to_string() const {
    std::ostringstream oss;
    oss << "Architecture: " << architecture_name() << "\n"
        << "AVX2:         " << (has_avx2 ? "Supported" : "Unsupported") << "\n"
        << "AVX:          " << (has_avx ? "Supported" : "Unsupported") << "\n"
        << "FMA:          " << (has_fma ? "Supported" : "Unsupported") << "\n"
        << "F16C:         " << (has_f16c ? "Supported" : "Unsupported") << "\n"
        << "SSE4.2:       " << (has_sse4_2 ? "Supported" : "Unsupported") << "\n"
        << "NEON:         " << (has_neon ? "Supported" : "Unsupported");
    return oss.str();
}

bool is_avx2_supported() noexcept {
    return CpuFeatures::detect().has_avx2;
}

std::string_view get_host_architecture_name() noexcept {
    return CpuFeatures::detect().architecture_name();
}

}  // namespace mirage::cpu

