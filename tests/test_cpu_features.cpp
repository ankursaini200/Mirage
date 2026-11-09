#include <gtest/gtest.h>

#include "mirage/cpu/cpu_features.hpp"

namespace mirage::tests {

TEST(CpuFeaturesTest, SingletonConsistency) {
    const auto& features1 = cpu::CpuFeatures::detect();
    const auto& features2 = cpu::CpuFeatures::detect();
    EXPECT_EQ(&features1, &features2);
}

TEST(CpuFeaturesTest, ArchitectureIdentification) {
    const auto& features = cpu::CpuFeatures::detect();
    const std::string_view arch_name = features.architecture_name();

    EXPECT_FALSE(arch_name.empty());
    EXPECT_NE(arch_name, "Unknown");
}

TEST(CpuFeaturesTest, Avx2StatusConsistency) {
    const auto& features = cpu::CpuFeatures::detect();
    const std::string_view status = features.avx2_status_string();

    if (features.has_avx2) {
        EXPECT_EQ(status, "Supported");
    } else {
        EXPECT_EQ(status, "Unsupported");
    }

    EXPECT_EQ(features.has_avx2, cpu::is_avx2_supported());
}

TEST(CpuFeaturesTest, DiagnosticStringFormatting) {
    const auto& features = cpu::CpuFeatures::detect();
    const std::string diagnostic = features.to_string();

    EXPECT_NE(diagnostic.find("Architecture:"), std::string::npos);
    EXPECT_NE(diagnostic.find("AVX2:"), std::string::npos);
    EXPECT_NE(diagnostic.find("AVX:"), std::string::npos);
    EXPECT_NE(diagnostic.find("FMA:"), std::string::npos);
    EXPECT_NE(diagnostic.find("F16C:"), std::string::npos);
    EXPECT_NE(diagnostic.find("SSE4.2:"), std::string::npos);
}

TEST(CpuFeaturesTest, ConvenienceFunctions) {
    EXPECT_EQ(cpu::get_host_architecture_name(), cpu::CpuFeatures::detect().architecture_name());
    EXPECT_EQ(cpu::is_avx2_supported(), cpu::CpuFeatures::detect().has_avx2);
}

}  // namespace mirage::tests

