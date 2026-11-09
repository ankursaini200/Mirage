#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <opencv2/core.hpp>

#include "mirage/cpu/cpu_features.hpp"
#include "mirage/core/half.hpp"
#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/image/tile_manager.hpp"
#include "mirage/precision/adaptive_precision.hpp"
#include "mirage/simd/avx2_kernels.hpp"
#include "mirage/simd/simd_backend.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__F16C__)
#include <immintrin.h>
#endif
#endif

using namespace mirage;
using namespace mirage::simd;
using namespace mirage::enhancement;

namespace {

cv::Mat create_random_image(int width, int height, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    cv::Mat img(height, width, CV_32FC3);
    for (int r = 0; r < height; ++r) {
        auto* ptr = img.ptr<cv::Vec3f>(r);
        for (int c = 0; c < width; ++c) {
            ptr[c] = cv::Vec3f(dis(gen), dis(gen), dis(gen));
        }
    }
    return img;
}

double max_absolute_diff(const cv::Mat& a, const cv::Mat& b) {
    EXPECT_EQ(a.size(), b.size());
    EXPECT_EQ(a.type(), b.type());
    double max_diff = 0.0;
    for (int r = 0; r < a.rows; ++r) {
        const auto* ptr_a = a.ptr<cv::Vec3f>(r);
        const auto* ptr_b = b.ptr<cv::Vec3f>(r);
        for (int c = 0; c < a.cols; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                const double diff = std::abs(static_cast<double>(ptr_a[c][ch]) - static_cast<double>(ptr_b[c][ch]));
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
    }
    return max_diff;
}

}  // namespace

// ============================================================================
// SIMD Backend Configuration Tests
// ============================================================================

TEST(SimdBackendTest, ParseBackendStrings) {
    EXPECT_EQ(parse_execution_backend("auto"), ExecutionBackend::Auto);
    EXPECT_EQ(parse_execution_backend("Auto"), ExecutionBackend::Auto);
    EXPECT_EQ(parse_execution_backend("AUTO"), ExecutionBackend::Auto);

    EXPECT_EQ(parse_execution_backend("scalar"), ExecutionBackend::Scalar);
    EXPECT_EQ(parse_execution_backend("Scalar"), ExecutionBackend::Scalar);
    EXPECT_EQ(parse_execution_backend("SCALAR"), ExecutionBackend::Scalar);

    EXPECT_EQ(parse_execution_backend("avx2"), ExecutionBackend::AVX2);
    EXPECT_EQ(parse_execution_backend("Avx2"), ExecutionBackend::AVX2);
    EXPECT_EQ(parse_execution_backend("AVX2"), ExecutionBackend::AVX2);

    EXPECT_THROW((void)parse_execution_backend("neon"), std::invalid_argument);
    EXPECT_THROW((void)parse_execution_backend("cuda"), std::invalid_argument);
    EXPECT_THROW((void)parse_execution_backend(""), std::invalid_argument);
}

TEST(SimdBackendTest, BackendAvailabilityAndResolution) {
    EXPECT_TRUE(is_backend_supported(ExecutionBackend::Auto));
    EXPECT_TRUE(is_backend_supported(ExecutionBackend::Scalar));

    const auto& cpu = cpu::CpuFeatures::detect();
    const bool expected_avx2 = cpu.has_avx2 && cpu.has_fma;
    EXPECT_EQ(is_backend_supported(ExecutionBackend::AVX2), expected_avx2);

    const auto resolved = resolve_backend(ExecutionBackend::Auto);
    if (expected_avx2) {
        EXPECT_EQ(resolved, ExecutionBackend::AVX2);
    } else {
        EXPECT_EQ(resolved, ExecutionBackend::Scalar);
    }

    EXPECT_EQ(resolve_backend(ExecutionBackend::Scalar), ExecutionBackend::Scalar);
    EXPECT_EQ(resolve_backend(ExecutionBackend::AVX2), ExecutionBackend::AVX2);
}

TEST(SimdBackendTest, FeatureDependentBackendSafety) {
    // 1. Both AVX2 and FMA present -> AVX2 supported, Auto resolves to AVX2
    cpu::CpuFeatures feat_both{};
    feat_both.has_avx2 = true;
    feat_both.has_fma = true;
    EXPECT_TRUE(is_backend_supported(ExecutionBackend::AVX2, feat_both));
    EXPECT_EQ(resolve_backend(ExecutionBackend::Auto, feat_both), ExecutionBackend::AVX2);

    // 2. AVX2 present but FMA missing -> AVX2 NOT supported, Auto resolves to Scalar
    cpu::CpuFeatures feat_no_fma{};
    feat_no_fma.has_avx2 = true;
    feat_no_fma.has_fma = false;
    EXPECT_FALSE(is_backend_supported(ExecutionBackend::AVX2, feat_no_fma));
    EXPECT_EQ(resolve_backend(ExecutionBackend::Auto, feat_no_fma), ExecutionBackend::Scalar);

    // 3. FMA present but AVX2 missing -> AVX2 NOT supported, Auto resolves to Scalar
    cpu::CpuFeatures feat_no_avx2{};
    feat_no_avx2.has_avx2 = false;
    feat_no_avx2.has_fma = true;
    EXPECT_FALSE(is_backend_supported(ExecutionBackend::AVX2, feat_no_avx2));
    EXPECT_EQ(resolve_backend(ExecutionBackend::Auto, feat_no_avx2), ExecutionBackend::Scalar);

    // 4. Neither present -> AVX2 NOT supported, Auto resolves to Scalar
    cpu::CpuFeatures feat_neither{};
    feat_neither.has_avx2 = false;
    feat_neither.has_fma = false;
    EXPECT_FALSE(is_backend_supported(ExecutionBackend::AVX2, feat_neither));
    EXPECT_EQ(resolve_backend(ExecutionBackend::Auto, feat_neither), ExecutionBackend::Scalar);
    EXPECT_TRUE(is_backend_supported(ExecutionBackend::Scalar, feat_neither));
    EXPECT_TRUE(is_backend_supported(ExecutionBackend::Auto, feat_neither));
}

// ============================================================================
// AVX2 Kernel Numerical Equivalence Tests
// ============================================================================

TEST(Avx2KernelsTest, BrightnessScalarEquivalence) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const cv::Mat src = create_random_image(128, 96, 101);
    const float deltas[] = {-0.75f, -0.3f, 0.0f, 0.15f, 0.85f};

    for (float delta : deltas) {
        cv::Mat dst_scalar;
        cv::Mat dst_avx2;

        EnhancementEngine::adjust_brightness_scalar(src, dst_scalar, delta);
        avx2::adjust_brightness_avx2(src, dst_avx2, delta);

        const double diff = max_absolute_diff(dst_scalar, dst_avx2);
        EXPECT_DOUBLE_EQ(diff, 0.0) << "Brightness difference detected for delta=" << delta;
    }
}

TEST(Avx2KernelsTest, ContrastScalarEquivalence) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const cv::Mat src = create_random_image(128, 96, 202);
    const float factors[] = {0.0f, 0.5f, 1.0f, 1.75f, 3.0f};
    const float midpoints[] = {0.2f, 0.5f, 0.8f};

    for (float factor : factors) {
        for (float mid : midpoints) {
            cv::Mat dst_scalar;
            cv::Mat dst_avx2;

            EnhancementEngine::adjust_contrast_scalar(src, dst_scalar, factor, mid);
            avx2::adjust_contrast_avx2(src, dst_avx2, factor, mid);

            const double diff = max_absolute_diff(dst_scalar, dst_avx2);
            EXPECT_LE(diff, 1e-7) << "Contrast diff for factor=" << factor << " mid=" << mid;
        }
    }
}

TEST(Avx2KernelsTest, SaturationScalarEquivalence) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const cv::Mat src = create_random_image(128, 96, 303);
    const float factors[] = {0.0f, 0.25f, 1.0f, 1.6f, 2.5f};

    for (float factor : factors) {
        cv::Mat dst_scalar;
        cv::Mat dst_avx2;

        EnhancementEngine::adjust_saturation_scalar(src, dst_scalar, factor);
        avx2::adjust_saturation_avx2(src, dst_avx2, factor);

        const double diff = max_absolute_diff(dst_scalar, dst_avx2);
        EXPECT_LE(diff, 1e-6) << "Saturation diff for factor=" << factor;
    }
}

TEST(Avx2KernelsTest, CombinedBrightnessContrastEquivalence) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const cv::Mat src = create_random_image(128, 96, 404);
    const float delta = 0.12f;
    const float factor = 1.45f;
    const float midpoint = 0.5f;

    // Sequential scalar: Brightness then Contrast
    cv::Mat tmp_scalar;
    cv::Mat dst_scalar;
    EnhancementEngine::adjust_brightness_scalar(src, tmp_scalar, delta);
    EnhancementEngine::adjust_contrast_scalar(tmp_scalar, dst_scalar, factor, midpoint);

    // Fused AVX2 combined kernel
    cv::Mat dst_avx2;
    avx2::adjust_brightness_contrast_avx2(src, dst_avx2, delta, factor, midpoint);

    const double diff = max_absolute_diff(dst_scalar, dst_avx2);
    EXPECT_LE(diff, 1e-7) << "Combined B+C difference vs sequential scalar";
}

TEST(Avx2KernelsTest, ClampingExtremes) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    cv::Mat src(4, 4, CV_32FC3, cv::Scalar(-0.5f, 1.5f, 2.0f));
    cv::Mat dst_b;
    cv::Mat dst_c;
    cv::Mat dst_s;

    avx2::adjust_brightness_avx2(src, dst_b, 0.1f);
    avx2::adjust_contrast_avx2(src, dst_c, 2.0f, 0.5f);
    avx2::adjust_saturation_avx2(src, dst_s, 2.0f);

    for (int r = 0; r < 4; ++r) {
        const auto* b_ptr = dst_b.ptr<cv::Vec3f>(r);
        const auto* c_ptr = dst_c.ptr<cv::Vec3f>(r);
        const auto* s_ptr = dst_s.ptr<cv::Vec3f>(r);
        for (int c = 0; c < 4; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                EXPECT_GE(b_ptr[c][ch], 0.0f);
                EXPECT_LE(b_ptr[c][ch], 1.0f);
                EXPECT_GE(c_ptr[c][ch], 0.0f);
                EXPECT_LE(c_ptr[c][ch], 1.0f);
                EXPECT_GE(s_ptr[c][ch], 0.0f);
                EXPECT_LE(s_ptr[c][ch], 1.0f);
            }
        }
    }
}

TEST(Avx2KernelsTest, NonVectorAlignedDimensions) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const int widths[] = {1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 31, 33};
    const int heights[] = {1, 3, 7};

    for (int w : widths) {
        for (int h : heights) {
            const cv::Mat src = create_random_image(w, h, static_cast<unsigned int>(w * 100 + h));

            cv::Mat dst_s_b, dst_a_b;
            EnhancementEngine::adjust_brightness_scalar(src, dst_s_b, 0.1f);
            avx2::adjust_brightness_avx2(src, dst_a_b, 0.1f);
            EXPECT_LE(max_absolute_diff(dst_s_b, dst_a_b), 1e-7);

            cv::Mat dst_s_c, dst_a_c;
            EnhancementEngine::adjust_contrast_scalar(src, dst_s_c, 1.2f, 0.5f);
            avx2::adjust_contrast_avx2(src, dst_a_c, 1.2f, 0.5f);
            EXPECT_LE(max_absolute_diff(dst_s_c, dst_a_c), 1e-7);

            cv::Mat dst_s_s, dst_a_s;
            EnhancementEngine::adjust_saturation_scalar(src, dst_s_s, 1.3f);
            avx2::adjust_saturation_avx2(src, dst_a_s, 1.3f);
            EXPECT_LE(max_absolute_diff(dst_s_s, dst_a_s), 1e-7);
        }
    }
}

TEST(Avx2KernelsTest, SmallImages) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const cv::Mat img1x1 = create_random_image(1, 1, 991);
    const cv::Mat img2x2 = create_random_image(2, 2, 992);
    const cv::Mat img3x5 = create_random_image(3, 5, 993);

    for (const auto& img : {img1x1, img2x2, img3x5}) {
        cv::Mat s_out, a_out;
        EnhancementEngine::adjust_brightness_scalar(img, s_out, 0.25f);
        avx2::adjust_brightness_avx2(img, a_out, 0.25f);
        EXPECT_LE(max_absolute_diff(s_out, a_out), 1e-7);
    }
}

TEST(Avx2KernelsTest, PartialBoundaryTiles) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const cv::Mat src = create_random_image(190, 130, 888);
    image::TileManager mgr(src, 32, 32);

    cv::Mat dst_scalar(src.size(), src.type());
    cv::Mat dst_avx2(src.size(), src.type());

    for (const auto& t : mgr.tiles()) {
        const cv::Rect rect(t.x, t.y, t.width, t.height);
        const cv::Mat tile_src = src(rect);
        cv::Mat tile_dst_s = dst_scalar(rect);
        cv::Mat tile_dst_a = dst_avx2(rect);

        EnhancementEngine::adjust_brightness_scalar(tile_src, tile_dst_s, 0.15f);
        avx2::adjust_brightness_avx2(tile_src, tile_dst_a, 0.15f);
    }

    EXPECT_LE(max_absolute_diff(dst_scalar, dst_avx2), 1e-7);
}

TEST(Avx2KernelsTest, DeterministicOutput) {
    if (!cpu::CpuFeatures::detect().has_avx2) {
        GTEST_SKIP() << "Host CPU does not support AVX2. Skipping AVX2 kernel test.";
    }

    const cv::Mat src = create_random_image(64, 64, 777);
    cv::Mat run1, run2, run3;

    avx2::adjust_brightness_contrast_avx2(src, run1, 0.1f, 1.3f, 0.5f);
    avx2::adjust_brightness_contrast_avx2(src, run2, 0.1f, 1.3f, 0.5f);
    avx2::adjust_brightness_contrast_avx2(src, run3, 0.1f, 1.3f, 0.5f);

    EXPECT_DOUBLE_EQ(max_absolute_diff(run1, run2), 0.0);
    EXPECT_DOUBLE_EQ(max_absolute_diff(run2, run3), 0.0);
}

// ============================================================================
// Runtime Dispatch & Fallback Tests
// ============================================================================

TEST(EnhancementEngineDispatchTest, AutoVsScalarVsAvx2) {
    const cv::Mat src = create_random_image(100, 80, 555);

    EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.1f;
    config.enable_contrast = true;
    config.contrast = 1.25f;
    config.enable_saturation = true;
    config.saturation = 1.4f;
    config.enable_gamma = true;
    config.gamma = 1.1f;

    const cv::Mat out_scalar = EnhancementEngine::process(src, config, ExecutionBackend::Scalar);
    const cv::Mat out_auto = EnhancementEngine::process(src, config, ExecutionBackend::Auto);

    if (cpu::CpuFeatures::detect().has_avx2) {
        const cv::Mat out_avx2 = EnhancementEngine::process(src, config, ExecutionBackend::AVX2);
        EXPECT_LE(max_absolute_diff(out_scalar, out_avx2), 1e-6);
        EXPECT_DOUBLE_EQ(max_absolute_diff(out_auto, out_avx2), 0.0);
    } else {
        EXPECT_DOUBLE_EQ(max_absolute_diff(out_auto, out_scalar), 0.0);
    }
}

TEST(EnhancementEngineDispatchTest, InvalidBackendHandling) {
    const cv::Mat src = create_random_image(16, 16);
    EnhancementConfig config{};

    // If host doesn't have AVX2, forcing AVX2 must throw EnhancementException
    if (!cpu::CpuFeatures::detect().has_avx2) {
        EXPECT_THROW(
            (void)EnhancementEngine::process(src, config, ExecutionBackend::AVX2),
            EnhancementException);
    }
}

TEST(AdaptiveEngineDispatchTest, AdaptiveBackendDispatch) {
    const cv::Mat src = create_random_image(128, 96, 666);
    EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.1f;
    config.enable_contrast = true;
    config.contrast = 1.2f;

    const auto res_scalar = precision::AdaptivePrecisionEngine::enhance_adaptive(
        src, config, 32, 32, 0.04, analysis::SensitivityWeights{}, ExecutionBackend::Scalar);

    const auto res_auto = precision::AdaptivePrecisionEngine::enhance_adaptive(
        src, config, 32, 32, 0.04, analysis::SensitivityWeights{}, ExecutionBackend::Auto);

    EXPECT_LE(max_absolute_diff(res_scalar.enhanced_image, res_auto.enhanced_image), 1e-7);
    EXPECT_EQ(res_scalar.decision.total_tiles, res_auto.decision.total_tiles);
    EXPECT_EQ(res_scalar.decision.fp32_count, res_auto.decision.fp32_count);
    EXPECT_EQ(res_scalar.decision.fp16_count, res_auto.decision.fp16_count);
}

// ============================================================================
// F16C Hardware vs Software Equivalence Investigation
// ============================================================================

TEST(F16CInvestigationTest, HardwareConversionEquivalence) {
    const bool host_has_f16c = cpu::CpuFeatures::detect().has_f16c;
    if (!host_has_f16c) {
        GTEST_SKIP() << "Host CPU does not support F16C. Skipping hardware conversion comparison.";
    }

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__F16C__)
    const float test_pixels[] = {
        0.0f, 0.1f, 0.25f, 0.333333f, 0.5f, 0.666667f, 0.75f, 0.9f, 1.0f,
        0.0001f, 0.001f, 0.01f, 1.5f, 2.0f
    };

    for (float px : test_pixels) {
        const float sw_rounded = core::round_to_fp16(px);

        const __m128 in_vec = _mm_set_ss(px);
        const __m128i ph = _mm_cvtps_ph(in_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m128 out_vec = _mm_cvtph_ps(ph);
        const float hw_rounded = _mm_cvtss_f32(out_vec);

        EXPECT_FLOAT_EQ(sw_rounded, hw_rounded)
            << "F16C hardware conversion mismatch for normalized pixel val=" << px;
    }
#endif
#endif
}
