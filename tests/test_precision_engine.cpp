#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/enhancement/precision_config.hpp"

namespace mirage::tests {

namespace {

cv::Mat create_synthetic_f32_image(int width, int height) {
    cv::Mat img(height, width, CV_32FC3);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            img.at<cv::Vec3f>(r, c) = cv::Vec3f(
                static_cast<float>(c) / static_cast<float>(width),
                static_cast<float>(r) / static_cast<float>(height),
                static_cast<float>(c + r) / static_cast<float>(width + height)
            );
        }
    }
    return img;
}

}  // namespace

TEST(PrecisionEngineTest, Fp32ModeMatchesDefaultReference) {
    const cv::Mat img = create_synthetic_f32_image(32, 32);

    enhancement::EnhancementConfig config{};
    config.enable_brightness = true; config.brightness = 0.1f;
    config.enable_contrast   = true; config.contrast   = 1.2f;
    config.enable_saturation = true; config.saturation = 1.3f;
    config.enable_gamma      = true; config.gamma      = 1.1f;

    const cv::Mat ref = enhancement::EnhancementEngine::process(img, config);
    const cv::Mat fp32_mode = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::FP32);

    // FP32 mode must be exactly identical to reference
    EXPECT_EQ(cv::norm(ref, fp32_mode, cv::NORM_INF), 0.0);
}

TEST(PrecisionEngineTest, Fp16NumericalOutputAndBoundedError) {
    const cv::Mat img = create_synthetic_f32_image(48, 48);

    enhancement::EnhancementConfig config{};
    config.enable_brightness = true; config.brightness = 0.12f;
    config.enable_contrast   = true; config.contrast   = 1.25f;
    config.enable_saturation = true; config.saturation = 1.35f;
    config.enable_gamma      = true; config.gamma      = 1.15f;

    const cv::Mat ref_fp32 = enhancement::EnhancementEngine::process(img, config);
    const cv::Mat fp16_out = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::FP16);

    EXPECT_EQ(fp16_out.size(), ref_fp32.size());
    EXPECT_EQ(fp16_out.type(), ref_fp32.type());

    // Quantitative Error Analysis against FP32 reference
    const enhancement::ErrorMetrics metrics =
        enhancement::compute_error_metrics(ref_fp32, fp16_out);

    // FP16 output differs from FP32 due to binary16 rounding, but error must be bounded:
    // Machine epsilon of FP16 is 2^-10 ~= 9.77e-4. Accumulated error across 4 operations:
    EXPECT_GT(metrics.max_absolute_error, 0.0);
    EXPECT_LT(metrics.max_absolute_error, 0.005);
    EXPECT_LT(metrics.mean_absolute_error, 0.001);
    EXPECT_LT(metrics.root_mean_square_error, 0.0015);
    EXPECT_GT(metrics.peak_signal_to_noise_ratio, 50.0); // > 50 dB PSNR
}

TEST(PrecisionEngineTest, MixedPrecisionExecutionAndIntermediateError) {
    const cv::Mat img = create_synthetic_f32_image(32, 32);

    enhancement::EnhancementConfig config{};
    config.enable_brightness = true; config.brightness = 0.15f;
    config.enable_contrast   = true; config.contrast   = 1.3f;
    config.enable_saturation = true; config.saturation = 1.4f;
    config.enable_gamma      = true; config.gamma      = 1.2f;

    const cv::Mat ref_fp32 = enhancement::EnhancementEngine::process(img, config);
    const cv::Mat fp16_out = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::FP16);

    enhancement::MixedPrecisionPolicy mixed_policy{};
    mixed_policy.brightness = enhancement::PrecisionMode::FP32; // FP32 brightness
    mixed_policy.contrast   = enhancement::PrecisionMode::FP16; // FP16 contrast
    mixed_policy.saturation = enhancement::PrecisionMode::FP16; // FP16 saturation
    mixed_policy.gamma      = enhancement::PrecisionMode::FP32; // FP32 gamma

    const cv::Mat mixed_out = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::Mixed, mixed_policy);

    const enhancement::ErrorMetrics mixed_metrics =
        enhancement::compute_error_metrics(ref_fp32, mixed_out);
    const enhancement::ErrorMetrics fp16_metrics =
        enhancement::compute_error_metrics(ref_fp32, fp16_out);

    // Mixed precision error should be non-zero but generally strictly less than or equal to full FP16
    EXPECT_GT(mixed_metrics.max_absolute_error, 0.0);
    EXPECT_LE(mixed_metrics.mean_absolute_error, fp16_metrics.mean_absolute_error * 1.05);
    EXPECT_GT(mixed_metrics.peak_signal_to_noise_ratio, 50.0);
}

TEST(PrecisionEngineTest, DeterministicExecutionAcrossAllModes) {
    const cv::Mat img = create_synthetic_f32_image(24, 24);

    enhancement::EnhancementConfig config{};
    config.enable_brightness = true; config.brightness = 0.1f;
    config.enable_contrast   = true; config.contrast   = 1.2f;

    // FP16 repeatability
    const cv::Mat fp16_run1 = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::FP16);
    const cv::Mat fp16_run2 = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::FP16);
    EXPECT_EQ(cv::norm(fp16_run1, fp16_run2, cv::NORM_INF), 0.0);

    // Mixed repeatability
    enhancement::MixedPrecisionPolicy policy{};
    policy.brightness = enhancement::PrecisionMode::FP16;
    policy.contrast = enhancement::PrecisionMode::FP32;

    const cv::Mat mixed_run1 = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::Mixed, policy);
    const cv::Mat mixed_run2 = enhancement::EnhancementEngine::process_with_precision(
        img, config, enhancement::PrecisionMode::Mixed, policy);
    EXPECT_EQ(cv::norm(mixed_run1, mixed_run2, cv::NORM_INF), 0.0);
}

TEST(PrecisionEngineTest, PrecisionModeParsing) {
    EXPECT_EQ(enhancement::parse_precision_mode("fp32"), enhancement::PrecisionMode::FP32);
    EXPECT_EQ(enhancement::parse_precision_mode("FP32"), enhancement::PrecisionMode::FP32);
    EXPECT_EQ(enhancement::parse_precision_mode("fp16"), enhancement::PrecisionMode::FP16);
    EXPECT_EQ(enhancement::parse_precision_mode("FP16"), enhancement::PrecisionMode::FP16);
    EXPECT_EQ(enhancement::parse_precision_mode("mixed"), enhancement::PrecisionMode::Mixed);
    EXPECT_EQ(enhancement::parse_precision_mode("MIXED"), enhancement::PrecisionMode::Mixed);

    EXPECT_THROW({ auto _ = enhancement::parse_precision_mode("unknown"); (void)_; },
                 std::invalid_argument);
}

}  // namespace mirage::tests

