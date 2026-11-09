#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "mirage/analysis/sensitivity_analyzer.hpp"
#include "mirage/enhancement/enhancement_engine.hpp"
#include "mirage/enhancement/precision_config.hpp"
#include "mirage/precision/adaptive_precision.hpp"

namespace mirage::tests {

namespace {

// Helper to create synthetic normalized BGR image (CV_32FC3, [0.0, 1.0])
cv::Mat create_solid_bgr(int width, int height, const cv::Scalar& bgr) {
    return cv::Mat(height, width, CV_32FC3, bgr);
}

cv::Mat create_checkerboard_bgr(int width, int height, int block_size) {
    cv::Mat img(height, width, CV_32FC3);
    const cv::Vec3f c1(0.1f, 0.2f, 0.8f);
    const cv::Vec3f c2(0.9f, 0.7f, 0.1f);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            const bool alt = ((r / block_size) % 2 == (c / block_size) % 2);
            img.at<cv::Vec3f>(r, c) = alt ? c1 : c2;
        }
    }
    return img;
}

analysis::SensitivityReport create_mock_report(const std::vector<double>& scores) {
    analysis::SensitivityReport report{};
    report.total_tiles = scores.size();
    report.tiles.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        analysis::TileSensitivity ts{};
        ts.id = i;
        ts.sensitivity_score = scores[i];
        report.tiles.push_back(ts);
    }
    return report;
}

}  // namespace

// ============================================================================
// Precision Assignment Tests
// ============================================================================

TEST(AdaptivePrecisionTest, ScoresBelowThresholdSelectFP16) {
    const auto report = create_mock_report({0.01, 0.02, 0.03});
    const auto decision = precision::AdaptivePrecisionEngine::decide_precisions(report, 0.05);

    EXPECT_EQ(decision.total_tiles, 3u);
    EXPECT_EQ(decision.fp16_count, 3u);
    EXPECT_EQ(decision.fp32_count, 0u);
    for (const auto& a : decision.assignments) {
        EXPECT_EQ(a.precision, enhancement::PrecisionMode::FP16);
    }
}

TEST(AdaptivePrecisionTest, ScoresEqualThresholdSelectFP32) {
    const auto report = create_mock_report({0.05});
    const auto decision = precision::AdaptivePrecisionEngine::decide_precisions(report, 0.05);

    EXPECT_EQ(decision.total_tiles, 1u);
    EXPECT_EQ(decision.fp32_count, 1u);
    EXPECT_EQ(decision.fp16_count, 0u);
    EXPECT_EQ(decision.assignments[0].precision, enhancement::PrecisionMode::FP32);
}

TEST(AdaptivePrecisionTest, ScoresAboveThresholdSelectFP32) {
    const auto report = create_mock_report({0.06, 0.10, 0.50});
    const auto decision = precision::AdaptivePrecisionEngine::decide_precisions(report, 0.05);

    EXPECT_EQ(decision.total_tiles, 3u);
    EXPECT_EQ(decision.fp32_count, 3u);
    EXPECT_EQ(decision.fp16_count, 0u);
    for (const auto& a : decision.assignments) {
        EXPECT_EQ(a.precision, enhancement::PrecisionMode::FP32);
    }
}

TEST(AdaptivePrecisionTest, ThresholdZeroAllFP32) {
    const auto report = create_mock_report({0.0, 0.02, 0.1, 0.8});
    const auto decision = precision::AdaptivePrecisionEngine::decide_precisions(report, 0.0);

    EXPECT_EQ(decision.total_tiles, 4u);
    EXPECT_EQ(decision.fp32_count, 4u);
    EXPECT_EQ(decision.fp16_count, 0u);
    EXPECT_DOUBLE_EQ(decision.fp32_percentage, 100.0);
    EXPECT_DOUBLE_EQ(decision.fp16_percentage, 0.0);
}

TEST(AdaptivePrecisionTest, ThresholdOneAllFP16) {
    const auto report = create_mock_report({0.0, 0.2, 0.5, 0.99});
    const auto decision = precision::AdaptivePrecisionEngine::decide_precisions(report, 1.0);

    EXPECT_EQ(decision.total_tiles, 4u);
    EXPECT_EQ(decision.fp16_count, 4u);
    EXPECT_EQ(decision.fp32_count, 0u);
    EXPECT_DOUBLE_EQ(decision.fp16_percentage, 100.0);
    EXPECT_DOUBLE_EQ(decision.fp32_percentage, 0.0);
}

TEST(AdaptivePrecisionTest, InvalidThresholdsThrow) {
    const auto report = create_mock_report({0.1, 0.2});

    EXPECT_THROW((void)precision::AdaptivePrecisionEngine::decide_precisions(report, -0.01),
                 precision::AdaptivePrecisionException);
    EXPECT_THROW((void)precision::AdaptivePrecisionEngine::decide_precisions(report, 1.01),
                 precision::AdaptivePrecisionException);
    EXPECT_THROW((void)precision::AdaptivePrecisionEngine::decide_precisions(report, std::nan("")),
                 precision::AdaptivePrecisionException);
}

TEST(AdaptivePrecisionTest, DeterministicAssignments) {
    const auto report = create_mock_report({0.01, 0.04, 0.05, 0.08, 0.03});
    const auto d1 = precision::AdaptivePrecisionEngine::decide_precisions(report, 0.04);
    const auto d2 = precision::AdaptivePrecisionEngine::decide_precisions(report, 0.04);

    EXPECT_EQ(d1.total_tiles, d2.total_tiles);
    EXPECT_EQ(d1.fp32_count, d2.fp32_count);
    EXPECT_EQ(d1.fp16_count, d2.fp16_count);
    for (size_t i = 0; i < d1.assignments.size(); ++i) {
        EXPECT_EQ(d1.assignments[i].precision, d2.assignments[i].precision);
    }
}

TEST(AdaptivePrecisionTest, PrecisionDistributionCountsSumToTotal) {
    const auto report = create_mock_report({0.02, 0.05, 0.01, 0.08, 0.03, 0.04});
    const auto decision = precision::AdaptivePrecisionEngine::decide_precisions(report, 0.04);

    EXPECT_EQ(decision.total_tiles, 6u);
    EXPECT_EQ(decision.fp32_count + decision.fp16_count, decision.total_tiles);
    EXPECT_NEAR(decision.fp32_percentage + decision.fp16_percentage, 100.0, 1e-6);
}

// ============================================================================
// Adaptive Image Enhancement Pipeline Tests
// ============================================================================

TEST(AdaptivePrecisionTest, AllTilesFP32MatchesFP32Reference) {
    const cv::Mat img = create_checkerboard_bgr(64, 64, 16);
    enhancement::EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.1f;
    config.enable_contrast = true;
    config.contrast = 1.2f;

    // Threshold 0.0 forces all tiles to FP32
    const auto res = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 0.0);

    EXPECT_EQ(res.decision.fp32_count, res.decision.total_tiles);
    EXPECT_EQ(res.decision.fp16_count, 0u);

    // FP32 reference
    const cv::Mat ref = enhancement::EnhancementEngine::process(img, config);

    // Output must be bit-identical
    const auto metrics = enhancement::compute_error_metrics(ref, res.enhanced_image);
    EXPECT_DOUBLE_EQ(metrics.max_absolute_error, 0.0);
    EXPECT_DOUBLE_EQ(metrics.mean_absolute_error, 0.0);
    EXPECT_DOUBLE_EQ(metrics.root_mean_square_error, 0.0);
    EXPECT_TRUE(std::isinf(metrics.peak_signal_to_noise_ratio));
    EXPECT_DOUBLE_EQ(metrics.relative_l2_error, 0.0);
}

TEST(AdaptivePrecisionTest, AllTilesFP16MatchesUniformFP16) {
    const cv::Mat img = create_checkerboard_bgr(64, 64, 16);
    enhancement::EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.1f;
    config.enable_contrast = true;
    config.contrast = 1.2f;

    // Threshold 1.0 forces all tiles to FP16 (since scores < 1.0)
    const auto res = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 1.0);

    EXPECT_EQ(res.decision.fp16_count, res.decision.total_tiles);
    EXPECT_EQ(res.decision.fp32_count, 0u);

    // Uniform FP16 output
    const cv::Mat fp16_out = enhancement::EnhancementEngine::process_fp16(img, config);

    // Must match uniform FP16
    const auto metrics = enhancement::compute_error_metrics(fp16_out, res.enhanced_image);
    EXPECT_DOUBLE_EQ(metrics.max_absolute_error, 0.0);
    EXPECT_DOUBLE_EQ(metrics.mean_absolute_error, 0.0);
    EXPECT_TRUE(std::isinf(metrics.peak_signal_to_noise_ratio));
}

TEST(AdaptivePrecisionTest, MixedFP16FP32Assignments) {
    // Top half is flat solid, bottom half has sharp checkerboard
    cv::Mat img = create_solid_bgr(64, 64, cv::Scalar(0.5f, 0.5f, 0.5f));
    const cv::Mat checker = create_checkerboard_bgr(64, 32, 8);
    checker.copyTo(img(cv::Rect(0, 32, 64, 32)));

    enhancement::EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.1f;
    config.enable_contrast = true;
    config.contrast = 1.2f;

    // With tile size 32, top tiles will have low sensitivity (0.0), bottom tiles high
    const auto res = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 0.02);

    // Should produce both FP32 and FP16 tiles
    EXPECT_GT(res.decision.fp32_count, 0u);
    EXPECT_GT(res.decision.fp16_count, 0u);
    EXPECT_EQ(res.decision.fp32_count + res.decision.fp16_count, res.decision.total_tiles);

    // Error compared to FP32 reference must be non-zero and bounded
    const cv::Mat ref = enhancement::EnhancementEngine::process(img, config);
    const auto metrics = enhancement::compute_error_metrics(ref, res.enhanced_image);

    EXPECT_GT(metrics.max_absolute_error, 0.0);
    EXPECT_LT(metrics.max_absolute_error, 0.005);
    EXPECT_GT(metrics.peak_signal_to_noise_ratio, 65.0);
}

TEST(AdaptivePrecisionTest, PartialBoundaryTilesAndOddDimensions) {
    // Non-divisible dimensions: 100 x 70 with 32x32 tiles -> 4 cols x 3 rows = 12 tiles
    const cv::Mat img = create_checkerboard_bgr(100, 70, 10);
    enhancement::EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.05f;

    const auto res = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 0.04);

    EXPECT_EQ(res.enhanced_image.cols, 100);
    EXPECT_EQ(res.enhanced_image.rows, 70);
    EXPECT_EQ(res.decision.total_tiles, 12u);
    EXPECT_EQ(res.decision.fp32_count + res.decision.fp16_count, 12u);
}

TEST(AdaptivePrecisionTest, ImageSmallerThanTileSize) {
    // 15 x 20 image with tile size 32 -> exactly 1 partial tile
    const cv::Mat img = create_solid_bgr(15, 20, cv::Scalar(0.3f, 0.6f, 0.9f));
    enhancement::EnhancementConfig config{};
    config.enable_contrast = true;
    config.contrast = 1.1f;

    const auto res = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 0.04);

    EXPECT_EQ(res.enhanced_image.cols, 15);
    EXPECT_EQ(res.enhanced_image.rows, 20);
    EXPECT_EQ(res.decision.total_tiles, 1u);
    EXPECT_EQ(res.decision.fp32_count + res.decision.fp16_count, 1u);
}

TEST(AdaptivePrecisionTest, UniformImageHandling) {
    const cv::Mat img = create_solid_bgr(64, 64, cv::Scalar(0.4f, 0.4f, 0.4f));
    enhancement::EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.1f;

    // Solid image has sensitivity = 0.0. With threshold = 0.01, all select FP16
    const auto res = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 0.01);

    EXPECT_EQ(res.decision.fp16_count, res.decision.total_tiles);
    EXPECT_EQ(res.decision.fp32_count, 0u);
}

TEST(AdaptivePrecisionTest, DeterministicOutput) {
    const cv::Mat img = create_checkerboard_bgr(64, 64, 12);
    enhancement::EnhancementConfig config{};
    config.enable_brightness = true;
    config.brightness = 0.05f;
    config.enable_contrast = true;
    config.contrast = 1.3f;

    const auto res1 = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 0.03);
    const auto res2 = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 32, 32, 0.03);

    EXPECT_EQ(res1.decision.fp32_count, res2.decision.fp32_count);
    EXPECT_EQ(res1.decision.fp16_count, res2.decision.fp16_count);

    const auto metrics = enhancement::compute_error_metrics(
        res1.enhanced_image, res2.enhanced_image);
    EXPECT_DOUBLE_EQ(metrics.max_absolute_error, 0.0);
    EXPECT_TRUE(std::isinf(metrics.peak_signal_to_noise_ratio));
}

TEST(AdaptivePrecisionTest, OutputDimensionsAndTypePreserved) {
    const cv::Mat img = create_checkerboard_bgr(80, 50, 10);
    enhancement::EnhancementConfig config{};
    config.enable_gamma = true;
    config.gamma = 1.2f;

    const auto res = precision::AdaptivePrecisionEngine::enhance_adaptive(
        img, config, 16, 16, 0.04);

    EXPECT_EQ(res.enhanced_image.cols, 80);
    EXPECT_EQ(res.enhanced_image.rows, 50);
    EXPECT_EQ(res.enhanced_image.channels(), 3);
    EXPECT_EQ(res.enhanced_image.type(), CV_32FC3);
}

TEST(AdaptivePrecisionTest, RelativeL2ErrorAndZeroReferenceHandling) {
    const cv::Mat black = create_solid_bgr(32, 32, cv::Scalar(0.0f, 0.0f, 0.0f));

    // Zero reference image vs zero target image should yield 0 relative error (no div-by-zero)
    const auto metrics = enhancement::compute_error_metrics(black, black);
    EXPECT_DOUBLE_EQ(metrics.relative_l2_error, 0.0);
    EXPECT_DOUBLE_EQ(metrics.mean_relative_error, 0.0);
    EXPECT_TRUE(std::isinf(metrics.peak_signal_to_noise_ratio));
}

}  // namespace mirage::tests
