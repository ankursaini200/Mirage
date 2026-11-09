#include <gtest/gtest.h>
#include <cmath>
#include <opencv2/core.hpp>

#include "mirage/enhancement/enhancement_engine.hpp"

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

TEST(EnhancementEngineTest, BrightnessAdjustmentAndClamping) {
    cv::Mat img(1, 3, CV_32FC3);
    img.at<cv::Vec3f>(0, 0) = cv::Vec3f(0.2f, 0.2f, 0.2f);
    img.at<cv::Vec3f>(0, 1) = cv::Vec3f(0.5f, 0.5f, 0.5f);
    img.at<cv::Vec3f>(0, 2) = cv::Vec3f(0.9f, 0.9f, 0.9f);

    // Positive delta with upper clamping
    const cv::Mat bright = enhancement::EnhancementEngine::adjust_brightness(img, 0.2f);
    EXPECT_NEAR(bright.at<cv::Vec3f>(0, 0)[0], 0.4f, 1e-5f);
    EXPECT_NEAR(bright.at<cv::Vec3f>(0, 1)[0], 0.7f, 1e-5f);
    EXPECT_NEAR(bright.at<cv::Vec3f>(0, 2)[0], 1.0f, 1e-5f);  // Clamped at 1.0f

    // Negative delta with lower clamping
    const cv::Mat dark = enhancement::EnhancementEngine::adjust_brightness(img, -0.4f);
    EXPECT_NEAR(dark.at<cv::Vec3f>(0, 0)[0], 0.0f, 1e-5f);    // Clamped at 0.0f
    EXPECT_NEAR(dark.at<cv::Vec3f>(0, 1)[0], 0.1f, 1e-5f);
    EXPECT_NEAR(dark.at<cv::Vec3f>(0, 2)[0], 0.5f, 1e-5f);
}

TEST(EnhancementEngineTest, ContrastAdjustmentAroundMidpoint) {
    cv::Mat img(1, 3, CV_32FC3);
    img.at<cv::Vec3f>(0, 0) = cv::Vec3f(0.3f, 0.3f, 0.3f);
    img.at<cv::Vec3f>(0, 1) = cv::Vec3f(0.5f, 0.5f, 0.5f);  // Exact midpoint
    img.at<cv::Vec3f>(0, 2) = cv::Vec3f(0.7f, 0.7f, 0.7f);

    const cv::Mat contrast = enhancement::EnhancementEngine::adjust_contrast(img, 2.0f, 0.5f);

    // 0.5 + 2.0 * (0.3 - 0.5) = 0.1
    EXPECT_NEAR(contrast.at<cv::Vec3f>(0, 0)[0], 0.1f, 1e-5f);
    // Midpoint is unchanged: 0.5 + 2.0 * (0.5 - 0.5) = 0.5
    EXPECT_NEAR(contrast.at<cv::Vec3f>(0, 1)[0], 0.5f, 1e-5f);
    // 0.5 + 2.0 * (0.7 - 0.5) = 0.9
    EXPECT_NEAR(contrast.at<cv::Vec3f>(0, 2)[0], 0.9f, 1e-5f);
}

TEST(EnhancementEngineTest, SaturationAndGrayscalePreservation) {
    cv::Mat img(1, 2, CV_32FC3);
    // Pixel 0: Neutral grayscale (B = G = R = 0.4)
    img.at<cv::Vec3f>(0, 0) = cv::Vec3f(0.4f, 0.4f, 0.4f);
    // Pixel 1: Colored (B = 0.2, G = 0.5, R = 0.8)
    img.at<cv::Vec3f>(0, 1) = cv::Vec3f(0.2f, 0.5f, 0.8f);

    // Scaling saturation on neutral grayscale must not alter values
    const cv::Mat boosted_sat = enhancement::EnhancementEngine::adjust_saturation(img, 2.5f);
    EXPECT_NEAR(boosted_sat.at<cv::Vec3f>(0, 0)[0], 0.4f, 1e-5f);
    EXPECT_NEAR(boosted_sat.at<cv::Vec3f>(0, 0)[1], 0.4f, 1e-5f);
    EXPECT_NEAR(boosted_sat.at<cv::Vec3f>(0, 0)[2], 0.4f, 1e-5f);

    // Saturation = 0.0 must produce exact Rec. 709 grayscale luminance
    const cv::Mat zero_sat = enhancement::EnhancementEngine::adjust_saturation(img, 0.0f);
    const float expected_y = 0.114f * 0.2f + 0.587f * 0.5f + 0.299f * 0.8f;
    EXPECT_NEAR(zero_sat.at<cv::Vec3f>(0, 1)[0], expected_y, 1e-5f);
    EXPECT_NEAR(zero_sat.at<cv::Vec3f>(0, 1)[1], expected_y, 1e-5f);
    EXPECT_NEAR(zero_sat.at<cv::Vec3f>(0, 1)[2], expected_y, 1e-5f);
}

TEST(EnhancementEngineTest, GammaCorrection) {
    cv::Mat img(1, 3, CV_32FC3);
    img.at<cv::Vec3f>(0, 0) = cv::Vec3f(0.0f, 0.0f, 0.0f);
    img.at<cv::Vec3f>(0, 1) = cv::Vec3f(0.25f, 0.25f, 0.25f);
    img.at<cv::Vec3f>(0, 2) = cv::Vec3f(1.0f, 1.0f, 1.0f);

    // gamma = 2.0 -> I_out = I_in ^ (1.0 / 2.0) = sqrt(I_in)
    const cv::Mat gamma_corrected = enhancement::EnhancementEngine::adjust_gamma(img, 2.0f);
    EXPECT_NEAR(gamma_corrected.at<cv::Vec3f>(0, 0)[0], 0.0f, 1e-5f);
    EXPECT_NEAR(gamma_corrected.at<cv::Vec3f>(0, 1)[0], 0.5f, 1e-5f);  // sqrt(0.25) = 0.5
    EXPECT_NEAR(gamma_corrected.at<cv::Vec3f>(0, 2)[0], 1.0f, 1e-5f);
}

TEST(EnhancementEngineTest, ParameterValidation) {
    const cv::Mat valid_img = create_synthetic_f32_image(16, 16);
    const cv::Mat empty_img;
    const cv::Mat uint8_img(16, 16, CV_8UC3, cv::Scalar(100, 100, 100));

    // Reject empty image
    EXPECT_THROW({ auto _ = enhancement::EnhancementEngine::adjust_brightness(empty_img, 0.1f); (void)_; },
                 enhancement::EnhancementException);

    // Reject non-CV_32FC3 image
    EXPECT_THROW({ auto _ = enhancement::EnhancementEngine::adjust_brightness(uint8_img, 0.1f); (void)_; },
                 enhancement::EnhancementException);

    // Reject invalid parameter ranges
    EXPECT_THROW({ auto _ = enhancement::EnhancementEngine::adjust_brightness(valid_img, 1.5f); (void)_; },
                 enhancement::EnhancementException);
    EXPECT_THROW({ auto _ = enhancement::EnhancementEngine::adjust_contrast(valid_img, -0.5f); (void)_; },
                 enhancement::EnhancementException);
    EXPECT_THROW({ auto _ = enhancement::EnhancementEngine::adjust_saturation(valid_img, -1.0f); (void)_; },
                 enhancement::EnhancementException);
    EXPECT_THROW({ auto _ = enhancement::EnhancementEngine::adjust_gamma(valid_img, 0.0f); (void)_; },
                 enhancement::EnhancementException);
    EXPECT_THROW({ auto _ = enhancement::EnhancementEngine::adjust_gamma(valid_img, -1.5f); (void)_; },
                 enhancement::EnhancementException);
}

TEST(EnhancementEngineTest, OutputDimensionsTypeAndValidRange) {
    const cv::Mat img = create_synthetic_f32_image(43, 29);

    enhancement::EnhancementConfig config{};
    config.enable_brightness = true; config.brightness = 0.15f;
    config.enable_contrast   = true; config.contrast   = 1.3f;
    config.enable_saturation = true; config.saturation = 1.2f;
    config.enable_gamma      = true; config.gamma      = 1.4f;

    const cv::Mat enhanced = enhancement::EnhancementEngine::process(img, config);

    EXPECT_EQ(enhanced.size(), img.size());
    EXPECT_EQ(enhanced.type(), CV_32FC3);

    // Verify all pixels remain strictly within [0.0f, 1.0f]
    for (int r = 0; r < enhanced.rows; ++r) {
        const auto* row = enhanced.ptr<cv::Vec3f>(r);
        for (int c = 0; c < enhanced.cols; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                EXPECT_GE(row[c][ch], 0.0f);
                EXPECT_LE(row[c][ch], 1.0f);
            }
        }
    }
}

TEST(EnhancementEngineTest, DeterministicOutput) {
    const cv::Mat img = create_synthetic_f32_image(32, 32);

    enhancement::EnhancementConfig config{};
    config.enable_brightness = true; config.brightness = -0.1f;
    config.enable_contrast   = true; config.contrast   = 1.25f;
    config.enable_saturation = true; config.saturation = 1.15f;
    config.enable_gamma      = true; config.gamma      = 0.9f;

    const cv::Mat run1 = enhancement::EnhancementEngine::process(img, config);
    const cv::Mat run2 = enhancement::EnhancementEngine::process(img, config);

    EXPECT_EQ(cv::norm(run1, run2, cv::NORM_INF), 0.0);
}

}  // namespace mirage::tests

