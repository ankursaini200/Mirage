#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "mirage/image/color_converter.hpp"

namespace mirage::tests {

TEST(ColorConverterTest, BgrToRgbChannelOrdering) {
    // 1x3 test image with known distinct colors: Blue, Green, Red
    cv::Mat bgr(1, 3, CV_8UC3);
    bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 0, 0);  // Pure Blue
    bgr.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);  // Pure Green
    bgr.at<cv::Vec3b>(0, 2) = cv::Vec3b(0, 0, 255);  // Pure Red

    const cv::Mat rgb = image::ColorConverter::bgr_to_rgb(bgr);
    ASSERT_EQ(rgb.size(), bgr.size());
    ASSERT_EQ(rgb.type(), bgr.type());

    // In RGB:
    // Pixel 0 (Blue) should have R=0, G=0, B=255
    EXPECT_EQ(rgb.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 255));
    // Pixel 1 (Green) should have R=0, G=255, B=0
    EXPECT_EQ(rgb.at<cv::Vec3b>(0, 1), cv::Vec3b(0, 255, 0));
    // Pixel 2 (Red) should have R=255, G=0, B=0
    EXPECT_EQ(rgb.at<cv::Vec3b>(0, 2), cv::Vec3b(255, 0, 0));
}

TEST(ColorConverterTest, RgbToBgrSymmetry) {
    cv::Mat original(32, 32, CV_8UC3);
    for (int r = 0; r < original.rows; ++r) {
        for (int c = 0; c < original.cols; ++c) {
            original.at<cv::Vec3b>(r, c) = cv::Vec3b(
                static_cast<uint8_t>(r * 7),
                static_cast<uint8_t>(c * 7),
                static_cast<uint8_t>((r + c) * 3)
            );
        }
    }

    const cv::Mat rgb = image::ColorConverter::bgr_to_rgb(original);
    const cv::Mat roundtrip = image::ColorConverter::rgb_to_bgr(rgb);

    EXPECT_EQ(cv::norm(original, roundtrip, cv::NORM_INF), 0.0);
}

TEST(ColorConverterTest, ScientificCieLabNumericalValidity) {
    // Generate normalized floating-point BGR gradient in [0.0, 1.0]
    cv::Mat bgr_float(24, 24, CV_32FC3);
    for (int r = 0; r < bgr_float.rows; ++r) {
        for (int c = 0; c < bgr_float.cols; ++c) {
            bgr_float.at<cv::Vec3f>(r, c) = cv::Vec3f(
                static_cast<float>(c) / 24.0f,
                static_cast<float>(r) / 24.0f,
                static_cast<float>(c + r) / 48.0f
            );
        }
    }

    const cv::Mat lab = image::ColorConverter::bgr_to_lab(bgr_float);

    EXPECT_EQ(lab.size(), bgr_float.size());
    EXPECT_EQ(lab.type(), CV_32FC3);

    // Verify CIELAB scientific numerical ranges
    for (int r = 0; r < lab.rows; ++r) {
        for (int c = 0; c < lab.cols; ++c) {
            const cv::Vec3f val = lab.at<cv::Vec3f>(r, c);
            const float L = val[0];
            const float a = val[1];
            const float b = val[2];

            // Lightness in [0.0, 100.0]
            EXPECT_GE(L, 0.0f);
            EXPECT_LE(L, 100.0f);

            // Chromaticity in [-127.0, +127.0]
            EXPECT_GE(a, -127.0f);
            EXPECT_LE(a, 127.0f);
            EXPECT_GE(b, -127.0f);
            EXPECT_LE(b, 127.0f);
        }
    }

    // Round-trip back to BGR (CIELAB nonlinear companding has ~1.4e-3 FP32 tolerance)
    const cv::Mat bgr_roundtrip = image::ColorConverter::lab_to_bgr(lab);
    EXPECT_EQ(bgr_roundtrip.type(), CV_32FC3);
    EXPECT_NEAR(cv::norm(bgr_float, bgr_roundtrip, cv::NORM_INF), 0.0, 2e-3);
}

TEST(ColorConverterTest, KnownColorLabReferences) {
    cv::Mat test_colors(1, 2, CV_32FC3);
    test_colors.at<cv::Vec3f>(0, 0) = cv::Vec3f(0.0f, 0.0f, 0.0f);  // Black
    test_colors.at<cv::Vec3f>(0, 1) = cv::Vec3f(1.0f, 1.0f, 1.0f);  // White

    const cv::Mat lab = image::ColorConverter::bgr_to_lab(test_colors);

    // Black: L* = 0, a* = 0, b* = 0
    const cv::Vec3f black_lab = lab.at<cv::Vec3f>(0, 0);
    EXPECT_NEAR(black_lab[0], 0.0f, 1e-3);
    EXPECT_NEAR(black_lab[1], 0.0f, 1e-3);
    EXPECT_NEAR(black_lab[2], 0.0f, 1e-3);

    // White: L* = 100, a* = 0, b* = 0
    const cv::Vec3f white_lab = lab.at<cv::Vec3f>(0, 1);
    EXPECT_NEAR(white_lab[0], 100.0f, 1e-2);
    EXPECT_NEAR(white_lab[1], 0.0f, 0.1f);
    EXPECT_NEAR(white_lab[2], 0.0f, 0.1f);
}

TEST(ColorConverterTest, InvalidInputValidation) {
    const cv::Mat empty_mat;
    EXPECT_THROW({ auto _ = image::ColorConverter::bgr_to_rgb(empty_mat); (void)_; }, image::ColorConversionException);

    // Single-channel grayscale matrix
    const cv::Mat gray(10, 10, CV_8UC1, cv::Scalar(128));
    EXPECT_THROW({ auto _ = image::ColorConverter::bgr_to_rgb(gray); (void)_; }, image::ColorConversionException);

    // Passing CV_8UC3 to scientific floating-point Lab conversion should throw
    const cv::Mat bgr8(10, 10, CV_8UC3, cv::Scalar(100, 150, 200));
    EXPECT_THROW({ auto _ = image::ColorConverter::bgr_to_lab(bgr8); (void)_; }, image::ColorConversionException);
}

}  // namespace mirage::tests
