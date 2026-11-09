#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "mirage/analysis/quality_metrics.hpp"

using namespace mirage::analysis;

TEST(QualityMetricsTest, CIEDE2000IdenticalColors) {
    const double de = compute_ciede2000(50.0, 0.0, 0.0, 50.0, 0.0, 0.0);
    EXPECT_NEAR(de, 0.0, 1e-6);
}

TEST(QualityMetricsTest, CIEDE2000SharmaReferencePairs) {
    // Sharma et al. (2005) published benchmark verification pairs
    // Pair 1:
    const double de1 = compute_ciede2000(50.0, 2.6772, -79.7751, 50.0, 0.0, -82.7485);
    EXPECT_NEAR(de1, 2.0425, 1e-3);

    // Pair 2:
    const double de2 = compute_ciede2000(50.0, 3.1571, -77.5521, 50.0, 0.0, -82.7485);
    EXPECT_NEAR(de2, 2.7998, 1e-3);

    // Pair 3:
    const double de3 = compute_ciede2000(50.0, 2.8361, -74.0530, 50.0, 0.0, -82.7485);
    EXPECT_NEAR(de3, 3.4333, 1e-3);

    // Symmetry test: compute_ciede2000(A, B) == compute_ciede2000(B, A)
    const double de1_rev = compute_ciede2000(50.0, 0.0, -82.7485, 50.0, 2.6772, -79.7751);
    EXPECT_NEAR(de1, de1_rev, 1e-6);
}

TEST(QualityMetricsTest, SSIMIdenticalImages) {
    cv::Mat img(64, 64, CV_32FC3);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            img.at<cv::Vec3f>(y, x) = cv::Vec3f(
                static_cast<float>(x) / 64.0f,
                static_cast<float>(y) / 64.0f,
                static_cast<float>(x + y) / 128.0f);
        }
    }

    const double ssim = compute_ssim(img, img);
    EXPECT_NEAR(ssim, 1.0, 1e-4);
}

TEST(QualityMetricsTest, SSIMDifferentImages) {
    cv::Mat img1(64, 64, CV_32FC3, cv::Scalar(0.1f, 0.1f, 0.1f));
    cv::Mat img2(64, 64, CV_32FC3, cv::Scalar(0.9f, 0.9f, 0.9f));

    const double ssim = compute_ssim(img1, img2);
    EXPECT_GE(ssim, 0.0);
    EXPECT_LE(ssim, 0.5);
}

TEST(QualityMetricsTest, EvaluateQualityIdenticalImages) {
    cv::Mat img(32, 32, CV_32FC3, cv::Scalar(0.5f, 0.4f, 0.3f));
    const auto q = evaluate_quality(img, img);

    EXPECT_NEAR(q.max_absolute_error, 0.0, 1e-6);
    EXPECT_NEAR(q.mean_absolute_error, 0.0, 1e-6);
    EXPECT_NEAR(q.root_mean_square_error, 0.0, 1e-6);
    EXPECT_GE(q.peak_signal_to_noise_ratio, 99.0);
    EXPECT_NEAR(q.relative_l2_error, 0.0, 1e-6);
    EXPECT_NEAR(q.ssim, 1.0, 1e-4);
    EXPECT_NEAR(q.mean_ciede2000, 0.0, 1e-5);
    EXPECT_NEAR(q.max_ciede2000, 0.0, 1e-5);
}

TEST(QualityMetricsTest, EvaluateQualityKnownShift) {
    cv::Mat img1(32, 32, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
    cv::Mat img2(32, 32, CV_32FC3, cv::Scalar(0.6f, 0.6f, 0.6f));

    const auto q = evaluate_quality(img1, img2);
    EXPECT_NEAR(q.mean_absolute_error, 0.10, 1e-4);
    EXPECT_NEAR(q.root_mean_square_error, 0.10, 1e-4);
    EXPECT_NEAR(q.max_absolute_error, 0.10, 1e-4);
    EXPECT_GT(q.mean_ciede2000, 0.0);
}
